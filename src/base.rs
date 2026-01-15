use std::collections::{HashMap, HashSet};

use smol_str::SmolStr;

use super::{
    SNAPSHOT_VERSION,
    index::Index,
    pipeline::{DefaultTokenizer, Pipeline},
    tokenizer::Token,
    types::{
        DerivedSpan, DerivedTerm, DocData, DomainLengths, InMemoryIndex, PositionEncoding,
        SnapshotData, TermDomain, TermFrequencyEntry, TermId, TermPositions, TokenStream,
    },
};

type DirtyDoc = (String, String, String, i64);
type DeletedDoc = HashMap<String, HashSet<String>>;

impl InMemoryIndex {
    /// Create an index that returns match spans in the given encoding.
    pub fn with_position_encoding(encoding: PositionEncoding) -> Self {
        Self {
            position_encoding: encoding,
            ..Default::default()
        }
    }

    /// Create an index that uses a custom dictionary for tokenization.
    pub fn with_dictionary_config(dictionary: crate::tokenizer::DictionaryConfig) -> Self {
        Self {
            dictionary: Some(dictionary),
            ..Default::default()
        }
    }

    /// Set the encoding (bytes or UTF-16) used when returning match spans.
    pub fn set_position_encoding(&mut self, encoding: PositionEncoding) {
        self.position_encoding = encoding;
    }

    /// Swap in or remove a dictionary config for future tokenization.
    pub fn set_dictionary_config(
        &mut self,
        dictionary: Option<crate::tokenizer::DictionaryConfig>,
    ) {
        self.dictionary = dictionary;
    }

    /// Add or replace a document in an index. Set `index` to false to stage content without
    /// tokenization (doc will exist but not be searchable).
    pub fn add_doc(&mut self, index_name: &str, doc_id: &str, text: &str, index: bool) {
        let token_stream = if index {
            self.document_pipeline().document_tokens(text)
        } else {
            TokenStream {
                tokens: Vec::new(),
                doc_len: 0,
            }
        };

        let mut maps = Index {
            state: self.index_state_mut(index_name),
        };

        let doc_idx = if let Some(existing) = maps.state.doc_index.get(doc_id) {
            *existing
        } else if let Some(reuse) = maps.state.free_docs.pop() {
            let doc_key = SmolStr::new(doc_id);
            if let Some(slot) = maps.state.doc_ids.get_mut(reuse as usize) {
                *slot = doc_key.clone();
            } else {
                maps.state
                    .doc_ids
                    .resize(reuse as usize + 1, SmolStr::default());
                maps.state.doc_ids[reuse as usize] = doc_key.clone();
            }
            if maps.state.docs.len() <= reuse as usize {
                maps.state.docs.resize(reuse as usize + 1, None);
            }
            maps.state.doc_index.insert(doc_key, reuse);
            reuse
        } else {
            let doc_key = SmolStr::new(doc_id);
            let id = maps.state.doc_ids.len() as super::types::DocId;
            maps.state.doc_ids.push(doc_key.clone());
            maps.state.docs.push(None);
            maps.state.doc_index.insert(doc_key, id);
            id
        };

        if let Some(old_data) = maps
            .state
            .docs
            .get_mut(doc_idx as usize)
            .and_then(|slot| slot.take())
        {
            maps.state.total_len -= old_data.doc_len;
            let old_domain_lengths = DomainLengths::from_doc(&old_data);
            old_domain_lengths.for_each_nonzero(|domain, len| {
                maps.state.domain_total_len.add(domain, -len);
            });
            maps.remove_doc_terms(doc_idx, &old_data);
        }

        let mut term_pos: HashMap<TermId, Vec<(u32, u32)>> = HashMap::new();
        let mut derived_candidates: Vec<(TermId, TermId, (u32, u32))> = Vec::new();
        let mut term_freqs: HashMap<TermId, [u32; super::types::TERM_DOMAIN_COUNT]> =
            HashMap::new();

        for token in &token_stream.tokens {
            let term_id = get_or_insert_term_id(maps.state, &token.term);
            let domain_idx = super::types::domain_index(token.domain);
            let counts = term_freqs
                .entry(term_id)
                .or_insert([0; super::types::TERM_DOMAIN_COUNT]);
            counts[domain_idx] += 1;

            if token.domain == TermDomain::Original {
                term_pos
                    .entry(term_id)
                    .or_default()
                    .push((token.span.0 as u32, token.span.1 as u32));
            } else {
                let base_term_id = get_or_insert_term_id(maps.state, &token.base_term);
                derived_candidates.push((
                    term_id,
                    base_term_id,
                    (token.span.0 as u32, token.span.1 as u32),
                ));
            }
        }

        let mut term_positions: Vec<TermPositions> = term_pos
            .into_iter()
            .map(|(term, mut positions)| {
                positions.sort();
                positions.dedup();
                TermPositions { term, positions }
            })
            .collect();
        term_positions.sort_by_key(|entry| entry.term);

        let base_terms: HashSet<TermId> = term_positions.iter().map(|entry| entry.term).collect();
        let mut derived_terms: Vec<DerivedTerm> = Vec::new();
        let mut derived_spans_map: HashMap<TermId, (u32, u32)> = HashMap::new();
        for (derived, base, span) in derived_candidates {
            if base_terms.contains(&base) {
                derived_terms.push(DerivedTerm { derived, base });
            } else {
                let span_len = span.1.saturating_sub(span.0);
                derived_spans_map
                    .entry(derived)
                    .and_modify(|existing| {
                        let existing_len = existing.1.saturating_sub(existing.0);
                        if span_len < existing_len {
                            *existing = span;
                        }
                    })
                    .or_insert(span);
            }
        }
        derived_terms.sort_by(|a, b| (a.derived, a.base).cmp(&(b.derived, b.base)));
        derived_terms.dedup_by(|a, b| a.derived == b.derived && a.base == b.base);
        let mut derived_spans: Vec<DerivedSpan> = derived_spans_map
            .into_iter()
            .map(|(derived, span)| DerivedSpan { derived, span })
            .collect();
        derived_spans.sort_by_key(|entry| entry.derived);

        let mut term_freqs_vec: Vec<TermFrequencyEntry> = term_freqs
            .into_iter()
            .map(|(term, counts)| TermFrequencyEntry { term, counts })
            .collect();
        term_freqs_vec.sort_by_key(|entry| entry.term);

        let doc_len = token_stream.doc_len;
        let mut domain_doc_len = DomainLengths::from_term_freqs(&term_freqs_vec);
        if domain_doc_len.is_zero() {
            domain_doc_len.add(TermDomain::Original, doc_len);
        }

        for entry in &term_freqs_vec {
            for (domain, count) in entry.positive_domains() {
                maps.add_posting(entry.term, domain, doc_idx, count);
            }
        }

        let doc_data = DocData {
            content: text.to_string(),
            doc_len,
            term_pos: term_positions,
            term_freqs: term_freqs_vec,
            domain_doc_len,
            derived_terms,
            derived_spans,
        };

        if maps.state.docs.len() <= doc_idx as usize {
            maps.state.docs.resize(doc_idx as usize + 1, None);
        }
        maps.state.docs[doc_idx as usize] = Some(doc_data);

        maps.state.total_len += doc_len;
        domain_doc_len.for_each_nonzero(|domain, len| {
            maps.state.domain_total_len.add(domain, len);
        });

        let doc_key = maps
            .state
            .doc_ids
            .get(doc_idx as usize)
            .cloned()
            .unwrap_or_else(|| SmolStr::new(doc_id));
        maps.state.dirty.insert(doc_key.clone());
        maps.state.deleted.remove(doc_key.as_str());
    }

    /// Remove a document and its postings from an index.
    pub fn remove_doc(&mut self, index_name: &str, doc_id: &str) {
        let mut maps = Index {
            state: self.index_state_mut(index_name),
        };
        let Some(&doc_idx) = maps.state.doc_index.get(doc_id) else {
            return;
        };

        if let Some(old_data) = maps
            .state
            .docs
            .get_mut(doc_idx as usize)
            .and_then(|slot| slot.take())
        {
            maps.state.total_len -= old_data.doc_len;
            let old_domain_lengths = DomainLengths::from_doc(&old_data);
            old_domain_lengths.for_each_nonzero(|domain, len| {
                maps.state.domain_total_len.add(domain, -len);
            });
            maps.remove_doc_terms(doc_idx, &old_data);
        }

        maps.state.doc_index.remove(doc_id);
        maps.state.free_docs.push(doc_idx);
        let doc_key = maps
            .state
            .doc_ids
            .get(doc_idx as usize)
            .cloned()
            .unwrap_or_else(|| SmolStr::new(doc_id));
        maps.state.deleted.insert(doc_key);
        maps.state.dirty.remove(doc_id);
    }

    /// Fetch raw document content by id, if present.
    pub fn get_doc(&self, index_name: &str, doc_id: &str) -> Option<String> {
        let state = self.indexes.get(index_name)?;
        let doc_idx = *state.doc_index.get(doc_id)? as usize;
        state
            .docs
            .get(doc_idx)
            .and_then(|doc| doc.as_ref())
            .map(|d| d.content.clone())
    }

    /// Return and clear the sets of dirty and deleted docs for persistence.
    pub fn take_dirty_and_deleted(&mut self) -> (Vec<DirtyDoc>, DeletedDoc) {
        let mut dirty_data = Vec::new();
        let mut deleted = HashMap::new();

        for (index_name, state) in self.indexes.iter_mut() {
            let dirty = std::mem::take(&mut state.dirty);
            let deleted_ids = std::mem::take(&mut state.deleted);

            for doc_id in dirty {
                if let Some(&doc_idx) = state.doc_index.get(&doc_id)
                    && let Some(doc) = state
                        .docs
                        .get(doc_idx as usize)
                        .and_then(|entry| entry.as_ref())
                {
                    dirty_data.push((
                        index_name.clone(),
                        doc_id.to_string(),
                        doc.content.clone(),
                        doc.doc_len,
                    ));
                }
            }

            if !deleted_ids.is_empty() {
                let deleted_strings: HashSet<String> = deleted_ids
                    .into_iter()
                    .map(|doc_id| doc_id.to_string())
                    .collect();
                deleted.insert(index_name.clone(), deleted_strings);
            }
        }

        (dirty_data, deleted)
    }

    /// Returns true if the index has new changes awaiting persistence.
    /// Pass `Some(name)` to check a specific index or `None` to check all.
    pub fn has_unpersisted_changes(&self, index_name: Option<&str>) -> bool {
        match index_name {
            Some(name) => self
                .indexes
                .get(name)
                .is_some_and(|state| !state.dirty.is_empty() || !state.deleted.is_empty()),
            None => self
                .indexes
                .values()
                .any(|state| !state.dirty.is_empty() || !state.deleted.is_empty()),
        }
    }

    /// Persist the given index only if there are pending changes.
    ///
    /// Returns `Ok(true)` if persistence was attempted (and succeeded), `Ok(false)` if skipped.
    /// The index is marked clean only after the provided callback returns `Ok`.
    pub fn persist_if_dirty<E>(
        &mut self,
        index_name: &str,
        mut persist: impl FnMut(SnapshotData) -> Result<(), E>,
    ) -> Result<bool, E> {
        if !self.has_unpersisted_changes(Some(index_name)) {
            return Ok(false);
        }

        let Some(snapshot) = self.get_snapshot_data(index_name) else {
            return Ok(false);
        };

        persist(snapshot)?;
        if let Some(state) = self.indexes.get_mut(index_name) {
            state.dirty.clear();
            state.deleted.clear();
        }
        Ok(true)
    }

    /// Get byte/UTF-16 spans for a query's terms within a document by re-tokenizing the query.
    pub fn get_matches(&self, index_name: &str, doc_id: &str, query: &str) -> Vec<(u32, u32)> {
        let query_terms: Vec<String> = self
            .tokenize_query(query)
            .into_iter()
            .map(|t| t.term)
            .collect();
        self.get_matches_for_terms(index_name, doc_id, &query_terms)
    }

    /// Get spans for specific terms within a document.
    pub fn get_matches_for_terms(
        &self,
        index_name: &str,
        doc_id: &str,
        terms: &[String],
    ) -> Vec<(u32, u32)> {
        let mut matches = Vec::new();
        let Some(state) = self.indexes.get(index_name) else {
            return matches;
        };
        let Some(&doc_idx) = state.doc_index.get(doc_id) else {
            return matches;
        };
        let Some(doc_data) = state
            .docs
            .get(doc_idx as usize)
            .and_then(|doc| doc.as_ref())
        else {
            return matches;
        };

        for term in terms {
            let Some(&term_id) = state.term_index.get(term.as_str()) else {
                continue;
            };

            let mut found = false;
            if let Some(positions) = find_term_positions(doc_data, term_id) {
                matches.extend(positions.iter().copied());
                found = true;
            }

            if !found {
                for base_term in find_base_terms(doc_data, term_id) {
                    if let Some(positions) = find_term_positions(doc_data, base_term) {
                        matches.extend(positions.iter().copied());
                        found = true;
                    }
                }
            }

            if !found {
                matches.extend(find_derived_spans(doc_data, term_id));
            }
        }

        if !matches.is_empty() {
            matches = convert_spans(&doc_data.content, &matches, self.position_encoding);
        }
        matches.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| (a.1 - a.0).cmp(&(b.1 - b.0))));
        matches = prune_overlapping_starts(&matches);
        matches
    }

    /// Get spans for previously returned matched terms (e.g., from `search_hits`).
    pub fn get_matches_for_matched_terms(
        &self,
        index_name: &str,
        doc_id: &str,
        terms: &[crate::types::MatchedTerm],
    ) -> Vec<(u32, u32)> {
        let term_strings: Vec<String> = terms.iter().map(|t| t.term.clone()).collect();
        self.get_matches_for_terms(index_name, doc_id, &term_strings)
    }

    /// Load a snapshot into an index, expecting all auxiliary structures to be present.
    pub fn load_snapshot(&mut self, index_name: &str, snapshot: SnapshotData) {
        if snapshot.version != SNAPSHOT_VERSION {
            return;
        }
        let version = {
            let mut maps = Index {
                state: self.index_state_mut(index_name),
            };
            maps.clear();
            maps.import_snapshot(snapshot);
            maps.state.version
        };
        if let Some(state) = self.indexes.get_mut(index_name) {
            state.version = version;
            state.dirty.clear();
            state.deleted.clear();
        }
    }

    /// Get a serializable snapshot of the given index, including postings.
    pub fn get_snapshot_data(&self, index_name: &str) -> Option<SnapshotData> {
        let state = self.indexes.get(index_name)?;
        if state.docs.iter().all(|d| d.is_none()) {
            return None;
        }

        Some(SnapshotData {
            version: state.version,
            terms: state.terms.clone(),
            docs: state.docs.clone(),
            doc_ids: state.doc_ids.clone(),
            domains: state.domains.clone(),
            total_len: state.total_len,
            domain_total_len: state.domain_total_len,
        })
    }

    fn document_pipeline(&self) -> Pipeline {
        if let Some(cfg) = &self.dictionary {
            Pipeline::with_dictionary(cfg.clone())
        } else {
            Pipeline::document_pipeline()
        }
    }

    pub(super) fn tokenize_query(&self, query: &str) -> Vec<Token> {
        if let Some(cfg) = &self.dictionary {
            Pipeline::new(DefaultTokenizer::for_queries().with_dictionary(cfg.clone()))
                .query_tokens(query)
                .tokens
                .into_iter()
                .map(|token| Token {
                    term: token.term,
                    start: token.span.0,
                    end: token.span.1,
                })
                .collect()
        } else {
            Pipeline::tokenize_query(query)
        }
    }
}

fn get_or_insert_term_id(state: &mut super::types::IndexState, term: &str) -> TermId {
    if let Some(&id) = state.term_index.get(term) {
        return id;
    }
    let id = state.terms.len() as TermId;
    let term_key = SmolStr::new(term);
    state.terms.push(term_key.clone());
    state.term_index.insert(term_key, id);
    id
}

fn find_term_positions(doc: &DocData, term: TermId) -> Option<&[(u32, u32)]> {
    let idx = doc
        .term_pos
        .binary_search_by_key(&term, |entry| entry.term)
        .ok()?;
    Some(&doc.term_pos[idx].positions)
}

fn find_base_terms(doc: &DocData, derived: TermId) -> Vec<TermId> {
    let list = &doc.derived_terms;
    let mut start = match list.binary_search_by_key(&derived, |entry| entry.derived) {
        Ok(idx) => idx,
        Err(_) => return Vec::new(),
    };
    while start > 0 && list[start - 1].derived == derived {
        start -= 1;
    }
    let mut terms = Vec::new();
    let mut idx = start;
    while idx < list.len() && list[idx].derived == derived {
        terms.push(list[idx].base);
        idx += 1;
    }
    terms
}

fn find_derived_spans(doc: &DocData, derived: TermId) -> Vec<(u32, u32)> {
    let list = &doc.derived_spans;
    let mut start = match list.binary_search_by_key(&derived, |entry| entry.derived) {
        Ok(idx) => idx,
        Err(_) => return Vec::new(),
    };
    while start > 0 && list[start - 1].derived == derived {
        start -= 1;
    }
    let mut spans = Vec::new();
    let mut idx = start;
    while idx < list.len() && list[idx].derived == derived {
        spans.push(list[idx].span);
        idx += 1;
    }
    spans
}

fn convert_spans(
    content: &str,
    spans: &[(u32, u32)],
    encoding: PositionEncoding,
) -> Vec<(u32, u32)> {
    match encoding {
        PositionEncoding::Bytes => spans.to_vec(),
        PositionEncoding::Utf16 => spans
            .iter()
            .map(|(start, end)| {
                let s = to_utf16_index(content, *start as usize);
                let e = to_utf16_index(content, *end as usize);
                (s as u32, e as u32)
            })
            .collect(),
    }
}

fn to_utf16_index(content: &str, byte_idx: usize) -> usize {
    if byte_idx == 0 {
        return 0;
    }
    let prefix = &content[..byte_idx.min(content.len())];
    prefix.encode_utf16().count()
}

fn prune_overlapping_starts(spans: &[(u32, u32)]) -> Vec<(u32, u32)> {
    if spans.is_empty() {
        return Vec::new();
    }
    let mut pruned = Vec::new();
    let mut i = 0;
    while i < spans.len() {
        let start = spans[i].0;
        let mut best = spans[i];
        let mut j = i + 1;
        while j < spans.len() && spans[j].0 == start {
            let best_len = best.1 - best.0;
            let cur_len = spans[j].1 - spans[j].0;
            if cur_len < best_len {
                best = spans[j];
            }
            j += 1;
        }
        pruned.push(best);
        i = j;
    }
    pruned
}
