use std::collections::{HashMap, HashSet};

use super::{
    pipeline::{DefaultTokenizer, Pipeline},
    tokenizer::Token,
    types::{
        DocData, DomainLengths, InMemoryIndex, PipelineToken, PositionEncoding, SNAPSHOT_VERSION,
        SnapshotData, TermDomain, TokenStream,
    },
};

type DirtyDoc = (String, String, String, i64);
type DeletedDoc = HashMap<String, HashSet<String>>;

impl InMemoryIndex {
    /// Create an index that returns match spans in the given encoding.
    pub fn with_position_encoding(encoding: PositionEncoding) -> Self {
        let mut index = Self::default();
        index.position_encoding = encoding;
        index
    }

    /// Create an index that uses a custom dictionary for tokenization.
    pub fn with_dictionary_config(dictionary: crate::tokenizer::DictionaryConfig) -> Self {
        let mut index = Self::default();
        index.dictionary = Some(dictionary);
        index
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
                term_freqs: HashMap::new(),
                doc_len: 0,
            }
        };

        let mut pos_map: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut derived_mapping: HashMap<String, HashSet<(u32, u32)>> = HashMap::new();
        for PipelineToken {
            term, span, domain, ..
        } in &token_stream.tokens
        {
            if *domain == TermDomain::Original {
                pos_map
                    .entry(term.clone())
                    .or_default()
                    .push((span.0 as u32, span.1 as u32));
            } else {
                derived_mapping
                    .entry(term.clone())
                    .or_default()
                    .insert((span.0 as u32, span.1 as u32));
            }
        }
        let doc_len = token_stream.doc_len;
        let term_freqs = token_stream.term_freqs;
        let mut domain_doc_len = DomainLengths::from_term_freqs(&term_freqs);
        if domain_doc_len.is_zero() {
            domain_doc_len.add(TermDomain::Original, doc_len);
        }

        if let Some(docs) = self.docs.get_mut(index_name) {
            if let Some(old_data) = docs.remove(doc_id) {
                *self.total_lens.entry(index_name.to_string()).or_default() -= old_data.doc_len;

                let old_domain_lengths = DomainLengths::from_doc(&old_data);
                if let Some(total_by_domain) = self.domain_total_lens.get_mut(index_name) {
                    old_domain_lengths.for_each_nonzero(|domain, len| {
                        total_by_domain.add(domain, -len);
                    });
                }

                self.index_maps_mut(index_name)
                    .remove_doc_terms(doc_id, &old_data);
            }
        }

        let mut writer = self.index_writer(index_name, doc_id);
        for (term, freqs) in &term_freqs {
            writer.add_term_frequency(term, freqs);
        }

        let doc_data = DocData {
            content: text.to_string(),
            doc_len,
            term_pos: pos_map,
            term_freqs,
            domain_doc_len: domain_doc_len.clone(),
            derived_terms: derived_mapping
                .into_iter()
                .map(|(k, v)| {
                    let mut spans: Vec<(u32, u32)> = v.into_iter().collect();
                    spans.sort();
                    spans.dedup();
                    if let Some(min_len) = spans.iter().map(|(s, e)| e - s).min() {
                        spans.retain(|(s, e)| e - s == min_len);
                    }
                    (k, spans)
                })
                .collect(),
        };

        self.docs
            .entry(index_name.to_string())
            .or_default()
            .insert(doc_id.to_string(), doc_data);
        *self.total_lens.entry(index_name.to_string()).or_default() += doc_len;
        let total_by_domain = self
            .domain_total_lens
            .entry(index_name.to_string())
            .or_default();
        domain_doc_len.for_each_nonzero(|domain, len| {
            total_by_domain.add(domain, len);
        });

        self.dirty
            .entry(index_name.to_string())
            .or_default()
            .insert(doc_id.to_string());
        if let Some(deleted) = self.deleted.get_mut(index_name) {
            deleted.remove(doc_id);
        }
    }

    /// Remove a document and its postings from an index.
    pub fn remove_doc(&mut self, index_name: &str, doc_id: &str) {
        if let Some(docs) = self.docs.get_mut(index_name) {
            if let Some(old_data) = docs.remove(doc_id) {
                *self.total_lens.entry(index_name.to_string()).or_default() -= old_data.doc_len;

                let old_domain_lengths = DomainLengths::from_doc(&old_data);
                if let Some(total_by_domain) = self.domain_total_lens.get_mut(index_name) {
                    old_domain_lengths.for_each_nonzero(|domain, len| {
                        total_by_domain.add(domain, -len);
                    });
                }

                self.index_maps_mut(index_name)
                    .remove_doc_terms(doc_id, &old_data);

                self.deleted
                    .entry(index_name.to_string())
                    .or_default()
                    .insert(doc_id.to_string());
                if let Some(dirty) = self.dirty.get_mut(index_name) {
                    dirty.remove(doc_id);
                }
            }
        }
    }

    /// Fetch raw document content by id, if present.
    pub fn get_doc(&self, index_name: &str, doc_id: &str) -> Option<String> {
        self.docs
            .get(index_name)
            .and_then(|docs| docs.get(doc_id))
            .map(|d| d.content.clone())
    }

    /// Return and clear the sets of dirty and deleted docs for persistence.
    pub fn take_dirty_and_deleted(&mut self) -> (Vec<DirtyDoc>, DeletedDoc) {
        let dirty = std::mem::take(&mut self.dirty);
        let deleted = std::mem::take(&mut self.deleted);

        let mut dirty_data = Vec::new();
        for (index_name, doc_ids) in &dirty {
            if let Some(docs) = self.docs.get(index_name) {
                for doc_id in doc_ids {
                    if let Some(data) = docs.get(doc_id) {
                        dirty_data.push((
                            index_name.clone(),
                            doc_id.clone(),
                            data.content.clone(),
                            data.doc_len,
                        ));
                    }
                }
            }
        }
        (dirty_data, deleted)
    }

    /// Returns true if the index has new changes awaiting persistence.
    /// Pass `Some(name)` to check a specific index or `None` to check all.
    pub fn has_unpersisted_changes(&self, index_name: Option<&str>) -> bool {
        match index_name {
            Some(name) => {
                self.dirty.get(name).map_or(false, |s| !s.is_empty())
                    || self.deleted.get(name).map_or(false, |s| !s.is_empty())
            }
            None => {
                self.dirty.values().any(|s| !s.is_empty())
                    || self.deleted.values().any(|s| !s.is_empty())
            }
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
        self.dirty.remove(index_name);
        self.deleted.remove(index_name);
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
        if let Some(docs) = self.docs.get(index_name) {
            if let Some(doc_data) = docs.get(doc_id) {
                for term in terms {
                    if let Some(positions) = doc_data.term_pos.get(term) {
                        matches.extend(positions.iter().cloned());
                        continue;
                    }
                    if let Some(positions) = doc_data.derived_terms.get(term) {
                        matches.extend(positions.iter().cloned());
                    }
                }
                if !matches.is_empty() {
                    matches = convert_spans(&doc_data.content, &matches, self.position_encoding);
                }
            }
        }
        matches.sort_by(|a, b| a.0.cmp(&b.0));
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
        assert_eq!(
            snapshot.version, SNAPSHOT_VERSION,
            "snapshot version {} does not match expected {}",
            snapshot.version, SNAPSHOT_VERSION
        );
        let version = {
            let mut maps = self.index_maps_mut(index_name);
            maps.clear(false);
            maps.import_snapshot(snapshot);
            maps.version
        };
        self.versions.insert(index_name.to_string(), version);
        self.dirty.remove(index_name);
        self.deleted.remove(index_name);
    }

    /// Get a serializable snapshot of the given index, including aux dictionaries/ngrams.
    pub fn get_snapshot_data(&self, index_name: &str) -> Option<SnapshotData> {
        self.docs.get(index_name).map(|docs| {
            let domains = self.domains.get(index_name).cloned().unwrap_or_default();

            SnapshotData {
                version: *self.versions.get(index_name).unwrap_or(&SNAPSHOT_VERSION),
                docs: docs.clone(),
                total_len: *self.total_lens.get(index_name).unwrap_or(&0),
                domain_total_len: self
                    .domain_total_lens
                    .get(index_name)
                    .cloned()
                    .unwrap_or_default(),
                domains,
            }
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
