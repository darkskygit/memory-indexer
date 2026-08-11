use roaring::RoaringBitmap;

use crate::{
    Error, FieldId, FieldType, MemoryIndex, Result, SearchMode, Value,
    index::{FieldIndex, TermId, TextIndex},
    ngram::{DEFAULT_FUZZY_PARAMS, collect_fuzzy_candidates},
    pipeline::{DefaultTokenizer, Pipeline},
    types::TermDomain,
};

use super::{Query, SearchOptions, Sort};

const MAX_QUERY_TOKENS: usize = 64;
const MAX_SEARCH_LIMIT: usize = 10_000;
const MIN_SHOULD_MATCH_RATIO: f32 = 0.6;

pub(super) struct EvalResult {
    pub(super) docs: RoaringBitmap,
    pub(super) scores: Vec<f32>,
}

impl MemoryIndex {
    pub(super) fn validate_search(&self, query: &Query, options: &SearchOptions) -> Result<()> {
        self.validate_query(query)?;
        if options.limit > MAX_SEARCH_LIMIT
            || options.offset.saturating_add(options.limit) > MAX_SEARCH_LIMIT
        {
            return Err(Error::InvalidQuery(format!(
                "search window exceeds {MAX_SEARCH_LIMIT}"
            )));
        }
        if options.after.is_some() && options.offset != 0 {
            return Err(Error::InvalidQuery(
                "offset and search-after are mutually exclusive".into(),
            ));
        }
        for sort in &options.sort {
            if let Sort::Field { field, .. } = sort {
                let field = self
                    .schema
                    .field(*field)
                    .ok_or_else(|| Error::InvalidQuery("unknown sort field".into()))?;
                if !field.options.sortable {
                    return Err(Error::InvalidQuery(format!(
                        "field {} is not sortable",
                        field.name
                    )));
                }
            }
        }
        for field in &options.stored_fields {
            let field = self
                .schema
                .field(*field)
                .ok_or_else(|| Error::InvalidQuery("unknown stored field".into()))?;
            if !field.options.stored {
                return Err(Error::InvalidQuery(format!(
                    "field {} is not stored",
                    field.name
                )));
            }
        }
        for field in &options.highlight_fields {
            let field = self
                .schema
                .field(*field)
                .ok_or_else(|| Error::InvalidQuery("unknown highlight field".into()))?;
            if !matches!(field.field_type, FieldType::Text(options) if options.positions)
                || !field.options.stored
            {
                return Err(Error::InvalidQuery(format!(
                    "field {} cannot be highlighted",
                    field.name
                )));
            }
        }
        Ok(())
    }

    pub(super) fn validate_query(&self, query: &Query) -> Result<()> {
        match query {
            Query::Text { field, query, .. } => {
                let field = self
                    .schema
                    .field(*field)
                    .ok_or_else(|| Error::InvalidQuery("unknown text field".into()))?;
                if !matches!(field.field_type, FieldType::Text(_)) || !field.options.indexed {
                    return Err(Error::InvalidQuery(
                        "Text query requires indexed Text field".into(),
                    ));
                }
                let count = self.query_tokens(query).len();
                if count > MAX_QUERY_TOKENS {
                    return Err(Error::QueryTooLarge {
                        limit: MAX_QUERY_TOKENS,
                        actual: count,
                    });
                }
            }
            Query::Term { field, value } => {
                let field = self
                    .schema
                    .field(*field)
                    .ok_or_else(|| Error::InvalidQuery("unknown term field".into()))?;
                let valid = field.options.indexed
                    && matches!(
                        (&field.field_type, value),
                        (FieldType::Keyword, Value::String(_))
                            | (FieldType::I64, Value::I64(_))
                            | (FieldType::Bool, Value::Bool(_))
                    );
                if !valid {
                    return Err(Error::InvalidQuery(
                        "Term query requires matching indexed Keyword/I64/Bool field".into(),
                    ));
                }
            }
            Query::Exists(field) => {
                self.schema
                    .field(*field)
                    .ok_or_else(|| Error::InvalidQuery("unknown exists field".into()))?;
            }
            Query::All => {}
            Query::Boolean {
                must,
                should,
                must_not,
                minimum_should_match,
            } => {
                if minimum_should_match.is_some_and(|minimum| minimum as usize > should.len()) {
                    return Err(Error::InvalidQuery(
                        "minimum_should_match exceeds should clauses".into(),
                    ));
                }
                for query in must.iter().chain(should).chain(must_not) {
                    self.validate_query(query)?;
                }
            }
            Query::Boost { query, factor } => {
                if !factor.is_finite() || *factor < 0.0 {
                    return Err(Error::InvalidQuery(
                        "boost factor must be finite and non-negative".into(),
                    ));
                }
                self.validate_query(query)?;
            }
        }
        Ok(())
    }

    pub(super) fn evaluate(
        &self,
        query: &Query,
        allowed: Option<&RoaringBitmap>,
    ) -> Result<EvalResult> {
        let capacity = self.state.documents.slots.len();
        let restrict = |mut docs: RoaringBitmap| {
            if let Some(allowed) = allowed {
                docs &= allowed;
            }
            docs
        };
        Ok(match query {
            Query::All => EvalResult {
                docs: restrict(self.state.live_docs.clone()),
                scores: vec![0.0; capacity],
            },
            Query::Exists(field) => EvalResult {
                docs: restrict(self.state.fields[field.index()].exists().clone()),
                scores: vec![0.0; capacity],
            },
            Query::Term { field, value } => {
                let docs = match (&self.state.fields[field.index()], value) {
                    (FieldIndex::Keyword(index), Value::String(value)) => index.term(value),
                    (FieldIndex::I64(index), Value::I64(value)) => index.term(*value),
                    (FieldIndex::Bool(index), Value::Bool(value)) => index.term(*value),
                    _ => unreachable!(),
                };
                EvalResult {
                    docs: restrict(docs),
                    scores: vec![0.0; capacity],
                }
            }
            Query::Text { field, query, mode } => {
                self.evaluate_text(*field, query, *mode, allowed)?
            }
            Query::Boost { query, factor } => {
                let mut result = self.evaluate(query, allowed)?;
                for doc in result.docs.iter() {
                    result.scores[doc as usize] *= *factor;
                }
                result
            }
            Query::Boolean {
                must,
                should,
                must_not,
                minimum_should_match,
            } => self.evaluate_boolean(must, should, must_not, *minimum_should_match, allowed)?,
        })
    }

    fn evaluate_boolean(
        &self,
        must: &[Query],
        should: &[Query],
        must_not: &[Query],
        minimum: Option<u16>,
        allowed: Option<&RoaringBitmap>,
    ) -> Result<EvalResult> {
        let capacity = self.state.documents.slots.len();
        let mut docs = allowed
            .cloned()
            .unwrap_or_else(|| self.state.live_docs.clone());
        for query in must_not {
            docs -= &self.evaluate(query, Some(&docs))?.docs;
        }
        let mut scores = vec![0.0; capacity];
        for query in must.iter().filter(|query| is_filter(query)) {
            docs &= self.evaluate(query, Some(&docs))?.docs;
        }
        for query in must.iter().filter(|query| !is_filter(query)) {
            let result = self.evaluate(query, Some(&docs))?;
            docs &= &result.docs;
            for doc in docs.iter() {
                scores[doc as usize] += result.scores[doc as usize];
            }
        }
        if should.is_empty() {
            return Ok(EvalResult { docs, scores });
        }
        let required = minimum.unwrap_or(if must.is_empty() { 1 } else { 0 }) as usize;
        let mut counts = vec![0u16; capacity];
        let mut should_scores = vec![0.0; capacity];
        for query in should {
            let result = self.evaluate(query, Some(&docs))?;
            for doc in result.docs.iter() {
                counts[doc as usize] += 1;
                should_scores[doc as usize] += result.scores[doc as usize];
            }
        }
        if required > 0 {
            docs = docs
                .iter()
                .filter(|doc| counts[*doc as usize] as usize >= required)
                .collect();
        }
        for doc in docs.iter() {
            scores[doc as usize] += should_scores[doc as usize];
        }
        Ok(EvalResult { docs, scores })
    }

    fn evaluate_text(
        &self,
        field: FieldId,
        query: &str,
        mode: SearchMode,
        allowed: Option<&RoaringBitmap>,
    ) -> Result<EvalResult> {
        if mode == SearchMode::Auto {
            let exact = self.evaluate_text(field, query, SearchMode::Exact, allowed)?;
            if !exact.docs.is_empty() {
                return Ok(exact);
            }
        }
        let FieldIndex::Text(index) = &self.state.fields[field.index()] else {
            unreachable!()
        };
        let tokens = self.query_tokens(query);
        let capacity = self.state.documents.slots.len();
        if tokens.is_empty() {
            return Ok(EvalResult {
                docs: RoaringBitmap::new(),
                scores: vec![0.0; capacity],
            });
        }
        let mut scores = vec![0.0f32; capacity];
        let mut masks = vec![0u64; capacity];
        let mut touched = Vec::new();
        for (token_index, token) in tokens.iter().enumerate() {
            for (term, domain, weight) in text_candidates(index, &token.term, mode) {
                let postings = index.postings(domain, term);
                if postings.is_empty() {
                    continue;
                }
                let live_count = index.exists.len().max(1) as f32;
                let idf = ((live_count - postings.len() as f32 + 0.5)
                    / (postings.len() as f32 + 0.5)
                    + 1.0)
                    .ln();
                let domain_index = crate::types::domain_index(domain);
                let avg_len = index.total_lengths[domain_index] as f32 / live_count;
                for posting in postings {
                    if allowed.is_some_and(|allowed| !allowed.contains(posting.doc)) {
                        continue;
                    }
                    let slot = posting.doc as usize;
                    if masks[slot] == 0 {
                        touched.push(posting.doc);
                    }
                    masks[slot] |= 1u64 << token_index;
                    let doc_len = index.field_lengths[slot][domain_index] as f32;
                    scores[slot] += bm25(posting.freq as f32, doc_len, avg_len, idf) * weight;
                }
            }
        }
        let required = ((tokens.len() as f32 * MIN_SHOULD_MATCH_RATIO).ceil() as u32).max(1);
        let docs = touched
            .into_iter()
            .filter(|doc| masks[*doc as usize].count_ones() >= required)
            .collect();
        Ok(EvalResult { docs, scores })
    }

    pub(super) fn query_tokens(&self, query: &str) -> Vec<crate::tokenizer::Token> {
        let pipeline = match self.schema.dictionary() {
            Some(dictionary) => {
                Pipeline::new(DefaultTokenizer::for_queries().with_dictionary(dictionary.clone()))
            }
            None => Pipeline::query_pipeline(),
        };
        pipeline
            .query_tokens(query)
            .tokens
            .into_iter()
            .map(|token| crate::tokenizer::Token {
                term: token.term,
                start: token.span.0,
                end: token.span.1,
            })
            .collect()
    }
}

fn is_filter(query: &Query) -> bool {
    match query {
        Query::Text { .. } => false,
        Query::Boost { query, .. } => is_filter(query),
        Query::Boolean {
            must,
            should,
            must_not,
            ..
        } => must.iter().chain(should).chain(must_not).all(is_filter),
        _ => true,
    }
}

pub(super) fn text_candidates(
    index: &TextIndex,
    query: &str,
    mode: SearchMode,
) -> Vec<(TermId, TermDomain, f32)> {
    let mut candidates = Vec::new();
    let exact = index.exact_term(query);
    if mode == SearchMode::Auto
        && let Some(term) = exact
        && !index.postings(TermDomain::Original, term).is_empty()
    {
        return vec![(term, TermDomain::Original, 1.0)];
    }
    if matches!(mode, SearchMode::Exact | SearchMode::Auto)
        && let Some(term) = exact
    {
        candidates.push((term, TermDomain::Original, 1.0));
    }
    if index.options.pinyin && matches!(mode, SearchMode::Pinyin | SearchMode::Auto) {
        if let Some(term) = exact {
            candidates.push((term, TermDomain::PinyinFull, 0.9));
            candidates.push((term, TermDomain::PinyinInitials, 0.8));
        }
        if index.options.prefix {
            for term in index.prefix_terms(query) {
                candidates.push((*term, TermDomain::PinyinFull, 0.85));
                candidates.push((*term, TermDomain::PinyinInitials, 0.75));
            }
        }
    }
    if index.options.fuzzy && matches!(mode, SearchMode::Fuzzy | SearchMode::Auto) {
        let term_ids = (0..index.terms.len() as u32).collect::<Vec<_>>();
        for (term, similarity) in collect_fuzzy_candidates(
            &index.ngrams,
            &term_ids,
            &index.terms,
            query,
            DEFAULT_FUZZY_PARAMS,
            exact,
        ) {
            candidates.push((term, TermDomain::Original, 0.7 * similarity as f32));
        }
    }
    candidates
        .sort_unstable_by_key(|(term, domain, _)| (*term, crate::types::domain_index(*domain)));
    candidates.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    candidates
}

fn bm25(freq: f32, doc_len: f32, avg_len: f32, idf: f32) -> f32 {
    let norm = if avg_len > 0.0 {
        doc_len / avg_len
    } else {
        0.0
    };
    idf * (freq * 2.2) / (freq + 1.2 * (0.25 + 0.75 * norm))
}
