use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use smol_str::SmolStr;

use super::{
    super::{
        ngram::{build_ngram_index, should_index_in_original_aux},
        tokenizer::Token,
        types::{
            DocData, DocId, InMemoryIndex, IndexState, Posting, SearchMode, TermDomain, TermId,
            domain_config,
        },
    },
    MatchedTerm, SearchHit,
    scoring::{
        MIN_SHOULD_MATCH_RATIO, bm25_component, compute_min_should_match, has_minimum_should_match,
        score_fuzzy_terms,
    },
};

const PINYIN_FULL_PREFIX_MIN: usize = 2;
const PINYIN_INITIALS_PREFIX_MIN: usize = 1;
const PINYIN_PREFIX_MAX: usize = 16;

struct TermView<'a> {
    term_id: TermId,
    term_text: String,
    postings: &'a [Posting],
    weight: f64,
    domain: TermDomain,
}

impl InMemoryIndex {
    /// Execute an auto-mode search and return doc ids with scores.
    pub fn search(&self, index_name: &str, query: &str) -> Vec<(String, f64)> {
        self.search_with_mode_hits(index_name, query, SearchMode::Auto)
            .into_iter()
            .map(|hit| (hit.doc_id, hit.score))
            .collect()
    }

    /// Execute an auto-mode search and return full hits including matched terms.
    pub fn search_hits(&self, index_name: &str, query: &str) -> Vec<SearchHit> {
        self.search_with_mode_hits(index_name, query, SearchMode::Auto)
    }

    /// Execute a search in the specified mode and return doc ids with scores.
    pub fn search_with_mode(
        &self,
        index_name: &str,
        query: &str,
        mode: SearchMode,
    ) -> Vec<(String, f64)> {
        self.search_with_mode_hits(index_name, query, mode)
            .into_iter()
            .map(|hit| (hit.doc_id, hit.score))
            .collect()
    }

    /// Execute a search in the specified mode and return full hits including matched terms.
    pub fn search_with_mode_hits(
        &self,
        index_name: &str,
        query: &str,
        mode: SearchMode,
    ) -> Vec<SearchHit> {
        if query == "*" || query.is_empty() {
            if let Some(state) = self.indexes.get(index_name) {
                return state
                    .doc_index
                    .keys()
                    .map(|doc_id| SearchHit {
                        doc_id: doc_id.to_string(),
                        score: 1.0,
                        matched_terms: Vec::new(),
                    })
                    .collect();
            }
            return vec![];
        }

        let query_terms = self.tokenize_query(query);
        if query_terms.is_empty() {
            return vec![];
        }

        match mode {
            SearchMode::Exact => self.bm25_search(index_name, &query_terms, TermDomain::Original),
            SearchMode::Pinyin => self.pinyin_search(index_name, &query_terms),
            SearchMode::Fuzzy => self.fuzzy_search(index_name, &query_terms),
            SearchMode::Auto => {
                let exact = self.bm25_search(index_name, &query_terms, TermDomain::Original);
                if has_minimum_should_match(&exact, query_terms.len()) {
                    // Stop at exact-domain hits when they already satisfy recall, so we don't
                    // dilute precision by falling through to fuzzier heuristics.
                    return exact;
                }

                if !is_ascii_alphanumeric_query(&query_terms) {
                    return self.fuzzy_search_internal(index_name, &query_terms, true);
                }

                let pinyin_prefix = self.pinyin_prefix_search(index_name, &query_terms);
                if has_minimum_should_match(&pinyin_prefix, query_terms.len()) {
                    return pinyin_prefix;
                }

                let pinyin_exact = self.pinyin_exact_search(index_name, &query_terms);
                if has_minimum_should_match(&pinyin_exact, query_terms.len()) {
                    return pinyin_exact;
                }

                if is_ascii_alphanumeric_query(&query_terms) {
                    let fuzzy_original = self.fuzzy_search(index_name, &query_terms);
                    if !fuzzy_original.is_empty() {
                        return fuzzy_original;
                    }
                } else {
                    let cjk_fuzzy = self.fuzzy_search_internal(index_name, &query_terms, true);
                    if !cjk_fuzzy.is_empty() {
                        return cjk_fuzzy;
                    }
                }

                self.fuzzy_pinyin_search(index_name, &query_terms)
            }
        }
    }

    fn bm25_search(
        &self,
        index_name: &str,
        query_terms: &[Token],
        domain: TermDomain,
    ) -> Vec<SearchHit> {
        if query_terms.is_empty() {
            return vec![];
        }

        let state = match self.indexes.get(index_name) {
            Some(state) => state,
            None => return vec![],
        };

        let domain_index = match state.domains.get(&domain) {
            Some(idx) => idx,
            None => return vec![],
        };

        let doc_count = state.doc_index.len();
        if doc_count == 0 {
            return vec![];
        }

        let mut term_views: Vec<TermView<'_>> = Vec::new();
        let weight = domain_config(domain).weight;

        for token in query_terms {
            let Some(&term_id) = state.term_index.get(token.term.as_str()) else {
                continue;
            };
            let Some(postings) = domain_index.postings.get(&term_id) else {
                continue;
            };
            if postings.is_empty() {
                continue;
            }
            let term_text = state
                .terms
                .get(term_id as usize)
                .map(|term| term.as_str().to_string())
                .unwrap_or_else(|| token.term.clone());
            term_views.push(TermView {
                term_id,
                term_text,
                postings,
                weight,
                domain,
            });
        }

        if term_views.is_empty() {
            return vec![];
        }

        let min_should_match =
            compute_min_should_match(query_terms.len(), term_views.len(), MIN_SHOULD_MATCH_RATIO);

        let n = doc_count as f64;
        let avgdl = average_doc_len(state, domain, doc_count);

        let mut idfs = HashMap::new();
        for view in &term_views {
            let n_q = view.postings.len() as f64;
            let idf = ((n - n_q + 0.5) / (n_q + 0.5) + 1.0).ln();
            idfs.insert(view.term_id, idf);
        }

        let mut matches: HashMap<DocId, HashSet<MatchedTerm>> = HashMap::new();
        let mut doc_scores: HashMap<DocId, f64> = HashMap::new();
        for view in &term_views {
            let idf = *idfs.get(&view.term_id).unwrap_or(&0.0);
            for posting in view.postings {
                let Some(doc_data) = state
                    .docs
                    .get(posting.doc as usize)
                    .and_then(|doc| doc.as_ref())
                else {
                    continue;
                };
                let component = bm25_component(
                    posting.freq as f64,
                    doc_len_for_domain(doc_data, view.domain),
                    avgdl,
                    idf,
                ) * view.weight;
                if component > 0.0 {
                    *doc_scores.entry(posting.doc).or_default() += component;
                    matches
                        .entry(posting.doc)
                        .or_default()
                        .insert(MatchedTerm::new(view.term_text.clone(), view.domain));
                }
            }
        }

        let mut scores: Vec<(DocId, f64)> = doc_scores
            .into_iter()
            .filter(|(doc_id, _)| {
                matches
                    .get(doc_id)
                    .map(|set| set.len() >= min_should_match)
                    .unwrap_or(false)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                let doc_name = state.doc_ids.get(doc_id as usize)?.to_string();
                Some(SearchHit {
                    doc_id: doc_name,
                    score,
                    matched_terms: matches
                        .remove(&doc_id)
                        .map(|s| s.into_iter().collect())
                        .unwrap_or_default(),
                })
            })
            .collect()
    }

    fn pinyin_search(&self, index_name: &str, query_terms: &[Token]) -> Vec<SearchHit> {
        if !is_ascii_alphanumeric_query(query_terms) {
            return vec![];
        }

        let exact = self.pinyin_exact_search(index_name, query_terms);
        if !exact.is_empty() {
            return exact;
        }

        self.pinyin_prefix_search(index_name, query_terms)
    }

    fn pinyin_prefix_search(&self, index_name: &str, query_terms: &[Token]) -> Vec<SearchHit> {
        let full_prefix = self.prefix_search_in_domain(
            index_name,
            query_terms,
            TermDomain::PinyinFull,
            PINYIN_FULL_PREFIX_MIN,
        );
        if !full_prefix.is_empty() {
            return full_prefix;
        }

        self.prefix_search_in_domain(
            index_name,
            query_terms,
            TermDomain::PinyinInitials,
            PINYIN_INITIALS_PREFIX_MIN,
        )
    }

    fn pinyin_exact_search(&self, index_name: &str, query_terms: &[Token]) -> Vec<SearchHit> {
        let full = self.bm25_search(index_name, query_terms, TermDomain::PinyinFull);
        if !full.is_empty() {
            return full;
        }

        self.bm25_search(index_name, query_terms, TermDomain::PinyinInitials)
    }

    fn fuzzy_search(&self, index_name: &str, query_terms: &[Token]) -> Vec<SearchHit> {
        self.fuzzy_search_internal(index_name, query_terms, false)
    }

    fn fuzzy_search_internal(
        &self,
        index_name: &str,
        query_terms: &[Token],
        allow_non_ascii: bool,
    ) -> Vec<SearchHit> {
        self.fuzzy_search_in_domain(
            index_name,
            query_terms,
            TermDomain::Original,
            allow_non_ascii,
        )
    }

    fn fuzzy_pinyin_search(&self, index_name: &str, query_terms: &[Token]) -> Vec<SearchHit> {
        if query_terms.is_empty() || !is_ascii_alphanumeric_query(query_terms) {
            return vec![];
        }

        let full =
            self.fuzzy_search_in_domain(index_name, query_terms, TermDomain::PinyinFull, false);
        if !full.is_empty() {
            return full;
        }

        self.fuzzy_search_in_domain(index_name, query_terms, TermDomain::PinyinInitials, false)
    }

    fn fuzzy_search_in_domain(
        &self,
        index_name: &str,
        query_terms: &[Token],
        domain: TermDomain,
        allow_non_ascii: bool,
    ) -> Vec<SearchHit> {
        if query_terms.is_empty() || (!allow_non_ascii && !is_ascii_alphanumeric_query(query_terms))
        {
            return vec![];
        }

        if !domain_config(domain).allow_fuzzy {
            return vec![];
        }

        let state = match self.indexes.get(index_name) {
            Some(state) => state,
            None => return vec![],
        };

        let domain_index = match state.domains.get(&domain) {
            Some(idx) => idx,
            None => return vec![],
        };

        let doc_count = state.doc_index.len();
        if doc_count == 0 {
            return vec![];
        }

        {
            let mut aux = domain_index.aux.write().unwrap();
            if aux.term_ids.is_none() {
                let mut ids: Vec<TermId> = domain_index
                    .postings
                    .keys()
                    .copied()
                    .filter(|term_id| {
                        if domain == TermDomain::Original {
                            state
                                .terms
                                .get(*term_id as usize)
                                .map(|term| should_index_in_original_aux(term.as_str()))
                                .unwrap_or(false)
                        } else {
                            true
                        }
                    })
                    .collect();
                ids.sort_unstable();
                aux.term_ids = Some(ids);
            }
            if aux.ngram_index.is_none() {
                let ids = aux.term_ids.as_ref().unwrap();
                aux.ngram_index = Some(build_ngram_index(ids, &state.terms));
            }
        }
        let aux = domain_index.aux.read().unwrap();
        let term_ids = aux.term_ids.as_ref().unwrap();
        let ngram_index = aux.ngram_index.as_ref().unwrap();

        let n = doc_count as f64;
        let avgdl = average_doc_len(state, domain, doc_count);

        let mut doc_scores: HashMap<DocId, f64> = HashMap::new();
        let mut matched_terms: HashMap<DocId, HashSet<MatchedTerm>> = HashMap::new();
        let weight = domain_config(domain).weight;
        let mut matched_query_tokens: HashMap<DocId, HashSet<usize>> = HashMap::new();
        let mut tokens_with_candidates: HashSet<usize> = HashSet::new();

        for (idx, token) in query_terms.iter().enumerate() {
            let exact_term = state.term_index.get(token.term.as_str()).copied();
            score_fuzzy_terms(
                &state.docs,
                domain_index,
                term_ids,
                &state.terms,
                ngram_index,
                n,
                avgdl,
                &mut doc_scores,
                &mut matched_terms,
                &mut matched_query_tokens,
                &mut tokens_with_candidates,
                domain,
                weight,
                &token.term,
                idx,
                exact_term,
            );
        }

        let available_terms = tokens_with_candidates.len();
        let min_should_match =
            // Only count query terms that actually produced fuzzy candidates; otherwise we
            // would unfairly drop hits because of tokens with zero recall paths.
            compute_min_should_match(query_terms.len(), available_terms, MIN_SHOULD_MATCH_RATIO);

        let mut scores: Vec<(DocId, f64)> = doc_scores
            .into_iter()
            .filter(|(doc_id, _)| {
                matched_query_tokens
                    .get(doc_id)
                    .map(|set| set.len() >= min_should_match)
                    .unwrap_or(false)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                let doc_name = state.doc_ids.get(doc_id as usize)?.to_string();
                Some(SearchHit {
                    matched_terms: matched_terms
                        .remove(&doc_id)
                        .map(|s| s.into_iter().collect())
                        .unwrap_or_default(),
                    doc_id: doc_name,
                    score,
                })
            })
            .collect()
    }

    fn prefix_search_in_domain(
        &self,
        index_name: &str,
        query_terms: &[Token],
        domain: TermDomain,
        min_prefix_len: usize,
    ) -> Vec<SearchHit> {
        if query_terms.is_empty() || !is_ascii_alphanumeric_query(query_terms) {
            return vec![];
        }

        let state = match self.indexes.get(index_name) {
            Some(state) => state,
            None => return vec![],
        };

        let domain_index = match state.domains.get(&domain) {
            Some(idx) => idx,
            None => return vec![],
        };

        let doc_count = state.doc_index.len();
        if doc_count == 0 {
            return vec![];
        }

        {
            let mut aux = domain_index.aux.write().unwrap();
            if aux.term_ids.is_none() {
                let mut ids: Vec<TermId> = domain_index.postings.keys().copied().collect();
                ids.sort_unstable();
                aux.term_ids = Some(ids);
            }
            if aux.prefix_index.is_none() {
                let mut prefix_index: HashMap<SmolStr, Vec<TermId>> = HashMap::new();
                let ids = aux.term_ids.as_ref().unwrap();
                for &term_id in ids {
                    let Some(term) = state.terms.get(term_id as usize) else {
                        continue;
                    };
                    if !term.as_str().is_ascii() {
                        continue;
                    }
                    let term_len = term.len();
                    if term_len < min_prefix_len {
                        continue;
                    }
                    let max = PINYIN_PREFIX_MAX.min(term_len);
                    for len in min_prefix_len..=max {
                        let prefix = SmolStr::new(&term.as_str()[..len]);
                        prefix_index.entry(prefix).or_default().push(term_id);
                    }
                }
                aux.prefix_index = Some(prefix_index);
            }
        }
        let aux = domain_index.aux.read().unwrap();
        let prefix_index = aux.prefix_index.as_ref().unwrap();

        let n = doc_count as f64;
        let avgdl = average_doc_len(state, domain, doc_count);

        let mut doc_scores: HashMap<DocId, f64> = HashMap::new();
        let mut matched_terms: HashMap<DocId, HashSet<MatchedTerm>> = HashMap::new();
        let weight = domain_config(domain).weight;
        let mut matched_query_tokens: HashMap<DocId, HashSet<usize>> = HashMap::new();
        let mut tokens_with_candidates: HashSet<usize> = HashSet::new();

        for (idx, token) in query_terms.iter().enumerate() {
            if token.term.len() < min_prefix_len || token.term.len() > PINYIN_PREFIX_MAX {
                continue;
            }

            let Some(candidates) = prefix_index.get(token.term.as_str()) else {
                continue;
            };
            if candidates.is_empty() {
                continue;
            }

            tokens_with_candidates.insert(idx);

            for &candidate in candidates {
                let Some(postings) = domain_index.postings.get(&candidate) else {
                    continue;
                };
                if postings.is_empty() {
                    continue;
                }

                let n_q = postings.len() as f64;
                let idf = ((n - n_q + 0.5) / (n_q + 0.5) + 1.0).ln();
                let candidate_text = state
                    .terms
                    .get(candidate as usize)
                    .map(|term| term.as_str().to_string())
                    .unwrap_or_else(|| token.term.clone());

                for posting in postings {
                    let Some(doc_data) = state
                        .docs
                        .get(posting.doc as usize)
                        .and_then(|doc| doc.as_ref())
                    else {
                        continue;
                    };
                    let term_score = bm25_component(
                        posting.freq as f64,
                        doc_len_for_domain(doc_data, domain),
                        avgdl,
                        idf,
                    ) * weight;
                    if term_score > 0.0 {
                        *doc_scores.entry(posting.doc).or_default() += term_score;
                        matched_terms
                            .entry(posting.doc)
                            .or_default()
                            .insert(MatchedTerm::new(candidate_text.clone(), domain));
                        matched_query_tokens
                            .entry(posting.doc)
                            .or_default()
                            .insert(idx);
                    }
                }
            }
        }

        let available_terms = tokens_with_candidates.len();
        let min_should_match =
            compute_min_should_match(query_terms.len(), available_terms, MIN_SHOULD_MATCH_RATIO);

        let mut scores: Vec<(DocId, f64)> = doc_scores
            .into_iter()
            .filter(|(doc_id, _)| {
                matched_query_tokens
                    .get(doc_id)
                    .map(|set| set.len() >= min_should_match)
                    .unwrap_or(false)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                let doc_name = state.doc_ids.get(doc_id as usize)?.to_string();
                Some(SearchHit {
                    matched_terms: matched_terms
                        .remove(&doc_id)
                        .map(|s| s.into_iter().collect())
                        .unwrap_or_default(),
                    doc_id: doc_name,
                    score,
                })
            })
            .collect()
    }
}

pub(super) fn is_ascii_alphanumeric_query(tokens: &[Token]) -> bool {
    tokens
        .iter()
        .all(|token| token.term.chars().all(|c| c.is_ascii_alphanumeric()))
}

fn doc_len_for_domain(doc_data: &DocData, domain: TermDomain) -> f64 {
    let len = doc_data.domain_doc_len.get(domain);
    if len > 0 {
        len as f64
    } else {
        doc_data.doc_len as f64
    }
}

fn average_doc_len(state: &IndexState, domain: TermDomain, doc_count: usize) -> f64 {
    if doc_count == 0 {
        return 0.0;
    }

    let total = state.domain_total_len.get(domain);
    if total <= 0 {
        0.0
    } else {
        total as f64 / doc_count as f64
    }
}
