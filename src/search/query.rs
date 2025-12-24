use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use super::{
    super::{
        tokenizer::Token,
        types::{DocData, InMemoryIndex, SearchMode, domain_config},
    },
    MatchedTerm, SearchHit, TermDomain,
    scoring::{
        MIN_SHOULD_MATCH_RATIO, bm25_component, compute_min_should_match, has_minimum_should_match,
        score_fuzzy_terms,
    },
};

struct TermView<'a> {
    term: String,
    postings: &'a HashMap<String, i64>,
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
            if let Some(docs) = self.docs.get(index_name) {
                return docs
                    .keys()
                    .map(|k| SearchHit {
                        doc_id: k.clone(),
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

        let domains = match self.domains.get(index_name) {
            Some(d) => d,
            None => return vec![],
        };

        let domain_index = match domains.get(&domain) {
            Some(idx) => idx,
            None => return vec![],
        };

        let docs = match self.docs.get(index_name) {
            Some(d) => d,
            None => return vec![],
        };

        let mut term_views: Vec<TermView<'_>> = Vec::new();
        let weight = domain_config(domain).weight;

        for token in query_terms {
            let Some(doc_map) = domain_index.postings.get(&token.term) else {
                continue;
            };

            if doc_map.is_empty() {
                continue;
            }

            term_views.push(TermView {
                term: token.term.clone(),
                postings: doc_map,
                weight,
                domain,
            });
        }

        if term_views.is_empty() {
            return vec![];
        }

        let min_should_match =
            compute_min_should_match(query_terms.len(), term_views.len(), MIN_SHOULD_MATCH_RATIO);

        let n = docs.len() as f64;
        if n <= 0.0 {
            return vec![];
        }
        let avgdl = average_doc_len(self, index_name, domain, docs.len());

        let mut idfs = HashMap::new();
        for view in &term_views {
            let n_q = view.postings.len() as f64;
            let idf = ((n - n_q + 0.5) / (n_q + 0.5) + 1.0).ln();
            idfs.insert(view.term.clone(), idf);
        }

        let mut matches: HashMap<String, HashSet<MatchedTerm>> = HashMap::new();
        let mut doc_scores: HashMap<String, f64> = HashMap::new();
        for view in &term_views {
            for (doc_id, freq) in view.postings {
                let Some(doc_data) = docs.get(doc_id) else {
                    continue;
                };
                let idf = *idfs.get(&view.term).unwrap_or(&0.0);
                let component = bm25_component(
                    *freq as f64,
                    doc_len_for_domain(doc_data, view.domain),
                    avgdl,
                    idf,
                ) * view.weight;
                if component > 0.0 {
                    *doc_scores.entry(doc_id.clone()).or_default() += component;
                    matches
                        .entry(doc_id.clone())
                        .or_default()
                        .insert(MatchedTerm::new(view.term.clone(), view.domain));
                }
            }
        }

        let mut scores: Vec<(String, f64)> = doc_scores
            .into_iter()
            .filter(|(doc_id, _)| {
                matches
                    .get(doc_id)
                    .map(|set| set.len() >= min_should_match)
                    .unwrap_or(false)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
            .into_iter()
            .map(|(doc_id, score)| SearchHit {
                doc_id: doc_id.clone(),
                score,
                matched_terms: matches
                    .remove(&doc_id)
                    .map(|s| s.into_iter().collect())
                    .unwrap_or_default(),
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
        let full_prefix = self.bm25_search(index_name, query_terms, TermDomain::PinyinFullPrefix);
        if !full_prefix.is_empty() {
            return full_prefix;
        }

        self.bm25_search(index_name, query_terms, TermDomain::PinyinInitialsPrefix)
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

        let docs = match self.docs.get(index_name) {
            Some(d) => d,
            None => return vec![],
        };

        let domains = match self.domains.get(index_name) {
            Some(d) => d,
            None => return vec![],
        };
        let domain_index = match domains.get(&domain) {
            Some(idx) => idx,
            None => return vec![],
        };

        let n = docs.len() as f64;
        if n <= 0.0 {
            return vec![];
        }
        let avgdl = average_doc_len(self, index_name, domain, docs.len());

        let mut doc_scores: HashMap<String, f64> = HashMap::new();
        let mut matched_terms: HashMap<String, HashSet<MatchedTerm>> = HashMap::new();
        let weight = domain_config(domain).weight;
        let mut matched_query_tokens: HashMap<String, HashSet<usize>> = HashMap::new();
        let mut tokens_with_candidates: HashSet<usize> = HashSet::new();

        for (idx, token) in query_terms.iter().enumerate() {
            score_fuzzy_terms(
                docs,
                domain_index,
                n,
                avgdl,
                &mut doc_scores,
                &mut matched_terms,
                &mut matched_query_tokens,
                &mut tokens_with_candidates,
                domain,
                weight,
                &token.term,
                &|doc_data| doc_len_for_domain(doc_data, domain),
                idx,
            );
        }

        let available_terms = tokens_with_candidates.len();
        let min_should_match =
            // Only count query terms that actually produced fuzzy candidates; otherwise we
            // would unfairly drop hits because of tokens with zero recall paths.
            compute_min_should_match(query_terms.len(), available_terms, MIN_SHOULD_MATCH_RATIO);

        let mut scores: Vec<(String, f64)> = doc_scores
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
            .map(|(doc_id, score)| SearchHit {
                matched_terms: matched_terms
                    .remove(&doc_id)
                    .map(|s| s.into_iter().collect())
                    .unwrap_or_default(),
                doc_id,
                score,
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
    if domain.is_prefix() {
        // Prefix domains reuse positions but skip length normalization so short prefixes
        // are not penalized compared to full tokens.
        return 0.0;
    }

    let len = doc_data.domain_doc_len.get(domain);
    if len > 0 {
        len as f64
    } else {
        doc_data.doc_len as f64
    }
}

fn average_doc_len(
    index: &InMemoryIndex,
    index_name: &str,
    domain: TermDomain,
    doc_count: usize,
) -> f64 {
    if domain.is_prefix() || doc_count == 0 {
        return 0.0;
    }

    let total = index
        .domain_total_lens
        .get(index_name)
        .map(|m| m.get(domain))
        .unwrap_or(0);
    if total <= 0 {
        0.0
    } else {
        total as f64 / doc_count as f64
    }
}
