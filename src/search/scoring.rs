use std::collections::{HashMap, HashSet};

use super::{
    super::{
        ngram::{DEFAULT_FUZZY_PARAMS, collect_fuzzy_candidates},
        types::{DocData, DomainIndex},
    },
    MatchedTerm, SearchHit, TermDomain,
};

// Standard BM25 defaults that balance term frequency and doc length without over-penalizing short docs.
pub(super) const BM25_K1: f64 = 1.2;
pub(super) const BM25_B: f64 = 0.75;
// Down-weight fuzzy matches relative to exact matches to favor precise hits when both exist.
pub(super) const FUZZY_WEIGHT: f64 = 0.7;
// Require ~60% of query tokens to match so multi-term queries still filter low-signal hits.
pub(super) const MIN_SHOULD_MATCH_RATIO: f64 = 0.6;
// Always insist on at least one match even for very short queries.
pub(super) const MIN_SHOULD_MATCH_FLOOR: usize = 1;

pub(super) fn bm25_component(freq: f64, doc_len: f64, avgdl: f64, idf: f64) -> f64 {
    if freq <= 0.0 || idf <= 0.0 {
        return 0.0;
    }
    let norm_dl = if avgdl > 0.0 { doc_len / avgdl } else { 0.0 };
    let numerator = freq * (BM25_K1 + 1.0);
    let denominator = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * norm_dl);
    if denominator == 0.0 {
        0.0
    } else {
        idf * (numerator / denominator)
    }
}

pub(super) fn compute_min_should_match(
    query_terms: usize,
    available_terms: usize,
    ratio: f64,
) -> usize {
    if available_terms == 0 {
        return 0;
    }
    let required = ((query_terms as f64) * ratio).ceil() as usize;
    let required = required.max(MIN_SHOULD_MATCH_FLOOR);
    required.min(available_terms)
}

pub(super) fn has_minimum_should_match(results: &[SearchHit], query_terms: usize) -> bool {
    if results.is_empty() {
        return false;
    }
    let required = compute_min_should_match(query_terms, query_terms, MIN_SHOULD_MATCH_RATIO);
    results
        .iter()
        .any(|hit| hit.matched_terms.len() >= required)
}

pub(super) fn score_fuzzy_terms(
    docs: &HashMap<String, DocData>,
    domain_index: &DomainIndex,
    n: f64,
    avgdl: f64,
    doc_scores: &mut HashMap<String, f64>,
    matched_terms: &mut HashMap<String, HashSet<MatchedTerm>>,
    matched_query_tokens: &mut HashMap<String, HashSet<usize>>,
    tokens_with_candidates: &mut HashSet<usize>,
    domain: TermDomain,
    weight: f64,
    query_term: &str,
    doc_len_for_domain: &dyn Fn(&DocData) -> f64,
    query_idx: usize,
) {
    let candidates = collect_fuzzy_candidates(
        &domain_index.ngram_index,
        &domain_index.term_dict,
        query_term,
        DEFAULT_FUZZY_PARAMS,
    );
    if candidates.is_empty() {
        return;
    }
    tokens_with_candidates.insert(query_idx);
    for (candidate_term, similarity) in candidates {
        let Some(doc_map) = domain_index.postings.get(&candidate_term) else {
            continue;
        };
        if doc_map.is_empty() {
            continue;
        }

        let n_q = doc_map.len() as f64;
        let idf = ((n - n_q + 0.5) / (n_q + 0.5) + 1.0).ln();

        for (doc_id, freq) in doc_map {
            if let Some(doc_data) = docs.get(doc_id) {
                let term_score =
                    bm25_component(*freq as f64, doc_len_for_domain(doc_data), avgdl, idf)
                        * weight
                        * FUZZY_WEIGHT
                        * similarity;
                if term_score > 0.0 {
                    *doc_scores.entry(doc_id.clone()).or_default() += term_score;
                    matched_terms
                        .entry(doc_id.clone())
                        .or_default()
                        .insert(MatchedTerm::new(candidate_term.clone(), domain));
                    matched_query_tokens
                        .entry(doc_id.clone())
                        .or_default()
                        .insert(query_idx);
                }
            }
        }
    }
}
