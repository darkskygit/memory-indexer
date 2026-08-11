use std::collections::HashMap;

use super::index::TermId;

use smol_str::SmolStr;
use strsim::normalized_levenshtein;

#[derive(Debug, Clone, Copy)]
pub struct FuzzyParams {
    pub sim_threshold: f64,
    pub max_candidates: usize,
    pub max_len_diff: usize,
}

// Tuned to keep recall for near-miss typos while capping candidate explosion on short CJK terms.
pub const DEFAULT_FUZZY_PARAMS: FuzzyParams = FuzzyParams {
    sim_threshold: 0.6,
    max_candidates: 20,
    max_len_diff: 2,
};

pub fn collect_fuzzy_candidates(
    ngram_index: &HashMap<SmolStr, Vec<TermId>>,
    term_ids: &[TermId],
    terms: &[SmolStr],
    term: &str,
    params: FuzzyParams,
    exact_term: Option<TermId>,
) -> Vec<(TermId, f64)> {
    let ngrams = generate_ngrams(term);
    if ngrams.is_empty() {
        if let Some(term_id) = exact_term {
            return vec![(term_id, 1.0)];
        }
        return collect_from_term_list(term_ids, terms, term, params);
    }

    let term_len = term.chars().count();
    let sim_threshold = fuzzy_threshold(term_len, params.sim_threshold);
    let mut counts: HashMap<TermId, usize> = HashMap::new();
    for gram in ngrams {
        if let Some(candidates) = ngram_index.get(gram.as_str()) {
            for &candidate in candidates {
                *counts.entry(candidate).or_insert(0) += 1;
            }
        }
    }

    let mut ranked: Vec<(TermId, usize)> = counts.into_iter().collect();
    ranked.sort_by_key(|item| std::cmp::Reverse(item.1));
    ranked.truncate(params.max_candidates.saturating_mul(2));

    let mut filtered = Vec::new();
    for (candidate, _) in ranked {
        let Some(candidate_term) = terms.get(candidate as usize) else {
            continue;
        };
        let candidate_len = candidate_term.chars().count();
        if length_gap_exceeds(term_len, candidate_len, params.max_len_diff) {
            continue;
        }
        let similarity = normalized_levenshtein(candidate_term.as_str(), term);
        if similarity >= sim_threshold {
            filtered.push((candidate, similarity));
        }
        if filtered.len() >= params.max_candidates {
            break;
        }
    }
    filtered
}

fn collect_from_term_list(
    term_ids: &[TermId],
    terms: &[SmolStr],
    term: &str,
    params: FuzzyParams,
) -> Vec<(TermId, f64)> {
    let term_len = term.chars().count();
    let sim_threshold = fuzzy_threshold(term_len, params.sim_threshold);
    let mut candidates: Vec<(TermId, f64)> = term_ids
        .iter()
        .filter_map(|&candidate| {
            let candidate_term = terms.get(candidate as usize)?;
            let candidate_len = candidate_term.chars().count();
            if length_gap_exceeds(term_len, candidate_len, params.max_len_diff) {
                return None;
            }
            let similarity = normalized_levenshtein(candidate_term.as_str(), term);
            (similarity >= sim_threshold).then_some((candidate, similarity))
        })
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(params.max_candidates);
    candidates
}

pub fn generate_ngrams(term: &str) -> Vec<String> {
    let chars: Vec<char> = term.chars().collect();
    match chars.len() {
        0 | 1 => Vec::new(),
        // Keep the whole bigram for very short terms so CJK 2-character words still
        // participate in n-gram lookups.
        2 => vec![chars.iter().collect()],
        _ => chars
            .windows(3)
            .map(|window| window.iter().collect())
            .collect(),
    }
}

fn length_gap_exceeds(a: usize, b: usize, max_gap: usize) -> bool {
    let diff = a.abs_diff(b);
    diff > max_gap
}

fn fuzzy_threshold(term_len: usize, base: f64) -> f64 {
    if term_len <= 2 { base.min(0.5) } else { base }
}
