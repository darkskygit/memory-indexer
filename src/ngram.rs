use std::collections::{HashMap, HashSet};

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
    ngram_index: &HashMap<String, Vec<String>>,
    term_dict: &HashSet<String>,
    term: &str,
    params: FuzzyParams,
) -> Vec<(String, f64)> {
    let ngrams = generate_ngrams(term);
    if ngrams.is_empty() {
        if term_dict.contains(term) {
            return vec![(term.to_string(), 1.0)];
        }
        return collect_from_term_dict(term_dict, term, params);
    }

    let term_len = term.chars().count();
    let sim_threshold = fuzzy_threshold(term_len, params.sim_threshold);
    let mut counts: HashMap<String, usize> = HashMap::new();
    for gram in ngrams {
        if let Some(terms) = ngram_index.get(&gram) {
            for candidate in terms {
                *counts.entry(candidate.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    ranked.truncate(params.max_candidates.saturating_mul(2));

    let mut filtered = Vec::new();
    for (candidate, _) in ranked {
        let candidate_len = candidate.chars().count();
        if length_gap_exceeds(term_len, candidate_len, params.max_len_diff) {
            continue;
        }
        let similarity = normalized_levenshtein(&candidate, term);
        if similarity >= sim_threshold {
            filtered.push((candidate, similarity));
        }
        if filtered.len() >= params.max_candidates {
            break;
        }
    }
    filtered
}

fn collect_from_term_dict(
    term_dict: &HashSet<String>,
    term: &str,
    params: FuzzyParams,
) -> Vec<(String, f64)> {
    let term_len = term.chars().count();
    let sim_threshold = fuzzy_threshold(term_len, params.sim_threshold);
    let mut candidates: Vec<(String, f64)> = term_dict
        .iter()
        .filter(|candidate| {
            let candidate_len = candidate.chars().count();
            !length_gap_exceeds(term_len, candidate_len, params.max_len_diff)
        })
        .filter_map(|candidate| {
            let similarity = normalized_levenshtein(candidate, term);
            (similarity >= sim_threshold).then_some((candidate.clone(), similarity))
        })
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(params.max_candidates);
    candidates
}

pub fn add_term_to_ngrams(index: &mut HashMap<String, Vec<String>>, term: &str) {
    for gram in generate_ngrams(term) {
        let terms = index.entry(gram).or_default();
        if !terms.contains(&term.to_string()) {
            terms.push(term.to_string());
        }
    }
}

pub fn remove_term_from_ngrams(index: &mut HashMap<String, Vec<String>>, term: &str) {
    for gram in generate_ngrams(term) {
        if let Some(terms) = index.get_mut(&gram) {
            terms.retain(|t| t != term);
            if terms.is_empty() {
                index.remove(&gram);
            }
        }
    }
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
    let diff = if a >= b { a - b } else { b - a };
    diff > max_gap
}

fn fuzzy_threshold(term_len: usize, base: f64) -> f64 {
    if term_len <= 2 { base.min(0.5) } else { base }
}

pub fn should_index_in_original_aux(term: &str) -> bool {
    term.chars().any(|c| c.is_alphanumeric())
}
