use std::collections::HashMap;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::{
    TextOptions,
    ngram::generate_ngrams,
    types::{TermDomain, domain_index},
};

use super::{DocId, TermId};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Posting {
    pub(crate) doc: DocId,
    pub(crate) freq: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct TermPosition {
    pub(crate) term: TermId,
    pub(crate) start: u32,
    pub(crate) end: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DerivedTerm {
    pub(crate) derived: TermId,
    pub(crate) base: TermId,
    pub(crate) span: (u32, u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TextValueState {
    pub(crate) positions: Vec<TermPosition>,
    pub(crate) derived: Vec<DerivedTerm>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct TextDocState {
    pub(crate) values: Vec<TextValueState>,
    pub(crate) terms: Vec<TextTermState>,
    pub(crate) lengths: [u32; 3],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct TextTermState {
    pub(crate) term: TermId,
    pub(crate) domains: u8,
}

#[derive(Debug)]
pub(crate) struct PreparedText {
    pub(crate) values: Vec<PreparedTextValue>,
}

#[derive(Debug)]
pub(crate) struct PreparedTextValue {
    pub(crate) tokens: Vec<crate::types::PipelineToken>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TextIndex {
    pub(crate) options: TextOptions,
    pub(crate) terms: Vec<SmolStr>,
    #[serde(skip, default)]
    pub(crate) term_ids: HashMap<SmolStr, TermId>,
    pub(crate) postings: [Vec<Vec<Posting>>; 3],
    pub(crate) exists: RoaringBitmap,
    pub(crate) field_lengths: Vec<[u32; 3]>,
    pub(crate) total_lengths: [u64; 3],
    pub(crate) doc_states: Vec<Option<TextDocState>>,
    pub(crate) ngrams: HashMap<SmolStr, Vec<TermId>>,
    pub(crate) prefixes: HashMap<SmolStr, Vec<TermId>>,
}

impl TextIndex {
    pub(crate) fn rebuild_lookup_map(&mut self) {
        self.term_ids = self
            .terms
            .iter()
            .enumerate()
            .map(|(id, term)| (term.clone(), id as TermId))
            .collect();
    }

    pub(crate) fn new(options: TextOptions) -> Self {
        Self {
            options,
            terms: Vec::new(),
            term_ids: HashMap::new(),
            postings: [Vec::new(), Vec::new(), Vec::new()],
            exists: RoaringBitmap::new(),
            field_lengths: Vec::new(),
            total_lengths: [0; 3],
            doc_states: Vec::new(),
            ngrams: HashMap::new(),
            prefixes: HashMap::new(),
        }
    }

    pub(crate) fn ensure_doc_capacity(&mut self, capacity: usize) {
        if self.field_lengths.len() < capacity {
            self.field_lengths.resize(capacity, [0; 3]);
        }
        if self.doc_states.len() < capacity {
            self.doc_states.resize_with(capacity, || None);
        }
    }

    fn intern(&mut self, term: &str) -> TermId {
        if let Some(id) = self.term_ids.get(term) {
            return *id;
        }
        let id = self.terms.len() as TermId;
        let term = SmolStr::new(term);
        self.terms.push(term.clone());
        self.term_ids.insert(term.clone(), id);
        for postings in &mut self.postings {
            postings.push(Vec::new());
        }
        id
    }

    fn register_recall(&mut self, term: TermId, domain: TermDomain) {
        let value = &self.terms[term as usize];
        if domain == TermDomain::Original {
            for gram in generate_ngrams(value.as_str()) {
                let terms = self.ngrams.entry(SmolStr::new(gram)).or_default();
                if terms.last() != Some(&term) {
                    terms.push(term);
                }
            }
        } else if self.options.prefix {
            let mut prefix = String::new();
            for character in value.chars().take(16) {
                prefix.push(character);
                let terms = self.prefixes.entry(SmolStr::new(&prefix)).or_default();
                if terms.last() != Some(&term) {
                    terms.push(term);
                }
            }
        }
    }

    pub(crate) fn insert(&mut self, doc: DocId, prepared: PreparedText) {
        let mut frequencies: HashMap<TermId, [u32; 3]> = HashMap::new();
        let mut values = Vec::with_capacity(prepared.values.len());
        let mut lengths = [0u32; 3];
        for prepared_value in prepared.values {
            let mut positions = Vec::new();
            let mut derived = Vec::new();
            for token in prepared_value.tokens {
                let term = self.intern(&token.term);
                self.register_recall(term, token.domain);
                let domain = domain_index(token.domain);
                frequencies.entry(term).or_insert([0; 3])[domain] += 1;
                lengths[domain] += 1;
                if token.domain == TermDomain::Original {
                    positions.push(TermPosition {
                        term,
                        start: token.span.0 as u32,
                        end: token.span.1 as u32,
                    });
                } else {
                    let base = self.intern(&token.base_term);
                    derived.push(DerivedTerm {
                        derived: term,
                        base,
                        span: (token.span.0 as u32, token.span.1 as u32),
                    });
                }
            }
            positions.sort_unstable_by_key(|entry| (entry.term, entry.start, entry.end));
            positions.dedup_by_key(|entry| (entry.term, entry.start, entry.end));
            derived.sort_unstable_by_key(|entry| (entry.derived, entry.base, entry.span));
            derived.dedup_by_key(|entry| (entry.derived, entry.base, entry.span));
            values.push(TextValueState { positions, derived });
        }
        let mut term_freqs = frequencies.into_iter().collect::<Vec<_>>();
        term_freqs.sort_unstable_by_key(|entry| entry.0);
        for (term, counts) in &term_freqs {
            for (domain, count) in counts.iter().copied().enumerate() {
                if count == 0 {
                    continue;
                }
                let postings = &mut self.postings[domain][*term as usize];
                match postings.binary_search_by_key(&doc, |posting| posting.doc) {
                    Ok(index) => postings[index].freq = count,
                    Err(index) => postings.insert(index, Posting { doc, freq: count }),
                }
            }
        }
        self.field_lengths[doc as usize] = lengths;
        for (total, length) in self.total_lengths.iter_mut().zip(lengths) {
            *total += length as u64;
        }
        self.exists.insert(doc);
        let terms = term_freqs
            .iter()
            .map(|(term, counts)| TextTermState {
                term: *term,
                domains: counts.iter().enumerate().fold(0, |mask, (domain, count)| {
                    mask | ((*count > 0) as u8) << domain
                }),
            })
            .collect();
        self.doc_states[doc as usize] = Some(TextDocState {
            values,
            terms,
            lengths,
        });
    }

    pub(crate) fn remove(&mut self, doc: DocId) {
        let Some(state) = self.doc_states[doc as usize].take() else {
            return;
        };
        for term in &state.terms {
            for domain in 0..3 {
                if term.domains & (1 << domain) == 0 {
                    continue;
                }
                let postings = &mut self.postings[domain][term.term as usize];
                if let Ok(index) = postings.binary_search_by_key(&doc, |posting| posting.doc) {
                    postings.remove(index);
                }
            }
        }
        for (total, length) in self.total_lengths.iter_mut().zip(state.lengths) {
            *total = total.saturating_sub(length as u64);
        }
        self.field_lengths[doc as usize] = [0; 3];
        self.exists.remove(doc);
    }

    pub(crate) fn exact_term(&self, term: &str) -> Option<TermId> {
        self.term_ids.get(term).copied()
    }

    pub(crate) fn prefix_terms(&self, prefix: &str) -> &[TermId] {
        self.prefixes
            .get(prefix)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub(crate) fn postings(&self, domain: TermDomain, term: TermId) -> &[Posting] {
        self.postings[domain_index(domain)]
            .get(term as usize)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub(crate) fn optimize(&mut self) -> (Vec<Option<TermId>>, usize) {
        let mut mapping = vec![None; self.terms.len()];
        let mut next = 0;
        for (old, mapped) in mapping.iter_mut().enumerate() {
            if self.postings.iter().any(|domain| !domain[old].is_empty()) {
                *mapped = Some(next);
                next += 1;
            }
        }
        let removed = mapping.len() - next as usize;
        if removed == 0 {
            return (mapping, 0);
        }
        self.terms = std::mem::take(&mut self.terms)
            .into_iter()
            .enumerate()
            .filter_map(|(old, term)| mapping[old].map(|_| term))
            .collect();
        for domain in &mut self.postings {
            *domain = std::mem::take(domain)
                .into_iter()
                .enumerate()
                .filter_map(|(old, postings)| mapping[old].map(|_| postings))
                .collect();
        }
        self.term_ids = self
            .terms
            .iter()
            .enumerate()
            .map(|(id, term)| (term.clone(), id as TermId))
            .collect();
        self.ngrams.clear();
        self.prefixes.clear();
        for id in 0..self.terms.len() as TermId {
            if !self.postings[0][id as usize].is_empty() {
                self.register_recall(id, TermDomain::Original);
            }
            if !self.postings[1][id as usize].is_empty() {
                self.register_recall(id, TermDomain::PinyinFull);
            }
            if !self.postings[2][id as usize].is_empty() {
                self.register_recall(id, TermDomain::PinyinInitials);
            }
        }
        (mapping, removed)
    }
}
