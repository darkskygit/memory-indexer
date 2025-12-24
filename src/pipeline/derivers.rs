use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
};

use crate::{
    tokenizer::build_pinyin_variants,
    types::{NormalizedTerm, PipelineToken, TermDomain, TermFrequency, TokenDraft},
};

// Prefix bounds to avoid exploding prefix variants: skip 1-char full prefixes and cap both forms.
const PINYIN_FULL_PREFIX_MIN: usize = 2;
const PINYIN_FULL_PREFIX_MAX: usize = 16;
const PINYIN_INITIALS_PREFIX_MIN: usize = 1;
const PINYIN_INITIALS_PREFIX_MAX: usize = 16;

pub struct PassthroughNormalizer;

impl PassthroughNormalizer {
    pub fn normalize(&self, draft: &TokenDraft) -> Vec<NormalizedTerm> {
        vec![NormalizedTerm {
            term: draft.text.clone(),
            span: draft.span,
            script: draft.script,
            mapping: draft.mapping.clone(),
        }]
    }
}

pub struct PinyinVariantDeriver {
    cache: RefCell<SimpleLru<String, Vec<(String, String)>>>,
}
// Cache derived pinyin variants to avoid recomputing polyphonic forms while staying memory-light.
const PINYIN_CACHE_CAPACITY: usize = 512;

impl PinyinVariantDeriver {
    pub fn new() -> Self {
        Self {
            cache: RefCell::new(SimpleLru::new(PINYIN_CACHE_CAPACITY)),
        }
    }
}

impl Default for PinyinVariantDeriver {
    fn default() -> Self {
        Self::new()
    }
}

impl PinyinVariantDeriver {
    pub fn derive(&self, term: &NormalizedTerm) -> Vec<PipelineToken> {
        let variants = {
            let mut cache = self.cache.borrow_mut();
            if let Some(cached) = cache.get(&term.term) {
                cached
            } else {
                let computed = build_pinyin_variants(&term.term);
                cache.put(term.term.clone(), computed.clone());
                computed
            }
        };

        let mut tokens = Vec::new();
        for (full, initials) in variants {
            if !full.is_empty() {
                tokens.push(PipelineToken {
                    term: full.clone(),
                    span: term.span,
                    domain: TermDomain::PinyinFull,
                    base_term: term.term.clone(),
                });
            }
            if !initials.is_empty() && initials != full {
                tokens.push(PipelineToken {
                    term: initials,
                    span: term.span,
                    domain: TermDomain::PinyinInitials,
                    base_term: term.term.clone(),
                });
            }
        }
        tokens
    }
}

pub struct PrefixDerivers {
    full: AsciiPrefixDeriver,
    initials: AsciiPrefixDeriver,
}

impl PrefixDerivers {
    pub fn pinyin_defaults() -> Self {
        Self {
            full: AsciiPrefixDeriver::new(
                TermDomain::PinyinFullPrefix,
                PINYIN_FULL_PREFIX_MIN,
                PINYIN_FULL_PREFIX_MAX,
                |freqs| freqs.get(TermDomain::PinyinFull) > 0,
            ),
            initials: AsciiPrefixDeriver::new(
                TermDomain::PinyinInitialsPrefix,
                PINYIN_INITIALS_PREFIX_MIN,
                PINYIN_INITIALS_PREFIX_MAX,
                |freqs| freqs.get(TermDomain::PinyinInitials) > 0,
            ),
        }
    }
}

impl PrefixDerivers {
    pub fn derive_prefixes(
        &self,
        token: &PipelineToken,
        term_freqs: &HashMap<String, TermFrequency>,
    ) -> Vec<PipelineToken> {
        match token.domain {
            TermDomain::PinyinFull => self.full.derive_prefixes(token, term_freqs),
            TermDomain::PinyinInitials => self.initials.derive_prefixes(token, term_freqs),
            _ => Vec::new(),
        }
    }
}

pub struct AsciiPrefixDeriver {
    domain: TermDomain,
    min_len: usize,
    max_len: usize,
    should_skip: fn(&TermFrequency) -> bool,
}

impl AsciiPrefixDeriver {
    pub fn new(
        domain: TermDomain,
        min_len: usize,
        max_len: usize,
        should_skip: fn(&TermFrequency) -> bool,
    ) -> Self {
        Self {
            domain,
            min_len,
            max_len,
            should_skip,
        }
    }
}

impl AsciiPrefixDeriver {
    pub fn derive_prefixes(
        &self,
        token: &PipelineToken,
        term_freqs: &HashMap<String, TermFrequency>,
    ) -> Vec<PipelineToken> {
        if !token.term.is_ascii() {
            return Vec::new();
        }
        let mut prefixes = Vec::new();
        let max_len = self.max_len.min(token.term.len().saturating_sub(1));
        if max_len < self.min_len {
            return prefixes;
        }

        for len in self.min_len..=max_len {
            let prefix = token.term[..len].to_string();
            if term_freqs
                .get(&prefix)
                .is_some_and(|freqs| (self.should_skip)(freqs))
            {
                continue;
            }
            prefixes.push(PipelineToken {
                term: prefix,
                span: token.span,
                domain: self.domain,
                base_term: token.base_term.clone(),
            });
        }
        prefixes
    }
}

pub struct NoopNgramDeriver;

impl NoopNgramDeriver {
    pub fn derive_ngrams(&self, _token: &PipelineToken) -> Vec<PipelineToken> {
        Vec::new()
    }
}

#[derive(Debug)]
struct SimpleLru<K, V> {
    map: HashMap<K, V>,
    order: VecDeque<K>,
    capacity: usize,
}

impl<K, V> SimpleLru<K, V>
where
    K: std::cmp::Eq + std::hash::Hash + Clone,
{
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    fn get(&mut self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        if self.map.contains_key(key) {
            self.touch(key);
        }
        self.map.get(key).cloned()
    }

    fn put(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            self.touch(&key);
            self.map.insert(key, value);
            return;
        }
        if self.map.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }

    fn touch(&mut self, key: &K) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
    }
}
