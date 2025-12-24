use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::tokenizer::{DictionaryConfig, OffsetMap, SegmentScript, TokenWithScript};

pub const SNAPSHOT_VERSION: u32 = 4;

/// Search execution strategy for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Only search original terms.
    Exact,
    /// Search derived pinyin domains.
    Pinyin,
    /// Allow fuzzy matching for tolerant recall.
    Fuzzy,
    /// Try exact first, then pinyin and fuzzy fallbacks.
    Auto,
}

/// Token domain representing how a term was derived or transformed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TermDomain {
    Original,
    PinyinFull,
    PinyinInitials,
    PinyinFullPrefix,
    PinyinInitialsPrefix,
}

const TERM_DOMAIN_COUNT: usize = 5;

const fn domain_index(domain: TermDomain) -> usize {
    match domain {
        TermDomain::Original => 0,
        TermDomain::PinyinFull => 1,
        TermDomain::PinyinInitials => 2,
        TermDomain::PinyinFullPrefix => 3,
        TermDomain::PinyinInitialsPrefix => 4,
    }
}

impl TermDomain {
    /// Returns true if the domain represents a pinyin-derived token.
    pub fn is_pinyin(&self) -> bool {
        matches!(
            self,
            TermDomain::PinyinFull
                | TermDomain::PinyinInitials
                | TermDomain::PinyinFullPrefix
                | TermDomain::PinyinInitialsPrefix
        )
    }

    /// Returns true if the domain stores prefix tokens rather than full terms.
    pub fn is_prefix(&self) -> bool {
        matches!(
            self,
            TermDomain::PinyinFullPrefix | TermDomain::PinyinInitialsPrefix
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DomainConfig {
    pub weight: f64,
    pub enable_ngrams: bool,
    pub allow_fuzzy: bool,
}

pub fn domain_config(domain: TermDomain) -> DomainConfig {
    match domain {
        TermDomain::Original => DomainConfig {
            weight: 1.0,
            enable_ngrams: true,
            allow_fuzzy: true,
        },
        TermDomain::PinyinFull => DomainConfig {
            weight: 0.9,
            enable_ngrams: true,
            allow_fuzzy: true,
        },
        TermDomain::PinyinInitials => DomainConfig {
            weight: 0.8,
            enable_ngrams: true,
            allow_fuzzy: true,
        },
        TermDomain::PinyinFullPrefix => DomainConfig {
            weight: 0.7,
            enable_ngrams: false,
            allow_fuzzy: false,
        },
        TermDomain::PinyinInitialsPrefix => DomainConfig {
            weight: 0.75,
            enable_ngrams: false,
            allow_fuzzy: false,
        },
    }
}

pub fn all_domains() -> &'static [TermDomain] {
    &[
        TermDomain::Original,
        TermDomain::PinyinFull,
        TermDomain::PinyinInitials,
        TermDomain::PinyinFullPrefix,
        TermDomain::PinyinInitialsPrefix,
    ]
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DomainIndex {
    pub postings: HashMap<String, HashMap<String, i64>>,
    pub term_dict: HashSet<String>,
    pub ngram_index: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TermFrequency {
    pub counts: HashMap<TermDomain, u32>,
}

impl TermFrequency {
    pub fn increment(&mut self, domain: TermDomain) {
        *self.counts.entry(domain).or_default() += 1;
    }

    pub fn get(&self, domain: TermDomain) -> u32 {
        *self.counts.get(&domain).unwrap_or(&0)
    }

    pub fn positive_domains(&self) -> Vec<(TermDomain, u32)> {
        let mut domains = Vec::new();
        for domain in all_domains() {
            if let Some(count) = self.counts.get(domain) {
                if *count > 0 {
                    domains.push((*domain, *count));
                }
            }
        }
        domains
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocData {
    pub content: String,
    /// Document length in normalized tokens.
    pub doc_len: i64,
    /// Term positions for original-domain tokens.
    pub term_pos: HashMap<String, Vec<(u32, u32)>>,
    #[serde(default)]
    pub term_freqs: HashMap<String, TermFrequency>,
    #[serde(default)]
    pub domain_doc_len: DomainLengths,
    #[serde(default)]
    pub derived_terms: HashMap<String, Vec<(u32, u32)>>,
}

/// In-memory inverted index supporting exact, pinyin, and fuzzy search over documents.
#[derive(Debug)]
pub struct InMemoryIndex {
    pub versions: HashMap<String, u32>,
    pub docs: HashMap<String, HashMap<String, DocData>>,
    pub domains: HashMap<String, HashMap<TermDomain, DomainIndex>>,
    pub total_lens: HashMap<String, i64>,
    pub domain_total_lens: HashMap<String, DomainLengths>,
    pub dirty: HashMap<String, HashSet<String>>,
    pub deleted: HashMap<String, HashSet<String>>,
    pub position_encoding: PositionEncoding,
    pub dictionary: Option<DictionaryConfig>,
}

#[derive(Debug, Clone, Copy)]
pub struct Segment<'a> {
    pub script: SegmentScript,
    pub text: &'a str,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenDraft {
    pub text: String,
    pub span: (usize, usize),
    pub script: SegmentScript,
    pub mapping: OffsetMap,
}

impl From<TokenWithScript> for TokenDraft {
    fn from(value: TokenWithScript) -> Self {
        Self {
            text: value.term,
            span: (value.start, value.end),
            script: value.script,
            mapping: value.offset_map,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedTerm {
    pub term: String,
    pub span: (usize, usize),
    pub script: SegmentScript,
    pub mapping: OffsetMap,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineToken {
    pub term: String,
    pub span: (usize, usize),
    pub domain: TermDomain,
    pub base_term: String,
}

pub struct TokenStream {
    pub tokens: Vec<PipelineToken>,
    pub term_freqs: HashMap<String, TermFrequency>,
    pub doc_len: i64,
}

/// Snapshot of per-domain auxiliary structures.
/// Persisted index state including documents and aux domain data.
#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotData {
    #[serde(default)]
    pub version: u32,
    pub docs: HashMap<String, DocData>,
    pub domains: HashMap<TermDomain, DomainIndex>,
    pub total_len: i64,
    pub domain_total_len: DomainLengths,
}

/// Term and domain that matched during search.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchedTerm {
    pub term: String,
    pub domain: TermDomain,
}

impl MatchedTerm {
    pub fn new(term: String, domain: TermDomain) -> Self {
        Self { term, domain }
    }
}

/// Encoding used when returning match spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEncoding {
    /// Return offsets in raw bytes.
    Bytes,
    /// Return offsets in UTF-16 code units (useful for JS/DOM).
    Utf16,
}

impl Default for InMemoryIndex {
    fn default() -> Self {
        Self {
            versions: HashMap::new(),
            docs: HashMap::new(),
            domains: HashMap::new(),
            total_lens: HashMap::new(),
            domain_total_lens: HashMap::new(),
            dirty: HashMap::new(),
            deleted: HashMap::new(),
            position_encoding: PositionEncoding::Utf16,
            dictionary: None,
        }
    }
}

/// Search hit containing the doc id, score, and matched terms/domains.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub doc_id: String,
    pub score: f64,
    pub matched_terms: Vec<MatchedTerm>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DomainLengths {
    lens: [i64; TERM_DOMAIN_COUNT],
}

impl Default for DomainLengths {
    fn default() -> Self {
        Self {
            lens: [0; TERM_DOMAIN_COUNT],
        }
    }
}

impl DomainLengths {
    pub fn get(&self, domain: TermDomain) -> i64 {
        self.lens[domain_index(domain)]
    }

    pub fn clear(&mut self) {
        self.lens = [0; TERM_DOMAIN_COUNT];
    }

    pub fn add(&mut self, domain: TermDomain, delta: i64) {
        let idx = domain_index(domain);
        self.lens[idx] += delta;
    }

    pub fn is_zero(&self) -> bool {
        self.lens.iter().all(|&v| v == 0)
    }

    pub fn for_each_nonzero(&self, mut f: impl FnMut(TermDomain, i64)) {
        for domain in all_domains() {
            let len = self.get(*domain);
            if len != 0 {
                f(*domain, len);
            }
        }
    }

    pub fn from_term_freqs(freqs: &HashMap<String, TermFrequency>) -> Self {
        let mut lengths = Self::default();
        for freqs in freqs.values() {
            for (domain, count) in freqs.positive_domains() {
                lengths.add(domain, count as i64);
            }
        }
        lengths
    }

    pub fn from_doc(doc: &DocData) -> Self {
        if !doc.domain_doc_len.is_zero() {
            return doc.domain_doc_len;
        }
        let mut lengths = Self::from_term_freqs(&doc.term_freqs);
        if lengths.is_zero() {
            lengths.add(TermDomain::Original, doc.doc_len);
        }
        lengths
    }
}
