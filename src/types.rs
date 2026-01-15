use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use super::tokenizer::{DictionaryConfig, OffsetMap, SegmentScript, TokenWithScript};

pub const SNAPSHOT_VERSION: u32 = 5;

pub type TermId = u32;
pub type DocId = u32;

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
}

pub(crate) const TERM_DOMAIN_COUNT: usize = 3;

pub(crate) const fn domain_index(domain: TermDomain) -> usize {
    match domain {
        TermDomain::Original => 0,
        TermDomain::PinyinFull => 1,
        TermDomain::PinyinInitials => 2,
    }
}

impl TermDomain {
    /// Returns true if the domain represents a pinyin-derived token.
    pub fn is_pinyin(&self) -> bool {
        matches!(self, TermDomain::PinyinFull | TermDomain::PinyinInitials)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DomainConfig {
    pub weight: f64,
    pub allow_fuzzy: bool,
}

pub fn domain_config(domain: TermDomain) -> DomainConfig {
    match domain {
        TermDomain::Original => DomainConfig {
            weight: 1.0,
            allow_fuzzy: true,
        },
        TermDomain::PinyinFull => DomainConfig {
            weight: 0.9,
            allow_fuzzy: true,
        },
        TermDomain::PinyinInitials => DomainConfig {
            weight: 0.8,
            allow_fuzzy: true,
        },
    }
}

pub fn all_domains() -> &'static [TermDomain] {
    &[
        TermDomain::Original,
        TermDomain::PinyinFull,
        TermDomain::PinyinInitials,
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Posting {
    pub doc: DocId,
    pub freq: u32,
}

#[derive(Debug, Default, Clone)]
pub struct DomainAux {
    pub term_ids: Option<Vec<TermId>>,
    pub ngram_index: Option<HashMap<SmolStr, Vec<TermId>>>,
    pub prefix_index: Option<HashMap<SmolStr, Vec<TermId>>>,
}

impl DomainAux {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    fn default_lock() -> RwLock<Self> {
        RwLock::new(Self::default())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DomainIndex {
    pub postings: HashMap<TermId, Vec<Posting>>,
    #[serde(skip, default = "DomainAux::default_lock")]
    pub aux: RwLock<DomainAux>,
}

impl Clone for DomainIndex {
    fn clone(&self) -> Self {
        Self {
            postings: self.postings.clone(),
            aux: RwLock::new(DomainAux::default()),
        }
    }
}

impl Default for DomainIndex {
    fn default() -> Self {
        Self {
            postings: HashMap::new(),
            aux: RwLock::new(DomainAux::default()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermPositions {
    pub term: TermId,
    pub positions: Vec<(u32, u32)>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TermFrequencyEntry {
    pub term: TermId,
    pub counts: [u32; TERM_DOMAIN_COUNT],
}

impl TermFrequencyEntry {
    pub fn positive_domains(&self) -> Vec<(TermDomain, u32)> {
        let mut domains = Vec::new();
        for domain in all_domains() {
            let count = self.counts[domain_index(*domain)];
            if count > 0 {
                domains.push((*domain, count));
            }
        }
        domains
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DerivedTerm {
    pub derived: TermId,
    pub base: TermId,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DerivedSpan {
    pub derived: TermId,
    pub span: (u32, u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocData {
    pub content: String,
    /// Document length in normalized tokens.
    pub doc_len: i64,
    /// Term positions for original-domain tokens (by term id).
    pub term_pos: Vec<TermPositions>,
    /// Per-term frequencies for removal and length accounting.
    pub term_freqs: Vec<TermFrequencyEntry>,
    #[serde(default)]
    pub domain_doc_len: DomainLengths,
    /// Mapping from derived term id -> base term id for highlighting.
    #[serde(default)]
    pub derived_terms: Vec<DerivedTerm>,
    /// Fallback spans for derived terms that lack base term positions.
    #[serde(default)]
    pub derived_spans: Vec<DerivedSpan>,
}

/// In-memory inverted index supporting exact, pinyin, and fuzzy search over documents.
#[derive(Debug)]
pub struct InMemoryIndex {
    pub indexes: HashMap<String, IndexState>,
    pub position_encoding: PositionEncoding,
    pub dictionary: Option<DictionaryConfig>,
}

#[derive(Debug, Default)]
pub struct IndexState {
    pub version: u32,
    pub terms: Vec<SmolStr>,
    pub term_index: HashMap<SmolStr, TermId>,
    pub docs: Vec<Option<DocData>>,
    pub doc_ids: Vec<SmolStr>,
    pub doc_index: HashMap<SmolStr, DocId>,
    pub free_docs: Vec<DocId>,
    pub domains: HashMap<TermDomain, DomainIndex>,
    pub total_len: i64,
    pub domain_total_len: DomainLengths,
    pub dirty: HashSet<SmolStr>,
    pub deleted: HashSet<SmolStr>,
}

impl InMemoryIndex {
    pub fn snapshot_version() -> u32 {
        SNAPSHOT_VERSION
    }
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
    pub doc_len: i64,
}

/// Persisted index state including documents and per-domain postings.
#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotData {
    #[serde(default)]
    pub version: u32,
    pub terms: Vec<SmolStr>,
    pub docs: Vec<Option<DocData>>,
    pub doc_ids: Vec<SmolStr>,
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
            indexes: HashMap::new(),
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

    pub fn from_term_freqs(freqs: &[TermFrequencyEntry]) -> Self {
        let mut lengths = Self::default();
        for entry in freqs {
            for (domain, count) in entry.positive_domains() {
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
