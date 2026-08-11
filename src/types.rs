use serde::{Deserialize, Serialize};

/// Search execution strategy for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PositionEncoding {
    Bytes,
    #[default]
    Utf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum TermDomain {
    Original,
    PinyinFull,
    PinyinInitials,
}

pub(crate) const fn domain_index(domain: TermDomain) -> usize {
    match domain {
        TermDomain::Original => 0,
        TermDomain::PinyinFull => 1,
        TermDomain::PinyinInitials => 2,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NormalizedTerm {
    pub term: String,
    pub span: (usize, usize),
    pub script: crate::tokenizer::SegmentScript,
    pub mapping: crate::tokenizer::OffsetMap,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PipelineToken {
    pub term: String,
    pub span: (usize, usize),
    pub domain: TermDomain,
    pub base_term: String,
}

pub(crate) struct TokenStream {
    pub tokens: Vec<PipelineToken>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Segment<'a> {
    pub script: crate::tokenizer::SegmentScript,
    pub text: &'a str,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenDraft {
    pub text: String,
    pub span: (usize, usize),
    pub script: crate::tokenizer::SegmentScript,
    pub mapping: crate::tokenizer::OffsetMap,
}

impl From<crate::tokenizer::TokenWithScript> for TokenDraft {
    fn from(value: crate::tokenizer::TokenWithScript) -> Self {
        Self {
            text: value.term,
            span: (value.start, value.end),
            script: value.script,
            mapping: value.offset_map,
        }
    }
}
