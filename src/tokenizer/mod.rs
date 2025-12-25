use std::collections::HashSet;

use jieba_rs::Jieba;
use once_cell::sync::Lazy;

pub mod dictionary;
mod normalize;
mod pinyin;
mod script;

pub use dictionary::{DictionaryConfig, DictionarySegmenter};
pub use normalize::{TextNormalizer, TextNormalizerRef, normalize_query, normalize_term};
pub use normalize::is_ascii_id_like;
pub use pinyin::{
    build_pinyin_variants, cjk_spans, contains_chinese_chars, should_derive_pinyin_for_span,
};
pub use script::{SegmentScript, is_cjk_char, script_runs};

static JIEBA: Lazy<Jieba> = Lazy::new(Jieba::new);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub term: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffsetMap {
    pub original: (usize, usize),
    pub normalized_to_original: Vec<usize>,
}

impl OffsetMap {
    pub fn identity(span: (usize, usize)) -> Self {
        Self {
            original: span,
            normalized_to_original: Vec::new(),
        }
    }

    pub fn with_mapping(span: (usize, usize), mapping: Vec<usize>) -> Self {
        Self {
            original: span,
            normalized_to_original: mapping,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenWithScript {
    pub term: String,
    pub start: usize,
    pub end: usize,
    pub script: SegmentScript,
    pub offset_map: OffsetMap,
}

impl From<TokenWithScript> for Token {
    fn from(value: TokenWithScript) -> Self {
        Token {
            term: value.term,
            start: value.start,
            end: value.end,
        }
    }
}

pub(crate) fn tokenize_chinese(
    segment: &str,
    base_start: usize,
    script: SegmentScript,
    normalizer: &dyn TextNormalizer,
    out: &mut Vec<TokenWithScript>,
    seen: &mut HashSet<(String, usize, usize)>,
) {
    let mut char_byte_offsets: Vec<usize> = segment.char_indices().map(|(i, _)| i).collect();
    char_byte_offsets.push(segment.len());

    let jieba_tokens = JIEBA.tokenize(segment, jieba_rs::TokenizeMode::Search, false);
    for token in jieba_tokens {
        let start = base_start + char_byte_offsets[token.start];
        let raw = &segment[char_byte_offsets[token.start]..char_byte_offsets[token.end]];
        normalizer.normalize(raw, start, script, out, seen);
    }
}

pub(crate) fn tokenize_japanese(
    segment: &str,
    base_start: usize,
    script: SegmentScript,
    normalizer: &dyn TextNormalizer,
    dictionary: Option<&DictionarySegmenter>,
    out: &mut Vec<TokenWithScript>,
    seen: &mut HashSet<(String, usize, usize)>,
) {
    if let Some(dict) = dictionary {
        if dict.segment(segment, base_start, script, normalizer, out, seen) {
            return;
        }
    }

    tokenize_char_ngrams(segment, base_start, script, normalizer, out, seen);
}

pub(crate) fn tokenize_hangul(
    segment: &str,
    base_start: usize,
    script: SegmentScript,
    normalizer: &dyn TextNormalizer,
    dictionary: Option<&DictionarySegmenter>,
    out: &mut Vec<TokenWithScript>,
    seen: &mut HashSet<(String, usize, usize)>,
) {
    if let Some(dict) = dictionary {
        if dict.segment(segment, base_start, script, normalizer, out, seen) {
            return;
        }
    }

    tokenize_char_ngrams(segment, base_start, script, normalizer, out, seen);
}

pub(crate) fn tokenize_char_ngrams(
    segment: &str,
    base_start: usize,
    script: SegmentScript,
    normalizer: &dyn TextNormalizer,
    out: &mut Vec<TokenWithScript>,
    seen: &mut HashSet<(String, usize, usize)>,
) {
    let chars: Vec<(usize, char)> = segment.char_indices().collect();
    for (i, &(offset, _)) in chars.iter().enumerate() {
        let end_offset = chars
            .get(i + 1)
            .map(|(idx, _)| *idx)
            .unwrap_or_else(|| segment.len());
        normalizer.normalize(
            &segment[offset..end_offset],
            base_start + offset,
            script,
            out,
            seen,
        );

        if let Some((_next_offset, _)) = chars.get(i + 1) {
            let end_offset = chars
                .get(i + 2)
                .map(|(idx, _)| *idx)
                .unwrap_or_else(|| segment.len());
            normalizer.normalize(
                &segment[offset..end_offset],
                base_start + offset,
                script,
                out,
                seen,
            );
        }
    }
}
