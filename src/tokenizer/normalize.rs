use std::{collections::HashSet, sync::Arc};

use once_cell::sync::Lazy;
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

use super::{OffsetMap, TokenWithScript, script::SegmentScript};

pub trait TextNormalizer: Send + Sync {
    fn normalize(
        &self,
        raw: &str,
        base_start: usize,
        script: SegmentScript,
        out: &mut Vec<TokenWithScript>,
        seen: &mut HashSet<(String, usize, usize)>,
    );
}

#[derive(Default)]
pub struct DefaultTextNormalizer;

impl DefaultTextNormalizer {
    fn normalize_ascii_split(
        raw: &str,
        base_start: usize,
        script: SegmentScript,
        out: &mut Vec<TokenWithScript>,
        seen: &mut HashSet<(String, usize, usize)>,
    ) {
        let bytes = raw.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            while i < bytes.len() && !bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let end = i;
            if start < end {
                if let Some(token) =
                    Self::normalize_span(&raw[start..end], base_start + start, script)
                {
                    Self::push_token(out, seen, token);
                }
            }
        }
    }

    fn normalize_span(
        raw: &str,
        start_offset: usize,
        script: SegmentScript,
    ) -> Option<TokenWithScript> {
        let (normalized, mapping) = Self::normalize_text(raw, start_offset);
        if normalized.is_empty() {
            return None;
        }
        Some(TokenWithScript {
            term: normalized,
            start: start_offset,
            end: start_offset + raw.len(),
            script,
            offset_map: OffsetMap::with_mapping((start_offset, start_offset + raw.len()), mapping),
        })
    }

    fn normalize_text(raw: &str, base_start: usize) -> (String, Vec<usize>) {
        let mut normalized = String::new();
        let mut mapping = Vec::new();

        for (orig_byte, ch) in raw.char_indices() {
            let mut decomposed = String::new();
            for d in ch.to_string().nfkd() {
                if !is_combining_mark(d) {
                    decomposed.push(d);
                }
            }

            for lower in decomposed.nfkc().flat_map(|c| c.to_lowercase()) {
                normalized.push(lower);
                mapping.push(base_start + orig_byte);
            }
        }

        (normalized, mapping)
    }

    fn push_token(
        out: &mut Vec<TokenWithScript>,
        seen: &mut HashSet<(String, usize, usize)>,
        token: TokenWithScript,
    ) {
        if seen.insert((token.term.clone(), token.start, token.end)) {
            out.push(token);
        }
    }
}

impl TextNormalizer for DefaultTextNormalizer {
    fn normalize(
        &self,
        raw: &str,
        base_start: usize,
        script: SegmentScript,
        out: &mut Vec<TokenWithScript>,
        seen: &mut HashSet<(String, usize, usize)>,
    ) {
        if raw.is_ascii() {
            Self::normalize_ascii_split(raw, base_start, script, out, seen);
            return;
        }

        if !raw.chars().any(|c| c.is_alphanumeric()) {
            return;
        }

        if let Some(token) = Self::normalize_span(raw, base_start, script) {
            Self::push_token(out, seen, token);
        }
    }
}

pub type TextNormalizerRef = Arc<dyn TextNormalizer>;

static DEFAULT_TEXT_NORMALIZER: Lazy<TextNormalizerRef> =
    Lazy::new(|| Arc::new(DefaultTextNormalizer));

pub fn default_text_normalizer() -> TextNormalizerRef {
    DEFAULT_TEXT_NORMALIZER.clone()
}

pub fn normalize_query() -> TextNormalizerRef {
    default_text_normalizer()
}

pub fn normalize_term() -> TextNormalizerRef {
    default_text_normalizer()
}
