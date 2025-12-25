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

const ID_LIKE_MIN_LEN: usize = 8;
const ID_LIKE_LEN_RELAXED: usize = 16;

#[derive(Default)]
struct IdLikeDetector {
    len: usize,
    alpha: u32,
    vowels: u32,
    // bit0=upper, bit1=lower, bit2=digit
    class_mask: u8,
}

impl IdLikeDetector {
    #[inline]
    fn is_vowel_lower(b: u8) -> bool {
        matches!(b, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
    }

    fn push(&mut self, b: u8) -> bool {
        match b {
            b'a'..=b'z' => {
                self.len += 1;
                self.alpha += 1;
                self.class_mask |= 1 << 1;
                self.vowels += Self::is_vowel_lower(b) as u32;
                true
            }
            b'A'..=b'Z' => {
                self.len += 1;
                self.alpha += 1;
                self.class_mask |= 1 << 0;
                self.vowels += Self::is_vowel_lower(b | 0x20) as u32;
                true
            }
            b'0'..=b'9' => {
                self.len += 1;
                self.class_mask |= 1 << 2;
                true
            }
            b'_' | b'-' => {
                self.len += 1;
                true
            }
            _ => false,
        }
    }

    fn is_id_like(&self) -> bool {
        let class_count = self.class_mask.count_ones() as u32;
        // relaxed: vowels/alpha <= 0.30  <=> 10*vowels <= 3*alpha
        // strict : vowels/alpha <= 0.25  <=> 4*vowels <= alpha
        let low_vowel_relaxed = self.alpha == 0 || self.vowels * 10 <= self.alpha * 3;
        let low_vowel_strict = self.alpha == 0 || self.vowels * 4 <= self.alpha;

        (self.len >= ID_LIKE_LEN_RELAXED && low_vowel_relaxed)
            || (class_count >= 2 && low_vowel_strict)
    }
}

/// Heuristic detector for high-entropy ASCII ids (UUID/nanoid-like) to avoid fuzzy indexing overhead.
#[inline]
pub fn is_ascii_id_like(token: &str) -> bool {
    let mut stats = IdLikeDetector::default();
    for &b in token.as_bytes() {
        if !stats.push(b) {
            return false;
        }
    }

    if stats.len < ID_LIKE_MIN_LEN {
        return false;
    }

    stats.is_id_like()
}

impl DefaultTextNormalizer {
    fn normalize_ascii_split(
        raw: &str,
        base_start: usize,
        script: SegmentScript,
        out: &mut Vec<TokenWithScript>,
        seen: &mut HashSet<(String, usize, usize)>,
    ) {
        let bytes = raw.as_bytes();
        let len = bytes.len();
        let mut spans: Vec<(usize, usize)> = Vec::new();

        let mut i = 0;
        while i < len {
            // Skip separators (characters that are not alnum, '_' or '-')
            while i < len && !(bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'-'))
            {
                i += 1;
            }
            if i >= len {
                break;
            }

            let start = i;

            // This loop does three things simultaneously:
            // 1) Find run_end
            // 2) Count alpha/vowels/class needed for id-like detection
            // 3) Record alnum spans needed for fallback (treating _/- as separators)
            spans.clear();

            let mut stats = IdLikeDetector::default();
            let mut cur_alnum_start: Option<usize> = None;

            while i < len && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'-')) {
                let b = bytes[i];

                if !stats.push(b) {
                    break;
                }

                match b {
                    b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' => {
                        if cur_alnum_start.is_none() {
                            cur_alnum_start = Some(i);
                        }
                    }
                    b'_' | b'-' => {
                        // fallback: treat _/- as separators, close the current span
                        if let Some(s) = cur_alnum_start.take() {
                            spans.push((s, i));
                        }
                    }
                    _ => {}
                }

                i += 1;
            }

            let end = i;

            // run end: if it ends with alnum, append the final span
            if let Some(s) = cur_alnum_start.take() {
                spans.push((s, end));
            }

            // First, quickly prune by length (to avoid doing too much work on short strings)
            if stats.len < ID_LIKE_MIN_LEN {
                // Fallback directly (spans are already available)
                for (s, e) in spans.iter().copied() {
                    if let Some(token) = Self::normalize_span(&raw[s..e], base_start + s, script) {
                        Self::push_token(out, seen, token);
                    }
                }
                continue;
            }

            if stats.is_id_like() {
                if let Some(token) =
                    Self::normalize_span(&raw[start..end], base_start + start, script)
                {
                    Self::push_token(out, seen, token);
                }
            } else {
                for (s, e) in spans.iter().copied() {
                    if let Some(token) = Self::normalize_span(&raw[s..e], base_start + s, script) {
                        Self::push_token(out, seen, token);
                    }
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
