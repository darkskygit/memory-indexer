use std::collections::HashSet;

use pinyin::{ToPinyin, ToPinyinMulti};

use super::script::{is_cjk_char, is_kana, next_char, prev_char};

pub const MAX_PINYIN_VARIANTS: usize = 4;
pub const MAX_PINYIN_VARIANT_LEN: usize = 48;
pub const MAX_POLYPHONIC_CHARS: usize = 4;
pub const MAX_TOKEN_CHARS_FOR_PINYIN: usize = 16;

pub fn contains_chinese_chars(term: &str) -> bool {
    term.chars().any(is_cjk_char)
}

pub fn should_derive_pinyin_for_span(text: &str, start: usize, end: usize) -> bool {
    if start >= end || start >= text.len() || end > text.len() {
        return false;
    }

    // If the span itself contains kana, treat it as Japanese and skip pinyin.
    if text[start..end].chars().any(is_kana) {
        return false;
    }

    // Heuristic: if there is Hiragana/Katakana adjacent to the span, avoid pinyin.
    if let Some(prev) = prev_char(text, start) {
        if is_kana(prev) {
            return false;
        }
    }
    if let Some(next) = next_char(text, end) {
        if is_kana(next) {
            return false;
        }
    }

    true
}

pub fn cjk_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut current_start: Option<usize> = None;

    for (i, c) in text.char_indices() {
        if is_cjk_char(c) {
            if current_start.is_none() {
                current_start = Some(i);
            }
        } else if let Some(start) = current_start.take() {
            spans.push((start, i));
        }
    }

    if let Some(start) = current_start.take() {
        spans.push((start, text.len()));
    }

    spans
}

pub fn build_pinyin_variants(term: &str) -> Vec<(String, String)> {
    let chars: Vec<char> = term.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    if chars.len() > MAX_TOKEN_CHARS_FOR_PINYIN {
        return Vec::new();
    }

    // Avoid combinatorial explosion: only expand multi-pronunciation for very short tokens.
    if chars.len() > MAX_POLYPHONIC_CHARS {
        return build_single_pinyin_variant(&chars)
            .into_iter()
            .collect::<Vec<_>>();
    }

    let mut per_char: Vec<Vec<pinyin::Pinyin>> = Vec::new();
    for ch in &chars {
        if let Some(multi) = ch.to_pinyin_multi() {
            let mut options: Vec<pinyin::Pinyin> = multi.into_iter().collect();
            if options.is_empty() {
                if let Some(py) = ch.to_pinyin() {
                    options.push(py);
                }
            }
            if options.is_empty() {
                return Vec::new();
            }
            per_char.push(options);
        } else if let Some(py) = ch.to_pinyin() {
            per_char.push(vec![py]);
        } else {
            return Vec::new();
        }
    }

    let mut variants = Vec::new();
    let mut seen = HashSet::new();

    fn dfs(
        idx: usize,
        per_char: &[Vec<pinyin::Pinyin>],
        full: &mut String,
        initials: &mut String,
        out: &mut Vec<(String, String)>,
        seen: &mut HashSet<(String, String)>,
    ) {
        if out.len() >= MAX_PINYIN_VARIANTS {
            return;
        }
        if idx == per_char.len() {
            let key = (full.clone(), initials.clone());
            if key.0.len() <= MAX_PINYIN_VARIANT_LEN && seen.insert(key.clone()) {
                out.push(key);
            }
            return;
        }

        for py in &per_char[idx] {
            full.push_str(py.plain());
            initials.push_str(py.first_letter());

            if full.len() <= MAX_PINYIN_VARIANT_LEN {
                dfs(idx + 1, per_char, full, initials, out, seen);
            }

            for _ in 0..py.plain().len() {
                full.pop();
            }
            for _ in 0..py.first_letter().len() {
                initials.pop();
            }
            if out.len() >= MAX_PINYIN_VARIANTS {
                break;
            }
        }
    }

    dfs(
        0,
        &per_char,
        &mut String::new(),
        &mut String::new(),
        &mut variants,
        &mut seen,
    );

    if variants.is_empty() {
        build_single_pinyin_variant(&chars)
            .into_iter()
            .collect::<Vec<_>>()
    } else {
        variants
    }
}

fn build_single_pinyin_variant(chars: &[char]) -> Option<(String, String)> {
    let mut full = String::new();
    let mut initials = String::new();

    for ch in chars {
        if let Some(py) = ch.to_pinyin() {
            full.push_str(py.plain());
            initials.push_str(py.first_letter());
        } else {
            return None;
        }
    }

    Some((full, initials))
}
