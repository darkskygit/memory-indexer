use unicode_script::{Script as UniScript, UnicodeScript};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentScript {
    Han,
    Hiragana,
    Katakana,
    Hangul,
    LatinDigit,
    Other,
}

pub fn classify_script(c: char) -> SegmentScript {
    if is_cjk_char(c) {
        SegmentScript::Han
    } else {
        match c.script() {
            UniScript::Hiragana => SegmentScript::Hiragana,
            UniScript::Katakana => SegmentScript::Katakana,
            UniScript::Hangul => SegmentScript::Hangul,
            UniScript::Latin => SegmentScript::LatinDigit,
            _ if c.is_ascii_alphanumeric() => SegmentScript::LatinDigit,
            _ => SegmentScript::Other,
        }
    }
}

pub fn script_runs(text: &str) -> Vec<(SegmentScript, usize, usize)> {
    let mut runs = Vec::new();
    if text.is_empty() {
        return runs;
    }

    let mut iter = text.char_indices().peekable();
    let mut current: Option<SegmentScript> = None;
    let mut start = 0usize;

    while let Some((idx, ch)) = iter.next() {
        let script = classify_script(ch);
        if let Some(cur) = current {
            if cur != script {
                runs.push((cur, start, idx));
                start = idx;
                current = Some(script);
            }
        } else {
            start = idx;
            current = Some(script);
        }

        if iter.peek().is_none() {
            runs.push((current.unwrap_or(script), start, idx + ch.len_utf8()));
        }
    }

    runs
}

pub fn is_kana(c: char) -> bool {
    matches!(c.script(), UniScript::Hiragana | UniScript::Katakana)
}

pub fn is_cjk_char(c: char) -> bool {
    matches!(
      c,
      '\u{3400}'..='\u{4DBF}'
        | '\u{4E00}'..='\u{9FFF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2B73F}'
        | '\u{2B740}'..='\u{2B81F}'
        | '\u{2B820}'..='\u{2CEAF}'
    )
}

pub fn prev_char(text: &str, start: usize) -> Option<char> {
    text.char_indices()
        .take_while(|(idx, _)| *idx < start)
        .map(|(_, ch)| ch)
        .last()
}

pub fn next_char(text: &str, end: usize) -> Option<char> {
    text.char_indices()
        .find_map(|(idx, ch)| (idx >= end).then_some(ch))
}
