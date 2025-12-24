use std::collections::{HashMap, HashSet};

use super::{
    super::tokenizer::{
        DictionarySegmenter, TextNormalizer, TextNormalizerRef, normalize_query, normalize_term,
        script_runs, tokenize_chinese, tokenize_hangul, tokenize_japanese,
    },
    DictionaryConfig, Segment, SegmentScript, TokenDraft,
};

pub trait ScriptTokenizerStrategy {
    fn tokenize(
        &self,
        segment: &Segment<'_>,
        normalizer: &dyn TextNormalizer,
        dictionary: Option<&DictionarySegmenter>,
        out: &mut Vec<TokenDraft>,
        seen: &mut HashSet<(String, usize, usize)>,
    );
}

pub struct DefaultScriptSegmenter;

impl DefaultScriptSegmenter {
    pub fn segment<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        script_runs(text)
            .into_iter()
            .map(|(script, start, end)| Segment {
                script,
                text: &text[start..end],
                offset: start,
            })
            .collect()
    }
}

pub struct HanTokenizer;

impl ScriptTokenizerStrategy for HanTokenizer {
    fn tokenize(
        &self,
        segment: &Segment<'_>,
        normalizer: &dyn TextNormalizer,
        _dictionary: Option<&DictionarySegmenter>,
        out: &mut Vec<TokenDraft>,
        seen: &mut HashSet<(String, usize, usize)>,
    ) {
        let mut tokens = Vec::new();
        tokenize_chinese(
            segment.text,
            segment.offset,
            segment.script,
            normalizer,
            &mut tokens,
            seen,
        );
        out.extend(tokens.into_iter().map(TokenDraft::from));
    }
}

pub struct KanaTokenizer;

impl ScriptTokenizerStrategy for KanaTokenizer {
    fn tokenize(
        &self,
        segment: &Segment<'_>,
        normalizer: &dyn TextNormalizer,
        dictionary: Option<&DictionarySegmenter>,
        out: &mut Vec<TokenDraft>,
        seen: &mut HashSet<(String, usize, usize)>,
    ) {
        let mut tokens = Vec::new();
        tokenize_japanese(
            segment.text,
            segment.offset,
            segment.script,
            normalizer,
            dictionary,
            &mut tokens,
            seen,
        );
        out.extend(tokens.into_iter().map(TokenDraft::from));
    }
}

pub struct HangulTokenizer;

impl ScriptTokenizerStrategy for HangulTokenizer {
    fn tokenize(
        &self,
        segment: &Segment<'_>,
        normalizer: &dyn TextNormalizer,
        dictionary: Option<&DictionarySegmenter>,
        out: &mut Vec<TokenDraft>,
        seen: &mut HashSet<(String, usize, usize)>,
    ) {
        let mut tokens = Vec::new();
        tokenize_hangul(
            segment.text,
            segment.offset,
            segment.script,
            normalizer,
            dictionary,
            &mut tokens,
            seen,
        );
        out.extend(tokens.into_iter().map(TokenDraft::from));
    }
}

pub struct LatinOtherTokenizer;

impl ScriptTokenizerStrategy for LatinOtherTokenizer {
    fn tokenize(
        &self,
        segment: &Segment<'_>,
        normalizer: &dyn TextNormalizer,
        _dictionary: Option<&DictionarySegmenter>,
        out: &mut Vec<TokenDraft>,
        seen: &mut HashSet<(String, usize, usize)>,
    ) {
        let mut tokens = Vec::new();
        normalizer.normalize(
            segment.text,
            segment.offset,
            segment.script,
            &mut tokens,
            seen,
        );
        out.extend(tokens.into_iter().map(TokenDraft::from));
    }
}

pub struct DefaultTokenizer {
    pub(crate) normalizer: TextNormalizerRef,
    pub(crate) tokenizers: HashMap<SegmentScript, Box<dyn ScriptTokenizerStrategy>>,
    pub(crate) fallback: Box<dyn ScriptTokenizerStrategy>,
    pub(crate) dictionary: Option<DictionarySegmenter>,
}

impl DefaultTokenizer {
    pub fn for_documents() -> Self {
        Self::with_default_scripts(normalize_term())
    }

    pub fn for_queries() -> Self {
        Self::with_default_scripts(normalize_query())
    }

    fn with_default_scripts(normalizer: TextNormalizerRef) -> Self {
        Self::new(normalizer, Box::new(LatinOtherTokenizer))
            .register_script_tokenizer(SegmentScript::Han, HanTokenizer)
            .register_script_tokenizer(SegmentScript::Hiragana, KanaTokenizer)
            .register_script_tokenizer(SegmentScript::Katakana, KanaTokenizer)
            .register_script_tokenizer(SegmentScript::Hangul, HangulTokenizer)
    }

    pub fn new(normalizer: TextNormalizerRef, fallback: Box<dyn ScriptTokenizerStrategy>) -> Self {
        Self {
            normalizer,
            tokenizers: HashMap::new(),
            fallback,
            dictionary: None,
        }
    }

    pub fn register_script_tokenizer<T>(mut self, script: SegmentScript, tokenizer: T) -> Self
    where
        T: ScriptTokenizerStrategy + 'static,
    {
        self.tokenizers.insert(script, Box::new(tokenizer));
        self
    }

    pub fn with_dictionary(mut self, config: DictionaryConfig) -> Self {
        self.dictionary = Some(DictionarySegmenter::new(config));
        self
    }

    pub fn tokenize_segment(
        &self,
        segment: &Segment<'_>,
        out: &mut Vec<TokenDraft>,
        seen: &mut HashSet<(String, usize, usize)>,
    ) {
        if let Some(tokenizer) = self.tokenizers.get(&segment.script) {
            tokenizer.tokenize(
                segment,
                self.normalizer.as_ref(),
                self.dictionary.as_ref(),
                out,
                seen,
            );
        } else {
            self.fallback.tokenize(
                segment,
                self.normalizer.as_ref(),
                self.dictionary.as_ref(),
                out,
                seen,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{OffsetMap, dictionary::ScriptDictionary, normalize_term};
    use std::collections::HashSet;

    struct FallbackTokenizer;

    impl ScriptTokenizerStrategy for FallbackTokenizer {
        fn tokenize(
            &self,
            segment: &Segment<'_>,
            _normalizer: &dyn TextNormalizer,
            _dictionary: Option<&DictionarySegmenter>,
            out: &mut Vec<TokenDraft>,
            _seen: &mut HashSet<(String, usize, usize)>,
        ) {
            out.push(TokenDraft {
                text: "fallback".to_string(),
                span: (segment.offset, segment.offset + segment.text.len()),
                script: segment.script,
                mapping: OffsetMap::identity((segment.offset, segment.offset + segment.text.len())),
            });
        }
    }

    #[test]
    fn allows_injecting_dictionary_config() {
        let config = DictionaryConfig {
            japanese: Some(ScriptDictionary {
                version: Some("v1".to_string()),
                entries: std::collections::HashSet::new(),
            }),
            hangul: None,
        };
        let tokenizer = DefaultTokenizer::for_documents().with_dictionary(config);
        assert!(tokenizer.dictionary.is_some());
    }

    #[test]
    fn falls_back_to_default_tokenizer_when_script_not_registered() {
        let tokenizer = DefaultTokenizer::new(normalize_term(), Box::new(FallbackTokenizer));

        let segment = Segment {
            script: SegmentScript::Other,
            text: "abc",
            offset: 0,
        };

        let mut out = Vec::new();
        let mut seen = HashSet::new();
        tokenizer.tokenize_segment(&segment, &mut out, &mut seen);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "fallback");
    }
}
