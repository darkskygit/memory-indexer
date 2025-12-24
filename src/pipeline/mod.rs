use std::collections::{HashMap, HashSet};

mod derivers;
mod tokenizers;

pub use derivers::{NoopNgramDeriver, PassthroughNormalizer, PinyinVariantDeriver, PrefixDerivers};
pub use tokenizers::{DefaultScriptSegmenter, DefaultTokenizer};

use super::{
    tokenizer::{
        DictionaryConfig, OffsetMap, SegmentScript, Token, cjk_spans, contains_chinese_chars,
        is_cjk_char, should_derive_pinyin_for_span,
    },
    types::{
        NormalizedTerm, PipelineToken, Segment, TermDomain, TermFrequency, TokenDraft, TokenStream,
    },
};

const MAX_CJK_SPAN_DERIVATION_CHARS: usize = 32;

pub struct Pipeline {
    segmenter: DefaultScriptSegmenter,
    tokenizer: DefaultTokenizer,
    normalizer: PassthroughNormalizer,
    variant_deriver: PinyinVariantDeriver,
    prefix_deriver: PrefixDerivers,
    ngram_deriver: NoopNgramDeriver,
}

#[derive(Clone, Copy)]
pub struct PipelineConfig {
    pub enable_variants: bool,
    pub enable_prefixes: bool,
}

impl PipelineConfig {
    pub fn document() -> Self {
        Self {
            enable_variants: true,
            enable_prefixes: true,
        }
    }

    pub fn query() -> Self {
        Self {
            enable_variants: false,
            enable_prefixes: false,
        }
    }
}

impl Pipeline {
    pub fn document_pipeline() -> Self {
        Self::new(DefaultTokenizer::for_documents())
    }

    pub fn query_pipeline() -> Self {
        Self::new(DefaultTokenizer::for_queries())
    }

    pub fn new(tokenizer: DefaultTokenizer) -> Self {
        Self {
            segmenter: DefaultScriptSegmenter,
            tokenizer,
            normalizer: PassthroughNormalizer,
            variant_deriver: PinyinVariantDeriver::default(),
            prefix_deriver: PrefixDerivers::pinyin_defaults(),
            ngram_deriver: NoopNgramDeriver,
        }
    }

    pub fn with_dictionary(dictionary: DictionaryConfig) -> Self {
        Self::new(DefaultTokenizer::for_documents().with_dictionary(dictionary))
    }

    pub fn tokenize_query(text: &str) -> Vec<Token> {
        Self::query_pipeline()
            .query_tokens(text)
            .tokens
            .into_iter()
            .map(|token| Token {
                term: token.term,
                start: token.span.0,
                end: token.span.1,
            })
            .collect()
    }

    pub fn document_tokens(&self, text: &str) -> TokenStream {
        self.run(text, PipelineConfig::document())
    }

    pub fn query_tokens(&self, text: &str) -> TokenStream {
        self.run(text, PipelineConfig::query())
    }

    fn run(&self, text: &str, config: PipelineConfig) -> TokenStream {
        let mut drafts = Vec::new();
        let mut seen = HashSet::new();
        for segment in self.segmenter.segment(text) {
            self.tokenizer
                .tokenize_segment(&segment, &mut drafts, &mut seen);
        }

        let mut tokens = Vec::new();
        let mut term_freqs: HashMap<String, TermFrequency> = HashMap::new();
        let mut covered_cjk_spans: HashSet<(usize, usize)> = HashSet::new();

        let mut doc_len: i64 = 0;

        for draft in &drafts {
            let normalized = self.normalizer.normalize(draft);
            doc_len += normalized.len() as i64;

            for norm in normalized {
                let span = norm.span;
                push_token(
                    &mut tokens,
                    &mut term_freqs,
                    PipelineToken {
                        term: norm.term.clone(),
                        span,
                        domain: TermDomain::Original,
                        base_term: norm.term.clone(),
                    },
                );

                if norm.term.chars().all(is_cjk_char) {
                    covered_cjk_spans.insert(span);
                }

                if config.enable_variants
                    && contains_chinese_chars(&norm.term)
                    && should_derive_pinyin_for_span(text, span.0, span.1)
                {
                    self.derive_variants(&norm, &config, &mut tokens, &mut term_freqs);
                }
            }
        }

        if config.enable_variants {
            for (start, end) in cjk_spans(text) {
                if covered_cjk_spans.contains(&(start, end)) {
                    continue;
                }
                let span = (start, end);
                if text[start..end].chars().count() > MAX_CJK_SPAN_DERIVATION_CHARS {
                    continue;
                }
                if !should_derive_pinyin_for_span(text, start, end) {
                    continue;
                }
                let term = text[start..end].to_string();
                let norm = NormalizedTerm {
                    term,
                    span,
                    script: SegmentScript::Han,
                    mapping: OffsetMap::identity(span),
                };
                self.derive_variants(&norm, &config, &mut tokens, &mut term_freqs);
            }
        }

        TokenStream {
            tokens,
            term_freqs,
            doc_len,
        }
    }

    fn derive_variants(
        &self,
        term: &NormalizedTerm,
        config: &PipelineConfig,
        tokens: &mut Vec<PipelineToken>,
        term_freqs: &mut HashMap<String, TermFrequency>,
    ) {
        for variant in self.variant_deriver.derive(term) {
            push_token(tokens, term_freqs, variant.clone());

            if config.enable_prefixes {
                for prefix in self.prefix_deriver.derive_prefixes(&variant, term_freqs) {
                    push_token(tokens, term_freqs, prefix);
                }
            }

            for ngram in self.ngram_deriver.derive_ngrams(&variant) {
                push_token(tokens, term_freqs, ngram);
            }
        }
    }
}

fn push_token(
    tokens: &mut Vec<PipelineToken>,
    term_freqs: &mut HashMap<String, TermFrequency>,
    token: PipelineToken,
) {
    term_freqs
        .entry(token.term.clone())
        .or_default()
        .increment(token.domain);
    tokens.push(token);
}
