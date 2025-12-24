use std::{
    collections::{HashMap, HashSet},
    time::{SystemTime, UNIX_EPOCH},
};

use super::{SegmentScript, TextNormalizer, TokenWithScript, script_runs, tokenize_char_ngrams};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DictionaryLanguage {
    Japanese,
    Hangul,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptDictionary {
    pub version: Option<String>,
    pub entries: HashSet<String>,
}

impl ScriptDictionary {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DictionaryConfig {
    pub japanese: Option<ScriptDictionary>,
    pub hangul: Option<ScriptDictionary>,
}

#[derive(Debug, Clone, Default)]
pub struct DictionarySegmenter {
    pub config: DictionaryConfig,
}

impl DictionarySegmenter {
    pub fn new(config: DictionaryConfig) -> Self {
        Self { config }
    }

    pub fn export(&self) -> Vec<DictionaryExport> {
        let mut exports = Vec::new();
        if let Some(dict) = &self.config.japanese {
            if let Some(export) = export_dictionary(DictionaryLanguage::Japanese, dict) {
                exports.push(export);
            }
        }
        if let Some(dict) = &self.config.hangul {
            if let Some(export) = export_dictionary(DictionaryLanguage::Hangul, dict) {
                exports.push(export);
            }
        }
        exports
    }

    pub fn segment(
        &self,
        segment: &str,
        base_start: usize,
        script: SegmentScript,
        normalizer: &dyn TextNormalizer,
        out: &mut Vec<TokenWithScript>,
        seen: &mut HashSet<(String, usize, usize)>,
    ) -> bool {
        let Some(dictionary) = self.dictionary_for_script(script) else {
            return false;
        };
        if dictionary.is_empty() {
            return false;
        }

        let mut char_offsets: Vec<usize> = segment.char_indices().map(|(i, _)| i).collect();
        char_offsets.push(segment.len());
        let char_len = char_offsets.len().saturating_sub(1);
        if char_len == 0 {
            return true;
        }

        let mut covered = vec![false; char_len];
        let mut matched_any = false;
        let mut idx = 0;

        while idx < char_len {
            let mut matched_range: Option<(usize, usize)> = None;
            for end in (idx + 1..=char_len).rev() {
                let start_byte = char_offsets[idx];
                let end_byte = char_offsets[end];
                let candidate = &segment[start_byte..end_byte];
                if dictionary.entries.contains(candidate) {
                    matched_range = Some((idx, end));
                    break;
                }
            }

            if let Some((start_idx, end_idx)) = matched_range {
                matched_any = true;
                let start_byte = char_offsets[start_idx];
                let end_byte = char_offsets[end_idx];
                normalizer.normalize(
                    &segment[start_byte..end_byte],
                    base_start + start_byte,
                    script,
                    out,
                    seen,
                );
                for i in start_idx..end_idx {
                    covered[i] = true;
                }
                idx = end_idx;
            } else {
                idx += 1;
            }
        }

        if !matched_any {
            tokenize_char_ngrams(segment, base_start, script, normalizer, out, seen);
            return true;
        }

        let mut start = 0;
        while start < char_len {
            if covered[start] {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < char_len && !covered[end] {
                end += 1;
            }
            let start_byte = char_offsets[start];
            let end_byte = char_offsets[end];
            tokenize_char_ngrams(
                &segment[start_byte..end_byte],
                base_start + start_byte,
                script,
                normalizer,
                out,
                seen,
            );
            start = end;
        }

        true
    }

    fn dictionary_for_script(&self, script: SegmentScript) -> Option<&ScriptDictionary> {
        match script {
            SegmentScript::Hiragana | SegmentScript::Katakana => self.config.japanese.as_ref(),
            SegmentScript::Hangul => self.config.hangul.as_ref(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DictionaryMetadata {
    pub language: DictionaryLanguage,
    pub version: String,
    pub entry_count: usize,
    pub generated_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct DictionaryExport {
    pub metadata: DictionaryMetadata,
    pub entries: Vec<String>,
}

pub fn export_dictionary(
    language: DictionaryLanguage,
    dictionary: &ScriptDictionary,
) -> Option<DictionaryExport> {
    if dictionary.is_empty() {
        return None;
    }
    let mut entries: Vec<String> = dictionary.entries.iter().cloned().collect();
    entries.sort();

    let metadata = DictionaryMetadata {
        language,
        version: dictionary
            .version
            .clone()
            .unwrap_or_else(|| format!("{}-unversioned", language_prefix(language))),
        entry_count: entries.len(),
        generated_at: SystemTime::now(),
    };

    Some(DictionaryExport { metadata, entries })
}

#[derive(Debug, Clone)]
pub struct DictionaryTrainingConfig {
    pub min_freq: usize,
    pub min_token_len: usize,
    pub max_token_len: usize,
    pub max_entries: usize,
    pub version: Option<String>,
}

impl Default for DictionaryTrainingConfig {
    fn default() -> Self {
        Self {
            min_freq: 2,
            min_token_len: 2,
            max_token_len: 8,
            max_entries: 8_000,
            version: None,
        }
    }
}

pub fn train_dictionary_for_language(
    corpus: &[String],
    language: DictionaryLanguage,
    config: DictionaryTrainingConfig,
) -> ScriptDictionary {
    let min_token_len = config.min_token_len.max(1);
    let max_token_len = config.max_token_len.max(min_token_len);
    let mut counts: HashMap<String, usize> = HashMap::new();

    for text in corpus {
        for (script, start, end) in script_runs(text) {
            if !matches_language(script, language) {
                continue;
            }
            let segment = &text[start..end];
            let mut char_offsets: Vec<usize> = segment.char_indices().map(|(i, _)| i).collect();
            char_offsets.push(segment.len());
            let char_len = char_offsets.len().saturating_sub(1);
            for i in 0..char_len {
                for len in min_token_len..=max_token_len {
                    if i + len > char_len {
                        break;
                    }
                    let start_byte = char_offsets[i];
                    let end_byte = char_offsets[i + len];
                    let candidate = &segment[start_byte..end_byte];
                    if candidate.chars().any(|c| c.is_whitespace()) {
                        continue;
                    }
                    *counts.entry(candidate.to_string()).or_insert(0) += 1;
                }
            }
        }
    }

    let mut entries: Vec<(String, usize)> = counts
        .into_iter()
        .filter(|(_, freq)| *freq >= config.min_freq)
        .collect();
    entries.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.0.len().cmp(&a.0.len()))
            .then_with(|| a.0.cmp(&b.0))
    });
    if config.max_entries > 0 && entries.len() > config.max_entries {
        entries.truncate(config.max_entries);
    }

    let entries_set: HashSet<String> = entries.into_iter().map(|(entry, _)| entry).collect();
    ScriptDictionary {
        version: Some(version_or_default(language, &config.version)),
        entries: entries_set,
    }
}

pub fn train_dictionary_config(
    corpus: &[String],
    config: DictionaryTrainingConfig,
) -> DictionaryConfig {
    let japanese =
        train_dictionary_for_language(corpus, DictionaryLanguage::Japanese, config.clone());
    let hangul = train_dictionary_for_language(corpus, DictionaryLanguage::Hangul, config);

    DictionaryConfig {
        japanese: (!japanese.is_empty()).then_some(japanese),
        hangul: (!hangul.is_empty()).then_some(hangul),
    }
}

fn version_or_default(language: DictionaryLanguage, provided: &Option<String>) -> String {
    if let Some(version) = provided {
        return version.clone();
    }
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}-{ts}", language_prefix(language))
}

fn language_prefix(language: DictionaryLanguage) -> &'static str {
    match language {
        DictionaryLanguage::Japanese => "ja",
        DictionaryLanguage::Hangul => "ko",
    }
}

fn matches_language(script: SegmentScript, language: DictionaryLanguage) -> bool {
    match (language, script) {
        (DictionaryLanguage::Japanese, SegmentScript::Hiragana)
        | (DictionaryLanguage::Japanese, SegmentScript::Katakana) => true,
        (DictionaryLanguage::Hangul, SegmentScript::Hangul) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{SegmentScript, normalize_query};
    use std::collections::HashSet;
    use tempfile::tempdir;

    #[test]
    fn segments_dictionary_tokens_and_fallbacks() {
        let mut entries = HashSet::new();
        entries.insert("こん".to_string());
        let config = DictionaryConfig {
            japanese: Some(ScriptDictionary {
                version: Some("v1".to_string()),
                entries,
            }),
            hangul: None,
        };
        let segmenter = DictionarySegmenter::new(config);
        let normalizer = normalize_query();
        let mut out = Vec::new();
        let mut seen = HashSet::new();

        let used = segmenter.segment(
            "こんにちは",
            0,
            SegmentScript::Hiragana,
            normalizer.as_ref(),
            &mut out,
            &mut seen,
        );

        assert!(used, "expected dictionary to be applied when provided");
        assert!(
            out.iter().any(|t| t.term == "こん"),
            "expected dictionary token present, got {:?}",
            out
        );
        assert!(
            out.iter().any(|t| t.start == 12),
            "expected fallback tokens for unmatched spans, got {:?}",
            out
        );
    }

    #[test]
    fn trains_and_exports_dictionaries() {
        let corpus = vec![
            "こんにちは世界".to_string(),
            "こんにちは友達".to_string(),
            "안녕하세요 세계".to_string(),
        ];
        let config = DictionaryTrainingConfig {
            min_freq: 1,
            min_token_len: 2,
            max_token_len: 3,
            max_entries: 4,
            version: Some("v1".to_string()),
        };

        let dictionaries = train_dictionary_config(&corpus, config);
        let segmenter = DictionarySegmenter::new(dictionaries.clone());
        let exports = segmenter.export();

        assert!(
            dictionaries.japanese.is_some(),
            "expected japanese dictionary"
        );
        assert!(dictionaries.hangul.is_some(), "expected hangul dictionary");
        assert_eq!(exports.len(), 2, "expected exports per language");
        let ja_export = exports
            .iter()
            .find(|e| matches!(e.metadata.language, DictionaryLanguage::Japanese))
            .expect("japanese export present");
        assert_eq!(ja_export.metadata.entry_count, ja_export.entries.len());
        assert!(
            ja_export
                .metadata
                .generated_at
                .elapsed()
                .unwrap_or_default()
                .as_secs()
                < 5,
            "expected recent generated_at, got {:?}",
            ja_export.metadata.generated_at
        );
        assert!(
            ja_export.metadata.version.starts_with("v1"),
            "expected provided version, got {}",
            ja_export.metadata.version
        );
    }

    #[test]
    fn saves_and_loads_dictionary_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dict.json");

        let mut entries = HashSet::new();
        entries.insert("こん".to_string());
        let config = DictionaryConfig {
            japanese: Some(ScriptDictionary {
                version: Some("v1".to_string()),
                entries,
            }),
            hangul: None,
        };

        save_dictionary(&path, &config).unwrap();
        let loaded = load_dictionary(&path).unwrap();
        assert_eq!(
            loaded.japanese.unwrap().entries.len(),
            1,
            "expected saved japanese entries"
        );
    }

    fn save_dictionary(path: &std::path::Path, config: &DictionaryConfig) -> std::io::Result<()> {
        let data = serde_json::to_vec(config).map_err(to_io_err)?;
        std::fs::write(path, data)
    }

    fn load_dictionary(path: &std::path::Path) -> std::io::Result<DictionaryConfig> {
        let data = std::fs::read(path)?;
        serde_json::from_slice(&data).map_err(to_io_err)
    }

    fn to_io_err(err: impl std::fmt::Display) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::Other, err.to_string())
    }
}
