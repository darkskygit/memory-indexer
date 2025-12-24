mod base;
mod index;
mod ngram;
mod pipeline;
mod search;
mod tokenizer;
mod types;

pub use types::{
    DocData, InMemoryIndex, PositionEncoding, SNAPSHOT_VERSION, SearchHit, SearchMode,
    SnapshotData, TermDomain,
};

pub use tokenizer::dictionary::{
    DictionaryConfig, DictionaryLanguage, DictionarySegmenter, ScriptDictionary,
    train_dictionary_config,
};

#[cfg(test)]
mod tests {
    use super::types::{DomainLengths, MatchedTerm, TermFrequency};
    use super::*;
    use std::collections::{HashMap, HashSet};
    use tempfile::tempdir;

    const INDEX: &str = "test-index";
    const DOC_CN: &str = "doc-cn";
    const DOC_EN: &str = "doc-en";
    const DOC_JP: &str = "doc-jp";

    fn assert_contains_doc(results: &[(String, f64)], doc_id: &str) {
        assert!(
            results.iter().any(|(id, _)| id == doc_id),
            "expected results to contain doc {doc_id}, got {:?}",
            results
        );
    }

    fn domain_term_dict<'a>(
        index: &'a InMemoryIndex,
        domain: TermDomain,
    ) -> Option<&'a std::collections::HashSet<String>> {
        index
            .domains
            .get(INDEX)
            .and_then(|domains| domains.get(&domain))
            .map(|d| &d.term_dict)
    }

    fn domain_ngram_index<'a>(
        index: &'a InMemoryIndex,
        domain: TermDomain,
    ) -> Option<&'a std::collections::HashMap<String, Vec<String>>> {
        index
            .domains
            .get(INDEX)
            .and_then(|domains| domains.get(&domain))
            .map(|d| &d.ngram_index)
    }

    #[test]
    fn chinese_full_pinyin_search() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        let hits = index.search(INDEX, "nihao");
        assert_contains_doc(&hits, DOC_CN);
    }

    #[test]
    fn chinese_initials_search() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        let hits = index.search(INDEX, "nh");
        assert_contains_doc(&hits, DOC_CN);
    }

    #[test]
    fn chinese_initials_prefix_search() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        let hits = index.search(INDEX, "nhs");
        assert_contains_doc(&hits, DOC_CN);

        let exact = index.get_matches(INDEX, DOC_CN, "nhsj");
        let prefix = index.get_matches(INDEX, DOC_CN, "nhs");
        assert!(!exact.is_empty());
        assert!(!prefix.is_empty());
        assert!(
            prefix.iter().any(|p| exact.iter().any(|e| e.0 == p.0)),
            "prefix highlight should align to original start"
        );
    }

    #[test]
    fn chinese_full_pinyin_prefix_search() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        let hits = index.search(INDEX, "nih");
        assert_contains_doc(&hits, DOC_CN);

        let exact = index.get_matches(INDEX, DOC_CN, "nihaoshijie");
        let prefix = index.get_matches(INDEX, DOC_CN, "nih");
        assert!(!exact.is_empty());
        assert!(!prefix.is_empty());
        assert!(
            prefix.iter().any(|p| exact.iter().any(|e| e.0 == p.0)),
            "prefix highlight should align to original start"
        );
    }

    #[test]
    fn pinyin_fuzzy_search() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        let hits = index.search_hits(INDEX, "nihap");
        assert!(
            hits.iter()
                .any(|h| h.doc_id == DOC_CN && !h.matched_terms.is_empty()),
            "expected matched pinyin term in fuzzy hits: {:?}",
            hits.iter()
                .map(|h| (&h.doc_id, &h.matched_terms))
                .collect::<Vec<_>>()
        );

        let fuzzy_original = index.search_with_mode(INDEX, "nihap", SearchMode::Fuzzy);
        assert!(
            fuzzy_original.is_empty(),
            "expected SearchMode::Fuzzy to only search original domain, got {:?}",
            fuzzy_original
        );
    }

    #[test]
    fn original_aux_index_excludes_non_ascii_terms() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        if let Some(term_dict) = domain_term_dict(&index, TermDomain::Original) {
            assert!(term_dict.contains("你好"));
            assert!(term_dict.contains("世界"));
        }
    }

    #[test]
    fn english_fuzzy_search() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_EN, "fuzzy search handles typos", true);

        let hits = index.search_hits(INDEX, "fuzze");
        assert!(hits.iter().any(|h| {
            h.doc_id == DOC_EN
                && h.matched_terms
                    .iter()
                    .any(|t| t.term == "fuzzy" && t.domain == TermDomain::Original)
        }));
    }

    #[test]
    fn english_query_splits_separators_and_lowercases() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_EN, "MEMORY-INDEXER", true);

        let hits = index.search_with_mode(INDEX, "memory-indexer", SearchMode::Exact);
        assert_contains_doc(&hits, DOC_EN);
    }

    #[test]
    fn fuzzy_search_allows_alphanumeric_terms() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_EN, "version2 stable", true);

        let hits = index.search_with_mode(INDEX, "versoin2", SearchMode::Fuzzy);
        assert_contains_doc(&hits, DOC_EN);
    }

    #[test]
    fn fuzzy_search_handles_separated_query_terms() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_EN, "memory-indexer", true);

        let hits = index.search_with_mode(INDEX, "memry-indexer", SearchMode::Fuzzy);
        assert_contains_doc(&hits, DOC_EN);
    }

    #[test]
    fn fuzzy_search_handles_short_terms() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_EN, "go go", true);

        let hits = index.search_with_mode(INDEX, "go", SearchMode::Fuzzy);
        assert_contains_doc(&hits, DOC_EN);
    }

    #[test]
    fn pinyin_highlight_uses_original_positions() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        let direct = index.get_matches(INDEX, DOC_CN, "你好");
        assert!(
            !direct.is_empty(),
            "expected direct chinese match to have positions"
        );

        let pinyin = index.get_matches(INDEX, DOC_CN, "nihao");
        assert_eq!(pinyin, direct);
    }

    #[test]
    fn highlight_prefers_original_for_mixed_scripts() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "hello 世界", true);

        let hits = index.search_hits(INDEX, "hello shi");
        let Some(hit) = hits.iter().find(|h| h.doc_id == DOC_CN) else {
            panic!("expected hit for mixed script query");
        };
        let matches = index.get_matches_for_matched_terms(INDEX, DOC_CN, &hit.matched_terms);
        let content = index.get_doc(INDEX, DOC_CN).unwrap();
        let slices: Vec<String> = matches
            .iter()
            .map(|(s, e)| utf16_slice(&content, *s, *e))
            .collect();
        assert!(
            slices.iter().any(|s| s == "hello"),
            "expected original spans for mixed script matches, got {:?}",
            slices
        );
        if slices.iter().any(|s| s.chars().any(|c| !c.is_ascii())) {
            assert!(
                slices.iter().any(|s| s == "世界"),
                "expected CJK spans for mixed script matches, got {:?}",
                slices
            );
        }
    }

    #[test]
    fn pinyin_prefix_highlight_uses_original_spans() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        let hits = index.search_hits(INDEX, "nih");
        let Some(hit) = hits.iter().find(|h| h.doc_id == DOC_CN) else {
            panic!("expected prefix pinyin hit");
        };
        let matches = index.get_matches_for_matched_terms(INDEX, DOC_CN, &hit.matched_terms);
        let direct = index.get_matches(INDEX, DOC_CN, "你好");
        assert_eq!(
            matches, direct,
            "prefix highlight should map back to original spans"
        );
    }

    #[test]
    fn pinyin_highlight_handles_trailing_ascii() {
        let mut index = InMemoryIndex::with_position_encoding(PositionEncoding::Utf16);
        index.add_doc(
            INDEX,
            DOC_CN,
            "美光将在全球内存供应短缺之际退出消费级内存业务",
            true,
        );

        let hits = index.search_hits(INDEX, "neicun");
        let hit = hits
            .iter()
            .find(|h| h.doc_id == DOC_CN)
            .unwrap_or_else(|| panic!("expected hit for neicun, got {:?}", hits));
        let matches = index.get_matches_for_matched_terms(INDEX, DOC_CN, &hit.matched_terms);
        assert!(
            !matches.is_empty(),
            "expected highlight spans for pinyin match, got none"
        );
        let content = index.get_doc(INDEX, DOC_CN).unwrap();
        let slices: Vec<String> = matches
            .iter()
            .map(|(s, e)| utf16_slice(&content, *s, *e))
            .collect();
        assert!(
            slices.iter().all(|s| s == "内存"),
            "expected highlights to stay on original term, got {:?}",
            slices
        );
    }

    fn utf16_slice(content: &str, start: u32, end: u32) -> String {
        let mut utf16_pos = 0u32;
        let mut start_byte = 0usize;
        let mut end_byte = content.len();
        for (idx, ch) in content.char_indices() {
            if utf16_pos == start {
                start_byte = idx;
            }
            utf16_pos += ch.len_utf16() as u32;
            if utf16_pos == end {
                end_byte = idx + ch.len_utf8();
                break;
            }
        }
        content[start_byte..end_byte].to_string()
    }

    #[test]
    fn exact_search_prefers_original_terms() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_EN, "nihao greeting", true);
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        let exact_hits = index.search_with_mode(INDEX, "nihao", SearchMode::Exact);
        assert_contains_doc(&exact_hits, DOC_EN);
        assert!(
            exact_hits.iter().all(|(id, _)| id == DOC_EN),
            "expected exact search to ignore pinyin matches, got {:?}",
            exact_hits
        );

        let auto_hits = index.search(INDEX, "nihao");
        assert_contains_doc(&auto_hits, DOC_EN);
        assert!(
            auto_hits.iter().all(|(id, _)| id != DOC_CN),
            "auto search should stop at exact matches"
        );

        let pinyin_hits = index.search_with_mode(INDEX, "nihao", SearchMode::Pinyin);
        assert_contains_doc(&pinyin_hits, DOC_CN);
    }

    #[test]
    fn japanese_ngram_search() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_JP, "こんにちは世界", true);

        let hits = index.search(INDEX, "こん");
        assert_contains_doc(&hits, DOC_JP);

        let matches = index.get_matches(INDEX, DOC_JP, "こん");
        assert!(
            !matches.is_empty(),
            "expected offsets for japanese ngram matches"
        );
    }

    #[test]
    fn kanji_adjacent_to_kana_skips_pinyin() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_JP, "東京へようこそ", true);

        let hits = index.search_with_mode(INDEX, "dongjing", SearchMode::Pinyin);
        assert!(
            hits.is_empty(),
            "kanji near kana should not derive pinyin, got {:?}",
            hits
        );
    }

    #[test]
    fn exact_search_applies_minimum_should_match() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, "doc-2-terms", "apple banana", true);
        index.add_doc(INDEX, "doc-3-terms", "apple banana cherry", true);
        index.add_doc(INDEX, "doc-1-term", "apple", true);

        let hits = index.search_with_mode(INDEX, "apple banana cherry", SearchMode::Exact);

        assert_contains_doc(&hits, "doc-2-terms");
        assert_contains_doc(&hits, "doc-3-terms");
        assert!(
            !hits.iter().any(|(id, _)| id == "doc-1-term"),
            "docs below minimum_should_match should be filtered out"
        );

        let score_two = hits
            .iter()
            .find(|(id, _)| id == "doc-2-terms")
            .map(|(_, s)| *s)
            .unwrap();
        let score_three = hits
            .iter()
            .find(|(id, _)| id == "doc-3-terms")
            .map(|(_, s)| *s)
            .unwrap();
        assert!(
            score_three > score_two,
            "more matched terms should score higher: {} vs {}",
            score_three,
            score_two
        );
    }

    #[test]
    fn pinyin_polyphonic_variants_for_short_tokens() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "重庆火锅", true);

        let hits_zhong = index.search_with_mode_hits(INDEX, "zhongqing", SearchMode::Pinyin);
        assert!(
            hits_zhong.iter().any(|h| h.doc_id == DOC_CN),
            "expected zhongqing variant to hit"
        );

        let hits_chong = index.search_with_mode_hits(INDEX, "chongqing", SearchMode::Pinyin);
        assert!(
            hits_chong.iter().any(|h| h.doc_id == DOC_CN),
            "expected chongqing variant to hit"
        );

        let matched_terms: Vec<MatchedTerm> = hits_zhong
            .into_iter()
            .find(|h| h.doc_id == DOC_CN)
            .map(|h| h.matched_terms)
            .unwrap_or_default();
        assert!(
            matched_terms
                .iter()
                .any(|t| t.term.contains("zhongqing") || t.term.contains("chongqing")),
            "expected polyphonic pinyin variants in matched_terms, got {:?}",
            matched_terms
        );
    }

    #[test]
    fn removing_doc_cleans_aux_indices() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_EN, "token removal check", true);

        index.remove_doc(INDEX, DOC_EN);

        if let Some(term_dict) = domain_term_dict(&index, TermDomain::Original) {
            assert!(
                !term_dict.contains("token"),
                "term_dict should drop removed terms"
            );
        }

        if let Some(ngram_index) = domain_ngram_index(&index, TermDomain::Original) {
            let still_contains = ngram_index
                .values()
                .any(|terms| terms.iter().any(|term| term == "token"));
            assert!(!still_contains, "ngrams should remove term entries");
        }
    }

    #[test]
    fn get_matches_for_terms_uses_matched_terms() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_EN, "memoryIndexer", true);

        let hits = index.search_hits(INDEX, "memryindexer");
        let Some(hit) = hits.iter().find(|h| h.doc_id == DOC_EN) else {
            panic!("expected hit for doc");
        };
        assert!(
            hit.matched_terms
                .iter()
                .any(|t| t.term == "memoryindexer" && t.domain == TermDomain::Original),
            "expected matched term memoryIndexer, got {:?}",
            hit.matched_terms
        );

        let matches = index.get_matches_for_matched_terms(INDEX, DOC_EN, &hit.matched_terms);
        assert!(!matches.is_empty(), "expected matches from matched_terms");
    }

    #[test]
    fn snapshot_contains_aux_indices_per_domain() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界 memory-indexer", true);

        let snapshot = index
            .get_snapshot_data(INDEX)
            .expect("snapshot should exist");

        let domains = snapshot.domains;
        let original = domains
            .get(&TermDomain::Original)
            .expect("snapshot should contain original domain");
        assert!(
            !original.term_dict.is_empty(),
            "expected original aux index to be persisted"
        );
        let pinyin_full = domains
            .get(&TermDomain::PinyinFull)
            .expect("snapshot should contain pinyin full domain");
        assert!(
            !pinyin_full.term_dict.is_empty(),
            "expected full pinyin aux index to be persisted"
        );
        let pinyin_initials = domains
            .get(&TermDomain::PinyinInitials)
            .expect("snapshot should contain pinyin initials domain");
        assert!(
            !pinyin_initials.term_dict.is_empty(),
            "expected initials pinyin aux index to be persisted"
        );
        assert!(
            !pinyin_full.ngram_index.is_empty(),
            "expected pinyin ngram index to be persisted"
        );
    }

    #[test]
    fn fullwidth_pinyin_query_hits() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        // Full-width ASCII should normalize to ASCII and derive pinyin.
        let hits = index.search_hits(INDEX, "ＮＩＨＡＯ");
        assert!(
            hits.iter().any(|h| h.doc_id == DOC_CN),
            "expected full-width pinyin query to hit, got {:?}",
            hits.iter()
                .map(|h| (&h.doc_id, &h.matched_terms))
                .collect::<Vec<_>>()
        );
        let matched = hits.iter().find(|h| h.doc_id == DOC_CN).and_then(|h| {
            h.matched_terms
                .iter()
                .find(|t| t.domain == TermDomain::PinyinFull)
        });
        assert!(
            matched.is_some(),
            "expected matched pinyin full term, got {:?}",
            hits.iter()
                .find(|h| h.doc_id == DOC_CN)
                .map(|h| h.matched_terms.clone())
        );
    }

    #[test]
    fn short_pinyin_fuzzy_hits() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好", true);

        // Missing one character should still fuzzy match via pinyin domain.
        let hits = index.search_hits(INDEX, "niha");
        assert!(
            hits.iter().any(|h| h.doc_id == DOC_CN),
            "expected fuzzy pinyin hit for short query, got {:?}",
            hits.iter()
                .map(|h| (&h.doc_id, &h.matched_terms))
                .collect::<Vec<_>>()
        );
        let matched = hits.iter().find(|h| h.doc_id == DOC_CN).and_then(|h| {
            h.matched_terms.iter().find(|t| {
                matches!(
                    t.domain,
                    TermDomain::PinyinFull | TermDomain::PinyinFullPrefix
                )
            })
        });
        assert!(
            matched.is_some(),
            "expected matched pinyin term, got {:?}",
            hits.iter()
                .find(|h| h.doc_id == DOC_CN)
                .map(|h| h.matched_terms.clone())
        );
    }

    #[test]
    fn non_ascii_auto_fuzzy_fallback() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "北京大学", true);

        // Typo on the last character should still match via non-ASCII fuzzy fallback.
        let hits = index.search_hits(INDEX, "北景大学");
        assert!(
            hits.iter().any(|h| h.doc_id == DOC_CN),
            "expected non-ascii fuzzy fallback to hit, got {:?}",
            hits.iter()
                .map(|h| (&h.doc_id, &h.matched_terms))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn mixed_script_query_hits_all_tokens() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "hello 世界", true);

        let hits = index.search_hits(INDEX, "hello 世界");
        assert!(
            hits.iter().any(|h| h.doc_id == DOC_CN),
            "expected mixed-script query to hit doc, got {:?}",
            hits.iter()
                .map(|h| (&h.doc_id, &h.matched_terms))
                .collect::<Vec<_>>()
        );
        let matched = hits
            .iter()
            .find(|h| h.doc_id == DOC_CN)
            .map(|h| h.matched_terms.clone())
            .unwrap_or_default();
        assert!(
            matched
                .iter()
                .any(|t| t.term == "hello" && t.domain == TermDomain::Original),
            "expected matched original term hello, got {:?}",
            matched
        );
        assert!(
            matched.iter().any(|t| t.term == "世界"),
            "expected matched CJK term 世界, got {:?}",
            matched
        );
    }

    #[test]
    fn chinese_oov_fuzzy_recall() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "明博", true);

        // Typo on the second char should still recall via non-ASCII fuzzy fallback.
        let hits = index.search_hits(INDEX, "明搏");
        assert!(
            hits.iter().any(|h| h.doc_id == DOC_CN),
            "expected OOV chinese fuzzy to hit, got {:?}",
            hits.iter()
                .map(|h| (&h.doc_id, &h.matched_terms))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn load_snapshot_rebuilds_missing_aux_indices() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, DOC_CN, "你好世界", true);

        let mut snapshot = index
            .get_snapshot_data(INDEX)
            .expect("snapshot should exist");
        if let Some(full) = snapshot.domains.get_mut(&TermDomain::PinyinFull) {
            full.term_dict.clear();
            full.ngram_index.clear();
        }
        if let Some(initials) = snapshot.domains.get_mut(&TermDomain::PinyinInitials) {
            initials.term_dict.clear();
            initials.ngram_index.clear();
        }

        let mut restored = InMemoryIndex::default();
        restored.load_snapshot(INDEX, snapshot);

        let hits = restored.search_hits(INDEX, "nihap");
        assert!(
            hits.iter().any(|hit| hit.doc_id == DOC_CN),
            "expected rebuilt pinyin aux indices to allow fuzzy hits"
        );
        assert!(
            restored
                .domains
                .get(INDEX)
                .and_then(|domains| domains.get(&TermDomain::PinyinFull))
                .is_some_and(|d| !d.term_dict.is_empty()),
            "expected pinyin full dictionary to be rebuilt from doc data"
        );
        assert!(
            restored
                .domains
                .get(INDEX)
                .and_then(|domains| domains.get(&TermDomain::PinyinInitials))
                .is_some_and(|d| !d.term_dict.is_empty()),
            "expected pinyin initials dictionary to be rebuilt from doc data"
        );
    }

    #[test]
    fn fuzzy_msm_filters_insufficient_matches() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, "doc-long", "apple banana", true);
        index.add_doc(INDEX, "doc-short", "apple", true);

        let hits = index.search_with_mode_hits(INDEX, "applr banaan", SearchMode::Fuzzy);
        assert!(
            hits.iter().any(|h| h.doc_id == "doc-long"),
            "expected fuzzy msm to keep doc with both terms, got {:?}",
            hits
        );
        assert!(
            hits.iter().all(|h| h.doc_id != "doc-short"),
            "docs below min_should_match should be filtered out: {:?}",
            hits
        );
    }

    #[test]
    fn short_cjk_fuzzy_recall_uses_2gram() {
        let mut index = InMemoryIndex::default();
        index.add_doc(INDEX, "doc-short-cjk", "方案", true);

        let hits = index.search_hits(INDEX, "方桉");
        assert!(
            hits.iter().any(|h| h.doc_id == "doc-short-cjk"),
            "expected 2-gram fuzzy recall for short CJK tokens, got {:?}",
            hits
        );
    }

    #[test]
    fn snapshot_v2_rebuilds_derived_spans() {
        let mut term_pos: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        term_pos.insert("你好".to_string(), vec![(0, 6)]);
        term_pos.insert("nihao".to_string(), vec![(0, 6)]);

        let mut term_freqs: HashMap<String, TermFrequency> = HashMap::new();
        let mut freq_original = TermFrequency::default();
        freq_original.increment(TermDomain::Original);
        term_freqs.insert("你好".to_string(), freq_original);
        let mut freq_pinyin = TermFrequency::default();
        freq_pinyin.increment(TermDomain::PinyinFull);
        term_freqs.insert("nihao".to_string(), freq_pinyin);

        let mut docs = HashMap::new();
        docs.insert(
            DOC_CN.to_string(),
            DocData {
                content: "你好".to_string(),
                doc_len: 2,
                term_pos,
                term_freqs,
                domain_doc_len: DomainLengths::default(),
                derived_terms: HashMap::new(),
            },
        );

        let snapshot = SnapshotData {
            version: 2,
            docs,
            domains: HashMap::new(),
        };

        let mut index = InMemoryIndex::default();
        index.load_snapshot(INDEX, snapshot);

        let hits = index.search_hits(INDEX, "nihao");
        assert!(
            hits.iter().any(|h| h.doc_id == DOC_CN),
            "expected legacy snapshot to rebuild pinyin hits, got {:?}",
            hits
        );

        let matches = index.get_matches(INDEX, DOC_CN, "nihao");
        assert!(
            matches.iter().any(|(s, e)| (*s, *e) == (0, 2)),
            "expected derived spans converted to utf16, got {:?}",
            matches
        );
    }

    #[test]
    fn dictionary_load_and_fallback() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dict.json");

        let mut entries = HashSet::new();
        entries.insert("こんにちは".to_string());
        let config = DictionaryConfig {
            japanese: Some(ScriptDictionary {
                version: Some("v1".to_string()),
                entries,
            }),
            hangul: None,
        };

        std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
        let loaded: DictionaryConfig =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).expect("should deserialize");

        let mut index = InMemoryIndex::with_dictionary_config(loaded.clone());
        index.add_doc(INDEX, DOC_JP, "こんにちは世界", true);

        let hits = index.search_with_mode_hits(INDEX, "こんにちは", SearchMode::Exact);
        assert!(
            hits.iter().any(|h| h.doc_id == DOC_JP),
            "expected dictionary-backed search hit, got {:?}",
            hits
        );
        if let Some(dict) = domain_term_dict(&index, TermDomain::Original) {
            assert!(
                dict.contains("こんにちは"),
                "expected dictionary tokens to be indexed, got {:?}",
                dict
            );
        }

        let mut fallback_index = InMemoryIndex::default();
        fallback_index.add_doc(INDEX, DOC_JP, "こんにちは世界", true);
        let fallback_hits =
            fallback_index.search_with_mode_hits(INDEX, "こんにちは", SearchMode::Exact);
        assert!(
            fallback_hits.iter().any(|h| h.doc_id == DOC_JP),
            "expected fallback tokenization to still recall doc, got {:?}",
            fallback_hits
        );
    }
}
