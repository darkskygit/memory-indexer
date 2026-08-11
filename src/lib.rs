mod checkpoint;
mod document;
mod error;
mod index;
mod ngram;
mod pipeline;
mod schema;
mod search;
mod tokenizer;
mod types;

pub use checkpoint::Checkpoint;
pub use document::{BatchResult, Document, Mutation, MutationResult, Value};
pub use error::{Error, Result};
pub use index::optimize::OptimizeResult;
pub use schema::{Field, FieldId, FieldOptions, FieldType, Schema, SchemaBuilder, TextOptions};
pub use search::{
    AggregationResult, Bucket, Highlight, Query, SearchHit, SearchOptions, SearchResult, Sort,
    SortOrder, SortValue, TermsAggregation,
};
pub use tokenizer::dictionary::{
    DictionaryConfig, DictionaryLanguage, DictionarySegmenter, ScriptDictionary,
    train_dictionary_config,
};
pub use types::{PositionEncoding, SearchMode};

use index::MemoryIndexState;

#[derive(Debug)]
pub struct MemoryIndex {
    schema: Schema,
    state: MemoryIndexState,
}

impl MemoryIndex {
    pub fn new(schema: Schema) -> Self {
        Self {
            state: MemoryIndexState::new(&schema),
            schema,
        }
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn len(&self) -> usize {
        self.state.live_docs.len() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.state.live_docs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        schema: Schema,
        workspace: FieldId,
        content: FieldId,
        rank: FieldId,
        active: FieldId,
    }

    fn fixture(encoding: PositionEncoding) -> Fixture {
        let mut builder = Schema::builder().position_encoding(encoding);
        let workspace = builder.keyword("workspace", FieldOptions::indexed_stored().multi_value());
        let content = builder.text(
            "content",
            TextOptions::multilingual()
                .with_pinyin()
                .with_prefix()
                .with_fuzzy()
                .with_positions(),
            FieldOptions::indexed_stored().multi_value(),
        );
        let rank = builder.i64("rank", FieldOptions::indexed_stored().sortable());
        let active = builder.bool("active", FieldOptions::indexed_stored().sortable());
        Fixture {
            schema: builder.build().unwrap(),
            workspace,
            content,
            rank,
            active,
        }
    }

    fn document(
        fixture: &Fixture,
        id: &str,
        workspace: &str,
        content: &str,
        rank: i64,
    ) -> Document {
        let mut document = Document::new(id);
        document.add(fixture.workspace, workspace);
        document.add(fixture.content, content);
        document.add(fixture.rank, rank);
        document.add(fixture.active, true);
        document
    }

    fn ids(result: SearchResult) -> Vec<String> {
        result.hits.into_iter().map(|hit| hit.id).collect()
    }

    #[test]
    fn multilingual_and_pinyin_semantics() {
        let cases = [
            (
                "latin",
                "fuzzy search handles typos",
                "fuzze",
                SearchMode::Auto,
            ),
            ("chinese-full", "你好世界", "nihao", SearchMode::Auto),
            ("chinese-initial", "你好世界", "nhs", SearchMode::Auto),
            ("japanese", "検索エンジン", "検索", SearchMode::Exact),
            ("korean", "검색 엔진", "검색", SearchMode::Exact),
            (
                "mixed",
                "memory 搜索 engine",
                "memory 搜索",
                SearchMode::Exact,
            ),
            ("polyphonic", "重庆银行", "chongqing", SearchMode::Auto),
        ];
        for (id, text, query, mode) in cases {
            let fixture = fixture(PositionEncoding::Bytes);
            let mut index = MemoryIndex::new(fixture.schema.clone());
            index
                .upsert(document(&fixture, id, "workspace-1", text, 1))
                .unwrap();
            let result = index
                .search(
                    &Query::text(fixture.content, query, mode),
                    SearchOptions::new(10),
                )
                .unwrap();
            assert_eq!(ids(result), vec![id], "case {id}");
        }
    }

    #[test]
    fn keyword_is_exact_case_sensitive_and_deduplicated() {
        let fixture = fixture(PositionEncoding::Bytes);
        let mut index = MemoryIndex::new(fixture.schema.clone());
        let mut first = document(&fixture, "one", "workspace-1", "search", 1);
        first.add(fixture.workspace, "workspace-1");
        index.upsert(first).unwrap();
        index
            .upsert(document(&fixture, "two", "workspace-2", "search", 2))
            .unwrap();
        assert_eq!(
            ids(index
                .search(
                    &Query::term(fixture.workspace, "workspace-1"),
                    SearchOptions::new(10)
                )
                .unwrap()),
            vec!["one"]
        );
        assert!(
            index
                .search(
                    &Query::term(fixture.workspace, "Workspace-1"),
                    SearchOptions::new(10)
                )
                .unwrap()
                .hits
                .is_empty()
        );
    }

    #[test]
    fn boolean_top_k_fields_highlight_and_search_after() {
        let fixture = fixture(PositionEncoding::Utf16);
        let mut index = MemoryIndex::new(fixture.schema.clone());
        for (id, workspace, text, rank) in [
            ("a", "w1", "你好 search", 3),
            ("b", "w1", "search", 2),
            ("c", "w2", "search", 1),
        ] {
            index
                .upsert(document(&fixture, id, workspace, text, rank))
                .unwrap();
        }
        let query = Query::boolean(
            vec![
                Query::term(fixture.workspace, "w1"),
                Query::text(fixture.content, "search", SearchMode::Exact),
            ],
            vec![],
            vec![Query::term(fixture.rank, 99i64)],
        );
        let options = SearchOptions {
            limit: 1,
            offset: 0,
            after: None,
            sort: vec![Sort::Field {
                field: fixture.rank,
                order: SortOrder::Desc,
            }],
            stored_fields: vec![fixture.content, fixture.rank],
            highlight_fields: vec![fixture.content],
        };
        let first = index.search(&query, options.clone()).unwrap();
        assert_eq!(first.total, 2);
        assert_eq!(first.hits[0].id, "a");
        assert!(!first.hits[0].highlights.is_empty());
        let mut second_options = options;
        second_options.after = Some(first.hits[0].sort_values.clone());
        assert_eq!(
            ids(index.search(&query, second_options).unwrap()),
            vec!["b"]
        );
    }

    #[test]
    fn mutation_batch_aggregation_and_checkpoint_contract() {
        let fixture = fixture(PositionEncoding::Bytes);
        let schema = fixture.schema.clone();
        let mut index = MemoryIndex::new(fixture.schema.clone());
        index
            .apply_batch(vec![
                Mutation::Upsert(document(&fixture, "a", "w1", "search", 1)),
                Mutation::Upsert(document(&fixture, "b", "w1", "search", 2)),
                Mutation::Upsert(document(&fixture, "c", "w2", "other", 3)),
            ])
            .unwrap();
        assert_eq!(index.change_sequence(), 1);
        let aggregation = index
            .aggregate(
                &Query::All,
                TermsAggregation {
                    field: fixture.workspace,
                    limit: 10,
                    offset: 0,
                    top_hits: Some(SearchOptions::new(1)),
                },
            )
            .unwrap();
        assert_eq!(aggregation.total, 3);
        assert_eq!(aggregation.buckets[0].count, 2);
        assert!(index.delete("b"));
        assert_eq!(
            index
                .search(&Query::All, SearchOptions::new(10))
                .unwrap()
                .total,
            2
        );
        let checkpoint = index.checkpoint().unwrap();
        index
            .upsert(document(&fixture, "d", "w3", "new", 4))
            .unwrap();
        index
            .mark_checkpoint_persisted(checkpoint.sequence)
            .unwrap();
        assert!(index.has_unpersisted_changes());
        let restored = MemoryIndex::from_checkpoint(schema, &checkpoint.bytes).unwrap();
        assert_eq!(restored.len(), 2);
        assert!(!restored.has_unpersisted_changes());
        let mut corrupted = checkpoint.bytes;
        *corrupted.last_mut().unwrap() ^= 1;
        assert!(MemoryIndex::from_checkpoint(restored.schema().clone(), &corrupted).is_err());
    }

    #[test]
    fn full_upsert_removes_fields_and_invalid_batch_is_atomic() {
        let fixture = fixture(PositionEncoding::Bytes);
        let mut index = MemoryIndex::new(fixture.schema.clone());
        index
            .upsert(document(&fixture, "a", "w1", "search", 1))
            .unwrap();
        let mut replacement = Document::new("a");
        replacement.add(fixture.content, "other");
        index.upsert(replacement).unwrap();
        assert!(
            index
                .search(
                    &Query::term(fixture.workspace, "w1"),
                    SearchOptions::new(10)
                )
                .unwrap()
                .hits
                .is_empty()
        );
        let sequence = index.change_sequence();
        let mut invalid = Document::new("bad");
        invalid.add(fixture.rank, "not-an-i64");
        assert!(
            index
                .apply_batch(vec![
                    Mutation::Delete("a".into()),
                    Mutation::Upsert(invalid)
                ])
                .is_err()
        );
        assert_eq!(index.change_sequence(), sequence);
        assert_eq!(index.len(), 1);

        let bulk = (0..2_100)
            .map(|item| {
                let mut document = Document::new(format!("bulk-{item}"));
                document.add(fixture.workspace, "bulk");
                Mutation::Upsert(document)
            })
            .collect();
        index.apply_batch(bulk).unwrap();
        index
            .apply_batch(
                (0..1_000)
                    .map(|item| {
                        let mut document = Document::new(format!("bulk-{item}"));
                        document.add(fixture.workspace, "bulk-updated");
                        Mutation::Upsert(document)
                    })
                    .collect(),
            )
            .unwrap();
        let deleted = index
            .apply_batch(
                (1_000..2_000)
                    .map(|item| Mutation::Delete(format!("bulk-{item}")))
                    .collect(),
            )
            .unwrap();
        assert_eq!(deleted.deleted, 1_000);
        assert_eq!(
            index
                .search(&Query::All, SearchOptions::new(0))
                .unwrap()
                .total,
            1_101
        );
        index.checkpoint().unwrap();
    }
}
