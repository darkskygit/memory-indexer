use std::{hint::black_box, sync::Arc};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use memory_indexer::{InMemoryIndex, SearchMode};

const INDEX_NAME: &str = "bench-index";

fn sample_docs(count: usize) -> Vec<(String, String)> {
    let templates = [
        "你好世界，memory-indexer 本地索引器基准测试。",
        "memory-indexer v2 adds better search and local-first sync.",
        "再来一段中文内容，覆盖拼音前缀与模糊搜索。",
        "Fuzzy search should recover memoryIndexer even with typos like memry-indexer.",
        "混合内容：你好世界 and local-first knowledge base powered by memory-indexer.",
    ];
    (0..count)
        .map(|i| {
            let text = templates[i % templates.len()].to_string();
            (format!("doc-{i}"), text)
        })
        .collect()
}

fn build_index(docs: &[(String, String)]) -> InMemoryIndex {
    let mut index = InMemoryIndex::default();
    for (doc_id, text) in docs {
        index.add_doc(INDEX_NAME, doc_id, text, true);
    }
    index
}

fn bench_index_build(c: &mut Criterion) {
    let docs = sample_docs(128);
    let mut group = c.benchmark_group("indexer_build");

    for &count in &[32usize, 128usize] {
        let docs = docs.clone();
        group.bench_function(BenchmarkId::from_parameter(count), move |b| {
            b.iter_batched(
                || InMemoryIndex::default(),
                |mut index| {
                    for (doc_id, text) in docs.iter().take(count) {
                        index.add_doc(INDEX_NAME, doc_id, text, true);
                    }
                    black_box(index);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_search_paths(c: &mut Criterion) {
    let docs = sample_docs(160);
    let index = Arc::new(build_index(&docs));
    let mut group = c.benchmark_group("indexer_search");

    let prefix_index = Arc::clone(&index);
    group.bench_function("pinyin_prefix", move |b| {
        b.iter(|| {
            black_box(
                prefix_index
                    .search_with_mode_hits(INDEX_NAME, "nhs", SearchMode::Pinyin)
                    .len(),
            );
        })
    });

    let fuzzy_pinyin_index = Arc::clone(&index);
    group.bench_function("pinyin_fuzzy", move |b| {
        b.iter(|| black_box(fuzzy_pinyin_index.search_hits(INDEX_NAME, "nihap").len()))
    });

    group.bench_function("english_fuzzy", move |b| {
        b.iter(|| {
            black_box(
                index
                    .search_with_mode_hits(INDEX_NAME, "affone-db", SearchMode::Fuzzy)
                    .len(),
            );
        })
    });

    group.finish();
}

fn bench_snapshot_serialization(c: &mut Criterion) {
    let docs = sample_docs(128);
    let index = build_index(&docs);
    let snapshot = index
        .get_snapshot_data(INDEX_NAME)
        .expect("snapshot should exist for serialization");
    let snapshot_size = serde_json::to_vec(&snapshot)
        .expect("serialize snapshot for size")
        .len() as u64;

    let mut group = c.benchmark_group("indexer_snapshot");
    group.throughput(Throughput::Bytes(snapshot_size));
    group.bench_function("serialize_snapshot_json", move |b| {
        b.iter(|| {
            let bytes = serde_json::to_vec(&snapshot).expect("serialize snapshot");
            black_box(bytes.len());
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_index_build,
    bench_search_paths,
    bench_snapshot_serialization
);
criterion_main!(benches);
