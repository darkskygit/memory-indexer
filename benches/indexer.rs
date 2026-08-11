use std::{
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{Criterion, criterion_group, criterion_main};
use memory_indexer::{
    Document, FieldId, FieldOptions, MemoryIndex, Mutation, PositionEncoding, Query, Schema,
    SearchMode, SearchOptions, Sort, SortOrder, TermsAggregation, TextOptions,
};
use serde::Serialize;

const DOCUMENTS: usize = 100_000;
const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

#[derive(Clone, Copy)]
struct Fields {
    workspace: FieldId,
    doc: FieldId,
    block: FieldId,
    content: FieldId,
    flavour: FieldId,
}

#[derive(Serialize)]
struct Report {
    documents: usize,
    build_ms: f64,
    broad_search: Latency,
    selective_search: Latency,
    aggregate: Latency,
    exact_filter: Latency,
    update_ms: f64,
    delete_ms: f64,
    checkpoint_ms: f64,
    checkpoint_compressed_bytes: usize,
    restore_ms: f64,
    live_rss_mib: Option<u64>,
    metadata: Metadata,
}

#[derive(Serialize)]
struct Latency {
    iterations: usize,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
    result_count: usize,
}

#[derive(Serialize)]
struct Metadata {
    os: String,
    arch: &'static str,
    cpu: String,
    warmup: usize,
    iterations: usize,
    rss: &'static str,
}

fn schema() -> (Schema, Fields) {
    let mut schema = Schema::builder().position_encoding(PositionEncoding::Utf16);
    let fields = Fields {
        workspace: schema.keyword("workspace_id", FieldOptions::indexed_stored()),
        doc: schema.keyword("doc_id", FieldOptions::indexed_stored().sortable()),
        block: schema.keyword("block_id", FieldOptions::indexed_stored().sortable()),
        content: schema.text(
            "content",
            TextOptions::multilingual()
                .with_pinyin()
                .with_prefix()
                .with_fuzzy()
                .with_positions(),
            FieldOptions::indexed_stored(),
        ),
        flavour: schema.keyword("flavour", FieldOptions::indexed_stored()),
    };
    (schema.build().unwrap(), fields)
}

fn content(index: usize) -> String {
    const TEMPLATES: &[&str] = &[
        "AFFiNE collaborative notes provide distributed search over local first documents and shared knowledge.",
        "这是一段中文笔记，包含全文搜索、知识管理、协作编辑和离线同步。",
        "Manticore and Elasticsearch provide distributed search with ranking highlighting and filtering.",
        "ローカルファーストのノートでは全文検索と共同編集を利用できます。",
        "Fast indexing matters for workspace search, backlinks, attachments, databases and document titles.",
        "다국어 문서 검색은 한국어 일본어 중국어와 영어 토큰을 함께 처리합니다.",
        "A selective benchmark token makes rare lookups measurable without changing the schema.",
        "Boolean filters, aggregation, top hits, cursor pagination and snippets define the provider contract.",
    ];
    format!(
        "{} rareterm{} sequence{}",
        TEMPLATES[index % TEMPLATES.len()],
        index % 1000,
        index
    )
}

fn document(fields: Fields, index: usize, suffix: &str) -> Document {
    let logical_doc = index / 5;
    let mut document = Document::new(format!(
        "workspace-{}/doc-{logical_doc}/block-{index}",
        logical_doc % 10
    ));
    document.add(fields.workspace, format!("workspace-{}", logical_doc % 10));
    document.add(fields.doc, format!("doc-{logical_doc}"));
    document.add(fields.block, format!("block-{index}"));
    document.add(fields.content, format!("{}{suffix}", content(index)));
    document.add(
        fields.flavour,
        if index.is_multiple_of(5) {
            "affine:page"
        } else {
            "affine:paragraph"
        },
    );
    document
}

fn build(count: usize) -> (Schema, Fields, MemoryIndex) {
    let (schema, fields) = schema();
    let mut index = MemoryIndex::new(schema.clone());
    for start in (0..count).step_by(1_000) {
        let mutations = (start..(start + 1_000).min(count))
            .map(|item| Mutation::Upsert(document(fields, item, "")))
            .collect();
        index.apply_batch(mutations).unwrap();
    }
    (schema, fields, index)
}

fn query(fields: Fields, text: &str) -> Query {
    Query::boolean(
        vec![
            Query::term(fields.workspace, "workspace-3"),
            Query::text(fields.content, text, SearchMode::Auto),
        ],
        vec![],
        vec![],
    )
}

fn search_options(fields: Fields) -> SearchOptions {
    SearchOptions {
        limit: 20,
        offset: 0,
        after: None,
        sort: vec![
            Sort::ScoreDesc,
            Sort::Field {
                field: fields.doc,
                order: SortOrder::Asc,
            },
        ],
        stored_fields: vec![fields.doc, fields.block, fields.content, fields.flavour],
        highlight_fields: vec![fields.content],
    }
}

fn measure(mut operation: impl FnMut() -> usize, warmup: usize, iterations: usize) -> Latency {
    for _ in 0..warmup {
        black_box(operation());
    }
    let mut samples = Vec::with_capacity(iterations);
    let mut result_count = 0;
    for _ in 0..iterations {
        let started = Instant::now();
        result_count = black_box(operation());
        samples.push(started.elapsed());
    }
    let mean_ms =
        samples.iter().map(Duration::as_secs_f64).sum::<f64>() * 1_000.0 / iterations as f64;
    let mut values = samples
        .into_iter()
        .map(|value| value.as_secs_f64() * 1_000.0)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    Latency {
        iterations,
        p50_ms: percentile(&values, 0.50),
        p95_ms: percentile(&values, 0.95),
        p99_ms: percentile(&values, 0.99),
        mean_ms,
        result_count,
    }
}

fn percentile(values: &[f64], ratio: f64) -> f64 {
    values[((values.len() - 1) as f64 * ratio).round() as usize]
}

fn write_report() {
    let Some(path) = std::env::var_os("MEMORY_INDEXER_REPORT") else {
        return;
    };
    let started = Instant::now();
    let (schema, fields, mut index) = build(DOCUMENTS);
    let build_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let live_rss_mib = rss_mib();
    let options = search_options(fields);
    let broad_search = measure(
        || {
            index
                .search(&query(fields, "distributed search"), options.clone())
                .unwrap()
                .hits
                .len()
        },
        WARMUP,
        ITERATIONS,
    );
    let selective_search = measure(
        || {
            index
                .search(&query(fields, "sequence17"), options.clone())
                .unwrap()
                .hits
                .len()
        },
        WARMUP,
        ITERATIONS,
    );
    let exact_filter = measure(
        || {
            index
                .search(&Query::term(fields.doc, "doc-1234"), SearchOptions::new(20))
                .unwrap()
                .hits
                .len()
        },
        WARMUP,
        ITERATIONS,
    );
    let aggregate = measure(
        || {
            index
                .aggregate(
                    &query(fields, "distributed search"),
                    TermsAggregation {
                        field: fields.doc,
                        limit: 20,
                        offset: 0,
                        top_hits: Some(options.clone()),
                    },
                )
                .unwrap()
                .buckets
                .len()
        },
        5,
        50,
    );

    let started = Instant::now();
    index
        .apply_batch(
            (0..1_000)
                .map(|item| Mutation::Upsert(document(fields, item, " updated")))
                .collect(),
        )
        .unwrap();
    let update_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = Instant::now();
    index
        .apply_batch(
            (1_000..2_000)
                .map(|item| {
                    Mutation::Delete(format!(
                        "workspace-{}/doc-{}/block-{item}",
                        (item / 5) % 10,
                        item / 5
                    ))
                })
                .collect(),
        )
        .unwrap();
    let delete_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let started = Instant::now();
    let checkpoint = index.checkpoint().unwrap();
    let checkpoint_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = Instant::now();
    let restored = MemoryIndex::from_checkpoint(schema, &checkpoint.bytes).unwrap();
    black_box(restored.len());
    let restore_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let report = Report {
        documents: DOCUMENTS,
        build_ms,
        broad_search,
        selective_search,
        aggregate,
        exact_filter,
        update_ms,
        delete_ms,
        checkpoint_ms,
        checkpoint_compressed_bytes: checkpoint.bytes.len(),
        restore_ms,
        live_rss_mib,
        metadata: Metadata {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH,
            cpu: cpu_name(),
            warmup: WARMUP,
            iterations: ITERATIONS,
            rss: "ps -o rss= -p <pid>, sampled after build",
        },
    };
    std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}

fn rss_mib() -> Option<u64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kib| kib / 1024)
}

fn cpu_name() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn benchmark(c: &mut Criterion) {
    write_report();
    if std::env::var_os("MEMORY_INDEXER_REPORT_ONLY").is_some() {
        return;
    }
    let (_, fields, index) = build(DOCUMENTS);
    let index = Arc::new(index);
    let options = search_options(fields);
    for (name, text) in [("broad", "distributed search"), ("selective", "sequence17")] {
        let index = Arc::clone(&index);
        let query = query(fields, text);
        let options = options.clone();
        c.bench_function(name, move |bench| {
            bench.iter(|| black_box(index.search(&query, options.clone()).unwrap().hits.len()))
        });
    }
    let index = Arc::clone(&index);
    c.bench_function("keyword_exact", move |bench| {
        bench.iter(|| {
            black_box(
                index
                    .search(&Query::term(fields.doc, "doc-1234"), SearchOptions::new(20))
                    .unwrap()
                    .hits
                    .len(),
            )
        })
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
