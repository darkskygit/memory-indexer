# memory-indexer

In-memory multilingual full-text indexer with pinyin-first search, prefix and fuzzy recall—built for chat memory, note-taking, or local knowledge bases.

## Highlights

- [x] Out-of-the-box CJK support

    - [x] chinese and pinyin fuzzy search

    - [x] japanese/korean n-grams with custom dictionaries

    - [x] mixed-script text supported

- [x] Ranking and routing

    - [x] BM25 with minimum-should-match

    - [x] ASCII queries auto-route exact → pinyin → fuzzy

    - [x] non-ASCII uses 2/3-gram + Levenshtein fuzzy

- [x] Highlight-friendly offsets: UTF-8/UTF-16 positions supported
- [x] Index snapshots: compressed binary format for persistence and fast loading
- [x] Pluggable dictionaries: inject or train Japanese/Hangul dictionaries for better tokenization

## Quick start

```rust
use memory_indexer::{InMemoryIndex, SearchMode};

let mut index = InMemoryIndex::default();
index.add_doc("kb", "doc-cn", "你好世界 memory-indexer", true);
index.add_doc("kb", "doc-en", "fuzzy search handles typos", true);

// Auto chooses between exact / pinyin / fuzzy
let hits = index.search_hits("kb", "nihao");

// Explicit modes
let fuzzy = index.search_with_mode("kb", "memry-indexer", SearchMode::Fuzzy);
let pinyin_prefix = index.search_with_mode_hits("kb", "nhs", SearchMode::Pinyin);

// Highlight spans (UTF-16 positions by default)
let spans = index.get_matches("kb", "doc-cn", "nihao");

// Snapshot persistence
let snapshot = index.get_snapshot_data("kb").unwrap();
// index.load_snapshot("kb", snapshot);
```

## Development

-   Tests: `cargo test`
-   Benchmarks: `cargo bench`

## License

> AGPL-3.0-or-later
