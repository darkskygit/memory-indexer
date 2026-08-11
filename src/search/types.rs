use crate::{FieldId, SearchMode, Value};

#[derive(Debug, Clone)]
pub enum Query {
    Text {
        field: FieldId,
        query: String,
        mode: SearchMode,
    },
    Term {
        field: FieldId,
        value: Value,
    },
    Exists(FieldId),
    All,
    Boolean {
        must: Vec<Query>,
        should: Vec<Query>,
        must_not: Vec<Query>,
        minimum_should_match: Option<u16>,
    },
    Boost {
        query: Box<Query>,
        factor: f32,
    },
}

impl Query {
    pub fn text(field: FieldId, query: impl Into<String>, mode: SearchMode) -> Self {
        Self::Text {
            field,
            query: query.into(),
            mode,
        }
    }

    pub fn term(field: FieldId, value: impl Into<Value>) -> Self {
        Self::Term {
            field,
            value: value.into(),
        }
    }

    pub fn boolean(must: Vec<Query>, should: Vec<Query>, must_not: Vec<Query>) -> Self {
        Self::Boolean {
            must,
            should,
            must_not,
            minimum_should_match: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sort {
    ScoreDesc,
    Field { field: FieldId, order: SortOrder },
    DocumentId,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SortValue {
    Score(f32),
    String(String),
    I64(i64),
    Bool(bool),
    Missing,
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit: usize,
    pub offset: usize,
    pub after: Option<Vec<SortValue>>,
    pub sort: Vec<Sort>,
    pub stored_fields: Vec<FieldId>,
    pub highlight_fields: Vec<FieldId>,
}

impl SearchOptions {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            offset: 0,
            after: None,
            sort: Vec::new(),
            stored_fields: Vec::new(),
            highlight_fields: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Highlight {
    pub field: FieldId,
    pub value_index: u16,
    pub spans: Vec<(u32, u32)>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub fields: Vec<(FieldId, Vec<Value>)>,
    pub sort_values: Vec<SortValue>,
    pub highlights: Vec<Highlight>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub total: usize,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone)]
pub struct TermsAggregation {
    pub field: FieldId,
    pub limit: usize,
    pub offset: usize,
    pub top_hits: Option<SearchOptions>,
}

#[derive(Debug, Clone)]
pub struct Bucket {
    pub key: Value,
    pub count: u64,
    pub max_score: f32,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone)]
pub struct AggregationResult {
    pub total: usize,
    pub buckets: Vec<Bucket>,
}
