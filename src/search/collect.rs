use std::{cmp::Ordering, collections::BinaryHeap};

use crate::{
    Error, FieldId, MemoryIndex, Result, Value,
    index::{DocId, FieldIndex},
};

use super::{EvalResult, Highlight, Query, SearchHit, SearchOptions, Sort, SortOrder, SortValue};

impl MemoryIndex {
    pub(super) fn collect_top_k(
        &self,
        evaluated: &EvalResult,
        options: &SearchOptions,
    ) -> Result<Vec<Candidate>> {
        let size = options.offset.saturating_add(options.limit);
        if size == 0 {
            return Ok(Vec::new());
        }
        let sorts = effective_sorts(&options.sort);
        let mut heap = BinaryHeap::with_capacity(size + 1);
        for doc in evaluated.docs.iter() {
            let candidate = self.candidate(doc, evaluated.scores[doc as usize], &sorts)?;
            if let Some(after) = &options.after {
                if after.len() != candidate.sort_values.len() {
                    return Err(Error::InvalidQuery(
                        "search-after value count does not match sort".into(),
                    ));
                }
                if compare_sort_values(&candidate.sort_values, after, &sorts) != Ordering::Greater {
                    continue;
                }
            }
            heap.push(candidate);
            if heap.len() > size {
                heap.pop();
            }
        }
        let mut candidates = heap.into_vec();
        candidates.sort_by(|a, b| compare_candidates(a, b, &sorts));
        Ok(candidates)
    }

    fn candidate(&self, doc: DocId, score: f32, sorts: &[Sort]) -> Result<Candidate> {
        let id = self
            .state
            .external_id(doc)
            .ok_or_else(|| Error::InvalidQuery("live document has no external id".into()))?
            .to_owned();
        let mut values = Vec::with_capacity(sorts.len());
        for sort in sorts {
            values.push(match sort {
                Sort::ScoreDesc => SortValue::Score(score),
                Sort::DocumentId => SortValue::String(id.clone()),
                Sort::Field { field, .. } => match &self.state.fields[field.index()] {
                    FieldIndex::Keyword(index) => index
                        .sort_value(doc)
                        .map(|value| SortValue::String(value.to_owned()))
                        .unwrap_or(SortValue::Missing),
                    FieldIndex::I64(index) => index
                        .values
                        .get(doc as usize)
                        .and_then(|value| *value)
                        .map(SortValue::I64)
                        .unwrap_or(SortValue::Missing),
                    FieldIndex::Bool(index) => index
                        .values
                        .get(doc as usize)
                        .and_then(|value| *value)
                        .map(SortValue::Bool)
                        .unwrap_or(SortValue::Missing),
                    FieldIndex::Text(_) => {
                        return Err(Error::InvalidQuery("Text fields cannot be sorted".into()));
                    }
                },
            });
        }
        Ok(Candidate {
            doc,
            id,
            score,
            sort_values: values,
            sorts: sorts.to_vec(),
        })
    }

    pub(super) fn materialize_hit(
        &self,
        query: &Query,
        candidate: Candidate,
        options: &SearchOptions,
    ) -> Result<SearchHit> {
        let fields = options
            .stored_fields
            .iter()
            .filter_map(|field| {
                self.state
                    .stored_values(candidate.doc, *field)
                    .map(|values| (*field, values.to_vec()))
            })
            .collect();
        let highlights = options
            .highlight_fields
            .iter()
            .filter_map(|field| self.highlight(query, candidate.doc, *field))
            .flatten()
            .collect();
        Ok(SearchHit {
            id: candidate.id,
            score: candidate.score,
            fields,
            sort_values: candidate.sort_values,
            highlights,
        })
    }

    fn highlight(&self, query: &Query, doc: DocId, field: FieldId) -> Option<Vec<Highlight>> {
        let FieldIndex::Text(index) = &self.state.fields[field.index()] else {
            return None;
        };
        let state = index.doc_states.get(doc as usize)?.as_ref()?;
        let mut matched = Vec::new();
        collect_text_queries(query, field, &mut matched);
        let mut terms = Vec::new();
        for (query, mode) in matched {
            for token in self.query_tokens(query) {
                terms.extend(
                    super::evaluate::text_candidates(index, &token.term, *mode)
                        .into_iter()
                        .map(|(term, _, _)| term),
                );
            }
        }
        terms.sort_unstable();
        terms.dedup();
        let stored = self.state.stored_values(doc, field)?;
        let mut highlights = Vec::new();
        for (value_index, value) in state.values.iter().enumerate() {
            let mut spans = Vec::new();
            for term in &terms {
                spans.extend(
                    value
                        .positions
                        .iter()
                        .filter(|entry| entry.term == *term)
                        .map(|entry| (entry.start, entry.end)),
                );
                for derived in value.derived.iter().filter(|entry| entry.derived == *term) {
                    let base_positions = value
                        .positions
                        .iter()
                        .filter(|entry| entry.term == derived.base)
                        .map(|entry| (entry.start, entry.end))
                        .collect::<Vec<_>>();
                    if !base_positions.is_empty() {
                        spans.extend(base_positions);
                    } else {
                        spans.push(derived.span);
                    }
                }
            }
            spans.sort_unstable();
            spans.dedup();
            spans = merge_spans(spans);
            if self.schema.position_encoding() == crate::PositionEncoding::Utf16
                && let Some(Value::String(text)) = stored.get(value_index)
            {
                spans = spans
                    .into_iter()
                    .map(|(start, end)| (utf16_offset(text, start), utf16_offset(text, end)))
                    .collect();
            }
            if !spans.is_empty() {
                highlights.push(Highlight {
                    field,
                    value_index: value_index as u16,
                    spans,
                });
            }
        }
        Some(highlights)
    }
}

fn effective_sorts(sorts: &[Sort]) -> Vec<Sort> {
    let mut result = if sorts.is_empty() {
        vec![Sort::ScoreDesc]
    } else {
        sorts.to_vec()
    };
    if !result.iter().any(|sort| matches!(sort, Sort::DocumentId)) {
        result.push(Sort::DocumentId);
    }
    result
}

#[derive(Debug)]
pub(super) struct Candidate {
    doc: DocId,
    id: String,
    score: f32,
    sort_values: Vec<SortValue>,
    sorts: Vec<Sort>,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.doc == other.doc
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_candidates(self, other, &self.sorts)
    }
}

fn compare_candidates(a: &Candidate, b: &Candidate, sorts: &[Sort]) -> Ordering {
    compare_sort_values(&a.sort_values, &b.sort_values, sorts)
}

fn compare_sort_values(a: &[SortValue], b: &[SortValue], sorts: &[Sort]) -> Ordering {
    for ((a, b), sort) in a.iter().zip(b).zip(sorts) {
        let order = sort_value_order(a, b);
        let order = match sort {
            Sort::ScoreDesc
            | Sort::Field {
                order: SortOrder::Desc,
                ..
            } => order.reverse(),
            _ => order,
        };
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}

fn sort_value_order(a: &SortValue, b: &SortValue) -> Ordering {
    match (a, b) {
        (SortValue::Missing, SortValue::Missing) => Ordering::Equal,
        (SortValue::Missing, _) => Ordering::Greater,
        (_, SortValue::Missing) => Ordering::Less,
        (SortValue::Score(a), SortValue::Score(b)) => a.total_cmp(b),
        (SortValue::String(a), SortValue::String(b)) => a.cmp(b),
        (SortValue::I64(a), SortValue::I64(b)) => a.cmp(b),
        (SortValue::Bool(a), SortValue::Bool(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

fn collect_text_queries<'a>(
    query: &'a Query,
    field: FieldId,
    output: &mut Vec<(&'a str, &'a crate::SearchMode)>,
) {
    match query {
        Query::Text {
            field: query_field,
            query,
            mode,
        } if *query_field == field => output.push((query, mode)),
        Query::Boolean {
            must,
            should,
            must_not,
            ..
        } => {
            for query in must.iter().chain(should).chain(must_not) {
                collect_text_queries(query, field, output);
            }
        }
        Query::Boost { query, .. } => collect_text_queries(query, field, output),
        _ => {}
    }
}

fn merge_spans(spans: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut()
            && span.0 <= last.1
        {
            last.1 = last.1.max(span.1);
        } else {
            merged.push(span);
        }
    }
    merged
}

fn utf16_offset(text: &str, byte: u32) -> u32 {
    text[..(byte as usize).min(text.len())]
        .encode_utf16()
        .count() as u32
}
