mod collect;
mod evaluate;
mod types;

use std::collections::HashMap;

use roaring::RoaringBitmap;

use crate::{Error, FieldType, MemoryIndex, Result, Value, index::FieldIndex};

pub use types::{
    AggregationResult, Bucket, Highlight, Query, SearchHit, SearchOptions, SearchResult, Sort,
    SortOrder, SortValue, TermsAggregation,
};

use evaluate::EvalResult;

impl MemoryIndex {
    pub fn search(&self, query: &Query, options: SearchOptions) -> Result<SearchResult> {
        self.validate_search(query, &options)?;
        let evaluated = self.evaluate(query, Some(&self.state.live_docs))?;
        let total = evaluated.docs.len() as usize;
        if options.limit == 0 {
            return Ok(SearchResult {
                total,
                hits: Vec::new(),
            });
        }
        let candidates = self.collect_top_k(&evaluated, &options)?;
        let hits = candidates
            .into_iter()
            .skip(options.offset)
            .map(|candidate| self.materialize_hit(query, candidate, &options))
            .collect::<Result<Vec<_>>>()?;
        Ok(SearchResult { total, hits })
    }

    pub fn aggregate(
        &self,
        query: &Query,
        aggregation: TermsAggregation,
    ) -> Result<AggregationResult> {
        let field = self
            .schema
            .field(aggregation.field)
            .ok_or_else(|| Error::InvalidQuery("unknown aggregation field".into()))?;
        if !matches!(field.field_type, FieldType::Keyword) || !field.options.indexed {
            return Err(Error::InvalidQuery(
                "terms aggregation requires indexed Keyword field".into(),
            ));
        }
        if let Some(options) = &aggregation.top_hits {
            self.validate_search(query, options)?;
        }
        let evaluated = self.evaluate(query, Some(&self.state.live_docs))?;
        let FieldIndex::Keyword(index) = &self.state.fields[aggregation.field.index()] else {
            unreachable!()
        };
        let mut buckets: HashMap<u32, BucketState> = HashMap::new();
        for doc in evaluated.docs.iter() {
            let values = &index.doc_values[doc as usize];
            if values.is_empty() {
                continue;
            }
            let score = evaluated
                .scores
                .get(doc as usize)
                .copied()
                .unwrap_or_default();
            for value in values {
                let bucket = buckets.entry(*value).or_insert_with(|| BucketState {
                    docs: RoaringBitmap::new(),
                    count: 0,
                    max_score: 0.0,
                });
                bucket.docs.insert(doc);
                bucket.count += 1;
                bucket.max_score = bucket.max_score.max(score);
            }
        }
        let total = evaluated.docs.len() as usize;
        let mut buckets = buckets.into_iter().collect::<Vec<_>>();
        buckets.sort_by(|(a_id, a), (b_id, b)| {
            b.max_score
                .total_cmp(&a.max_score)
                .then_with(|| index.values[*a_id as usize].cmp(&index.values[*b_id as usize]))
        });
        let buckets = buckets
            .into_iter()
            .skip(aggregation.offset)
            .take(aggregation.limit)
            .map(|(value, bucket)| {
                let hits = if let Some(options) = &aggregation.top_hits {
                    let bucket_eval = EvalResult {
                        docs: bucket.docs,
                        scores: evaluated.scores.clone(),
                    };
                    self.collect_top_k(&bucket_eval, options)?
                        .into_iter()
                        .skip(options.offset)
                        .map(|candidate| self.materialize_hit(query, candidate, options))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    Vec::new()
                };
                Ok(Bucket {
                    key: Value::String(index.values[value as usize].to_string()),
                    count: bucket.count,
                    max_score: bucket.max_score,
                    hits,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(AggregationResult { total, buckets })
    }

    pub fn delete_by_query(&mut self, query: &Query) -> Result<usize> {
        self.validate_query(query)?;
        let docs = self
            .evaluate(query, Some(&self.state.live_docs))?
            .docs
            .iter()
            .collect::<Vec<_>>();
        for doc in &docs {
            assert!(self.remove_doc(*doc, true));
        }
        if !docs.is_empty() {
            self.state.change_sequence += 1;
        }
        Ok(docs.len())
    }
}

struct BucketState {
    docs: RoaringBitmap,
    count: u64,
    max_score: f32,
}
