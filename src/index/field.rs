use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::{Field, FieldType};

use super::{DocId, TextIndex};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum FieldIndex {
    Text(TextIndex),
    Keyword(KeywordIndex),
    I64(I64Index),
    Bool(BoolIndex),
}

impl FieldIndex {
    pub(crate) fn new(field: &Field) -> Self {
        match field.field_type {
            FieldType::Text(options) => Self::Text(TextIndex::new(options)),
            FieldType::Keyword => Self::Keyword(KeywordIndex::new(field.options.sortable)),
            FieldType::I64 => {
                Self::I64(I64Index::new(field.options.indexed, field.options.sortable))
            }
            FieldType::Bool => Self::Bool(BoolIndex::new(field.options.sortable)),
        }
    }

    pub(crate) fn exists(&self) -> &RoaringBitmap {
        match self {
            Self::Text(index) => &index.exists,
            Self::Keyword(index) => &index.exists,
            Self::I64(index) => &index.exists,
            Self::Bool(index) => &index.exists,
        }
    }

    pub(crate) fn ensure_doc_capacity(&mut self, capacity: usize) {
        match self {
            Self::Text(index) => index.ensure_doc_capacity(capacity),
            Self::Keyword(index) => index.ensure_doc_capacity(capacity),
            Self::I64(index) => index.ensure_doc_capacity(capacity),
            Self::Bool(index) => index.ensure_doc_capacity(capacity),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct KeywordIndex {
    pub(crate) values: Vec<SmolStr>,
    #[serde(skip, default)]
    pub(crate) value_ids: HashMap<SmolStr, u32>,
    pub(crate) postings: Vec<RoaringBitmap>,
    pub(crate) exists: RoaringBitmap,
    pub(crate) sortable_values: Option<Vec<Option<u32>>>,
    pub(crate) doc_values: Vec<Vec<u32>>,
}

impl KeywordIndex {
    pub(crate) fn rebuild_lookup_map(&mut self) {
        self.value_ids = self
            .values
            .iter()
            .enumerate()
            .map(|(id, value)| (value.clone(), id as u32))
            .collect();
    }

    fn new(sortable: bool) -> Self {
        Self {
            values: Vec::new(),
            value_ids: HashMap::new(),
            postings: Vec::new(),
            exists: RoaringBitmap::new(),
            sortable_values: sortable.then(Vec::new),
            doc_values: Vec::new(),
        }
    }

    pub(crate) fn ensure_doc_capacity(&mut self, capacity: usize) {
        if self.doc_values.len() < capacity {
            self.doc_values.resize_with(capacity, Vec::new);
        }
        if let Some(values) = &mut self.sortable_values
            && values.len() < capacity
        {
            values.resize(capacity, None);
        }
    }

    pub(crate) fn insert(&mut self, doc: DocId, values: &[SmolStr]) {
        let mut ids = values
            .iter()
            .map(|value| {
                if let Some(id) = self.value_ids.get(value) {
                    *id
                } else {
                    let id = self.values.len() as u32;
                    self.values.push(value.clone());
                    self.value_ids.insert(value.clone(), id);
                    self.postings.push(RoaringBitmap::new());
                    id
                }
            })
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        for id in &ids {
            self.postings[*id as usize].insert(doc);
        }
        self.exists.insert(doc);
        if let Some(sortable) = &mut self.sortable_values {
            sortable[doc as usize] = ids.first().copied();
        }
        self.doc_values[doc as usize] = ids.clone();
    }

    pub(crate) fn remove(&mut self, doc: DocId) {
        for value in std::mem::take(&mut self.doc_values[doc as usize]) {
            if let Some(postings) = self.postings.get_mut(value as usize) {
                postings.remove(doc);
            }
        }
        self.exists.remove(doc);
        if let Some(sortable) = &mut self.sortable_values
            && let Some(value) = sortable.get_mut(doc as usize)
        {
            *value = None;
        }
    }

    pub(crate) fn term(&self, value: &str) -> RoaringBitmap {
        self.value_ids
            .get(value)
            .and_then(|id| self.postings.get(*id as usize))
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn sort_value(&self, doc: DocId) -> Option<&str> {
        let id = self.sortable_values.as_ref()?.get(doc as usize)?.as_ref()?;
        self.values.get(*id as usize).map(SmolStr::as_str)
    }

    pub(crate) fn optimize(&mut self) -> (Vec<Option<u32>>, usize) {
        let mut mapping = vec![None; self.values.len()];
        let mut values = Vec::new();
        let mut postings = Vec::new();
        for (old, (value, docs)) in std::mem::take(&mut self.values)
            .into_iter()
            .zip(std::mem::take(&mut self.postings))
            .enumerate()
        {
            if docs.is_empty() {
                continue;
            }
            let new = values.len() as u32;
            mapping[old] = Some(new);
            values.push(value);
            postings.push(docs);
        }
        let removed = mapping.iter().filter(|value| value.is_none()).count();
        self.value_ids = values
            .iter()
            .enumerate()
            .map(|(id, value)| (value.clone(), id as u32))
            .collect();
        self.values = values;
        self.postings = postings;
        if let Some(sortable) = &mut self.sortable_values {
            for value in sortable.iter_mut().flatten() {
                *value = mapping[*value as usize].expect("live sortable value must be retained");
            }
        }
        for values in &mut self.doc_values {
            for value in values {
                *value = mapping[*value as usize].expect("live keyword value must be retained");
            }
        }
        (mapping, removed)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct I64Index {
    pub(crate) exact: Option<BTreeMap<i64, RoaringBitmap>>,
    pub(crate) values: Vec<Option<i64>>,
    pub(crate) exists: RoaringBitmap,
}

impl I64Index {
    fn new(indexed: bool, _sortable: bool) -> Self {
        Self {
            exact: indexed.then(BTreeMap::new),
            values: Vec::new(),
            exists: RoaringBitmap::new(),
        }
    }

    pub(crate) fn ensure_doc_capacity(&mut self, capacity: usize) {
        if self.values.len() < capacity {
            self.values.resize(capacity, None);
        }
    }

    pub(crate) fn insert(&mut self, doc: DocId, value: i64) {
        if let Some(exact) = &mut self.exact {
            exact.entry(value).or_default().insert(doc);
        }
        self.values[doc as usize] = Some(value);
        self.exists.insert(doc);
    }

    pub(crate) fn remove(&mut self, doc: DocId) {
        let Some(value) = self.values[doc as usize].take() else {
            return;
        };
        if let Some(exact) = &mut self.exact
            && let Some(postings) = exact.get_mut(&value)
        {
            postings.remove(doc);
        }
        self.exists.remove(doc);
    }

    pub(crate) fn term(&self, value: i64) -> RoaringBitmap {
        self.exact
            .as_ref()
            .and_then(|exact| exact.get(&value))
            .cloned()
            .unwrap_or_default()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BoolIndex {
    pub(crate) false_docs: RoaringBitmap,
    pub(crate) true_docs: RoaringBitmap,
    pub(crate) values: Vec<Option<bool>>,
    pub(crate) exists: RoaringBitmap,
}

impl BoolIndex {
    fn new(_sortable: bool) -> Self {
        Self {
            false_docs: RoaringBitmap::new(),
            true_docs: RoaringBitmap::new(),
            values: Vec::new(),
            exists: RoaringBitmap::new(),
        }
    }

    pub(crate) fn ensure_doc_capacity(&mut self, capacity: usize) {
        if self.values.len() < capacity {
            self.values.resize(capacity, None);
        }
    }

    pub(crate) fn insert(&mut self, doc: DocId, value: bool) {
        if value {
            self.true_docs.insert(doc);
        } else {
            self.false_docs.insert(doc);
        }
        self.values[doc as usize] = Some(value);
        self.exists.insert(doc);
    }

    pub(crate) fn remove(&mut self, doc: DocId) {
        let Some(value) = self.values[doc as usize].take() else {
            return;
        };
        if value {
            self.true_docs.remove(doc);
        } else {
            self.false_docs.remove(doc);
        }
        self.exists.remove(doc);
    }

    pub(crate) fn term(&self, value: bool) -> RoaringBitmap {
        if value {
            self.true_docs.clone()
        } else {
            self.false_docs.clone()
        }
    }
}
