mod field;
mod mutation;
pub(crate) mod optimize;
mod text;

use std::collections::HashMap;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::{FieldId, Schema, Value};

pub(crate) type DocId = u32;
pub(crate) type TermId = u32;

pub(crate) use field::FieldIndex;
pub(crate) use text::TextIndex;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MemoryIndexState {
    pub(crate) documents: DocumentStore,
    pub(crate) fields: Vec<FieldIndex>,
    pub(crate) live_docs: RoaringBitmap,
    pub(crate) change_sequence: u64,
    pub(crate) persisted_sequence: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct DocumentStore {
    pub(crate) slots: Vec<Option<DocumentSlot>>,
    #[serde(skip, default)]
    pub(crate) external_ids: HashMap<SmolStr, DocId>,
    pub(crate) free: Vec<DocId>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DocumentSlot {
    pub(crate) external_id: SmolStr,
    pub(crate) stored: Vec<Option<StoredValues>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum StoredValues {
    One(Value),
    Many(Vec<Value>),
}

impl StoredValues {
    pub(crate) fn new(mut values: Vec<Value>) -> Self {
        if values.len() == 1 {
            Self::One(values.pop().unwrap())
        } else {
            Self::Many(values)
        }
    }

    pub(crate) fn as_slice(&self) -> &[Value] {
        match self {
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug)]
pub(crate) struct PreparedDocument {
    pub(crate) external_id: SmolStr,
    pub(crate) stored: Vec<Option<StoredValues>>,
    pub(crate) indexed: Vec<PreparedField>,
}

#[derive(Debug)]
pub(crate) enum PreparedField {
    Missing,
    Text(text::PreparedText),
    Keyword(Vec<SmolStr>),
    I64(i64),
    Bool(bool),
}

impl MemoryIndexState {
    pub(crate) fn new(schema: &Schema) -> Self {
        Self {
            documents: DocumentStore::default(),
            fields: schema.fields().iter().map(FieldIndex::new).collect(),
            live_docs: RoaringBitmap::new(),
            change_sequence: 0,
            persisted_sequence: 0,
        }
    }

    pub(crate) fn external_id(&self, doc: DocId) -> Option<&str> {
        self.documents
            .slots
            .get(doc as usize)
            .and_then(Option::as_ref)
            .map(|slot| slot.external_id.as_str())
    }

    pub(crate) fn stored_values(&self, doc: DocId, field: FieldId) -> Option<&[Value]> {
        self.documents
            .slots
            .get(doc as usize)
            .and_then(Option::as_ref)
            .and_then(|slot| slot.stored.get(field.index()))
            .and_then(Option::as_ref)
            .map(StoredValues::as_slice)
    }

    pub(crate) fn validation_error(&self, schema: &Schema) -> Option<String> {
        if self.fields.len() != schema.fields().len()
            || self.documents.slots.len() > u32::MAX as usize
        {
            return Some("field or document capacity mismatch".into());
        }
        for doc in self.live_docs.iter() {
            let Some(Some(slot)) = self.documents.slots.get(doc as usize) else {
                return Some(format!("live document {doc} has no slot"));
            };
            if self.documents.external_ids.get(&slot.external_id) != Some(&doc) {
                return Some(format!("live document {doc} has inconsistent external id"));
            }
            if slot.stored.len() != self.fields.len() {
                return Some(format!("live document {doc} has inconsistent field state"));
            }
        }
        None
    }

    pub(crate) fn rebuild_lookup_maps(&mut self) {
        self.documents.external_ids = self
            .documents
            .slots
            .iter()
            .enumerate()
            .filter_map(|(doc, slot)| {
                slot.as_ref()
                    .map(|slot| (slot.external_id.clone(), doc as DocId))
            })
            .collect();
        for field in &mut self.fields {
            match field {
                FieldIndex::Keyword(index) => index.rebuild_lookup_map(),
                FieldIndex::Text(index) => index.rebuild_lookup_map(),
                FieldIndex::I64(_) | FieldIndex::Bool(_) => {}
            }
        }
    }
}
