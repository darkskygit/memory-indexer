use std::collections::HashSet;

use smol_str::SmolStr;

use crate::{
    BatchResult, Document, Error, FieldType, MemoryIndex, Mutation, MutationResult, Result, Value,
    pipeline::{DefaultTokenizer, Pipeline},
};

use super::{
    DocId, DocumentSlot, FieldIndex, PreparedDocument, PreparedField, StoredValues,
    text::{PreparedText, PreparedTextValue},
};

impl MemoryIndex {
    pub fn upsert(&mut self, document: Document) -> Result<MutationResult> {
        let inserted = !self
            .state
            .documents
            .external_ids
            .contains_key(document.id.as_str());
        let prepared = self.prepare_document(document)?;
        self.apply_prepared(prepared);
        self.state.change_sequence += 1;
        Ok(MutationResult {
            inserted,
            sequence: self.state.change_sequence,
        })
    }

    pub fn delete(&mut self, id: &str) -> bool {
        let Some(doc) = self.state.documents.external_ids.get(id).copied() else {
            return false;
        };
        if !self.remove_doc(doc, true) {
            return false;
        }
        self.state.change_sequence += 1;
        true
    }

    pub fn apply_batch(&mut self, mutations: Vec<Mutation>) -> Result<BatchResult> {
        let mut prepared = Vec::with_capacity(mutations.len());
        for mutation in mutations {
            prepared.push(match mutation {
                Mutation::Upsert(document) => {
                    PreparedMutation::Upsert(self.prepare_document(document)?)
                }
                Mutation::Delete(id) => PreparedMutation::Delete(id),
            });
        }
        let mut upserted = 0;
        let mut deleted = 0;
        for mutation in prepared {
            match mutation {
                PreparedMutation::Upsert(document) => {
                    self.apply_prepared(document);
                    upserted += 1;
                }
                PreparedMutation::Delete(id) => {
                    if let Some(doc) = self.state.documents.external_ids.get(id.as_str()).copied()
                        && self.remove_doc(doc, true)
                    {
                        deleted += 1;
                    }
                }
            }
        }
        if upserted > 0 || deleted > 0 {
            self.state.change_sequence += 1;
        }
        Ok(BatchResult {
            upserted,
            deleted,
            sequence: self.state.change_sequence,
        })
    }

    fn prepare_document(&self, document: Document) -> Result<PreparedDocument> {
        if document.id.is_empty() {
            return Err(Error::InvalidDocument("document id cannot be empty".into()));
        }
        let field_count = self.schema.fields().len();
        let mut source = vec![None; field_count];
        for (field, values) in document.fields {
            let definition = self
                .schema
                .field(field)
                .ok_or_else(|| Error::InvalidDocument(format!("unknown field id {}", field.0)))?;
            if source[field.index()].is_some() {
                return Err(Error::InvalidDocument(format!(
                    "field {} occurs more than once",
                    definition.name
                )));
            }
            if values.is_empty() {
                continue;
            }
            if !definition.options.multi_value && values.len() > 1 {
                return Err(Error::InvalidDocument(format!(
                    "field {} is single-value",
                    definition.name
                )));
            }
            for value in &values {
                let valid = matches!(
                    (&definition.field_type, value),
                    (FieldType::Text(_), Value::String(_))
                        | (FieldType::Keyword, Value::String(_))
                        | (FieldType::I64, Value::I64(_))
                        | (FieldType::Bool, Value::Bool(_))
                );
                if !valid {
                    return Err(Error::InvalidDocument(format!(
                        "value type does not match field {}",
                        definition.name
                    )));
                }
            }
            source[field.index()] = Some(values);
        }

        let pipeline = match self.schema.dictionary() {
            Some(dictionary) => {
                Pipeline::new(DefaultTokenizer::for_documents().with_dictionary(dictionary.clone()))
            }
            None => Pipeline::document_pipeline(),
        };
        let mut stored = std::iter::repeat_with(|| None)
            .take(field_count)
            .collect::<Vec<_>>();
        let mut indexed = Vec::with_capacity(field_count);
        for field in self.schema.fields() {
            let Some(values) = source[field.id.index()].take() else {
                indexed.push(PreparedField::Missing);
                continue;
            };
            let prepared = if !field.options.indexed && !field.options.sortable {
                PreparedField::Missing
            } else {
                match field.field_type {
                    FieldType::Text(_) => PreparedField::Text(PreparedText {
                        values: values
                            .iter()
                            .map(|value| match value {
                                Value::String(text) => PreparedTextValue {
                                    tokens: pipeline.document_tokens(text).tokens,
                                },
                                _ => unreachable!(),
                            })
                            .collect(),
                    }),
                    FieldType::Keyword => {
                        let mut unique = HashSet::new();
                        let mut values = values
                            .iter()
                            .filter_map(|value| match value {
                                Value::String(value) if unique.insert(value.as_str()) => {
                                    Some(SmolStr::new(value))
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        values.sort();
                        PreparedField::Keyword(values)
                    }
                    FieldType::I64 => match values[0] {
                        Value::I64(value) => PreparedField::I64(value),
                        _ => unreachable!(),
                    },
                    FieldType::Bool => match values[0] {
                        Value::Bool(value) => PreparedField::Bool(value),
                        _ => unreachable!(),
                    },
                }
            };
            indexed.push(prepared);
            if field.options.stored {
                stored[field.id.index()] = Some(StoredValues::new(values));
            }
        }
        Ok(PreparedDocument {
            external_id: SmolStr::new(document.id),
            stored,
            indexed,
        })
    }

    fn apply_prepared(&mut self, prepared: PreparedDocument) {
        let doc = if let Some(existing) = self
            .state
            .documents
            .external_ids
            .get(&prepared.external_id)
            .copied()
        {
            assert!(
                self.remove_doc(existing, false),
                "external id {} points to empty document slot {existing}",
                prepared.external_id
            );
            existing
        } else if let Some(reused) = self.state.documents.free.pop() {
            reused
        } else {
            self.state.documents.slots.len() as DocId
        };
        let capacity = doc as usize + 1;
        if self.state.documents.slots.len() < capacity {
            self.state.documents.slots.resize_with(capacity, || None);
        }
        for field in &mut self.state.fields {
            field.ensure_doc_capacity(capacity);
        }

        for (field, value) in self.state.fields.iter_mut().zip(prepared.indexed) {
            match (field, value) {
                (_, PreparedField::Missing) => {}
                (FieldIndex::Text(index), PreparedField::Text(value)) => {
                    index.insert(doc, value);
                }
                (FieldIndex::Keyword(index), PreparedField::Keyword(values)) => {
                    index.insert(doc, &values);
                }
                (FieldIndex::I64(index), PreparedField::I64(value)) => {
                    index.insert(doc, value);
                }
                (FieldIndex::Bool(index), PreparedField::Bool(value)) => {
                    index.insert(doc, value);
                }
                _ => unreachable!("prepared field type was validated against schema"),
            }
        }
        self.state.live_docs.insert(doc);
        self.state
            .documents
            .external_ids
            .insert(prepared.external_id.clone(), doc);
        self.state.documents.slots[doc as usize] = Some(DocumentSlot {
            external_id: prepared.external_id,
            stored: prepared.stored,
        });
    }

    pub(crate) fn remove_doc(&mut self, doc: DocId, release: bool) -> bool {
        let Some(slot) = self
            .state
            .documents
            .slots
            .get_mut(doc as usize)
            .and_then(Option::take)
        else {
            return false;
        };
        for field in &mut self.state.fields {
            match field {
                FieldIndex::Text(index) => index.remove(doc),
                FieldIndex::Keyword(index) => index.remove(doc),
                FieldIndex::I64(index) => index.remove(doc),
                FieldIndex::Bool(index) => index.remove(doc),
            }
        }
        self.state.live_docs.remove(doc);
        if release {
            self.state.documents.external_ids.remove(&slot.external_id);
            self.state.documents.free.push(doc);
        }
        true
    }
}

enum PreparedMutation {
    Upsert(PreparedDocument),
    Delete(String),
}
