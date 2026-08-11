use serde::{Deserialize, Serialize};

use crate::FieldId;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Value {
    String(String),
    I64(i64),
    Bool(bool),
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub id: String,
    pub fields: Vec<(FieldId, Vec<Value>)>,
}

impl Document {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: Vec::new(),
        }
    }

    pub fn add(&mut self, field: FieldId, value: impl Into<Value>) -> &mut Self {
        self.add_values(field, [value.into()])
    }

    pub fn add_values(
        &mut self,
        field: FieldId,
        values: impl IntoIterator<Item = Value>,
    ) -> &mut Self {
        let values = values.into_iter().collect::<Vec<_>>();
        if let Some((_, current)) = self.fields.iter_mut().find(|(id, _)| *id == field) {
            current.extend(values);
        } else {
            self.fields.push((field, values));
        }
        self
    }
}

#[derive(Debug, Clone)]
pub enum Mutation {
    Upsert(Document),
    Delete(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutationResult {
    pub inserted: bool,
    pub sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchResult {
    pub upserted: usize,
    pub deleted: usize,
    pub sequence: u64,
}
