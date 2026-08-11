use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{DictionaryConfig, Error, PositionEncoding, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldId(pub u16);

impl FieldId {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldOptions {
    pub indexed: bool,
    pub stored: bool,
    pub sortable: bool,
    pub multi_value: bool,
}

impl FieldOptions {
    pub const fn new() -> Self {
        Self {
            indexed: false,
            stored: false,
            sortable: false,
            multi_value: false,
        }
    }

    pub const fn indexed(mut self) -> Self {
        self.indexed = true;
        self
    }

    pub const fn stored(mut self) -> Self {
        self.stored = true;
        self
    }

    pub const fn sortable(mut self) -> Self {
        self.sortable = true;
        self
    }

    pub const fn multi_value(mut self) -> Self {
        self.multi_value = true;
        self
    }

    pub const fn indexed_stored() -> Self {
        Self::new().indexed().stored()
    }
}

impl Default for FieldOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextOptions {
    pub pinyin: bool,
    pub prefix: bool,
    pub fuzzy: bool,
    pub positions: bool,
}

impl TextOptions {
    pub const fn multilingual() -> Self {
        Self {
            pinyin: false,
            prefix: false,
            fuzzy: false,
            positions: false,
        }
    }

    pub const fn with_pinyin(mut self) -> Self {
        self.pinyin = true;
        self
    }

    pub const fn with_prefix(mut self) -> Self {
        self.prefix = true;
        self
    }

    pub const fn with_fuzzy(mut self) -> Self {
        self.fuzzy = true;
        self
    }

    pub const fn with_positions(mut self) -> Self {
        self.positions = true;
        self
    }
}

impl Default for TextOptions {
    fn default() -> Self {
        Self::multilingual()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    Text(TextOptions),
    Keyword,
    I64,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Field {
    pub id: FieldId,
    pub name: String,
    pub field_type: FieldType,
    pub options: FieldOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    fields: Vec<Field>,
    position_encoding: PositionEncoding,
    dictionary: Option<DictionaryConfig>,
    fingerprint: [u8; 32],
}

impl Schema {
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    pub fn field(&self, id: FieldId) -> Option<&Field> {
        self.fields.get(id.index())
    }

    pub fn field_by_name(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|field| field.name == name)
    }

    pub fn position_encoding(&self) -> PositionEncoding {
        self.position_encoding
    }

    pub fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    pub(crate) fn dictionary(&self) -> Option<&DictionaryConfig> {
        self.dictionary.as_ref()
    }

    pub(crate) fn validate_fingerprint(&self) -> bool {
        self.fingerprint
            == schema_fingerprint(
                &self.fields,
                self.position_encoding,
                self.dictionary.as_ref(),
            )
    }
}

#[derive(Debug, Default)]
pub struct SchemaBuilder {
    fields: Vec<(String, FieldType, FieldOptions)>,
    position_encoding: PositionEncoding,
    dictionary: Option<DictionaryConfig>,
}

impl SchemaBuilder {
    pub fn position_encoding(mut self, encoding: PositionEncoding) -> Self {
        self.position_encoding = encoding;
        self
    }

    pub fn dictionary(mut self, dictionary: DictionaryConfig) -> Self {
        self.dictionary = Some(dictionary);
        self
    }

    pub fn text(
        &mut self,
        name: impl Into<String>,
        text: TextOptions,
        options: FieldOptions,
    ) -> FieldId {
        self.push(name.into(), FieldType::Text(text), options)
    }

    pub fn keyword(&mut self, name: impl Into<String>, options: FieldOptions) -> FieldId {
        self.push(name.into(), FieldType::Keyword, options)
    }

    pub fn i64(&mut self, name: impl Into<String>, options: FieldOptions) -> FieldId {
        self.push(name.into(), FieldType::I64, options)
    }

    pub fn bool(&mut self, name: impl Into<String>, options: FieldOptions) -> FieldId {
        self.push(name.into(), FieldType::Bool, options)
    }

    fn push(&mut self, name: String, field_type: FieldType, options: FieldOptions) -> FieldId {
        let id = FieldId(u16::try_from(self.fields.len()).expect("schema field limit exceeded"));
        self.fields.push((name, field_type, options));
        id
    }

    pub fn build(self) -> Result<Schema> {
        let mut names = HashSet::new();
        let mut fields = Vec::with_capacity(self.fields.len());
        for (index, (name, field_type, options)) in self.fields.into_iter().enumerate() {
            if name.is_empty() || !names.insert(name.clone()) {
                return Err(Error::InvalidSchema(format!(
                    "duplicate or empty field name: {name}"
                )));
            }
            if !options.indexed && !options.stored && !options.sortable {
                return Err(Error::InvalidSchema(format!(
                    "field {name} must be indexed, stored, or sortable"
                )));
            }
            if options.sortable && options.multi_value {
                return Err(Error::InvalidSchema(format!(
                    "sortable field {name} cannot be multi-value"
                )));
            }
            if options.sortable && matches!(field_type, FieldType::Text(_)) {
                return Err(Error::InvalidSchema(format!(
                    "text field {name} cannot be sortable"
                )));
            }
            if let FieldType::Text(text) = field_type
                && text.positions
                && !options.indexed
            {
                return Err(Error::InvalidSchema(format!(
                    "text positions require indexed field {name}"
                )));
            }
            fields.push(Field {
                id: FieldId(index as u16),
                name,
                field_type,
                options,
            });
        }
        let fingerprint =
            schema_fingerprint(&fields, self.position_encoding, self.dictionary.as_ref());
        Ok(Schema {
            fields,
            position_encoding: self.position_encoding,
            dictionary: self.dictionary,
            fingerprint,
        })
    }
}

fn schema_fingerprint(
    fields: &[Field],
    position_encoding: PositionEncoding,
    dictionary: Option<&DictionaryConfig>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"memory-indexer-schema-v1");
    hasher.update([position_encoding as u8]);
    for field in fields {
        hasher.update(field.id.0.to_le_bytes());
        hasher.update((field.name.len() as u64).to_le_bytes());
        hasher.update(field.name.as_bytes());
        match field.field_type {
            FieldType::Text(options) => {
                hasher.update([
                    0,
                    options.pinyin as u8,
                    options.prefix as u8,
                    options.fuzzy as u8,
                    options.positions as u8,
                ]);
            }
            FieldType::Keyword => hasher.update([1]),
            FieldType::I64 => hasher.update([2]),
            FieldType::Bool => hasher.update([3]),
        }
        hasher.update([
            field.options.indexed as u8,
            field.options.stored as u8,
            field.options.sortable as u8,
            field.options.multi_value as u8,
        ]);
    }
    for dictionary in [
        dictionary.and_then(|d| d.japanese.as_ref()),
        dictionary.and_then(|d| d.hangul.as_ref()),
    ] {
        match dictionary {
            Some(dictionary) => {
                hasher.update([1]);
                if let Some(version) = &dictionary.version {
                    hasher.update(version.as_bytes());
                }
                let mut entries: Vec<_> = dictionary.entries.iter().collect();
                entries.sort();
                for entry in entries {
                    hasher.update((entry.len() as u64).to_le_bytes());
                    hasher.update(entry.as_bytes());
                }
            }
            None => hasher.update([0]),
        }
    }
    hasher.finalize().into()
}
