use std::io::{Cursor, Read};

use serde::{Deserialize, Serialize};

use crate::{Error, MemoryIndex, Result, Schema, index::MemoryIndexState};

const MAGIC: &[u8; 8] = b"MEMIDX01";
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 8 + 4 + 32 + 8 + 8 + 4;

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub sequence: u64,
    pub bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct Payload {
    schema: Schema,
    state: MemoryIndexState,
}

impl MemoryIndex {
    pub fn checkpoint(&self) -> Result<Checkpoint> {
        let sequence = self.state.change_sequence;
        let encoded = bincode::serde::encode_to_vec(
            PayloadRef {
                schema: &self.schema,
                state: &self.state,
            },
            bincode::config::standard().with_fixed_int_encoding(),
        )
        .map_err(|error| Error::Codec(error.to_string()))?;
        let compressed =
            zstd::bulk::compress(&encoded, 1).map_err(|error| Error::Codec(error.to_string()))?;
        let checksum = crc32fast::hash(&compressed);
        let mut bytes = Vec::with_capacity(HEADER_LEN + compressed.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.schema.fingerprint());
        bytes.extend_from_slice(&sequence.to_le_bytes());
        bytes.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&checksum.to_le_bytes());
        bytes.extend_from_slice(&compressed);
        Ok(Checkpoint { sequence, bytes })
    }

    pub fn from_checkpoint(expected_schema: Schema, bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN || &bytes[..8] != MAGIC {
            return Err(Error::InvalidCheckpoint(
                "invalid checkpoint magic or truncated header".into(),
            ));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(Error::InvalidCheckpoint(format!(
                "unsupported checkpoint version {version}"
            )));
        }
        let fingerprint: [u8; 32] = bytes[12..44].try_into().unwrap();
        if fingerprint != expected_schema.fingerprint() {
            return Err(Error::InvalidCheckpoint(
                "checkpoint schema does not match expected schema".into(),
            ));
        }
        let sequence = u64::from_le_bytes(bytes[44..52].try_into().unwrap());
        let payload_len = u64::from_le_bytes(bytes[52..60].try_into().unwrap()) as usize;
        let checksum = u32::from_le_bytes(bytes[60..64].try_into().unwrap());
        let payload = bytes
            .get(HEADER_LEN..)
            .ok_or_else(|| Error::InvalidCheckpoint("checkpoint payload is truncated".into()))?;
        if payload.len() != payload_len || crc32fast::hash(payload) != checksum {
            return Err(Error::InvalidCheckpoint(
                "checkpoint payload length or checksum is invalid".into(),
            ));
        }
        let mut decoder = zstd::stream::read::Decoder::new(Cursor::new(payload))
            .map_err(|error| Error::InvalidCheckpoint(error.to_string()))?;
        let decoded: Payload = bincode::serde::decode_from_std_read(
            &mut decoder,
            bincode::config::standard().with_fixed_int_encoding(),
        )
        .map_err(|error| Error::InvalidCheckpoint(error.to_string()))?;
        let mut trailing = [0u8; 1];
        if decoder
            .read(&mut trailing)
            .map_err(|error| Error::InvalidCheckpoint(error.to_string()))?
            != 0
        {
            return Err(Error::InvalidCheckpoint(
                "checkpoint contains trailing decoded bytes".into(),
            ));
        }
        if decoded.schema.fingerprint() != fingerprint || !decoded.schema.validate_fingerprint() {
            return Err(Error::InvalidCheckpoint(
                "checkpoint schema fingerprint is inconsistent".into(),
            ));
        }
        let mut state = decoded.state;
        state.rebuild_lookup_maps();
        if state.change_sequence != sequence {
            return Err(Error::InvalidCheckpoint(
                "checkpoint change sequence is inconsistent".into(),
            ));
        }
        if let Some(error) = state.validation_error(&expected_schema) {
            return Err(Error::InvalidCheckpoint(format!(
                "checkpoint document state is inconsistent: {error}"
            )));
        }
        state.persisted_sequence = sequence;
        Ok(Self {
            schema: expected_schema,
            state,
        })
    }

    pub fn mark_checkpoint_persisted(&mut self, sequence: u64) -> Result<()> {
        if sequence > self.state.change_sequence {
            return Err(Error::CheckpointSequenceInFuture {
                sequence,
                current: self.state.change_sequence,
            });
        }
        self.state.persisted_sequence = self.state.persisted_sequence.max(sequence);
        Ok(())
    }

    pub fn has_unpersisted_changes(&self) -> bool {
        self.state.change_sequence > self.state.persisted_sequence
    }

    pub fn change_sequence(&self) -> u64 {
        self.state.change_sequence
    }
}

#[derive(Serialize)]
struct PayloadRef<'a> {
    schema: &'a Schema,
    state: &'a MemoryIndexState,
}
