use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidSchema(String),
    InvalidDocument(String),
    InvalidQuery(String),
    QueryTooLarge { limit: usize, actual: usize },
    InvalidCheckpoint(String),
    CheckpointSequenceInFuture { sequence: u64, current: u64 },
    Codec(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchema(message)
            | Self::InvalidDocument(message)
            | Self::InvalidQuery(message)
            | Self::InvalidCheckpoint(message)
            | Self::Codec(message) => f.write_str(message),
            Self::QueryTooLarge { limit, actual } => {
                write!(f, "query has {actual} tokens; the limit is {limit}")
            }
            Self::CheckpointSequenceInFuture { sequence, current } => write!(
                f,
                "checkpoint sequence {sequence} is newer than index sequence {current}"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
