use std::path::PathBuf;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid dataset manifest JSON: {0}")]
    ManifestJson(#[from] serde_json::Error),
    #[error("dataset validation failed: {0}")]
    Validation(String),
    #[error("invalid window request: {0}")]
    InvalidWindow(String),
    #[error("exact alignment failed: {0}")]
    Alignment(String),
    #[error("source {path} is {actual} bytes but at least {required} bytes are required")]
    SourceTooShort {
        path: PathBuf,
        actual: u64,
        required: u64,
    },
    #[error("source SHA-256 mismatch at {path}: expected {expected}, got {actual}")]
    SourceHashMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("runtime worker terminated before producing all requested windows")]
    WorkerTerminated,
}

impl RuntimeError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
