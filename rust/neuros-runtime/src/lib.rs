#![forbid(unsafe_op_in_unsafe_fn)]

//! neurOS native data plane.
//!
//! The crate intentionally owns only transport and representation concerns:
//! validated dataset manifests, memory mapping, deterministic window planning,
//! bounded prefetch, zero-copy Arrow views, and content provenance. Scientific
//! transforms and model semantics remain in the Python control plane unless
//! promoted into explicit, versioned runtime adapters.

mod content_identity;
mod dataset;
mod error;
mod manifest;

pub use content_identity::{DATASET_CONTENT_DOMAIN, declared_dataset_content_sha256};
pub use dataset::{
    Dataset, SourceVerificationState, StreamSelector, WindowHandle, WindowSpec, WindowStream,
};
pub use error::{Result, RuntimeError};
pub use manifest::{
    ClockSpec, DType, DatasetManifest, MANIFEST_FILE, MANIFEST_SCHEMA_VERSION, Record,
};
