#![forbid(unsafe_op_in_unsafe_fn)]

//! neurOS native data plane.
//!
//! The crate intentionally owns only transport and representation concerns:
//! validated dataset manifests, memory mapping, deterministic window planning,
//! bounded prefetch, exact clock-domain synchronization, and zero-copy Arrow
//! views. Scientific transforms and model semantics remain in the Python control
//! plane unless promoted into explicit, versioned runtime adapters.

mod dataset;
mod error;
mod manifest;
mod sync;

pub use dataset::{Dataset, StreamSelector, WindowHandle, WindowSpec, WindowStream};
pub use error::{Result, RuntimeError};
pub use manifest::{
    ClockSpec, DType, DatasetManifest, MANIFEST_FILE, MANIFEST_SCHEMA_VERSION, Record,
};
pub use sync::{
    AlignedRecordPlan, AlignmentPolicy, EXACT_ALIGNMENT_PLAN_SCHEMA_VERSION, ExactAlignmentPlan,
    ExactAlignmentSpec, plan_exact_alignment,
};
