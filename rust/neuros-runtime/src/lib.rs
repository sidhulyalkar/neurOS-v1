#![forbid(unsafe_op_in_unsafe_fn)]

//! neurOS native data plane.
//!
//! The crate intentionally owns only transport and representation concerns:
//! validated dataset manifests, memory mapping, deterministic window planning,
//! bounded prefetch, zero-copy Arrow views, content provenance, exact
//! integer-clock synchronization planning, and execution of already-qualified
//! exact alignment plans. Scientific transforms and model semantics remain in
//! the Python control plane unless promoted into explicit, versioned runtime
//! adapters.

mod content_identity;
mod dataset {
    include!("dataset.rs");

    mod aligned_fresh {
        include!("aligned_fresh.rs");
    }

    mod aligned {
        include!("aligned.rs");
    }

    pub use aligned::{AlignedWindowHandle, AlignedWindowStream};
}
mod error;
mod manifest;
mod sync;

pub use content_identity::{DATASET_CONTENT_DOMAIN, declared_dataset_content_sha256};
pub use dataset::{
    AlignedWindowHandle, AlignedWindowStream, Dataset, SourceVerificationState, StreamSelector,
    WindowHandle, WindowSpec, WindowStream,
};
pub use error::{Result, RuntimeError};
pub use manifest::{
    ClockSpec, DType, DatasetManifest, MANIFEST_FILE, MANIFEST_SCHEMA_VERSION, Record,
};
pub use sync::{
    AlignedRecordPlan, AlignmentPolicy, EXACT_ALIGNMENT_PLAN_SCHEMA_VERSION, ExactAlignmentPlan,
    ExactAlignmentSpec, plan_exact_alignment,
};

#[cfg(test)]
impl std::fmt::Debug for WindowHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WindowHandle")
            .field("record_id", &self.record_id())
            .field("start_frame", &self.start_frame())
            .field("end_frame_exclusive", &self.end_frame_exclusive())
            .finish_non_exhaustive()
    }
}
