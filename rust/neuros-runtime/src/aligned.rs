use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender, bounded};
use tracing::debug;

use crate::dataset::{Dataset, WindowDescriptor, WindowHandle};
use crate::error::{Result, RuntimeError};
use crate::manifest::Record;
use crate::sync::{
    AlignedRecordPlan, AlignmentPolicy, EXACT_ALIGNMENT_PLAN_SCHEMA_VERSION, ExactAlignmentPlan,
};

#[derive(Clone)]
struct PreparedRecord {
    plan: AlignedRecordPlan,
    record: Arc<Record>,
    frame_elements: usize,
}

#[derive(Clone)]
struct PreparedAlignmentPlan {
    plan: ExactAlignmentPlan,
    plan_sha256: String,
    records: Vec<PreparedRecord>,
}

#[derive(Clone)]
pub struct AlignedWindowHandle {
    plan_sha256: String,
    dataset_content_sha256: String,
    manifest_sha256: String,
    sync_group: String,
    window_index: usize,
    start_ns: i64,
    end_ns: i64,
    windows: Vec<WindowHandle>,
}

enum AlignedStreamMessage {
    Window(Result<AlignedWindowHandle>),
    Finished,
}

pub struct AlignedWindowStream {
    receiver: Receiver<AlignedStreamMessage>,
    finished: bool,
}

impl Dataset {
    /// Execute a previously qualified exact-alignment plan with bounded prefetch.
    ///
    /// This method does not solve synchronization again. It validates the stored
    /// plan directly against the current manifest, performs a fresh physical-file
    /// integrity pass, and derives every source slice only from the plan's stored
    /// frame arithmetic.
    pub fn stream_aligned(
        self: &Arc<Self>,
        plan: &ExactAlignmentPlan,
        prefetch: usize,
    ) -> Result<AlignedWindowStream> {
        if prefetch == 0 {
            return Err(RuntimeError::Alignment(
                "aligned prefetch must be at least one".into(),
            ));
        }
        let prepared = prepare_alignment_plan(self.as_ref(), plan)?;
        let (sender, receiver) = bounded(prefetch);
        let dataset = Arc::clone(self);

        thread::Builder::new()
            .name("neuros-aligned-prefetch".into())
            .spawn(move || dataset.run_aligned_worker(prepared, sender))
            .map_err(|source| RuntimeError::io("<aligned-prefetch-thread>", source))?;

        Ok(AlignedWindowStream {
            receiver,
            finished: false,
        })
    }

    fn run_aligned_worker(
        self: Arc<Self>,
        prepared: PreparedAlignmentPlan,
        sender: Sender<AlignedStreamMessage>,
    ) {
        for window_index in 0..prepared.plan.window_count {
            match self.open_aligned_window(&prepared, window_index) {
                Ok(window) => {
                    if sender.send(AlignedStreamMessage::Window(Ok(window))).is_err() {
                        debug!("aligned window consumer dropped; cancelling prefetch worker");
                        return;
                    }
                }
                Err(error) => {
                    if sender
                        .send(AlignedStreamMessage::Window(Err(error)))
                        .is_ok()
                    {
                        let _ = sender.send(AlignedStreamMessage::Finished);
                    }
                    return;
                }
            }
        }
        let _ = sender.send(AlignedStreamMessage::Finished);
    }

    fn open_aligned_window(
        &self,
        prepared: &PreparedAlignmentPlan,
        window_index: usize,
    ) -> Result<AlignedWindowHandle> {
        let start_ns = checked_window_start_ns(&prepared.plan, window_index)?;
        let end_ns = checked_i128_to_i64(
            i128::from(start_ns)
                .checked_add(i128::from(prepared.plan.duration_ns))
                .ok_or_else(|| RuntimeError::Alignment("aligned end_ns overflowed".into()))?,
            "aligned end_ns",
        )?;
        let mut windows = Vec::with_capacity(prepared.records.len());

        for prepared_record in &prepared.records {
            let start_frame = prepared_record
                .plan
                .start_frame_for_window(window_index)?;
            let descriptor = WindowDescriptor {
                record: Arc::clone(&prepared_record.record),
                start_frame,
                length_frames: prepared_record.plan.frames_per_window,
                frame_elements: prepared_record.frame_elements,
            };
            let window = self.open_window(descriptor)?;
            if window.verified_dataset_content_sha256()
                != Some(prepared.plan.dataset_content_sha256.as_str())
            {
                return Err(RuntimeError::Alignment(format!(
                    "record {:?} opened without the plan's verified dataset content identity",
                    prepared_record.plan.record_id
                )));
            }
            if window.verified_source_sha256()
                != Some(prepared_record.plan.source_sha256.as_str())
            {
                return Err(RuntimeError::Alignment(format!(
                    "record {:?} opened without the plan's verified source identity",
                    prepared_record.plan.record_id
                )));
            }
            windows.push(window);
        }

        Ok(AlignedWindowHandle {
            plan_sha256: prepared.plan_sha256.clone(),
            dataset_content_sha256: prepared.plan.dataset_content_sha256.clone(),
            manifest_sha256: prepared.plan.manifest_sha256.clone(),
            sync_group: prepared.plan.sync_group.clone(),
            window_index,
            start_ns,
            end_ns,
            windows,
        })
    }
}

impl AlignedWindowHandle {
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }

    pub fn dataset_content_sha256(&self) -> &str {
        &self.dataset_content_sha256
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn sync_group(&self) -> &str {
        &self.sync_group
    }

    pub const fn window_index(&self) -> usize {
        self.window_index
    }

    pub const fn start_ns(&self) -> i64 {
        self.start_ns
    }

    pub const fn end_ns(&self) -> i64 {
        self.end_ns
    }

    pub fn windows(&self) -> &[WindowHandle] {
        &self.windows
    }

    pub fn window(&self, modality: &str) -> Option<&WindowHandle> {
        self.windows
            .iter()
            .find(|window| window.modality() == modality)
    }
}

impl Iterator for AlignedWindowStream {
    type Item = Result<AlignedWindowHandle>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.receiver.recv() {
            Ok(AlignedStreamMessage::Window(window)) => Some(window),
            Ok(AlignedStreamMessage::Finished) => {
                self.finished = true;
                None
            }
            Err(_) => {
                self.finished = true;
                Some(Err(RuntimeError::WorkerTerminated))
            }
        }
    }
}

fn prepare_alignment_plan(dataset: &Dataset, plan: &ExactAlignmentPlan) -> Result<PreparedAlignmentPlan> {
    validate_plan_header(dataset, plan)?;

    let freshly_verified = dataset.verify_content_fresh()?.ok_or_else(|| {
        RuntimeError::Alignment(
            "aligned execution requires a complete freshly verified dataset content identity"
                .into(),
        )
    })?;
    if freshly_verified != plan.dataset_content_sha256 {
        return Err(RuntimeError::Alignment(format!(
            "alignment plan dataset content {} does not match freshly verified dataset content {}",
            plan.dataset_content_sha256, freshly_verified
        )));
    }

    let mut seen_records = HashSet::with_capacity(plan.entries.len());
    let mut seen_modalities = HashSet::with_capacity(plan.entries.len());
    let mut records = Vec::with_capacity(plan.entries.len());
    let mut overlap_start = i128::MIN;
    let mut overlap_end = i128::MAX;

    for (index, entry) in plan.entries.iter().enumerate() {
        if !seen_records.insert(entry.record_id.as_str()) {
            return Err(RuntimeError::Alignment(format!(
                "alignment plan repeats record {:?}",
                entry.record_id
            )));
        }
        if !seen_modalities.insert(entry.modality.as_str()) {
            return Err(RuntimeError::Alignment(format!(
                "alignment plan repeats modality {:?}",
                entry.modality
            )));
        }
        if index > 0 {
            let previous = &plan.entries[index - 1];
            if (previous.modality.as_str(), previous.record_id.as_str())
                >= (entry.modality.as_str(), entry.record_id.as_str())
            {
                return Err(RuntimeError::Alignment(
                    "alignment plan entries are not in canonical modality/record order".into(),
                ));
            }
        }

        let record = dataset
            .manifest()
            .records
            .iter()
            .find(|candidate| candidate.id == entry.record_id)
            .ok_or_else(|| {
                RuntimeError::Alignment(format!(
                    "alignment plan record {:?} is absent from the current manifest",
                    entry.record_id
                ))
            })?;
        validate_entry(plan, entry, record)?;

        let clock = record.clock.as_ref().expect("validated by validate_entry");
        let record_start = i128::from(clock.start_ns);
        let record_frames = i128::try_from(record.shape[0]).map_err(|_| {
            RuntimeError::Alignment(format!(
                "record {:?} frame count cannot be represented for execution validation",
                record.id
            ))
        })?;
        let record_span = record_frames
            .checked_mul(i128::from(clock.period_ns))
            .ok_or_else(|| {
                RuntimeError::Alignment(format!(
                    "record {:?} clock span overflowed during execution validation",
                    record.id
                ))
            })?;
        let record_end = record_start.checked_add(record_span).ok_or_else(|| {
            RuntimeError::Alignment(format!(
                "record {:?} clock end overflowed during execution validation",
                record.id
            ))
        })?;
        overlap_start = overlap_start.max(record_start);
        overlap_end = overlap_end.min(record_end);

        records.push(PreparedRecord {
            plan: entry.clone(),
            record: Arc::new(record.clone()),
            frame_elements: record.frame_elements()?,
        });
    }

    if overlap_start > i128::from(plan.start_ns) {
        return Err(RuntimeError::Alignment(
            "alignment plan starts before the selected records' common overlap".into(),
        ));
    }
    if overlap_end != i128::from(plan.overlap_end_ns) {
        return Err(RuntimeError::Alignment(format!(
            "alignment plan overlap_end_ns={} does not match current selected-record overlap end {}",
            plan.overlap_end_ns, overlap_end
        )));
    }

    let latest_start = i128::from(plan.overlap_end_ns)
        .checked_sub(i128::from(plan.duration_ns))
        .ok_or_else(|| RuntimeError::Alignment("alignment duration underflowed overlap".into()))?;
    if i128::from(plan.start_ns) > latest_start {
        return Err(RuntimeError::Alignment(
            "alignment plan contains no legal execution window".into(),
        ));
    }
    let expected_count = usize::try_from(
        latest_start
            .checked_sub(i128::from(plan.start_ns))
            .ok_or_else(|| RuntimeError::Alignment("alignment count span underflowed".into()))?
            / i128::from(plan.stride_ns)
            + 1,
    )
    .map_err(|_| RuntimeError::Alignment("alignment window count overflowed usize".into()))?;
    if expected_count != plan.window_count {
        return Err(RuntimeError::Alignment(format!(
            "alignment plan window_count={} does not match direct overlap arithmetic {}",
            plan.window_count, expected_count
        )));
    }

    Ok(PreparedAlignmentPlan {
        plan: plan.clone(),
        plan_sha256: plan.sha256()?,
        records,
    })
}

fn validate_plan_header(dataset: &Dataset, plan: &ExactAlignmentPlan) -> Result<()> {
    if plan.schema_version != EXACT_ALIGNMENT_PLAN_SCHEMA_VERSION {
        return Err(RuntimeError::Alignment(format!(
            "unsupported alignment plan schema {}; expected {}",
            plan.schema_version, EXACT_ALIGNMENT_PLAN_SCHEMA_VERSION
        )));
    }
    if plan.policy != AlignmentPolicy::Exact {
        return Err(RuntimeError::Alignment(
            "aligned execution currently accepts exact plans only".into(),
        ));
    }
    if plan.dataset_id != dataset.manifest().dataset_id {
        return Err(RuntimeError::Alignment(format!(
            "alignment plan dataset_id {:?} does not match current dataset {:?}",
            plan.dataset_id,
            dataset.manifest().dataset_id
        )));
    }
    if plan.manifest_sha256 != dataset.manifest_sha256() {
        return Err(RuntimeError::Alignment(format!(
            "alignment plan manifest {} does not match current manifest {}",
            plan.manifest_sha256,
            dataset.manifest_sha256()
        )));
    }
    if dataset.declared_dataset_content_sha256()
        != Some(plan.dataset_content_sha256.as_str())
    {
        return Err(RuntimeError::Alignment(
            "alignment plan dataset content identity does not match the current manifest declaration"
                .into(),
        ));
    }
    if plan.sync_group.trim().is_empty() || plan.sync_group != plan.sync_group.trim() {
        return Err(RuntimeError::Alignment(
            "alignment plan sync_group is not canonical".into(),
        ));
    }
    if plan.duration_ns == 0 || plan.stride_ns == 0 || plan.window_count == 0 {
        return Err(RuntimeError::Alignment(
            "alignment plan duration, stride and window_count must be positive".into(),
        ));
    }
    if plan.entries.len() < 2 {
        return Err(RuntimeError::Alignment(
            "aligned execution requires at least two plan entries".into(),
        ));
    }
    if plan.start_ns >= plan.overlap_end_ns {
        return Err(RuntimeError::Alignment(
            "alignment plan has an empty temporal overlap".into(),
        ));
    }
    Ok(())
}

fn validate_entry(
    plan: &ExactAlignmentPlan,
    entry: &AlignedRecordPlan,
    record: &Record,
) -> Result<()> {
    if record.sync_group.as_deref() != Some(plan.sync_group.as_str()) {
        return Err(RuntimeError::Alignment(format!(
            "record {:?} no longer belongs to sync_group {:?}",
            record.id, plan.sync_group
        )));
    }
    if record.subject != entry.subject
        || record.modality != entry.modality
        || record.source_sha256.as_deref() != Some(entry.source_sha256.as_str())
        || record.offset_bytes != entry.offset_bytes
        || record.dtype != entry.dtype
        || record.shape != entry.shape
    {
        return Err(RuntimeError::Alignment(format!(
            "alignment entry {:?} no longer matches the current record interpretation/source identity",
            entry.record_id
        )));
    }
    let clock = record.clock.as_ref().ok_or_else(|| {
        RuntimeError::Alignment(format!(
            "alignment entry {:?} record no longer has a clock",
            entry.record_id
        ))
    })?;
    if clock.id != entry.clock_id
        || clock.start_ns != entry.clock_start_ns
        || clock.period_ns != entry.period_ns
    {
        return Err(RuntimeError::Alignment(format!(
            "alignment entry {:?} clock identity changed",
            entry.record_id
        )));
    }
    if entry.frames_per_window == 0 || entry.frame_stride == 0 {
        return Err(RuntimeError::Alignment(format!(
            "alignment entry {:?} has a zero frame extent/stride",
            entry.record_id
        )));
    }

    let period = i128::from(entry.period_ns);
    let entry_start = i128::from(entry.clock_start_ns)
        .checked_add(
            i128::try_from(entry.start_frame)
                .map_err(|_| RuntimeError::Alignment("start_frame overflowed i128".into()))?
                .checked_mul(period)
                .ok_or_else(|| RuntimeError::Alignment("start-frame clock product overflowed".into()))?,
        )
        .ok_or_else(|| RuntimeError::Alignment("entry start time overflowed".into()))?;
    if entry_start != i128::from(plan.start_ns) {
        return Err(RuntimeError::Alignment(format!(
            "alignment entry {:?} start-frame equation does not reproduce plan.start_ns",
            entry.record_id
        )));
    }
    let duration = i128::try_from(entry.frames_per_window)
        .map_err(|_| RuntimeError::Alignment("frames_per_window overflowed i128".into()))?
        .checked_mul(period)
        .ok_or_else(|| RuntimeError::Alignment("window-duration clock product overflowed".into()))?;
    if duration != i128::from(plan.duration_ns) {
        return Err(RuntimeError::Alignment(format!(
            "alignment entry {:?} frame count does not reproduce plan.duration_ns",
            entry.record_id
        )));
    }
    let stride = i128::try_from(entry.frame_stride)
        .map_err(|_| RuntimeError::Alignment("frame_stride overflowed i128".into()))?
        .checked_mul(period)
        .ok_or_else(|| RuntimeError::Alignment("window-stride clock product overflowed".into()))?;
    if stride != i128::from(plan.stride_ns) {
        return Err(RuntimeError::Alignment(format!(
            "alignment entry {:?} frame stride does not reproduce plan.stride_ns",
            entry.record_id
        )));
    }

    let final_start = entry
        .start_frame_for_window(plan.window_count - 1)?;
    let final_end = final_start
        .checked_add(entry.frames_per_window)
        .ok_or_else(|| RuntimeError::Alignment("final aligned frame extent overflowed usize".into()))?;
    if final_end > record.shape[0] {
        return Err(RuntimeError::Alignment(format!(
            "alignment entry {:?} final window extends beyond record frame extent",
            entry.record_id
        )));
    }
    Ok(())
}

fn checked_window_start_ns(plan: &ExactAlignmentPlan, window_index: usize) -> Result<i64> {
    let index = i128::try_from(window_index)
        .map_err(|_| RuntimeError::Alignment("window index overflowed i128".into()))?;
    let offset = index
        .checked_mul(i128::from(plan.stride_ns))
        .ok_or_else(|| RuntimeError::Alignment("aligned time stride overflowed".into()))?;
    checked_i128_to_i64(
        i128::from(plan.start_ns)
            .checked_add(offset)
            .ok_or_else(|| RuntimeError::Alignment("aligned start_ns overflowed".into()))?,
        "aligned start_ns",
    )
}

fn checked_i128_to_i64(value: i128, label: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        RuntimeError::Alignment(format!("{label} falls outside the signed 64-bit clock domain"))
    })
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use sha2::{Digest, Sha256};
    use tempfile::tempdir;

    use super::*;
    use crate::manifest::{ClockSpec, DType, DatasetManifest, MANIFEST_FILE, Record};
    use crate::sync::{ExactAlignmentSpec, plan_exact_alignment};

    fn build_fixture() -> (tempfile::TempDir, Arc<Dataset>, ExactAlignmentPlan) {
        let directory = tempdir().unwrap();
        let write_source = |name: &str, count: u32| -> String {
            let path = directory.path().join(name);
            let mut file = File::create(&path).unwrap();
            for value in 0..count {
                file.write_all(&(value as f32).to_le_bytes()).unwrap();
            }
            drop(file);
            format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
        };
        let fmri_sha = write_source("fmri.f32", 40);
        let behavior_sha = write_source("behavior.f32", 40);
        let manifest = DatasetManifest {
            schema_version: 1,
            dataset_id: "aligned-execution".into(),
            records: vec![
                Record {
                    id: "fmri".into(),
                    subject: "sub-01".into(),
                    modality: "fmri".into(),
                    sync_group: Some("sub-01/run-01".into()),
                    path: "fmri.f32".into(),
                    source_sha256: Some(fmri_sha),
                    offset_bytes: 0,
                    dtype: DType::Float32Le,
                    shape: vec![10, 4],
                    sampling_hz: Some(0.5),
                    clock: Some(ClockSpec {
                        id: "scanner".into(),
                        start_ns: 0,
                        period_ns: 2_000_000_000,
                    }),
                },
                Record {
                    id: "behavior".into(),
                    subject: "sub-01".into(),
                    modality: "behavior".into(),
                    sync_group: Some("sub-01/run-01".into()),
                    path: "behavior.f32".into(),
                    source_sha256: Some(behavior_sha),
                    offset_bytes: 0,
                    dtype: DType::Float32Le,
                    shape: vec![40, 1],
                    sampling_hz: Some(2.0),
                    clock: Some(ClockSpec {
                        id: "behavior-clock".into(),
                        start_ns: 0,
                        period_ns: 500_000_000,
                    }),
                },
            ],
        };
        std::fs::write(
            directory.path().join(MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let dataset = Dataset::open(directory.path()).unwrap();
        let plan = plan_exact_alignment(
            dataset.as_ref(),
            "sub-01/run-01",
            &["fmri".into(), "behavior".into()],
            ExactAlignmentSpec::new(4_000_000_000, 2_000_000_000).unwrap(),
        )
        .unwrap();
        (directory, dataset, plan)
    }

    #[test]
    fn executes_exact_plan_without_replanning() {
        let (_directory, dataset, plan) = build_fixture();
        let plan_sha = plan.sha256().unwrap();
        let mut stream = dataset.stream_aligned(&plan, 2).unwrap();
        let first = stream.next().unwrap().unwrap();
        assert_eq!(first.plan_sha256(), plan_sha);
        assert_eq!(first.window_index(), 0);
        assert_eq!(first.start_ns(), 0);
        assert_eq!(first.end_ns(), 4_000_000_000);
        assert_eq!(first.windows()[0].modality(), "behavior");
        assert_eq!(first.windows()[0].start_frame(), 0);
        assert_eq!(first.windows()[0].end_frame_exclusive(), 8);
        assert_eq!(first.windows()[1].modality(), "fmri");
        assert_eq!(first.windows()[1].start_frame(), 0);
        assert_eq!(first.windows()[1].end_frame_exclusive(), 2);

        let fourth = stream.nth(2).unwrap().unwrap();
        assert_eq!(fourth.window_index(), 3);
        assert_eq!(fourth.start_ns(), 6_000_000_000);
        assert_eq!(fourth.window("behavior").unwrap().start_frame(), 12);
        assert_eq!(fourth.window("behavior").unwrap().end_frame_exclusive(), 20);
        assert_eq!(fourth.window("fmri").unwrap().start_frame(), 3);
        assert_eq!(fourth.window("fmri").unwrap().end_frame_exclusive(), 5);
    }

    #[test]
    fn stale_exact_manifest_plan_rejects_even_when_content_identity_matches() {
        let (directory, dataset, plan) = build_fixture();
        let original_content = dataset.declared_dataset_content_sha256().unwrap().to_owned();
        let mut manifest = dataset.manifest().clone();
        manifest.records.reverse();
        std::fs::write(
            directory.path().join(MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let reopened = Dataset::open(directory.path()).unwrap();
        assert_eq!(
            reopened.declared_dataset_content_sha256(),
            Some(original_content.as_str())
        );
        assert!(reopened.stream_aligned(&plan, 1).is_err());
    }

    #[test]
    fn tampered_frame_equation_rejects_without_replanning() {
        let (_directory, dataset, mut plan) = build_fixture();
        plan.entries[0].frame_stride += 1;
        let error = dataset.stream_aligned(&plan, 1).unwrap_err();
        assert!(error.to_string().contains("frame stride"));
    }

    #[test]
    fn fresh_execution_verification_ignores_stale_mmap_verification_cache() {
        let (directory, dataset, plan) = build_fixture();
        let mut single = dataset
            .stream(
                crate::dataset::StreamSelector {
                    subjects: vec![],
                    modalities: vec!["fmri".into()],
                },
                crate::dataset::WindowSpec::new(2, 2).unwrap(),
                1,
            )
            .unwrap();
        let cached_window = single.next().unwrap().unwrap();
        assert!(cached_window.verified_source_sha256().is_some());

        let path = directory.path().join("fmri.f32");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 0xff;
        std::fs::write(&path, bytes).unwrap();

        let error = dataset.stream_aligned(&plan, 1).unwrap_err();
        assert!(matches!(error, RuntimeError::SourceHashMismatch { .. }));
        assert_eq!(cached_window.record_id(), "fmri");
    }

    #[test]
    fn final_window_is_in_bounds() {
        let (_directory, dataset, plan) = build_fixture();
        let mut stream = dataset.stream_aligned(&plan, 2).unwrap();
        let last = stream.nth(plan.window_count - 1).unwrap().unwrap();
        assert_eq!(last.window_index(), plan.window_count - 1);
        assert_eq!(last.window("fmri").unwrap().end_frame_exclusive(), 10);
        assert_eq!(last.window("behavior").unwrap().end_frame_exclusive(), 40);
    }

    #[test]
    fn zero_prefetch_rejects() {
        let (_directory, dataset, plan) = build_fixture();
        assert!(dataset.stream_aligned(&plan, 0).is_err());
    }
}