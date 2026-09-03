use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender, bounded};
use tracing::debug;

use crate::dataset::{Dataset, WindowHandle};
use crate::error::{Result, RuntimeError};
use crate::manifest::Record;
use crate::sync::{AlignmentPolicy, EXACT_ALIGNMENT_PLAN_SCHEMA_VERSION, ExactAlignmentPlan};

#[derive(Clone, Debug)]
struct AlignedExecutionEntry {
    record: Arc<Record>,
    start_frame: usize,
    length_frames: usize,
    frame_stride: usize,
}

#[derive(Clone, Debug)]
struct AlignedExecutionSpec {
    plan_sha256: String,
    dataset_content_sha256: String,
    manifest_sha256: String,
    start_ns: i64,
    duration_ns: u64,
    stride_ns: u64,
    window_count: usize,
    entries: Vec<AlignedExecutionEntry>,
}

pub struct AlignedBatch {
    plan_sha256: String,
    dataset_content_sha256: String,
    manifest_sha256: String,
    window_index: usize,
    start_ns: i64,
    end_ns: i64,
    windows: Vec<WindowHandle>,
}

enum AlignedStreamMessage {
    Batch(Result<AlignedBatch>),
    Finished,
}

pub struct AlignedBatchStream {
    receiver: Receiver<AlignedStreamMessage>,
    finished: bool,
}

impl Dataset {
    /// Execute a previously qualified exact alignment plan without re-planning clocks.
    ///
    /// The complete declared dataset identity is freshly verified before the worker
    /// starts. Each emitted batch contains stable-ordered zero-copy child windows
    /// backed by the same mmap ownership path as the v0 single-modality runtime.
    pub fn stream_aligned(
        self: &Arc<Self>,
        plan: &ExactAlignmentPlan,
        prefetch: usize,
    ) -> Result<AlignedBatchStream> {
        if prefetch == 0 {
            return Err(RuntimeError::InvalidWindow(
                "aligned prefetch must be at least one".into(),
            ));
        }
        let execution = self.validate_aligned_plan(plan)?;
        let (sender, receiver) = bounded(prefetch);
        let dataset = Arc::clone(self);

        thread::Builder::new()
            .name("neuros-aligned-prefetch".into())
            .spawn(move || dataset.run_aligned_stream_worker(execution, sender))
            .map_err(|source| RuntimeError::io("<aligned-prefetch-thread>", source))?;

        Ok(AlignedBatchStream {
            receiver,
            finished: false,
        })
    }

    fn validate_aligned_plan(&self, plan: &ExactAlignmentPlan) -> Result<AlignedExecutionSpec> {
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
        if plan.dataset_id != self.manifest().dataset_id {
            return Err(RuntimeError::Alignment(format!(
                "alignment plan dataset_id {:?} does not match opened dataset {:?}",
                plan.dataset_id,
                self.manifest().dataset_id
            )));
        }
        if plan.manifest_sha256 != self.manifest_sha256() {
            return Err(RuntimeError::Alignment(
                "alignment plan manifest_sha256 does not match the opened manifest".into(),
            ));
        }
        if plan.sync_group.is_empty() || plan.sync_group != plan.sync_group.trim() {
            return Err(RuntimeError::Alignment(
                "alignment plan sync_group is not a canonical acquisition identifier".into(),
            ));
        }
        if plan.duration_ns == 0 || plan.stride_ns == 0 || plan.window_count == 0 {
            return Err(RuntimeError::Alignment(
                "alignment plan requires positive duration, stride, and window_count".into(),
            ));
        }
        if plan.entries.len() < 2 {
            return Err(RuntimeError::Alignment(
                "aligned execution requires at least two plan entries".into(),
            ));
        }

        let declared_dataset_content_sha256 =
            self.declared_dataset_content_sha256().ok_or_else(|| {
                RuntimeError::Alignment(
                    "aligned execution requires a complete declared dataset content identity"
                        .into(),
                )
            })?;
        if plan.dataset_content_sha256 != declared_dataset_content_sha256 {
            return Err(RuntimeError::Alignment(
                "alignment plan dataset_content_sha256 does not match the opened dataset".into(),
            ));
        }

        let mut seen_records = HashSet::with_capacity(plan.entries.len());
        let mut seen_modalities = HashSet::with_capacity(plan.entries.len());
        let mut previous_key: Option<(&str, &str)> = None;
        let mut overlap_start = i128::MIN;
        let mut overlap_end = i128::MAX;
        let mut common_period = 1i128;
        let start_ns = i128::from(plan.start_ns);
        let duration_ns = i128::from(plan.duration_ns);
        let stride_ns = i128::from(plan.stride_ns);
        let mut execution_entries = Vec::with_capacity(plan.entries.len());

        for entry in &plan.entries {
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
            let key = (entry.modality.as_str(), entry.record_id.as_str());
            if previous_key.is_some_and(|previous| previous >= key) {
                return Err(RuntimeError::Alignment(
                    "alignment plan entries are not in canonical modality/record order".into(),
                ));
            }
            previous_key = Some(key);

            let record = self
                .records()
                .iter()
                .find(|record| record.id == entry.record_id)
                .cloned()
                .ok_or_else(|| {
                    RuntimeError::Alignment(format!(
                        "alignment plan references missing record {:?}",
                        entry.record_id
                    ))
                })?;
            if record.subject != entry.subject
                || record.modality != entry.modality
                || record.sync_group.as_deref() != Some(plan.sync_group.as_str())
                || record.source_sha256.as_deref() != Some(entry.source_sha256.as_str())
                || record.offset_bytes != entry.offset_bytes
                || record.dtype != entry.dtype
                || record.shape != entry.shape
            {
                return Err(RuntimeError::Alignment(format!(
                    "alignment plan record descriptor no longer matches manifest record {:?}",
                    entry.record_id
                )));
            }
            let clock = record.clock.as_ref().ok_or_else(|| {
                RuntimeError::Alignment(format!(
                    "alignment plan record {:?} no longer has a clock",
                    entry.record_id
                ))
            })?;
            if clock.id != entry.clock_id
                || clock.start_ns != entry.clock_start_ns
                || clock.period_ns != entry.period_ns
            {
                return Err(RuntimeError::Alignment(format!(
                    "alignment plan clock descriptor no longer matches record {:?}",
                    entry.record_id
                )));
            }

            let period_ns = i128::from(clock.period_ns);
            overlap_start = overlap_start.max(i128::from(clock.start_ns));
            common_period = checked_lcm(common_period, period_ns)?;
            let frame_offset_ns = start_ns
                .checked_sub(i128::from(clock.start_ns))
                .ok_or_else(|| {
                    RuntimeError::Alignment(
                        "alignment plan frame offset overflowed clock arithmetic".into(),
                    )
                })?;
            if frame_offset_ns < 0
                || frame_offset_ns % period_ns != 0
                || duration_ns % period_ns != 0
                || stride_ns % period_ns != 0
            {
                return Err(RuntimeError::Alignment(format!(
                    "alignment plan arithmetic is not exact on record {:?}",
                    entry.record_id
                )));
            }
            let expected_start_frame =
                usize::try_from(frame_offset_ns / period_ns).map_err(|_| {
                    RuntimeError::Alignment("alignment plan start frame overflowed usize".into())
                })?;
            let expected_length = usize::try_from(duration_ns / period_ns).map_err(|_| {
                RuntimeError::Alignment("alignment plan frames-per-window overflowed usize".into())
            })?;
            let expected_stride = usize::try_from(stride_ns / period_ns).map_err(|_| {
                RuntimeError::Alignment("alignment plan frame stride overflowed usize".into())
            })?;
            if entry.start_frame != expected_start_frame
                || entry.frames_per_window != expected_length
                || entry.frame_stride != expected_stride
            {
                return Err(RuntimeError::Alignment(format!(
                    "alignment plan derived frame arithmetic is stale for record {:?}",
                    entry.record_id
                )));
            }

            let frames = i128::try_from(record.shape[0]).map_err(|_| {
                RuntimeError::Alignment(
                    "record frame count cannot be represented for aligned execution".into(),
                )
            })?;
            let record_end = i128::from(clock.start_ns)
                .checked_add(frames.checked_mul(period_ns).ok_or_else(|| {
                    RuntimeError::Alignment(
                        "record clock span overflowed aligned execution arithmetic".into(),
                    )
                })?)
                .ok_or_else(|| {
                    RuntimeError::Alignment(
                        "record clock end overflowed aligned execution arithmetic".into(),
                    )
                })?;
            overlap_end = overlap_end.min(record_end);

            let final_index = plan.window_count - 1;
            let final_start_frame = entry
                .start_frame
                .checked_add(final_index.checked_mul(entry.frame_stride).ok_or_else(|| {
                    RuntimeError::Alignment("final aligned frame offset overflowed usize".into())
                })?)
                .ok_or_else(|| {
                    RuntimeError::Alignment("final aligned start frame overflowed usize".into())
                })?;
            let final_end_frame = final_start_frame
                .checked_add(entry.frames_per_window)
                .ok_or_else(|| {
                    RuntimeError::Alignment("final aligned end frame overflowed usize".into())
                })?;
            if final_end_frame > record.shape[0] {
                return Err(RuntimeError::Alignment(format!(
                    "alignment plan final window exceeds record {:?}",
                    entry.record_id
                )));
            }

            execution_entries.push(AlignedExecutionEntry {
                record,
                start_frame: entry.start_frame,
                length_frames: entry.frames_per_window,
                frame_stride: entry.frame_stride,
            });
        }

        let previous_common_boundary = start_ns.checked_sub(common_period).ok_or_else(|| {
            RuntimeError::Alignment(
                "alignment plan previous common boundary overflowed clock arithmetic".into(),
            )
        })?;
        if start_ns < overlap_start || previous_common_boundary >= overlap_start {
            return Err(RuntimeError::Alignment(
                "alignment plan does not begin at the earliest exact common boundary".into(),
            ));
        }

        let expected_overlap_end = i64::try_from(overlap_end).map_err(|_| {
            RuntimeError::Alignment(
                "aligned overlap end falls outside signed 64-bit clock domain".into(),
            )
        })?;
        if plan.overlap_end_ns != expected_overlap_end {
            return Err(RuntimeError::Alignment(
                "alignment plan overlap_end_ns does not match selected records".into(),
            ));
        }
        let latest_start = overlap_end.checked_sub(duration_ns).ok_or_else(|| {
            RuntimeError::Alignment("aligned latest-start arithmetic overflowed".into())
        })?;
        if start_ns > latest_start {
            return Err(RuntimeError::Alignment(
                "alignment plan has no executable exact window".into(),
            ));
        }
        let expected_window_count = usize::try_from(
            latest_start.checked_sub(start_ns).ok_or_else(|| {
                RuntimeError::Alignment("aligned window-count arithmetic underflowed".into())
            })? / stride_ns
                + 1,
        )
        .map_err(|_| RuntimeError::Alignment("aligned window count overflowed usize".into()))?;
        if plan.window_count != expected_window_count {
            return Err(RuntimeError::Alignment(format!(
                "alignment plan window_count={} is not maximal exact count {expected_window_count}",
                plan.window_count
            )));
        }

        // Planning verified these bytes when the plan was created. Execution performs a
        // fresh complete re-hash so a source changed between planning and worker launch
        // cannot silently inherit the earlier verification cache.
        let verified_dataset_content_sha256 = self.verify_content_fresh()?.ok_or_else(|| {
            RuntimeError::Alignment(
                "aligned execution requires complete fresh source verification".into(),
            )
        })?;
        if verified_dataset_content_sha256 != plan.dataset_content_sha256 {
            return Err(RuntimeError::Alignment(
                "fresh dataset verification does not match alignment plan content identity".into(),
            ));
        }

        Ok(AlignedExecutionSpec {
            plan_sha256: plan.sha256()?,
            dataset_content_sha256: verified_dataset_content_sha256,
            manifest_sha256: self.manifest_sha256().to_owned(),
            start_ns: plan.start_ns,
            duration_ns: plan.duration_ns,
            stride_ns: plan.stride_ns,
            window_count: plan.window_count,
            entries: execution_entries,
        })
    }

    fn run_aligned_stream_worker(
        self: Arc<Self>,
        execution: AlignedExecutionSpec,
        sender: Sender<AlignedStreamMessage>,
    ) {
        for window_index in 0..execution.window_count {
            let start_ns =
                match aligned_window_time(execution.start_ns, execution.stride_ns, window_index) {
                    Ok(value) => value,
                    Err(error) => {
                        send_terminal_error(&sender, error);
                        return;
                    }
                };
            let end_ns = match i128::from(start_ns)
                .checked_add(i128::from(execution.duration_ns))
                .and_then(|value| i64::try_from(value).ok())
            {
                Some(value) => value,
                None => {
                    send_terminal_error(
                        &sender,
                        RuntimeError::Alignment(
                            "aligned batch end_ns overflowed signed clock domain".into(),
                        ),
                    );
                    return;
                }
            };

            let mut windows = Vec::with_capacity(execution.entries.len());
            for entry in &execution.entries {
                let start_frame = match window_index
                    .checked_mul(entry.frame_stride)
                    .and_then(|offset| entry.start_frame.checked_add(offset))
                {
                    Some(value) => value,
                    None => {
                        send_terminal_error(
                            &sender,
                            RuntimeError::Alignment(
                                "aligned batch frame index overflowed usize".into(),
                            ),
                        );
                        return;
                    }
                };
                match self.open_record_window(
                    Arc::clone(&entry.record),
                    start_frame,
                    entry.length_frames,
                ) {
                    Ok(window) => windows.push(window),
                    Err(error) => {
                        send_terminal_error(&sender, error);
                        return;
                    }
                }
            }

            let batch = AlignedBatch {
                plan_sha256: execution.plan_sha256.clone(),
                dataset_content_sha256: execution.dataset_content_sha256.clone(),
                manifest_sha256: execution.manifest_sha256.clone(),
                window_index,
                start_ns,
                end_ns,
                windows,
            };
            if sender.send(AlignedStreamMessage::Batch(Ok(batch))).is_err() {
                debug!("aligned batch consumer dropped; cancelling prefetch worker");
                return;
            }
        }
        let _ = sender.send(AlignedStreamMessage::Finished);
    }
}

impl AlignedBatch {
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }

    pub fn dataset_content_sha256(&self) -> &str {
        &self.dataset_content_sha256
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
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
}

impl Iterator for AlignedBatchStream {
    type Item = Result<AlignedBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        match self.receiver.recv() {
            Ok(AlignedStreamMessage::Batch(batch)) => Some(batch),
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

fn aligned_window_time(start_ns: i64, stride_ns: u64, index: usize) -> Result<i64> {
    let offset = i128::try_from(index)
        .ok()
        .and_then(|value| value.checked_mul(i128::from(stride_ns)))
        .ok_or_else(|| RuntimeError::Alignment("aligned batch time offset overflowed".into()))?;
    i128::from(start_ns)
        .checked_add(offset)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or_else(|| {
            RuntimeError::Alignment("aligned batch start_ns overflowed signed clock domain".into())
        })
}

fn checked_lcm(left: i128, right: i128) -> Result<i128> {
    let divisor = gcd_positive(left, right);
    left.checked_div(divisor)
        .and_then(|value| value.checked_mul(right))
        .ok_or_else(|| {
            RuntimeError::Alignment(
                "alignment plan common clock period overflowed integer arithmetic".into(),
            )
        })
}

fn gcd_positive(mut left: i128, mut right: i128) -> i128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.abs()
}

fn send_terminal_error(sender: &Sender<AlignedStreamMessage>, error: RuntimeError) {
    if sender.send(AlignedStreamMessage::Batch(Err(error))).is_ok() {
        let _ = sender.send(AlignedStreamMessage::Finished);
    }
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

    fn aligned_fixture() -> (tempfile::TempDir, Arc<Dataset>, ExactAlignmentPlan) {
        let directory = tempdir().unwrap();
        let write_values = |name: &str, values: &[f32]| -> String {
            let path = directory.path().join(name);
            let mut file = File::create(&path).unwrap();
            for value in values {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            drop(file);
            format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
        };
        let fmri_values: Vec<f32> = (0..40).map(|value| value as f32).collect();
        let behavior_values: Vec<f32> = (0..40).map(|value| 100.0 + value as f32).collect();
        let fmri_sha = write_values("fmri.f32", &fmri_values);
        let behavior_sha = write_values("behavior.f32", &behavior_values);
        let manifest = DatasetManifest {
            schema_version: 1,
            dataset_id: "aligned-execution-test".into(),
            records: vec![
                Record {
                    id: "fmri-run-01".into(),
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
                        id: "scanner-clock".into(),
                        start_ns: 0,
                        period_ns: 2_000_000_000,
                    }),
                },
                Record {
                    id: "behavior-run-01".into(),
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
    fn aligned_stream_executes_exact_plan_with_zero_copy_children() {
        let (_directory, dataset, plan) = aligned_fixture();
        let expected_plan_sha = plan.sha256().unwrap();
        let mut batches = dataset
            .stream_aligned(&plan, 2)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 9);

        let first = &batches[0];
        assert_eq!(first.plan_sha256(), expected_plan_sha);
        assert_eq!(first.dataset_content_sha256(), plan.dataset_content_sha256);
        assert_eq!(first.manifest_sha256(), plan.manifest_sha256);
        assert_eq!(first.window_index(), 0);
        assert_eq!(first.start_ns(), 0);
        assert_eq!(first.end_ns(), 4_000_000_000);
        assert_eq!(first.windows()[0].modality(), "behavior");
        assert_eq!(first.windows()[1].modality(), "fmri");
        assert_eq!(
            first.windows()[0].arrow_values().unwrap().values().as_ref(),
            &[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        );
        assert_eq!(
            first.windows()[1].arrow_values().unwrap().values().as_ref(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );

        let middle = &batches[3];
        assert_eq!(middle.start_ns(), 6_000_000_000);
        assert_eq!(middle.end_ns(), 10_000_000_000);
        assert_eq!(middle.windows()[0].start_frame(), 12);
        assert_eq!(middle.windows()[0].end_frame_exclusive(), 20);
        assert_eq!(middle.windows()[1].start_frame(), 3);
        assert_eq!(middle.windows()[1].end_frame_exclusive(), 5);
        assert_eq!(
            middle.windows()[0]
                .arrow_values()
                .unwrap()
                .values()
                .as_ref(),
            &[112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0]
        );
        assert_eq!(
            middle.windows()[1]
                .arrow_values()
                .unwrap()
                .values()
                .as_ref(),
            &[12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        );

        let final_batch = batches.pop().unwrap();
        assert_eq!(final_batch.window_index(), 8);
        assert_eq!(final_batch.start_ns(), 16_000_000_000);
        assert_eq!(final_batch.end_ns(), 20_000_000_000);
        assert_eq!(
            final_batch.windows()[0]
                .arrow_values()
                .unwrap()
                .values()
                .as_ref(),
            &[132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0]
        );
        assert_eq!(
            final_batch.windows()[1]
                .arrow_values()
                .unwrap()
                .values()
                .as_ref(),
            &[32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]
        );
        assert!(final_batch.windows().iter().all(|window| {
            window.source_verification_state()
                == crate::dataset::SourceVerificationState::VerifiedAtOpen
        }));
    }

    #[test]
    fn aligned_stream_rejects_tampered_plan_before_worker_launch() {
        let (_directory, dataset, plan) = aligned_fixture();
        let mut tampered = plan.clone();
        tampered.entries[0].start_frame += 1;
        let error = dataset.stream_aligned(&tampered, 1).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("derived frame arithmetic is stale")
        );

        let mut truncated = plan.clone();
        truncated.window_count -= 1;
        let error = dataset.stream_aligned(&truncated, 1).err().unwrap();
        assert!(error.to_string().contains("not maximal exact count"));

        let mut delayed = plan.clone();
        delayed.start_ns += delayed.stride_ns as i64;
        delayed.window_count -= 1;
        for entry in &mut delayed.entries {
            entry.start_frame += entry.frame_stride;
        }
        let error = dataset.stream_aligned(&delayed, 1).err().unwrap();
        assert!(error.to_string().contains("earliest exact common boundary"));
    }

    #[test]
    fn aligned_stream_fresh_verification_rejects_post_plan_source_mutation() {
        let (directory, dataset, plan) = aligned_fixture();
        let path = directory.path().join("fmri.f32");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 0xff;
        std::fs::write(&path, bytes).unwrap();

        let error = dataset.stream_aligned(&plan, 1).err().unwrap();
        assert!(matches!(error, RuntimeError::SourceHashMismatch { .. }));
    }

    #[test]
    fn aligned_stream_rejects_zero_prefetch() {
        let (_directory, dataset, plan) = aligned_fixture();
        let error = dataset.stream_aligned(&plan, 0).err().unwrap();
        assert!(matches!(error, RuntimeError::InvalidWindow(_)));
    }
}
