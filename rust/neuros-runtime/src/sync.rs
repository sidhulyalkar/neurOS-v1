use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Result, RuntimeError};
use crate::manifest::{DatasetManifest, Record};

pub const EXACT_ALIGNMENT_PLAN_SCHEMA_VERSION: u16 = 1;
const PLAN_HASH_DOMAIN: &[u8] = b"neuros.exact_alignment_plan.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlignmentPolicy {
    Exact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactAlignmentSpec {
    pub duration_ns: u64,
    pub stride_ns: u64,
}

impl ExactAlignmentSpec {
    pub fn new(duration_ns: u64, stride_ns: u64) -> Result<Self> {
        if duration_ns == 0 || stride_ns == 0 {
            return Err(RuntimeError::Alignment(
                "duration_ns and stride_ns must both be positive".into(),
            ));
        }
        Ok(Self {
            duration_ns,
            stride_ns,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AlignedRecordPlan {
    pub record_id: String,
    pub subject: String,
    pub modality: String,
    pub clock_id: String,
    pub clock_start_ns: i64,
    pub period_ns: u64,
    pub start_frame: usize,
    pub frames_per_window: usize,
    pub frame_stride: usize,
}

impl AlignedRecordPlan {
    pub fn start_frame_for_window(&self, window_index: usize) -> Result<usize> {
        window_index
            .checked_mul(self.frame_stride)
            .and_then(|offset| self.start_frame.checked_add(offset))
            .ok_or_else(|| RuntimeError::Alignment("aligned frame index overflowed usize".into()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExactAlignmentPlan {
    pub schema_version: u16,
    pub policy: AlignmentPolicy,
    pub dataset_id: String,
    pub manifest_sha256: String,
    pub sync_group: String,
    pub start_ns: i64,
    pub overlap_end_ns: i64,
    pub duration_ns: u64,
    pub stride_ns: u64,
    pub window_count: usize,
    pub entries: Vec<AlignedRecordPlan>,
}

impl ExactAlignmentPlan {
    pub fn sha256(&self) -> Result<String> {
        let payload = serde_json::to_vec(self).map_err(|error| {
            RuntimeError::Alignment(format!("failed to serialize alignment plan: {error}"))
        })?;
        let mut digest = Sha256::new();
        digest.update(PLAN_HASH_DOMAIN);
        digest.update(payload);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|error| {
            RuntimeError::Alignment(format!("failed to serialize alignment plan: {error}"))
        })
    }
}

pub fn plan_exact_alignment(
    manifest: &DatasetManifest,
    manifest_sha256: &str,
    sync_group: &str,
    modalities: &[String],
    spec: ExactAlignmentSpec,
) -> Result<ExactAlignmentPlan> {
    let sync_group = sync_group.trim();
    if sync_group.is_empty() {
        return Err(RuntimeError::Alignment(
            "sync_group must be non-empty".into(),
        ));
    }
    if modalities.len() < 2 {
        return Err(RuntimeError::Alignment(
            "exact multimodal planning requires at least two modalities".into(),
        ));
    }

    let mut requested = HashSet::with_capacity(modalities.len());
    for modality in modalities {
        let normalized = modality.trim();
        if normalized.is_empty() {
            return Err(RuntimeError::Alignment(
                "requested modalities must be non-empty".into(),
            ));
        }
        if !requested.insert(normalized) {
            return Err(RuntimeError::Alignment(format!(
                "requested modality {normalized:?} is duplicated"
            )));
        }
    }

    let mut by_modality: HashMap<&str, Vec<&Record>> = HashMap::new();
    for record in &manifest.records {
        if record.sync_group.as_deref() != Some(sync_group) {
            continue;
        }
        if requested.contains(record.modality.as_str()) {
            by_modality
                .entry(record.modality.as_str())
                .or_default()
                .push(record);
        }
    }

    let mut records = Vec::with_capacity(requested.len());
    for modality in requested {
        match by_modality.remove(modality) {
            None => {
                return Err(RuntimeError::Alignment(format!(
                    "sync_group {sync_group:?} has no record for requested modality {modality:?}"
                )));
            }
            Some(matches) if matches.len() != 1 => {
                return Err(RuntimeError::Alignment(format!(
                    "sync_group {sync_group:?} has {} records for modality {modality:?}; exact planning requires one unambiguous record",
                    matches.len()
                )));
            }
            Some(mut matches) => records.push(matches.remove(0)),
        }
    }
    records.sort_by(|left, right| {
        left.modality
            .cmp(&right.modality)
            .then_with(|| left.id.cmp(&right.id))
    });

    let subject = records[0].subject.as_str();
    if records.iter().any(|record| record.subject != subject) {
        return Err(RuntimeError::Alignment(format!(
            "sync_group {sync_group:?} crosses subjects; acquisition groups must be subject-local"
        )));
    }

    let mut overlap_start = i128::MIN;
    let mut overlap_end = i128::MAX;
    let mut congruence: Option<(i128, i128)> = None;

    for record in &records {
        let clock = record.clock.as_ref().ok_or_else(|| {
            RuntimeError::Alignment(format!(
                "record {:?} is missing clock metadata required for exact multimodal planning",
                record.id
            ))
        })?;
        let period = i128::from(clock.period_ns);
        let start = i128::from(clock.start_ns);
        let frames = i128::try_from(record.shape[0]).map_err(|_| {
            RuntimeError::Alignment(format!(
                "record {:?} frame count cannot be represented for clock arithmetic",
                record.id
            ))
        })?;
        let span = frames.checked_mul(period).ok_or_else(|| {
            RuntimeError::Alignment(format!("record {:?} clock span overflowed", record.id))
        })?;
        let end = start.checked_add(span).ok_or_else(|| {
            RuntimeError::Alignment(format!("record {:?} clock end overflowed", record.id))
        })?;
        overlap_start = overlap_start.max(start);
        overlap_end = overlap_end.min(end);
        congruence = Some(match congruence {
            None => (start.rem_euclid(period), period),
            Some((residue, modulus)) => combine_congruence(
                residue,
                modulus,
                start.rem_euclid(period),
                period,
            )?
            .ok_or_else(|| {
                RuntimeError::Alignment(format!(
                    "clock phase for record {:?} has no exact common boundary with the selected modalities",
                    record.id
                ))
            })?,
        });
    }

    if overlap_start >= overlap_end {
        return Err(RuntimeError::Alignment(format!(
            "sync_group {sync_group:?} has no common recording-time overlap"
        )));
    }

    let (residue, common_period) = congruence.expect("records is guaranteed non-empty");
    let duration = i128::from(spec.duration_ns);
    let stride = i128::from(spec.stride_ns);
    if duration % common_period != 0 {
        return Err(RuntimeError::Alignment(format!(
            "duration_ns={} does not end on every selected clock boundary; exact common period is {common_period} ns",
            spec.duration_ns
        )));
    }
    if stride % common_period != 0 {
        return Err(RuntimeError::Alignment(format!(
            "stride_ns={} does not preserve every selected clock boundary; exact common period is {common_period} ns",
            spec.stride_ns
        )));
    }

    let first_start = first_congruent_at_or_after(overlap_start, residue, common_period)?;
    let latest_start = overlap_end.checked_sub(duration).ok_or_else(|| {
        RuntimeError::Alignment("aligned window duration overflowed clock arithmetic".into())
    })?;
    if first_start > latest_start {
        return Err(RuntimeError::Alignment(format!(
            "sync_group {sync_group:?} has common overlap but no exact window of {} ns",
            spec.duration_ns
        )));
    }
    let count_span = latest_start.checked_sub(first_start).ok_or_else(|| {
        RuntimeError::Alignment("aligned window count span underflowed clock arithmetic".into())
    })?;
    let count_minus_one = count_span / stride;
    let window_count = usize::try_from(count_minus_one + 1)
        .map_err(|_| RuntimeError::Alignment("aligned window count overflowed usize".into()))?;

    let mut entries = Vec::with_capacity(records.len());
    for record in records {
        let clock = record
            .clock
            .as_ref()
            .expect("clock presence validated above");
        let period = i128::from(clock.period_ns);
        let frame_offset = first_start
            .checked_sub(i128::from(clock.start_ns))
            .ok_or_else(|| {
                RuntimeError::Alignment(format!(
                    "record {:?} frame offset overflowed clock arithmetic",
                    record.id
                ))
            })?;
        if frame_offset < 0 || frame_offset % period != 0 {
            return Err(RuntimeError::Alignment(format!(
                "internal exact-alignment invariant failed for record {:?}",
                record.id
            )));
        }
        let start_frame = usize::try_from(frame_offset / period).map_err(|_| {
            RuntimeError::Alignment(format!(
                "record {:?} start frame overflowed usize",
                record.id
            ))
        })?;
        let frames_per_window = usize::try_from(duration / period).map_err(|_| {
            RuntimeError::Alignment(format!(
                "record {:?} frames-per-window overflowed usize",
                record.id
            ))
        })?;
        let frame_stride = usize::try_from(stride / period).map_err(|_| {
            RuntimeError::Alignment(format!(
                "record {:?} frame stride overflowed usize",
                record.id
            ))
        })?;

        entries.push(AlignedRecordPlan {
            record_id: record.id.clone(),
            subject: record.subject.clone(),
            modality: record.modality.clone(),
            clock_id: clock.id.clone(),
            clock_start_ns: clock.start_ns,
            period_ns: clock.period_ns,
            start_frame,
            frames_per_window,
            frame_stride,
        });
    }

    Ok(ExactAlignmentPlan {
        schema_version: EXACT_ALIGNMENT_PLAN_SCHEMA_VERSION,
        policy: AlignmentPolicy::Exact,
        dataset_id: manifest.dataset_id.clone(),
        manifest_sha256: manifest_sha256.to_owned(),
        sync_group: sync_group.to_owned(),
        start_ns: i64::try_from(first_start).map_err(|_| {
            RuntimeError::Alignment(
                "aligned start_ns falls outside the signed 64-bit clock domain".into(),
            )
        })?,
        overlap_end_ns: i64::try_from(overlap_end).map_err(|_| {
            RuntimeError::Alignment(
                "common overlap end falls outside the signed 64-bit clock domain".into(),
            )
        })?,
        duration_ns: spec.duration_ns,
        stride_ns: spec.stride_ns,
        window_count,
        entries,
    })
}

fn combine_congruence(
    left_residue: i128,
    left_modulus: i128,
    right_residue: i128,
    right_modulus: i128,
) -> Result<Option<(i128, i128)>> {
    let gcd = gcd(left_modulus, right_modulus);
    let difference = right_residue - left_residue;
    if difference % gcd != 0 {
        return Ok(None);
    }

    let left_reduced = left_modulus / gcd;
    let right_reduced = right_modulus / gcd;
    let (_, inverse, _) = checked_extended_gcd(left_reduced, right_reduced)?;
    let step = (difference / gcd)
        .checked_mul(inverse)
        .ok_or_else(|| {
            RuntimeError::Alignment("clock congruence multiplication overflowed".into())
        })?
        .rem_euclid(right_reduced);
    let modulus = left_modulus.checked_mul(right_reduced).ok_or_else(|| {
        RuntimeError::Alignment("exact common clock period overflowed integer arithmetic".into())
    })?;
    let residue = left_residue
        .checked_add(left_modulus.checked_mul(step).ok_or_else(|| {
            RuntimeError::Alignment("clock congruence step overflowed integer arithmetic".into())
        })?)
        .ok_or_else(|| RuntimeError::Alignment("clock congruence residue overflowed".into()))?
        .rem_euclid(modulus);
    Ok(Some((residue, modulus)))
}

fn first_congruent_at_or_after(start: i128, residue: i128, modulus: i128) -> Result<i128> {
    let delta = residue.checked_sub(start).ok_or_else(|| {
        RuntimeError::Alignment("aligned boundary delta overflowed integer arithmetic".into())
    })?;
    start
        .checked_add(delta.rem_euclid(modulus))
        .ok_or_else(|| RuntimeError::Alignment("aligned boundary search overflowed".into()))
}

fn gcd(mut left: i128, mut right: i128) -> i128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.abs()
}

fn checked_extended_gcd(left: i128, right: i128) -> Result<(i128, i128, i128)> {
    if right == 0 {
        return Ok((left, 1, 0));
    }

    let (gcd, x1, y1) = checked_extended_gcd(right, left % right)?;
    let quotient_times_y1 = (left / right).checked_mul(y1).ok_or_else(|| {
        RuntimeError::Alignment("extended clock congruence multiplication overflowed".into())
    })?;
    let y = x1.checked_sub(quotient_times_y1).ok_or_else(|| {
        RuntimeError::Alignment("extended clock congruence subtraction overflowed".into())
    })?;
    Ok((gcd, y1, y))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::manifest::{ClockSpec, DType};

    fn record(id: &str, modality: &str, frames: usize, start_ns: i64, period_ns: u64) -> Record {
        Record {
            id: id.into(),
            subject: "sub-01".into(),
            modality: modality.into(),
            sync_group: Some("sub-01/run-01".into()),
            path: PathBuf::from(format!("{id}.f32")),
            offset_bytes: 0,
            dtype: DType::Float32Le,
            shape: vec![frames, 1],
            sampling_hz: None,
            clock: Some(ClockSpec {
                id: format!("clock-{modality}"),
                start_ns,
                period_ns,
            }),
        }
    }

    fn manifest(records: Vec<Record>) -> DatasetManifest {
        DatasetManifest {
            schema_version: 1,
            dataset_id: "aligned-test".into(),
            records,
        }
    }

    #[test]
    fn exact_plan_maps_modalities_without_materializing_windows() {
        let manifest = manifest(vec![
            record("fmri", "fmri", 10, 0, 2_000_000_000),
            record("behavior", "behavior", 40, 0, 500_000_000),
        ]);
        let plan = plan_exact_alignment(
            &manifest,
            "a".repeat(64).as_str(),
            "sub-01/run-01",
            &["fmri".into(), "behavior".into()],
            ExactAlignmentSpec::new(4_000_000_000, 2_000_000_000).unwrap(),
        )
        .unwrap();

        assert_eq!(plan.start_ns, 0);
        assert_eq!(plan.overlap_end_ns, 20_000_000_000);
        assert_eq!(plan.window_count, 9);
        assert_eq!(plan.entries[0].modality, "behavior");
        assert_eq!(plan.entries[0].frames_per_window, 8);
        assert_eq!(plan.entries[0].frame_stride, 4);
        assert_eq!(plan.entries[1].modality, "fmri");
        assert_eq!(plan.entries[1].frames_per_window, 2);
        assert_eq!(plan.entries[1].frame_stride, 1);
        assert_eq!(plan.entries[0].start_frame_for_window(3).unwrap(), 12);
    }

    #[test]
    fn incompatible_clock_phases_reject() {
        let manifest = manifest(vec![
            record("fmri", "fmri", 10, 250_000_000, 2_000_000_000),
            record("behavior", "behavior", 40, 0, 500_000_000),
        ]);
        assert!(
            plan_exact_alignment(
                &manifest,
                "b".repeat(64).as_str(),
                "sub-01/run-01",
                &["fmri".into(), "behavior".into()],
                ExactAlignmentSpec::new(4_000_000_000, 2_000_000_000).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn partial_overlap_starts_at_first_exact_common_boundary() {
        let manifest = manifest(vec![
            record("fmri", "fmri", 10, 2_000_000_000, 2_000_000_000),
            record("behavior", "behavior", 40, 0, 500_000_000),
        ]);
        let plan = plan_exact_alignment(
            &manifest,
            "c".repeat(64).as_str(),
            "sub-01/run-01",
            &["behavior".into(), "fmri".into()],
            ExactAlignmentSpec::new(4_000_000_000, 2_000_000_000).unwrap(),
        )
        .unwrap();
        assert_eq!(plan.start_ns, 2_000_000_000);
        assert_eq!(plan.overlap_end_ns, 20_000_000_000);
        assert_eq!(plan.window_count, 8);
    }

    #[test]
    fn duplicate_modality_in_group_is_ambiguous() {
        let manifest = manifest(vec![
            record("fmri-a", "fmri", 10, 0, 2_000_000_000),
            record("fmri-b", "fmri", 10, 0, 2_000_000_000),
            record("behavior", "behavior", 40, 0, 500_000_000),
        ]);
        assert!(
            plan_exact_alignment(
                &manifest,
                "d".repeat(64).as_str(),
                "sub-01/run-01",
                &["fmri".into(), "behavior".into()],
                ExactAlignmentSpec::new(4_000_000_000, 2_000_000_000).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn plan_order_and_hash_ignore_manifest_record_order() {
        let left = manifest(vec![
            record("fmri", "fmri", 10, 0, 2_000_000_000),
            record("behavior", "behavior", 40, 0, 500_000_000),
        ]);
        let right = manifest(vec![
            record("behavior", "behavior", 40, 0, 500_000_000),
            record("fmri", "fmri", 10, 0, 2_000_000_000),
        ]);
        let modalities = ["fmri".into(), "behavior".into()];
        let spec = ExactAlignmentSpec::new(4_000_000_000, 2_000_000_000).unwrap();
        let left_plan = plan_exact_alignment(
            &left,
            "e".repeat(64).as_str(),
            "sub-01/run-01",
            &modalities,
            spec,
        )
        .unwrap();
        let right_plan = plan_exact_alignment(
            &right,
            "e".repeat(64).as_str(),
            "sub-01/run-01",
            &modalities,
            spec,
        )
        .unwrap();
        assert_eq!(left_plan, right_plan);
        assert_eq!(left_plan.sha256().unwrap(), right_plan.sha256().unwrap());
    }

    #[test]
    fn duration_and_stride_must_preserve_all_clock_boundaries() {
        let manifest = manifest(vec![
            record("fmri", "fmri", 10, 0, 2_000_000_000),
            record("behavior", "behavior", 40, 0, 500_000_000),
        ]);
        assert!(
            plan_exact_alignment(
                &manifest,
                "f".repeat(64).as_str(),
                "sub-01/run-01",
                &["fmri".into(), "behavior".into()],
                ExactAlignmentSpec::new(3_000_000_000, 2_000_000_000).unwrap(),
            )
            .is_err()
        );
        assert!(
            plan_exact_alignment(
                &manifest,
                "f".repeat(64).as_str(),
                "sub-01/run-01",
                &["fmri".into(), "behavior".into()],
                ExactAlignmentSpec::new(4_000_000_000, 1_000_000_000).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn exact_common_period_overflow_rejects_explicitly() {
        let manifest = manifest(vec![
            record("a", "fmri", 2, 0, u64::MAX),
            record("b", "behavior", 2, 0, u64::MAX - 1),
        ]);
        let error = plan_exact_alignment(
            &manifest,
            "1".repeat(64).as_str(),
            "sub-01/run-01",
            &["fmri".into(), "behavior".into()],
            ExactAlignmentSpec::new(u64::MAX, u64::MAX).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("common clock period overflowed"));
    }
}
