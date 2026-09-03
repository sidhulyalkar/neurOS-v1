use sha2::{Digest, Sha256};

use crate::error::{Result, RuntimeError};
use crate::manifest::{DType, DatasetManifest};

pub const DATASET_CONTENT_DOMAIN: &str = "neuros.dataset_content.v1";

/// Return the canonical dataset byte/interpretation identity when every record
/// declares a source SHA-256.
///
/// The identity intentionally excludes path, sampling rate, synchronization group,
/// and clock metadata. Those remain bound by the manifest SHA-256. This content
/// identity instead binds the stable record ID, full-file source digest, record
/// byte offset, dtype, and shape using a domain-separated length-prefixed encoding.
pub fn declared_dataset_content_sha256(manifest: &DatasetManifest) -> Result<Option<String>> {
    let mut records: Vec<_> = manifest.records.iter().collect();
    records.sort_by(|left, right| left.id.cmp(&right.id));

    let mut hasher = Sha256::new();
    update_bytes(&mut hasher, DATASET_CONTENT_DOMAIN.as_bytes())?;
    for record in records {
        let Some(source_sha256) = record.source_sha256.as_deref() else {
            return Ok(None);
        };
        update_bytes(&mut hasher, record.id.as_bytes())?;
        update_bytes(&mut hasher, source_sha256.as_bytes())?;
        hasher.update(record.offset_bytes.to_le_bytes());
        update_bytes(&mut hasher, dtype_tag(record.dtype))?;
        let rank = u64::try_from(record.shape.len())
            .map_err(|_| RuntimeError::Validation("record rank does not fit in u64".into()))?;
        hasher.update(rank.to_le_bytes());
        for dimension in &record.shape {
            let dimension = u64::try_from(*dimension).map_err(|_| {
                RuntimeError::Validation("record shape dimension does not fit in u64".into())
            })?;
            hasher.update(dimension.to_le_bytes());
        }
    }
    Ok(Some(format!("{:x}", hasher.finalize())))
}

fn update_bytes(hasher: &mut Sha256, value: &[u8]) -> Result<()> {
    let length = u64::try_from(value.len()).map_err(|_| {
        RuntimeError::Validation("identity field length does not fit in u64".into())
    })?;
    hasher.update(length.to_le_bytes());
    hasher.update(value);
    Ok(())
}

const fn dtype_tag(dtype: DType) -> &'static [u8] {
    match dtype {
        DType::Float32Le => b"float32-le",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{DatasetManifest, Record};

    fn record(id: &str, path: &str, source_sha256: Option<&str>) -> Record {
        Record {
            id: id.into(),
            subject: "sub-01".into(),
            modality: "fmri".into(),
            sync_group: None,
            path: path.into(),
            source_sha256: source_sha256.map(str::to_owned),
            offset_bytes: 0,
            dtype: DType::Float32Le,
            shape: vec![6, 4],
            sampling_hz: Some(0.5),
            clock: None,
        }
    }

    fn manifest(records: Vec<Record>) -> DatasetManifest {
        DatasetManifest {
            schema_version: 1,
            dataset_id: "example".into(),
            records,
        }
    }

    #[test]
    fn identity_is_stable_under_record_order_and_path_rename() {
        let hash_a = "a".repeat(64);
        let hash_b = "b".repeat(64);
        let first = manifest(vec![
            record("r1", "first-name.f32", Some(&hash_a)),
            record("r2", "second-name.f32", Some(&hash_b)),
        ]);
        let renamed_reordered = manifest(vec![
            record("r2", "renamed-b.f32", Some(&hash_b)),
            record("r1", "renamed-a.f32", Some(&hash_a)),
        ]);
        assert_eq!(
            declared_dataset_content_sha256(&first).unwrap(),
            declared_dataset_content_sha256(&renamed_reordered).unwrap()
        );
    }

    #[test]
    fn sync_group_does_not_redefine_dataset_byte_identity() {
        let hash = "d".repeat(64);
        let first = manifest(vec![record("r1", "shared.f32", Some(&hash))]);
        let mut regrouped = first.clone();
        regrouped.records[0].sync_group = Some("sub-01/run-02".into());
        assert_eq!(
            declared_dataset_content_sha256(&first).unwrap(),
            declared_dataset_content_sha256(&regrouped).unwrap()
        );
    }

    #[test]
    fn changed_record_interpretation_changes_identity() {
        let hash = "c".repeat(64);
        let first = manifest(vec![record("r1", "shared.f32", Some(&hash))]);
        let mut changed = first.clone();
        changed.records[0].shape = vec![4, 6];
        assert_ne!(
            declared_dataset_content_sha256(&first).unwrap(),
            declared_dataset_content_sha256(&changed).unwrap()
        );
    }

    #[test]
    fn unhashed_record_makes_content_identity_unavailable() {
        let partial = manifest(vec![record("r1", "shared.f32", None)]);
        assert_eq!(declared_dataset_content_sha256(&partial).unwrap(), None);
    }
}
