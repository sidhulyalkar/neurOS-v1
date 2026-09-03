use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::dataset::Dataset;
use crate::error::{Result, RuntimeError};

impl Dataset {
    /// Re-read every declared source from the current filesystem and verify its
    /// whole-file SHA-256 without trusting the mmap verification cache.
    ///
    /// This establishes freshness at the execution-start verification point. It
    /// does not claim that an external writer cannot mutate a file afterwards.
    pub(crate) fn verify_content_fresh(&self) -> Result<Option<String>> {
        let Some(expected_dataset_sha256) = self.declared_dataset_content_sha256.clone() else {
            return Ok(None);
        };

        let mut records: Vec<_> = self.records.iter().collect();
        records.sort_by(|left, right| left.id.cmp(&right.id));
        let mut verified_paths: HashMap<PathBuf, String> = HashMap::new();

        for record in records {
            let expected_source_sha256 = record.source_sha256.as_deref().ok_or_else(|| {
                RuntimeError::Validation(
                    "fresh dataset verification requires every record source hash".into(),
                )
            })?;
            let (path, _) = self.resolve_record_source(record)?;

            if let Some(previous_expected) = verified_paths.get(&path) {
                if previous_expected != expected_source_sha256 {
                    return Err(RuntimeError::Validation(format!(
                        "records sharing source {} declare conflicting SHA-256 values",
                        path.display()
                    )));
                }
                continue;
            }

            verify_file_sha256_fresh(&path, expected_source_sha256)?;
            verified_paths.insert(path, expected_source_sha256.to_owned());
        }

        let mut state = self.verified_dataset_content_sha256.lock().map_err(|_| {
            RuntimeError::Validation("dataset verification state lock was poisoned".into())
        })?;
        *state = Some(expected_dataset_sha256.clone());
        Ok(Some(expected_dataset_sha256))
    }
}

fn verify_file_sha256_fresh(path: &Path, expected: &str) -> Result<()> {
    let mut file = File::open(path).map_err(|source| RuntimeError::io(path, source))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];

    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| RuntimeError::io(path, source))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }

    let actual = format!("{:x}", digest.finalize());
    if actual != expected {
        return Err(RuntimeError::SourceHashMismatch {
            path: path.to_path_buf(),
            expected: expected.to_owned(),
            actual,
        });
    }
    Ok(())
}
