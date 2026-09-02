use std::collections::HashSet;
use std::path::{Component, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, RuntimeError};

pub const MANIFEST_FILE: &str = "neuros.dataset.json";
pub const MANIFEST_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub records: Vec<Record>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub subject: String,
    pub modality: String,
    pub path: PathBuf,
    #[serde(default)]
    pub offset_bytes: u64,
    pub dtype: DType,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub sampling_hz: Option<f64>,
    #[serde(default)]
    pub clock: Option<ClockSpec>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DType {
    Float32Le,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClockSpec {
    pub id: String,
    pub start_ns: i64,
    pub period_ns: u64,
}

impl DatasetManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(RuntimeError::Validation(format!(
                "unsupported manifest schema {}; expected {}",
                self.schema_version, MANIFEST_SCHEMA_VERSION
            )));
        }
        if self.dataset_id.trim().is_empty() {
            return Err(RuntimeError::Validation(
                "dataset_id must not be empty".into(),
            ));
        }
        if self.records.is_empty() {
            return Err(RuntimeError::Validation(
                "manifest must contain at least one record".into(),
            ));
        }

        let mut ids = HashSet::with_capacity(self.records.len());
        for record in &self.records {
            record.validate()?;
            if !ids.insert(record.id.as_str()) {
                return Err(RuntimeError::Validation(format!(
                    "duplicate record id {:?}",
                    record.id
                )));
            }
        }
        Ok(())
    }
}

impl Record {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty()
            || self.subject.trim().is_empty()
            || self.modality.trim().is_empty()
        {
            return Err(RuntimeError::Validation(format!(
                "record {:?} requires non-empty id, subject, and modality",
                self.id
            )));
        }
        if self.path.as_os_str().is_empty() || !is_safe_relative_path(&self.path) {
            return Err(RuntimeError::Validation(format!(
                "record {:?} path must be a safe relative path: {:?}",
                self.id, self.path
            )));
        }
        if self.shape.is_empty() || self.shape.iter().any(|&dimension| dimension == 0) {
            return Err(RuntimeError::Validation(format!(
                "record {:?} shape dimensions must all be positive",
                self.id
            )));
        }
        let alignment = self.dtype.size_bytes() as u64;
        if self.offset_bytes % alignment != 0 {
            return Err(RuntimeError::Validation(format!(
                "record {:?} offset_bytes={} is not aligned to {} bytes for {:?}",
                self.id, self.offset_bytes, alignment, self.dtype
            )));
        }
        if let Some(rate) = self.sampling_hz {
            if !rate.is_finite() || rate <= 0.0 {
                return Err(RuntimeError::Validation(format!(
                    "record {:?} sampling_hz must be finite and positive",
                    self.id
                )));
            }
        }
        if let Some(clock) = &self.clock {
            if clock.id.trim().is_empty() || clock.period_ns == 0 {
                return Err(RuntimeError::Validation(format!(
                    "record {:?} clock requires a non-empty id and positive period_ns",
                    self.id
                )));
            }
        }
        self.required_end_byte()?;
        Ok(())
    }

    pub fn element_count(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |acc, &dimension| {
            acc.checked_mul(dimension).ok_or_else(|| {
                RuntimeError::Validation(format!("record {:?} shape overflows usize", self.id))
            })
        })
    }

    pub fn frame_elements(&self) -> Result<usize> {
        self.shape
            .iter()
            .skip(1)
            .try_fold(1usize, |acc, &dimension| {
                acc.checked_mul(dimension).ok_or_else(|| {
                    RuntimeError::Validation(format!(
                        "record {:?} frame shape overflows usize",
                        self.id
                    ))
                })
            })
    }

    pub fn required_end_byte(&self) -> Result<u64> {
        let payload = self
            .element_count()?
            .checked_mul(self.dtype.size_bytes())
            .ok_or_else(|| {
                RuntimeError::Validation(format!("record {:?} byte size overflows usize", self.id))
            })?;
        self.offset_bytes
            .checked_add(payload as u64)
            .ok_or_else(|| {
                RuntimeError::Validation(format!("record {:?} byte extent overflows u64", self.id))
            })
    }
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Float32Le => 4,
        }
    }
}

fn is_safe_relative_path(path: &PathBuf) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> Record {
        Record {
            id: "fmri-01".into(),
            subject: "sub-01".into(),
            modality: "fmri".into(),
            path: "sub-01/fmri.f32".into(),
            offset_bytes: 0,
            dtype: DType::Float32Le,
            shape: vec![10, 4],
            sampling_hz: Some(0.5),
            clock: None,
        }
    }

    #[test]
    fn rejects_path_escape() {
        let mut candidate = record();
        candidate.path = "../outside.f32".into();
        assert!(candidate.validate().is_err());
    }

    #[test]
    fn rejects_misaligned_offset() {
        let mut candidate = record();
        candidate.offset_bytes = 2;
        assert!(candidate.validate().is_err());
    }

    #[test]
    fn computes_required_extent() {
        let mut candidate = record();
        candidate.offset_bytes = 32;
        assert_eq!(candidate.required_end_byte().unwrap(), 32 + 10 * 4 * 4);
    }
}
