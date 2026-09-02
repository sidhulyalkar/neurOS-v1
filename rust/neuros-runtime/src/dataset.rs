use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Weak};
use std::thread;

use arrow_array::Float32Array;
use arrow_buffer::{Buffer, ScalarBuffer};
use crossbeam_channel::{bounded, Receiver};
use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};
use tracing::{debug, instrument};

use crate::error::{Result, RuntimeError};
use crate::manifest::{DatasetManifest, Record, MANIFEST_FILE};

#[derive(Clone, Debug)]
pub struct StreamSelector {
    pub subjects: Vec<String>,
    pub modalities: Vec<String>,
}

impl Default for StreamSelector {
    fn default() -> Self {
        Self {
            subjects: Vec::new(),
            modalities: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WindowSpec {
    pub length: usize,
    pub stride: usize,
}

impl WindowSpec {
    pub fn new(length: usize, stride: usize) -> Result<Self> {
        if length == 0 || stride == 0 {
            return Err(RuntimeError::InvalidWindow(
                "length and stride must both be positive".into(),
            ));
        }
        Ok(Self { length, stride })
    }
}

#[derive(Clone, Debug)]
pub struct WindowDescriptor {
    record: Record,
    start_frame: usize,
    length_frames: usize,
    frame_elements: usize,
}

pub struct Dataset {
    root: PathBuf,
    manifest: DatasetManifest,
    manifest_sha256: String,
    mmap_cache: Mutex<HashMap<PathBuf, Weak<MappedRegion>>>,
}

#[derive(Debug)]
struct MappedRegion {
    mmap: Mmap,
}

#[derive(Clone)]
pub struct WindowHandle {
    region: Arc<MappedRegion>,
    record: Record,
    start_frame: usize,
    length_frames: usize,
    frame_elements: usize,
    source_size_bytes: u64,
    manifest_sha256: String,
}

pub struct WindowStream {
    receiver: Receiver<Result<WindowHandle>>,
}

impl Dataset {
    #[instrument(skip(root), fields(root = %root.as_ref().display()))]
    pub fn open(root: impl AsRef<Path>) -> Result<Arc<Self>> {
        if cfg!(target_endian = "big") {
            return Err(RuntimeError::Validation(
                "the v0 runtime supports little-endian hosts only".into(),
            ));
        }

        let root = root.as_ref().to_path_buf();
        let manifest_path = root.join(MANIFEST_FILE);
        let bytes = std::fs::read(&manifest_path)
            .map_err(|source| RuntimeError::io(&manifest_path, source))?;
        let manifest: DatasetManifest = serde_json::from_slice(&bytes)?;
        manifest.validate()?;
        let manifest_sha256 = format!("{:x}", Sha256::digest(&bytes));

        Ok(Arc::new(Self {
            root,
            manifest,
            manifest_sha256,
            mmap_cache: Mutex::new(HashMap::new()),
        }))
    }

    pub fn manifest(&self) -> &DatasetManifest {
        &self.manifest
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn plan_windows(
        &self,
        selector: &StreamSelector,
        spec: WindowSpec,
    ) -> Result<Vec<WindowDescriptor>> {
        let subjects: HashSet<&str> = selector.subjects.iter().map(String::as_str).collect();
        let modalities: HashSet<&str> = selector.modalities.iter().map(String::as_str).collect();
        let mut windows = Vec::new();

        for record in &self.manifest.records {
            if !subjects.is_empty() && !subjects.contains(record.subject.as_str()) {
                continue;
            }
            if !modalities.is_empty() && !modalities.contains(record.modality.as_str()) {
                continue;
            }

            let frames = record.shape[0];
            if frames < spec.length {
                continue;
            }
            let frame_elements = record.frame_elements()?;
            let final_start = frames - spec.length;
            let mut start = 0usize;
            loop {
                windows.push(WindowDescriptor {
                    record: record.clone(),
                    start_frame: start,
                    length_frames: spec.length,
                    frame_elements,
                });
                if start > final_start.saturating_sub(spec.stride) {
                    break;
                }
                start += spec.stride;
            }
        }
        Ok(windows)
    }

    pub fn stream(
        self: &Arc<Self>,
        selector: StreamSelector,
        spec: WindowSpec,
        prefetch: usize,
    ) -> Result<WindowStream> {
        if prefetch == 0 {
            return Err(RuntimeError::InvalidWindow(
                "prefetch must be at least one".into(),
            ));
        }
        let plan = self.plan_windows(&selector, spec)?;
        let (sender, receiver) = bounded(prefetch);
        let dataset = Arc::clone(self);

        thread::Builder::new()
            .name("neuros-data-prefetch".into())
            .spawn(move || {
                for descriptor in plan {
                    if sender.send(dataset.open_window(descriptor)).is_err() {
                        debug!("window consumer dropped; cancelling prefetch worker");
                        return;
                    }
                }
            })
            .map_err(|source| RuntimeError::io("<prefetch-thread>", source))?;

        Ok(WindowStream { receiver })
    }

    fn open_window(&self, descriptor: WindowDescriptor) -> Result<WindowHandle> {
        let path = self.root.join(&descriptor.record.path);
        let required_end = descriptor.record.required_end_byte()?;
        let metadata = std::fs::metadata(&path).map_err(|source| RuntimeError::io(&path, source))?;
        if metadata.len() < required_end {
            return Err(RuntimeError::SourceTooShort {
                path,
                actual: metadata.len(),
                required: required_end,
            });
        }
        let region = self.map_source(&path)?;
        Ok(WindowHandle {
            region,
            record: descriptor.record,
            start_frame: descriptor.start_frame,
            length_frames: descriptor.length_frames,
            frame_elements: descriptor.frame_elements,
            source_size_bytes: metadata.len(),
            manifest_sha256: self.manifest_sha256.clone(),
        })
    }

    fn map_source(&self, path: &Path) -> Result<Arc<MappedRegion>> {
        {
            let cache = self
                .mmap_cache
                .lock()
                .map_err(|_| RuntimeError::Validation("mmap cache lock was poisoned".into()))?;
            if let Some(region) = cache.get(path).and_then(Weak::upgrade) {
                return Ok(region);
            }
        }

        let file = File::open(path).map_err(|source| RuntimeError::io(path, source))?;
        // SAFETY: this is a read-only mapping of a live File. memmap2 owns the mapping
        // independently after creation, and WindowHandle/Arrow buffers retain the Arc.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|source| RuntimeError::io(path, source))?;
        let region = Arc::new(MappedRegion { mmap });

        let mut cache = self
            .mmap_cache
            .lock()
            .map_err(|_| RuntimeError::Validation("mmap cache lock was poisoned".into()))?;
        cache.insert(path.to_path_buf(), Arc::downgrade(&region));
        Ok(region)
    }
}

impl WindowHandle {
    pub fn record_id(&self) -> &str {
        &self.record.id
    }

    pub fn subject(&self) -> &str {
        &self.record.subject
    }

    pub fn modality(&self) -> &str {
        &self.record.modality
    }

    pub fn start_frame(&self) -> usize {
        self.start_frame
    }

    pub fn shape(&self) -> Vec<usize> {
        let mut shape = self.record.shape.clone();
        shape[0] = self.length_frames;
        shape
    }

    pub fn sampling_hz(&self) -> Option<f64> {
        self.record.sampling_hz
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn source_size_bytes(&self) -> u64 {
        self.source_size_bytes
    }

    pub fn element_len(&self) -> Result<usize> {
        self.length_frames
            .checked_mul(self.frame_elements)
            .ok_or_else(|| RuntimeError::InvalidWindow("window element count overflowed usize".into()))
    }

    /// Create an Arrow Float32Array that owns a reference to the mmap and therefore
    /// views the source bytes without copying them.
    pub fn arrow_values(&self) -> Result<Float32Array> {
        let element_len = self.element_len()?;
        let element_offset = self
            .start_frame
            .checked_mul(self.frame_elements)
            .ok_or_else(|| RuntimeError::InvalidWindow("window offset overflowed usize".into()))?;
        let relative_byte_offset = element_offset
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::InvalidWindow("window byte offset overflowed usize".into()))?;
        let byte_offset = usize::try_from(self.record.offset_bytes)
            .ok()
            .and_then(|base| base.checked_add(relative_byte_offset))
            .ok_or_else(|| RuntimeError::InvalidWindow("source byte offset overflowed usize".into()))?;
        let byte_len = element_len
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::InvalidWindow("window byte length overflowed usize".into()))?;
        let byte_end = byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| RuntimeError::InvalidWindow("window byte extent overflowed usize".into()))?;
        if byte_end > self.region.mmap.len() {
            return Err(RuntimeError::InvalidWindow(
                "window extends beyond mapped source".into(),
            ));
        }

        // SAFETY: byte_offset..byte_end was bounds checked above. Mmap is page-aligned,
        // manifest offsets are validated by the manifest contract, and the Arc passed as
        // the Arrow allocation owner keeps the mapping alive for the full array lifetime.
        let ptr = unsafe { self.region.mmap.as_ptr().add(byte_offset) as *mut u8 };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| RuntimeError::InvalidWindow("mapped source returned a null pointer".into()))?;
        let owner = Arc::clone(&self.region);
        let buffer = unsafe { Buffer::from_custom_allocation(ptr, byte_len, owner) };
        let values = ScalarBuffer::<f32>::new(buffer, 0, element_len);
        Ok(Float32Array::new(values, None))
    }
}

impl Iterator for WindowStream {
    type Item = Result<WindowHandle>;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;
    use crate::manifest::{DType, DatasetManifest};

    fn fixture() -> (tempfile::TempDir, Arc<Dataset>) {
        let directory = tempdir().unwrap();
        let data_path = directory.path().join("fmri.f32");
        let mut data = File::create(&data_path).unwrap();
        for value in 0..24u32 {
            data.write_all(&(value as f32).to_le_bytes()).unwrap();
        }

        let manifest = DatasetManifest {
            schema_version: 1,
            dataset_id: "test".into(),
            records: vec![Record {
                id: "r1".into(),
                subject: "sub-01".into(),
                modality: "fmri".into(),
                path: "fmri.f32".into(),
                offset_bytes: 0,
                dtype: DType::Float32Le,
                shape: vec![6, 4],
                sampling_hz: Some(0.5),
                clock: None,
            }],
        };
        std::fs::write(
            directory.path().join(MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let dataset = Dataset::open(directory.path()).unwrap();
        (directory, dataset)
    }

    #[test]
    fn plans_deterministic_windows() {
        let (_directory, dataset) = fixture();
        let spec = WindowSpec::new(3, 2).unwrap();
        let plan = dataset
            .plan_windows(&StreamSelector::default(), spec)
            .unwrap();
        let starts: Vec<_> = plan.iter().map(|window| window.start_frame).collect();
        assert_eq!(starts, vec![0, 2]);
    }

    #[test]
    fn arrow_window_is_zero_copy_over_mmap() {
        let (_directory, dataset) = fixture();
        let mut stream = dataset
            .stream(StreamSelector::default(), WindowSpec::new(2, 2).unwrap(), 2)
            .unwrap();
        let window = stream.next().unwrap().unwrap();
        let values = window.arrow_values().unwrap();
        assert_eq!(values.values().as_ref(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(window.shape(), vec![2, 4]);
    }
}
