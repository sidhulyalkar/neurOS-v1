use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Weak};
use std::thread;

use arrow_array::Float32Array;
use arrow_buffer::{Buffer, ScalarBuffer};
use crossbeam_channel::{Receiver, Sender, bounded};
use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};
use tracing::{debug, instrument};

use crate::content_identity::declared_dataset_content_sha256 as compute_dataset_content_sha256;
use crate::error::{Result, RuntimeError};
use crate::manifest::{DatasetManifest, MANIFEST_FILE, Record};

#[derive(Clone, Debug, Default)]
pub struct StreamSelector {
    pub subjects: Vec<String>,
    pub modalities: Vec<String>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceVerificationState {
    Unverified,
    VerifiedAtOpen,
}

impl SourceVerificationState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unverified => "unverified",
            Self::VerifiedAtOpen => "verified_at_open",
        }
    }
}

#[derive(Clone, Debug)]
struct WindowDescriptor {
    record: Arc<Record>,
    start_frame: usize,
    length_frames: usize,
    frame_elements: usize,
}

pub struct Dataset {
    root: PathBuf,
    manifest: DatasetManifest,
    records: Vec<Arc<Record>>,
    manifest_sha256: String,
    declared_dataset_content_sha256: Option<String>,
    verified_dataset_content_sha256: Mutex<Option<String>>,
    mmap_cache: Mutex<HashMap<PathBuf, Weak<MappedRegion>>>,
}

#[derive(Debug)]
struct MappedRegion {
    mmap: Mmap,
    verified_source_sha256: Mutex<Option<String>>,
}

#[derive(Clone)]
pub struct WindowHandle {
    region: Arc<MappedRegion>,
    record: Arc<Record>,
    start_frame: usize,
    end_frame_exclusive: usize,
    length_frames: usize,
    frame_elements: usize,
    record_byte_end_exclusive: u64,
    source_size_bytes: u64,
    manifest_sha256: String,
    declared_source_sha256: Option<String>,
    verified_source_sha256: Option<String>,
    source_verification_state: SourceVerificationState,
    declared_dataset_content_sha256: Option<String>,
    verified_dataset_content_sha256: Option<String>,
}

enum StreamMessage {
    Window(Result<WindowHandle>),
    Finished,
}

pub struct WindowStream {
    receiver: Receiver<StreamMessage>,
    finished: bool,
}

impl Dataset {
    #[instrument(skip(root), fields(root = %root.as_ref().display()))]
    pub fn open(root: impl AsRef<Path>) -> Result<Arc<Self>> {
        if cfg!(target_endian = "big") {
            return Err(RuntimeError::Validation(
                "the v0 runtime supports little-endian hosts only".into(),
            ));
        }

        let requested_root = root.as_ref();
        let root = std::fs::canonicalize(requested_root)
            .map_err(|source| RuntimeError::io(requested_root, source))?;
        let root_metadata =
            std::fs::metadata(&root).map_err(|source| RuntimeError::io(&root, source))?;
        if !root_metadata.is_dir() {
            return Err(RuntimeError::Validation(format!(
                "dataset root is not a directory: {}",
                root.display()
            )));
        }

        let manifest_path = root.join(MANIFEST_FILE);
        let bytes = std::fs::read(&manifest_path)
            .map_err(|source| RuntimeError::io(&manifest_path, source))?;
        let manifest: DatasetManifest = serde_json::from_slice(&bytes)?;
        manifest.validate()?;
        let declared_dataset_content_sha256 = compute_dataset_content_sha256(&manifest)?;
        let records = manifest.records.iter().cloned().map(Arc::new).collect();
        let manifest_sha256 = format!("{:x}", Sha256::digest(&bytes));

        Ok(Arc::new(Self {
            root,
            manifest,
            records,
            manifest_sha256,
            declared_dataset_content_sha256,
            verified_dataset_content_sha256: Mutex::new(None),
            mmap_cache: Mutex::new(HashMap::new()),
        }))
    }

    pub fn manifest(&self) -> &DatasetManifest {
        &self.manifest
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn declared_dataset_content_sha256(&self) -> Option<&str> {
        self.declared_dataset_content_sha256.as_deref()
    }

    pub fn verified_dataset_content_sha256(&self) -> Result<Option<String>> {
        self.verified_dataset_content_sha256
            .lock()
            .map(|value| value.clone())
            .map_err(|_| {
                RuntimeError::Validation("dataset verification state lock was poisoned".into())
            })
    }

    /// Verify every declared source needed by the canonical dataset content identity.
    ///
    /// Returns `Ok(None)` for a partially hashed manifest. No dataset-level verified
    /// identity is claimed unless every record declares a source SHA-256 and every
    /// referenced mapped source matches its declaration.
    pub fn verify_content(&self) -> Result<Option<String>> {
        let Some(expected_dataset_sha256) = self.declared_dataset_content_sha256.clone() else {
            return Ok(None);
        };

        let mut records: Vec<_> = self.records.iter().collect();
        records.sort_by(|left, right| left.id.cmp(&right.id));
        // The process-wide mmap cache intentionally stores weak references. Hold
        // verified mappings strongly for this pass so multiple records referencing
        // one source share verification work even when no window is alive yet.
        let mut verified_regions = Vec::new();
        for record in records {
            let expected_source_sha256 = record.source_sha256.as_deref().ok_or_else(|| {
                RuntimeError::Validation(
                    "declared dataset content identity requires every record source hash".into(),
                )
            })?;
            let (path, required_end) = self.resolve_record_source(record)?;
            let (region, verified_source_sha256) =
                self.map_source(&path, Some(expected_source_sha256))?;
            let mapped_size = mapped_size_bytes(&region)?;
            if mapped_size < required_end {
                return Err(RuntimeError::SourceTooShort {
                    path,
                    actual: mapped_size,
                    required: required_end,
                });
            }
            if verified_source_sha256.as_deref() != Some(expected_source_sha256) {
                return Err(RuntimeError::Validation(
                    "source verification completed without the declared digest".into(),
                ));
            }
            verified_regions.push(region);
        }

        let mut state = self.verified_dataset_content_sha256.lock().map_err(|_| {
            RuntimeError::Validation("dataset verification state lock was poisoned".into())
        })?;
        *state = Some(expected_dataset_sha256.clone());
        Ok(Some(expected_dataset_sha256))
    }

    pub fn plan_windows(
        &self,
        selector: &StreamSelector,
        spec: WindowSpec,
    ) -> Result<Vec<(String, usize)>> {
        let mut windows = Vec::new();
        self.visit_window_descriptors(selector, spec, |descriptor| {
            windows.push((descriptor.record.id.clone(), descriptor.start_frame));
            Ok(true)
        })?;
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
        let (sender, receiver) = bounded(prefetch);
        let dataset = Arc::clone(self);

        thread::Builder::new()
            .name("neuros-data-prefetch".into())
            .spawn(move || dataset.run_stream_worker(selector, spec, sender))
            .map_err(|source| RuntimeError::io("<prefetch-thread>", source))?;

        Ok(WindowStream {
            receiver,
            finished: false,
        })
    }

    fn visit_window_descriptors<F>(
        &self,
        selector: &StreamSelector,
        spec: WindowSpec,
        mut visitor: F,
    ) -> Result<bool>
    where
        F: FnMut(WindowDescriptor) -> Result<bool>,
    {
        let subjects: HashSet<&str> = selector.subjects.iter().map(String::as_str).collect();
        let modalities: HashSet<&str> = selector.modalities.iter().map(String::as_str).collect();

        for record in &self.records {
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
                if !visitor(WindowDescriptor {
                    record: Arc::clone(record),
                    start_frame: start,
                    length_frames: spec.length,
                    frame_elements,
                })? {
                    return Ok(false);
                }

                let Some(next_start) = start.checked_add(spec.stride) else {
                    return Err(RuntimeError::InvalidWindow(
                        "window stride overflowed usize".into(),
                    ));
                };
                if next_start > final_start {
                    break;
                }
                start = next_start;
            }
        }
        Ok(true)
    }

    fn run_stream_worker(
        self: Arc<Self>,
        selector: StreamSelector,
        spec: WindowSpec,
        sender: Sender<StreamMessage>,
    ) {
        let result = self.visit_window_descriptors(&selector, spec, |descriptor| {
            match self.open_window(descriptor) {
                Ok(window) => {
                    if sender.send(StreamMessage::Window(Ok(window))).is_err() {
                        debug!("window consumer dropped; cancelling prefetch worker");
                        return Ok(false);
                    }
                }
                Err(error) => {
                    if sender.send(StreamMessage::Window(Err(error))).is_ok() {
                        let _ = sender.send(StreamMessage::Finished);
                    }
                    return Ok(false);
                }
            }
            Ok(true)
        });

        match result {
            Ok(true) => {
                let _ = sender.send(StreamMessage::Finished);
            }
            Ok(false) => {}
            Err(error) => {
                if sender.send(StreamMessage::Window(Err(error))).is_ok() {
                    let _ = sender.send(StreamMessage::Finished);
                }
            }
        }
    }

    fn open_window(&self, descriptor: WindowDescriptor) -> Result<WindowHandle> {
        let (path, required_end) = self.resolve_record_source(&descriptor.record)?;
        let declared_source_sha256 = descriptor.record.source_sha256.clone();
        let (region, verified_source_sha256) =
            self.map_source(&path, declared_source_sha256.as_deref())?;
        let mapped_size = mapped_size_bytes(&region)?;
        if mapped_size < required_end {
            return Err(RuntimeError::SourceTooShort {
                path,
                actual: mapped_size,
                required: required_end,
            });
        }
        let source_verification_state = if verified_source_sha256.is_some() {
            SourceVerificationState::VerifiedAtOpen
        } else {
            SourceVerificationState::Unverified
        };
        let end_frame_exclusive = descriptor
            .start_frame
            .checked_add(descriptor.length_frames)
            .ok_or_else(|| {
                RuntimeError::InvalidWindow("window frame extent overflowed usize".into())
            })?;
        let verified_dataset_content_sha256 = self.verified_dataset_content_sha256()?;

        Ok(WindowHandle {
            region,
            record: descriptor.record,
            start_frame: descriptor.start_frame,
            end_frame_exclusive,
            length_frames: descriptor.length_frames,
            frame_elements: descriptor.frame_elements,
            record_byte_end_exclusive: required_end,
            source_size_bytes: mapped_size,
            manifest_sha256: self.manifest_sha256.clone(),
            declared_source_sha256,
            verified_source_sha256,
            source_verification_state,
            declared_dataset_content_sha256: self.declared_dataset_content_sha256.clone(),
            verified_dataset_content_sha256,
        })
    }

    fn resolve_record_source(&self, record: &Record) -> Result<(PathBuf, u64)> {
        let requested_path = self.root.join(&record.path);
        let path = std::fs::canonicalize(&requested_path)
            .map_err(|source| RuntimeError::io(&requested_path, source))?;
        if !path.starts_with(&self.root) {
            return Err(RuntimeError::Validation(format!(
                "record {:?} resolves outside the dataset root: {}",
                record.id,
                path.display()
            )));
        }

        let required_end = record.required_end_byte()?;
        let metadata =
            std::fs::metadata(&path).map_err(|source| RuntimeError::io(&path, source))?;
        if !metadata.is_file() {
            return Err(RuntimeError::Validation(format!(
                "record {:?} source is not a regular file: {}",
                record.id,
                path.display()
            )));
        }
        if metadata.len() < required_end {
            return Err(RuntimeError::SourceTooShort {
                path,
                actual: metadata.len(),
                required: required_end,
            });
        }
        Ok((path, required_end))
    }

    fn map_source(
        &self,
        path: &Path,
        expected_sha256: Option<&str>,
    ) -> Result<(Arc<MappedRegion>, Option<String>)> {
        let cached_region = {
            let cache = self
                .mmap_cache
                .lock()
                .map_err(|_| RuntimeError::Validation("mmap cache lock was poisoned".into()))?;
            cache.get(path).and_then(Weak::upgrade)
        };
        if let Some(region) = cached_region {
            let verified = Self::ensure_source_verified(path, &region, expected_sha256)?;
            return Ok((region, verified));
        }

        let file = File::open(path).map_err(|source| RuntimeError::io(path, source))?;
        // SAFETY: this is a read-only mapping of a live File. memmap2 owns the mapping
        // independently after creation, and WindowHandle/Arrow buffers retain the Arc.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|source| RuntimeError::io(path, source))?;
        let verified_source_sha256 = match expected_sha256 {
            Some(expected) => Some(Self::verify_mapped_sha256(path, &mmap, expected)?),
            None => None,
        };
        let region = Arc::new(MappedRegion {
            mmap,
            verified_source_sha256: Mutex::new(verified_source_sha256.clone()),
        });

        let mut cache = self
            .mmap_cache
            .lock()
            .map_err(|_| RuntimeError::Validation("mmap cache lock was poisoned".into()))?;
        cache.insert(path.to_path_buf(), Arc::downgrade(&region));
        Ok((region, verified_source_sha256))
    }

    fn ensure_source_verified(
        path: &Path,
        region: &MappedRegion,
        expected_sha256: Option<&str>,
    ) -> Result<Option<String>> {
        let Some(expected) = expected_sha256 else {
            return Ok(None);
        };

        let mut state = region.verified_source_sha256.lock().map_err(|_| {
            RuntimeError::Validation("source verification cache lock was poisoned".into())
        })?;
        if let Some(actual) = state.as_ref() {
            if actual != expected {
                return Err(RuntimeError::SourceHashMismatch {
                    path: path.to_path_buf(),
                    expected: expected.to_owned(),
                    actual: actual.clone(),
                });
            }
            return Ok(Some(actual.clone()));
        }

        let actual = Self::verify_mapped_sha256(path, &region.mmap, expected)?;
        *state = Some(actual.clone());
        Ok(Some(actual))
    }

    fn verify_mapped_sha256(path: &Path, mmap: &Mmap, expected: &str) -> Result<String> {
        let actual = format!("{:x}", Sha256::digest(mmap.as_ref()));
        if actual != expected {
            return Err(RuntimeError::SourceHashMismatch {
                path: path.to_path_buf(),
                expected: expected.to_owned(),
                actual,
            });
        }
        Ok(actual)
    }
}

fn mapped_size_bytes(region: &MappedRegion) -> Result<u64> {
    u64::try_from(region.mmap.len())
        .map_err(|_| RuntimeError::Validation("mapped source size does not fit in u64".into()))
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

    pub const fn end_frame_exclusive(&self) -> usize {
        self.end_frame_exclusive
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

    pub fn record_byte_start(&self) -> u64 {
        self.record.offset_bytes
    }

    pub const fn record_byte_end_exclusive(&self) -> u64 {
        self.record_byte_end_exclusive
    }

    pub fn declared_source_sha256(&self) -> Option<&str> {
        self.declared_source_sha256.as_deref()
    }

    pub fn verified_source_sha256(&self) -> Option<&str> {
        self.verified_source_sha256.as_deref()
    }

    pub const fn source_verification_state(&self) -> SourceVerificationState {
        self.source_verification_state
    }

    pub fn declared_dataset_content_sha256(&self) -> Option<&str> {
        self.declared_dataset_content_sha256.as_deref()
    }

    pub fn verified_dataset_content_sha256(&self) -> Option<&str> {
        self.verified_dataset_content_sha256.as_deref()
    }

    pub fn element_len(&self) -> Result<usize> {
        self.length_frames
            .checked_mul(self.frame_elements)
            .ok_or_else(|| {
                RuntimeError::InvalidWindow("window element count overflowed usize".into())
            })
    }

    /// Create an Arrow Float32Array that owns a reference to the mmap and therefore
    /// views the source bytes without copying them.
    pub fn arrow_values(&self) -> Result<Float32Array> {
        let element_len = self.element_len()?;
        let element_offset = self
            .start_frame
            .checked_mul(self.frame_elements)
            .ok_or_else(|| RuntimeError::InvalidWindow("window offset overflowed usize".into()))?;
        let relative_byte_offset = element_offset.checked_mul(4).ok_or_else(|| {
            RuntimeError::InvalidWindow("window byte offset overflowed usize".into())
        })?;
        let byte_offset = usize::try_from(self.record.offset_bytes)
            .ok()
            .and_then(|base| base.checked_add(relative_byte_offset))
            .ok_or_else(|| {
                RuntimeError::InvalidWindow("source byte offset overflowed usize".into())
            })?;
        let byte_len = element_len.checked_mul(4).ok_or_else(|| {
            RuntimeError::InvalidWindow("window byte length overflowed usize".into())
        })?;
        let byte_end = byte_offset.checked_add(byte_len).ok_or_else(|| {
            RuntimeError::InvalidWindow("window byte extent overflowed usize".into())
        })?;
        if byte_end > self.region.mmap.len() {
            return Err(RuntimeError::InvalidWindow(
                "window extends beyond mapped source".into(),
            ));
        }

        // SAFETY: byte_offset..byte_end was bounds checked above. Mmap is page-aligned,
        // manifest offsets are validated by the manifest contract, and the Arc passed as
        // the Arrow allocation owner keeps the mapping alive for the full array lifetime.
        let ptr = unsafe { self.region.mmap.as_ptr().add(byte_offset) as *mut u8 };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            RuntimeError::InvalidWindow("mapped source returned a null pointer".into())
        })?;
        let owner = Arc::clone(&self.region);
        let buffer = unsafe { Buffer::from_custom_allocation(ptr, byte_len, owner) };
        let values = ScalarBuffer::<f32>::new(buffer, 0, element_len);
        Ok(Float32Array::new(values, None))
    }
}

impl Iterator for WindowStream {
    type Item = Result<WindowHandle>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        match self.receiver.recv() {
            Ok(StreamMessage::Window(window)) => Some(window),
            Ok(StreamMessage::Finished) => {
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;
    use crate::manifest::{DType, DatasetManifest};

    fn fixture_with_declared_hash(declare_hash: bool) -> (tempfile::TempDir, Arc<Dataset>, String) {
        let directory = tempdir().unwrap();
        let data_path = directory.path().join("fmri.f32");
        let mut data = File::create(&data_path).unwrap();
        for value in 0..24u32 {
            data.write_all(&(value as f32).to_le_bytes()).unwrap();
        }
        drop(data);
        let source_bytes = std::fs::read(&data_path).unwrap();
        let source_sha256 = format!("{:x}", Sha256::digest(&source_bytes));

        let manifest = DatasetManifest {
            schema_version: 1,
            dataset_id: "test".into(),
            records: vec![Record {
                id: "r1".into(),
                subject: "sub-01".into(),
                modality: "fmri".into(),
                sync_group: None,
                path: "fmri.f32".into(),
                source_sha256: declare_hash.then(|| source_sha256.clone()),
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
        (directory, dataset, source_sha256)
    }

    fn fixture() -> (tempfile::TempDir, Arc<Dataset>) {
        let (directory, dataset, _) = fixture_with_declared_hash(false);
        (directory, dataset)
    }

    #[test]
    fn plans_deterministic_windows() {
        let (_directory, dataset) = fixture();
        let spec = WindowSpec::new(3, 2).unwrap();
        let plan = dataset
            .plan_windows(&StreamSelector::default(), spec)
            .unwrap();
        let starts: Vec<_> = plan.iter().map(|(_, start)| *start).collect();
        assert_eq!(starts, vec![0, 2]);
    }

    #[test]
    fn stride_larger_than_final_start_never_emits_out_of_bounds_window() {
        let (_directory, dataset) = fixture();
        let plan = dataset
            .plan_windows(&StreamSelector::default(), WindowSpec::new(3, 4).unwrap())
            .unwrap();
        let starts: Vec<_> = plan.iter().map(|(_, start)| *start).collect();
        assert_eq!(starts, vec![0]);
    }

    #[test]
    fn arrow_window_is_zero_copy_over_mmap() {
        let (_directory, dataset) = fixture();
        let mut stream = dataset
            .stream(StreamSelector::default(), WindowSpec::new(2, 2).unwrap(), 2)
            .unwrap();
        let window = stream.next().unwrap().unwrap();
        let values = window.arrow_values().unwrap();
        assert_eq!(
            values.values().as_ref(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(window.shape(), vec![2, 4]);
        assert_eq!(window.start_frame(), 0);
        assert_eq!(window.end_frame_exclusive(), 2);
        assert_eq!(window.record_byte_start(), 0);
        assert_eq!(window.record_byte_end_exclusive(), 96);
        assert_eq!(
            window.source_verification_state(),
            SourceVerificationState::Unverified
        );
        assert_eq!(window.declared_source_sha256(), None);
        assert_eq!(window.verified_source_sha256(), None);
        assert_eq!(window.declared_dataset_content_sha256(), None);
        assert_eq!(window.verified_dataset_content_sha256(), None);
    }

    #[test]
    fn declared_source_hash_is_verified_before_window_is_returned() {
        let (_directory, dataset, source_sha256) = fixture_with_declared_hash(true);
        assert!(dataset.declared_dataset_content_sha256().is_some());
        assert_eq!(dataset.verified_dataset_content_sha256().unwrap(), None);
        let mut stream = dataset
            .stream(StreamSelector::default(), WindowSpec::new(2, 2).unwrap(), 1)
            .unwrap();
        let window = stream.next().unwrap().unwrap();
        assert_eq!(
            window.declared_source_sha256(),
            Some(source_sha256.as_str())
        );
        assert_eq!(
            window.verified_source_sha256(),
            Some(source_sha256.as_str())
        );
        assert_eq!(
            window.source_verification_state(),
            SourceVerificationState::VerifiedAtOpen
        );
        assert!(window.declared_dataset_content_sha256().is_some());
        assert_eq!(window.verified_dataset_content_sha256(), None);
    }

    #[test]
    fn explicit_dataset_verification_promotes_dataset_content_identity() {
        let (_directory, dataset, source_sha256) = fixture_with_declared_hash(true);
        let declared = dataset
            .declared_dataset_content_sha256()
            .unwrap()
            .to_owned();
        assert_eq!(dataset.verify_content().unwrap(), Some(declared.clone()));
        assert_eq!(
            dataset.verified_dataset_content_sha256().unwrap(),
            Some(declared.clone())
        );

        let mut stream = dataset
            .stream(StreamSelector::default(), WindowSpec::new(2, 2).unwrap(), 1)
            .unwrap();
        let window = stream.next().unwrap().unwrap();
        assert_eq!(
            window.verified_source_sha256(),
            Some(source_sha256.as_str())
        );
        assert_eq!(
            window.verified_dataset_content_sha256(),
            Some(declared.as_str())
        );
    }

    #[test]
    fn partial_hash_manifest_cannot_claim_verified_dataset_identity() {
        let (_directory, dataset) = fixture();
        assert_eq!(dataset.declared_dataset_content_sha256(), None);
        assert_eq!(dataset.verify_content().unwrap(), None);
        assert_eq!(dataset.verified_dataset_content_sha256().unwrap(), None);
    }

    #[test]
    fn source_mutation_rejects_before_first_window() {
        let (directory, dataset, source_sha256) = fixture_with_declared_hash(true);
        let data_path = directory.path().join("fmri.f32");
        let mut bytes = std::fs::read(&data_path).unwrap();
        bytes[0] ^= 0xff;
        std::fs::write(&data_path, bytes).unwrap();

        let mut stream = dataset
            .stream(StreamSelector::default(), WindowSpec::new(2, 2).unwrap(), 1)
            .unwrap();
        let error = stream.next().unwrap().unwrap_err();
        match error {
            RuntimeError::SourceHashMismatch {
                expected, actual, ..
            } => {
                assert_eq!(expected, source_sha256);
                assert_ne!(actual, expected);
            }
            other => panic!("expected source hash mismatch, got {other:?}"),
        }
        assert!(stream.next().is_none());
    }

    #[test]
    fn cached_unverified_mapping_is_upgraded_when_hash_is_declared() {
        let directory = tempdir().unwrap();
        let data_path = directory.path().join("shared.f32");
        let mut data = File::create(&data_path).unwrap();
        for value in 0..24u32 {
            data.write_all(&(value as f32).to_le_bytes()).unwrap();
        }
        drop(data);
        let actual = format!("{:x}", Sha256::digest(std::fs::read(&data_path).unwrap()));

        let record = |id: &str, source_sha256: Option<String>| Record {
            id: id.into(),
            subject: "sub-01".into(),
            modality: "fmri".into(),
            sync_group: None,
            path: "shared.f32".into(),
            source_sha256,
            offset_bytes: 0,
            dtype: DType::Float32Le,
            shape: vec![6, 4],
            sampling_hz: Some(0.5),
            clock: None,
        };
        let manifest = DatasetManifest {
            schema_version: 1,
            dataset_id: "shared-upgrade".into(),
            records: vec![record("r1", None), record("r2", Some(actual.clone()))],
        };
        std::fs::write(
            directory.path().join(MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let dataset = Dataset::open(directory.path()).unwrap();
        let mut stream = dataset
            .stream(StreamSelector::default(), WindowSpec::new(6, 6).unwrap(), 1)
            .unwrap();

        let first = stream.next().unwrap().unwrap();
        assert_eq!(
            first.source_verification_state(),
            SourceVerificationState::Unverified
        );
        assert_eq!(first.verified_source_sha256(), None);

        let second = stream.next().unwrap().unwrap();
        assert_eq!(second.declared_source_sha256(), Some(actual.as_str()));
        assert_eq!(second.verified_source_sha256(), Some(actual.as_str()));
        assert_eq!(
            second.source_verification_state(),
            SourceVerificationState::VerifiedAtOpen
        );
        assert!(stream.next().is_none());

        // Keep the first window alive through the upgrade so the second record must
        // reuse and upgrade the same cached mmap rather than create a fresh mapping.
        assert_eq!(first.record_id(), "r1");
    }

    #[test]
    fn cached_mapping_does_not_accept_a_conflicting_declared_hash() {
        let directory = tempdir().unwrap();
        let data_path = directory.path().join("shared.f32");
        let mut data = File::create(&data_path).unwrap();
        for value in 0..24u32 {
            data.write_all(&(value as f32).to_le_bytes()).unwrap();
        }
        drop(data);
        let actual = format!("{:x}", Sha256::digest(std::fs::read(&data_path).unwrap()));
        let wrong = "0".repeat(64);
        assert_ne!(actual, wrong);

        let record = |id: &str, source_sha256: String| Record {
            id: id.into(),
            subject: "sub-01".into(),
            modality: "fmri".into(),
            sync_group: None,
            path: "shared.f32".into(),
            source_sha256: Some(source_sha256),
            offset_bytes: 0,
            dtype: DType::Float32Le,
            shape: vec![6, 4],
            sampling_hz: Some(0.5),
            clock: None,
        };
        let manifest = DatasetManifest {
            schema_version: 1,
            dataset_id: "shared".into(),
            records: vec![record("r1", actual.clone()), record("r2", wrong)],
        };
        std::fs::write(
            directory.path().join(MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let dataset = Dataset::open(directory.path()).unwrap();
        let mut stream = dataset
            .stream(StreamSelector::default(), WindowSpec::new(6, 6).unwrap(), 1)
            .unwrap();
        let first = stream.next().unwrap().unwrap();
        assert_eq!(first.verified_source_sha256(), Some(actual.as_str()));
        assert!(matches!(
            stream.next().unwrap(),
            Err(RuntimeError::SourceHashMismatch { .. })
        ));
        assert!(stream.next().is_none());
    }

    #[test]
    fn source_error_is_reported_once_then_stream_finishes() {
        let (directory, dataset) = fixture();
        std::fs::write(directory.path().join("fmri.f32"), [0_u8; 4]).unwrap();
        let mut stream = dataset
            .stream(StreamSelector::default(), WindowSpec::new(2, 2).unwrap(), 2)
            .unwrap();
        assert!(stream.next().unwrap().is_err());
        assert!(stream.next().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape_from_dataset_root() {
        use std::os::unix::fs::symlink;

        let (directory, dataset) = fixture();
        let outside = tempdir().unwrap();
        let outside_file = outside.path().join("outside.f32");
        std::fs::write(&outside_file, [0_u8; 96]).unwrap();
        std::fs::remove_file(directory.path().join("fmri.f32")).unwrap();
        symlink(&outside_file, directory.path().join("fmri.f32")).unwrap();

        let mut stream = dataset
            .stream(StreamSelector::default(), WindowSpec::new(2, 2).unwrap(), 2)
            .unwrap();
        assert!(stream.next().unwrap().is_err());
        assert!(stream.next().is_none());
    }
}
