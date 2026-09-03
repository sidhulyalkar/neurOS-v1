from pathlib import Path


def replace_once(text: str, needle: str, replacement: str, label: str) -> str:
    count = text.count(needle)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(needle, replacement, 1)


def patch_aligned() -> None:
    path = Path("rust/neuros-runtime/src/aligned.rs")
    text = path.read_text()

    needle = "        let mut overlap_end = i128::MAX;\n"
    replacement = (
        "        let mut overlap_start = i128::MIN;\n"
        "        let mut overlap_end = i128::MAX;\n"
        "        let mut common_period = 1i128;\n"
    )
    text = replace_once(text, needle, replacement, "aligned overlap state")

    needle = "            let period_ns = i128::from(clock.period_ns);\n"
    replacement = (
        needle
        + "            overlap_start = overlap_start.max(i128::from(clock.start_ns));\n"
        + "            common_period = checked_lcm(common_period, period_ns)?;\n"
    )
    text = replace_once(text, needle, replacement, "aligned exact-period state")

    needle = "        let expected_overlap_end = i64::try_from(overlap_end).map_err(|_| {\n"
    earliest = """        let previous_common_boundary = start_ns.checked_sub(common_period).ok_or_else(|| {
            RuntimeError::Alignment(
                "alignment plan previous common boundary overflowed clock arithmetic".into(),
            )
        })?;
        if start_ns < overlap_start || previous_common_boundary >= overlap_start {
            return Err(RuntimeError::Alignment(
                "alignment plan does not begin at the earliest exact common boundary".into(),
            ));
        }

"""
    text = replace_once(text, needle, earliest + needle, "earliest common boundary")

    helper_anchor = "fn send_terminal_error(sender: &Sender<AlignedStreamMessage>, error: RuntimeError) {\n"
    helpers = """fn checked_lcm(left: i128, right: i128) -> Result<i128> {
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

"""
    if "fn checked_lcm(" in text:
        raise RuntimeError("checked_lcm already present before bootstrap")
    text = replace_once(text, helper_anchor, helpers + helper_anchor, "LCM helper anchor")

    text = text.replace(".unwrap_err();", ".err().unwrap();")
    if ".unwrap_err();" in text:
        raise RuntimeError("unwrap_err remained after test normalization")

    needle = '        assert!(error.to_string().contains("not maximal exact count"));\n'
    delayed_test = needle + """
        let mut delayed = plan.clone();
        delayed.start_ns += delayed.stride_ns as i64;
        delayed.window_count -= 1;
        for entry in &mut delayed.entries {
            entry.start_frame += entry.frame_stride;
        }
        let error = dataset.stream_aligned(&delayed, 1).err().unwrap();
        assert!(error
            .to_string()
            .contains("earliest exact common boundary"));
"""
    text = replace_once(text, needle, delayed_test, "delayed-plan adversarial test")
    path.write_text(text)


def patch_dataset() -> None:
    path = Path("rust/neuros-runtime/src/dataset.rs")
    text = path.read_text()

    manifest_anchor = """    pub fn manifest(&self) -> &DatasetManifest {
        &self.manifest
    }
"""
    records_method = manifest_anchor + """
    pub(crate) fn records(&self) -> &[Arc<Record>] {
        &self.records
    }
"""
    if "pub(crate) fn records(&self)" in text:
        raise RuntimeError("records hook already present before bootstrap")
    text = replace_once(text, manifest_anchor, records_method, "dataset records hook")

    plan_windows_anchor = "    pub fn plan_windows(\n"
    fresh_verify = """    /// Re-hash every declared source even when a prior verification cache exists.
    ///
    /// Exact aligned execution uses this at worker authorization time so a source
    /// changed after planning cannot inherit an older cached verification result.
    pub(crate) fn verify_content_fresh(&self) -> Result<Option<String>> {
        let Some(expected_dataset_sha256) = self.declared_dataset_content_sha256.clone() else {
            return Ok(None);
        };

        let mut records: Vec<_> = self.records.iter().collect();
        records.sort_by(|left, right| left.id.cmp(&right.id));
        let mut verified_regions = Vec::new();
        for record in records {
            let expected_source_sha256 = record.source_sha256.as_deref().ok_or_else(|| {
                RuntimeError::Validation(
                    "declared dataset content identity requires every record source hash".into(),
                )
            })?;
            let (path, required_end) = self.resolve_record_source(record)?;
            let (region, _) = self.map_source(&path, None)?;
            let mapped_size = mapped_size_bytes(&region)?;
            if mapped_size < required_end {
                return Err(RuntimeError::SourceTooShort {
                    path,
                    actual: mapped_size,
                    required: required_end,
                });
            }
            let actual = Self::verify_mapped_sha256(&path, &region.mmap, expected_source_sha256)?;
            let mut state = region.verified_source_sha256.lock().map_err(|_| {
                RuntimeError::Validation("source verification cache lock was poisoned".into())
            })?;
            *state = Some(actual);
            drop(state);
            verified_regions.push(region);
        }

        let mut state = self.verified_dataset_content_sha256.lock().map_err(|_| {
            RuntimeError::Validation("dataset verification state lock was poisoned".into())
        })?;
        *state = Some(expected_dataset_sha256.clone());
        Ok(Some(expected_dataset_sha256))
    }

"""
    if "pub(crate) fn verify_content_fresh" in text:
        raise RuntimeError("fresh verification hook already present before bootstrap")
    text = replace_once(
        text,
        plan_windows_anchor,
        fresh_verify + plan_windows_anchor,
        "fresh verification hook",
    )

    open_window_anchor = "    fn open_window(&self, descriptor: WindowDescriptor) -> Result<WindowHandle> {\n"
    open_record_window = """    pub(crate) fn open_record_window(
        &self,
        record: Arc<Record>,
        start_frame: usize,
        length_frames: usize,
    ) -> Result<WindowHandle> {
        if length_frames == 0 {
            return Err(RuntimeError::InvalidWindow(
                "aligned child window length must be positive".into(),
            ));
        }
        let end_frame_exclusive = start_frame.checked_add(length_frames).ok_or_else(|| {
            RuntimeError::InvalidWindow("aligned child frame extent overflowed usize".into())
        })?;
        if end_frame_exclusive > record.shape[0] {
            return Err(RuntimeError::InvalidWindow(format!(
                "aligned child window [{start_frame}, {end_frame_exclusive}) exceeds record {:?} with {} frames",
                record.id, record.shape[0]
            )));
        }
        let frame_elements = record.frame_elements()?;
        self.open_window(WindowDescriptor {
            record,
            start_frame,
            length_frames,
            frame_elements,
        })
    }

"""
    if "pub(crate) fn open_record_window" in text:
        raise RuntimeError("open-record-window hook already present before bootstrap")
    text = replace_once(
        text,
        open_window_anchor,
        open_record_window + open_window_anchor,
        "open-record-window hook",
    )
    path.write_text(text)


def patch_lib() -> None:
    path = Path("rust/neuros-runtime/src/lib.rs")
    text = path.read_text()
    if "mod aligned;" in text:
        raise RuntimeError("aligned module already wired before bootstrap")
    text = replace_once(
        text,
        "mod content_identity;\n",
        "mod aligned;\nmod content_identity;\n",
        "aligned module declaration",
    )
    export_anchor = "pub use content_identity::{DATASET_CONTENT_DOMAIN, declared_dataset_content_sha256};\n"
    text = replace_once(
        text,
        export_anchor,
        "pub use aligned::{AlignedBatch, AlignedBatchStream};\n" + export_anchor,
        "aligned public exports",
    )
    path.write_text(text)


def main() -> None:
    patch_aligned()
    patch_dataset()
    patch_lib()


if __name__ == "__main__":
    main()
