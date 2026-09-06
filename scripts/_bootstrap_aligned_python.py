from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


# PyO3 binding
path = Path("rust/neuros-runtime-py/src/lib.rs")
text = path.read_text()
text = replace_once(
    text,
    "use neuros_runtime::{\n    Dataset, ExactAlignmentPlan, ExactAlignmentSpec, StreamSelector, WindowHandle, WindowSpec,\n    WindowStream, plan_exact_alignment,\n};",
    "use neuros_runtime::{\n    AlignedBatch, AlignedBatchStream, Dataset, ExactAlignmentPlan, ExactAlignmentSpec,\n    StreamSelector, WindowHandle, WindowSpec, WindowStream, plan_exact_alignment,\n};",
    "native imports",
)
stream_marker = "    #[pyo3(signature = (*, subjects=None, modalities=None, window, stride=None, prefetch=8))]\n"
stream_aligned = '''    #[pyo3(signature = (*, plan, prefetch=8))]
    fn stream_aligned(
        &self,
        py: Python<'_>,
        plan: PyRef<'_, NativeAlignmentPlan>,
        prefetch: usize,
    ) -> PyResult<NativeAlignedBatchStream> {
        let plan = plan.inner.clone();
        let stream = py
            .detach(|| self.inner.stream_aligned(&plan, prefetch))
            .map_err(runtime_error)?;
        Ok(NativeAlignedBatchStream {
            inner: Mutex::new(stream),
        })
    }

'''
text = replace_once(text, stream_marker, stream_aligned + stream_marker, "native stream_aligned")
class_marker = "#[pyclass]\nstruct NativeWindowStream {\n"
aligned_classes = '''#[pyclass]
struct NativeAlignedBatchStream {
    inner: Mutex<AlignedBatchStream>,
}

#[pymethods]
impl NativeAlignedBatchStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<NativeAlignedBatch>> {
        let next = py.detach(|| {
            self.inner
                .lock()
                .map_err(|_| "native aligned stream lock was poisoned".to_owned())
                .map(|mut stream| stream.next())
        });
        let next = next.map_err(PyRuntimeError::new_err)?;
        match next {
            None => Ok(None),
            Some(Ok(batch)) => Ok(Some(NativeAlignedBatch { inner: batch })),
            Some(Err(error)) => Err(runtime_error(error)),
        }
    }
}

#[pyclass]
struct NativeAlignedBatch {
    inner: AlignedBatch,
}

#[pymethods]
impl NativeAlignedBatch {
    #[getter]
    fn plan_sha256(&self) -> String {
        self.inner.plan_sha256().to_owned()
    }

    #[getter]
    fn dataset_content_sha256(&self) -> String {
        self.inner.dataset_content_sha256().to_owned()
    }

    #[getter]
    fn manifest_sha256(&self) -> String {
        self.inner.manifest_sha256().to_owned()
    }

    #[getter]
    fn window_index(&self) -> usize {
        self.inner.window_index()
    }

    #[getter]
    fn start_ns(&self) -> i64 {
        self.inner.start_ns()
    }

    #[getter]
    fn end_ns(&self) -> i64 {
        self.inner.end_ns()
    }

    #[getter]
    fn windows(&self) -> Vec<NativeWindow> {
        self.inner
            .windows()
            .iter()
            .cloned()
            .map(|window| NativeWindow { inner: window })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeAlignedBatch(index={}, start_ns={}, end_ns={}, modalities={})",
            self.inner.window_index(),
            self.inner.start_ns(),
            self.inner.end_ns(),
            self.inner.windows().len(),
        )
    }
}

'''
text = replace_once(text, class_marker, aligned_classes + class_marker, "native aligned classes")
module_marker = "    module.add_class::<NativeAlignmentPlan>()?;\n"
text = replace_once(
    text,
    module_marker,
    module_marker + "    module.add_class::<NativeAlignedBatchStream>()?;\n    module.add_class::<NativeAlignedBatch>()?;\n",
    "native module classes",
)
path.write_text(text)


# High-level Python facade
path = Path("packages/neuros/src/neuros/dataset.py")
text = path.read_text()
text = text.replace(
    "Single-modality streaming remains the v0 execution contract. Multimodal v1\nstarts with a provenance-bound exact-clock planning authority: neurOS verifies\nthe complete declared dataset content and proves a cross-modal frame mapping\nbefore it is allowed to execute one.",
    "Single-modality streaming remains the v0 execution contract. Multimodal v1\nuses a provenance-bound exact-clock planning authority followed by an executor\nthat consumes that exact plan without silently recomputing synchronization.",
    1,
)
dataset_marker = "\n\nclass Dataset:\n"
aligned_batch = '''

@dataclass(frozen=True, slots=True)
class AlignedBatch:
    """One provenance-bound exact multimodal batch.

    Child windows retain the native mmap owner and therefore preserve the same
    zero-copy Arrow semantics as single-modality ``DataWindow`` instances.
    """

    _native_batch: Any

    @property
    def plan_sha256(self) -> str:
        return str(self._native_batch.plan_sha256)

    @property
    def dataset_content_sha256(self) -> str:
        return str(self._native_batch.dataset_content_sha256)

    @property
    def manifest_sha256(self) -> str:
        return str(self._native_batch.manifest_sha256)

    @property
    def window_index(self) -> int:
        return int(self._native_batch.window_index)

    @property
    def start_ns(self) -> int:
        return int(self._native_batch.start_ns)

    @property
    def end_ns(self) -> int:
        return int(self._native_batch.end_ns)

    @property
    def windows(self) -> tuple[DataWindow, ...]:
        return tuple(DataWindow(window) for window in self._native_batch.windows)

    @property
    def by_modality(self) -> dict[str, DataWindow]:
        windows = self.windows
        mapping = {window.modality: window for window in windows}
        if len(mapping) != len(windows):  # pragma: no cover - native invariant
            raise RuntimeError("aligned batch contains duplicate modalities")
        return mapping

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "dataset_content_sha256": self.dataset_content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "window_index": self.window_index,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "slices": tuple(window.provenance for window in self.windows),
            "verification_boundary": (
                "dataset content was freshly verified at aligned stream authorization; "
                "backing files are not claimed immutable afterward"
            ),
        }

    def __getattr__(self, name: str) -> DataWindow:
        normalized = name.replace("_", "-").lower()
        for modality, window in self.by_modality.items():
            if modality.lower() == normalized or modality.replace("-", "_").lower() == name.lower():
                return window
        raise AttributeError(name)
'''
text = replace_once(text, dataset_marker, aligned_batch + dataset_marker, "python AlignedBatch")
method_marker = "        return AlignmentPlan(native_plan)\n\n    def to_orion_lineage(\n"
stream_method = '''        return AlignmentPlan(native_plan)

    def stream_aligned(
        self,
        plan: AlignmentPlan,
        *,
        prefetch: int = 8,
    ) -> Iterator[AlignedBatch]:
        """Execute an existing exact alignment authority with bounded prefetch.

        The API deliberately accepts an ``AlignmentPlan`` rather than modality
        names. Native execution revalidates the plan against the opened manifest,
        freshly re-hashes complete declared dataset content, and rejects stale,
        tampered, truncated, or non-exact plans before returning the stream.
        """

        if not isinstance(plan, AlignmentPlan):
            raise TypeError("stream_aligned requires an AlignmentPlan returned by plan_aligned")
        if prefetch <= 0:
            raise ValueError("prefetch must be at least one")
        native_stream = self._native_dataset.stream_aligned(
            plan=plan._native_plan,
            prefetch=prefetch,
        )
        return (AlignedBatch(batch) for batch in native_stream)

    def to_orion_lineage(
'''
text = replace_once(text, method_marker, stream_method, "python stream_aligned")
path.write_text(text)


# Public namespace
path = Path("packages/neuros/src/neuros/__init__.py")
text = path.read_text()
text = replace_once(
    text,
    "from neuros.dataset import (  # noqa: E402,F401\n    AlignmentPlan,\n",
    "from neuros.dataset import (  # noqa: E402,F401\n    AlignedBatch,\n    AlignmentPlan,\n",
    "public import",
)
text = replace_once(
    text,
    "__all__ = [\n    \"AlignmentPlan\",\n",
    "__all__ = [\n    \"AlignedBatch\",\n    \"AlignmentPlan\",\n",
    "public all",
)
path.write_text(text)


# Exact-head installed-wheel execution smoke in canonical runtime CI.
path = Path(".github/workflows/rust-runtime-ci.yml")
text = path.read_text()
native_marker = "              # Preserve the promoted v0 zero-copy + provenance path after planning.\n"
native_smoke = '''              # Execute the exact plan itself. The executor must consume the
              # qualified plan, preserve canonical modality order, and carry the
              # plan/content/manifest identities on every batch.
              aligned_stream = dataset.stream_aligned(plan=plan, prefetch=2)
              first_batch = next(aligned_stream)
              assert first_batch.plan_sha256 == plan.sha256
              assert first_batch.dataset_content_sha256 == declared_dataset_sha256
              assert first_batch.manifest_sha256 == dataset.manifest_sha256
              assert first_batch.window_index == 0
              assert first_batch.start_ns == 0
              assert first_batch.end_ns == 4_000_000_000
              first_windows = first_batch.windows
              assert [window.modality for window in first_windows] == ["behavior", "fmri"]
              assert [(window.start_frame, window.end_frame_exclusive) for window in first_windows] == [
                  (0, 8),
                  (0, 2),
              ]
              assert [window.shape for window in first_windows] == [[8, 1], [2, 4]]
              assert all(window.verified_dataset_content_sha256 == declared_dataset_sha256 for window in first_windows)
              assert all(len(window.to_arrow()) == 8 for window in first_windows)

              last_batch = first_batch
              for candidate in aligned_stream:
                  last_batch = candidate
              assert last_batch.window_index == 8
              assert last_batch.start_ns == 16_000_000_000
              assert last_batch.end_ns == 20_000_000_000
              last_windows = last_batch.windows
              assert [(window.start_frame, window.end_frame_exclusive) for window in last_windows] == [
                  (32, 40),
                  (8, 10),
              ]
              assert all(len(window.to_arrow()) == 8 for window in last_windows)

'''
text = replace_once(text, native_marker, native_smoke + native_marker, "native wheel aligned smoke")
public_marker = '''              assert by_modality["behavior"]["source_sha256"] == behavior_sha256
              assert by_modality["behavior"]["start_frame"] == 12
              assert by_modality["behavior"]["stop_frame"] == 20
'''
public_smoke = public_marker + '''

              public_stream = study.stream_aligned(public_plan, prefetch=2)
              public_first = next(public_stream)
              assert public_first.plan_sha256 == public_plan.sha256
              assert public_first.dataset_content_sha256 == declared_dataset_sha256
              assert public_first.manifest_sha256 == study.manifest_sha256
              assert public_first.window_index == 0
              assert public_first.start_ns == 0
              assert public_first.end_ns == 4_000_000_000
              assert tuple(window.modality for window in public_first.windows) == ("behavior", "fmri")
              assert public_first.behavior.start_frame == 0
              assert public_first.behavior.end_frame_exclusive == 8
              assert public_first.fmri.start_frame == 0
              assert public_first.fmri.end_frame_exclusive == 2
              public_last = public_first
              for candidate in public_stream:
                  public_last = candidate
              assert public_last.window_index == 8
              assert public_last.behavior.start_frame == 32
              assert public_last.behavior.end_frame_exclusive == 40
              assert public_last.fmri.start_frame == 8
              assert public_last.fmri.end_frame_exclusive == 10
'''
text = replace_once(text, public_marker, public_smoke, "public wheel aligned smoke")
path.write_text(text)
