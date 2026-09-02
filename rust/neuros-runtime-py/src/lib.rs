use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field};
use neuros_runtime::{Dataset, StreamSelector, WindowHandle, WindowSpec, WindowStream};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

fn runtime_error(error: neuros_runtime::RuntimeError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pyclass]
struct NativeDataset {
    inner: Arc<Dataset>,
}

#[pymethods]
impl NativeDataset {
    #[staticmethod]
    fn open(root: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: Dataset::open(root).map_err(runtime_error)?,
        })
    }

    #[getter]
    fn dataset_id(&self) -> String {
        self.inner.manifest().dataset_id.clone()
    }

    #[getter]
    fn manifest_sha256(&self) -> String {
        self.inner.manifest_sha256().to_owned()
    }

    #[getter]
    fn record_count(&self) -> usize {
        self.inner.manifest().records.len()
    }

    #[pyo3(signature = (*, subjects=None, modalities=None, window, stride=None, prefetch=8))]
    fn stream(
        &self,
        subjects: Option<Vec<String>>,
        modalities: Option<Vec<String>>,
        window: usize,
        stride: Option<usize>,
        prefetch: usize,
    ) -> PyResult<NativeWindowStream> {
        let spec = WindowSpec::new(window, stride.unwrap_or(window)).map_err(runtime_error)?;
        let selector = StreamSelector {
            subjects: subjects.unwrap_or_default(),
            modalities: modalities.unwrap_or_default(),
        };
        let stream = self
            .inner
            .stream(selector, spec, prefetch)
            .map_err(runtime_error)?;
        Ok(NativeWindowStream {
            inner: Mutex::new(stream),
        })
    }
}

#[pyclass]
struct NativeWindowStream {
    inner: Mutex<WindowStream>,
}

#[pymethods]
impl NativeWindowStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<NativeWindow>> {
        let next = py.detach(|| {
            self.inner
                .lock()
                .map_err(|_| "native stream lock was poisoned".to_owned())
                .map(|mut stream| stream.next())
        });
        let next = next.map_err(PyRuntimeError::new_err)?;
        match next {
            None => Ok(None),
            Some(Ok(window)) => Ok(Some(NativeWindow { inner: window })),
            Some(Err(error)) => Err(runtime_error(error)),
        }
    }
}

#[pyclass]
struct NativeWindow {
    inner: WindowHandle,
}

#[pymethods]
impl NativeWindow {
    #[getter]
    fn record_id(&self) -> String {
        self.inner.record_id().to_owned()
    }

    #[getter]
    fn subject(&self) -> String {
        self.inner.subject().to_owned()
    }

    #[getter]
    fn modality(&self) -> String {
        self.inner.modality().to_owned()
    }

    #[getter]
    fn start_frame(&self) -> usize {
        self.inner.start_frame()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    #[getter]
    fn sampling_hz(&self) -> Option<f64> {
        self.inner.sampling_hz()
    }

    #[getter]
    fn manifest_sha256(&self) -> String {
        self.inner.manifest_sha256().to_owned()
    }

    #[getter]
    fn source_size_bytes(&self) -> u64 {
        self.inner.source_size_bytes()
    }

    /// Return an arro3 Array backed directly by the memory-mapped source bytes.
    /// The Arrow allocation retains the mmap owner, so the view remains valid even
    /// after this NativeWindow instance is dropped.
    fn to_arrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let values = self.inner.arrow_values().map_err(runtime_error)?;
        let array: ArrayRef = Arc::new(values);
        let field = Arc::new(Field::new("values", DataType::Float32, false));
        PyArray::new(array, field).into_arro3(py)
    }

    /// Convenience adapter for environments that already carry PyArrow.
    /// This uses Arrow's capsule interface and does not copy the underlying values.
    fn to_pyarrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let values = self.inner.arrow_values().map_err(runtime_error)?;
        let array: ArrayRef = Arc::new(values);
        let field = Arc::new(Field::new("values", DataType::Float32, false));
        PyArray::new(array, field).into_pyarrow(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeWindow(subject={:?}, modality={:?}, start_frame={}, shape={:?})",
            self.inner.subject(),
            self.inner.modality(),
            self.inner.start_frame(),
            self.inner.shape(),
        )
    }
}

#[pyfunction]
fn runtime_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn require_single_modality(modalities: Vec<String>) -> PyResult<()> {
    if modalities.len() > 1 {
        return Err(PyValueError::new_err(
            "v0 native streaming does not infer cross-modal synchronization; select one modality until an explicit clock/resampling policy is supplied",
        ));
    }
    Ok(())
}

#[pymodule]
fn neuros_runtime_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeDataset>()?;
    module.add_class::<NativeWindowStream>()?;
    module.add_class::<NativeWindow>()?;
    module.add_function(wrap_pyfunction!(runtime_version, module)?)?;
    module.add_function(wrap_pyfunction!(require_single_modality, module)?)?;
    Ok(())
}
