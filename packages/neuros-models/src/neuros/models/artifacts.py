"""Portable, non-pickle state artifacts for inspectable PyTorch decoders.

The artifact format is intentionally data-only: deterministic JSON plus NumPy
``.npy`` tensors. Reading an artifact never imports or executes code named by
the artifact. A caller must provide an already-constructed compatible decoder
before restoration.

Two state identities are exposed:

``parameter_state_sha256``
    Hash of the exact PyTorch ``state_dict`` names, dtypes, shapes, and bytes.
    This preserves the hash semantics already used by longitudinal evidence.

``learning_state_sha256``
    Parameter state plus PyTorch CPU/CUDA RNG state and the trained-state gate.
    This is the stronger rollback identity for continued stochastic fine-tuning.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_KIND = "neuros-torch-decoder-state"

_CONFIG_FIELDS = (
    "n_channels",
    "n_classes",
    "learning_rate",
    "weight_decay",
    "n_epochs",
    "batch_size",
    "device_spec",
    "random_state",
    "temporal_filters",
    "depth_multiplier",
    "separable_filters",
    "temporal_kernel",
    "separable_kernel",
    "embedding_dim",
    "pool_length",
    "pool_stride",
    "n_heads",
    "n_layers",
    "feedforward_multiplier",
    "dropout",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact metadata/configuration cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            normalized_key = str(key)
            if not normalized_key.strip():
                raise ValueError("artifact mapping keys must be non-empty")
            if normalized_key in normalized:
                raise ValueError(
                    "artifact mapping keys collide after string normalization: "
                    f"{normalized_key!r}"
                )
            normalized[normalized_key] = _jsonable(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(
        "artifact values must be deterministic JSON-compatible primitives; "
        f"got {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    normalized = _jsonable(value)
    if isinstance(normalized, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in normalized.items()}
        )
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly_array(name: str, value: Any, *, require_uint8: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{name} cannot use object dtype")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"{name} must use a numeric or boolean dtype")
    if require_uint8 and array.dtype != np.uint8:
        raise TypeError(f"{name} must use uint8 dtype")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _tensor_content_sha256(name: str, array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"neuros.tensor.v1\0")
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def parameter_state_sha256_from_tensors(tensors: Mapping[str, np.ndarray]) -> str:
    """Hash tensors with the exact legacy longitudinal state-hash semantics."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        array = np.ascontiguousarray(np.asarray(tensors[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def learning_state_sha256(
    *,
    parameter_state_sha256: str,
    cpu_rng_state: np.ndarray,
    cuda_rng_states: Sequence[np.ndarray],
    is_trained: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"neuros.torch-learning-state.v1\0")
    digest.update(parameter_state_sha256.encode("ascii"))
    digest.update(b"\0cpu\0")
    cpu = np.ascontiguousarray(cpu_rng_state)
    digest.update(cpu.tobytes(order="C"))
    for index, state in enumerate(cuda_rng_states):
        digest.update(f"\0cuda:{index}\0".encode("ascii"))
        digest.update(np.ascontiguousarray(state).tobytes(order="C"))
    digest.update(b"\0trained\0")
    digest.update(b"1" if is_trained else b"0")
    return digest.hexdigest()


def resolved_torch_decoder_config(model: Any) -> dict[str, Any]:
    """Return the explicit constructor-relevant configuration we can validate."""

    result: dict[str, Any] = {}
    for name in _CONFIG_FIELDS:
        if not hasattr(model, name):
            continue
        value = getattr(model, name)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, bool, int, float)) or value is None:
            result[name] = _jsonable(value)
    return result


def _state_arrays(model: Any) -> dict[str, np.ndarray]:
    module = model.analysis_model()
    state = module.state_dict()
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        arrays[name] = _readonly_array(f"state_dict[{name!r}]", tensor.numpy())
    if not arrays:
        raise ValueError("decoder state_dict is empty")
    return arrays


def torch_parameter_state_sha256(model: Any) -> str:
    """Hash a live decoder using the maintained longitudinal parameter identity."""

    return parameter_state_sha256_from_tensors(_state_arrays(model))


@dataclass(frozen=True, slots=True)
class TorchDecoderStateSnapshot:
    """Exact data-only snapshot of one inspectable PyTorch decoder state."""

    model_type: str
    model_version: str
    resolved_config: Mapping[str, Any]
    analysis_manifest_fingerprint: str
    tensors: Mapping[str, np.ndarray]
    cpu_rng_state: np.ndarray
    cuda_rng_states: tuple[np.ndarray, ...] = ()
    is_trained: bool = False
    training_history: tuple[Mapping[str, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parameter_state_sha256: str | None = None
    learning_state_sha256: str | None = None
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"TorchDecoderStateSnapshot schema_version must be {ARTIFACT_SCHEMA_VERSION}"
            )
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("model_type must be a non-empty string")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        if (
            not isinstance(self.analysis_manifest_fingerprint, str)
            or not self.analysis_manifest_fingerprint.strip()
        ):
            raise ValueError("analysis_manifest_fingerprint must be a non-empty string")
        if not isinstance(self.tensors, Mapping) or not self.tensors:
            raise ValueError("tensors must be a non-empty mapping")

        config = _freeze_json(self.resolved_config)
        metadata = _freeze_json(self.metadata)
        if not isinstance(config, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("resolved_config and metadata must be mappings")

        normalized_tensors: dict[str, np.ndarray] = {}
        for raw_name, value in self.tensors.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError("tensor names must be non-empty strings")
            if raw_name in normalized_tensors:
                raise ValueError(f"duplicate tensor name {raw_name!r}")
            normalized_tensors[raw_name] = _readonly_array(
                f"tensors[{raw_name!r}]", value
            )

        cpu = _readonly_array("cpu_rng_state", self.cpu_rng_state, require_uint8=True)
        cuda = tuple(
            _readonly_array(f"cuda_rng_states[{index}]", value, require_uint8=True)
            for index, value in enumerate(self.cuda_rng_states)
        )
        history: list[Mapping[str, float]] = []
        for index, row in enumerate(self.training_history):
            if not isinstance(row, Mapping):
                raise TypeError(f"training_history[{index}] must be a mapping")
            normalized_row: dict[str, float] = {}
            for key, value in row.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("training-history keys must be non-empty strings")
                if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                    raise ValueError(f"training_history[{index}][{key!r}] must be numeric")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"training_history[{index}][{key!r}] must be finite")
                normalized_row[key] = number
            history.append(MappingProxyType(normalized_row))

        parameter_hash = parameter_state_sha256_from_tensors(normalized_tensors)
        if self.parameter_state_sha256 is not None and self.parameter_state_sha256 != parameter_hash:
            raise ValueError("parameter_state_sha256 does not match snapshot tensors")
        learning_hash = learning_state_sha256(
            parameter_state_sha256=parameter_hash,
            cpu_rng_state=cpu,
            cuda_rng_states=cuda,
            is_trained=bool(self.is_trained),
        )
        if self.learning_state_sha256 is not None and self.learning_state_sha256 != learning_hash:
            raise ValueError("learning_state_sha256 does not match snapshot learning state")

        object.__setattr__(self, "model_type", self.model_type.strip())
        object.__setattr__(self, "model_version", self.model_version.strip())
        object.__setattr__(self, "resolved_config", config)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "tensors", MappingProxyType(normalized_tensors))
        object.__setattr__(self, "cpu_rng_state", cpu)
        object.__setattr__(self, "cuda_rng_states", cuda)
        object.__setattr__(self, "training_history", tuple(history))
        object.__setattr__(self, "is_trained", bool(self.is_trained))
        object.__setattr__(self, "parameter_state_sha256", parameter_hash)
        object.__setattr__(self, "learning_state_sha256", learning_hash)

    @property
    def snapshot_fingerprint(self) -> str:
        return _canonical_sha256(self.manifest(include_fingerprint=False))

    def manifest(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": ARTIFACT_KIND,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "resolved_config": _thaw_json(self.resolved_config),
            "analysis_manifest_fingerprint": self.analysis_manifest_fingerprint,
            "is_trained": self.is_trained,
            "parameter_state_sha256": self.parameter_state_sha256,
            "learning_state_sha256": self.learning_state_sha256,
            "tensors": {
                name: {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "content_sha256": _tensor_content_sha256(name, array),
                }
                for name, array in sorted(self.tensors.items())
            },
            "cpu_rng_state": {
                "dtype": str(self.cpu_rng_state.dtype),
                "shape": list(self.cpu_rng_state.shape),
                "content_sha256": _tensor_content_sha256("cpu_rng_state", self.cpu_rng_state),
            },
            "cuda_rng_states": [
                {
                    "index": index,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "content_sha256": _tensor_content_sha256(
                        f"cuda_rng_states[{index}]", array
                    ),
                }
                for index, array in enumerate(self.cuda_rng_states)
            ],
            "training_history": [dict(row) for row in self.training_history],
            "metadata": _thaw_json(self.metadata),
        }
        if include_fingerprint:
            payload["snapshot_fingerprint"] = self.snapshot_fingerprint
        return payload


def snapshot_torch_decoder_state(
    model: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> TorchDecoderStateSnapshot:
    torch, _ = model._torch()
    arrays = _state_arrays(model)
    cpu_rng = torch.get_rng_state().detach().cpu().numpy()
    cuda_rng: tuple[np.ndarray, ...] = ()
    if torch.cuda.is_available():
        cuda_rng = tuple(
            state.detach().cpu().numpy() for state in torch.cuda.get_rng_state_all()
        )
    history = tuple(dict(row) for row in getattr(model, "training_history", ()))
    return TorchDecoderStateSnapshot(
        model_type=type(model).__name__,
        model_version=str(model.model_version),
        resolved_config=resolved_torch_decoder_config(model),
        analysis_manifest_fingerprint=model.analysis_manifest().fingerprint(),
        tensors=arrays,
        cpu_rng_state=cpu_rng,
        cuda_rng_states=cuda_rng,
        is_trained=bool(model.is_trained),
        training_history=history,
        metadata={} if metadata is None else metadata,
    )


def _validate_restore_target(model: Any, snapshot: TorchDecoderStateSnapshot) -> None:
    if type(model).__name__ != snapshot.model_type:
        raise ValueError(
            f"model type mismatch: target={type(model).__name__}, snapshot={snapshot.model_type}"
        )
    if str(model.model_version) != snapshot.model_version:
        raise ValueError("model version differs from snapshot")
    if resolved_torch_decoder_config(model) != _thaw_json(snapshot.resolved_config):
        raise ValueError("decoder configuration differs from snapshot")
    if model.analysis_manifest().fingerprint() != snapshot.analysis_manifest_fingerprint:
        raise ValueError("analysis-manifest identity differs from snapshot")

    live = model.analysis_model().state_dict()
    if set(live) != set(snapshot.tensors):
        raise ValueError("state_dict key set differs from snapshot")
    for name in sorted(live):
        target = live[name].detach().cpu().contiguous().numpy()
        source = snapshot.tensors[name]
        if target.shape != source.shape:
            raise ValueError(f"state tensor {name!r} shape differs from snapshot")
        if target.dtype != source.dtype:
            raise ValueError(f"state tensor {name!r} dtype differs from snapshot")

    torch, _ = model._torch()
    if snapshot.cuda_rng_states:
        if not torch.cuda.is_available():
            raise ValueError("snapshot contains CUDA RNG state but CUDA is unavailable")
        if len(snapshot.cuda_rng_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from snapshot RNG authority")


def restore_torch_decoder_state(model: Any, snapshot: TorchDecoderStateSnapshot) -> None:
    """Restore an exact compatible state after validating all geometry first."""

    if not isinstance(snapshot, TorchDecoderStateSnapshot):
        raise TypeError("snapshot must be a TorchDecoderStateSnapshot")
    _validate_restore_target(model, snapshot)
    torch, _ = model._torch()
    module = model.analysis_model()
    live = module.state_dict()

    prepared: dict[str, Any] = {}
    for name in sorted(live):
        prepared[name] = torch.as_tensor(
            np.array(snapshot.tensors[name], copy=True),
            dtype=live[name].dtype,
            device=live[name].device,
        )

    # No live mutation occurs before all compatibility checks and tensor
    # conversions have succeeded.
    module.load_state_dict(prepared, strict=True)
    torch.set_rng_state(
        torch.as_tensor(np.array(snapshot.cpu_rng_state, copy=True), dtype=torch.uint8)
    )
    if snapshot.cuda_rng_states:
        torch.cuda.set_rng_state_all(
            [
                torch.as_tensor(np.array(value, copy=True), dtype=torch.uint8)
                for value in snapshot.cuda_rng_states
            ]
        )
    model.is_trained = snapshot.is_trained
    model.training_history = [dict(row) for row in snapshot.training_history]

    restored = snapshot_torch_decoder_state(model, metadata=_thaw_json(snapshot.metadata))
    if restored.parameter_state_sha256 != snapshot.parameter_state_sha256:
        raise RuntimeError("restored parameter state does not match snapshot SHA-256")
    if restored.learning_state_sha256 != snapshot.learning_state_sha256:
        raise RuntimeError("restored learning state does not match snapshot SHA-256")


def _write_npy(path: Path, array: np.ndarray) -> str:
    np.save(path, np.asarray(array), allow_pickle=False)
    return _file_sha256(path)


def write_torch_decoder_artifact(
    model_or_snapshot: Any,
    output: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a deterministic data-only decoder artifact directory."""

    snapshot = (
        model_or_snapshot
        if isinstance(model_or_snapshot, TorchDecoderStateSnapshot)
        else snapshot_torch_decoder_state(model_or_snapshot, metadata=metadata)
    )
    root = Path(output)
    if root.exists():
        if not overwrite and any(root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty artifact directory {root}")
        if overwrite:
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    tensor_dir = root / "tensors"
    tensor_dir.mkdir()

    manifest = snapshot.manifest(include_fingerprint=True)
    tensor_files: dict[str, Any] = {}
    for index, (name, array) in enumerate(sorted(snapshot.tensors.items())):
        filename = f"{index:04d}.npy"
        file_sha = _write_npy(tensor_dir / filename, array)
        tensor_files[name] = {"file": f"tensors/{filename}", "file_sha256": file_sha}

    cpu_file = "cpu_rng.npy"
    cpu_file_sha = _write_npy(root / cpu_file, snapshot.cpu_rng_state)
    cuda_files: list[dict[str, Any]] = []
    for index, array in enumerate(snapshot.cuda_rng_states):
        filename = f"cuda_rng_{index:04d}.npy"
        cuda_files.append(
            {
                "index": index,
                "file": filename,
                "file_sha256": _write_npy(root / filename, array),
            }
        )

    manifest["tensor_files"] = tensor_files
    manifest["cpu_rng_file"] = {"file": cpu_file, "file_sha256": cpu_file_sha}
    manifest["cuda_rng_files"] = cuda_files
    manifest_without_artifact_hash = dict(manifest)
    manifest["artifact_manifest_sha256"] = _canonical_sha256(manifest_without_artifact_hash)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return root


def _load_npy_checked(root: Path, file_entry: Mapping[str, Any], *, name: str) -> np.ndarray:
    relative = file_entry.get("file")
    expected_sha = file_entry.get("file_sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{name} artifact file path is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} artifact file escapes artifact directory") from exc
    if not path.is_file():
        raise ValueError(f"{name} artifact file is missing")
    if not isinstance(expected_sha, str) or _file_sha256(path) != expected_sha:
        raise ValueError(f"{name} artifact file SHA-256 mismatch")
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{name} artifact file is not a valid NumPy array") from exc


def read_torch_decoder_artifact(path: str | Path) -> TorchDecoderStateSnapshot:
    """Read and verify a data-only artifact without executing model code."""

    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("artifact manifest.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("artifact manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be a JSON object")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported torch decoder artifact schema_version")
    if manifest.get("kind") != ARTIFACT_KIND:
        raise ValueError("unsupported torch decoder artifact kind")

    expected_manifest_sha = manifest.get("artifact_manifest_sha256")
    without_hash = dict(manifest)
    without_hash.pop("artifact_manifest_sha256", None)
    if not isinstance(expected_manifest_sha, str) or _canonical_sha256(without_hash) != expected_manifest_sha:
        raise ValueError("artifact_manifest_sha256 does not match manifest content")

    specs = manifest.get("tensors")
    files = manifest.get("tensor_files")
    if not isinstance(specs, dict) or not isinstance(files, dict) or set(specs) != set(files):
        raise ValueError("artifact tensor specs/files are incomplete")
    tensors: dict[str, np.ndarray] = {}
    for name in sorted(specs):
        if not isinstance(name, str) or not name:
            raise ValueError("artifact tensor names must be non-empty strings")
        spec = specs[name]
        entry = files[name]
        if not isinstance(spec, dict) or not isinstance(entry, dict):
            raise ValueError(f"artifact tensor {name!r} spec/file entry is invalid")
        array = _load_npy_checked(root, entry, name=f"tensor {name!r}")
        if str(array.dtype) != spec.get("dtype") or list(array.shape) != spec.get("shape"):
            raise ValueError(f"artifact tensor {name!r} dtype/shape mismatch")
        if _tensor_content_sha256(name, array) != spec.get("content_sha256"):
            raise ValueError(f"artifact tensor {name!r} content SHA-256 mismatch")
        tensors[name] = array

    cpu_entry = manifest.get("cpu_rng_file")
    cpu_spec = manifest.get("cpu_rng_state")
    if not isinstance(cpu_entry, dict) or not isinstance(cpu_spec, dict):
        raise ValueError("artifact CPU RNG metadata is incomplete")
    cpu = _load_npy_checked(root, cpu_entry, name="CPU RNG")
    if str(cpu.dtype) != cpu_spec.get("dtype") or list(cpu.shape) != cpu_spec.get("shape"):
        raise ValueError("artifact CPU RNG dtype/shape mismatch")
    if _tensor_content_sha256("cpu_rng_state", cpu) != cpu_spec.get("content_sha256"):
        raise ValueError("artifact CPU RNG content SHA-256 mismatch")

    cuda_specs = manifest.get("cuda_rng_states")
    cuda_files = manifest.get("cuda_rng_files")
    if not isinstance(cuda_specs, list) or not isinstance(cuda_files, list):
        raise ValueError("artifact CUDA RNG metadata must be lists")
    if len(cuda_specs) != len(cuda_files):
        raise ValueError("artifact CUDA RNG specs/files length mismatch")
    cuda: list[np.ndarray] = []
    for index, (spec, entry) in enumerate(zip(cuda_specs, cuda_files, strict=True)):
        if not isinstance(spec, dict) or not isinstance(entry, dict):
            raise ValueError("artifact CUDA RNG entry is invalid")
        if spec.get("index") != index or entry.get("index") != index:
            raise ValueError("artifact CUDA RNG indices are not canonical")
        array = _load_npy_checked(root, entry, name=f"CUDA RNG {index}")
        if str(array.dtype) != spec.get("dtype") or list(array.shape) != spec.get("shape"):
            raise ValueError(f"artifact CUDA RNG {index} dtype/shape mismatch")
        if _tensor_content_sha256(f"cuda_rng_states[{index}]", array) != spec.get("content_sha256"):
            raise ValueError(f"artifact CUDA RNG {index} content SHA-256 mismatch")
        cuda.append(array)

    snapshot = TorchDecoderStateSnapshot(
        model_type=manifest.get("model_type"),
        model_version=manifest.get("model_version"),
        resolved_config=manifest.get("resolved_config"),
        analysis_manifest_fingerprint=manifest.get("analysis_manifest_fingerprint"),
        tensors=tensors,
        cpu_rng_state=cpu,
        cuda_rng_states=tuple(cuda),
        is_trained=manifest.get("is_trained"),
        training_history=tuple(manifest.get("training_history", ())),
        metadata=manifest.get("metadata", {}),
        parameter_state_sha256=manifest.get("parameter_state_sha256"),
        learning_state_sha256=manifest.get("learning_state_sha256"),
        schema_version=manifest.get("schema_version"),
    )
    expected_snapshot_fingerprint = manifest.get("snapshot_fingerprint")
    if expected_snapshot_fingerprint != snapshot.snapshot_fingerprint:
        raise ValueError("snapshot_fingerprint does not match reconstructed artifact")
    return snapshot
