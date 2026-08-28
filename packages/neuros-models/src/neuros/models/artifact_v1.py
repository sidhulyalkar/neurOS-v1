"""Hardened Model Artifact v1 implementation.

A promoted neurOS decoder artifact is an immutable-by-contract directory of
canonical JSON provenance plus tensor-only safetensors weights. Artifact content
never supplies an import path and loading never executes pickle.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from neuros.contracts import DecoderCapabilities, DecoderOutput, StreamDescriptor
from neuros.models.base_model import BaseModel
from neuros.models.torch_base import TorchDecoderModel

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_NAME = "manifest.json"
_WEIGHTS_NAME = "weights.safetensors"
_WEIGHTS_FORMAT = "safetensors.torch"
_PROBABILITY_SEMANTICS = {
    "uncalibrated_softmax",
    "calibrated_probability",
    "unavailable",
}


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    value = value.strip().lower()
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return value


def _optional_sha256(name: str, value: Any) -> str | None:
    return None if value is None else _require_sha256(name, value)


def _sha_tuple(name: str, values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of SHA-256 values")
    try:
        result = tuple(_require_sha256(name, value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of SHA-256 values") from exc
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate SHA-256 values")
    return result


def _string_tuple(name: str, values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    return tuple(value.strip() for value in result)


def _canonical_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("model artifact metadata cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("model artifact metadata cannot contain object arrays")
        return _canonical_json(value.tolist())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("model artifact mapping keys must be non-empty strings")
            key = key.strip()
            if key in normalized:
                raise ValueError("model artifact mapping keys collide after normalization")
            normalized[key] = _canonical_json(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("unordered sets are not valid model artifact metadata")
    raise TypeError(
        "model artifact metadata must be deterministic JSON-compatible values; "
        f"got {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    normalized = _canonical_json(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _canonical_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_load(path: Path) -> Mapping[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in model artifact manifest")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value!r} is not permitted")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model artifact manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("model artifact manifest root must be a JSON object")
    return value


def _exact_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer without coercion")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _finite_positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric without coercion")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


@dataclass(frozen=True, slots=True)
class ModelInputContract:
    """Exact array assumptions and optional stream-descriptor authority."""

    axes: tuple[str, ...]
    shape: tuple[int | None, ...]
    dtype: str = "float32"
    channel_names: tuple[str, ...] = ()
    sample_rate_hz: float | None = None
    signal_unit: str | None = None
    stream_descriptor_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ModelInputContract schema_version must be 1")
        axes = _string_tuple("axes", self.axes)
        if not axes:
            raise ValueError("axes must be non-empty")
        if len(set(axes)) != len(axes):
            raise ValueError("axes cannot contain duplicate names")
        try:
            raw_shape = tuple(self.shape)
        except TypeError as exc:
            raise ValueError("shape must be a sequence of positive integers or None") from exc
        if len(raw_shape) != len(axes):
            raise ValueError("shape must align one-to-one with axes")
        shape: list[int | None] = []
        for index, value in enumerate(raw_shape):
            shape.append(None if value is None else _exact_int(f"shape[{index}]", value, minimum=1))
        dtype = _nonempty("dtype", self.dtype)
        try:
            np.dtype(dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported numpy dtype {dtype!r}") from exc
        channels = _string_tuple("channel_names", self.channel_names)
        if len(set(channels)) != len(channels):
            raise ValueError("channel_names cannot contain duplicates")
        if channels:
            if "channel" not in axes:
                raise ValueError("channel_names require a 'channel' axis")
            channel_size = shape[axes.index("channel")]
            if channel_size is None:
                raise ValueError("channel_names require a fixed channel dimension")
            if channel_size != len(channels):
                raise ValueError("channel_names length must equal the channel dimension")
        sample_rate = None if self.sample_rate_hz is None else _finite_positive_float(
            "sample_rate_hz", self.sample_rate_hz
        )
        signal_unit = None if self.signal_unit is None else _nonempty("signal_unit", self.signal_unit)
        descriptor_sha = _optional_sha256("stream_descriptor_sha256", self.stream_descriptor_sha256)
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "channel_names", channels)
        object.__setattr__(self, "sample_rate_hz", sample_rate)
        object.__setattr__(self, "signal_unit", signal_unit)
        object.__setattr__(self, "stream_descriptor_sha256", descriptor_sha)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "axes": list(self.axes),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "channel_names": list(self.channel_names),
            "sample_rate_hz": self.sample_rate_hz,
            "signal_unit": self.signal_unit,
            "stream_descriptor_sha256": self.stream_descriptor_sha256,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelInputContract":
        if not isinstance(payload, Mapping):
            raise TypeError("input_contract must be a mapping")
        allowed = {
            "schema_version", "axes", "shape", "dtype", "channel_names",
            "sample_rate_hz", "signal_unit", "stream_descriptor_sha256", "metadata",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"input_contract contains unknown fields {sorted(unknown)}")
        return cls(
            axes=tuple(payload.get("axes", ())),
            shape=tuple(payload.get("shape", ())),
            dtype=payload.get("dtype", "float32"),
            channel_names=tuple(payload.get("channel_names", ())),
            sample_rate_hz=payload.get("sample_rate_hz"),
            signal_unit=payload.get("signal_unit"),
            stream_descriptor_sha256=payload.get("stream_descriptor_sha256"),
            metadata=payload.get("metadata", {}),
            schema_version=_exact_int("input_contract schema_version", payload.get("schema_version", 1)),
        )

    def validate_array(self, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim != len(self.shape):
            raise ValueError(
                f"artifact input rank mismatch: expected {len(self.shape)}, received {array.ndim}"
            )
        for index, expected in enumerate(self.shape):
            if expected is not None and array.shape[index] != expected:
                raise ValueError(
                    f"artifact input shape mismatch at axis {self.axes[index]!r}: "
                    f"expected {expected}, received {array.shape[index]}"
                )
        expected_dtype = np.dtype(self.dtype)
        if array.dtype != expected_dtype:
            raise ValueError(
                f"artifact input dtype mismatch: expected {expected_dtype}, received {array.dtype}"
            )
        return array

    def validate_stream_descriptor(self, descriptor: StreamDescriptor) -> None:
        """Bind array assumptions to the canonical neurOS stream identity."""

        if not isinstance(descriptor, StreamDescriptor):
            raise TypeError("descriptor must be a StreamDescriptor")
        if self.stream_descriptor_sha256 is not None:
            if descriptor.fingerprint() != self.stream_descriptor_sha256:
                raise ValueError("stream descriptor SHA-256 does not match artifact input authority")
        if self.channel_names and tuple(descriptor.channel_names) != self.channel_names:
            raise ValueError("stream descriptor channel_names do not match artifact input authority")
        if self.sample_rate_hz is not None and not math.isclose(
            descriptor.nominal_sample_rate_hz,
            self.sample_rate_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("stream descriptor sample rate does not match artifact input authority")
        if self.signal_unit is not None and descriptor.units:
            if any(unit != self.signal_unit for unit in descriptor.units):
                raise ValueError("stream descriptor units do not match artifact input authority")


@dataclass(frozen=True, slots=True)
class ModelOutputContract:
    """Explicit semantic meaning of decoder scores and probabilities."""

    class_labels: tuple[str, ...]
    task: str = "classification"
    score_semantics: str = "class_logits"
    probability_semantics: str = "uncalibrated_softmax"
    uncertainty_semantics: str = "none"
    probability_calibration_method: str | None = None
    probability_calibration_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ModelOutputContract schema_version must be 1")
        labels = _string_tuple("class_labels", self.class_labels)
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("class_labels must contain at least two unique labels")
        task = _nonempty("task", self.task)
        score_semantics = _nonempty("score_semantics", self.score_semantics)
        probability_semantics = _nonempty("probability_semantics", self.probability_semantics)
        if probability_semantics not in _PROBABILITY_SEMANTICS:
            raise ValueError(
                f"probability_semantics must be one of {sorted(_PROBABILITY_SEMANTICS)}"
            )
        uncertainty_semantics = _nonempty("uncertainty_semantics", self.uncertainty_semantics)
        method = self.probability_calibration_method
        calibration_sha = self.probability_calibration_sha256
        if probability_semantics == "calibrated_probability":
            method = _nonempty("probability_calibration_method", method)
            calibration_sha = _require_sha256("probability_calibration_sha256", calibration_sha)
        elif method is not None or calibration_sha is not None:
            raise ValueError(
                "probability calibration identity may only be declared for calibrated_probability"
            )
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "class_labels", labels)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "score_semantics", score_semantics)
        object.__setattr__(self, "probability_semantics", probability_semantics)
        object.__setattr__(self, "uncertainty_semantics", uncertainty_semantics)
        object.__setattr__(self, "probability_calibration_method", method)
        object.__setattr__(self, "probability_calibration_sha256", calibration_sha)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "class_labels": list(self.class_labels),
            "task": self.task,
            "score_semantics": self.score_semantics,
            "probability_semantics": self.probability_semantics,
            "uncertainty_semantics": self.uncertainty_semantics,
            "probability_calibration_method": self.probability_calibration_method,
            "probability_calibration_sha256": self.probability_calibration_sha256,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelOutputContract":
        if not isinstance(payload, Mapping):
            raise TypeError("output_contract must be a mapping")
        allowed = {
            "schema_version", "class_labels", "task", "score_semantics",
            "probability_semantics", "uncertainty_semantics",
            "probability_calibration_method", "probability_calibration_sha256", "metadata",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"output_contract contains unknown fields {sorted(unknown)}")
        return cls(
            class_labels=tuple(payload.get("class_labels", ())),
            task=payload.get("task", "classification"),
            score_semantics=payload.get("score_semantics", "class_logits"),
            probability_semantics=payload.get("probability_semantics", "uncalibrated_softmax"),
            uncertainty_semantics=payload.get("uncertainty_semantics", "none"),
            probability_calibration_method=payload.get("probability_calibration_method"),
            probability_calibration_sha256=payload.get("probability_calibration_sha256"),
            metadata=payload.get("metadata", {}),
            schema_version=_exact_int("output_contract schema_version", payload.get("schema_version", 1)),
        )


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    """Canonical identity and provenance for one promoted decoder artifact."""

    artifact_id: str
    factory_id: str
    model_type: str
    backend: str
    backend_version: str
    model_config: Mapping[str, Any]
    input_contract: ModelInputContract
    output_contract: ModelOutputContract
    weights_sha256: str
    interpretability_manifest: Mapping[str, Any]
    interpretability_manifest_sha256: str
    git_sha: str
    package_versions: Mapping[str, str]
    training_authority_sha256s: tuple[str, ...] = ()
    evaluation_authority_sha256s: tuple[str, ...] = ()
    preprocessing_state_sha256s: tuple[str, ...] = ()
    calibration_state_sha256s: tuple[str, ...] = ()
    scientific_study_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    weights_format: str = _WEIGHTS_FORMAT
    weights_file: str = _WEIGHTS_NAME
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ModelArtifactManifest schema_version must be 1")
        for name in ("artifact_id", "factory_id", "model_type", "backend", "backend_version"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.weights_format != _WEIGHTS_FORMAT:
            raise ValueError(f"weights_format must be {_WEIGHTS_FORMAT!r} for Model Artifact v1")
        if self.weights_file != _WEIGHTS_NAME:
            raise ValueError(f"weights_file must be {_WEIGHTS_NAME!r} for Model Artifact v1")
        if not isinstance(self.input_contract, ModelInputContract):
            raise TypeError("input_contract must be a ModelInputContract")
        if not isinstance(self.output_contract, ModelOutputContract):
            raise TypeError("output_contract must be a ModelOutputContract")
        model_config = _freeze_json(self.model_config)
        interpretability = _freeze_json(self.interpretability_manifest)
        metadata = _freeze_json(self.metadata)
        if not isinstance(model_config, Mapping) or not isinstance(interpretability, Mapping):
            raise TypeError("model_config and interpretability_manifest must be mappings")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        weights_sha = _require_sha256("weights_sha256", self.weights_sha256)
        interpretability_sha = _require_sha256(
            "interpretability_manifest_sha256", self.interpretability_manifest_sha256
        )
        actual_interpretability_sha = hashlib.sha256(
            json.dumps(
                _thaw_json(interpretability),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if actual_interpretability_sha != interpretability_sha:
            raise ValueError("interpretability_manifest_sha256 does not match embedded manifest")
        if not isinstance(self.git_sha, str) or not _GIT_SHA_RE.fullmatch(self.git_sha):
            raise ValueError("git_sha must be a 40-character lowercase Git commit SHA")
        if not isinstance(self.package_versions, Mapping) or not self.package_versions:
            raise ValueError("package_versions must be a non-empty mapping")
        versions: dict[str, str] = {}
        for raw_name, raw_version in self.package_versions.items():
            name = _nonempty("package name", raw_name)
            version = _nonempty(f"package_versions[{name!r}]", raw_version)
            if name in versions:
                raise ValueError("package_versions keys cannot duplicate after normalization")
            versions[name] = version
        object.__setattr__(self, "model_config", model_config)
        object.__setattr__(self, "interpretability_manifest", interpretability)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "weights_sha256", weights_sha)
        object.__setattr__(self, "interpretability_manifest_sha256", interpretability_sha)
        object.__setattr__(self, "package_versions", MappingProxyType(dict(sorted(versions.items()))))
        object.__setattr__(self, "training_authority_sha256s", _sha_tuple("training_authority_sha256s", self.training_authority_sha256s))
        object.__setattr__(self, "evaluation_authority_sha256s", _sha_tuple("evaluation_authority_sha256s", self.evaluation_authority_sha256s))
        object.__setattr__(self, "preprocessing_state_sha256s", _sha_tuple("preprocessing_state_sha256s", self.preprocessing_state_sha256s))
        object.__setattr__(self, "calibration_state_sha256s", _sha_tuple("calibration_state_sha256s", self.calibration_state_sha256s))
        object.__setattr__(self, "scientific_study_sha256", _optional_sha256("scientific_study_sha256", self.scientific_study_sha256))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "factory_id": self.factory_id,
            "model_type": self.model_type,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "model_config": _thaw_json(self.model_config),
            "input_contract": self.input_contract.to_dict(),
            "output_contract": self.output_contract.to_dict(),
            "weights_format": self.weights_format,
            "weights_file": self.weights_file,
            "weights_sha256": self.weights_sha256,
            "interpretability_manifest": _thaw_json(self.interpretability_manifest),
            "interpretability_manifest_sha256": self.interpretability_manifest_sha256,
            "git_sha": self.git_sha,
            "package_versions": dict(self.package_versions),
            "training_authority_sha256s": list(self.training_authority_sha256s),
            "evaluation_authority_sha256s": list(self.evaluation_authority_sha256s),
            "preprocessing_state_sha256s": list(self.preprocessing_state_sha256s),
            "calibration_state_sha256s": list(self.calibration_state_sha256s),
            "scientific_study_sha256": self.scientific_study_sha256,
            "metadata": _thaw_json(self.metadata),
        }

    @property
    def manifest_sha256(self) -> str:
        return _canonical_sha256(self._identity_payload())

    @property
    def artifact_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"neuros.model-artifact.v1\0")
        digest.update(bytes.fromhex(self.manifest_sha256))
        digest.update(bytes.fromhex(self.weights_sha256))
        return digest.hexdigest()

    @property
    def display_fingerprint(self) -> str:
        return self.artifact_sha256[:16]

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["manifest_sha256"] = self.manifest_sha256
        payload["artifact_sha256"] = self.artifact_sha256
        payload["display_fingerprint"] = self.display_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelArtifactManifest":
        if not isinstance(payload, Mapping):
            raise TypeError("model artifact manifest must be a mapping")
        required = {
            "schema_version", "artifact_id", "factory_id", "model_type", "backend",
            "backend_version", "model_config", "input_contract", "output_contract",
            "weights_format", "weights_file", "weights_sha256", "interpretability_manifest",
            "interpretability_manifest_sha256", "git_sha", "package_versions",
            "training_authority_sha256s", "evaluation_authority_sha256s",
            "preprocessing_state_sha256s", "calibration_state_sha256s",
            "scientific_study_sha256", "metadata", "manifest_sha256",
            "artifact_sha256", "display_fingerprint",
        }
        missing = required - set(payload)
        unknown = set(payload) - required
        if missing:
            raise ValueError(f"model artifact manifest is missing fields {sorted(missing)}")
        if unknown:
            raise ValueError(f"model artifact manifest contains unknown fields {sorted(unknown)}")
        value = cls(
            artifact_id=payload["artifact_id"],
            factory_id=payload["factory_id"],
            model_type=payload["model_type"],
            backend=payload["backend"],
            backend_version=payload["backend_version"],
            model_config=payload["model_config"],
            input_contract=ModelInputContract.from_dict(payload["input_contract"]),
            output_contract=ModelOutputContract.from_dict(payload["output_contract"]),
            weights_format=payload["weights_format"],
            weights_file=payload["weights_file"],
            weights_sha256=payload["weights_sha256"],
            interpretability_manifest=payload["interpretability_manifest"],
            interpretability_manifest_sha256=payload["interpretability_manifest_sha256"],
            git_sha=payload["git_sha"],
            package_versions=payload["package_versions"],
            training_authority_sha256s=tuple(payload["training_authority_sha256s"]),
            evaluation_authority_sha256s=tuple(payload["evaluation_authority_sha256s"]),
            preprocessing_state_sha256s=tuple(payload["preprocessing_state_sha256s"]),
            calibration_state_sha256s=tuple(payload["calibration_state_sha256s"]),
            scientific_study_sha256=payload["scientific_study_sha256"],
            metadata=payload["metadata"],
            schema_version=_exact_int("schema_version", payload["schema_version"]),
        )
        if _require_sha256("manifest_sha256", payload["manifest_sha256"]) != value.manifest_sha256:
            raise ValueError("manifest_sha256 does not match serialized model artifact content")
        if _require_sha256("artifact_sha256", payload["artifact_sha256"]) != value.artifact_sha256:
            raise ValueError("artifact_sha256 does not match serialized model artifact content")
        if payload["display_fingerprint"] != value.display_fingerprint:
            raise ValueError("display_fingerprint does not match artifact_sha256")
        return value


def _builtin_torch_factories() -> dict[str, type[TorchDecoderModel]]:
    from neuros.models.attention_fusion_model import AttentionFusionModel
    from neuros.models.cnn_model import CNNModel
    from neuros.models.eeg_conformer_model import EEGConformerModel
    from neuros.models.eegnet_model import EEGNetModel
    from neuros.models.lstm_model import LSTMModel
    from neuros.models.transformer_model import TransformerModel

    return {
        "neuros.attention_fusion.v1": AttentionFusionModel,
        "neuros.cnn.v1": CNNModel,
        "neuros.eeg_conformer.v1": EEGConformerModel,
        "neuros.eegnet.v1": EEGNetModel,
        "neuros.lstm.v1": LSTMModel,
        "neuros.transformer.v1": TransformerModel,
    }


def _factory_id_for_model(model: TorchDecoderModel) -> str:
    matches = [factory_id for factory_id, factory in _builtin_torch_factories().items() if type(model) is factory]
    if len(matches) != 1:
        raise ValueError(
            f"{type(model).__name__} has no unique built-in safe artifact factory; "
            "Model Artifact v1 does not fall back to arbitrary import paths"
        )
    return matches[0]


def _constructor_config(model: TorchDecoderModel) -> Mapping[str, Any]:
    signature = inspect.signature(type(model).__init__)
    config: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if name == "device":
            continue
        if hasattr(model, name):
            value = getattr(model, name)
        elif parameter.default is not inspect.Parameter.empty:
            value = parameter.default
        else:
            raise ValueError(
                f"cannot derive constructor field {name!r} from {type(model).__name__}; "
                "safe artifact export requires explicit reconstructability"
            )
        config[name] = value
    frozen = _freeze_json(config)
    if not isinstance(frozen, Mapping):
        raise TypeError("derived model configuration must be a mapping")
    return frozen


def _default_package_versions() -> Mapping[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("neuros-models", "neuros-core", "numpy", "torch", "safetensors"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"cannot promote artifact without installed distribution {distribution!r}"
            ) from exc
    return MappingProxyType(versions)


def _require_safetensors() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise ImportError(
            "Safe promoted PyTorch artifacts require safetensors. Install `neuros-models[artifact]`."
        ) from exc
    return load_file, save_file


def _artifact_entries(root: Path) -> set[str]:
    entries: set[str] = set()
    for entry in root.iterdir():
        if entry.is_symlink():
            raise ValueError("model artifact directories cannot contain symbolic links")
        if not entry.is_file():
            raise ValueError("model artifact directories can contain only regular files")
        entries.add(entry.name)
    return entries


def verify_model_artifact(path: str | Path) -> ModelArtifactManifest:
    """Verify the immutable JSON/tensor envelope without constructing a model."""

    root = Path(path)
    if root.is_symlink():
        raise ValueError("model artifact root cannot be a symbolic link")
    if not root.is_dir():
        raise FileNotFoundError(f"model artifact directory not found: {root}")
    expected = {_MANIFEST_NAME, _WEIGHTS_NAME}
    entries = _artifact_entries(root)
    if entries != expected:
        raise ValueError(f"model artifact must contain exactly {sorted(expected)}; found {sorted(entries)}")
    manifest = ModelArtifactManifest.from_dict(_strict_json_load(root / _MANIFEST_NAME))
    if _file_sha256(root / _WEIGHTS_NAME) != manifest.weights_sha256:
        raise ValueError("weights SHA-256 does not match model artifact manifest")
    return manifest


def _default_output_contract(model: TorchDecoderModel) -> ModelOutputContract:
    n_classes = getattr(model, "n_classes", None)
    if not isinstance(n_classes, int) or n_classes < 2:
        raise ValueError("safe classifier artifact export requires an integer n_classes >= 2")
    return ModelOutputContract(class_labels=tuple(str(index) for index in range(n_classes)))


def export_model_artifact(
    model: TorchDecoderModel,
    output_dir: str | Path,
    *,
    artifact_id: str,
    input_contract: ModelInputContract,
    git_sha: str,
    output_contract: ModelOutputContract | None = None,
    package_versions: Mapping[str, str] | None = None,
    training_authority_sha256s: Sequence[str] = (),
    evaluation_authority_sha256s: Sequence[str] = (),
    preprocessing_state_sha256s: Sequence[str] = (),
    calibration_state_sha256s: Sequence[str] = (),
    scientific_study_sha256: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ModelArtifactManifest:
    """Export one trained built-in PyTorch decoder without pickle."""

    if not isinstance(model, TorchDecoderModel):
        raise TypeError("Model Artifact v1 safe export currently supports TorchDecoderModel only")
    if not model.is_trained:
        raise RuntimeError("only trained models can be promoted to a Model Artifact v1")
    if not isinstance(input_contract, ModelInputContract):
        raise TypeError("input_contract must be a ModelInputContract")
    output_contract = _default_output_contract(model) if output_contract is None else output_contract
    if not isinstance(output_contract, ModelOutputContract):
        raise TypeError("output_contract must be a ModelOutputContract")
    if len(output_contract.class_labels) != getattr(model, "n_classes", -1):
        raise ValueError("output contract class_labels must match model n_classes")
    if output_contract.probability_semantics != "uncalibrated_softmax":
        raise ValueError(
            "built-in TorchDecoderModel v1 factories emit uncalibrated softmax only; "
            "a calibrated probability artifact requires a qualified calibrated factory"
        )

    torch, _nn = model._torch()
    _load_file, save_file = _require_safetensors()
    analysis_manifest = model.analysis_manifest()
    if tuple(analysis_manifest.input_axes) != input_contract.axes:
        raise ValueError("input contract axes must exactly match model interpretability manifest input_axes")
    model_config = _constructor_config(model)
    if "n_channels" in model_config and "channel" in input_contract.axes:
        channel_dim = input_contract.shape[input_contract.axes.index("channel")]
        if channel_dim != model_config["n_channels"]:
            raise ValueError("input contract channel dimension must match model n_channels")
    if hasattr(model, "total_dim") and len(input_contract.shape) >= 2:
        if input_contract.shape[1] != getattr(model, "total_dim"):
            raise ValueError("input contract feature dimension must match model total_dim")

    module = model._ensure_model()
    tensors: dict[str, Any] = {}
    for name, value in module.state_dict().items():
        if not isinstance(name, str) or not name:
            raise ValueError("state_dict keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict value {name!r} is not a tensor")
        tensors[name] = value.detach().cpu().contiguous()

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"promoted model artifacts are immutable; destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent)))
    try:
        weights_path = temporary / _WEIGHTS_NAME
        save_file(tensors, str(weights_path))
        manifest = ModelArtifactManifest(
            artifact_id=artifact_id,
            factory_id=_factory_id_for_model(model),
            model_type=type(model).__name__,
            backend="pytorch",
            backend_version=str(torch.__version__),
            model_config=model_config,
            input_contract=input_contract,
            output_contract=output_contract,
            weights_sha256=_file_sha256(weights_path),
            interpretability_manifest=analysis_manifest.to_dict(),
            interpretability_manifest_sha256=analysis_manifest.sha256(),
            git_sha=git_sha,
            package_versions=_default_package_versions() if package_versions is None else package_versions,
            training_authority_sha256s=tuple(training_authority_sha256s),
            evaluation_authority_sha256s=tuple(evaluation_authority_sha256s),
            preprocessing_state_sha256s=tuple(preprocessing_state_sha256s),
            calibration_state_sha256s=tuple(calibration_state_sha256s),
            scientific_study_sha256=scientific_study_sha256,
            metadata={} if metadata is None else metadata,
        )
        (temporary / _MANIFEST_NAME).write_text(
            json.dumps(manifest.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        verify_model_artifact(temporary)
        os.replace(temporary, destination)
        temporary = None
        return manifest
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


class ArtifactBackedDecoder(BaseModel):
    """Read-only public decoder interface loaded from a verified artifact.

    Python cannot make an object cryptographically immutable against a caller
    deliberately reaching into private attributes. The public API therefore
    prevents mutation and never exposes the live inference module. Mechanistic
    inspection receives a detached snapshot whose mutation cannot alter future
    deployment inference.
    """

    def __init__(self, model: TorchDecoderModel, manifest: ModelArtifactManifest) -> None:
        super().__init__()
        self._model = model
        self.artifact_manifest = manifest
        self.is_trained = True
        self.model_version = manifest.display_fingerprint

    @property
    def capabilities(self) -> DecoderCapabilities:
        base = self._model.capabilities
        return DecoderCapabilities(
            probabilities=base.probabilities,
            uncertainty=base.uncertainty,
            online_fit=False,
            streaming_state=base.streaming_state,
            embeddings=base.embeddings,
        )

    def _validated(self, X: np.ndarray) -> np.ndarray:
        return self.artifact_manifest.input_contract.validate_array(X)

    def validate_stream_descriptor(self, descriptor: StreamDescriptor) -> None:
        self.artifact_manifest.input_contract.validate_stream_descriptor(descriptor)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        del X, y
        raise RuntimeError("promoted model artifacts are immutable and cannot be retrained")

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        del X, y
        raise RuntimeError("promoted model artifacts are immutable and cannot be updated")

    def adapt(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("promoted model artifacts are immutable and cannot be adapted in place")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._validated(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return self._model.predict_proba(self._validated(X))

    def predict_logits(self, X: np.ndarray) -> np.ndarray | None:
        return self._model.predict_logits(self._validated(X))

    def encode(self, X: np.ndarray) -> np.ndarray | None:
        return self._model.encode(self._validated(X))

    def analysis_manifest(self) -> Any:
        return self._model.analysis_manifest()

    def analysis_model(self) -> Any:
        snapshot = copy.deepcopy(self._model.analysis_model())
        if hasattr(snapshot, "eval"):
            snapshot.eval()
        if hasattr(snapshot, "parameters"):
            for parameter in snapshot.parameters():
                parameter.requires_grad_(False)
        return snapshot

    def infer(self, X: np.ndarray) -> DecoderOutput:
        output = self._model.infer(self._validated(X))
        metadata = dict(output.metadata)
        contract = self.artifact_manifest.output_contract
        metadata.update(
            {
                "artifact_id": self.artifact_manifest.artifact_id,
                "artifact_sha256": self.artifact_manifest.artifact_sha256,
                "artifact_manifest_sha256": self.artifact_manifest.manifest_sha256,
                "interpretability_manifest_sha256": self.artifact_manifest.interpretability_manifest_sha256,
                "promoted_artifact": True,
                "class_labels": contract.class_labels,
                "score_semantics": contract.score_semantics,
                "probability_semantics": contract.probability_semantics,
                "uncertainty_semantics": contract.uncertainty_semantics,
            }
        )
        return replace(
            output,
            model_version=self.artifact_manifest.display_fingerprint,
            metadata=metadata,
        )


def _verify_environment(manifest: ModelArtifactManifest) -> None:
    for distribution, expected_version in manifest.package_versions.items():
        try:
            actual_version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"artifact requires distribution {distribution!r}={expected_version}, but it is not installed"
            ) from exc
        if actual_version != expected_version:
            raise RuntimeError(
                f"artifact requires {distribution}=={expected_version}, but the environment has {actual_version}"
            )


def load_model_artifact(path: str | Path, *, device: str = "cpu") -> ArtifactBackedDecoder:
    """Verify and load a promoted artifact through a built-in safe factory."""

    manifest = verify_model_artifact(path)
    _verify_environment(manifest)
    factory = _builtin_torch_factories().get(manifest.factory_id)
    if factory is None:
        raise ValueError(
            f"artifact factory {manifest.factory_id!r} is not a built-in safe Model Artifact v1 factory"
        )
    if factory.__name__ != manifest.model_type:
        raise ValueError("artifact model_type does not match its registered safe factory")
    config = _thaw_json(manifest.model_config)
    if not isinstance(config, dict):
        raise TypeError("model_config must decode to a mapping")
    signature = inspect.signature(factory.__init__)
    if "device" in signature.parameters:
        config["device"] = device
    model = factory(**config)
    if model.analysis_manifest().sha256() != manifest.interpretability_manifest_sha256:
        raise ValueError("installed model interpretability contract differs from the promoted artifact")
    if len(manifest.output_contract.class_labels) != getattr(model, "n_classes", -1):
        raise ValueError("artifact output contract class count differs from reconstructed model")
    if manifest.output_contract.probability_semantics != "uncalibrated_softmax":
        raise ValueError("registered v1 torch factory does not implement the declared probability semantics")

    load_file, _save_file = _require_safetensors()
    tensors = load_file(str(Path(path) / _WEIGHTS_NAME), device="cpu")
    module = model._ensure_model()
    expected_state = module.state_dict()
    if set(tensors) != set(expected_state):
        missing = sorted(set(expected_state) - set(tensors))
        extra = sorted(set(tensors) - set(expected_state))
        raise ValueError(
            "artifact tensor names do not match the registered model state; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    for name, expected in expected_state.items():
        actual = tensors[name]
        if tuple(actual.shape) != tuple(expected.shape):
            raise ValueError(
                f"artifact tensor {name!r} shape mismatch: expected {tuple(expected.shape)}, "
                f"received {tuple(actual.shape)}"
            )
        if actual.dtype != expected.dtype:
            raise ValueError(
                f"artifact tensor {name!r} dtype mismatch: expected {expected.dtype}, received {actual.dtype}"
            )
    module.load_state_dict(tensors, strict=True)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    model.is_trained = True
    return ArtifactBackedDecoder(model, manifest)


__all__ = [
    "ArtifactBackedDecoder",
    "ModelArtifactManifest",
    "ModelInputContract",
    "ModelOutputContract",
    "export_model_artifact",
    "load_model_artifact",
    "verify_model_artifact",
]
