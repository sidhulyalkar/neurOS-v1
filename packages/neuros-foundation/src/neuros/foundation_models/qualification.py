"""Peer-facing contracts for neural-system qualification.

neurOS owns the authority around a comparison, not the external researcher's
training code. The authority chain is deliberately explicit:

    protocol -> method spec -> run contract -> learned-state identity

Benchmark metadata never becomes executable import code. Target information can
enter an external method only through a separately declared authority path.
Full SHA references compose with Scientific Authority without making this
package depend on ORION's implementation classes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

ProbabilitySemantics = Literal[
    "uncalibrated_softmax",
    "calibrated_probability",
    "unavailable",
]
ProtocolStatus = Literal["draft", "frozen", "retired"]
StateIdentityKind = Literal[
    "tensor_sha256",
    "checkpoint_sha256",
    "opaque_unverified",
]
TargetAdaptationMode = Literal["none", "unlabeled"]

_SHA256_HEX = frozenset("0123456789abcdef")


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    value = value.strip()
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 digest")
    return value


def _optional_sha256(name: str, value: Any) -> str | None:
    return None if value is None else _sha256(name, value)


def _sha_tuple(name: str, values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of SHA-256 digests")
    try:
        result = tuple(_sha256(name, value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of SHA-256 digests") from exc
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate SHA-256 digests")
    return result


def _strings(name: str, values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        result = tuple(_nonempty(name, value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _exact_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer without coercion")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_nonnegative_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric without coercion")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _canonical_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("qualification metadata cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _nonempty("metadata key", key)
            if normalized in result:
                raise ValueError("qualification metadata keys collide after normalization")
            result[normalized] = _canonical_json(item)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (tuple, list)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("unordered sets are not valid qualification metadata")
    raise TypeError(
        "qualification metadata must use deterministic JSON-compatible values; "
        f"got {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    normalized = _canonical_json(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze(item) for item in normalized)
    return normalized


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _identity_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"schema": schema, "payload": _canonical_json(payload)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class QualificationProtocolSpec:
    """Model-independent scientific question and evaluation authority.

    Human-readable metric names remain useful for tables, but a frozen protocol
    must also bind the full SHA-256 of its immutable metric scorecard. That
    scorecard can be produced by Scientific Authority v2 without introducing an
    ORION dependency into neuros-foundation.
    """

    protocol_id: str
    dataset_id: str
    dataset_lineage_sha256: str
    task_id: str
    independent_unit: str
    grouping_hierarchy: tuple[str, ...]
    calibration_budgets_per_class: tuple[int, ...]
    primary_metric: str = "balanced_accuracy"
    secondary_metrics: tuple[str, ...] = (
        "accuracy",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
    )
    metric_scorecard_sha256: str | None = None
    robustness_axes: tuple[str, ...] = (
        "session",
        "subject",
        "channel_drop",
        "artifact_sensitivity",
    )
    final_assessment_role: str = "untouched_final_assessment"
    protocol_status: ProtocolStatus = "draft"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("QualificationProtocolSpec schema_version must be 1")
        protocol_id = _nonempty("protocol_id", self.protocol_id)
        dataset_id = _nonempty("dataset_id", self.dataset_id)
        dataset_lineage = _sha256("dataset_lineage_sha256", self.dataset_lineage_sha256)
        task_id = _nonempty("task_id", self.task_id)
        independent_unit = _nonempty("independent_unit", self.independent_unit)
        hierarchy = _strings("grouping_hierarchy", self.grouping_hierarchy)
        if not hierarchy or hierarchy[0] != independent_unit:
            raise ValueError("grouping_hierarchy must start with the declared independent_unit")
        if isinstance(self.calibration_budgets_per_class, (str, bytes)):
            raise ValueError("calibration_budgets_per_class must be a sequence of integers")
        budgets = tuple(
            _exact_nonnegative_int("calibration budget", value)
            for value in self.calibration_budgets_per_class
        )
        if not budgets or budgets[0] != 0:
            raise ValueError("calibration_budgets_per_class must start at zero")
        if tuple(sorted(set(budgets))) != budgets:
            raise ValueError("calibration_budgets_per_class must be unique and strictly increasing")
        primary = _nonempty("primary_metric", self.primary_metric)
        secondary = _strings("secondary_metrics", self.secondary_metrics)
        if primary in secondary:
            raise ValueError("primary_metric must not be duplicated in secondary_metrics")
        metric_scorecard = _optional_sha256(
            "metric_scorecard_sha256", self.metric_scorecard_sha256
        )
        robustness = _strings("robustness_axes", self.robustness_axes)
        final_role = _nonempty("final_assessment_role", self.final_assessment_role)
        if final_role != "untouched_final_assessment":
            raise ValueError("v1 final_assessment_role must be 'untouched_final_assessment'")
        if self.protocol_status not in {"draft", "frozen", "retired"}:
            raise ValueError("protocol_status must be draft, frozen, or retired")
        if self.protocol_status == "frozen" and metric_scorecard is None:
            raise ValueError("frozen qualification protocol requires metric_scorecard_sha256")
        metadata = _freeze(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "dataset_lineage_sha256", dataset_lineage)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "independent_unit", independent_unit)
        object.__setattr__(self, "grouping_hierarchy", hierarchy)
        object.__setattr__(self, "calibration_budgets_per_class", budgets)
        object.__setattr__(self, "primary_metric", primary)
        object.__setattr__(self, "secondary_metrics", secondary)
        object.__setattr__(self, "metric_scorecard_sha256", metric_scorecard)
        object.__setattr__(self, "robustness_axes", robustness)
        object.__setattr__(self, "final_assessment_role", final_role)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "dataset_id": self.dataset_id,
            "dataset_lineage_sha256": self.dataset_lineage_sha256,
            "task_id": self.task_id,
            "independent_unit": self.independent_unit,
            "grouping_hierarchy": list(self.grouping_hierarchy),
            "calibration_budgets_per_class": list(self.calibration_budgets_per_class),
            "primary_metric": self.primary_metric,
            "secondary_metrics": list(self.secondary_metrics),
            "metric_scorecard_sha256": self.metric_scorecard_sha256,
            "robustness_axes": list(self.robustness_axes),
            "final_assessment_role": self.final_assessment_role,
            "protocol_status": self.protocol_status,
            "metadata": _thaw(self.metadata),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256("neuros.qualification_protocol.v1", self.to_dict())

    @property
    def display_fingerprint(self) -> str:
        return self.sha256[:16]


@dataclass(frozen=True, slots=True)
class ExternalDecoderMethodSpec:
    """Stable external algorithm/configuration identity, excluding learned state.

    ``model_lineage_sha256=None`` means lineage is unknown, not disjoint. A
    foundation/pretrained method can therefore participate while Scientific
    Authority correctly refuses a verified-disjoint pretraining claim until its
    lineage record is supplied and audited.
    """

    method_id: str
    implementation: str
    implementation_version: str
    input_axes: tuple[str, ...]
    probability_semantics: ProbabilitySemantics
    target_adaptation_mode: TargetAdaptationMode = "none"
    uncertainty_semantics: str = "none"
    model_lineage_sha256: str | None = None
    source_reference: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ExternalDecoderMethodSpec schema_version must be 1")
        method_id = _nonempty("method_id", self.method_id)
        implementation = _nonempty("implementation", self.implementation)
        implementation_version = _nonempty("implementation_version", self.implementation_version)
        axes = _strings("input_axes", self.input_axes)
        if not axes:
            raise ValueError("input_axes must be non-empty")
        if self.probability_semantics not in {
            "uncalibrated_softmax",
            "calibrated_probability",
            "unavailable",
        }:
            raise ValueError("unsupported probability_semantics")
        if self.target_adaptation_mode not in {"none", "unlabeled"}:
            raise ValueError("target_adaptation_mode must be 'none' or 'unlabeled'")
        uncertainty = _nonempty("uncertainty_semantics", self.uncertainty_semantics)
        model_lineage = _optional_sha256("model_lineage_sha256", self.model_lineage_sha256)
        source = None if self.source_reference is None else _nonempty(
            "source_reference", self.source_reference
        )
        metadata = _freeze(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "implementation_version", implementation_version)
        object.__setattr__(self, "input_axes", axes)
        object.__setattr__(self, "uncertainty_semantics", uncertainty)
        object.__setattr__(self, "model_lineage_sha256", model_lineage)
        object.__setattr__(self, "source_reference", source)
        object.__setattr__(self, "metadata", metadata)

    @property
    def lineage_known(self) -> bool:
        return self.model_lineage_sha256 is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "input_axes": list(self.input_axes),
            "probability_semantics": self.probability_semantics,
            "target_adaptation_mode": self.target_adaptation_mode,
            "uncertainty_semantics": self.uncertainty_semantics,
            "model_lineage_sha256": self.model_lineage_sha256,
            "source_reference": self.source_reference,
            "metadata": _thaw(self.metadata),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256("neuros.external_decoder_method.v1", self.to_dict())

    @property
    def display_fingerprint(self) -> str:
        return self.sha256[:16]


@dataclass(frozen=True, slots=True)
class QualificationRunContract:
    """Exact target-information and preprocessing authority for one execution."""

    protocol_sha256: str
    method_spec_sha256: str
    case_authority_sha256: str
    labeled_target_examples: int = 0
    unlabeled_target_examples: int = 0
    unlabeled_target_seconds: float = 0.0
    preprocessing_authority_sha256s: tuple[str, ...] = ()
    calibration_authority_sha256s: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("QualificationRunContract schema_version must be 1")
        object.__setattr__(self, "protocol_sha256", _sha256("protocol_sha256", self.protocol_sha256))
        object.__setattr__(self, "method_spec_sha256", _sha256("method_spec_sha256", self.method_spec_sha256))
        object.__setattr__(self, "case_authority_sha256", _sha256("case_authority_sha256", self.case_authority_sha256))
        object.__setattr__(self, "labeled_target_examples", _exact_nonnegative_int("labeled_target_examples", self.labeled_target_examples))
        object.__setattr__(self, "unlabeled_target_examples", _exact_nonnegative_int("unlabeled_target_examples", self.unlabeled_target_examples))
        object.__setattr__(self, "unlabeled_target_seconds", _finite_nonnegative_float("unlabeled_target_seconds", self.unlabeled_target_seconds))
        object.__setattr__(self, "preprocessing_authority_sha256s", _sha_tuple("preprocessing_authority_sha256s", self.preprocessing_authority_sha256s))
        object.__setattr__(self, "calibration_authority_sha256s", _sha_tuple("calibration_authority_sha256s", self.calibration_authority_sha256s))
        metadata = _freeze(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def zero_shot(self) -> bool:
        return (
            self.labeled_target_examples == 0
            and self.unlabeled_target_examples == 0
            and self.unlabeled_target_seconds == 0.0
        )

    @property
    def consumes_unlabeled_target(self) -> bool:
        return self.unlabeled_target_examples > 0 or self.unlabeled_target_seconds > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "method_spec_sha256": self.method_spec_sha256,
            "case_authority_sha256": self.case_authority_sha256,
            "labeled_target_examples": self.labeled_target_examples,
            "unlabeled_target_examples": self.unlabeled_target_examples,
            "unlabeled_target_seconds": self.unlabeled_target_seconds,
            "preprocessing_authority_sha256s": list(self.preprocessing_authority_sha256s),
            "calibration_authority_sha256s": list(self.calibration_authority_sha256s),
            "metadata": _thaw(self.metadata),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256("neuros.qualification_run.v1", self.to_dict())


@dataclass(frozen=True, slots=True)
class ExternalLearnedState:
    """Adapter-reported learned-state identity after one authorized fit/adaptation."""

    state_identity_kind: StateIdentityKind = "opaque_unverified"
    state_sha256: str | None = None
    calibration_state_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ExternalLearnedState schema_version must be 1")
        if self.state_identity_kind not in {
            "tensor_sha256",
            "checkpoint_sha256",
            "opaque_unverified",
        }:
            raise ValueError("unsupported state_identity_kind")
        state_sha = _optional_sha256("state_sha256", self.state_sha256)
        calibration_sha = _optional_sha256("calibration_state_sha256", self.calibration_state_sha256)
        if self.state_identity_kind == "opaque_unverified" and state_sha is not None:
            raise ValueError("opaque_unverified state cannot claim state_sha256")
        if self.state_identity_kind != "opaque_unverified" and state_sha is None:
            raise ValueError("verified state identity requires state_sha256")
        metadata = _freeze(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "state_sha256", state_sha)
        object.__setattr__(self, "calibration_state_sha256", calibration_sha)
        object.__setattr__(self, "metadata", metadata)

    @property
    def state_addressable(self) -> bool:
        return self.state_identity_kind != "opaque_unverified" and self.state_sha256 is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_identity_kind": self.state_identity_kind,
            "state_sha256": self.state_sha256,
            "calibration_state_sha256": self.calibration_state_sha256,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class QualificationModelState:
    """System-created binding of one learned state to the exact run that produced it."""

    method_spec_sha256: str
    run_contract_sha256: str
    probability_semantics: ProbabilitySemantics
    learned_state: ExternalLearnedState
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("QualificationModelState schema_version must be 1")
        method_sha = _sha256("method_spec_sha256", self.method_spec_sha256)
        run_sha = _sha256("run_contract_sha256", self.run_contract_sha256)
        if self.probability_semantics not in {
            "uncalibrated_softmax",
            "calibrated_probability",
            "unavailable",
        }:
            raise ValueError("unsupported probability_semantics")
        if not isinstance(self.learned_state, ExternalLearnedState):
            raise TypeError("learned_state must be an ExternalLearnedState")
        calibration_sha = self.learned_state.calibration_state_sha256
        if self.probability_semantics == "calibrated_probability":
            if calibration_sha is None:
                raise ValueError("calibrated_probability requires calibration_state_sha256 for this fitted state")
        elif calibration_sha is not None:
            raise ValueError("calibration_state_sha256 may only accompany calibrated_probability")
        object.__setattr__(self, "method_spec_sha256", method_sha)
        object.__setattr__(self, "run_contract_sha256", run_sha)

    @property
    def state_addressable(self) -> bool:
        return self.learned_state.state_addressable

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_spec_sha256": self.method_spec_sha256,
            "run_contract_sha256": self.run_contract_sha256,
            "probability_semantics": self.probability_semantics,
            "learned_state": self.learned_state.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256("neuros.qualification_model_state.v1", self.to_dict())


def bind_learned_state(
    method_spec: ExternalDecoderMethodSpec,
    run_contract: QualificationRunContract,
    learned_state: ExternalLearnedState,
) -> QualificationModelState:
    """Bind adapter-reported state to the exact method/run that produced it."""

    if run_contract.method_spec_sha256 != method_spec.sha256:
        raise ValueError("run contract does not authorize this external method specification")
    return QualificationModelState(
        method_spec_sha256=method_spec.sha256,
        run_contract_sha256=run_contract.sha256,
        probability_semantics=method_spec.probability_semantics,
        learned_state=learned_state,
    )


@runtime_checkable
class ExternalQualificationDecoder(Protocol):
    """Minimal fitted-model surface used by the qualification runner."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def learned_state(self) -> ExternalLearnedState:
        ...


@runtime_checkable
class ExternalUnlabeledTargetAdapter(Protocol):
    """Separate authority channel for methods that consume unlabeled target data."""

    def adapt_unlabeled(self, X: np.ndarray) -> None:
        ...


@runtime_checkable
class ExternalQualificationFactory(Protocol):
    """Trusted-code factory used to obtain a fresh decoder for every budget."""

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        ...

    def create(self) -> ExternalQualificationDecoder:
        ...


def validate_run_capabilities(
    method_spec: ExternalDecoderMethodSpec,
    run_contract: QualificationRunContract,
    decoder: ExternalQualificationDecoder,
) -> None:
    """Ensure the declared target-information budget has a matching data path."""

    if run_contract.method_spec_sha256 != method_spec.sha256:
        raise ValueError("run contract does not authorize this external method specification")
    if run_contract.consumes_unlabeled_target:
        if method_spec.target_adaptation_mode != "unlabeled":
            raise ValueError(
                "run consumes unlabeled target information but method does not declare unlabeled adaptation"
            )
        if not isinstance(decoder, ExternalUnlabeledTargetAdapter):
            raise TypeError("method declares unlabeled adaptation but decoder lacks adapt_unlabeled(X)")


def validate_probability_output(
    method_spec: ExternalDecoderMethodSpec,
    probability: Any,
    *,
    expected_samples: int,
    expected_classes: int,
    atol: float = 1e-6,
) -> np.ndarray:
    """Fail closed on malformed probability output without renormalizing it."""

    if method_spec.probability_semantics == "unavailable":
        raise ValueError("model declares probability_semantics='unavailable'")
    probs = np.asarray(probability)
    if probs.ndim != 2 or probs.shape != (expected_samples, expected_classes):
        raise ValueError(f"probability output must have exact shape ({expected_samples}, {expected_classes})")
    if not np.issubdtype(probs.dtype, np.floating):
        raise ValueError("probability output must use a floating dtype")
    if not np.isfinite(probs).all():
        raise ValueError("probability output must be finite")
    if np.any(probs < -atol) or np.any(probs > 1.0 + atol):
        raise ValueError("probability output must remain within [0, 1]")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=atol):
        raise ValueError("probability rows must sum to one; neurOS will not renormalize them")
    return probs


__all__ = [
    "ExternalDecoderMethodSpec",
    "ExternalLearnedState",
    "ExternalQualificationDecoder",
    "ExternalQualificationFactory",
    "ExternalUnlabeledTargetAdapter",
    "ProbabilitySemantics",
    "ProtocolStatus",
    "QualificationModelState",
    "QualificationProtocolSpec",
    "QualificationRunContract",
    "StateIdentityKind",
    "TargetAdaptationMode",
    "bind_learned_state",
    "validate_probability_output",
    "validate_run_capabilities",
]
