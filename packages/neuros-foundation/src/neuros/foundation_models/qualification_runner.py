"""Executable referee for Neural System Qualification (NSQ) v1.

The runner owns evidence authority, not external model training. It composes the
existing :class:`LongitudinalCaseAuthority`, exposes only authorized observations
to a fresh external decoder at each calibration budget, validates output
semantics, and preserves every attempted external run including failures.

The v1 longitudinal executor is intentionally *labeled-target only*. The generic
NSQ participation schema can describe unlabeled target adaptation, but this
executor refuses to manufacture an unlabeled pool from leftover calibration
rows. A future executor must receive a separately frozen unlabeled-target
observation authority before ``adapt_unlabeled(X)`` may be called.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .longitudinal_authority import LongitudinalCaseAuthority
from .qualification import (
    ExternalDecoderMethodSpec,
    ExternalLearnedState,
    ExternalProbabilityDecoder,
    ExternalQualificationDecoder,
    ExternalQualificationFactory,
    QualificationModelState,
    QualificationProtocolSpec,
    QualificationRunContract,
    bind_learned_state,
    validate_prediction_output,
    validate_probability_output,
    validate_run_capabilities,
)
from .real_world import GroupedEvaluationData

QualificationStatus = Literal[
    "success",
    "failed",
    "skipped",
    "unavailable",
    "nonconverged",
    "oom",
]
MetricAvailability = Literal[
    "available",
    "unavailable_probability_output",
    "unavailable_class_support",
]

_SHA256_HEX = frozenset("0123456789abcdef")


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    normalized = value.strip()
    if len(normalized) != 64 or any(char not in _SHA256_HEX for char in normalized):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 digest")
    return normalized


def _sha_tuple(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of SHA-256 digests")
    result = tuple(_sha256(name, value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate SHA-256 digests")
    return result


def _strict_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer without coercion")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric without coercion")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("qualification identity cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("qualification identity cannot contain object arrays")
        return _canonical(value.tolist())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("qualification mapping keys must be non-empty strings")
            name = key.strip()
            if name in normalized:
                raise ValueError("qualification mapping keys collide after normalization")
            normalized[name] = _canonical(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("unordered sets are not valid qualification identity values")
    raise TypeError(
        "qualification identity must use deterministic JSON-compatible values; "
        f"got {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    normalized = _canonical(value)
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
    raw = json.dumps(
        {"schema": schema, "payload": _canonical(payload)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _index_set_sha256(
    role: str,
    processed_data_sha256: str,
    indices: Sequence[int] | np.ndarray,
) -> str:
    values = np.asarray(indices)
    if values.ndim != 1:
        raise ValueError(f"{role} indices must be one-dimensional")
    if values.dtype == np.bool_:
        raise ValueError(f"{role} indices cannot be booleans")
    integer = values.astype(np.int64)
    if not np.array_equal(values, integer):
        raise ValueError(f"{role} indices must be integers without coercion")
    return _identity_sha256(
        "neuros.qualification_observation_set.v1",
        {
            "role": role,
            "processed_data_sha256": _sha256(
                "processed_data_sha256", processed_data_sha256
            ),
            "indices": [int(value) for value in integer.tolist()],
        },
    )


class QualificationUnavailableError(RuntimeError):
    """External method/runtime is unavailable for the attempted case."""


class QualificationSkippedError(RuntimeError):
    """Case was intentionally skipped under a predeclared scientific rule."""


class QualificationNonConvergenceError(RuntimeError):
    """External method completed its budget but did not satisfy convergence."""


@runtime_checkable
class ExternalProbabilityClassOrderProvider(Protocol):
    """Fitted class-column order required for probability-capable methods."""

    def probability_class_labels(self) -> Sequence[Any]:
        ...


@dataclass(frozen=True, slots=True)
class QualificationExecutionContext:
    """Observed upstream provenance for one qualification execution.

    ``observed_dataset_lineage_sha256`` identifies the upstream dataset/revision
    actually loaded. It is intentionally distinct from the case authority's
    processed-array SHA-256.

    ``unlabeled_target_examples`` is retained in the context schema so callers
    cannot accidentally erase that intent. The v1 ``LongitudinalCaseAuthority``
    executor refuses any nonzero value because that authority does not freeze a
    scientifically distinct unlabeled-adaptation pool.
    """

    observed_dataset_lineage_sha256: str
    preprocessing_authority_sha256s: tuple[str, ...] = ()
    calibration_authority_sha256s: tuple[str, ...] = ()
    unlabeled_target_examples: int = 0
    target_example_duration_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("QualificationExecutionContext schema_version must be 1")
        object.__setattr__(
            self,
            "observed_dataset_lineage_sha256",
            _sha256(
                "observed_dataset_lineage_sha256",
                self.observed_dataset_lineage_sha256,
            ),
        )
        object.__setattr__(
            self,
            "preprocessing_authority_sha256s",
            _sha_tuple(
                "preprocessing_authority_sha256s",
                self.preprocessing_authority_sha256s,
            ),
        )
        object.__setattr__(
            self,
            "calibration_authority_sha256s",
            _sha_tuple(
                "calibration_authority_sha256s",
                self.calibration_authority_sha256s,
            ),
        )
        count = _strict_nonnegative_int(
            "unlabeled_target_examples", self.unlabeled_target_examples
        )
        object.__setattr__(self, "unlabeled_target_examples", count)
        if self.target_example_duration_s is not None:
            duration = _strict_positive_float(
                "target_example_duration_s", self.target_example_duration_s
            )
            if count == 0:
                raise ValueError(
                    "target_example_duration_s requires unlabeled_target_examples > 0"
                )
            object.__setattr__(self, "target_example_duration_s", duration)
        metadata = _freeze(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def unlabeled_target_seconds(self) -> float:
        if self.target_example_duration_s is None:
            return 0.0
        return float(self.unlabeled_target_examples * self.target_example_duration_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observed_dataset_lineage_sha256": self.observed_dataset_lineage_sha256,
            "preprocessing_authority_sha256s": list(
                self.preprocessing_authority_sha256s
            ),
            "calibration_authority_sha256s": list(
                self.calibration_authority_sha256s
            ),
            "unlabeled_target_examples": self.unlabeled_target_examples,
            "target_example_duration_s": self.target_example_duration_s,
            "metadata": _thaw(self.metadata),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.qualification_execution_context.v1", self.to_dict()
        )


@dataclass(frozen=True, slots=True)
class QualificationScore:
    """Metric values plus explicit availability for one successful execution."""

    metrics: Mapping[str, float | None]
    availability: Mapping[str, MetricAvailability]

    def __post_init__(self) -> None:
        metrics = dict(self.metrics)
        availability = dict(self.availability)
        if set(metrics) != set(availability):
            raise ValueError("metrics and availability must contain identical keys")
        normalized: dict[str, float | None] = {}
        for name, value in metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("metric names must be non-empty strings")
            state = availability[name]
            if state not in {
                "available",
                "unavailable_probability_output",
                "unavailable_class_support",
            }:
                raise ValueError(f"unsupported availability state {state!r}")
            if value is None:
                if state == "available":
                    raise ValueError(
                        f"metric {name!r} cannot be available with a null value"
                    )
                normalized[name] = None
            else:
                if state != "available":
                    raise ValueError(
                        f"metric {name!r} has a value but is marked unavailable"
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"metric {name!r} must be finite")
                normalized[name] = number
        object.__setattr__(self, "metrics", MappingProxyType(normalized))
        object.__setattr__(self, "availability", MappingProxyType(availability))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "availability": dict(self.availability),
        }


@runtime_checkable
class QualificationScorecard(Protocol):
    """Trusted-code metric authority bound by full SHA-256 in the protocol."""

    @property
    def sha256(self) -> str:
        ...

    @property
    def metric_names(self) -> tuple[str, ...]:
        ...

    def score(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probability: np.ndarray | None,
        class_labels: tuple[str, ...],
    ) -> QualificationScore:
        ...


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _binary_auc(y_true: np.ndarray, positive_score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=bool)
    scores = np.asarray(positive_score, dtype=np.float64)
    n_positive = int(np.sum(y))
    n_negative = int(len(y) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None
    ranks = _average_ranks(scores)
    rank_sum_positive = float(np.sum(ranks[y]))
    return float(
        (rank_sum_positive - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


@dataclass(frozen=True, slots=True)
class ClassificationScorecardV1:
    """Dependency-light classification scorecard for NSQ v1.

    Semantics:
    - balanced accuracy: macro mean recall over the source-derived vocabulary;
    - ROC AUC: binary rank AUC, positive class = second canonical source label;
    - Brier: mean per-sample multiclass sum-squared probability error;
    - ECE: 10 equal-width top-label confidence bins by default.
    """

    ece_bins: int = 10
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ClassificationScorecardV1 schema_version must be 1")
        if isinstance(self.ece_bins, bool) or not isinstance(self.ece_bins, int):
            raise ValueError("ece_bins must be an integer without coercion")
        if self.ece_bins <= 1:
            raise ValueError("ece_bins must be greater than one")

    @property
    def metric_names(self) -> tuple[str, ...]:
        return (
            "balanced_accuracy",
            "accuracy",
            "roc_auc",
            "brier_score",
            "expected_calibration_error",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_semantics": {
                "balanced_accuracy": {
                    "direction": "higher",
                    "definition": "macro_mean_recall_over_source_class_vocabulary",
                },
                "accuracy": {
                    "direction": "higher",
                    "definition": "fraction_exact_task_labels",
                },
                "roc_auc": {
                    "direction": "higher",
                    "definition": "binary_rank_auc",
                    "positive_class": "second_canonical_source_label",
                    "undefined_policy": "unavailable_class_support",
                },
                "brier_score": {
                    "direction": "lower",
                    "definition": "mean_multiclass_sum_squared_probability_error",
                    "undefined_policy": "unavailable_probability_output",
                },
                "expected_calibration_error": {
                    "direction": "lower",
                    "definition": "top_label_ece_equal_width",
                    "bins": self.ece_bins,
                    "undefined_policy": "unavailable_probability_output",
                },
            },
            "aggregation_unit": "trial_within_case",
            "implementation": "neuros-foundation:numpy",
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256("neuros.classification_scorecard.v1", self.to_dict())

    def score(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probability: np.ndarray | None,
        class_labels: tuple[str, ...],
    ) -> QualificationScore:
        truth = np.asarray(y_true).astype(str)
        prediction = np.asarray(y_pred).astype(str)
        if truth.shape != prediction.shape or truth.ndim != 1:
            raise ValueError("scorecard expects aligned one-dimensional labels")
        if not class_labels:
            raise ValueError("class_labels must be non-empty")

        recalls = [
            float(np.mean(prediction[truth == label] == label))
            for label in class_labels
            if np.any(truth == label)
        ]
        if not recalls:
            raise ValueError("evaluation set contains no declared task classes")

        metrics: dict[str, float | None] = {
            "balanced_accuracy": float(np.mean(recalls)),
            "accuracy": float(np.mean(prediction == truth)),
            "roc_auc": None,
            "brier_score": None,
            "expected_calibration_error": None,
        }
        availability: dict[str, MetricAvailability] = {
            "balanced_accuracy": "available",
            "accuracy": "available",
            "roc_auc": "unavailable_probability_output",
            "brier_score": "unavailable_probability_output",
            "expected_calibration_error": "unavailable_probability_output",
        }
        if probability is None:
            return QualificationScore(metrics=metrics, availability=availability)

        probs = np.asarray(probability, dtype=np.float64)
        n_classes = len(class_labels)
        label_to_index = {label: index for index, label in enumerate(class_labels)}
        encoded = np.asarray(
            [label_to_index[label] for label in truth], dtype=np.int64
        )
        one_hot = np.eye(n_classes, dtype=np.float64)[encoded]
        metrics["brier_score"] = float(
            np.mean(np.sum((probs - one_hot) ** 2, axis=1))
        )
        availability["brier_score"] = "available"

        predicted_index = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        correct = (predicted_index == encoded).astype(np.float64)
        edges = np.linspace(0.0, 1.0, self.ece_bins + 1)
        ece = 0.0
        for index in range(self.ece_bins):
            if index == self.ece_bins - 1:
                mask = (
                    (confidence >= edges[index])
                    & (confidence <= edges[index + 1])
                )
            else:
                mask = (
                    (confidence >= edges[index])
                    & (confidence < edges[index + 1])
                )
            if np.any(mask):
                ece += float(np.mean(mask)) * abs(
                    float(np.mean(correct[mask]))
                    - float(np.mean(confidence[mask]))
                )
        metrics["expected_calibration_error"] = float(ece)
        availability["expected_calibration_error"] = "available"

        if n_classes == 2:
            auc = _binary_auc(truth == class_labels[1], probs[:, 1])
            if auc is None:
                availability["roc_auc"] = "unavailable_class_support"
            else:
                metrics["roc_auc"] = auc
                availability["roc_auc"] = "available"
        else:
            availability["roc_auc"] = "unavailable_class_support"

        return QualificationScore(metrics=metrics, availability=availability)


DEFAULT_CLASSIFICATION_SCORECARD = ClassificationScorecardV1()


@dataclass(frozen=True, slots=True)
class QualificationBudgetResult:
    """Failure-preserving result for one case/method/calibration budget."""

    status: QualificationStatus
    case_id: str
    method_id: str
    calibration_per_class: int
    protocol_sha256: str
    case_authority_sha256: str
    method_spec_sha256: str
    run_contract_sha256: str
    execution_context_sha256: str
    metric_scorecard_sha256: str
    processed_data_sha256: str
    observed_dataset_lineage_sha256: str
    source_train_indices_sha256: str
    labeled_target_indices_sha256: str
    fit_indices_sha256: str
    evaluation_indices_sha256: str
    qualification_model_state_sha256: str | None
    external_learned_state_sha256: str | None
    external_state_identity_kind: str | None
    learned_state_addressable: bool
    source_train_samples: int
    labeled_target_examples: int
    evaluation_samples: int
    class_labels: tuple[str, ...]
    score: QualificationScore | None = None
    probability_available: bool = False
    fit_s: float | None = None
    inference_s: float | None = None
    failure_type: str | None = None
    failure_reason: str | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.status not in {
            "success",
            "failed",
            "skipped",
            "unavailable",
            "nonconverged",
            "oom",
        }:
            raise ValueError(f"unsupported qualification status {self.status!r}")
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            raise ValueError("QualificationBudgetResult schema_version must be 2")
        for name in (
            "protocol_sha256",
            "case_authority_sha256",
            "method_spec_sha256",
            "run_contract_sha256",
            "execution_context_sha256",
            "metric_scorecard_sha256",
            "processed_data_sha256",
            "observed_dataset_lineage_sha256",
            "source_train_indices_sha256",
            "labeled_target_indices_sha256",
            "fit_indices_sha256",
            "evaluation_indices_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        for name in (
            "qualification_model_state_sha256",
            "external_learned_state_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha256(name, value))

        if self.learned_state_addressable:
            if self.external_learned_state_sha256 is None:
                raise ValueError(
                    "addressable learned state requires external_learned_state_sha256"
                )
        elif self.external_learned_state_sha256 is not None:
            raise ValueError(
                "opaque learned state cannot expose external_learned_state_sha256"
            )

        if self.status == "success":
            if self.score is None or self.qualification_model_state_sha256 is None:
                raise ValueError(
                    "successful result requires score and qualification model-state binding"
                )
            if self.external_state_identity_kind is None:
                raise ValueError(
                    "successful result requires external_state_identity_kind"
                )
            if self.failure_type is not None or self.failure_reason is not None:
                raise ValueError("successful result cannot contain failure metadata")
        else:
            if self.score is not None:
                raise ValueError("non-success result cannot contain a score")
            if not self.failure_type or not self.failure_reason:
                raise ValueError(
                    "non-success result requires failure_type and failure_reason"
                )

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "case_id": self.case_id,
            "method_id": self.method_id,
            "calibration_per_class": self.calibration_per_class,
            "protocol_sha256": self.protocol_sha256,
            "case_authority_sha256": self.case_authority_sha256,
            "method_spec_sha256": self.method_spec_sha256,
            "run_contract_sha256": self.run_contract_sha256,
            "execution_context_sha256": self.execution_context_sha256,
            "metric_scorecard_sha256": self.metric_scorecard_sha256,
            "processed_data_sha256": self.processed_data_sha256,
            "observed_dataset_lineage_sha256": self.observed_dataset_lineage_sha256,
            "source_train_indices_sha256": self.source_train_indices_sha256,
            "labeled_target_indices_sha256": self.labeled_target_indices_sha256,
            "fit_indices_sha256": self.fit_indices_sha256,
            "evaluation_indices_sha256": self.evaluation_indices_sha256,
            "qualification_model_state_sha256": self.qualification_model_state_sha256,
            "external_learned_state_sha256": self.external_learned_state_sha256,
            "external_state_identity_kind": self.external_state_identity_kind,
            "learned_state_addressable": self.learned_state_addressable,
            "source_train_samples": self.source_train_samples,
            "labeled_target_examples": self.labeled_target_examples,
            "unlabeled_target_examples": 0,
            "unlabeled_target_seconds": 0.0,
            "evaluation_samples": self.evaluation_samples,
            "class_labels": list(self.class_labels),
            "score": None if self.score is None else self.score.to_dict(),
            "probability_available": self.probability_available,
            "fit_s": self.fit_s,
            "inference_s": self.inference_s,
            "failure_type": self.failure_type,
            "failure_reason": self.failure_reason,
        }
        if include_sha256:
            payload["result_sha256"] = self.sha256
        return payload

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.qualification_budget_result.v2",
            self.to_dict(include_sha256=False),
        )


@dataclass(frozen=True, slots=True)
class QualificationCaseResult:
    """Complete failure-preserving budget frontier for one case and method."""

    protocol_sha256: str
    case_authority_sha256: str
    method_spec_sha256: str
    execution_context_sha256: str
    metric_scorecard_sha256: str
    rows: tuple[QualificationBudgetResult, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            raise ValueError("QualificationCaseResult schema_version must be 2")
        for name in (
            "protocol_sha256",
            "case_authority_sha256",
            "method_spec_sha256",
            "execution_context_sha256",
            "metric_scorecard_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        if not self.rows:
            raise ValueError(
                "qualification case result must contain at least one budget row"
            )
        budgets = tuple(row.calibration_per_class for row in self.rows)
        if len(set(budgets)) != len(budgets):
            raise ValueError("qualification result cannot duplicate calibration budgets")
        for row in self.rows:
            if row.protocol_sha256 != self.protocol_sha256:
                raise ValueError("row protocol SHA differs from case result")
            if row.case_authority_sha256 != self.case_authority_sha256:
                raise ValueError("row case-authority SHA differs from case result")
            if row.method_spec_sha256 != self.method_spec_sha256:
                raise ValueError("row method SHA differs from case result")
            if row.execution_context_sha256 != self.execution_context_sha256:
                raise ValueError("row execution-context SHA differs from case result")
            if row.metric_scorecard_sha256 != self.metric_scorecard_sha256:
                raise ValueError("row metric-scorecard SHA differs from case result")

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "case_authority_sha256": self.case_authority_sha256,
            "method_spec_sha256": self.method_spec_sha256,
            "execution_context_sha256": self.execution_context_sha256,
            "metric_scorecard_sha256": self.metric_scorecard_sha256,
            "rows": [row.to_dict() for row in self.rows],
        }
        if include_sha256:
            payload["result_sha256"] = self.sha256
        return payload

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.qualification_case_result.v2",
            self.to_dict(include_sha256=False),
        )


def _canonical_class_labels(
    y: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[str, ...]:
    source_labels = np.asarray(y)[source_indices].astype(str)
    labels = tuple(sorted(np.unique(source_labels).tolist()))
    if len(labels) < 2:
        raise ValueError("source history must contain at least two task classes")
    all_labels = np.asarray(y).astype(str)
    unknown = sorted(set(all_labels.tolist()) - set(labels))
    if unknown:
        raise ValueError(
            "loaded data contains labels absent from source-history vocabulary: "
            f"{unknown}"
        )
    return labels


def _failure_status(exc: Exception) -> QualificationStatus:
    if isinstance(exc, QualificationSkippedError):
        return "skipped"
    if isinstance(exc, QualificationUnavailableError) or isinstance(exc, ImportError):
        return "unavailable"
    if isinstance(exc, QualificationNonConvergenceError):
        return "nonconverged"
    if isinstance(exc, MemoryError) or "OutOfMemoryError" in type(exc).__name__:
        return "oom"
    return "failed"


def _failure_reason(exc: Exception) -> tuple[str, str]:
    kind = type(exc).__name__
    reason = str(exc).strip() or kind
    if len(reason) > 512:
        reason = reason[:509] + "..."
    return kind, reason


def _validate_preflight(
    data: GroupedEvaluationData,
    authority: LongitudinalCaseAuthority,
    protocol: QualificationProtocolSpec,
    factory: ExternalQualificationFactory,
    execution_context: QualificationExecutionContext,
    scorecard: QualificationScorecard,
) -> tuple[Any, ExternalDecoderMethodSpec, tuple[str, ...]]:
    if protocol.protocol_status != "frozen":
        raise ValueError("qualification execution requires protocol_status='frozen'")
    if protocol.dataset_id != data.dataset_id:
        raise ValueError(
            f"protocol dataset_id={protocol.dataset_id!r} does not match loaded "
            f"dataset_id={data.dataset_id!r}"
        )
    if (
        execution_context.observed_dataset_lineage_sha256
        != protocol.dataset_lineage_sha256
    ):
        raise ValueError(
            "observed dataset lineage SHA-256 differs from frozen protocol"
        )
    if protocol.metric_scorecard_sha256 != scorecard.sha256:
        raise ValueError("metric scorecard SHA-256 differs from frozen protocol")
    declared_metrics = (protocol.primary_metric, *protocol.secondary_metrics)
    if tuple(declared_metrics) != tuple(scorecard.metric_names):
        raise ValueError("protocol metric names/order differ from scorecard authority")
    if execution_context.unlabeled_target_examples:
        raise ValueError(
            "LongitudinalCaseAuthority NSQ runner v1 is labeled-target only; "
            "unlabeled adaptation requires a separately frozen unlabeled-target "
            "observation authority and cannot reuse leftover calibration rows"
        )

    split = authority.restore(data)
    if authority.split_unit not in protocol.grouping_hierarchy:
        raise ValueError(
            f"case split unit {authority.split_unit!r} is absent from protocol "
            "grouping hierarchy"
        )
    for budget in protocol.calibration_budgets_per_class:
        if budget > split.max_budget_per_class:
            raise ValueError(
                f"protocol budget {budget}/class exceeds frozen authority maximum "
                f"{split.max_budget_per_class}/class"
            )

    method_spec = factory.method_spec
    if not isinstance(method_spec, ExternalDecoderMethodSpec):
        raise TypeError("factory.method_spec must be an ExternalDecoderMethodSpec")
    X = np.asarray(data.X)
    if not method_spec.input_axes or method_spec.input_axes[0] != "sample":
        raise ValueError("external method input_axes must begin with 'sample'")
    if len(method_spec.input_axes) != X.ndim:
        raise ValueError(
            "external method input_axes dimensionality differs from processed neural array"
        )

    labels = _canonical_class_labels(
        np.asarray(data.y), np.asarray(split.source_train_indices, dtype=np.int64)
    )
    return split, method_spec, labels


def run_external_qualification_case(
    data: GroupedEvaluationData,
    authority: LongitudinalCaseAuthority,
    protocol: QualificationProtocolSpec,
    factory: ExternalQualificationFactory,
    *,
    execution_context: QualificationExecutionContext,
    scorecard: QualificationScorecard = DEFAULT_CLASSIFICATION_SCORECARD,
) -> QualificationCaseResult:
    """Execute one external method across a frozen labeled-calibration frontier.

    Scientific-authority failures abort before any external model is created.
    External model/runtime failures after a valid run contract exists are retained
    as explicit per-budget rows.
    """

    split, method_spec, class_labels = _validate_preflight(
        data,
        authority,
        protocol,
        factory,
        execution_context,
        scorecard,
    )

    X = np.asarray(data.X)
    y = np.asarray(data.y).astype(str)
    evaluation_indices = np.asarray(split.evaluation_indices, dtype=np.int64)
    source_indices = np.asarray(split.source_train_indices, dtype=np.int64)
    source_sha = _index_set_sha256(
        "supervised_source_history",
        authority.processed_data_sha256,
        source_indices,
    )
    evaluation_sha = _index_set_sha256(
        "untouched_final_assessment",
        authority.processed_data_sha256,
        evaluation_indices,
    )
    rows: list[QualificationBudgetResult] = []

    for budget in protocol.calibration_budgets_per_class:
        calibration_indices = np.asarray(
            split.calibration_indices(budget), dtype=np.int64
        )
        train_indices = np.asarray(
            split.train_indices_for_budget(budget), dtype=np.int64
        )
        if np.intersect1d(train_indices, evaluation_indices).size:
            raise RuntimeError(
                "internal authority error: fit set overlaps final assessment"
            )
        labeled_sha = _index_set_sha256(
            "labeled_target_calibration",
            authority.processed_data_sha256,
            calibration_indices,
        )
        fit_sha = _index_set_sha256(
            "supervised_fit",
            authority.processed_data_sha256,
            train_indices,
        )

        run_contract = QualificationRunContract(
            protocol_sha256=protocol.sha256,
            method_spec_sha256=method_spec.sha256,
            case_authority_sha256=authority.authority_sha256,
            labeled_target_examples=int(len(calibration_indices)),
            unlabeled_target_examples=0,
            unlabeled_target_seconds=0.0,
            preprocessing_authority_sha256s=(
                execution_context.preprocessing_authority_sha256s
            ),
            calibration_authority_sha256s=(
                execution_context.calibration_authority_sha256s
            ),
            metadata={
                "calibration_per_class": int(budget),
                "source_train_indices_sha256": source_sha,
                "labeled_target_indices_sha256": labeled_sha,
                "fit_indices_sha256": fit_sha,
                "evaluation_indices_sha256": evaluation_sha,
                "observation_authority_schema": "neuros.nsq_observation_roles.v1",
            },
        )

        fit_s: float | None = None
        inference_s: float | None = None
        bound_state: QualificationModelState | None = None
        learned_state: ExternalLearnedState | None = None
        try:
            decoder = factory.create()
            if not isinstance(decoder, ExternalQualificationDecoder):
                raise TypeError(
                    "factory.create() must return an ExternalQualificationDecoder"
                )
            validate_run_capabilities(method_spec, run_contract, decoder)

            started = time.perf_counter()
            decoder.fit(X[train_indices], y[train_indices])
            fit_s = float(time.perf_counter() - started)

            learned_state = decoder.learned_state()
            if not isinstance(learned_state, ExternalLearnedState):
                raise TypeError(
                    "decoder.learned_state() must return ExternalLearnedState"
                )
            bound_state = bind_learned_state(
                method_spec,
                run_contract,
                learned_state,
            )

            started = time.perf_counter()
            raw_prediction = decoder.predict(X[evaluation_indices])
            inference_s = float(time.perf_counter() - started)
            prediction = validate_prediction_output(
                raw_prediction,
                expected_samples=len(evaluation_indices),
                allowed_labels=class_labels,
            ).astype(str)

            probability: np.ndarray | None = None
            probability_available = (
                method_spec.probability_semantics != "unavailable"
            )
            if probability_available:
                if not isinstance(decoder, ExternalProbabilityDecoder):
                    raise TypeError(
                        "method declares probability output but decoder lacks "
                        "predict_proba(X)"
                    )
                if not isinstance(
                    decoder, ExternalProbabilityClassOrderProvider
                ):
                    raise TypeError(
                        "probability-capable decoder must expose "
                        "probability_class_labels()"
                    )
                probability_labels = tuple(
                    str(value) for value in decoder.probability_class_labels()
                )
                if probability_labels != class_labels:
                    raise ValueError(
                        "probability class order differs from canonical "
                        "source-derived class order"
                    )
                probability = validate_probability_output(
                    method_spec,
                    decoder.predict_proba(X[evaluation_indices]),
                    expected_samples=len(evaluation_indices),
                    expected_classes=len(class_labels),
                )

            score = scorecard.score(
                y_true=y[evaluation_indices],
                y_pred=prediction,
                probability=probability,
                class_labels=class_labels,
            )
            rows.append(
                QualificationBudgetResult(
                    status="success",
                    case_id=authority.case_id,
                    method_id=method_spec.method_id,
                    calibration_per_class=int(budget),
                    protocol_sha256=protocol.sha256,
                    case_authority_sha256=authority.authority_sha256,
                    method_spec_sha256=method_spec.sha256,
                    run_contract_sha256=run_contract.sha256,
                    execution_context_sha256=execution_context.sha256,
                    metric_scorecard_sha256=scorecard.sha256,
                    processed_data_sha256=authority.processed_data_sha256,
                    observed_dataset_lineage_sha256=(
                        execution_context.observed_dataset_lineage_sha256
                    ),
                    source_train_indices_sha256=source_sha,
                    labeled_target_indices_sha256=labeled_sha,
                    fit_indices_sha256=fit_sha,
                    evaluation_indices_sha256=evaluation_sha,
                    qualification_model_state_sha256=bound_state.sha256,
                    external_learned_state_sha256=learned_state.state_sha256,
                    external_state_identity_kind=(
                        learned_state.state_identity_kind
                    ),
                    learned_state_addressable=bound_state.state_addressable,
                    source_train_samples=int(len(source_indices)),
                    labeled_target_examples=int(len(calibration_indices)),
                    evaluation_samples=int(len(evaluation_indices)),
                    class_labels=class_labels,
                    score=score,
                    probability_available=probability_available,
                    fit_s=fit_s,
                    inference_s=inference_s,
                )
            )
        except Exception as exc:
            failure_type, failure_reason = _failure_reason(exc)
            rows.append(
                QualificationBudgetResult(
                    status=_failure_status(exc),
                    case_id=authority.case_id,
                    method_id=method_spec.method_id,
                    calibration_per_class=int(budget),
                    protocol_sha256=protocol.sha256,
                    case_authority_sha256=authority.authority_sha256,
                    method_spec_sha256=method_spec.sha256,
                    run_contract_sha256=run_contract.sha256,
                    execution_context_sha256=execution_context.sha256,
                    metric_scorecard_sha256=scorecard.sha256,
                    processed_data_sha256=authority.processed_data_sha256,
                    observed_dataset_lineage_sha256=(
                        execution_context.observed_dataset_lineage_sha256
                    ),
                    source_train_indices_sha256=source_sha,
                    labeled_target_indices_sha256=labeled_sha,
                    fit_indices_sha256=fit_sha,
                    evaluation_indices_sha256=evaluation_sha,
                    qualification_model_state_sha256=(
                        None if bound_state is None else bound_state.sha256
                    ),
                    external_learned_state_sha256=(
                        None if learned_state is None else learned_state.state_sha256
                    ),
                    external_state_identity_kind=(
                        None
                        if learned_state is None
                        else learned_state.state_identity_kind
                    ),
                    learned_state_addressable=(
                        False
                        if bound_state is None
                        else bound_state.state_addressable
                    ),
                    source_train_samples=int(len(source_indices)),
                    labeled_target_examples=int(len(calibration_indices)),
                    evaluation_samples=int(len(evaluation_indices)),
                    class_labels=class_labels,
                    score=None,
                    probability_available=False,
                    fit_s=fit_s,
                    inference_s=inference_s,
                    failure_type=failure_type,
                    failure_reason=failure_reason,
                )
            )

    return QualificationCaseResult(
        protocol_sha256=protocol.sha256,
        case_authority_sha256=authority.authority_sha256,
        method_spec_sha256=method_spec.sha256,
        execution_context_sha256=execution_context.sha256,
        metric_scorecard_sha256=scorecard.sha256,
        rows=tuple(rows),
    )


__all__ = [
    "ClassificationScorecardV1",
    "DEFAULT_CLASSIFICATION_SCORECARD",
    "ExternalProbabilityClassOrderProvider",
    "MetricAvailability",
    "QualificationBudgetResult",
    "QualificationCaseResult",
    "QualificationExecutionContext",
    "QualificationNonConvergenceError",
    "QualificationScore",
    "QualificationScorecard",
    "QualificationSkippedError",
    "QualificationStatus",
    "QualificationUnavailableError",
    "run_external_qualification_case",
]
