"""Discovery-frozen, held-out causal feature correspondence studies."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import combinations
from math import comb
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from neuros_mechint.core.manifest import stable_hash

CORRESPONDENCE_ARTIFACT_SCHEMA = "neuros-mechint.feature-correspondence-artifact.v1"
CORRESPONDENCE_STUDY_SCHEMA = "neuros-mechint.feature-correspondence-study.v1"


class CorrespondenceSplit(str, Enum):
    """Discovery versus held-out validation examples."""

    DISCOVERY = "discovery"
    VALIDATION = "validation"


class CorrespondenceKind(str, Enum):
    """Supported correspondence shapes."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    SUBSPACE = "subspace"


@dataclass(frozen=True, slots=True)
class FeatureSpaceIdentity:
    """Immutable scientific identity for a representation feature space."""

    space_id: str
    model_id: str
    model_revision: str
    representation_id: str
    feature_names: tuple[str, ...]
    architecture: str
    tokenizer_id: str
    tokenizer_revision: str
    dataset_id: str
    dataset_revision: str
    session_id: str
    checkpoint: str
    subject_id: str | None = None
    feature_semantics: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "space_id",
            "model_id",
            "model_revision",
            "representation_id",
            "architecture",
            "tokenizer_id",
            "tokenizer_revision",
            "dataset_id",
            "dataset_revision",
            "session_id",
            "checkpoint",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        feature_names = tuple(dict.fromkeys(str(item) for item in self.feature_names))
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if len(feature_names) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        semantics = {str(key): str(value) for key, value in self.feature_semantics.items()}
        unknown = sorted(set(semantics) - set(feature_names))
        if unknown:
            raise ValueError(f"feature_semantics contains unknown feature(s): {unknown}")
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "feature_semantics", MappingProxyType(semantics))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "checkpoint": self.checkpoint,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "feature_names": list(self.feature_names),
            "feature_semantics": dict(self.feature_semantics),
            "metadata": dict(self.metadata),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "representation_id": self.representation_id,
            "session_id": self.session_id,
            "space_id": self.space_id,
            "subject_id": self.subject_id,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True, slots=True)
class FactorialCorrespondenceOrigin:
    """Optional provenance link to an estimable v0.7 factorial contrast."""

    factorial_study_fingerprint: str
    contrast_id: str
    cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.factorial_study_fingerprint or not self.contrast_id:
            raise ValueError("factorial origin requires a study fingerprint and contrast_id")
        if not self.cell_ids:
            raise ValueError("factorial origin requires at least one source cell")
        object.__setattr__(self, "cell_ids", tuple(str(item) for item in self.cell_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_ids": list(self.cell_ids),
            "contrast_id": self.contrast_id,
            "factorial_study_fingerprint": self.factorial_study_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class FeaturePairExample:
    """One semantically paired source/target representation observation."""

    example_id: str
    semantic_trial_id: str
    split: CorrespondenceSplit | str
    partition_id: str
    source_activation: np.ndarray
    target_activation: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id or not self.semantic_trial_id or not self.partition_id:
            raise ValueError("example_id, semantic_trial_id, and partition_id must be non-empty")
        split = CorrespondenceSplit(self.split)
        source = np.asarray(self.source_activation, dtype=np.float64).reshape(-1).copy()
        target = np.asarray(self.target_activation, dtype=np.float64).reshape(-1).copy()
        if not source.size or not target.size:
            raise ValueError("source_activation and target_activation must be non-empty vectors")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("feature activations must be finite")
        source.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "source_activation", source)
        object.__setattr__(self, "target_activation", target)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def pair_hash(self) -> str:
        return stable_hash(
            {
                "source_activation": self.source_activation,
                "target_activation": self.target_activation,
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "metadata": dict(self.metadata),
            "pair_hash": self.pair_hash,
            "partition_id": self.partition_id,
            "semantic_trial_id": self.semantic_trial_id,
            "split": self.split.value,
        }


@dataclass(frozen=True, slots=True)
class FeatureCorrespondencePolicy:
    """Promotion thresholds for causal correspondence."""

    min_discovery_examples: int = 6
    min_validation_examples: int = 6
    min_valid_transfer_fraction: float = 0.80
    min_validation_predictive_r2: float = 0.50
    min_median_causal_recovery: float = 0.75
    min_source_effect: float = 1e-6
    min_target_effect: float = 1e-6
    min_random_percentile: float = 0.90
    min_shuffled_margin: float = 0.20
    min_random_margin: float = 0.20
    max_discovery_validation_r2_drop: float = 0.40
    min_intervention_effect_correlation: float | None = None
    reject_duplicate_activation_content: bool = True

    def __post_init__(self) -> None:
        if self.min_discovery_examples < 2 or self.min_validation_examples < 2:
            raise ValueError("discovery and validation example minima must be at least 2")
        for name in ("min_valid_transfer_fraction", "min_random_percentile"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in (
            "min_validation_predictive_r2",
            "min_median_causal_recovery",
            "min_source_effect",
            "min_target_effect",
            "min_shuffled_margin",
            "min_random_margin",
            "max_discovery_validation_r2_drop",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.min_intervention_effect_correlation is not None:
            value = float(self.min_intervention_effect_correlation)
            if not -1.0 <= value <= 1.0:
                raise ValueError("min_intervention_effect_correlation must lie in [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CONTEXT_AXES = (
    "model_id",
    "model_revision",
    "architecture",
    "tokenizer_id",
    "tokenizer_revision",
    "dataset_id",
    "dataset_revision",
    "session_id",
    "subject_id",
    "checkpoint",
)


@dataclass(frozen=True, slots=True)
class FeatureCorrespondenceSpec:
    """Frozen design for one source-to-target feature correspondence study."""

    study_id: str
    source_space: FeatureSpaceIdentity
    target_space: FeatureSpaceIdentity
    source_features: tuple[str, ...]
    target_features: tuple[str, ...]
    kind: CorrespondenceKind | str
    discovery_partition_id: str
    validation_partition_id: str
    declared_context_differences: tuple[str, ...]
    ridge_alpha: float = 1e-6
    random_controls: int = 16
    seed: int = 0
    higher_is_better: bool = True
    policy: FeatureCorrespondencePolicy = field(default_factory=FeatureCorrespondencePolicy)
    factorial_origin: FactorialCorrespondenceOrigin | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CORRESPONDENCE_STUDY_SCHEMA

    def __post_init__(self) -> None:
        if not self.study_id or not self.discovery_partition_id or not self.validation_partition_id:
            raise ValueError("study and partition IDs must be non-empty")
        if self.discovery_partition_id == self.validation_partition_id:
            raise ValueError("discovery and validation partition IDs must differ")
        kind = CorrespondenceKind(self.kind)
        source_features = tuple(dict.fromkeys(str(item) for item in self.source_features))
        target_features = tuple(dict.fromkeys(str(item) for item in self.target_features))
        if not source_features or not target_features:
            raise ValueError("source_features and target_features must be non-empty")
        missing_source = sorted(set(source_features) - set(self.source_space.feature_names))
        missing_target = sorted(set(target_features) - set(self.target_space.feature_names))
        if missing_source or missing_target:
            raise ValueError(
                f"candidate contains unknown features: source={missing_source}, target={missing_target}"
            )
        if kind is CorrespondenceKind.ONE_TO_ONE and (
            len(source_features) != 1 or len(target_features) != 1
        ):
            raise ValueError("one_to_one correspondence requires exactly 1 source and 1 target")
        if kind is CorrespondenceKind.ONE_TO_MANY and (
            len(source_features) != 1 or len(target_features) < 2
        ):
            raise ValueError("one_to_many correspondence requires 1 source and at least 2 targets")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative")
        if self.random_controls <= 0:
            raise ValueError("random_controls must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        declared = tuple(dict.fromkeys(str(item) for item in self.declared_context_differences))
        unknown = sorted(set(declared) - set(_CONTEXT_AXES))
        if unknown:
            raise ValueError(f"unknown declared context difference(s): {unknown}")
        actual = {
            axis
            for axis in _CONTEXT_AXES
            if getattr(self.source_space, axis) != getattr(self.target_space, axis)
        }
        undeclared = sorted(actual - set(declared))
        if undeclared:
            raise ValueError(f"undeclared source/target context difference(s): {undeclared}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_features", source_features)
        object.__setattr__(self, "target_features", target_features)
        object.__setattr__(self, "declared_context_differences", declared)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_context_differences": list(self.declared_context_differences),
            "discovery_partition_id": self.discovery_partition_id,
            "factorial_origin": (
                None if self.factorial_origin is None else self.factorial_origin.to_dict()
            ),
            "higher_is_better": self.higher_is_better,
            "kind": self.kind.value,
            "metadata": dict(self.metadata),
            "policy": self.policy.to_dict(),
            "random_controls": self.random_controls,
            "ridge_alpha": self.ridge_alpha,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "source_features": list(self.source_features),
            "source_space": self.source_space.to_dict(),
            "study_id": self.study_id,
            "target_features": list(self.target_features),
            "target_space": self.target_space.to_dict(),
            "validation_partition_id": self.validation_partition_id,
        }


@dataclass(frozen=True, slots=True)
class FeatureCorrespondenceCandidate:
    """Frozen linear mapping discovered without held-out examples."""

    candidate_id: str
    kind: CorrespondenceKind
    source_features: tuple[str, ...]
    target_features: tuple[str, ...]
    mapping_matrix: tuple[tuple[float, ...], ...]
    intercept: tuple[float, ...]
    discovery_method: str
    discovery_example_ids: tuple[str, ...]
    activation_correlation: float
    geometric_similarity: float
    semantic_label_overlap: float | None
    predictive_r2: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.mapping_matrix, dtype=np.float64)
        if matrix.shape != (len(self.target_features), len(self.source_features)):
            raise ValueError(
                "mapping_matrix shape must equal (len(target_features), len(source_features))"
            )
        if len(self.intercept) != len(self.target_features):
            raise ValueError("intercept length must match target_features")
        if not np.isfinite(matrix).all() or not np.isfinite(np.asarray(self.intercept)).all():
            raise ValueError("mapping coefficients must be finite")

    def predict(self, source_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(source_values, dtype=np.float64).reshape(-1)
        if values.size != len(self.source_features):
            raise ValueError("source_values length does not match candidate source_features")
        matrix = np.asarray(self.mapping_matrix, dtype=np.float64)
        intercept = np.asarray(self.intercept, dtype=np.float64)
        return matrix @ values + intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_correlation": self.activation_correlation,
            "candidate_id": self.candidate_id,
            "discovery_example_ids": list(self.discovery_example_ids),
            "discovery_method": self.discovery_method,
            "geometric_similarity": self.geometric_similarity,
            "intercept": list(self.intercept),
            "kind": self.kind.value,
            "mapping_matrix": [list(row) for row in self.mapping_matrix],
            "predictive_r2": self.predictive_r2,
            "semantic_label_overlap": self.semantic_label_overlap,
            "source_features": list(self.source_features),
            "target_features": list(self.target_features),
        }


@dataclass(frozen=True, slots=True)
class CausalSubstitutionMetrics:
    """Model metrics returned by a causal substitution backend."""

    source_clean_metric: float
    source_ablated_metric: float
    target_clean_metric: float
    target_ablated_metric: float
    target_substituted_metric: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.source_clean_metric,
                self.source_ablated_metric,
                self.target_clean_metric,
                self.target_ablated_metric,
                self.target_substituted_metric,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("causal substitution metrics must be finite")


class CausalSubstitutionEvaluator(Protocol):
    """Backend contract for held-out paired feature interventions."""

    def __call__(
        self,
        *,
        target_example_id: str,
        source_example_id: str,
        source_features: tuple[str, ...],
        target_features: tuple[str, ...],
        replacement_values: np.ndarray,
    ) -> CausalSubstitutionMetrics:
        ...


@dataclass(frozen=True, slots=True)
class CausalTransferCase:
    """One candidate or matched-control held-out substitution."""

    target_example_id: str
    source_example_id: str
    control_kind: str
    control_id: str
    source_features: tuple[str, ...]
    target_features: tuple[str, ...]
    valid: bool
    invalid_reason: str | None
    source_effect: float
    target_effect: float
    recovery: float | None
    causal_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorrespondenceValidationMetrics:
    """Held-out predictive, geometric, intervention, and causal-transfer evidence."""

    activation_correlation: float
    geometric_similarity: float
    predictive_r2: float
    discovery_to_validation_r2_drop: float
    intervention_effect_correlation: float | None
    valid_transfer_fraction: float
    median_source_effect: float
    median_target_effect: float
    median_causal_recovery: float
    median_causal_score: float
    shuffled_median_causal_score: float
    random_median_causal_score: float
    random_control_percentile: float | None
    shuffled_margin: float
    random_margin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorrespondencePromotionDecision:
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons)}


@dataclass(frozen=True, slots=True)
class UnmatchedFeatures:
    source_features: tuple[str, ...]
    target_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_features": list(self.source_features),
            "target_features": list(self.target_features),
        }


@dataclass(frozen=True, slots=True)
class FeatureCorrespondenceResult:
    """Complete v0.8 correspondence result without raw activation payloads."""

    spec: FeatureCorrespondenceSpec
    candidate: FeatureCorrespondenceCandidate
    example_summaries: tuple[Mapping[str, Any], ...]
    validation_metrics: CorrespondenceValidationMetrics
    transfer_cases: tuple[CausalTransferCase, ...]
    promotion: CorrespondencePromotionDecision
    unmatched_features: UnmatchedFeatures

    @property
    def study_fingerprint(self) -> str:
        return stable_hash(
            {
                "candidate": self.candidate.to_dict(),
                "examples": [dict(item) for item in self.example_summaries],
                "promotion": self.promotion.to_dict(),
                "spec": self.spec.to_dict(),
                "transfer_cases": [item.to_dict() for item in self.transfer_cases],
                "validation_metrics": self.validation_metrics.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "example_summaries": [dict(item) for item in self.example_summaries],
            "promotion": self.promotion.to_dict(),
            "schema_version": self.spec.schema_version,
            "spec": self.spec.to_dict(),
            "study_fingerprint": self.study_fingerprint,
            "transfer_cases": [item.to_dict() for item in self.transfer_cases],
            "unmatched_features": self.unmatched_features.to_dict(),
            "validation_metrics": self.validation_metrics.to_dict(),
        }


def _feature_indices(space: FeatureSpaceIdentity, names: Sequence[str]) -> tuple[int, ...]:
    index = {name: i for i, name in enumerate(space.feature_names)}
    return tuple(index[name] for name in names)


def _matrix(
    examples: Sequence[FeaturePairExample],
    *,
    side: str,
    indices: Sequence[int],
) -> np.ndarray:
    if side not in {"source", "target"}:
        raise ValueError("side must be 'source' or 'target'")
    values = []
    for example in examples:
        vector = example.source_activation if side == "source" else example.target_activation
        values.append(vector[np.asarray(indices, dtype=np.int64)])
    return np.asarray(values, dtype=np.float64)


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError("linear CKA requires the same number of observations")
    x = left - left.mean(axis=0, keepdims=True)
    y = right - right.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    left_norm = float(np.linalg.norm(x.T @ x, ord="fro"))
    right_norm = float(np.linalg.norm(y.T @ y, ord="fro"))
    denominator = left_norm * right_norm
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _predictive_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = float(np.sum((actual - predicted) ** 2))
    centered = actual - actual.mean(axis=0, keepdims=True)
    total = float(np.sum(centered**2))
    if total <= 1e-12:
        return 1.0 if residual <= 1e-12 else 0.0
    return 1.0 - residual / total


def _semantic_overlap(
    source_space: FeatureSpaceIdentity,
    target_space: FeatureSpaceIdentity,
    source_features: Sequence[str],
    target_features: Sequence[str],
) -> float | None:
    source_labels = {
        source_space.feature_semantics[name].strip().lower()
        for name in source_features
        if source_space.feature_semantics.get(name, "").strip()
    }
    target_labels = {
        target_space.feature_semantics[name].strip().lower()
        for name in target_features
        if target_space.feature_semantics.get(name, "").strip()
    }
    if not source_labels or not target_labels:
        return None
    return len(source_labels & target_labels) / len(source_labels | target_labels)


def _fit_linear_mapping(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = source.mean(axis=0)
    y_mean = target.mean(axis=0)
    x_centered = source - x_mean
    y_centered = target - y_mean
    gram = x_centered.T @ x_centered
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * ridge_alpha
    coefficients = np.linalg.lstsq(
        gram + regularizer,
        x_centered.T @ y_centered,
        rcond=None,
    )[0]
    matrix = coefficients.T
    intercept = y_mean - matrix @ x_mean
    return matrix, intercept


def _candidate_from_features(
    *,
    spec: FeatureCorrespondenceSpec,
    discovery_examples: Sequence[FeaturePairExample],
    source_features: tuple[str, ...],
    candidate_id: str,
) -> FeatureCorrespondenceCandidate:
    source_indices = _feature_indices(spec.source_space, source_features)
    target_indices = _feature_indices(spec.target_space, spec.target_features)
    source = _matrix(discovery_examples, side="source", indices=source_indices)
    target = _matrix(discovery_examples, side="target", indices=target_indices)
    matrix, intercept = _fit_linear_mapping(source, target, ridge_alpha=spec.ridge_alpha)
    predicted = source @ matrix.T + intercept
    return FeatureCorrespondenceCandidate(
        candidate_id=candidate_id,
        kind=spec.kind,
        source_features=source_features,
        target_features=spec.target_features,
        mapping_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        intercept=tuple(float(value) for value in intercept),
        discovery_method="ridge_linear_discovery_only",
        discovery_example_ids=tuple(item.example_id for item in discovery_examples),
        activation_correlation=_correlation(predicted, target),
        geometric_similarity=_linear_cka(source, target),
        semantic_label_overlap=_semantic_overlap(
            spec.source_space,
            spec.target_space,
            source_features,
            spec.target_features,
        ),
        predictive_r2=_predictive_r2(target, predicted),
    )


def fit_feature_correspondence_candidate(
    spec: FeatureCorrespondenceSpec,
    discovery_examples: Sequence[FeaturePairExample],
) -> FeatureCorrespondenceCandidate:
    """Fit the preregistered source->target linear mapping on discovery only."""

    return _candidate_from_features(
        spec=spec,
        discovery_examples=discovery_examples,
        source_features=spec.source_features,
        candidate_id=f"{spec.study_id}:candidate",
    )


def _validate_examples(
    spec: FeatureCorrespondenceSpec,
    examples: Sequence[FeaturePairExample],
) -> tuple[tuple[FeaturePairExample, ...], tuple[FeaturePairExample, ...]]:
    if not examples:
        raise ValueError("correspondence study requires paired examples")
    ids = [item.example_id for item in examples]
    semantic_ids = [item.semantic_trial_id for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("example_id values must be unique")
    if len(semantic_ids) != len(set(semantic_ids)):
        raise ValueError("semantic_trial_id values must be unique across discovery and validation")

    discovery = []
    validation = []
    discovery_hashes: set[str] = set()
    validation_hashes: set[str] = set()
    for example in examples:
        if example.source_activation.size != len(spec.source_space.feature_names):
            raise ValueError(
                f"source activation length for {example.example_id!r} does not match source feature space"
            )
        if example.target_activation.size != len(spec.target_space.feature_names):
            raise ValueError(
                f"target activation length for {example.example_id!r} does not match target feature space"
            )
        if example.split is CorrespondenceSplit.DISCOVERY:
            if example.partition_id != spec.discovery_partition_id:
                raise ValueError("discovery example partition_id does not match study specification")
            discovery.append(example)
            discovery_hashes.add(example.pair_hash)
        else:
            if example.partition_id != spec.validation_partition_id:
                raise ValueError("validation example partition_id does not match study specification")
            validation.append(example)
            validation_hashes.add(example.pair_hash)

    if len(discovery) < spec.policy.min_discovery_examples:
        raise ValueError(
            f"need at least {spec.policy.min_discovery_examples} discovery examples, got {len(discovery)}"
        )
    if len(validation) < spec.policy.min_validation_examples:
        raise ValueError(
            f"need at least {spec.policy.min_validation_examples} validation examples, got {len(validation)}"
        )
    if (
        spec.policy.reject_duplicate_activation_content
        and discovery_hashes.intersection(validation_hashes)
    ):
        raise ValueError("exact paired activation content appears in both discovery and validation")
    return tuple(discovery), tuple(validation)


def _random_source_feature_sets(
    spec: FeatureCorrespondenceSpec,
) -> tuple[tuple[str, ...], ...]:
    size = len(spec.source_features)
    universe = tuple(spec.source_space.feature_names)
    candidate_set = frozenset(spec.source_features)
    possible_count = comb(len(universe), size) - 1
    if possible_count <= 0:
        return ()

    rng = np.random.default_rng(spec.seed)
    target_count = min(spec.random_controls, possible_count)
    enumeration_limit = max(256, spec.random_controls * 4)
    if possible_count <= enumeration_limit:
        possibilities = [
            tuple(items)
            for items in combinations(universe, size)
            if frozenset(items) != candidate_set
        ]
        if len(possibilities) <= target_count:
            return tuple(possibilities)
        chosen = rng.choice(len(possibilities), size=target_count, replace=False)
        return tuple(possibilities[int(index)] for index in sorted(chosen))

    chosen_sets: set[tuple[str, ...]] = set()
    max_attempts = max(1000, target_count * 100)
    attempts = 0
    while len(chosen_sets) < target_count and attempts < max_attempts:
        indices = tuple(
            sorted(
                int(index)
                for index in rng.choice(len(universe), size=size, replace=False)
            )
        )
        features = tuple(universe[index] for index in indices)
        if frozenset(features) != candidate_set:
            chosen_sets.add(features)
        attempts += 1
    if len(chosen_sets) != target_count:
        raise RuntimeError(
            "failed to sample the requested number of unique random source feature controls"
        )
    return tuple(sorted(chosen_sets))


def _deranged_donor_indices(count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("shuffle control requires at least two validation examples")
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(1, count))
    return np.roll(np.arange(count), offset)


def _orient(value: float, higher_is_better: bool) -> float:
    return float(value) if higher_is_better else -float(value)


def _transfer_case(
    *,
    spec: FeatureCorrespondenceSpec,
    evaluator: CausalSubstitutionEvaluator,
    target_example_id: str,
    source_example_id: str,
    source_features: tuple[str, ...],
    target_features: tuple[str, ...],
    replacement_values: np.ndarray,
    control_kind: str,
    control_id: str,
) -> CausalTransferCase:
    metrics = evaluator(
        target_example_id=target_example_id,
        source_example_id=source_example_id,
        source_features=source_features,
        target_features=target_features,
        replacement_values=np.asarray(replacement_values, dtype=np.float64),
    )
    source_clean = _orient(metrics.source_clean_metric, spec.higher_is_better)
    source_ablated = _orient(metrics.source_ablated_metric, spec.higher_is_better)
    target_clean = _orient(metrics.target_clean_metric, spec.higher_is_better)
    target_ablated = _orient(metrics.target_ablated_metric, spec.higher_is_better)
    target_substituted = _orient(metrics.target_substituted_metric, spec.higher_is_better)
    source_effect = source_clean - source_ablated
    target_effect = target_clean - target_ablated
    if target_effect <= 1e-12:
        return CausalTransferCase(
            target_example_id=target_example_id,
            source_example_id=source_example_id,
            control_kind=control_kind,
            control_id=control_id,
            source_features=source_features,
            target_features=target_features,
            valid=False,
            invalid_reason="target feature ablation does not reduce the oriented metric",
            source_effect=float(source_effect),
            target_effect=float(target_effect),
            recovery=None,
            causal_score=0.0,
        )
    recovery = (target_substituted - target_ablated) / target_effect
    relevance = (
        source_effect >= spec.policy.min_source_effect
        and target_effect >= spec.policy.min_target_effect
    )
    causal_score = float(np.clip(recovery, 0.0, 1.0)) if relevance else 0.0
    return CausalTransferCase(
        target_example_id=target_example_id,
        source_example_id=source_example_id,
        control_kind=control_kind,
        control_id=control_id,
        source_features=source_features,
        target_features=target_features,
        valid=True,
        invalid_reason=None,
        source_effect=float(source_effect),
        target_effect=float(target_effect),
        recovery=float(recovery),
        causal_score=causal_score,
    )


def _validation_similarity(
    spec: FeatureCorrespondenceSpec,
    candidate: FeatureCorrespondenceCandidate,
    validation_examples: Sequence[FeaturePairExample],
) -> tuple[float, float, float]:
    source_indices = _feature_indices(spec.source_space, candidate.source_features)
    target_indices = _feature_indices(spec.target_space, candidate.target_features)
    source = _matrix(validation_examples, side="source", indices=source_indices)
    target = _matrix(validation_examples, side="target", indices=target_indices)
    matrix = np.asarray(candidate.mapping_matrix, dtype=np.float64)
    intercept = np.asarray(candidate.intercept, dtype=np.float64)
    predicted = source @ matrix.T + intercept
    return (
        _correlation(predicted, target),
        _linear_cka(source, target),
        _predictive_r2(target, predicted),
    )


def _effect_correlation(cases: Sequence[CausalTransferCase]) -> float | None:
    valid = [item for item in cases if item.valid]
    if len(valid) < 3:
        return None
    source = np.asarray([item.source_effect for item in valid], dtype=np.float64)
    target = np.asarray([item.target_effect for item in valid], dtype=np.float64)
    if np.std(source) <= 1e-12 or np.std(target) <= 1e-12:
        return None
    return float(np.corrcoef(source, target)[0, 1])


def _median_score(cases: Sequence[CausalTransferCase]) -> float:
    if not cases:
        return 0.0
    return float(np.median([item.causal_score for item in cases]))


def _strict_percentile(candidate: float, controls: Sequence[float]) -> float | None:
    if not controls:
        return None
    return float(np.mean(np.asarray(controls, dtype=np.float64) < float(candidate)))


def _promotion(
    spec: FeatureCorrespondenceSpec,
    metrics: CorrespondenceValidationMetrics,
) -> CorrespondencePromotionDecision:
    reasons = []
    policy = spec.policy
    if metrics.valid_transfer_fraction < policy.min_valid_transfer_fraction:
        reasons.append(
            f"valid transfer fraction {metrics.valid_transfer_fraction:.3f} < "
            f"{policy.min_valid_transfer_fraction:.3f}"
        )
    if metrics.predictive_r2 < policy.min_validation_predictive_r2:
        reasons.append(
            f"validation predictive R2 {metrics.predictive_r2:.3f} < "
            f"{policy.min_validation_predictive_r2:.3f}"
        )
    if metrics.median_source_effect < policy.min_source_effect:
        reasons.append(
            f"median source intervention effect {metrics.median_source_effect:.6g} < "
            f"{policy.min_source_effect:.6g}"
        )
    if metrics.median_target_effect < policy.min_target_effect:
        reasons.append(
            f"median target intervention effect {metrics.median_target_effect:.6g} < "
            f"{policy.min_target_effect:.6g}"
        )
    if metrics.median_causal_recovery < policy.min_median_causal_recovery:
        reasons.append(
            f"median causal recovery {metrics.median_causal_recovery:.3f} < "
            f"{policy.min_median_causal_recovery:.3f}"
        )
    if metrics.random_control_percentile is None:
        reasons.append("no same-cardinality random-source controls were available")
    elif metrics.random_control_percentile < policy.min_random_percentile:
        reasons.append(
            f"random-control percentile {metrics.random_control_percentile:.3f} < "
            f"{policy.min_random_percentile:.3f}"
        )
    if metrics.shuffled_margin < policy.min_shuffled_margin:
        reasons.append(
            f"shuffled-pair causal margin {metrics.shuffled_margin:.3f} < "
            f"{policy.min_shuffled_margin:.3f}"
        )
    if metrics.random_margin < policy.min_random_margin:
        reasons.append(
            f"random-source causal margin {metrics.random_margin:.3f} < "
            f"{policy.min_random_margin:.3f}"
        )
    if metrics.discovery_to_validation_r2_drop > policy.max_discovery_validation_r2_drop:
        reasons.append(
            f"discovery-to-validation R2 drop {metrics.discovery_to_validation_r2_drop:.3f} > "
            f"{policy.max_discovery_validation_r2_drop:.3f}"
        )
    if policy.min_intervention_effect_correlation is not None:
        if metrics.intervention_effect_correlation is None:
            reasons.append("intervention-effect correlation is undefined")
        elif metrics.intervention_effect_correlation < policy.min_intervention_effect_correlation:
            reasons.append(
                f"intervention-effect correlation {metrics.intervention_effect_correlation:.3f} < "
                f"{policy.min_intervention_effect_correlation:.3f}"
            )
    return CorrespondencePromotionDecision(passed=not reasons, reasons=tuple(reasons))


def run_feature_correspondence_study(
    spec: FeatureCorrespondenceSpec,
    examples: Sequence[FeaturePairExample],
    *,
    evaluator: CausalSubstitutionEvaluator,
    candidate_fit: Callable[
        [FeatureCorrespondenceSpec, Sequence[FeaturePairExample]],
        FeatureCorrespondenceCandidate,
    ] = fit_feature_correspondence_candidate,
) -> FeatureCorrespondenceResult:
    """Fit correspondence on discovery only and test causal substitution held out."""

    discovery, validation = _validate_examples(spec, examples)
    candidate = candidate_fit(spec, discovery)
    if set(candidate.discovery_example_ids) - {item.example_id for item in discovery}:
        raise ValueError("candidate references examples outside the discovery split")
    if candidate.source_features != spec.source_features:
        raise ValueError("candidate source_features do not match the preregistered study")
    if candidate.target_features != spec.target_features:
        raise ValueError("candidate target_features do not match the preregistered study")

    activation_corr, geometric_similarity, validation_r2 = _validation_similarity(
        spec, candidate, validation
    )
    transfer_cases: list[CausalTransferCase] = []

    source_indices = _feature_indices(spec.source_space, candidate.source_features)
    for example in validation:
        source_values = example.source_activation[np.asarray(source_indices, dtype=np.int64)]
        mapped = candidate.predict(source_values)
        transfer_cases.append(
            _transfer_case(
                spec=spec,
                evaluator=evaluator,
                target_example_id=example.example_id,
                source_example_id=example.example_id,
                source_features=candidate.source_features,
                target_features=candidate.target_features,
                replacement_values=mapped,
                control_kind="candidate",
                control_id=candidate.candidate_id,
            )
        )

    shuffled_indices = _deranged_donor_indices(len(validation), spec.seed + 17)
    for target_index, source_index in enumerate(shuffled_indices):
        target_example = validation[target_index]
        source_example = validation[int(source_index)]
        source_values = source_example.source_activation[np.asarray(source_indices, dtype=np.int64)]
        mapped = candidate.predict(source_values)
        transfer_cases.append(
            _transfer_case(
                spec=spec,
                evaluator=evaluator,
                target_example_id=target_example.example_id,
                source_example_id=source_example.example_id,
                source_features=candidate.source_features,
                target_features=candidate.target_features,
                replacement_values=mapped,
                control_kind="shuffled_pair",
                control_id="shuffled_pair",
            )
        )

    random_candidates = [
        _candidate_from_features(
            spec=spec,
            discovery_examples=discovery,
            source_features=features,
            candidate_id=f"{spec.study_id}:random_source:{index}",
        )
        for index, features in enumerate(_random_source_feature_sets(spec))
    ]
    for random_candidate in random_candidates:
        indices = _feature_indices(spec.source_space, random_candidate.source_features)
        for example in validation:
            source_values = example.source_activation[np.asarray(indices, dtype=np.int64)]
            mapped = random_candidate.predict(source_values)
            transfer_cases.append(
                _transfer_case(
                    spec=spec,
                    evaluator=evaluator,
                    target_example_id=example.example_id,
                    source_example_id=example.example_id,
                    source_features=random_candidate.source_features,
                    target_features=random_candidate.target_features,
                    replacement_values=mapped,
                    control_kind="random_source",
                    control_id=random_candidate.candidate_id,
                )
            )

    candidate_cases = [item for item in transfer_cases if item.control_kind == "candidate"]
    shuffled_cases = [item for item in transfer_cases if item.control_kind == "shuffled_pair"]
    random_cases = [item for item in transfer_cases if item.control_kind == "random_source"]
    valid_candidate = [item for item in candidate_cases if item.valid]
    valid_fraction = len(valid_candidate) / len(candidate_cases)
    recoveries = [float(item.recovery) for item in valid_candidate if item.recovery is not None]
    median_recovery = float(np.median(recoveries)) if recoveries else 0.0
    candidate_score = _median_score(candidate_cases)
    shuffled_score = _median_score(shuffled_cases)

    random_by_id: dict[str, list[CausalTransferCase]] = defaultdict(list)
    for item in random_cases:
        random_by_id[item.control_id].append(item)
    random_scores = [_median_score(items) for _, items in sorted(random_by_id.items())]
    random_median = float(np.median(random_scores)) if random_scores else 0.0
    percentile = _strict_percentile(candidate_score, random_scores)

    metrics = CorrespondenceValidationMetrics(
        activation_correlation=activation_corr,
        geometric_similarity=geometric_similarity,
        predictive_r2=validation_r2,
        discovery_to_validation_r2_drop=float(candidate.predictive_r2 - validation_r2),
        intervention_effect_correlation=_effect_correlation(candidate_cases),
        valid_transfer_fraction=float(valid_fraction),
        median_source_effect=float(np.median([item.source_effect for item in candidate_cases])),
        median_target_effect=float(np.median([item.target_effect for item in candidate_cases])),
        median_causal_recovery=median_recovery,
        median_causal_score=candidate_score,
        shuffled_median_causal_score=shuffled_score,
        random_median_causal_score=random_median,
        random_control_percentile=percentile,
        shuffled_margin=float(candidate_score - shuffled_score),
        random_margin=float(candidate_score - random_median),
    )
    promotion = _promotion(spec, metrics)
    unmatched = UnmatchedFeatures(
        source_features=tuple(
            name for name in spec.source_space.feature_names if name not in candidate.source_features
        ),
        target_features=tuple(
            name for name in spec.target_space.feature_names if name not in candidate.target_features
        ),
    )
    return FeatureCorrespondenceResult(
        spec=spec,
        candidate=candidate,
        example_summaries=tuple(item.summary() for item in examples),
        validation_metrics=metrics,
        transfer_cases=tuple(transfer_cases),
        promotion=promotion,
        unmatched_features=unmatched,
    )


def write_correspondence_artifact(
    result: FeatureCorrespondenceResult,
    path: str | Path,
) -> str:
    """Write a self-checking correspondence JSON artifact."""

    payload_result = result.to_dict()
    artifact_hash = stable_hash(payload_result)
    payload = {
        "artifact_hash": artifact_hash,
        "artifact_schema": CORRESPONDENCE_ARTIFACT_SCHEMA,
        "result": payload_result,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_hash


def read_correspondence_artifact(path: str | Path) -> dict[str, Any]:
    """Validate and return a serialized correspondence result."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_schema") != CORRESPONDENCE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported correspondence artifact schema")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("correspondence artifact result must be an object")
    if stable_hash(result) != payload.get("artifact_hash"):
        raise ValueError("correspondence artifact hash mismatch")
    if result.get("schema_version") != CORRESPONDENCE_STUDY_SCHEMA:
        raise ValueError("unsupported correspondence study schema")
    return result
