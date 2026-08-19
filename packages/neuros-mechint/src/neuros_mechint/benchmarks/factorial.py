"""Preregistered factorial architecture x tokenizer mechanism studies.

v0.7 treats each architecture/tokenizer condition as a completed held-out
evidence-pack cell. This module never re-discovers circuits. It asks whether
differences between already-frozen cell-level mechanisms are estimable under a
matched factorial design, and it refuses contrasts with missing or confounded
cells.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, median
from types import MappingProxyType
from typing import Any

import numpy as np

from neuros_mechint.core.manifest import stable_hash

from .evidence_pack import EvidencePackResult
from .stability import EffectMapStability, compare_effect_maps

FACTORIAL_ARTIFACT_SCHEMA = "neuros-mechint.factorial-mechanism-artifact.v1"
FACTORIAL_STUDY_SCHEMA = "neuros-mechint.factorial-mechanism-study.v1"


class FactorialContrastKind(str, Enum):
    """Supported preregistered contrasts."""

    ARCHITECTURE_MAIN = "architecture_main"
    TOKENIZER_MAIN = "tokenizer_main"
    ARCHITECTURE_TOKENIZER_INTERACTION = "architecture_tokenizer_interaction"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True, slots=True)
class MatchedCovariate:
    """A design covariate that must match before a contrast is estimable."""

    name: str
    tolerance: float = 0.0
    relative: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("covariate name must be non-empty")
        if self.tolerance < 0.0:
            raise ValueError("covariate tolerance must be non-negative")

    def difference(self, left: Any, right: Any) -> float | None:
        if isinstance(left, bool) or isinstance(right, bool):
            return None
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(right) - float(left))
        return None

    def matches(self, left: Any, right: Any) -> bool:
        difference = self.difference(left, right)
        if difference is None:
            return left == right
        if not self.relative:
            return difference <= self.tolerance
        scale = max(abs(float(left)), abs(float(right)), 1e-12)
        return difference / scale <= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FactorialCellSpec:
    """One intended cell in an architecture x tokenizer factorial design."""

    cell_id: str
    architecture: str
    tokenizer_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dataset_id: str
    dataset_revision: str
    session_id: str
    metric_name: str
    discovery_method: str
    discovery_partition_id: str
    validation_partition_id: str
    training_seed: int
    checkpoint: str
    checkpoint_maturity: float
    target_universe: tuple[str, ...]
    subject_id: str | None = None
    covariates: Mapping[str, Any] = field(default_factory=dict)
    available: bool = True
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        required = (
            "cell_id",
            "architecture",
            "tokenizer_id",
            "model_id",
            "dataset_id",
            "session_id",
            "metric_name",
            "discovery_method",
            "discovery_partition_id",
            "validation_partition_id",
            "checkpoint",
        )
        for name in required:
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")
        if not np.isfinite(float(self.checkpoint_maturity)):
            raise ValueError("checkpoint_maturity must be finite")
        targets = tuple(dict.fromkeys(str(item) for item in self.target_universe))
        if not targets:
            raise ValueError("target_universe must not be empty")
        if self.available:
            for name in ("model_revision", "tokenizer_revision", "dataset_revision"):
                if not getattr(self, name):
                    raise ValueError(f"available cell requires pinned {name}")
            if self.missing_reason is not None:
                raise ValueError("available cell cannot declare missing_reason")
        elif not self.missing_reason:
            raise ValueError("missing cell must declare missing_reason")
        object.__setattr__(self, "target_universe", targets)
        object.__setattr__(self, "covariates", MappingProxyType(dict(self.covariates)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "available": self.available,
            "cell_id": self.cell_id,
            "checkpoint": self.checkpoint,
            "checkpoint_maturity": self.checkpoint_maturity,
            "covariates": dict(self.covariates),
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "discovery_method": self.discovery_method,
            "discovery_partition_id": self.discovery_partition_id,
            "metric_name": self.metric_name,
            "missing_reason": self.missing_reason,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "target_universe": list(self.target_universe),
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "training_seed": self.training_seed,
            "validation_partition_id": self.validation_partition_id,
        }


@dataclass(frozen=True, slots=True)
class FactorialContrastSpec:
    """A preregistered matched contrast over named factorial levels."""

    contrast_id: str
    kind: FactorialContrastKind | str
    architectures: tuple[str, ...]
    tokenizers: tuple[str, ...]
    checkpoints: tuple[str, ...] = ()
    fixed_axes: Mapping[str, Any] = field(default_factory=dict)
    replication_group: str | None = None

    def __post_init__(self) -> None:
        if not self.contrast_id:
            raise ValueError("contrast_id must be non-empty")
        kind = FactorialContrastKind(self.kind)
        object.__setattr__(self, "kind", kind)
        architectures = tuple(self.architectures)
        tokenizers = tuple(self.tokenizers)
        checkpoints = tuple(self.checkpoints)
        if kind is FactorialContrastKind.ARCHITECTURE_MAIN:
            if len(architectures) != 2 or len(tokenizers) != 1 or len(checkpoints) > 1:
                raise ValueError("architecture contrast requires 2 architectures and 1 tokenizer")
        elif kind is FactorialContrastKind.TOKENIZER_MAIN:
            if len(architectures) != 1 or len(tokenizers) != 2 or len(checkpoints) > 1:
                raise ValueError("tokenizer contrast requires 1 architecture and 2 tokenizers")
        elif kind is FactorialContrastKind.ARCHITECTURE_TOKENIZER_INTERACTION:
            if len(architectures) != 2 or len(tokenizers) != 2 or len(checkpoints) > 1:
                raise ValueError("interaction requires exactly 2 architectures and 2 tokenizers")
        elif kind is FactorialContrastKind.CHECKPOINT:
            if len(architectures) != 1 or len(tokenizers) != 1 or len(checkpoints) != 2:
                raise ValueError("checkpoint contrast requires 1 architecture, 1 tokenizer, 2 checkpoints")
        if len(set(architectures)) != len(architectures):
            raise ValueError("architecture levels must be unique within a contrast")
        if len(set(tokenizers)) != len(tokenizers):
            raise ValueError("tokenizer levels must be unique within a contrast")
        if len(set(checkpoints)) != len(checkpoints):
            raise ValueError("checkpoint levels must be unique within a contrast")
        object.__setattr__(self, "architectures", architectures)
        object.__setattr__(self, "tokenizers", tokenizers)
        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "fixed_axes", MappingProxyType(dict(self.fixed_axes)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "architectures": list(self.architectures),
            "checkpoints": list(self.checkpoints),
            "contrast_id": self.contrast_id,
            "fixed_axes": dict(self.fixed_axes),
            "kind": self.kind.value,
            "replication_group": self.replication_group,
            "tokenizers": list(self.tokenizers),
        }


@dataclass(frozen=True, slots=True)
class FactorialAnalysisPolicy:
    """Confound and comparability thresholds for factorial contrasts."""

    max_task_metric_delta: float = 0.05
    max_checkpoint_maturity_delta: float = 0.05
    min_shared_target_fraction: float = 0.75
    require_exact_target_universe: bool = True
    min_replication_contrasts: int = 2

    def __post_init__(self) -> None:
        if self.max_task_metric_delta < 0.0:
            raise ValueError("max_task_metric_delta must be non-negative")
        if self.max_checkpoint_maturity_delta < 0.0:
            raise ValueError("max_checkpoint_maturity_delta must be non-negative")
        if not 0.0 < self.min_shared_target_fraction <= 1.0:
            raise ValueError("min_shared_target_fraction must lie in (0, 1]")
        if self.min_replication_contrasts < 2:
            raise ValueError("min_replication_contrasts must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FactorialMechanismSpec:
    """Frozen factorial design and preregistered contrast list."""

    study_id: str
    cells: tuple[FactorialCellSpec, ...]
    contrasts: tuple[FactorialContrastSpec, ...]
    matched_covariates: tuple[MatchedCovariate, ...] = ()
    policy: FactorialAnalysisPolicy = field(default_factory=FactorialAnalysisPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = FACTORIAL_STUDY_SCHEMA

    def __post_init__(self) -> None:
        if not self.study_id:
            raise ValueError("study_id must be non-empty")
        if not self.cells:
            raise ValueError("factorial study requires declared cells")
        if not self.contrasts:
            raise ValueError("factorial study requires preregistered contrasts")
        cell_ids = [item.cell_id for item in self.cells]
        contrast_ids = [item.contrast_id for item in self.contrasts]
        covariate_names = [item.name for item in self.matched_covariates]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("factorial cell_id values must be unique")
        if len(contrast_ids) != len(set(contrast_ids)):
            raise ValueError("factorial contrast_id values must be unique")
        if len(covariate_names) != len(set(covariate_names)):
            raise ValueError("matched covariate names must be unique")
        for cell in self.cells:
            missing = [
                rule.name for rule in self.matched_covariates if rule.name not in cell.covariates
            ]
            if missing:
                raise ValueError(
                    f"cell {cell.cell_id!r} is missing matched covariate(s): {missing}"
                )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [item.to_dict() for item in self.cells],
            "contrasts": [item.to_dict() for item in self.contrasts],
            "matched_covariates": [item.to_dict() for item in self.matched_covariates],
            "metadata": dict(self.metadata),
            "policy": self.policy.to_dict(),
            "schema_version": self.schema_version,
            "study_id": self.study_id,
        }


@dataclass(frozen=True, slots=True)
class FactorialCellOutcome:
    """Held-out mechanism outcomes for one completed factorial cell."""

    task_metric: float
    candidate_size: int
    validation_sufficiency: float
    validation_necessity: float
    validation_joint_faithfulness: float
    validation_joint_random_percentile: float
    discovery_to_validation_drop: float
    intervention_baseline_sensitivity: float | None
    promotion_passed: bool
    source_study_fingerprint: str
    source_run_hash: str
    evidence_protocol_fingerprint: str
    effect_map: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_size <= 0:
            raise ValueError("candidate_size must be positive")
        scalar_values = (
            self.task_metric,
            self.validation_sufficiency,
            self.validation_necessity,
            self.validation_joint_faithfulness,
            self.validation_joint_random_percentile,
            self.discovery_to_validation_drop,
        )
        if not np.isfinite(np.asarray(scalar_values, dtype=np.float64)).all():
            raise ValueError("factorial outcome contains non-finite scalar values")
        if (
            self.intervention_baseline_sensitivity is not None
            and not np.isfinite(float(self.intervention_baseline_sensitivity))
        ):
            raise ValueError("intervention_baseline_sensitivity must be finite")
        effects = {str(key): float(value) for key, value in self.effect_map.items()}
        if effects and not np.isfinite(
            np.asarray(list(effects.values()), dtype=np.float64)
        ).all():
            raise ValueError("factorial effect_map contains non-finite values")
        if (
            not self.source_study_fingerprint
            or not self.source_run_hash
            or not self.evidence_protocol_fingerprint
        ):
            raise ValueError("factorial outcome requires source evidence identities and protocol")
        object.__setattr__(self, "effect_map", MappingProxyType(effects))

    def scalar_outcomes(self) -> dict[str, float]:
        payload = {
            "candidate_size": float(self.candidate_size),
            "discovery_to_validation_drop": self.discovery_to_validation_drop,
            "task_metric": self.task_metric,
            "validation_joint_faithfulness": self.validation_joint_faithfulness,
            "validation_joint_random_percentile": self.validation_joint_random_percentile,
            "validation_necessity": self.validation_necessity,
            "validation_sufficiency": self.validation_sufficiency,
        }
        if self.intervention_baseline_sensitivity is not None:
            payload["intervention_baseline_sensitivity"] = self.intervention_baseline_sensitivity
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scalar_outcomes(),
            "effect_map": dict(self.effect_map),
            "evidence_protocol_fingerprint": self.evidence_protocol_fingerprint,
            "promotion_passed": self.promotion_passed,
            "source_run_hash": self.source_run_hash,
            "source_study_fingerprint": self.source_study_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class FactorialCellResult:
    cell: FactorialCellSpec
    outcome: FactorialCellOutcome | None

    @property
    def observed(self) -> bool:
        return self.cell.available and self.outcome is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell.to_dict(),
            "observed": self.observed,
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FactorialContrastResult:
    contrast_id: str
    kind: FactorialContrastKind
    cell_ids: tuple[str, ...]
    estimable: bool
    reasons: tuple[str, ...]
    outcome_effects: Mapping[str, float]
    task_metric_range: float | None
    shared_target_fraction: float | None
    effect_map_stability: EffectMapStability | None = None
    interaction_effect_map: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome_effects",
            MappingProxyType(
                {str(key): float(value) for key, value in self.outcome_effects.items()}
            ),
        )
        object.__setattr__(
            self,
            "interaction_effect_map",
            MappingProxyType(
                {str(key): float(value) for key, value in self.interaction_effect_map.items()}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_ids": list(self.cell_ids),
            "contrast_id": self.contrast_id,
            "effect_map_stability": (
                None if self.effect_map_stability is None else self.effect_map_stability.to_dict()
            ),
            "estimable": self.estimable,
            "interaction_effect_map": dict(self.interaction_effect_map),
            "kind": self.kind.value,
            "outcome_effects": dict(self.outcome_effects),
            "reasons": list(self.reasons),
            "shared_target_fraction": self.shared_target_fraction,
            "task_metric_range": self.task_metric_range,
        }


@dataclass(frozen=True, slots=True)
class FactorialReplicationSummary:
    replication_group: str
    contrast_ids: tuple[str, ...]
    estimable_count: int
    session_ids: tuple[str, ...]
    cross_session: bool
    replication_ready: bool
    outcome_sign_agreement: Mapping[str, float]
    outcome_median_effect: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome_sign_agreement",
            MappingProxyType(dict(self.outcome_sign_agreement)),
        )
        object.__setattr__(
            self,
            "outcome_median_effect",
            MappingProxyType(dict(self.outcome_median_effect)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast_ids": list(self.contrast_ids),
            "cross_session": self.cross_session,
            "estimable_count": self.estimable_count,
            "outcome_median_effect": dict(self.outcome_median_effect),
            "outcome_sign_agreement": dict(self.outcome_sign_agreement),
            "replication_group": self.replication_group,
            "replication_ready": self.replication_ready,
            "session_ids": list(self.session_ids),
        }


@dataclass(frozen=True, slots=True)
class FactorialMechanismReport:
    spec: FactorialMechanismSpec
    cells: tuple[FactorialCellResult, ...]
    contrasts: tuple[FactorialContrastResult, ...]
    replications: tuple[FactorialReplicationSummary, ...]

    @property
    def missing_cell_ids(self) -> tuple[str, ...]:
        return tuple(item.cell.cell_id for item in self.cells if not item.observed)

    @property
    def estimable_contrast_ids(self) -> tuple[str, ...]:
        return tuple(item.contrast_id for item in self.contrasts if item.estimable)

    @property
    def nonestimable_contrast_ids(self) -> tuple[str, ...]:
        return tuple(item.contrast_id for item in self.contrasts if not item.estimable)

    @property
    def study_fingerprint(self) -> str:
        return stable_hash(
            {
                "cells": [item.to_dict() for item in self.cells],
                "contrasts": [item.to_dict() for item in self.contrasts],
                "replications": [item.to_dict() for item in self.replications],
                "spec": self.spec.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [item.to_dict() for item in self.cells],
            "contrasts": [item.to_dict() for item in self.contrasts],
            "estimable_contrast_ids": list(self.estimable_contrast_ids),
            "missing_cell_ids": list(self.missing_cell_ids),
            "nonestimable_contrast_ids": list(self.nonestimable_contrast_ids),
            "replications": [item.to_dict() for item in self.replications],
            "schema_version": self.spec.schema_version,
            "spec": self.spec.to_dict(),
            "study_fingerprint": self.study_fingerprint,
        }


def _paired_baseline_sensitivity(result: EvidencePackResult) -> float | None:
    by_baseline: dict[str, list[float]] = defaultdict(list)
    for case in result.candidate_cases:
        if case.split.value != "validation" or case.report is None:
            continue
        by_baseline[case.intervention_baseline].append(case.report.joint_faithfulness)
    if len(by_baseline) < 2:
        return None
    means = [mean(values) for _, values in sorted(by_baseline.items()) if values]
    if len(means) < 2:
        return None
    return float(max(means) - min(means))


def _validation_task_metric(result: EvidencePackResult) -> float:
    per_example: dict[str, list[float]] = defaultdict(list)
    for case in result.candidate_cases:
        if case.split.value != "validation" or case.report is None:
            continue
        per_example[case.example_id].append(case.report.baseline_metric)
    if not per_example:
        raise ValueError("evidence pack has no valid validation baseline metrics")
    return float(mean(mean(values) for _, values in sorted(per_example.items())))


def _evidence_protocol_fingerprint(result: EvidencePackResult) -> str:
    return stable_hash(
        {
            "discovery_method": result.spec.discovery_method,
            "faithfulness_policy": result.faithfulness_policy.to_dict(),
            "intervention_baselines": list(result.spec.intervention_baselines),
            "metric_name": result.spec.metric_name,
            "pack_policy": result.policy.to_dict(),
            "random_trials": result.spec.random_trials,
            "target_universe": list(result.spec.target_universe),
        }
    )


def outcome_from_evidence_pack(
    result: EvidencePackResult,
    *,
    effect_map: Mapping[str, float] | None = None,
) -> FactorialCellOutcome:
    """Convert one completed v0.6 pack into a v0.7 cell outcome."""

    return FactorialCellOutcome(
        task_metric=_validation_task_metric(result),
        candidate_size=len(result.candidate.targets),
        validation_sufficiency=result.validation_aggregate.mean_sufficiency,
        validation_necessity=result.validation_aggregate.mean_necessity,
        validation_joint_faithfulness=result.validation_aggregate.median_joint_faithfulness,
        validation_joint_random_percentile=(
            result.validation_aggregate.mean_joint_random_percentile
        ),
        discovery_to_validation_drop=result.promotion.joint_generalization_drop,
        intervention_baseline_sensitivity=_paired_baseline_sensitivity(result),
        promotion_passed=result.promotion.passed,
        source_study_fingerprint=result.study_fingerprint,
        source_run_hash=result.run_hash,
        evidence_protocol_fingerprint=_evidence_protocol_fingerprint(result),
        effect_map=dict(effect_map or {}),
    )


_ALLOWED_FIXED_AXES = {
    "architecture",
    "checkpoint",
    "checkpoint_maturity",
    "dataset_id",
    "dataset_revision",
    "discovery_method",
    "discovery_partition_id",
    "metric_name",
    "session_id",
    "subject_id",
    "tokenizer_id",
    "training_seed",
    "validation_partition_id",
}


def _matches_fixed(cell: FactorialCellSpec, fixed: Mapping[str, Any]) -> bool:
    unknown = sorted(set(fixed) - _ALLOWED_FIXED_AXES)
    if unknown:
        raise ValueError(f"unsupported fixed axis name(s): {unknown}")
    return all(getattr(cell, name) == value for name, value in fixed.items())


def _select_cells(
    spec: FactorialMechanismSpec,
    contrast: FactorialContrastSpec,
) -> tuple[FactorialCellSpec, ...]:
    selected = []
    for cell in spec.cells:
        if not _matches_fixed(cell, contrast.fixed_axes):
            continue
        if cell.architecture not in contrast.architectures:
            continue
        if cell.tokenizer_id not in contrast.tokenizers:
            continue
        if contrast.checkpoints and cell.checkpoint not in contrast.checkpoints:
            continue
        selected.append(cell)

    def _key(cell: FactorialCellSpec) -> tuple[int, int, int]:
        arch = contrast.architectures.index(cell.architecture)
        tok = contrast.tokenizers.index(cell.tokenizer_id)
        checkpoint = 0 if not contrast.checkpoints else contrast.checkpoints.index(cell.checkpoint)
        return arch, tok, checkpoint

    return tuple(sorted(selected, key=_key))


def _expected_cell_count(kind: FactorialContrastKind) -> int:
    if kind is FactorialContrastKind.ARCHITECTURE_TOKENIZER_INTERACTION:
        return 4
    return 2


def _shared_target_fraction(cells: Sequence[FactorialCellSpec]) -> float:
    target_sets = [set(item.target_universe) for item in cells]
    shared = set.intersection(*target_sets)
    union = set.union(*target_sets)
    return 1.0 if not union else len(shared) / len(union)


def _nonvaried_confounds(
    left: FactorialCellSpec,
    right: FactorialCellSpec,
    *,
    varied_axes: set[str],
    spec: FactorialMechanismSpec,
    left_outcome: FactorialCellOutcome,
    right_outcome: FactorialCellOutcome,
) -> list[str]:
    reasons = []
    design_axes = (
        "architecture",
        "tokenizer_id",
        "training_seed",
        "checkpoint",
        "dataset_id",
        "dataset_revision",
        "discovery_method",
        "discovery_partition_id",
        "metric_name",
        "session_id",
        "subject_id",
        "validation_partition_id",
    )
    for axis in design_axes:
        canonical = "tokenizer" if axis == "tokenizer_id" else axis
        if canonical in varied_axes:
            continue
        if getattr(left, axis) != getattr(right, axis):
            reasons.append(f"non-varied axis {axis} differs")

    if "checkpoint" not in varied_axes:
        maturity_delta = abs(left.checkpoint_maturity - right.checkpoint_maturity)
        if maturity_delta > spec.policy.max_checkpoint_maturity_delta:
            reasons.append(
                f"checkpoint maturity delta {maturity_delta:.3f} > "
                f"{spec.policy.max_checkpoint_maturity_delta:.3f}"
            )

    task_delta = abs(left_outcome.task_metric - right_outcome.task_metric)
    if task_delta > spec.policy.max_task_metric_delta:
        reasons.append(
            f"task metric delta {task_delta:.3f} > {spec.policy.max_task_metric_delta:.3f}"
        )

    if (
        left_outcome.evidence_protocol_fingerprint
        != right_outcome.evidence_protocol_fingerprint
    ):
        reasons.append("evidence protocol fingerprint differs")

    for rule in spec.matched_covariates:
        left_value = left.covariates[rule.name]
        right_value = right.covariates[rule.name]
        if not rule.matches(left_value, right_value):
            reasons.append(f"matched covariate {rule.name!r} differs")

    if (
        spec.policy.require_exact_target_universe
        and set(left.target_universe) != set(right.target_universe)
    ):
        reasons.append("target universe differs")
    return reasons


def _effect_map_stability(
    left: FactorialCellOutcome,
    right: FactorialCellOutcome,
) -> EffectMapStability | None:
    if not left.effect_map or not right.effect_map:
        return None
    return compare_effect_maps(left.effect_map, right.effect_map, top_k=5)


def _pair_effects(
    left: FactorialCellOutcome,
    right: FactorialCellOutcome,
) -> dict[str, float]:
    left_values = left.scalar_outcomes()
    right_values = right.scalar_outcomes()
    shared = sorted(set(left_values) & set(right_values))
    return {name: right_values[name] - left_values[name] for name in shared}


def _interaction_effects(
    a1_t1: FactorialCellOutcome,
    a2_t1: FactorialCellOutcome,
    a1_t2: FactorialCellOutcome,
    a2_t2: FactorialCellOutcome,
) -> dict[str, float]:
    values = [item.scalar_outcomes() for item in (a1_t1, a2_t1, a1_t2, a2_t2)]
    shared = sorted(set.intersection(*(set(item) for item in values)))
    return {
        name: (values[3][name] - values[2][name]) - (values[1][name] - values[0][name])
        for name in shared
    }


def _interaction_map(
    a1_t1: FactorialCellOutcome,
    a2_t1: FactorialCellOutcome,
    a1_t2: FactorialCellOutcome,
    a2_t2: FactorialCellOutcome,
) -> tuple[dict[str, float], float | None]:
    maps = [item.effect_map for item in (a1_t1, a2_t1, a1_t2, a2_t2)]
    if any(not item for item in maps):
        return {}, None
    shared = set.intersection(*(set(item) for item in maps))
    union = set.union(*(set(item) for item in maps))
    fraction = 1.0 if not union else len(shared) / len(union)
    interaction = {
        key: (maps[3][key] - maps[2][key]) - (maps[1][key] - maps[0][key])
        for key in sorted(shared)
    }
    return interaction, fraction


def _contrast_result(
    spec: FactorialMechanismSpec,
    contrast: FactorialContrastSpec,
    outcome_by_id: Mapping[str, FactorialCellOutcome],
) -> FactorialContrastResult:
    cells = _select_cells(spec, contrast)
    reasons = []
    expected = _expected_cell_count(contrast.kind)
    if len(cells) != expected:
        reasons.append(f"contrast resolved {len(cells)} cell(s); expected {expected}")

    if len({item.cell_id for item in cells}) != len(cells):
        reasons.append("contrast contains duplicate cell IDs")
    if any(not item.available for item in cells):
        reasons.append("contrast contains an explicitly missing cell")
    if any(item.cell_id not in outcome_by_id for item in cells if item.available):
        reasons.append("contrast is missing one or more observed cell outcomes")

    if reasons:
        return FactorialContrastResult(
            contrast_id=contrast.contrast_id,
            kind=contrast.kind,
            cell_ids=tuple(item.cell_id for item in cells),
            estimable=False,
            reasons=tuple(reasons),
            outcome_effects={},
            task_metric_range=None,
            shared_target_fraction=None,
        )

    outcomes = [outcome_by_id[item.cell_id] for item in cells]
    task_values = [item.task_metric for item in outcomes]
    task_range = max(task_values) - min(task_values)
    shared_fraction = _shared_target_fraction(cells)
    if shared_fraction < spec.policy.min_shared_target_fraction:
        reasons.append(
            f"shared target fraction {shared_fraction:.3f} < "
            f"{spec.policy.min_shared_target_fraction:.3f}"
        )

    effect_stability = None
    interaction_map: dict[str, float] = {}
    if contrast.kind is FactorialContrastKind.ARCHITECTURE_MAIN:
        reasons.extend(
            _nonvaried_confounds(
                cells[0],
                cells[1],
                varied_axes={"architecture"},
                spec=spec,
                left_outcome=outcomes[0],
                right_outcome=outcomes[1],
            )
        )
        outcome_effects = _pair_effects(outcomes[0], outcomes[1])
        effect_stability = _effect_map_stability(outcomes[0], outcomes[1])
    elif contrast.kind is FactorialContrastKind.TOKENIZER_MAIN:
        reasons.extend(
            _nonvaried_confounds(
                cells[0],
                cells[1],
                varied_axes={"tokenizer"},
                spec=spec,
                left_outcome=outcomes[0],
                right_outcome=outcomes[1],
            )
        )
        outcome_effects = _pair_effects(outcomes[0], outcomes[1])
        effect_stability = _effect_map_stability(outcomes[0], outcomes[1])
    elif contrast.kind is FactorialContrastKind.CHECKPOINT:
        reasons.extend(
            _nonvaried_confounds(
                cells[0],
                cells[1],
                varied_axes={"checkpoint"},
                spec=spec,
                left_outcome=outcomes[0],
                right_outcome=outcomes[1],
            )
        )
        outcome_effects = _pair_effects(outcomes[0], outcomes[1])
        effect_stability = _effect_map_stability(outcomes[0], outcomes[1])
    else:
        lookup = {
            (cell.architecture, cell.tokenizer_id): (cell, outcome)
            for cell, outcome in zip(cells, outcomes, strict=True)
        }
        a1, a2 = contrast.architectures
        t1, t2 = contrast.tokenizers
        ordered_pairs = [
            lookup[(a1, t1)],
            lookup[(a2, t1)],
            lookup[(a1, t2)],
            lookup[(a2, t2)],
        ]
        ordered_cells = [item[0] for item in ordered_pairs]
        ordered_outcomes = [item[1] for item in ordered_pairs]
        for left_index, right_index, varied in (
            (0, 1, {"architecture"}),
            (2, 3, {"architecture"}),
            (0, 2, {"tokenizer"}),
            (1, 3, {"tokenizer"}),
        ):
            reasons.extend(
                _nonvaried_confounds(
                    ordered_cells[left_index],
                    ordered_cells[right_index],
                    varied_axes=varied,
                    spec=spec,
                    left_outcome=ordered_outcomes[left_index],
                    right_outcome=ordered_outcomes[right_index],
                )
            )
        outcome_effects = _interaction_effects(*ordered_outcomes)
        interaction_map, map_fraction = _interaction_map(*ordered_outcomes)
        if map_fraction is not None:
            shared_fraction = min(shared_fraction, map_fraction)

    return FactorialContrastResult(
        contrast_id=contrast.contrast_id,
        kind=contrast.kind,
        cell_ids=tuple(item.cell_id for item in cells),
        estimable=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        outcome_effects={} if reasons else outcome_effects,
        task_metric_range=task_range,
        shared_target_fraction=shared_fraction,
        effect_map_stability=None if reasons else effect_stability,
        interaction_effect_map={} if reasons else interaction_map,
    )


def _replication_summaries(
    spec: FactorialMechanismSpec,
    contrasts: Sequence[FactorialContrastResult],
) -> tuple[FactorialReplicationSummary, ...]:
    contrast_by_id = {item.contrast_id: item for item in contrasts}
    grouped: dict[str, list[FactorialContrastResult]] = defaultdict(list)
    for contrast_spec in spec.contrasts:
        if contrast_spec.replication_group is None:
            continue
        grouped[contrast_spec.replication_group].append(contrast_by_id[contrast_spec.contrast_id])

    summaries = []
    for group, items in sorted(grouped.items()):
        estimable = [item for item in items if item.estimable]
        outcome_names = (
            sorted(set.intersection(*(set(item.outcome_effects) for item in estimable)))
            if estimable
            else []
        )
        sign_agreement = {}
        medians = {}
        for name in outcome_names:
            values = [item.outcome_effects[name] for item in estimable]
            nonzero = [np.sign(value) for value in values if abs(value) > 1e-12]
            if not nonzero:
                agreement = 1.0
            else:
                positive = sum(sign > 0 for sign in nonzero)
                negative = sum(sign < 0 for sign in nonzero)
                agreement = max(positive, negative) / len(nonzero)
            sign_agreement[name] = float(agreement)
            medians[name] = float(median(values))
        cell_ids = {cell_id for item in estimable for cell_id in item.cell_ids}
        declared = {cell.cell_id: cell for cell in spec.cells}
        sessions = tuple(
            sorted({declared[cell_id].session_id for cell_id in cell_ids if cell_id in declared})
        )
        cross_session = len(sessions) >= 2
        summaries.append(
            FactorialReplicationSummary(
                replication_group=group,
                contrast_ids=tuple(item.contrast_id for item in items),
                estimable_count=len(estimable),
                session_ids=sessions,
                cross_session=cross_session,
                replication_ready=(
                    len(estimable) >= spec.policy.min_replication_contrasts and cross_session
                ),
                outcome_sign_agreement=sign_agreement,
                outcome_median_effect=medians,
            )
        )
    return tuple(summaries)


def preregister_2x2_contrasts(
    *,
    prefix: str,
    architectures: tuple[str, str],
    tokenizers: tuple[str, str],
    fixed_axes: Mapping[str, Any],
    replication_namespace: str | None = None,
) -> tuple[FactorialContrastSpec, ...]:
    """Materialize the five primary contrasts for one matched 2 x 2 slice.

    Calling this once per session with the same ``replication_namespace`` makes
    the corresponding contrasts explicit cross-session replication groups.
    """

    if not prefix:
        raise ValueError("contrast prefix must be non-empty")
    a1, a2 = architectures
    t1, t2 = tokenizers

    def _group(label: str) -> str | None:
        if replication_namespace is None:
            return None
        return f"{replication_namespace}:{label}"

    shared = dict(fixed_axes)
    return (
        FactorialContrastSpec(
            contrast_id=f"{prefix}:architecture@{t1}",
            kind=FactorialContrastKind.ARCHITECTURE_MAIN,
            architectures=(a1, a2),
            tokenizers=(t1,),
            fixed_axes=shared,
            replication_group=_group(f"architecture@{t1}"),
        ),
        FactorialContrastSpec(
            contrast_id=f"{prefix}:architecture@{t2}",
            kind=FactorialContrastKind.ARCHITECTURE_MAIN,
            architectures=(a1, a2),
            tokenizers=(t2,),
            fixed_axes=shared,
            replication_group=_group(f"architecture@{t2}"),
        ),
        FactorialContrastSpec(
            contrast_id=f"{prefix}:tokenizer@{a1}",
            kind=FactorialContrastKind.TOKENIZER_MAIN,
            architectures=(a1,),
            tokenizers=(t1, t2),
            fixed_axes=shared,
            replication_group=_group(f"tokenizer@{a1}"),
        ),
        FactorialContrastSpec(
            contrast_id=f"{prefix}:tokenizer@{a2}",
            kind=FactorialContrastKind.TOKENIZER_MAIN,
            architectures=(a2,),
            tokenizers=(t1, t2),
            fixed_axes=shared,
            replication_group=_group(f"tokenizer@{a2}"),
        ),
        FactorialContrastSpec(
            contrast_id=f"{prefix}:interaction",
            kind=FactorialContrastKind.ARCHITECTURE_TOKENIZER_INTERACTION,
            architectures=(a1, a2),
            tokenizers=(t1, t2),
            fixed_axes=shared,
            replication_group=_group("interaction"),
        ),
    )


def analyze_factorial_mechanisms(
    spec: FactorialMechanismSpec,
    outcomes: Mapping[str, FactorialCellOutcome],
) -> FactorialMechanismReport:
    """Evaluate only preregistered, confound-checked factorial contrasts."""

    declared = {item.cell_id: item for item in spec.cells}
    unknown = sorted(set(outcomes) - set(declared))
    if unknown:
        raise ValueError(f"outcomes supplied for undeclared cell(s): {unknown}")
    forbidden = sorted(cell_id for cell_id in outcomes if not declared[cell_id].available)
    if forbidden:
        raise ValueError(f"outcomes supplied for explicitly missing cell(s): {forbidden}")

    cells = tuple(
        FactorialCellResult(cell=cell, outcome=outcomes.get(cell.cell_id)) for cell in spec.cells
    )
    contrast_results = tuple(
        _contrast_result(spec, contrast, outcomes) for contrast in spec.contrasts
    )
    replications = _replication_summaries(spec, contrast_results)
    return FactorialMechanismReport(
        spec=spec,
        cells=cells,
        contrasts=contrast_results,
        replications=replications,
    )


def write_factorial_artifact(
    report: FactorialMechanismReport,
    path: str | Path,
) -> str:
    """Write a self-checking factorial-study JSON artifact and return its hash."""

    result = report.to_dict()
    artifact_hash = stable_hash(result)
    payload = {
        "artifact_hash": artifact_hash,
        "artifact_schema": FACTORIAL_ARTIFACT_SCHEMA,
        "result": result,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_hash


def read_factorial_artifact(path: str | Path) -> dict[str, Any]:
    """Validate and return a serialized factorial-study result."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_schema") != FACTORIAL_ARTIFACT_SCHEMA:
        raise ValueError("unsupported factorial artifact schema")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("factorial artifact result must be an object")
    if stable_hash(result) != payload.get("artifact_hash"):
        raise ValueError("factorial artifact hash mismatch")
    if result.get("schema_version") != FACTORIAL_STUDY_SCHEMA:
        raise ValueError("unsupported factorial study schema")
    return result
