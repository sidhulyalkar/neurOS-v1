"""Held-out real-model evidence packs for circuit-faithfulness studies.

The v0.6 evidence-pack layer makes discovery/validation separation an executable
contract. Candidate discovery callbacks and intervention-donor estimation receive
discovery examples only. The selected candidate and donor statistics are frozen
before any validation example is evaluated.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from statistics import mean, median
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from neuros_mechint.adapters.base import ModelAdapter
from neuros_mechint.core.evidence import EvidenceTier
from neuros_mechint.core.manifest import ExperimentManifest, stable_hash, stable_hash_or_none
from neuros_mechint.core.metrics import ScalarMetric

from .faithfulness import (
    CircuitCandidate,
    CircuitFaithfulnessReport,
    FaithfulnessPolicy,
    evaluate_adapter_circuit_faithfulness,
)

EVIDENCE_ARTIFACT_SCHEMA = "neuros-mechint.evidence-pack-artifact.v1"
EVIDENCE_PACK_SCHEMA = "neuros-mechint.evidence-pack.v1"

_NORMALIZATION_FAILURES = (
    "all-target and null metrics are indistinguishable",
    "null intervention outperforms the all-target baseline",
)


class EvidenceSplit(str, Enum):
    """Immutable scientific role assigned to one example."""

    DISCOVERY = "discovery"
    VALIDATION = "validation"

    @classmethod
    def coerce(cls, value: EvidenceSplit | str) -> EvidenceSplit:
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


@dataclass(frozen=True, slots=True)
class EvidenceExample:
    """Opaque model input with an explicit discovery or validation role."""

    example_id: str
    inputs: Any
    split: EvidenceSplit | str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be non-empty")
        object.__setattr__(self, "split", EvidenceSplit.coerce(self.split))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def input_hash(self) -> str | None:
        return stable_hash_or_none(self.inputs)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "input_hash": self.input_hash,
            "metadata": dict(self.metadata),
            "split": self.split.value,
        }


@dataclass(frozen=True, slots=True)
class EvidencePackSpec:
    """Frozen design parameters for one held-out circuit study."""

    pack_id: str
    model_id: str
    dataset_id: str
    metric_name: str
    target_universe: tuple[str, ...]
    discovery_method: str
    model_revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    dataset_revision: str | None = None
    intervention_baselines: tuple[str, ...] = ("zero", "mean")
    random_trials: int = 100
    seed: int = 0
    evidence_tier: EvidenceTier = EvidenceTier.INTEGRATION
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVIDENCE_PACK_SCHEMA

    def __post_init__(self) -> None:
        if not self.pack_id or not self.model_id or not self.dataset_id:
            raise ValueError("pack_id, model_id, and dataset_id must be non-empty")
        if not self.metric_name or not self.discovery_method:
            raise ValueError("metric_name and discovery_method must be non-empty")
        targets = tuple(dict.fromkeys(str(target) for target in self.target_universe))
        if not targets:
            raise ValueError("target_universe must not be empty")
        baselines = tuple(dict.fromkeys(str(item) for item in self.intervention_baselines))
        if not baselines:
            raise ValueError("intervention_baselines must not be empty")
        unsupported = [item for item in baselines if item not in {"mean", "zero"}]
        if unsupported:
            raise ValueError(f"unsupported intervention baseline(s): {unsupported}")
        if self.random_trials <= 0:
            raise ValueError("random_trials must be positive")
        object.__setattr__(self, "target_universe", targets)
        object.__setattr__(self, "intervention_baselines", baselines)
        object.__setattr__(self, "evidence_tier", EvidenceTier.coerce(self.evidence_tier))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "discovery_method": self.discovery_method,
            "evidence_tier": {
                "label": self.evidence_tier.label,
                "level": int(self.evidence_tier),
            },
            "intervention_baselines": list(self.intervention_baselines),
            "metadata": dict(self.metadata),
            "metric_name": self.metric_name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "pack_id": self.pack_id,
            "random_trials": self.random_trials,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "target_universe": list(self.target_universe),
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True, slots=True)
class EvidencePackPolicy:
    """Cross-example criteria for promoting a discovered circuit."""

    min_validation_examples: int = 2
    min_validation_pass_rate: float = 0.80
    min_validation_joint_median: float = 0.50
    max_joint_generalization_drop: float = 0.25
    min_validation_joint_advantage_vs_magnitude: float = 0.0
    require_all_cases_valid: bool = True
    require_multiple_intervention_baselines: bool = True
    bootstrap_samples: int = 1000

    def __post_init__(self) -> None:
        if self.min_validation_examples <= 0:
            raise ValueError("min_validation_examples must be positive")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        for name in (
            "max_joint_generalization_drop",
            "min_validation_joint_median",
            "min_validation_pass_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bootstrap_samples": self.bootstrap_samples,
            "max_joint_generalization_drop": self.max_joint_generalization_drop,
            "min_validation_examples": self.min_validation_examples,
            "min_validation_joint_advantage_vs_magnitude": (
                self.min_validation_joint_advantage_vs_magnitude
            ),
            "min_validation_joint_median": self.min_validation_joint_median,
            "min_validation_pass_rate": self.min_validation_pass_rate,
            "require_all_cases_valid": self.require_all_cases_valid,
            "require_multiple_intervention_baselines": self.require_multiple_intervention_baselines,
        }


@dataclass(frozen=True, slots=True)
class EvidenceCaseResult:
    """One candidate evaluated on one example with one intervention baseline."""

    example_id: str
    split: EvidenceSplit
    intervention_baseline: str
    input_hash: str
    report: CircuitFaithfulnessReport | None
    invalid_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.report is None) == (self.invalid_reason is None):
            raise ValueError("exactly one of report and invalid_reason must be present")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def valid(self) -> bool:
        return self.report is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "input_hash": self.input_hash,
            "intervention_baseline": self.intervention_baseline,
            "invalid_reason": self.invalid_reason,
            "metadata": dict(self.metadata),
            "report": None if self.report is None else self.report.to_dict(),
            "split": self.split.value,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class EvidenceAggregate:
    """Summary across paired examples and intervention baselines."""

    n_cases: int
    n_valid_cases: int
    n_invalid_cases: int
    n_examples: int
    pass_rate: float
    valid_case_rate: float
    mean_sufficiency: float
    mean_necessity: float
    mean_joint_faithfulness: float
    median_joint_faithfulness: float
    mean_joint_random_percentile: float
    joint_mean_ci95_low: float
    joint_mean_ci95_high: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_mean_ci95_high": self.joint_mean_ci95_high,
            "joint_mean_ci95_low": self.joint_mean_ci95_low,
            "mean_joint_faithfulness": self.mean_joint_faithfulness,
            "mean_joint_random_percentile": self.mean_joint_random_percentile,
            "mean_necessity": self.mean_necessity,
            "mean_sufficiency": self.mean_sufficiency,
            "median_joint_faithfulness": self.median_joint_faithfulness,
            "n_cases": self.n_cases,
            "n_examples": self.n_examples,
            "n_invalid_cases": self.n_invalid_cases,
            "n_valid_cases": self.n_valid_cases,
            "pass_rate": self.pass_rate,
            "valid_case_rate": self.valid_case_rate,
        }


@dataclass(frozen=True, slots=True)
class EvidencePromotionDecision:
    """Held-out promotion result and explicit rejection reasons."""

    passed: bool
    reasons: tuple[str, ...]
    discovery_joint_median: float
    validation_joint_median: float
    joint_generalization_drop: float
    validation_pass_rate: float
    validation_joint_advantage_vs_magnitude: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovery_joint_median": self.discovery_joint_median,
            "joint_generalization_drop": self.joint_generalization_drop,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "validation_joint_advantage_vs_magnitude": (
                self.validation_joint_advantage_vs_magnitude
            ),
            "validation_joint_median": self.validation_joint_median,
            "validation_pass_rate": self.validation_pass_rate,
        }


@dataclass(frozen=True, slots=True)
class EvidenceTelemetry:
    """Runtime and memory observations for reproducible evidence runs."""

    wall_time_s: float
    peak_python_memory_mb: float
    peak_cuda_memory_mb: float | None
    package_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_versions", MappingProxyType(dict(self.package_versions)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_versions": dict(self.package_versions),
            "peak_cuda_memory_mb": self.peak_cuda_memory_mb,
            "peak_python_memory_mb": self.peak_python_memory_mb,
            "wall_time_s": self.wall_time_s,
        }


@dataclass(frozen=True, slots=True)
class EvidencePackResult:
    """Complete discovery/held-out evidence record for one frozen candidate."""

    spec: EvidencePackSpec
    policy: EvidencePackPolicy
    faithfulness_policy: FaithfulnessPolicy
    candidate: CircuitCandidate
    magnitude_candidate: CircuitCandidate | None
    candidate_cases: tuple[EvidenceCaseResult, ...]
    magnitude_cases: tuple[EvidenceCaseResult, ...]
    discovery_aggregate: EvidenceAggregate
    validation_aggregate: EvidenceAggregate
    magnitude_validation_aggregate: EvidenceAggregate | None
    promotion: EvidencePromotionDecision
    manifest: ExperimentManifest
    telemetry: EvidenceTelemetry
    discovery_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    example_identities: tuple[Mapping[str, Any], ...]
    mean_ablation_references: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean_ablation_references",
            MappingProxyType(dict(self.mean_ablation_references)),
        )

    @property
    def publication_issues(self) -> tuple[str, ...]:
        """Reproducibility gaps that should be resolved before publishing a pack."""

        issues = []
        if self.spec.model_revision is None:
            issues.append("model_revision is not pinned")
        if self.spec.dataset_revision is None:
            issues.append("dataset_revision is not pinned")
        if self.spec.tokenizer_id is not None and self.spec.tokenizer_revision is None:
            issues.append("tokenizer_revision is not pinned")
        return tuple(issues)

    @property
    def publication_ready(self) -> bool:
        return not self.publication_issues

    @property
    def study_fingerprint(self) -> str:
        """Deterministic identity excluding timestamped runtime provenance."""

        return stable_hash(
            {
                "candidate": self.candidate.to_dict(),
                "candidate_cases": [item.to_dict() for item in self.candidate_cases],
                "discovery_example_ids": self.discovery_example_ids,
                "faithfulness_policy": self.faithfulness_policy.to_dict(),
                "magnitude_candidate": (
                    None if self.magnitude_candidate is None else self.magnitude_candidate.to_dict()
                ),
                "magnitude_cases": [item.to_dict() for item in self.magnitude_cases],
                "mean_ablation_references": dict(self.mean_ablation_references),
                "policy": self.policy.to_dict(),
                "spec": self.spec.to_dict(),
                "validation_example_ids": self.validation_example_ids,
            }
        )

    @property
    def run_hash(self) -> str:
        return stable_hash(
            {
                "manifest_hash": self.manifest.content_hash,
                "study_fingerprint": self.study_fingerprint,
                "telemetry": self.telemetry.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "candidate_cases": [item.to_dict() for item in self.candidate_cases],
            "discovery_aggregate": self.discovery_aggregate.to_dict(),
            "discovery_example_ids": list(self.discovery_example_ids),
            "example_identities": [dict(item) for item in self.example_identities],
            "faithfulness_policy": self.faithfulness_policy.to_dict(),
            "magnitude_candidate": (
                None if self.magnitude_candidate is None else self.magnitude_candidate.to_dict()
            ),
            "magnitude_cases": [item.to_dict() for item in self.magnitude_cases],
            "magnitude_validation_aggregate": (
                None
                if self.magnitude_validation_aggregate is None
                else self.magnitude_validation_aggregate.to_dict()
            ),
            "manifest": self.manifest.to_dict(),
            "manifest_hash": self.manifest.content_hash,
            "mean_ablation_references": dict(self.mean_ablation_references),
            "policy": self.policy.to_dict(),
            "promotion": self.promotion.to_dict(),
            "publication_issues": list(self.publication_issues),
            "publication_ready": self.publication_ready,
            "run_hash": self.run_hash,
            "schema_version": self.spec.schema_version,
            "spec": self.spec.to_dict(),
            "study_fingerprint": self.study_fingerprint,
            "telemetry": self.telemetry.to_dict(),
            "validation_aggregate": self.validation_aggregate.to_dict(),
            "validation_example_ids": list(self.validation_example_ids),
        }


CandidateDiscovery = Callable[
    [ModelAdapter, tuple[EvidenceExample, ...], tuple[str, ...]], CircuitCandidate
]


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in (
        "circuit-tracer",
        "neuros-mechint",
        "nnsight",
        "sae-lens",
        "torch",
        "transformer-lens",
    ):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty value sequence")
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    boot = array[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def aggregate_evidence_cases(
    cases: Sequence[EvidenceCaseResult],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> EvidenceAggregate:
    """Aggregate cases while preserving example-level pairing in uncertainty."""

    cases = tuple(cases)
    if not cases:
        raise ValueError("cannot aggregate an empty evidence case sequence")
    reports = [item.report for item in cases if item.report is not None]
    if not reports:
        raise ValueError("no evidence cases contain a valid faithfulness normalization")
    joint = [item.joint_faithfulness for item in reports]
    percentiles = [
        item.joint_random_percentile
        for item in reports
        if item.joint_random_percentile is not None
    ]
    if len(percentiles) != len(reports):
        raise ValueError("all valid evidence cases must contain random-control percentiles")

    example_joint: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        if case.report is not None:
            example_joint[case.example_id].append(case.report.joint_faithfulness)
    paired_values = [mean(values) for _, values in sorted(example_joint.items())]
    low, high = _bootstrap_mean_ci(paired_values, samples=bootstrap_samples, seed=seed)
    return EvidenceAggregate(
        n_cases=len(cases),
        n_valid_cases=len(reports),
        n_invalid_cases=len(cases) - len(reports),
        n_examples=len({item.example_id for item in cases}),
        pass_rate=sum(float(item.passed) for item in reports) / len(cases),
        valid_case_rate=len(reports) / len(cases),
        mean_sufficiency=mean(item.sufficiency_fraction for item in reports),
        mean_necessity=mean(item.necessity_fraction for item in reports),
        mean_joint_faithfulness=mean(joint),
        median_joint_faithfulness=median(joint),
        mean_joint_random_percentile=mean(float(item) for item in percentiles),
        joint_mean_ci95_low=low,
        joint_mean_ci95_high=high,
    )


def discover_activation_magnitude_candidate(
    adapter: ModelAdapter,
    discovery_examples: Sequence[EvidenceExample],
    all_targets: Sequence[str],
    *,
    k: int,
    name: str = "activation-magnitude-baseline",
) -> CircuitCandidate:
    """Top-k activation-magnitude baseline fitted on discovery examples only."""

    if k <= 0:
        raise ValueError("k must be positive")
    targets = tuple(dict.fromkeys(str(target) for target in all_targets))
    if k > len(targets):
        raise ValueError("k cannot exceed the target universe size")
    examples = tuple(discovery_examples)
    if not examples:
        raise ValueError("activation-magnitude discovery requires discovery examples")
    if any(item.split is not EvidenceSplit.DISCOVERY for item in examples):
        raise ValueError("activation-magnitude discovery received a validation example")

    totals = {target: 0.0 for target in targets}
    for example in examples:
        captured = adapter.capture_outputs(example.inputs, targets)
        for target in targets:
            value = captured[target]
            totals[target] += float(value.detach().abs().mean().cpu().item())
    scores = {target: value / len(examples) for target, value in totals.items()}
    ranked = sorted(scores, key=lambda target: (-scores[target], target))
    chosen = tuple(ranked[:k])
    return CircuitCandidate(name=name, targets=chosen, scores=scores, source="activation-magnitude")


def discover_ablation_effect_candidate(
    adapter: ModelAdapter,
    discovery_examples: Sequence[EvidenceExample],
    all_targets: Sequence[str],
    *,
    metric: ScalarMetric,
    k: int,
    name: str = "discovery-single-target-ablation",
) -> CircuitCandidate:
    """Rank components by zero-ablation effect using discovery examples only."""

    if k <= 0:
        raise ValueError("k must be positive")
    targets = tuple(dict.fromkeys(str(target) for target in all_targets))
    if k > len(targets):
        raise ValueError("k cannot exceed the target universe size")
    examples = tuple(discovery_examples)
    if not examples:
        raise ValueError("ablation discovery requires discovery examples")
    if any(item.split is not EvidenceSplit.DISCOVERY for item in examples):
        raise ValueError("ablation discovery received a validation example")

    totals = {target: 0.0 for target in targets}
    for example in examples:
        baseline = metric(adapter.forward(example.inputs))
        captured = adapter.capture_outputs(example.inputs, targets)
        for target in targets:
            ablated = torch.zeros_like(captured[target])
            output = adapter.forward_with_replacements(example.inputs, {target: ablated})
            totals[target] += abs(metric(output) - baseline)
    scores = {target: value / len(examples) for target, value in totals.items()}
    ranked = sorted(scores, key=lambda target: (-scores[target], target))
    return CircuitCandidate(
        name=name,
        targets=tuple(ranked[:k]),
        scores=scores,
        source="discovery-single-target-ablation",
    )


def fit_discovery_mean_references(
    adapter: ModelAdapter,
    discovery_examples: Sequence[EvidenceExample],
    all_targets: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Fit scalar per-target mean donors exclusively on discovery activations."""

    examples = tuple(discovery_examples)
    if not examples:
        raise ValueError("mean-reference fitting requires discovery examples")
    if any(item.split is not EvidenceSplit.DISCOVERY for item in examples):
        raise ValueError("mean-reference fitting received a validation example")
    targets = tuple(dict.fromkeys(str(target) for target in all_targets))
    sums = {target: 0.0 for target in targets}
    counts = {target: 0 for target in targets}
    for example in examples:
        captured = adapter.capture_outputs(example.inputs, targets)
        for target in targets:
            value = captured[target].detach().to(dtype=torch.float64, device="cpu")
            sums[target] += float(value.sum().item())
            counts[target] += int(value.numel())
    references = {}
    for target in targets:
        if counts[target] <= 0:
            raise ValueError(f"target {target!r} produced no values for mean-reference fitting")
        references[target] = torch.tensor(sums[target] / counts[target], dtype=torch.float64)
    return references


def _validate_examples(
    examples: Sequence[EvidenceExample],
) -> tuple[tuple[EvidenceExample, ...], tuple[EvidenceExample, ...]]:
    examples = tuple(examples)
    if not examples:
        raise ValueError("evidence pack requires examples")
    ids = [item.example_id for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("example_id values must be unique")
    hashes = [item.input_hash for item in examples]
    if any(item is None for item in hashes):
        raise ValueError("all evidence inputs must support deterministic content hashing")
    concrete_hashes = [str(item) for item in hashes]
    if len(concrete_hashes) != len(set(concrete_hashes)):
        raise ValueError("duplicate input content across evidence examples is not allowed")
    discovery = tuple(item for item in examples if item.split is EvidenceSplit.DISCOVERY)
    validation = tuple(item for item in examples if item.split is EvidenceSplit.VALIDATION)
    if not discovery:
        raise ValueError("evidence pack requires at least one discovery example")
    if not validation:
        raise ValueError("evidence pack requires at least one validation example")
    return discovery, validation


def _normalization_failure(error: ValueError) -> bool:
    message = str(error)
    return any(message.startswith(prefix) for prefix in _NORMALIZATION_FAILURES)


def _evaluate_cases(
    *,
    adapter: ModelAdapter,
    metric: ScalarMetric,
    examples: Sequence[EvidenceExample],
    candidate: CircuitCandidate,
    spec: EvidencePackSpec,
    faithfulness_policy: FaithfulnessPolicy,
    mean_references: Mapping[str, torch.Tensor],
    seed_offset: int,
) -> tuple[EvidenceCaseResult, ...]:
    cases = []
    for example_index, example in enumerate(examples):
        input_hash = example.input_hash
        assert input_hash is not None
        for baseline_index, baseline in enumerate(spec.intervention_baselines):
            case_seed = spec.seed + seed_offset + 1000 * example_index + baseline_index
            references = mean_references if baseline == "mean" else None
            try:
                report = evaluate_adapter_circuit_faithfulness(
                    adapter=adapter,
                    inputs=example.inputs,
                    metric=metric,
                    all_targets=spec.target_universe,
                    candidate=candidate,
                    ablation_mode=baseline,
                    ablation_references=references,
                    random_trials=spec.random_trials,
                    seed=case_seed,
                    policy=faithfulness_policy,
                )
            except ValueError as error:
                if not _normalization_failure(error):
                    raise
                cases.append(
                    EvidenceCaseResult(
                        example_id=example.example_id,
                        split=example.split,
                        intervention_baseline=baseline,
                        input_hash=input_hash,
                        report=None,
                        invalid_reason=str(error),
                        metadata=example.metadata,
                    )
                )
                continue
            cases.append(
                EvidenceCaseResult(
                    example_id=example.example_id,
                    split=example.split,
                    intervention_baseline=baseline,
                    input_hash=input_hash,
                    report=report,
                    metadata=example.metadata,
                )
            )
    return tuple(cases)


def _promotion_decision(
    *,
    discovery: EvidenceAggregate,
    validation: EvidenceAggregate,
    magnitude_validation: EvidenceAggregate | None,
    spec: EvidencePackSpec,
    policy: EvidencePackPolicy,
) -> EvidencePromotionDecision:
    reasons = []
    if validation.n_examples < policy.min_validation_examples:
        reasons.append(
            f"validation examples {validation.n_examples} < {policy.min_validation_examples}"
        )
    if policy.require_all_cases_valid and discovery.n_invalid_cases:
        reasons.append(f"discovery contains {discovery.n_invalid_cases} invalid case(s)")
    if policy.require_all_cases_valid and validation.n_invalid_cases:
        reasons.append(f"validation contains {validation.n_invalid_cases} invalid case(s)")
    if validation.pass_rate < policy.min_validation_pass_rate:
        reasons.append(
            f"validation pass rate {validation.pass_rate:.3f} < "
            f"{policy.min_validation_pass_rate:.3f}"
        )
    if validation.median_joint_faithfulness < policy.min_validation_joint_median:
        reasons.append(
            f"validation joint median {validation.median_joint_faithfulness:.3f} < "
            f"{policy.min_validation_joint_median:.3f}"
        )
    drop = discovery.median_joint_faithfulness - validation.median_joint_faithfulness
    if drop > policy.max_joint_generalization_drop:
        reasons.append(
            f"joint generalization drop {drop:.3f} > {policy.max_joint_generalization_drop:.3f}"
        )
    if policy.require_multiple_intervention_baselines and len(spec.intervention_baselines) < 2:
        reasons.append("multiple intervention baselines are required")
    advantage = None
    if magnitude_validation is not None:
        advantage = (
            validation.median_joint_faithfulness
            - magnitude_validation.median_joint_faithfulness
        )
        if advantage < policy.min_validation_joint_advantage_vs_magnitude:
            reasons.append(
                f"validation joint advantage vs magnitude {advantage:.3f} < "
                f"{policy.min_validation_joint_advantage_vs_magnitude:.3f}"
            )
    return EvidencePromotionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        discovery_joint_median=discovery.median_joint_faithfulness,
        validation_joint_median=validation.median_joint_faithfulness,
        joint_generalization_drop=drop,
        validation_pass_rate=validation.pass_rate,
        validation_joint_advantage_vs_magnitude=advantage,
    )


def _model_hash(adapter: ModelAdapter) -> str | None:
    return stable_hash_or_none(adapter.model_fingerprint_payload())


def run_adapter_evidence_pack(
    *,
    spec: EvidencePackSpec,
    adapter: ModelAdapter,
    metric: ScalarMetric,
    examples: Sequence[EvidenceExample],
    candidate: CircuitCandidate | None = None,
    discover_candidate: CandidateDiscovery | None = None,
    pack_policy: EvidencePackPolicy | None = None,
    faithfulness_policy: FaithfulnessPolicy | None = None,
    include_magnitude_baseline: bool = True,
) -> EvidencePackResult:
    """Run discovery then held-out validation without exposing validation to discovery.

    Exactly one of ``candidate`` and ``discover_candidate`` must be supplied. If
    a discovery callback is used, it receives only examples labeled discovery.
    Candidate selection and mean-donor fitting finish before validation
    interventions begin. Deterministically hashable model state is checked for
    mutation after discovery and after intervention evaluation.
    """

    if (candidate is None) == (discover_candidate is None):
        raise ValueError("provide exactly one of candidate or discover_candidate")
    if metric.name != spec.metric_name:
        raise ValueError(
            f"metric name {metric.name!r} does not match evidence spec {spec.metric_name!r}"
        )

    discovery_examples, validation_examples = _validate_examples(examples)
    identities = tuple(item.identity_dict() for item in examples)
    dataset_hash = stable_hash(identities)
    pack_policy = pack_policy or EvidencePackPolicy()
    faithfulness_policy = faithfulness_policy or FaithfulnessPolicy()
    initial_model_hash = _model_hash(adapter)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        if candidate is None:
            assert discover_candidate is not None
            candidate = discover_candidate(adapter, discovery_examples, spec.target_universe)
        candidate_set = set(candidate.targets)
        if not candidate_set.issubset(set(spec.target_universe)):
            raise ValueError("discovered candidate contains targets outside target_universe")

        magnitude_candidate = None
        if include_magnitude_baseline:
            magnitude_candidate = discover_activation_magnitude_candidate(
                adapter,
                discovery_examples,
                spec.target_universe,
                k=len(candidate.targets),
            )

        mean_references = {}
        if "mean" in spec.intervention_baselines:
            mean_references = fit_discovery_mean_references(
                adapter,
                discovery_examples,
                spec.target_universe,
            )
        mean_reference_values = {
            target: float(value.item()) for target, value in mean_references.items()
        }

        post_discovery_hash = _model_hash(adapter)
        if (
            initial_model_hash is not None
            and post_discovery_hash is not None
            and initial_model_hash != post_discovery_hash
        ):
            raise RuntimeError("model state changed during candidate discovery")

        candidate_cases = _evaluate_cases(
            adapter=adapter,
            metric=metric,
            examples=(*discovery_examples, *validation_examples),
            candidate=candidate,
            spec=spec,
            faithfulness_policy=faithfulness_policy,
            mean_references=mean_references,
            seed_offset=0,
        )
        magnitude_cases = ()
        if magnitude_candidate is not None:
            magnitude_cases = _evaluate_cases(
                adapter=adapter,
                metric=metric,
                examples=(*discovery_examples, *validation_examples),
                candidate=magnitude_candidate,
                spec=spec,
                faithfulness_policy=faithfulness_policy,
                mean_references=mean_references,
                seed_offset=1_000_000,
            )

        final_model_hash = _model_hash(adapter)
        if (
            initial_model_hash is not None
            and final_model_hash is not None
            and initial_model_hash != final_model_hash
        ):
            raise RuntimeError("model state changed during evidence-pack evaluation")
    finally:
        wall_time_s = time.perf_counter() - started
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    discovery_cases = tuple(
        item for item in candidate_cases if item.split is EvidenceSplit.DISCOVERY
    )
    validation_cases = tuple(
        item for item in candidate_cases if item.split is EvidenceSplit.VALIDATION
    )
    discovery_aggregate = aggregate_evidence_cases(
        discovery_cases,
        bootstrap_samples=pack_policy.bootstrap_samples,
        seed=spec.seed,
    )
    validation_aggregate = aggregate_evidence_cases(
        validation_cases,
        bootstrap_samples=pack_policy.bootstrap_samples,
        seed=spec.seed + 1,
    )

    magnitude_validation_aggregate = None
    if magnitude_cases:
        magnitude_validation_cases = tuple(
            item for item in magnitude_cases if item.split is EvidenceSplit.VALIDATION
        )
        magnitude_validation_aggregate = aggregate_evidence_cases(
            magnitude_validation_cases,
            bootstrap_samples=pack_policy.bootstrap_samples,
            seed=spec.seed + 2,
        )

    promotion = _promotion_decision(
        discovery=discovery_aggregate,
        validation=validation_aggregate,
        magnitude_validation=magnitude_validation_aggregate,
        spec=spec,
        policy=pack_policy,
    )

    manifest = ExperimentManifest(
        experiment_name=spec.pack_id,
        method="held_out_circuit_evidence_pack",
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        model_hash=initial_model_hash,
        dataset_id=spec.dataset_id,
        dataset_hash=dataset_hash,
        method_version="1",
        parameters={
            "candidate": candidate.to_dict(),
            "discovery_example_ids": [item.example_id for item in discovery_examples],
            "discovery_method": spec.discovery_method,
            "faithfulness_policy": faithfulness_policy.to_dict(),
            "intervention_baselines": list(spec.intervention_baselines),
            "mean_ablation_references": mean_reference_values,
            "pack_policy": pack_policy.to_dict(),
            "random_trials": spec.random_trials,
            "target_universe": list(spec.target_universe),
            "tokenizer_id": spec.tokenizer_id,
            "tokenizer_revision": spec.tokenizer_revision,
            "validation_example_ids": [item.example_id for item in validation_examples],
        },
        seed=spec.seed,
        evidence_tier=spec.evidence_tier,
    )
    cuda_peak = None
    if torch.cuda.is_available():
        cuda_peak = float(torch.cuda.max_memory_allocated() / (1024**2))
    telemetry = EvidenceTelemetry(
        wall_time_s=wall_time_s,
        peak_python_memory_mb=float(peak_python / (1024**2)),
        peak_cuda_memory_mb=cuda_peak,
        package_versions=_package_versions(),
    )
    return EvidencePackResult(
        spec=spec,
        policy=pack_policy,
        faithfulness_policy=faithfulness_policy,
        candidate=candidate,
        magnitude_candidate=magnitude_candidate,
        candidate_cases=candidate_cases,
        magnitude_cases=tuple(magnitude_cases),
        discovery_aggregate=discovery_aggregate,
        validation_aggregate=validation_aggregate,
        magnitude_validation_aggregate=magnitude_validation_aggregate,
        promotion=promotion,
        manifest=manifest,
        telemetry=telemetry,
        discovery_example_ids=tuple(item.example_id for item in discovery_examples),
        validation_example_ids=tuple(item.example_id for item in validation_examples),
        example_identities=identities,
        mean_ablation_references=mean_reference_values,
    )


def write_evidence_pack_artifact(
    result: EvidencePackResult,
    path: str | Path,
) -> Path:
    """Write a self-checking JSON artifact that intentionally excludes raw inputs."""

    destination = Path(path)
    payload = result.to_dict()
    envelope = {
        "artifact_hash": stable_hash(payload),
        "artifact_schema": EVIDENCE_ARTIFACT_SCHEMA,
        "result": payload,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(envelope, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def read_evidence_pack_artifact(path: str | Path) -> dict[str, Any]:
    """Load an evidence artifact and reject schema or hash corruption."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_schema") != EVIDENCE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported evidence-pack artifact schema")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("evidence-pack artifact result must be an object")
    expected = payload.get("artifact_hash")
    observed = stable_hash(result)
    if expected != observed:
        raise ValueError("evidence-pack artifact hash mismatch")
    if result.get("schema_version") != EVIDENCE_PACK_SCHEMA:
        raise ValueError("unsupported evidence-pack result schema")
    return result
