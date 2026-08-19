"""End-to-end shared-computation studies for ORION representations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from orion.contracts import RepresentationBatch

from neuros_mechint.benchmarks import (
    CausalEffectRecord,
    HypothesisPolicy,
    MechanismContext,
    SharedComputationAnalysis,
    analyze_shared_computation,
)
from neuros_mechint.core import (
    EvidenceTier,
    InputCausalExperiment,
    InputMetric,
    stable_hash,
)
from neuros_mechint.core.results import InputExperimentResult

from .orion import RepresentationFeatureAblation, representation_window_audit


@dataclass(frozen=True, slots=True)
class OrionRepresentationContext:
    """One ORION representation/scorer pair in a comparative study."""

    context: MechanismContext
    representation: RepresentationBatch
    scorer: Callable[[RepresentationBatch], float]
    model_id: str | None = None
    alignment_origin_ns: int | None = None
    alignment_label: str | None = None
    evidence_tier: EvidenceTier = EvidenceTier.UNIT

    def resolved_alignment_origin_ns(self) -> int:
        if self.alignment_origin_ns is not None:
            return int(self.alignment_origin_ns)
        timestamps = np.asarray(self.representation.timestamps_ns, dtype=np.int64)
        if len(timestamps) == 0:
            raise ValueError("cannot infer alignment origin from an empty representation")
        return int(timestamps.min())


@dataclass(frozen=True, slots=True)
class RepresentationFeatureShuffle:
    """Matched feature control that preserves values but permutes their time association."""

    feature_indices: tuple[int, ...]
    seed: int = 0
    name: str = "orion_representation_feature_shuffle"

    def __init__(
        self,
        feature_indices: Sequence[int],
        seed: int = 0,
        name: str = "orion_representation_feature_shuffle",
    ) -> None:
        indices = tuple(sorted({int(index) for index in feature_indices}))
        if not indices:
            raise ValueError("feature_indices must not be empty")
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "name", name)

    @property
    def target(self) -> str:
        return "representation_features:" + ",".join(map(str, self.feature_indices))

    def apply(self, reference: RepresentationBatch) -> RepresentationBatch:
        values = np.asarray(reference.values).copy()
        if values.ndim != 2:
            raise ValueError("feature shuffle currently expects [time, features]")
        indices = np.asarray(self.feature_indices, dtype=np.int64)
        if (indices < 0).any() or (indices >= values.shape[1]).any():
            raise IndexError("feature index is outside the representation dimension")
        if len(values) > 1:
            rng = np.random.default_rng(self.seed)
            permutation = rng.permutation(len(values))
            original = values.copy()
            values[:, indices] = original[permutation][:, indices]
        metadata = dict(reference.metadata)
        metadata.update(
            {
                "mechint_intervention": self.name,
                "mechint_feature_indices": list(self.feature_indices),
            }
        )
        return RepresentationBatch(
            values=values,
            timestamps_ns=np.asarray(reference.timestamps_ns).copy(),
            mask=(
                None
                if reference.mask is None
                else np.asarray(reference.mask, dtype=bool).copy()
            ),
            metadata=metadata,
        )

    def metadata(self) -> Mapping[str, Any]:
        return {"feature_indices": list(self.feature_indices), "seed": self.seed}


@dataclass(frozen=True, slots=True)
class OrionContextAudit:
    """Raw ORION causal audit plus its canonical cross-context effect record."""

    context: MechanismContext
    alignment_origin_ns: int
    alignment_label: str
    result: InputExperimentResult
    record: CausalEffectRecord
    feature_result: InputExperimentResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "alignment_origin_ns": self.alignment_origin_ns,
            "alignment_label": self.alignment_label,
            "audit": self.result.to_dict(),
            "feature_audit": (
                None if self.feature_result is None else self.feature_result.to_dict()
            ),
            "canonical_effect_record": self.record.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OrionSharedComputationStudyResult:
    """Complete representation -> intervention -> hypothesis study artifact."""

    audits: tuple[OrionContextAudit, ...]
    analysis: SharedComputationAnalysis
    window_ns: int
    stride_ns: int | None
    ablation_mode: str
    top_k: int
    seed: int
    include_shuffle_controls: bool
    include_feature_audits: bool

    @property
    def context_manifest_hashes(self) -> dict[str, str]:
        return {
            audit.context.context_id: audit.result.manifest.content_hash
            for audit in self.audits
        }

    @property
    def study_fingerprint(self) -> str:
        """Deterministic fingerprint of the scientific design and measured effects."""

        contexts = []
        for audit in self.audits:
            context = audit.context
            record = audit.record
            contexts.append(
                {
                    "context_id": context.context_id,
                    "architecture": context.architecture,
                    "dataset_id": context.dataset_id,
                    "session_id": context.session_id,
                    "subject_id": context.subject_id,
                    "checkpoint": context.checkpoint,
                    "alignment_origin_ns": audit.alignment_origin_ns,
                    "alignment_label": audit.alignment_label,
                    "baseline_metric": record.baseline_metric,
                    "metric_name": record.metric_name,
                    "effect_map": dict(record.effect_map),
                    "control_map": dict(record.control_map),
                    "feature_audit": (
                        None
                        if audit.feature_result is None
                        else {
                            "baseline_metric": audit.feature_result.baseline_metric,
                            "metric_name": audit.feature_result.metric_name,
                            "effects": {
                                item.target: item.effect for item in audit.feature_result.effects
                            },
                            "controls": {
                                item.target: item.effect for item in audit.feature_result.controls
                            },
                        }
                    ),
                }
            )
        return stable_hash(
            {
                "parameters": {
                    "window_ns": self.window_ns,
                    "stride_ns": self.stride_ns,
                    "ablation_mode": self.ablation_mode,
                    "top_k": self.top_k,
                    "seed": self.seed,
                    "include_shuffle_controls": self.include_shuffle_controls,
                    "include_feature_audits": self.include_feature_audits,
                },
                "contexts": contexts,
                "analysis": self.analysis.to_dict(),
            }
        )

    @property
    def study_hash(self) -> str:
        """Compatibility name for the deterministic scientific fingerprint."""

        return self.study_fingerprint

    @property
    def run_hash(self) -> str:
        """Run-specific hash that also includes timestamped environment manifests."""

        return stable_hash(
            {
                "study_fingerprint": self.study_fingerprint,
                "context_manifests": self.context_manifest_hashes,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_fingerprint": self.study_fingerprint,
            "study_hash": self.study_hash,
            "run_hash": self.run_hash,
            "context_manifest_hashes": self.context_manifest_hashes,
            "parameters": {
                "window_ns": self.window_ns,
                "stride_ns": self.stride_ns,
                "ablation_mode": self.ablation_mode,
                "top_k": self.top_k,
                "seed": self.seed,
                "include_shuffle_controls": self.include_shuffle_controls,
                "include_feature_audits": self.include_feature_audits,
            },
            "audits": [audit.to_dict() for audit in self.audits],
            "analysis": self.analysis.to_dict(),
        }


def _canonical_target(effect: Any, *, origin_ns: int) -> str:
    metadata = dict(getattr(effect, "metadata", {}))
    start_ns = metadata.get("start_ns")
    end_ns = metadata.get("end_ns")
    if start_ns is None or end_ns is None:
        return str(effect.target)
    return f"representation_relative[{int(start_ns) - origin_ns}:{int(end_ns) - origin_ns}]"


def _canonical_map(items: Sequence[Any], *, origin_ns: int) -> dict[str, float]:
    effect_map: dict[str, float] = {}
    for item in items:
        target = _canonical_target(item, origin_ns=origin_ns)
        if target in effect_map:
            raise ValueError(f"duplicate canonical intervention target: {target}")
        effect_map[target] = float(item.effect)
    return effect_map


def audit_representation_features(
    study_context: OrionRepresentationContext,
    *,
    feature_groups: Sequence[Sequence[int]] | None = None,
    mode: str = "zero",
    seed: int = 0,
    include_shuffle_controls: bool = True,
) -> InputExperimentResult:
    """Measure feature-group necessity with matched temporal-permutation controls."""

    representation = study_context.representation
    values = np.asarray(representation.values)
    if values.ndim != 2:
        raise ValueError("feature audit currently expects [time, features]")
    groups = (
        tuple((index,) for index in range(values.shape[1]))
        if feature_groups is None
        else tuple(tuple(int(index) for index in group) for group in feature_groups)
    )
    if not groups or any(not group for group in groups):
        raise ValueError("feature_groups must contain at least one non-empty group")

    interventions = tuple(
        RepresentationFeatureAblation(group, mode=mode) for group in groups
    )
    controls: tuple[RepresentationFeatureShuffle, ...] = ()
    if include_shuffle_controls:
        controls = tuple(
            RepresentationFeatureShuffle(group, seed=seed + index)
            for index, group in enumerate(groups)
        )
    context = study_context.context
    experiment = InputCausalExperiment(
        reference=representation,
        metric=InputMetric(study_context.scorer, name="orion_representation_feature_score"),
        experiment_name=f"representation-feature-audit:{context.context_id}",
        model_id=study_context.model_id or f"{context.architecture}:{context.context_id}",
        dataset_id=context.dataset_id,
        seed=seed,
        evidence_tier=study_context.evidence_tier,
        metadata={"feature_groups": [list(group) for group in groups], "mode": mode},
    )
    return experiment.run(interventions, controls=controls)


def audit_orion_representation_context(
    study_context: OrionRepresentationContext,
    *,
    window_ns: int,
    stride_ns: int | None = None,
    mode: str = "zero",
    seed: int = 0,
    include_shuffle_controls: bool = True,
    include_feature_audit: bool = False,
    feature_groups: Sequence[Sequence[int]] | None = None,
) -> OrionContextAudit:
    """Audit one ORION representation and canonicalize effects to relative time."""

    context = study_context.context
    origin_ns = study_context.resolved_alignment_origin_ns()
    alignment_label = study_context.alignment_label or (
        "recording_start" if study_context.alignment_origin_ns is None else "unspecified_event"
    )
    result = representation_window_audit(
        study_context.representation,
        study_context.scorer,
        window_ns=window_ns,
        stride_ns=stride_ns,
        mode=mode,
        model_id=study_context.model_id or f"{context.architecture}:{context.context_id}",
        dataset_id=context.dataset_id,
        experiment_name=f"shared-computation:{context.context_id}",
        seed=seed,
        evidence_tier=study_context.evidence_tier,
        include_shuffle_controls=include_shuffle_controls,
    )
    feature_result = None
    if include_feature_audit:
        feature_result = audit_representation_features(
            study_context,
            feature_groups=feature_groups,
            mode=mode,
            seed=seed,
            include_shuffle_controls=include_shuffle_controls,
        )
    record = CausalEffectRecord(
        context=context,
        baseline_metric=result.baseline_metric,
        effect_map=_canonical_map(result.effects, origin_ns=origin_ns),
        control_map=_canonical_map(result.controls, origin_ns=origin_ns),
        metric_name=result.metric_name,
    )
    return OrionContextAudit(
        context=context,
        alignment_origin_ns=origin_ns,
        alignment_label=alignment_label,
        result=result,
        record=record,
        feature_result=feature_result,
    )


def run_shared_representation_study(
    contexts: Sequence[OrionRepresentationContext],
    *,
    window_ns: int,
    stride_ns: int | None = None,
    mode: str = "zero",
    top_k: int = 5,
    seed: int = 0,
    include_shuffle_controls: bool = True,
    include_feature_audits: bool = False,
    feature_groups: Sequence[Sequence[int]] | None = None,
    policy: HypothesisPolicy | None = None,
) -> OrionSharedComputationStudyResult:
    """Execute the complete ORION shared-neural-computation analysis chain.

    The pipeline is intentionally deterministic and explicit:

    ``RepresentationBatch -> independent causal windows -> canonical effect maps
    -> cross-context/architecture stability -> falsifiable candidate hypotheses``.

    Each context uses its own absolute timestamps for intervention execution but
    effect-map keys are converted to event-relative time using
    ``alignment_origin_ns``. If no origin is supplied, the first timestamp is
    used. For scientific event-aligned studies callers should provide the same
    semantic event origin (stimulus onset, movement onset, etc.) per context.

    Optional feature audits are retained per context but are not automatically
    compared across architectures because raw latent dimensions are not assumed
    to share semantic identity without an explicit feature alignment.
    """

    contexts = tuple(contexts)
    if len(contexts) < 2:
        raise ValueError("at least two ORION representation contexts are required")
    if window_ns <= 0:
        raise ValueError("window_ns must be positive")
    if stride_ns is not None and stride_ns <= 0:
        raise ValueError("stride_ns must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    explicit_labels = {
        context.alignment_label for context in contexts if context.alignment_label is not None
    }
    if len(explicit_labels) > 1:
        raise ValueError(
            "all explicitly labeled contexts must use the same semantic alignment event"
        )

    audits = tuple(
        audit_orion_representation_context(
            context,
            window_ns=window_ns,
            stride_ns=stride_ns,
            mode=mode,
            seed=seed + index,
            include_shuffle_controls=include_shuffle_controls,
            include_feature_audit=include_feature_audits,
            feature_groups=feature_groups,
        )
        for index, context in enumerate(contexts)
    )
    analysis = analyze_shared_computation(
        [audit.record for audit in audits],
        top_k=top_k,
        policy=policy,
    )
    return OrionSharedComputationStudyResult(
        audits=audits,
        analysis=analysis,
        window_ns=window_ns,
        stride_ns=stride_ns,
        ablation_mode=mode,
        top_k=top_k,
        seed=seed,
        include_shuffle_controls=include_shuffle_controls,
        include_feature_audits=include_feature_audits,
    )


def context_from_metadata(
    *,
    context_id: str,
    architecture: str,
    dataset_id: str,
    session_id: str,
    subject_id: str | None = None,
    checkpoint: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MechanismContext:
    """Convenience constructor for notebook and experiment-manifest workflows."""

    return MechanismContext(
        context_id=context_id,
        architecture=architecture,
        dataset_id=dataset_id,
        session_id=session_id,
        subject_id=subject_id,
        checkpoint=checkpoint,
        metadata=dict(metadata or {}),
    )
