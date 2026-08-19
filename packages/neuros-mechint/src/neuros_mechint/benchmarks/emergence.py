"""Track how causal intervention profiles emerge across training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .shared_computation import CausalEffectRecord
from .stability import EffectMapStability, compare_effect_maps


@dataclass(frozen=True, slots=True)
class CheckpointMechanismState:
    """One causal-effect record attached to an ordered training step."""

    step: int
    record: CausalEffectRecord

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "record": self.record.to_dict()}


@dataclass(frozen=True, slots=True)
class TargetEmergence:
    """When one intervention target becomes a stable part of the final mechanism."""

    target: str
    first_detected_step: int | None
    first_stable_step: int | None
    final_effect: float
    peak_absolute_effect: float
    final_sign: int

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CheckpointSimilarity:
    """Similarity of one checkpoint's causal map to the final checkpoint."""

    step: int
    context_id: str
    stability: EffectMapStability

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "context_id": self.context_id,
            "stability": self.stability.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MechanismEmergenceReport:
    """Longitudinal summary of causal-map formation during training."""

    architecture: str
    dataset_id: str
    session_id: str
    subject_id: str | None
    metric_name: str
    final_step: int
    final_context_id: str
    global_stable_step: int | None
    target_emergence: tuple[TargetEmergence, ...]
    checkpoint_similarity: tuple[CheckpointSimilarity, ...]
    parameters: Mapping[str, int | float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def stable_target_fraction(self) -> float:
        if not self.target_emergence:
            return 0.0
        stable = sum(item.first_stable_step is not None for item in self.target_emergence)
        return stable / len(self.target_emergence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "metric_name": self.metric_name,
            "final_step": self.final_step,
            "final_context_id": self.final_context_id,
            "global_stable_step": self.global_stable_step,
            "stable_target_fraction": self.stable_target_fraction,
            "target_emergence": [item.to_dict() for item in self.target_emergence],
            "checkpoint_similarity": [item.to_dict() for item in self.checkpoint_similarity],
            "parameters": dict(self.parameters),
        }


def _same_longitudinal_context(
    left: CausalEffectRecord,
    right: CausalEffectRecord,
) -> bool:
    a = left.context
    b = right.context
    return (
        a.architecture == b.architecture
        and a.dataset_id == b.dataset_id
        and a.session_id == b.session_id
        and a.subject_id == b.subject_id
        and left.metric_name == right.metric_name
    )


def _qualifies_global(
    stability: EffectMapStability,
    *,
    stable_spearman: float,
    stable_sign_agreement: float,
    min_shared_target_fraction: float,
) -> bool:
    return (
        stability.spearman_r is not None
        and stability.spearman_r >= stable_spearman
        and stability.sign_agreement >= stable_sign_agreement
        and stability.shared_target_fraction >= min_shared_target_fraction
    )


def analyze_mechanism_emergence(
    states: Sequence[CheckpointMechanismState],
    *,
    effect_fraction: float = 0.5,
    stable_spearman: float = 0.8,
    stable_sign_agreement: float = 0.8,
    min_shared_target_fraction: float = 0.75,
    consecutive_checkpoints: int = 2,
    top_k: int = 5,
) -> MechanismEmergenceReport:
    """Find when a checkpoint trajectory acquires its final causal structure.

    All records must describe the same architecture, dataset, session, subject,
    and metric. Checkpoint is the only scientific context axis allowed to vary.
    A target is considered detected once it has the final sign and reaches
    ``effect_fraction`` of its final magnitude. It is considered stable at the
    first checkpoint from which that criterion remains true through all later
    checkpoints.

    ``global_stable_step`` is the earliest checkpoint from which the requested
    number of consecutive maps all satisfy rank, sign, and target-coverage
    thresholds relative to the final map.
    """

    states = tuple(sorted(states, key=lambda item: item.step))
    if len(states) < 2:
        raise ValueError("at least two checkpoint states are required")
    if len({item.step for item in states}) != len(states):
        raise ValueError("checkpoint steps must be unique")
    if not 0.0 < effect_fraction <= 1.0:
        raise ValueError("effect_fraction must lie in (0, 1]")
    if not 0.0 <= stable_spearman <= 1.0:
        raise ValueError("stable_spearman must lie in [0, 1]")
    if not 0.0 <= stable_sign_agreement <= 1.0:
        raise ValueError("stable_sign_agreement must lie in [0, 1]")
    if not 0.0 < min_shared_target_fraction <= 1.0:
        raise ValueError("min_shared_target_fraction must lie in (0, 1]")
    if consecutive_checkpoints <= 0:
        raise ValueError("consecutive_checkpoints must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    reference = states[0].record
    if any(not _same_longitudinal_context(reference, item.record) for item in states[1:]):
        raise ValueError(
            "mechanism-emergence analysis requires architecture, dataset, session, subject, "
            "and metric_name to remain fixed"
        )

    final_state = states[-1]
    final_map = dict(final_state.record.effect_map)

    similarities = []
    for state in states:
        similarities.append(
            CheckpointSimilarity(
                step=state.step,
                context_id=state.record.context.context_id,
                stability=compare_effect_maps(
                    state.record.effect_map,
                    final_map,
                    top_k=top_k,
                ),
            )
        )

    global_stable_step = None
    required = min(consecutive_checkpoints, len(states))
    for start_index in range(len(states) - required + 1):
        window = similarities[start_index : start_index + required]
        if all(
            _qualifies_global(
                item.stability,
                stable_spearman=stable_spearman,
                stable_sign_agreement=stable_sign_agreement,
                min_shared_target_fraction=min_shared_target_fraction,
            )
            for item in window
        ):
            global_stable_step = window[0].step
            break

    target_reports = []
    for target in sorted(final_map):
        final_effect = float(final_map[target])
        final_sign = int(np.sign(final_effect))
        threshold = abs(final_effect) * effect_fraction

        qualifies = []
        absolute_effects = []
        for state in states:
            value = state.record.effect_map.get(target)
            if value is None:
                qualifies.append(False)
                continue
            value = float(value)
            absolute_effects.append(abs(value))
            sign_matches = final_sign == 0 or int(np.sign(value)) == final_sign
            magnitude_matches = abs(value) >= threshold
            qualifies.append(sign_matches and magnitude_matches)

        first_detected_step = next(
            (state.step for state, ok in zip(states, qualifies, strict=True) if ok),
            None,
        )
        first_stable_step = None
        for index, ok in enumerate(qualifies):
            if ok and all(qualifies[index:]):
                first_stable_step = states[index].step
                break

        target_reports.append(
            TargetEmergence(
                target=target,
                first_detected_step=first_detected_step,
                first_stable_step=first_stable_step,
                final_effect=final_effect,
                peak_absolute_effect=max(absolute_effects, default=0.0),
                final_sign=final_sign,
            )
        )

    context = final_state.record.context
    return MechanismEmergenceReport(
        architecture=context.architecture,
        dataset_id=context.dataset_id,
        session_id=context.session_id,
        subject_id=context.subject_id,
        metric_name=final_state.record.metric_name,
        final_step=final_state.step,
        final_context_id=context.context_id,
        global_stable_step=global_stable_step,
        target_emergence=tuple(target_reports),
        checkpoint_similarity=tuple(similarities),
        parameters={
            "effect_fraction": effect_fraction,
            "stable_spearman": stable_spearman,
            "stable_sign_agreement": stable_sign_agreement,
            "min_shared_target_fraction": min_shared_target_fraction,
            "consecutive_checkpoints": consecutive_checkpoints,
            "top_k": top_k,
        },
    )
