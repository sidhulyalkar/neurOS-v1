"""Mechanistic studies over sequence-aligned neural foundation-model states.

The integration is intentionally duck-typed: a NeuroFM-family model only needs
to be a ``torch.nn.Module`` with an addressable component path. This keeps
``neuros-mechint`` compatible with NeuroFM-X checkpoints without making the
experiment kernel depend on NeuroFM implementation details.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from orion.contracts import RepresentationBatch
from torch import nn

from neuros_mechint.adapters import ModelCall, PyTorchAdapter
from neuros_mechint.benchmarks import (
    CheckpointMechanismState,
    MechanismContext,
    MechanismEmergenceReport,
    analyze_mechanism_emergence,
)
from neuros_mechint.core import EvidenceTier

from .orion_study import (
    OrionRepresentationContext,
    OrionSharedComputationStudyResult,
    run_shared_representation_study,
)


@dataclass(frozen=True, slots=True)
class NeuroFMProbeSpec:
    """Describe the internal module state to expose as an ORION representation.

    ``batch_index=None`` means the captured module output is already unbatched
    with its first axis representing time. The default assumes ordinary batched
    neural-model tensors and selects batch element zero before interpreting the
    remaining first axis as time.
    """

    component_path: str = "backbone"
    batch_index: int | None = 0

    def __post_init__(self) -> None:
        if not self.component_path:
            raise ValueError("component_path must be non-empty")
        if self.batch_index is not None and self.batch_index < 0:
            raise ValueError("batch_index must be non-negative or None")


class NeuroFMRepresentationProbe:
    """Capture a tensor-valued NeuroFM module output as ``RepresentationBatch``."""

    def __init__(self, model: nn.Module, spec: NeuroFMProbeSpec | None = None) -> None:
        self.model = model
        self.spec = spec or NeuroFMProbeSpec()
        self.adapter = PyTorchAdapter(model)

    def capture(
        self,
        model_inputs: Any,
        *,
        input_timestamps_ns: Sequence[int],
        representation_timestamps_ns: Sequence[int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RepresentationBatch:
        """Capture one state and preserve only defensible time alignment.

        Batched outputs select ``batch_index`` first. Set ``batch_index=None``
        only when a component is known to return an already-unbatched
        ``[time, ...]`` representation.

        If the resulting time dimension equals the input timestamp count,
        ``input_timestamps_ns`` are reused. If a component compresses or expands
        time, callers must supply ``representation_timestamps_ns`` explicitly.
        The adapter never invents timestamps for Perceiver latents, pooled
        states, or other compressed representations.
        """

        was_training = self.model.training
        self.model.eval()
        try:
            captured = self.adapter.capture_outputs(
                model_inputs,
                [self.spec.component_path],
            )[self.spec.component_path]
        finally:
            self.model.train(was_training)

        if captured.ndim < 2:
            raise ValueError(
                "captured representation must have batch/features or time/features axes"
            )

        if self.spec.batch_index is None:
            values = captured
        else:
            if self.spec.batch_index >= captured.shape[0]:
                raise IndexError("batch_index is outside the captured batch dimension")
            values = captured[self.spec.batch_index]

        if values.ndim == 0:
            raise ValueError("captured representation has no feature dimension after batching")
        if values.ndim == 1:
            values = values[None, :]
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)

        input_timestamps = np.asarray(input_timestamps_ns, dtype=np.int64)
        if representation_timestamps_ns is None:
            if len(input_timestamps) != values.shape[0]:
                raise ValueError(
                    "captured time dimension differs from input timestamps; "
                    "provide representation_timestamps_ns explicitly"
                )
            timestamps = input_timestamps
        else:
            timestamps = np.asarray(representation_timestamps_ns, dtype=np.int64)
            if len(timestamps) != values.shape[0]:
                raise ValueError(
                    "representation_timestamps_ns length must match captured time dimension"
                )

        payload = {
            "source": "neurofm_module_probe",
            "component_path": self.spec.component_path,
            "batch_index": self.spec.batch_index,
            "model_module": type(self.model).__module__,
            "model_class": type(self.model).__qualname__,
        }
        payload.update(dict(metadata or {}))
        return RepresentationBatch(
            values=values.detach().cpu().numpy(),
            timestamps_ns=timestamps,
            metadata=payload,
        )


@dataclass(frozen=True, slots=True)
class NeuroFMCheckpointContext:
    """One model checkpoint, input batch, scorer, and experimental identity."""

    context: MechanismContext
    training_step: int
    model: nn.Module
    model_inputs: Any
    input_timestamps_ns: Sequence[int]
    scorer: Callable[[RepresentationBatch], float]
    probe_spec: NeuroFMProbeSpec = field(default_factory=NeuroFMProbeSpec)
    representation_timestamps_ns: Sequence[int] | None = None
    alignment_origin_ns: int | None = None
    alignment_label: str | None = None
    evidence_tier: EvidenceTier = EvidenceTier.UNIT

    def __post_init__(self) -> None:
        if self.training_step < 0:
            raise ValueError("training_step must be non-negative")

    def resolved_context(self) -> MechanismContext:
        checkpoint = self.context.checkpoint or f"step:{self.training_step}"
        return replace(self.context, checkpoint=checkpoint)

    def capture_representation(self) -> RepresentationBatch:
        return NeuroFMRepresentationProbe(self.model, self.probe_spec).capture(
            self.model_inputs,
            input_timestamps_ns=self.input_timestamps_ns,
            representation_timestamps_ns=self.representation_timestamps_ns,
            metadata={
                "training_step": self.training_step,
                "architecture": self.context.architecture,
                "checkpoint": self.resolved_context().checkpoint,
            },
        )


@dataclass(frozen=True, slots=True)
class NeuroFMMechanismLabResult:
    """Architecture comparison plus checkpoint-emergence reports."""

    shared_study: OrionSharedComputationStudyResult
    emergence_reports: Mapping[str, MechanismEmergenceReport]
    checkpoint_steps: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_study": self.shared_study.to_dict(),
            "emergence_reports": {
                key: value.to_dict() for key, value in self.emergence_reports.items()
            },
            "checkpoint_steps": dict(self.checkpoint_steps),
        }


def _trajectory_key(context: MechanismContext) -> str:
    subject = context.subject_id or "none"
    return f"{context.architecture}|{context.dataset_id}|{context.session_id}|{subject}"


def run_neurofm_mechanism_lab(
    checkpoints: Sequence[NeuroFMCheckpointContext],
    *,
    window_ns: int,
    stride_ns: int | None = None,
    mode: str = "zero",
    top_k: int = 5,
    seed: int = 0,
    include_shuffle_controls: bool = True,
    include_feature_audits: bool = False,
    feature_groups: Sequence[Sequence[int]] | None = None,
    emergence_effect_fraction: float = 0.5,
    emergence_stable_spearman: float = 0.8,
    emergence_stable_sign_agreement: float = 0.8,
    emergence_min_shared_target_fraction: float = 0.75,
    emergence_consecutive_checkpoints: int = 2,
) -> NeuroFMMechanismLabResult:
    """Run architecture and longitudinal mechanism studies over NeuroFM checkpoints.

    Architecture comparison is delegated to the v0.3 shared-computation layer.
    Longitudinal emergence is only computed within groups where architecture,
    dataset, session, subject, and metric are fixed, so checkpoint is the sole
    varying scientific axis.
    """

    checkpoints = tuple(checkpoints)
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoint contexts are required")

    context_ids = [item.resolved_context().context_id for item in checkpoints]
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("context_id values must be unique")

    orion_contexts = []
    steps_by_context = {}
    for checkpoint in checkpoints:
        context = checkpoint.resolved_context()
        steps_by_context[context.context_id] = checkpoint.training_step
        orion_contexts.append(
            OrionRepresentationContext(
                context=context,
                representation=checkpoint.capture_representation(),
                scorer=checkpoint.scorer,
                model_id=f"{context.architecture}:{context.checkpoint}",
                alignment_origin_ns=checkpoint.alignment_origin_ns,
                alignment_label=checkpoint.alignment_label,
                evidence_tier=checkpoint.evidence_tier,
            )
        )

    shared = run_shared_representation_study(
        orion_contexts,
        window_ns=window_ns,
        stride_ns=stride_ns,
        mode=mode,
        top_k=top_k,
        seed=seed,
        include_shuffle_controls=include_shuffle_controls,
        include_feature_audits=include_feature_audits,
        feature_groups=feature_groups,
    )

    record_by_context = {audit.context.context_id: audit.record for audit in shared.audits}
    grouped: dict[str, list[CheckpointMechanismState]] = defaultdict(list)
    for checkpoint in checkpoints:
        context = checkpoint.resolved_context()
        grouped[_trajectory_key(context)].append(
            CheckpointMechanismState(
                step=checkpoint.training_step,
                record=record_by_context[context.context_id],
            )
        )

    emergence = {}
    for key, states in grouped.items():
        if len(states) < 2:
            continue
        emergence[key] = analyze_mechanism_emergence(
            states,
            effect_fraction=emergence_effect_fraction,
            stable_spearman=emergence_stable_spearman,
            stable_sign_agreement=emergence_stable_sign_agreement,
            min_shared_target_fraction=emergence_min_shared_target_fraction,
            consecutive_checkpoints=emergence_consecutive_checkpoints,
            top_k=top_k,
        )

    return NeuroFMMechanismLabResult(
        shared_study=shared,
        emergence_reports=emergence,
        checkpoint_steps=steps_by_context,
    )


def model_call(
    tokens: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> ModelCall:
    """Build a PyTorch adapter call compatible with NeuroFM-X ``forward``."""

    kwargs: dict[str, Any] = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return ModelCall(args=(tokens,), kwargs=kwargs)
