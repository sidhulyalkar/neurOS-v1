"""Causal audits for ORION token and representation inputs.

The integration intentionally lives in ``neuros-mechint`` rather than ORION.
ORION owns representation contracts and tokenizers; mechanistic-interpretability
experiments may depend on those stable contracts without reversing the
dependency direction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from orion.contracts import NeuroTokenBatch, RepresentationBatch

from neuros_mechint.core import EvidenceTier, InputCausalExperiment, InputMetric
from neuros_mechint.core.results import InputExperimentResult


def _copy_batch(
    batch: NeuroTokenBatch,
    *,
    token_ids: np.ndarray | None = None,
    timestamps_ns: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    side_features: Mapping[str, np.ndarray] | None = None,
    metadata_update: Mapping[str, Any] | None = None,
) -> NeuroTokenBatch:
    metadata = dict(batch.metadata)
    metadata.update(metadata_update or {})
    return NeuroTokenBatch(
        token_ids=np.asarray(batch.token_ids if token_ids is None else token_ids).copy(),
        timestamps_ns=np.asarray(
            batch.timestamps_ns if timestamps_ns is None else timestamps_ns
        ).copy(),
        mask=(
            np.asarray(batch.mask, dtype=bool).copy()
            if mask is None and batch.mask is not None
            else (None if mask is None else np.asarray(mask, dtype=bool).copy())
        ),
        side_features={
            key: np.asarray(value).copy()
            for key, value in (
                batch.side_features if side_features is None else side_features
            ).items()
        },
        metadata=metadata,
    )


def _copy_representation(
    batch: RepresentationBatch,
    *,
    values: np.ndarray | None = None,
    metadata_update: Mapping[str, Any] | None = None,
) -> RepresentationBatch:
    metadata = dict(batch.metadata)
    metadata.update(metadata_update or {})
    return RepresentationBatch(
        values=np.asarray(batch.values if values is None else values).copy(),
        timestamps_ns=np.asarray(batch.timestamps_ns).copy(),
        mask=(
            None
            if batch.mask is None
            else np.asarray(batch.mask, dtype=bool).copy()
        ),
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class TokenTimeWindowMask:
    """Replace token IDs inside a temporal window while preserving timestamps."""

    start_ns: int
    end_ns: int
    replacement_token_id: int = 0
    name: str = "orion_token_time_window_mask"

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")

    @property
    def target(self) -> str:
        return f"tokens[{self.start_ns}:{self.end_ns}]"

    def apply(self, reference: NeuroTokenBatch) -> NeuroTokenBatch:
        ids = np.asarray(reference.token_ids).copy()
        selected = (
            (np.asarray(reference.timestamps_ns) >= self.start_ns)
            & (np.asarray(reference.timestamps_ns) < self.end_ns)
        )
        ids[selected] = self.replacement_token_id
        return _copy_batch(
            reference,
            token_ids=ids,
            metadata_update={
                "mechint_intervention": self.name,
                "mechint_selected_tokens": int(selected.sum()),
            },
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "replacement_token_id": self.replacement_token_id,
        }


@dataclass(frozen=True, slots=True)
class TokenTypeAblation:
    """Replace every occurrence of selected token IDs."""

    token_ids: tuple[int, ...]
    replacement_token_id: int = 0
    name: str = "orion_token_type_ablation"

    def __init__(
        self,
        token_ids: Sequence[int],
        replacement_token_id: int = 0,
        name: str = "orion_token_type_ablation",
    ) -> None:
        object.__setattr__(self, "token_ids", tuple(sorted({int(item) for item in token_ids})))
        object.__setattr__(self, "replacement_token_id", int(replacement_token_id))
        object.__setattr__(self, "name", name)
        if not self.token_ids:
            raise ValueError("token_ids must not be empty")

    @property
    def target(self) -> str:
        return "token_types:" + ",".join(map(str, self.token_ids))

    def apply(self, reference: NeuroTokenBatch) -> NeuroTokenBatch:
        ids = np.asarray(reference.token_ids).copy()
        selected = np.isin(ids, np.asarray(self.token_ids, dtype=ids.dtype))
        ids[selected] = self.replacement_token_id
        return _copy_batch(
            reference,
            token_ids=ids,
            metadata_update={
                "mechint_intervention": self.name,
                "mechint_selected_tokens": int(selected.sum()),
            },
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "replacement_token_id": self.replacement_token_id,
        }


@dataclass(frozen=True, slots=True)
class SideFeatureAblation:
    """Zero one ORION token side-feature without changing token identity."""

    feature: str
    name: str = "orion_side_feature_ablation"

    @property
    def target(self) -> str:
        return f"side_feature:{self.feature}"

    def apply(self, reference: NeuroTokenBatch) -> NeuroTokenBatch:
        if self.feature not in reference.side_features:
            raise KeyError(f"unknown side feature: {self.feature!r}")
        side_features = {
            key: np.asarray(value).copy()
            for key, value in reference.side_features.items()
        }
        side_features[self.feature] = np.zeros_like(side_features[self.feature])
        return _copy_batch(
            reference,
            side_features=side_features,
            metadata_update={"mechint_intervention": self.name},
        )

    def metadata(self) -> Mapping[str, Any]:
        return {"feature": self.feature}


@dataclass(frozen=True, slots=True)
class TokenTimeWindowShuffle:
    """Matched control that shuffles token content within a time window."""

    start_ns: int
    end_ns: int
    seed: int = 0
    name: str = "orion_token_time_window_shuffle"

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")

    @property
    def target(self) -> str:
        return f"tokens[{self.start_ns}:{self.end_ns}]"

    def apply(self, reference: NeuroTokenBatch) -> NeuroTokenBatch:
        timestamps = np.asarray(reference.timestamps_ns)
        selected_indices = np.flatnonzero(
            (timestamps >= self.start_ns) & (timestamps < self.end_ns)
        )
        ids = np.asarray(reference.token_ids).copy()
        if len(selected_indices) > 1:
            rng = np.random.default_rng(self.seed)
            permutation = rng.permutation(len(selected_indices))
            source_indices = selected_indices[permutation]
            original_ids = ids.copy()
            ids[selected_indices] = original_ids[source_indices]

            side_features = {}
            for key, value in reference.side_features.items():
                array = np.asarray(value).copy()
                if array.ndim >= 1 and array.shape[0] == len(ids):
                    original = array.copy()
                    array[selected_indices] = original[source_indices]
                side_features[key] = array
        else:
            side_features = {
                key: np.asarray(value).copy()
                for key, value in reference.side_features.items()
            }
        return _copy_batch(
            reference,
            token_ids=ids,
            side_features=side_features,
            metadata_update={
                "mechint_intervention": self.name,
                "mechint_selected_tokens": len(selected_indices),
            },
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class RepresentationTimeWindowAblation:
    """Ablate continuous ORION representations inside a temporal window."""

    start_ns: int
    end_ns: int
    mode: str = "zero"
    name: str = "orion_representation_time_window_ablation"

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")
        if self.mode not in {"zero", "mean"}:
            raise ValueError("mode must be 'zero' or 'mean'")

    @property
    def target(self) -> str:
        return f"representation[{self.start_ns}:{self.end_ns}]"

    def apply(self, reference: RepresentationBatch) -> RepresentationBatch:
        values = np.asarray(reference.values).copy()
        timestamps = np.asarray(reference.timestamps_ns)
        selected = (timestamps >= self.start_ns) & (timestamps < self.end_ns)
        if self.mode == "zero":
            values[selected] = 0
        elif selected.any():
            donor = values[~selected]
            baseline = (donor if len(donor) else values).mean(axis=0)
            values[selected] = baseline
        return _copy_representation(
            reference,
            values=values,
            metadata_update={
                "mechint_intervention": self.name,
                "mechint_selected_steps": int(selected.sum()),
                "mechint_ablation_mode": self.mode,
            },
        )

    def metadata(self) -> Mapping[str, Any]:
        return {"start_ns": self.start_ns, "end_ns": self.end_ns, "mode": self.mode}


@dataclass(frozen=True, slots=True)
class RepresentationFeatureAblation:
    """Ablate selected feature dimensions in a [time, features] representation."""

    feature_indices: tuple[int, ...]
    mode: str = "zero"
    name: str = "orion_representation_feature_ablation"

    def __init__(
        self,
        feature_indices: Sequence[int],
        mode: str = "zero",
        name: str = "orion_representation_feature_ablation",
    ) -> None:
        indices = tuple(sorted({int(item) for item in feature_indices}))
        if not indices:
            raise ValueError("feature_indices must not be empty")
        if mode not in {"zero", "mean"}:
            raise ValueError("mode must be 'zero' or 'mean'")
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "name", name)

    @property
    def target(self) -> str:
        return "representation_features:" + ",".join(map(str, self.feature_indices))

    def apply(self, reference: RepresentationBatch) -> RepresentationBatch:
        values = np.asarray(reference.values).copy()
        if values.ndim != 2:
            raise ValueError("feature ablation currently expects [time, features]")
        indices = np.asarray(self.feature_indices, dtype=np.int64)
        if (indices < 0).any() or (indices >= values.shape[1]).any():
            raise IndexError("feature index is outside the representation dimension")
        if self.mode == "zero":
            values[:, indices] = 0
        else:
            values[:, indices] = values[:, indices].mean(axis=0)
        return _copy_representation(
            reference,
            values=values,
            metadata_update={
                "mechint_intervention": self.name,
                "mechint_feature_indices": list(self.feature_indices),
                "mechint_ablation_mode": self.mode,
            },
        )

    def metadata(self) -> Mapping[str, Any]:
        return {"feature_indices": list(self.feature_indices), "mode": self.mode}


@dataclass(frozen=True, slots=True)
class RepresentationTimeWindowShuffle:
    """Matched control that shuffles latent vectors within a temporal window."""

    start_ns: int
    end_ns: int
    seed: int = 0
    name: str = "orion_representation_time_window_shuffle"

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")

    @property
    def target(self) -> str:
        return f"representation[{self.start_ns}:{self.end_ns}]"

    def apply(self, reference: RepresentationBatch) -> RepresentationBatch:
        timestamps = np.asarray(reference.timestamps_ns)
        selected_indices = np.flatnonzero(
            (timestamps >= self.start_ns) & (timestamps < self.end_ns)
        )
        values = np.asarray(reference.values).copy()
        if len(selected_indices) > 1:
            rng = np.random.default_rng(self.seed)
            source_indices = selected_indices[rng.permutation(len(selected_indices))]
            original = values.copy()
            values[selected_indices] = original[source_indices]
        return _copy_representation(
            reference,
            values=values,
            metadata_update={
                "mechint_intervention": self.name,
                "mechint_selected_steps": len(selected_indices),
            },
        )

    def metadata(self) -> Mapping[str, Any]:
        return {"start_ns": self.start_ns, "end_ns": self.end_ns, "seed": self.seed}


def build_temporal_window_interventions(
    batch: NeuroTokenBatch,
    *,
    window_ns: int,
    stride_ns: int | None = None,
    replacement_token_id: int = 0,
) -> tuple[TokenTimeWindowMask, ...]:
    """Cover a token batch with deterministic half-open temporal windows."""

    if window_ns <= 0:
        raise ValueError("window_ns must be positive")
    stride = window_ns if stride_ns is None else stride_ns
    if stride <= 0:
        raise ValueError("stride_ns must be positive")
    timestamps = np.asarray(batch.timestamps_ns, dtype=np.int64)
    if len(timestamps) == 0:
        return ()
    start = int(timestamps.min())
    stop = int(timestamps.max()) + 1
    return tuple(
        TokenTimeWindowMask(
            start_ns=left,
            end_ns=left + window_ns,
            replacement_token_id=replacement_token_id,
        )
        for left in range(start, stop, stride)
    )


def build_representation_window_interventions(
    batch: RepresentationBatch,
    *,
    window_ns: int,
    stride_ns: int | None = None,
    mode: str = "zero",
) -> tuple[RepresentationTimeWindowAblation, ...]:
    """Cover a representation batch with deterministic half-open time windows."""

    if window_ns <= 0:
        raise ValueError("window_ns must be positive")
    stride = window_ns if stride_ns is None else stride_ns
    if stride <= 0:
        raise ValueError("stride_ns must be positive")
    timestamps = np.asarray(batch.timestamps_ns, dtype=np.int64)
    if len(timestamps) == 0:
        return ()
    start = int(timestamps.min())
    stop = int(timestamps.max()) + 1
    return tuple(
        RepresentationTimeWindowAblation(
            start_ns=left,
            end_ns=left + window_ns,
            mode=mode,
        )
        for left in range(start, stop, stride)
    )


def temporal_window_audit(
    batch: NeuroTokenBatch,
    scorer: Callable[[NeuroTokenBatch], float],
    *,
    window_ns: int,
    stride_ns: int | None = None,
    replacement_token_id: int = 0,
    model_id: str = "orion-downstream-scorer",
    dataset_id: str = "in_memory",
    experiment_name: str = "orion-temporal-window-audit",
    seed: int = 0,
    evidence_tier: EvidenceTier = EvidenceTier.UNIT,
    include_shuffle_controls: bool = True,
) -> InputExperimentResult:
    """Run a temporal necessity sweep over an ORION token batch."""

    interventions = build_temporal_window_interventions(
        batch,
        window_ns=window_ns,
        stride_ns=stride_ns,
        replacement_token_id=replacement_token_id,
    )
    controls: Iterable[TokenTimeWindowShuffle] = ()
    if include_shuffle_controls:
        controls = tuple(
            TokenTimeWindowShuffle(
                start_ns=item.start_ns,
                end_ns=item.end_ns,
                seed=seed + index,
            )
            for index, item in enumerate(interventions)
        )
    experiment = InputCausalExperiment(
        reference=batch,
        metric=InputMetric(scorer, name="orion_downstream_score"),
        experiment_name=experiment_name,
        model_id=model_id,
        dataset_id=dataset_id,
        seed=seed,
        evidence_tier=evidence_tier,
        metadata={
            "window_ns": window_ns,
            "stride_ns": stride_ns,
            "replacement_token_id": replacement_token_id,
        },
    )
    return experiment.run(interventions, controls=controls)


def representation_window_audit(
    batch: RepresentationBatch,
    scorer: Callable[[RepresentationBatch], float],
    *,
    window_ns: int,
    stride_ns: int | None = None,
    mode: str = "zero",
    model_id: str = "orion-representation-scorer",
    dataset_id: str = "in_memory",
    experiment_name: str = "orion-representation-window-audit",
    seed: int = 0,
    evidence_tier: EvidenceTier = EvidenceTier.UNIT,
    include_shuffle_controls: bool = True,
) -> InputExperimentResult:
    """Run a temporal necessity sweep over continuous ORION representations."""

    interventions = build_representation_window_interventions(
        batch,
        window_ns=window_ns,
        stride_ns=stride_ns,
        mode=mode,
    )
    controls: Iterable[RepresentationTimeWindowShuffle] = ()
    if include_shuffle_controls:
        controls = tuple(
            RepresentationTimeWindowShuffle(
                start_ns=item.start_ns,
                end_ns=item.end_ns,
                seed=seed + index,
            )
            for index, item in enumerate(interventions)
        )
    experiment = InputCausalExperiment(
        reference=batch,
        metric=InputMetric(scorer, name="orion_representation_downstream_score"),
        experiment_name=experiment_name,
        model_id=model_id,
        dataset_id=dataset_id,
        seed=seed,
        evidence_tier=evidence_tier,
        metadata={"window_ns": window_ns, "stride_ns": stride_ns, "mode": mode},
    )
    return experiment.run(interventions, controls=controls)
