"""Apply the generic circuit-faithfulness benchmark to SAE feature subsets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from neuros_mechint.adapters.sae_lens import SAELensFeatureAdapter

from .faithfulness import (
    CircuitCandidate,
    CircuitFaithfulnessReport,
    FaithfulnessPolicy,
    evaluate_circuit_faithfulness,
)


def sae_feature_name(index: int) -> str:
    index = int(index)
    if index < 0:
        raise ValueError("SAE feature index must be non-negative")
    return f"sae_feature:{index}"


def _parse_feature_name(value: str) -> int:
    prefix = "sae_feature:"
    if not value.startswith(prefix):
        raise ValueError(f"invalid SAE feature target {value!r}")
    return int(value[len(prefix) :])


def evaluate_sae_feature_faithfulness(
    *,
    adapter: SAELensFeatureAdapter,
    activations: torch.Tensor,
    scorer: Callable[[torch.Tensor], Any],
    target_features: Sequence[int],
    candidate_features: Sequence[int],
    candidate_name: str = "sae-feature-circuit",
    random_trials: int = 100,
    seed: int = 0,
    higher_is_better: bool = True,
    policy: FaithfulnessPolicy | None = None,
) -> CircuitFaithfulnessReport:
    """Score an SAE feature subset relative to the SAE reconstruction baseline.

    The original-versus-reconstruction metric gap is always recorded in report
    metadata. Necessity and sufficiency are normalized within the reconstruction
    model, preventing SAE reconstruction error from being silently attributed to
    the nominated features.
    """

    universe_indices = tuple(dict.fromkeys(int(index) for index in target_features))
    candidate_indices = tuple(dict.fromkeys(int(index) for index in candidate_features))
    universe = tuple(sae_feature_name(index) for index in universe_indices)
    candidate = CircuitCandidate(
        name=candidate_name,
        targets=tuple(sae_feature_name(index) for index in candidate_indices),
        source="sae-feature-intervention",
    )
    reconstruction = adapter.reconstruction_audit(activations, scorer)

    def _subset_metric(retained: tuple[str, ...]) -> float:
        retained_indices = tuple(_parse_feature_name(value) for value in retained)
        return adapter.feature_metric(
            activations,
            scorer,
            target_features=universe_indices,
            retained_features=retained_indices,
        )

    return evaluate_circuit_faithfulness(
        all_targets=universe,
        candidate=candidate,
        subset_metric=_subset_metric,
        random_trials=random_trials,
        seed=seed,
        higher_is_better=higher_is_better,
        policy=policy,
        metadata={
            "adapter": type(adapter).__qualname__,
            "original_metric": reconstruction.original_metric,
            "reconstruction_metric": reconstruction.reconstruction_metric,
            "reconstruction_gap": reconstruction.reconstruction_gap,
            "activation_shape": list(reconstruction.activation_shape),
            "feature_shape": list(reconstruction.feature_shape),
        },
    )
