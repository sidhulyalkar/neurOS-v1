"""SAELens-compatible feature adapter with explicit reconstruction accounting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class SAEReconstructionAudit:
    """Quantify the SAE reconstruction confound before feature interventions."""

    original_metric: float
    reconstruction_metric: float
    reconstruction_gap: float
    activation_shape: tuple[int, ...]
    feature_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scalar_score(scorer: Callable[[torch.Tensor], Any], value: torch.Tensor) -> float:
    score = scorer(value)
    if isinstance(score, torch.Tensor):
        if score.numel() != 1:
            raise ValueError("SAE scorer must return a scalar")
        score = score.detach().cpu().item()
    elif isinstance(score, np.ndarray):
        if score.size != 1:
            raise ValueError("SAE scorer must return a scalar")
        score = score.item()
    return float(score)


class SAELensFeatureAdapter:
    """Use an SAE through the stable ``encode``/``decode`` feature interface.

    The implementation is duck-typed and does not import SAELens. This mirrors
    current SAELens inference guidance, where pretrained SAEs can encode and
    decode arbitrary PyTorch activations. Scientific feature interventions are
    evaluated relative to the SAE reconstruction, not silently against the
    original activation, so reconstruction error remains visible.
    """

    def __init__(self, sae: Any) -> None:
        for method in ("encode", "decode"):
            if not callable(getattr(sae, method, None)):
                raise TypeError(f"SAE must expose callable {method}()")
        self.sae = sae

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        features = self.sae.encode(activations)
        if not isinstance(features, torch.Tensor):
            raise TypeError("SAE encode() must return a tensor")
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        activations = self.sae.decode(features)
        if not isinstance(activations, torch.Tensor):
            raise TypeError("SAE decode() must return a tensor")
        return activations

    def reconstruct(self, activations: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(activations))

    def reconstruction_audit(
        self,
        activations: torch.Tensor,
        scorer: Callable[[torch.Tensor], Any],
    ) -> SAEReconstructionAudit:
        features = self.encode(activations)
        reconstructed = self.decode(features)
        original_metric = _scalar_score(scorer, activations)
        reconstruction_metric = _scalar_score(scorer, reconstructed)
        return SAEReconstructionAudit(
            original_metric=original_metric,
            reconstruction_metric=reconstruction_metric,
            reconstruction_gap=reconstruction_metric - original_metric,
            activation_shape=tuple(activations.shape),
            feature_shape=tuple(features.shape),
        )

    def reconstruct_with_feature_subset(
        self,
        activations: torch.Tensor,
        *,
        target_features: Sequence[int],
        retained_features: Sequence[int],
        baseline: str = "zero",
    ) -> torch.Tensor:
        """Decode after ablating target features not included in ``retained_features``.

        Features outside ``target_features`` are left untouched. Thus an empty
        retained set means "remove the audited feature universe", not "zero the
        entire SAE dictionary". This makes the all-vs-null span explicit for a
        nominated feature set.
        """

        if baseline != "zero":
            raise ValueError("SAE feature subset interventions currently support baseline='zero'")
        codes = self.encode(activations).clone()
        if codes.ndim == 0:
            raise ValueError("SAE feature codes must have a feature dimension")
        width = codes.shape[-1]
        universe = tuple(dict.fromkeys(int(index) for index in target_features))
        retained = {int(index) for index in retained_features}
        invalid = [index for index in universe if index < 0 or index >= width]
        if invalid:
            raise IndexError(f"SAE feature indices outside width {width}: {invalid}")
        if not retained.issubset(set(universe)):
            raise ValueError("retained_features must be a subset of target_features")
        removed = [index for index in universe if index not in retained]
        if removed:
            codes[..., removed] = 0
        return self.decode(codes)

    def feature_metric(
        self,
        activations: torch.Tensor,
        scorer: Callable[[torch.Tensor], Any],
        *,
        target_features: Sequence[int],
        retained_features: Sequence[int],
    ) -> float:
        reconstructed = self.reconstruct_with_feature_subset(
            activations,
            target_features=target_features,
            retained_features=retained_features,
        )
        return _scalar_score(scorer, reconstructed)
