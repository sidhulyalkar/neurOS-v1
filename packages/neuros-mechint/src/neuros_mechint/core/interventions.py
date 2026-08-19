"""Causal interventions for model-component experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .components import ComponentRef


class Intervention(Protocol):
    name: str
    component: ComponentRef

    def replacement(
        self,
        *,
        clean_value: torch.Tensor,
        corrupted_value: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def metadata(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class PatchIntervention:
    """Replace a corrupted module output with its clean counterpart."""

    component: ComponentRef
    name: str = "module_activation_patching"

    def replacement(
        self,
        *,
        clean_value: torch.Tensor,
        corrupted_value: torch.Tensor,
    ) -> torch.Tensor:
        del corrupted_value
        return clean_value

    def metadata(self) -> Mapping[str, Any]:
        return {"ablation": None}


@dataclass(frozen=True, slots=True)
class AblationIntervention:
    """Ablate a module output using zero or mean replacement."""

    component: ComponentRef
    mode: str = "zero"
    name: str = "module_output_ablation"

    def replacement(
        self,
        *,
        clean_value: torch.Tensor,
        corrupted_value: torch.Tensor,
    ) -> torch.Tensor:
        del clean_value
        if self.mode == "zero":
            return torch.zeros_like(corrupted_value)
        if self.mode == "mean":
            return torch.full_like(corrupted_value, corrupted_value.mean())
        raise ValueError(f"unsupported ablation mode: {self.mode!r}")

    def metadata(self) -> Mapping[str, Any]:
        return {"ablation": self.mode}
