"""Scalar metrics used to define mechanistic experiment effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


class ScalarMetric(Protocol):
    name: str

    def __call__(self, output: torch.Tensor) -> float:
        ...


@dataclass(frozen=True, slots=True)
class OutputMetric:
    """Wrap a callable that maps a model output to a scalar score."""

    fn: Callable[[torch.Tensor], torch.Tensor | float]
    name: str = "output_metric"

    def __call__(self, output: torch.Tensor) -> float:
        value = self.fn(output)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"metric {self.name!r} must return a scalar, got shape {tuple(value.shape)}"
                )
            value = value.detach().cpu().item()
        return float(value)


def logit_difference(positive_index: int, negative_index: int) -> OutputMetric:
    """Create a mean logit-difference metric over the batch."""

    def _metric(output: torch.Tensor) -> torch.Tensor:
        if output.ndim < 2:
            raise ValueError("logit_difference expects output with a class/logit dimension")
        return (output[..., positive_index] - output[..., negative_index]).mean()

    return OutputMetric(
        _metric,
        name=f"logit_difference[{positive_index}-{negative_index}]",
    )
