"""Model-adapter contract used by mechanistic experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch


class ModelAdapter(ABC):
    """Minimal tracing/intervention interface independent of model framework."""

    @abstractmethod
    def forward(self, inputs: Any) -> torch.Tensor:
        ...

    @abstractmethod
    def capture_outputs(self, inputs: Any, paths: Sequence[str]) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def forward_with_replacements(
        self,
        inputs: Any,
        replacements: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        ...

    def model_fingerprint_payload(self) -> Any | None:
        """Return deterministic model state suitable for full-content hashing."""

        return None
