"""Addressable model components used by tracing and interventions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class ComponentRef:
    """Reference to an addressable component inside a model."""

    path: str
    kind: str = "module"
    index: int | None = None
    position: int | None = None

    @property
    def label(self) -> str:
        parts = [self.path]
        if self.kind != "module":
            parts.append(self.kind)
        if self.index is not None:
            parts.append(str(self.index))
        if self.position is not None:
            parts.append(f"pos={self.position}")
        return ":".join(parts)
