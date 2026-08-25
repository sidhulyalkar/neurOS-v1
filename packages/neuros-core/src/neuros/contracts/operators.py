"""Protocols and explicit emission contracts for runtime operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Protocol, runtime_checkable

from .models import DecoderOutput
from .signal import SignalFrame, StreamDescriptor


@dataclass(frozen=True, slots=True)
class TransformEmission:
    """Explicit fan-out from one transform invocation.

    Returning a plain list/tuple from a transform is ambiguous because arrays,
    feature collections, and structured model inputs may themselves be
    sequences. ``TransformEmission`` is therefore the only kernel-level signal
    that one transform input should produce multiple downstream items.
    """

    items: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("TransformEmission must contain at least one item")

    @classmethod
    def from_iterable(cls, items: Iterable[Any]) -> "TransformEmission":
        return cls(tuple(items))


@runtime_checkable
class Source(Protocol):
    @property
    def descriptor(self) -> StreamDescriptor:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def frames(self) -> AsyncIterator[SignalFrame]:
        ...


@runtime_checkable
class Transform(Protocol):
    def transform(self, item: Any) -> Any:
        ...


@runtime_checkable
class Sink(Protocol):
    async def write(self, item: Any) -> None:
        ...


@runtime_checkable
class Monitor(Protocol):
    def update(self, item: Any) -> None:
        ...

    def result(self) -> dict[str, Any]:
        ...


@runtime_checkable
class OutputSubscriber(Protocol):
    async def on_output(self, output: DecoderOutput) -> None:
        ...
