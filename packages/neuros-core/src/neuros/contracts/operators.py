"""Protocols for runtime operators."""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from .models import DecoderOutput
from .signal import SignalFrame, StreamDescriptor


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
