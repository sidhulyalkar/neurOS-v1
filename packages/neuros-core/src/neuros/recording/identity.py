"""Source wrappers for canonical runtime stream identity."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, AsyncIterator

from neuros.contracts import SignalFrame, StreamDescriptor


class StreamIdentitySource:
    """Expose a source under the stream ID chosen by runtime configuration.

    Hardware drivers may have a device-native or class-derived stream ID. Once a
    source is bound to a configured RuntimeGraph stream, that configured ID is
    the canonical identity used for recording, replay, fusion, and provenance.
    The original driver-provided ID is retained in metadata.
    """

    def __init__(self, source: Any, stream_id: str) -> None:
        if not stream_id:
            raise ValueError("stream_id must be non-empty")
        self.source = source
        self.stream_id = stream_id

    @property
    def descriptor(self) -> StreamDescriptor:
        descriptor = self.source.descriptor
        if descriptor.stream_id == self.stream_id:
            return descriptor
        return replace(
            descriptor,
            stream_id=self.stream_id,
            metadata={
                **dict(descriptor.metadata),
                "source_stream_id": descriptor.stream_id,
            },
        )

    async def start(self) -> None:
        await self.source.start()

    async def stop(self) -> None:
        await self.source.stop()

    async def frames(self) -> AsyncIterator[SignalFrame]:
        async for frame in self.source.frames():
            if frame.stream_id == self.stream_id:
                yield frame
                continue
            yield replace(
                frame,
                stream_id=self.stream_id,
                metadata={
                    **dict(frame.metadata),
                    "source_stream_id": frame.stream_id,
                },
            )
