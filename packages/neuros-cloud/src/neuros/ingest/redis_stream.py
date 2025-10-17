"""
Redis Streams producer for edge ingestion.

This module implements a simple wrapper around Redis Streams for
publishing sensor events.  Redis Streams provide an append‑only log
with at‑least‑once semantics, making them suitable for edge devices
where local durability is required before forwarding to the central
Kafka cluster.

See the Redis Streams documentation for details:
https://redis.io/docs/data-types/streams/
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import redis  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class RedisEvent:
    """A simplified event for Redis Streams."""

    subject_id: str
    session_id: str
    device_id: str
    modality: str
    timestamp_ns: int
    seq: int
    payload: Any
    meta: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "device_id": self.device_id,
            "modality": self.modality,
            "timestamp_ns": str(self.timestamp_ns),  # Redis stores values as strings
            "seq": str(self.seq),
            "payload": json.dumps(self.payload),
            "meta": json.dumps(self.meta or {}),
        }


class RedisStreamWriter:
    """Write events to a Redis Stream.

    Parameters
    ----------
    host:
        Redis server hostname.
    port:
        Redis server port.
    stream_name:
        Name of the Redis Stream to append to.
    maxlen:
        Optional maximum length for the stream; older entries are trimmed.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, stream_name: str = "sensor", maxlen: int | None = None) -> None:
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.stream_name = stream_name
        self.maxlen = maxlen
        logger.info("Connected to Redis at %s:%d; stream=%s", host, port, stream_name)

    def write(self, event: RedisEvent) -> str:
        """Append an event to the Redis Stream.

        Returns the ID assigned by Redis (time-seq)."""
        entry_id = self.client.xadd(
            self.stream_name,
            event.to_dict(),
            maxlen=self.maxlen,
            approximate=True if self.maxlen else False,
        )
        return entry_id


def demo_edge_publishing(host: str = "localhost", port: int = 6379, duration: float = 5.0) -> None:
    """Demonstrate publishing of dummy events to Redis Streams.

    This function generates random events similar to those in the ZeroMQ
    example and writes them to a Redis stream called "sensor".
    """
    writer = RedisStreamWriter(host=host, port=port, stream_name="sensor")
    seq = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        timestamp_ns = int(time.time_ns())
        payload = {"value": seq}
        evt = RedisEvent(
            subject_id="sub0",
            session_id="sess0",
            device_id="dev0",
            modality="eeg",
            timestamp_ns=timestamp_ns,
            seq=seq,
            payload=payload,
        )
        entry_id = writer.write(evt)
        logger.debug("Wrote event %s to Redis stream sensor", entry_id)
        seq += 1
        time.sleep(0.01)
    logger.info("Finished writing %d events to Redis", seq)


__all__ = ["RedisEvent", "RedisStreamWriter", "demo_edge_publishing"]