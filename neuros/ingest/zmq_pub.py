"""
ZeroMQ publisher for local low‑latency fan‑out.

This module implements a simple publisher that emits synthetic sensor
events over a PUB socket.  In a real deployment the ``publish``
function would be called with encoded payloads read from actual
devices.  For demonstration purposes the ``run_zmq_publisher``
function generates a short stream of dummy values tagged with device
metadata and timestamps.

The publisher uses the PUB/SUB pattern, binding to a user‑specified
address (default: ``tcp://127.0.0.1:5555``).  Subscribers can
connect to this address and receive messages without backpressure.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict

import zmq  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class SensorEvent:
    """A simple representation of a sensor event.

    This dataclass does not use Protobuf to avoid requiring code
    generation; instead it holds a minimal set of fields.  For
    production use consider using the Protobuf schema defined under
    ``neuros/utils/protobuf/brain_event.proto``.
    """

    subject_id: str
    session_id: str
    device_id: str
    modality: str
    timestamp_ns: int
    seq: int
    payload: Any
    meta: Dict[str, Any] | None = None

    def to_json(self) -> bytes:
        """Serialise the event to a JSON bytes buffer."""
        return json.dumps(self.__dict__).encode("utf-8")


class ZMQPublisher:
    """Publish sensor events over ZeroMQ PUB socket."""

    def __init__(self, bind_address: str = "tcp://127.0.0.1:5555") -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(bind_address)
        logger.info("ZMQ publisher bound to %s", bind_address)

    def publish(self, event: SensorEvent) -> None:
        """Publish a single event to all subscribers."""
        self.socket.send(event.to_json())

    def close(self) -> None:
        self.socket.close()
        self.context.term()


def run_zmq_publisher(
    modality: str,
    device: str,
    subject: str,
    duration: float = 5.0,
    bind_address: str = "tcp://127.0.0.1:5555",
) -> None:
    """Generate and publish dummy sensor events for demonstration.

    Parameters
    ----------
    modality:
        The modality of the dummy sensor (e.g. ``eeg`` or ``fnirs``).
    device:
        Device identifier.
    subject:
        Subject identifier.
    duration:
        How long to publish data for, in seconds.
    bind_address:
        Address on which to bind the PUB socket.
    """
    pub = ZMQPublisher(bind_address=bind_address)
    seq = 0
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < duration:
            timestamp_ns = int(time.time_ns())
            # Generate a random floating point payload (e.g. sensor reading)
            payload = random.random()
            event = SensorEvent(
                subject_id=subject,
                session_id="sess1",
                device_id=device,
                modality=modality,
                timestamp_ns=timestamp_ns,
                seq=seq,
                payload=payload,
            )
            pub.publish(event)
            seq += 1
            # 100 Hz publishing rate (10 ms)
            time.sleep(0.01)
    finally:
        pub.close()
        logger.info("Finished publishing %d events for device %s", seq, device)


__all__ = ["SensorEvent", "ZMQPublisher", "run_zmq_publisher"]