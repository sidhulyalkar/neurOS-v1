"""
Kafka producer and consumer for neurOS.

This module provides high‑level wrappers around the ``confluent_kafka``
Producer and Consumer classes with sensible defaults for reliability
and throughput.  In production a schema registry should be used to
enforce Protobuf or Avro schemas – this example serialises events as
JSON for simplicity.  See ``neuros/utils/protobuf/brain_event.proto``
for the canonical schema definition.

Usage:

.. code-block:: python

    producer = KafkaWriter(bootstrap_servers="localhost:9092")
    producer.write(topic="raw.eeg.dev0", value=my_event_dict)
    producer.close()

"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from confluent_kafka import Producer, Consumer, KafkaException

logger = logging.getLogger(__name__)


@dataclass
class KafkaEvent:
    """Simplified representation of an event to be sent to Kafka."""

    subject_id: str
    session_id: str
    device_id: str
    modality: str
    timestamp_ns: int
    seq: int
    payload: Any
    meta: Dict[str, Any] | None = None

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__).encode("utf-8")


class KafkaWriter:
    """High‑level Kafka producer wrapper.

    The producer is configured for idempotent writes with acknowledgements
    from all replicas (acks='all') to guarantee at least once delivery.
    Batch sizes and linger settings can be tuned via parameters.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str = "neuros-producer",
        linger_ms: int = 20,
        batch_num_messages: int = 1000,
    ) -> None:
        self.producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "client.id": client_id,
                "enable.idempotence": True,
                "acks": "all",
                "linger.ms": linger_ms,
                "batch.num.messages": batch_num_messages,
            }
        )
        logger.info("Kafka producer initialised for %s", bootstrap_servers)

    def write(self, topic: str, value: Dict[str, Any], key: Optional[str] = None) -> None:
        """Serialize ``value`` as JSON and send it to the given topic."""
        payload = json.dumps(value).encode("utf-8")
        self.producer.produce(topic=topic, value=payload, key=key)
        self.producer.poll(0)

    def flush(self) -> None:
        self.producer.flush()

    def close(self) -> None:
        self.flush()


class KafkaReader:
    """High‑level Kafka consumer wrapper."""

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: Iterable[str],
        auto_offset_reset: str = "earliest",
    ) -> None:
        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": auto_offset_reset,
            }
        )
        self.consumer.subscribe(list(topics))
        logger.info("Kafka consumer subscribed to %s", topics)

    def __iter__(self):  # pragma: no cover
        return self

    def __next__(self) -> Dict[str, Any]:  # pragma: no cover
        msg = self.consumer.poll(1.0)
        if msg is None:
            raise StopIteration
        if msg.error():
            raise KafkaException(msg.error())
        return json.loads(msg.value().decode("utf-8"))

    def close(self) -> None:
        self.consumer.close()


__all__ = ["KafkaEvent", "KafkaWriter", "KafkaReader"]