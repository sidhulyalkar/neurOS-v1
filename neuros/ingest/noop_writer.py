"""No‑operation writer for local testing without Kafka.

This module defines a minimal writer with the same interface as
``KafkaWriter`` but which simply discards events.  Use this class
when running the Constellation pipeline in a "dry‑run" mode
without a backing Kafka broker.  Events will be counted and
available via the ``published_count`` attribute for simple
verification.

Example
-------

    from neuros.ingest.noop_writer import NoopWriter

    writer = NoopWriter()
    writer.write("topic", {"foo": "bar"})
    assert writer.published_count == 1

"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class NoopWriter:
    """Drop‑in replacement for ``KafkaWriter`` that does nothing.

    This class implements a ``write`` method that accepts a topic
    string and a payload dictionary.  The payload is not sent
    anywhere, but a counter of published messages is incremented.
    The class can be used as a context manager but has no real
    external resources to close.

    Attributes
    ----------
    published_count : int
        Total number of messages "sent" through this writer.
    """

    def __init__(self) -> None:
        self.published_count = 0

    def write(self, topic: str, payload: Dict[str, Any]) -> None:
        """Pretend to publish a message.

        Parameters
        ----------
        topic : str
            Name of the topic (ignored).
        payload : dict
            Message payload (ignored).
        """
        self.published_count += 1
        logger.debug("NoopWriter: discarded message for topic %s", topic)

    def close(self) -> None:
        """No‑op close method for compatibility with KafkaWriter."""
        logger.debug("NoopWriter: close called (nothing to do)")

    def __enter__(self) -> "NoopWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()