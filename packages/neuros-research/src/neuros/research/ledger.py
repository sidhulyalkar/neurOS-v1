"""Tamper-evident append-only experiment ledger."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ._canonical import canonical_json, canonical_sha256, freeze_json, require_nonempty, thaw_json

LedgerEventType = Literal[
    "packet_registered",
    "evidence_attached",
    "decision_attached",
    "insight_published",
]
GENESIS_HASH = "0" * 64
_ALLOWED_EVENT_TYPES = {"packet_registered", "evidence_attached", "decision_attached", "insight_published"}


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    index: int
    event_type: LedgerEventType
    experiment_id: str
    payload: Mapping[str, Any]
    previous_hash: str
    event_hash: str

    @classmethod
    def create(
        cls,
        *,
        index: int,
        event_type: LedgerEventType,
        experiment_id: str,
        payload: Mapping[str, Any],
        previous_hash: str,
    ) -> LedgerEvent:
        if index < 0:
            raise ValueError("ledger index must be non-negative")
        if event_type not in _ALLOWED_EVENT_TYPES:
            raise ValueError(f"unsupported ledger event_type {event_type!r}")
        experiment_id = require_nonempty(experiment_id, name="experiment_id")
        if len(previous_hash) != 64 or any(ch not in "0123456789abcdef" for ch in previous_hash):
            raise ValueError("previous_hash must be a full hexadecimal SHA-256")
        frozen_payload = freeze_json(payload, path="ledger.payload")
        unsigned = {
            "index": index,
            "event_type": event_type,
            "experiment_id": experiment_id,
            "payload": thaw_json(frozen_payload),
            "previous_hash": previous_hash,
        }
        event_hash = canonical_sha256(unsigned)
        return cls(
            index=index,
            event_type=event_type,
            experiment_id=experiment_id,
            payload=frozen_payload,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "event_type": self.event_type,
            "experiment_id": self.experiment_id,
            "payload": thaw_json(self.payload),
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "event_hash": self.event_hash}

    def verify(self, *, expected_index: int, expected_previous_hash: str) -> None:
        if self.index != expected_index:
            raise ValueError(
                f"ledger index mismatch: expected {expected_index}, observed {self.index}"
            )
        if self.previous_hash != expected_previous_hash:
            raise ValueError(f"ledger previous_hash mismatch at index {self.index}")
        expected_hash = canonical_sha256(self.unsigned_dict())
        if self.event_hash != expected_hash:
            raise ValueError(f"ledger event hash mismatch at index {self.index}")


class EvidenceLedger:
    """In-memory append-only hash chain with deterministic JSONL interchange."""

    def __init__(self, events: Iterable[LedgerEvent] = ()) -> None:
        self._events: list[LedgerEvent] = list(events)
        self.verify()

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._events)

    @property
    def head_hash(self) -> str:
        return self._events[-1].event_hash if self._events else GENESIS_HASH

    def append(
        self,
        event_type: LedgerEventType,
        experiment_id: str,
        payload: Mapping[str, Any],
    ) -> LedgerEvent:
        event = LedgerEvent.create(
            index=len(self._events),
            event_type=event_type,
            experiment_id=experiment_id,
            payload=payload,
            previous_hash=self.head_hash,
        )
        self._events.append(event)
        return event

    def verify(self) -> None:
        previous_hash = GENESIS_HASH
        for index, event in enumerate(self._events):
            event.verify(expected_index=index, expected_previous_hash=previous_hash)
            previous_hash = event.event_hash

    def to_jsonl(self) -> str:
        return "\n".join(canonical_json(event.to_dict()) for event in self._events) + (
            "\n" if self._events else ""
        )

    @classmethod
    def from_jsonl(cls, raw: str) -> EvidenceLedger:
        events: list[LedgerEvent] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                event_type = payload["event_type"]
                if event_type not in _ALLOWED_EVENT_TYPES:
                    raise ValueError(f"unsupported ledger event_type {event_type!r}")
                event = LedgerEvent(
                    index=int(payload["index"]),
                    event_type=event_type,
                    experiment_id=str(payload["experiment_id"]),
                    payload=freeze_json(payload["payload"], path=f"ledger[{line_number}].payload"),
                    previous_hash=str(payload["previous_hash"]),
                    event_hash=str(payload["event_hash"]),
                )
            except KeyError as exc:
                raise ValueError(f"ledger line {line_number} missing field {exc.args[0]!r}") from exc
            events.append(event)
        return cls(events)
