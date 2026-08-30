"""Atomic pre-invocation transport authority for promoted Kumar2024 execution.

FleetAuthority defines which immutable shard attempt may run.  This module owns a
narrower transport invariant required before an external scheduler is allowed to
launch that attempt:

    exactly one transport owner may acquire an attempt, and exactly one durable
    invocation marker may grant permission to start it.

The store contract is intentionally tiny.  A production backend must implement
``create_if_absent`` with a real atomic conditional create primitive.  The local
POSIX implementation is provided only for deterministic qualification and
single/shared-filesystem integrations; it is not evidence that an arbitrary
network filesystem or object store has the same atomic semantics.

A durable invocation marker is written *before* process launch.  If a scheduler
loses ownership after that marker exists, this module fails closed: replaying the
same marker does not grant a second launch.  Recovery/adjudication of an orphaned
invocation belongs to the later trusted-outcome transport layer and must prove
that retry cannot duplicate scientific execution.

Nothing here runs a model, reads neural data, scores a result, or permits ORION
comparison.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from ._kumar2024_promoted_fleet_common import (
    PromotedShardLease,
    _identity_sha256,
    _require_serialized_payload,
    _sha256,
)


class TransportClaimConflict(RuntimeError):
    """A different transport owner or payload already owns an immutable key."""


class TransportClaimStateError(RuntimeError):
    """Persisted transport state is absent, malformed, or inconsistent."""


def _owner_token_sha256(owner_token: bytes) -> str:
    if not isinstance(owner_token, bytes):
        raise TypeError("transport owner_token must be bytes")
    if len(owner_token) < 32:
        raise ValueError("transport owner_token must contain at least 32 bytes of entropy")
    return hashlib.sha256(
        b"neuros.kumar2024_promoted_transport_owner.v1\0" + owner_token
    ).hexdigest()


def _store_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("transport store key must be a non-empty string")
    text = value.strip()
    if "\\" in text:
        raise ValueError("transport store key must use POSIX separators")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("transport store key must be canonical and traversal-free")
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class PromotedTransportClaim:
    """Durable ownership claim for one immutable fleet lease."""

    lease: PromotedShardLease
    owner_token_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedTransportClaim schema_version must be 1")
        if not isinstance(self.lease, PromotedShardLease):
            raise TypeError("transport claim lease must be PromotedShardLease")
        object.__setattr__(
            self,
            "owner_token_sha256",
            _sha256("owner_token_sha256", self.owner_token_sha256),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lease": self.lease.to_dict(),
            "owner_token_sha256": self.owner_token_sha256,
            "claim_semantics": (
                "only the owner whose secret token hashes to owner_token_sha256 may "
                "attempt to create the invocation marker; claim replay never grants "
                "a second scientific invocation"
            ),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_transport_claim.v1", self.to_dict()
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotedTransportClaim":
        raw_lease = payload.get("lease")
        if not isinstance(raw_lease, Mapping):
            raise ValueError("serialized transport claim requires lease")
        value = cls(
            lease=PromotedShardLease.from_dict(raw_lease),
            owner_token_sha256=payload["owner_token_sha256"],
            schema_version=payload.get("schema_version", 1),
        )
        _require_serialized_payload(
            payload,
            value.to_dict(),
            object_name="transport claim",
            digest_field="transport_claim_sha256",
            digest=value.sha256,
        )
        return value


@dataclass(frozen=True, slots=True)
class PromotedInvocationMarker:
    """Write-once marker whose fresh creation is the sole launch permission."""

    transport_claim_sha256: str
    lease_sha256: str
    owner_token_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedInvocationMarker schema_version must be 1")
        for name in (
            "transport_claim_sha256",
            "lease_sha256",
            "owner_token_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "transport_claim_sha256": self.transport_claim_sha256,
            "lease_sha256": self.lease_sha256,
            "owner_token_sha256": self.owner_token_sha256,
            "launch_semantics": (
                "only a newly-created invocation marker grants permission to launch; "
                "an existing identical marker is replay evidence and grants no launch"
            ),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_transport_invocation.v1", self.to_dict()
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotedInvocationMarker":
        value = cls(
            transport_claim_sha256=payload["transport_claim_sha256"],
            lease_sha256=payload["lease_sha256"],
            owner_token_sha256=payload["owner_token_sha256"],
            schema_version=payload.get("schema_version", 1),
        )
        _require_serialized_payload(
            payload,
            value.to_dict(),
            object_name="transport invocation marker",
            digest_field="invocation_marker_sha256",
            digest=value.sha256,
        )
        return value


@dataclass(frozen=True, slots=True)
class TransportClaimDecision:
    claim: PromotedTransportClaim
    created: bool

    @property
    def newly_acquired(self) -> bool:
        return self.created

    @property
    def launch_permitted(self) -> bool:
        # Claim acquisition alone is deliberately insufficient to start a worker.
        return False


@dataclass(frozen=True, slots=True)
class InvocationDecision:
    marker: PromotedInvocationMarker
    created: bool

    @property
    def launch_permitted(self) -> bool:
        return self.created


@runtime_checkable
class AtomicCreateStore(Protocol):
    """Trusted store primitive required by the pre-invocation transport layer."""

    def create_if_absent(self, key: str, payload: bytes) -> bool:
        """Atomically create key and return True, or return False if it exists."""
        ...

    def read(self, key: str) -> bytes | None:
        """Return exact persisted bytes, or None when the key is absent."""
        ...


class LocalAtomicCreateStore:
    """POSIX O_EXCL reference store for qualification and local integration.

    Production distributed execution must use a backend whose atomic conditional
    create semantics are independently qualified.  This class must not be used as
    proof that a generic remote/network filesystem provides equivalent behavior.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        canonical = _store_key(key)
        return self.root.joinpath(*canonical.split("/"))

    def create_if_absent(self, key: str, payload: bytes) -> bool:
        if not isinstance(payload, bytes):
            raise TypeError("atomic store payload must be bytes")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:  # pragma: no cover - defensive OS failure guard
                    raise OSError("atomic claim store wrote zero bytes")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return True

    def read(self, key: str) -> bytes | None:
        path = self._path(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None


def transport_claim_key(lease: PromotedShardLease) -> str:
    if not isinstance(lease, PromotedShardLease):
        raise TypeError("lease must be PromotedShardLease")
    return _store_key(f"{lease.artifact_key}/transport/claim.json")


def transport_invocation_key(claim: PromotedTransportClaim) -> str:
    if not isinstance(claim, PromotedTransportClaim):
        raise TypeError("claim must be PromotedTransportClaim")
    return _store_key(f"{claim.lease.artifact_key}/transport/invocation.json")


def _serialize_claim(claim: PromotedTransportClaim) -> bytes:
    payload = {**claim.to_dict(), "transport_claim_sha256": claim.sha256}
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _deserialize_claim(payload: bytes) -> PromotedTransportClaim:
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportClaimStateError("persisted transport claim is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise TransportClaimStateError("persisted transport claim must be a JSON object")
    return PromotedTransportClaim.from_dict(raw)


def _serialize_invocation(marker: PromotedInvocationMarker) -> bytes:
    payload = {**marker.to_dict(), "invocation_marker_sha256": marker.sha256}
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _deserialize_invocation(payload: bytes) -> PromotedInvocationMarker:
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportClaimStateError("persisted invocation marker is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise TransportClaimStateError("persisted invocation marker must be a JSON object")
    return PromotedInvocationMarker.from_dict(raw)


def _require_store(store: AtomicCreateStore) -> None:
    if not isinstance(store, AtomicCreateStore):
        raise TypeError("store must implement AtomicCreateStore")


def acquire_attempt_claim(
    store: AtomicCreateStore,
    lease: PromotedShardLease,
    *,
    owner_token: bytes,
) -> TransportClaimDecision:
    """Atomically acquire one lease for one secret transport owner.

    ``created=True`` means this owner created the claim.  A replay by the same
    owner returns ``created=False``; a different owner fails closed.  Claim
    acquisition never grants permission to launch a worker.
    """

    _require_store(store)
    if not isinstance(lease, PromotedShardLease):
        raise TypeError("lease must be PromotedShardLease")
    claim = PromotedTransportClaim(
        lease=lease,
        owner_token_sha256=_owner_token_sha256(owner_token),
    )
    key = transport_claim_key(lease)
    encoded = _serialize_claim(claim)
    if store.create_if_absent(key, encoded):
        return TransportClaimDecision(claim=claim, created=True)

    persisted = store.read(key)
    if persisted is None:
        raise TransportClaimStateError(
            "atomic store reported an existing claim but it cannot be read"
        )
    observed = _deserialize_claim(persisted)
    if observed.lease != lease:
        raise TransportClaimConflict("existing claim names a different immutable lease")
    if observed.owner_token_sha256 != claim.owner_token_sha256:
        raise TransportClaimConflict("immutable lease is already owned by another transport")
    if observed.sha256 != claim.sha256 or persisted != encoded:
        raise TransportClaimConflict("existing transport claim bytes differ from canonical replay")
    return TransportClaimDecision(claim=observed, created=False)


def begin_attempt_invocation(
    store: AtomicCreateStore,
    claim: PromotedTransportClaim,
    *,
    owner_token: bytes,
) -> InvocationDecision:
    """Persist the write-once pre-launch marker.

    Only ``created=True`` grants launch permission.  If the marker already exists,
    even for the same owner, this call returns ``created=False`` and therefore
    forbids a second launch.  That conservative rule intentionally converts
    scheduler-loss recovery into an explicit adjudication problem instead of a
    timeout-based scientific retry.
    """

    _require_store(store)
    if not isinstance(claim, PromotedTransportClaim):
        raise TypeError("claim must be PromotedTransportClaim")
    owner_sha = _owner_token_sha256(owner_token)
    if owner_sha != claim.owner_token_sha256:
        raise TransportClaimConflict("owner token does not control this transport claim")

    claim_key = transport_claim_key(claim.lease)
    persisted_claim = store.read(claim_key)
    if persisted_claim is None:
        raise TransportClaimStateError("transport claim must be durably persisted before invocation")
    observed_claim = _deserialize_claim(persisted_claim)
    if observed_claim != claim or persisted_claim != _serialize_claim(claim):
        raise TransportClaimConflict("persisted transport claim differs from supplied claim")

    marker = PromotedInvocationMarker(
        transport_claim_sha256=claim.sha256,
        lease_sha256=claim.lease.sha256,
        owner_token_sha256=owner_sha,
    )
    key = transport_invocation_key(claim)
    encoded = _serialize_invocation(marker)
    if store.create_if_absent(key, encoded):
        return InvocationDecision(marker=marker, created=True)

    persisted = store.read(key)
    if persisted is None:
        raise TransportClaimStateError(
            "atomic store reported an invocation marker but it cannot be read"
        )
    observed = _deserialize_invocation(persisted)
    if observed != marker or persisted != encoded:
        raise TransportClaimConflict("existing invocation marker differs from canonical replay")
    return InvocationDecision(marker=observed, created=False)


__all__ = [
    "AtomicCreateStore",
    "InvocationDecision",
    "LocalAtomicCreateStore",
    "PromotedInvocationMarker",
    "PromotedTransportClaim",
    "TransportClaimConflict",
    "TransportClaimDecision",
    "TransportClaimStateError",
    "acquire_attempt_claim",
    "begin_attempt_invocation",
    "transport_claim_key",
    "transport_invocation_key",
]
