from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from _kumar2024_fleet_test_helpers import _fleet, _small_plan
from neuros.evidence.kumar2024_promoted_fleet import (
    PromotedShardLease,
    record_infrastructure_failure,
)
from neuros.evidence.kumar2024_promoted_fleet_transport import (
    LocalAtomicCreateStore,
    PromotedTransportClaim,
    TransportClaimConflict,
    TransportClaimStateError,
    acquire_attempt_claim,
    begin_attempt_invocation,
    transport_claim_key,
    transport_invocation_key,
)


class _BlackHoleCreateStore:
    """Adversarial backend that lies that it committed a key."""

    def create_if_absent(self, key: str, payload: bytes) -> bool:
        return True

    def read(self, key: str) -> bytes | None:
        return None


class _DropInvocationCreateStore:
    """Delegate existing state but lie only when publishing invocation."""

    def __init__(self, delegate: LocalAtomicCreateStore) -> None:
        self.delegate = delegate

    def create_if_absent(self, key: str, payload: bytes) -> bool:
        if key.endswith("/transport/invocation.json"):
            return True
        return self.delegate.create_if_absent(key, payload)

    def read(self, key: str) -> bytes | None:
        return self.delegate.read(key)


def _ledger_and_lease():
    _, authority, ledger = _fleet(_small_plan())
    shard_sha = authority.shard_spec_sha256_by_id[0][1]
    return ledger, ledger.next_lease(shard_sha)


def test_claim_acquisition_is_write_once_and_claim_alone_never_launches(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    ledger, lease = _ledger_and_lease()
    owner = b"a" * 32
    first = acquire_attempt_claim(store, ledger, lease, owner_token=owner)
    second = acquire_attempt_claim(store, ledger, lease, owner_token=owner)
    assert first.created is True
    assert first.newly_acquired is True
    assert first.launch_permitted is False
    assert second.created is False
    assert second.claim == first.claim
    assert second.launch_permitted is False


def test_transport_claim_rejects_skipped_or_stale_attempt_index(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    ledger, lease0 = _ledger_and_lease()
    future = PromotedShardLease(
        fleet_authority_sha256=lease0.fleet_authority_sha256,
        execution_plan_sha256=lease0.execution_plan_sha256,
        shard_id=lease0.shard_id,
        shard_spec_sha256=lease0.shard_spec_sha256,
        attempt_index=1,
    )
    with pytest.raises(ValueError, match="exact next FleetAuthority lease"):
        acquire_attempt_claim(store, ledger, future, owner_token=b"a" * 32)

    advanced = record_infrastructure_failure(
        ledger,
        lease0,
        failure_code="runner_unavailable",
    )
    with pytest.raises(ValueError, match="exact next FleetAuthority lease"):
        acquire_attempt_claim(store, advanced, lease0, owner_token=b"a" * 32)


def test_different_transport_owner_cannot_take_existing_lease(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    ledger, lease = _ledger_and_lease()
    acquire_attempt_claim(store, ledger, lease, owner_token=b"a" * 32)
    with pytest.raises(TransportClaimConflict):
        acquire_attempt_claim(store, ledger, lease, owner_token=b"b" * 32)


def test_concurrent_transport_owners_produce_exactly_one_winner(tmp_path):
    ledger, lease = _ledger_and_lease()

    def attempt(index: int) -> bool:
        store = LocalAtomicCreateStore(tmp_path)
        token = bytes([index + 1]) * 32
        try:
            return acquire_attempt_claim(store, ledger, lease, owner_token=token).created
        except TransportClaimConflict:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 7
    persisted = LocalAtomicCreateStore(tmp_path).read(transport_claim_key(lease))
    assert persisted is not None
    payload = json.loads(persisted)
    assert payload["lease"]["lease_sha256"] == lease.sha256
    assert payload["transport_claim_sha256"]


def test_claim_creation_must_be_durably_readable_before_acquisition():
    ledger, lease = _ledger_and_lease()
    with pytest.raises(TransportClaimStateError, match="cannot be read"):
        acquire_attempt_claim(
            _BlackHoleCreateStore(),
            ledger,
            lease,
            owner_token=b"a" * 32,
        )


def test_owner_secret_is_never_persisted(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    ledger, lease = _ledger_and_lease()
    owner = b"this-owner-token-is-secret-32bytes!!"
    assert len(owner) >= 32
    decision = acquire_attempt_claim(store, ledger, lease, owner_token=owner)
    persisted = store.read(transport_claim_key(lease))
    assert persisted is not None
    assert owner not in persisted
    assert decision.claim.owner_token_sha256.encode("ascii") in persisted


def test_invocation_marker_is_the_only_one_time_launch_permission(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    ledger, lease = _ledger_and_lease()
    owner = b"a" * 32
    claim = acquire_attempt_claim(store, ledger, lease, owner_token=owner).claim
    first = begin_attempt_invocation(store, ledger, claim, owner_token=owner)
    replay = begin_attempt_invocation(store, ledger, claim, owner_token=owner)
    assert first.created is True
    assert first.launch_permitted is True
    assert replay.created is False
    assert replay.launch_permitted is False
    assert replay.marker == first.marker
    assert store.read(transport_invocation_key(claim)) is not None


def test_invocation_creation_must_be_durably_readable_before_launch(tmp_path):
    durable = LocalAtomicCreateStore(tmp_path)
    ledger, lease = _ledger_and_lease()
    owner = b"a" * 32
    claim = acquire_attempt_claim(durable, ledger, lease, owner_token=owner).claim
    lying = _DropInvocationCreateStore(durable)
    with pytest.raises(TransportClaimStateError, match="cannot be read"):
        begin_attempt_invocation(lying, ledger, claim, owner_token=owner)
    assert durable.read(transport_invocation_key(claim)) is None


def test_claim_that_becomes_stale_before_launch_cannot_invoke(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    ledger, lease = _ledger_and_lease()
    owner = b"a" * 32
    claim = acquire_attempt_claim(store, ledger, lease, owner_token=owner).claim
    advanced = record_infrastructure_failure(
        ledger,
        lease,
        failure_code="runner_unavailable",
    )
    with pytest.raises(ValueError, match="exact next FleetAuthority lease"):
        begin_attempt_invocation(store, advanced, claim, owner_token=owner)
    assert store.read(transport_invocation_key(claim)) is None


def test_wrong_owner_cannot_create_invocation_marker(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    ledger, lease = _ledger_and_lease()
    owner = b"a" * 32
    claim = acquire_attempt_claim(store, ledger, lease, owner_token=owner).claim
    with pytest.raises(TransportClaimConflict, match="does not control"):
        begin_attempt_invocation(store, ledger, claim, owner_token=b"b" * 32)
    assert store.read(transport_invocation_key(claim)) is None


def test_claim_serialization_rejects_misleading_derived_fields(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    ledger, lease = _ledger_and_lease()
    claim = acquire_attempt_claim(store, ledger, lease, owner_token=b"a" * 32).claim
    payload = {**claim.to_dict(), "transport_claim_sha256": claim.sha256}
    payload["claim_semantics"] = "scheduler may retry based on accuracy"
    with pytest.raises(ValueError, match="differs from canonical value"):
        PromotedTransportClaim.from_dict(payload)


def test_transport_store_rejects_path_traversal(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    for key in ("../claim.json", "a/../claim.json", "/absolute/claim.json", "a\\b"):
        with pytest.raises(ValueError):
            store.create_if_absent(key, b"payload")


def test_transport_keys_are_scientific_lease_derived_not_scheduler_named(tmp_path):
    store = LocalAtomicCreateStore(tmp_path)
    ledger, lease = _ledger_and_lease()
    claim = acquire_attempt_claim(store, ledger, lease, owner_token=b"a" * 32).claim
    claim_key = transport_claim_key(lease).lower()
    invocation_key = transport_invocation_key(claim).lower()
    for text in (claim_key, invocation_key):
        assert lease.shard_spec_sha256 in text
        assert "github" not in text
        assert "runner" not in text
        assert "accuracy" not in text
        assert "score" not in text
