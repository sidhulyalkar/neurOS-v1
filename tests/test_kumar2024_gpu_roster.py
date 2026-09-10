from __future__ import annotations

import copy
import hashlib
import importlib.util
import random
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "packages" / "neuros" / "src" / "neuros" / "evidence" / "kumar2024_gpu_roster.py"
)
if not MODULE_PATH.is_file():
    MODULE_PATH = Path(__file__).with_name("kumar2024_gpu_roster.py")
spec = importlib.util.spec_from_file_location("roster_target", MODULE_PATH)
assert spec and spec.loader
roster = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = roster
spec.loader.exec_module(roster)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


FLEET = digest("fleet")
BINDING = digest("binding")
PLAN = digest("plan")
ENV = digest("environment")


def lease(ordinal: int, *, subject: int | None = None, session: str = "1") -> dict:
    payload = {
        "schema_version": 1,
        "fleet_authority_sha256": FLEET,
        "gpu_binding_sha256": BINDING,
        "execution_plan_sha256": PLAN,
        "environment_authority_sha256": ENV,
        "shard_spec_sha256": digest(f"shard-{ordinal}"),
        "ordinal": ordinal,
        "subject": subject if subject is not None else ordinal + 1,
        "target_session": session,
        "split_seed": 2026 + ordinal,
        "method_id": roster.EEGNET_METHOD_ID,
        "model_seed": 31415 + ordinal,
        "budgets_per_class": list(roster.CALIBRATION_FRONTIER),
        "max_attempts": 3,
    }
    return {**payload, "lease_sha256": roster._identity(roster.LEASE_SCHEMA, payload)}


def build(items):
    return roster.build_fleet_roster(
        items,
        fleet_authority_sha256=FLEET,
        expected_lease_count=len(items),
        gpu_binding_sha256=BINDING,
        execution_plan_sha256=PLAN,
        environment_authority_sha256=ENV,
        max_attempts_per_lease=3,
    )


def test_build_is_input_order_independent_and_all_proofs_verify():
    leases = [lease(i) for i in range(9)]
    authority_a, packages_a = build(leases)
    shuffled = leases[:]
    random.Random(2026).shuffle(shuffled)
    authority_b, packages_b = build(shuffled)
    assert authority_a.to_dict() == authority_b.to_dict()
    assert [item.to_dict() for item in packages_a] == [item.to_dict() for item in packages_b]
    assert authority_a.merkle_algorithm == roster.MERKLE_ALGORITHM
    assert authority_a.odd_node_policy == "duplicate_last"
    assert all(
        roster.verify_proof_carrying_lease(package, authority_a) == package.lease_sha256
        for package in packages_a
    )


def test_singleton_is_explicit_and_empty_rejects():
    authority, packages = build([lease(0)])
    assert authority.merkle_root_sha256 == roster._leaf_hash(packages[0].lease_sha256)
    assert packages[0].authentication_path == ()
    roster.verify_proof_carrying_lease(packages[0], authority)
    with pytest.raises(ValueError, match="positive integer"):
        roster.build_fleet_roster(
            [],
            fleet_authority_sha256=FLEET,
            expected_lease_count=0,
            gpu_binding_sha256=BINDING,
            execution_plan_sha256=PLAN,
            environment_authority_sha256=ENV,
            max_attempts_per_lease=3,
        )


def test_altered_lease_payload_with_unchanged_proof_rejects():
    authority, packages = build([lease(i) for i in range(5)])
    raw = packages[2].to_dict()
    raw["lease"]["subject"] = 18
    with pytest.raises(ValueError, match="lease_sha256"):
        roster.ProofCarryingLease(
            raw["lease"], raw["lease_sha256"], raw["roster_authority_sha256"],
            raw["leaf_index"], packages[2].authentication_path,
        )


def test_altered_ordinal_rejects_before_path_verification():
    authority, packages = build([lease(i) for i in range(5)])
    payload = dict(packages[2].lease)
    payload["ordinal"] = 3
    payload_without_sha = {k: v for k, v in payload.items() if k != "lease_sha256"}
    payload["lease_sha256"] = roster._identity(roster.LEASE_SCHEMA, payload_without_sha)
    package = roster.ProofCarryingLease(
        payload, payload["lease_sha256"], authority.sha256, 2, packages[2].authentication_path
    )
    with pytest.raises(ValueError, match="ordinal"):
        roster.verify_proof_carrying_lease(package, authority)


def test_swapped_proof_order_rejects():
    authority, packages = build([lease(i) for i in range(9)])
    path = list(packages[4].authentication_path)
    path[0], path[1] = path[1], path[0]
    package = roster.ProofCarryingLease(
        packages[4].lease, packages[4].lease_sha256, authority.sha256, 4, tuple(path)
    )
    with pytest.raises(ValueError):
        roster.verify_proof_carrying_lease(package, authority)


def test_flipped_sibling_side_rejects():
    authority, packages = build([lease(i) for i in range(5)])
    path = list(packages[1].authentication_path)
    first = path[0]
    path[0] = roster.MerkleProofEntry("right" if first.side == "left" else "left", first.sibling_sha256)
    package = roster.ProofCarryingLease(
        packages[1].lease, packages[1].lease_sha256, authority.sha256, 1, tuple(path)
    )
    with pytest.raises(ValueError, match="side"):
        roster.verify_proof_carrying_lease(package, authority)


def test_foreign_sibling_hash_rejects():
    authority, packages = build([lease(i) for i in range(5)])
    path = list(packages[1].authentication_path)
    path[0] = roster.MerkleProofEntry(path[0].side, digest("foreign"))
    package = roster.ProofCarryingLease(
        packages[1].lease, packages[1].lease_sha256, authority.sha256, 1, tuple(path)
    )
    with pytest.raises(ValueError, match="root"):
        roster.verify_proof_carrying_lease(package, authority)


def test_proof_from_different_roster_rejects():
    authority_a, packages_a = build([lease(i) for i in range(5)])
    foreign = [lease(i) for i in range(5)]
    foreign[4] = lease(4, subject=18)
    # reseal changed LeaseSpec
    p = {k: v for k, v in foreign[4].items() if k != "lease_sha256"}
    foreign[4]["lease_sha256"] = roster._identity(roster.LEASE_SCHEMA, p)
    authority_b, _ = build(foreign)
    with pytest.raises(ValueError, match="foreign roster"):
        roster.verify_proof_carrying_lease(packages_a[2], authority_b)
    assert authority_a.sha256 != authority_b.sha256


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("environment_authority_sha256", "environment"),
        ("execution_plan_sha256", "execution plan"),
    ],
)
def test_wrong_bound_authority_rejects_during_build(field, message):
    item = lease(0)
    item[field] = digest("wrong")
    payload = {k: v for k, v in item.items() if k != "lease_sha256"}
    item["lease_sha256"] = roster._identity(roster.LEASE_SCHEMA, payload)
    with pytest.raises(ValueError, match=message):
        build([item])


def test_duplicate_lease_identity_rejects():
    item = lease(0)
    with pytest.raises(ValueError, match="duplicate lease"):
        roster.build_fleet_roster(
            [item, copy.deepcopy(item)], fleet_authority_sha256=FLEET,
            expected_lease_count=2, gpu_binding_sha256=BINDING,
            execution_plan_sha256=PLAN, environment_authority_sha256=ENV,
            max_attempts_per_lease=3,
        )


def test_duplicate_shard_identity_rejects_even_when_lease_is_resealed():
    items = [lease(0), lease(1)]
    items[1]["shard_spec_sha256"] = items[0]["shard_spec_sha256"]
    p = {k: v for k, v in items[1].items() if k != "lease_sha256"}
    items[1]["lease_sha256"] = roster._identity(roster.LEASE_SCHEMA, p)
    with pytest.raises(ValueError, match="duplicate shard"):
        build(items)


@pytest.mark.parametrize("ordinals", [[0, 2], [1, 2], [0, 1, 3]])
def test_missing_or_noncontiguous_ordinal_rejects(ordinals):
    items = [lease(value) for value in ordinals]
    with pytest.raises(ValueError, match="contiguous"):
        roster.build_fleet_roster(
            items,
            fleet_authority_sha256=FLEET,
            expected_lease_count=len(items),
            gpu_binding_sha256=BINDING,
            execution_plan_sha256=PLAN,
            environment_authority_sha256=ENV,
            max_attempts_per_lease=3,
        )


def test_missing_or_extra_leaf_rejects_against_frozen_count():
    items = [lease(i) for i in range(4)]
    with pytest.raises(ValueError, match="count"):
        roster.build_fleet_roster(
            items[:-1], fleet_authority_sha256=FLEET, expected_lease_count=4,
            gpu_binding_sha256=BINDING, execution_plan_sha256=PLAN, environment_authority_sha256=ENV,
            max_attempts_per_lease=3,
        )
    with pytest.raises(ValueError, match="count"):
        roster.build_fleet_roster(
            [*items, lease(4)], fleet_authority_sha256=FLEET, expected_lease_count=4,
            gpu_binding_sha256=BINDING, execution_plan_sha256=PLAN, environment_authority_sha256=ENV,
            max_attempts_per_lease=3,
        )


def test_scientifically_contaminated_payload_rejects():
    item = lease(0)
    item["balanced_accuracy"] = 0.99
    with pytest.raises(ValueError, match="scientific field"):
        build([item])


def test_verifier_never_sorts_proof_nodes():
    authority, packages = build([lease(i) for i in range(6)])
    package = packages[3]
    reversed_path = tuple(reversed(package.authentication_path))
    contaminated = roster.ProofCarryingLease(
        package.lease, package.lease_sha256, authority.sha256, package.leaf_index, reversed_path
    )
    with pytest.raises(ValueError):
        roster.verify_proof_carrying_lease(contaminated, authority)


def test_duplicate_last_policy_is_explicitly_verified():
    authority, packages = build([lease(i) for i in range(5)])
    package = packages[-1]
    assert package.authentication_path[0].side == "right"
    assert package.authentication_path[0].sibling_sha256 == roster._leaf_hash(package.lease_sha256)
    path = list(package.authentication_path)
    path[0] = roster.MerkleProofEntry("right", digest("not-self"))
    contaminated = roster.ProofCarryingLease(
        package.lease, package.lease_sha256, authority.sha256, package.leaf_index, tuple(path)
    )
    with pytest.raises(ValueError, match="duplicate-last"):
        roster.verify_proof_carrying_lease(contaminated, authority)


def test_wrong_gpu_binding_rejects_even_when_lease_is_resealed():
    item = lease(0)
    item["gpu_binding_sha256"] = digest("foreign-binding")
    payload = {k: v for k, v in item.items() if k != "lease_sha256"}
    item["lease_sha256"] = roster._identity(roster.LEASE_SCHEMA, payload)
    with pytest.raises(ValueError, match="GPU binding"):
        build([item])


def test_wrong_retry_ceiling_rejects_even_when_lease_is_resealed():
    item = lease(0)
    item["max_attempts"] = 4
    payload = {k: v for k, v in item.items() if k != "lease_sha256"}
    item["lease_sha256"] = roster._identity(roster.LEASE_SCHEMA, payload)
    with pytest.raises(ValueError, match="retry ceiling"):
        build([item])


def test_canonical_authority_and_proof_bytes_are_input_order_independent():
    items = [lease(i) for i in range(7)]
    authority_a, packages_a = build(items)
    authority_b, packages_b = build(list(reversed(items)))
    assert roster.serialize_roster_authority(authority_a) == roster.serialize_roster_authority(authority_b)
    assert [roster.serialize_proof_carrying_lease(x) for x in packages_a] == [
        roster.serialize_proof_carrying_lease(x) for x in packages_b
    ]


def test_production_shape_810_synthetic_identities_are_deterministic():
    split_seeds = (2026, 3407, 9109)
    model_seeds = (31415, 384165836, 3991196546)
    items = []
    ordinal = 0
    for subject in range(1, 19):
        for session in ("1", "2", "3", "4", "5"):
            for split_seed in split_seeds:
                for model_seed in model_seeds:
                    payload = {
                        "schema_version": 1,
                        "fleet_authority_sha256": FLEET,
                        "gpu_binding_sha256": BINDING,
                        "execution_plan_sha256": PLAN,
                        "environment_authority_sha256": ENV,
                        "shard_spec_sha256": digest(f"synthetic-shard-{ordinal}"),
                        "ordinal": ordinal,
                        "subject": subject,
                        "target_session": session,
                        "split_seed": split_seed,
                        "method_id": roster.EEGNET_METHOD_ID,
                        "model_seed": model_seed,
                        "budgets_per_class": list(roster.CALIBRATION_FRONTIER),
                        "max_attempts": 3,
                    }
                    items.append({**payload, "lease_sha256": roster._identity(roster.LEASE_SCHEMA, payload)})
                    ordinal += 1
    assert len(items) == 810
    authority_a, packages_a = roster.build_fleet_roster(
        items, fleet_authority_sha256=FLEET, expected_lease_count=810,
        gpu_binding_sha256=BINDING, execution_plan_sha256=PLAN, environment_authority_sha256=ENV,
        max_attempts_per_lease=3,
    )
    authority_b, packages_b = roster.build_fleet_roster(
        reversed(items), fleet_authority_sha256=FLEET, expected_lease_count=810,
        gpu_binding_sha256=BINDING, execution_plan_sha256=PLAN, environment_authority_sha256=ENV,
        max_attempts_per_lease=3,
    )
    assert authority_a.to_dict() == authority_b.to_dict()
    assert packages_a[0].to_dict() == packages_b[0].to_dict()
    assert packages_a[-1].to_dict() == packages_b[-1].to_dict()
    for index in (0, 1, 404, 809):
        roster.verify_proof_carrying_lease(packages_a[index], authority_a)
