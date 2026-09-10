"""Merkle roster authority for proof-carrying Kumar2024 GPU fleet leases.

This module is a distribution-integrity layer only. It independently validates
canonical LeaseSpec records, commits their exact ordered identities into a
versioned Merkle tree, and verifies compact inclusion proofs before a provider
or model invocation is allowed to begin.

It intentionally does not import the fleet scheduler, model code, provider
SDKs, or scientific-result code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

ROSTER_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_roster_authority.v1"
ORDERED_LEAVES_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_roster_ordered_leaves.v1"
LEAF_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_roster_leaf.v1"
NODE_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_roster_node.v1"
LEASE_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_lease.v1"
MERKLE_ALGORITHM = "sha256-canonical-json-domain-separated-duplicate-last-v1"
ODD_NODE_POLICY = "duplicate_last"
EEGNET_METHOD_ID = "braindecode-eegnet"
CALIBRATION_FRONTIER = (0, 1, 2, 5, 10)

LEASE_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "fleet_authority_sha256",
        "gpu_binding_sha256",
        "execution_plan_sha256",
        "environment_authority_sha256",
        "shard_spec_sha256",
        "ordinal",
        "subject",
        "target_session",
        "split_seed",
        "method_id",
        "model_seed",
        "budgets_per_class",
        "max_attempts",
    }
)
LEASE_RECORD_KEYS = LEASE_PAYLOAD_KEYS | {"lease_sha256"}
FORBIDDEN_CONTROL_KEYS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "auc",
        "auroc",
        "f1",
        "loss",
        "logits",
        "method_ranking",
        "predictions",
        "probabilities",
        "scientific_score",
        "final_assessment_metric",
    }
)


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _identity(schema: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({"schema": schema, "payload": payload})).hexdigest()


def _sha(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical lowercase SHA-256")
    if (
        value != value.strip()
        or value != value.lower()
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a canonical lowercase SHA-256")
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _frontier(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("budgets_per_class must be the frozen calibration frontier")
    result = tuple(value)
    if result != CALIBRATION_FRONTIER or any(
        isinstance(item, bool) or not isinstance(item, int) for item in result
    ):
        raise ValueError(f"budgets_per_class must equal {CALIBRATION_FRONTIER}")
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def reject_scientific_fields(value: Any, path: str = "$") -> None:
    """Fail closed if a scientific outcome enters the roster boundary."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string roster key at {path}")
            if key.lower() in FORBIDDEN_CONTROL_KEYS:
                raise ValueError(f"scientific field {key!r} is forbidden at {path}")
            reject_scientific_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_scientific_fields(item, f"{path}[{index}]")


def _normalized_lease(record: Mapping[str, Any]) -> dict[str, Any]:
    reject_scientific_fields(record)
    if set(record) != LEASE_RECORD_KEYS:
        raise ValueError(
            "lease record keys differ: "
            f"missing={sorted(LEASE_RECORD_KEYS - set(record))}, "
            f"extra={sorted(set(record) - LEASE_RECORD_KEYS)}"
        )
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("lease schema_version must be 1")
    normalized = {
        "schema_version": 1,
        "fleet_authority_sha256": _sha(
            "fleet_authority_sha256", record["fleet_authority_sha256"]
        ),
        "gpu_binding_sha256": _sha("gpu_binding_sha256", record["gpu_binding_sha256"]),
        "execution_plan_sha256": _sha(
            "execution_plan_sha256", record["execution_plan_sha256"]
        ),
        "environment_authority_sha256": _sha(
            "environment_authority_sha256", record["environment_authority_sha256"]
        ),
        "shard_spec_sha256": _sha("shard_spec_sha256", record["shard_spec_sha256"]),
        "ordinal": _nonnegative_int("ordinal", record["ordinal"]),
        "subject": _positive_int("subject", record["subject"]),
        "target_session": _identifier("target_session", record["target_session"]),
        "split_seed": _nonnegative_int("split_seed", record["split_seed"]),
        "method_id": _identifier("method_id", record["method_id"]),
        "model_seed": _nonnegative_int("model_seed", record["model_seed"]),
        "budgets_per_class": list(_frontier(record["budgets_per_class"])),
        "max_attempts": _positive_int("max_attempts", record["max_attempts"]),
    }
    if normalized["method_id"] != EEGNET_METHOD_ID:
        raise ValueError("roster lease is not EEGNet")
    expected_sha = _identity(LEASE_SCHEMA, normalized)
    provided_sha = _sha("lease_sha256", record["lease_sha256"])
    if provided_sha != expected_sha:
        raise ValueError("lease_sha256 does not match canonical LeaseSpec payload")
    return {**normalized, "lease_sha256": provided_sha}


def _leaf_hash(lease_sha256: str) -> str:
    return _identity(LEAF_SCHEMA, {"lease_sha256": _sha("lease_sha256", lease_sha256)})


def _node_hash(left_sha256: str, right_sha256: str) -> str:
    return _identity(
        NODE_SCHEMA,
        {
            "left_sha256": _sha("left_sha256", left_sha256),
            "right_sha256": _sha("right_sha256", right_sha256),
        },
    )


def _merkle_root(leaf_hashes: Sequence[str]) -> str:
    """Compute the v1 root. Empty trees are invalid; singleton root is its leaf."""
    level = tuple(_sha("leaf_sha256", item) for item in leaf_hashes)
    if not level:
        raise ValueError("roster cannot be empty")
    while len(level) > 1:
        next_level = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            next_level.append(_node_hash(left, right))
        level = tuple(next_level)
    return level[0]


@dataclass(frozen=True)
class MerkleProofEntry:
    side: str
    sibling_sha256: str

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("Merkle proof side must be 'left' or 'right'")
        object.__setattr__(
            self,
            "sibling_sha256",
            _sha("sibling_sha256", self.sibling_sha256),
        )

    def to_dict(self) -> dict[str, str]:
        return {"side": self.side, "sibling_sha256": self.sibling_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MerkleProofEntry":
        if set(value) != {"side", "sibling_sha256"}:
            raise ValueError("Merkle proof entry has unexpected keys")
        return cls(value["side"], value["sibling_sha256"])


@dataclass(frozen=True)
class FleetRosterAuthority:
    fleet_authority_sha256: str
    expected_lease_count: int
    ordered_leaf_commitment_sha256: str
    merkle_root_sha256: str
    gpu_binding_sha256: str
    execution_plan_sha256: str
    environment_authority_sha256: str
    max_attempts_per_lease: int
    merkle_algorithm: str = MERKLE_ALGORITHM
    odd_node_policy: str = ODD_NODE_POLICY
    lease_schema: str = LEASE_SCHEMA
    method_id: str = EEGNET_METHOD_ID
    budgets_per_class: tuple[int, ...] = CALIBRATION_FRONTIER

    def __post_init__(self) -> None:
        for field in (
            "fleet_authority_sha256",
            "ordered_leaf_commitment_sha256",
            "merkle_root_sha256",
            "gpu_binding_sha256",
            "execution_plan_sha256",
            "environment_authority_sha256",
        ):
            object.__setattr__(self, field, _sha(field, getattr(self, field)))
        object.__setattr__(
            self,
            "expected_lease_count",
            _positive_int("expected_lease_count", self.expected_lease_count),
        )
        object.__setattr__(
            self,
            "max_attempts_per_lease",
            _positive_int("max_attempts_per_lease", self.max_attempts_per_lease),
        )
        if self.merkle_algorithm != MERKLE_ALGORITHM:
            raise ValueError("unsupported Merkle algorithm")
        if self.odd_node_policy != ODD_NODE_POLICY:
            raise ValueError("unsupported odd-node policy")
        if self.lease_schema != LEASE_SCHEMA:
            raise ValueError("roster must commit the canonical LeaseSpec v1 schema")
        if self.method_id != EEGNET_METHOD_ID:
            raise ValueError("roster v1 is fixed to EEGNet")
        object.__setattr__(self, "budgets_per_class", _frontier(self.budgets_per_class))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "kumar2024_gpu_fleet_merkle_roster_authority",
            "fleet_authority_sha256": self.fleet_authority_sha256,
            "expected_lease_count": self.expected_lease_count,
            "ordered_leaf_commitment_sha256": self.ordered_leaf_commitment_sha256,
            "merkle_algorithm": self.merkle_algorithm,
            "odd_node_policy": self.odd_node_policy,
            "leaf_domain": LEAF_SCHEMA,
            "node_domain": NODE_SCHEMA,
            "lease_schema": self.lease_schema,
            "merkle_root_sha256": self.merkle_root_sha256,
            "gpu_binding_sha256": self.gpu_binding_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "environment_authority_sha256": self.environment_authority_sha256,
            "max_attempts_per_lease": self.max_attempts_per_lease,
            "method_id": self.method_id,
            "budgets_per_class": list(self.budgets_per_class),
            "empty_tree_policy": "reject",
            "singleton_tree_policy": "root_equals_leaf",
            "proof_ordering": "leaf_index_and_explicit_sibling_side_no_sorting",
            "membership_implies_execution": False,
            "membership_implies_numerical_efficacy": False,
            "membership_implies_fleet_completeness": False,
            "orion_comparison_permitted": False,
        }

    @property
    def sha256(self) -> str:
        return _identity(ROSTER_SCHEMA, self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "roster_authority_sha256": self.sha256}


@dataclass(frozen=True)
class ProofCarryingLease:
    lease: Mapping[str, Any]
    lease_sha256: str
    roster_authority_sha256: str
    leaf_index: int
    authentication_path: tuple[MerkleProofEntry, ...]

    def __post_init__(self) -> None:
        normalized = _normalized_lease(self.lease)
        lease_sha = _sha("lease_sha256", self.lease_sha256)
        if normalized["lease_sha256"] != lease_sha:
            raise ValueError("proof package lease_sha256 differs from LeaseSpec")
        object.__setattr__(self, "lease", _freeze(normalized))
        object.__setattr__(self, "lease_sha256", lease_sha)
        object.__setattr__(
            self,
            "roster_authority_sha256",
            _sha("roster_authority_sha256", self.roster_authority_sha256),
        )
        object.__setattr__(self, "leaf_index", _nonnegative_int("leaf_index", self.leaf_index))
        path = tuple(self.authentication_path)
        if any(not isinstance(item, MerkleProofEntry) for item in path):
            raise TypeError("authentication_path must contain MerkleProofEntry values")
        object.__setattr__(self, "authentication_path", path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "lease": _thaw(self.lease),
            "lease_sha256": self.lease_sha256,
            "roster_authority_sha256": self.roster_authority_sha256,
            "leaf_index": self.leaf_index,
            "authentication_path": [item.to_dict() for item in self.authentication_path],
        }


def _proof_for_index(leaf_hashes: Sequence[str], leaf_index: int) -> tuple[MerkleProofEntry, ...]:
    count = len(leaf_hashes)
    if count == 0:
        raise ValueError("roster cannot be empty")
    if leaf_index < 0 or leaf_index >= count:
        raise ValueError("leaf_index outside roster")
    level = tuple(_sha("leaf_sha256", item) for item in leaf_hashes)
    index = leaf_index
    proof: list[MerkleProofEntry] = []
    while len(level) > 1:
        if index % 2:
            proof.append(MerkleProofEntry("left", level[index - 1]))
        else:
            sibling = level[index + 1] if index + 1 < len(level) else level[index]
            proof.append(MerkleProofEntry("right", sibling))
        next_level = []
        for offset in range(0, len(level), 2):
            left = level[offset]
            right = level[offset + 1] if offset + 1 < len(level) else left
            next_level.append(_node_hash(left, right))
        index //= 2
        level = tuple(next_level)
    return tuple(proof)


def build_fleet_roster(
    leases: Iterable[Mapping[str, Any]],
    *,
    fleet_authority_sha256: str,
    expected_lease_count: int,
    gpu_binding_sha256: str,
    execution_plan_sha256: str,
    environment_authority_sha256: str,
    max_attempts_per_lease: int,
) -> tuple[FleetRosterAuthority, tuple[ProofCarryingLease, ...]]:
    """Build one deterministic roster and one compact proof per canonical lease."""
    expected_fleet = _sha("fleet_authority_sha256", fleet_authority_sha256)
    expected_count = _positive_int("expected_lease_count", expected_lease_count)
    expected_binding = _sha("gpu_binding_sha256", gpu_binding_sha256)
    expected_plan = _sha("execution_plan_sha256", execution_plan_sha256)
    expected_environment = _sha(
        "environment_authority_sha256", environment_authority_sha256
    )
    expected_max_attempts = _positive_int(
        "max_attempts_per_lease", max_attempts_per_lease
    )
    normalized = [_normalized_lease(item) for item in leases]
    if len(normalized) != expected_count:
        raise ValueError("roster lease count differs from frozen fleet authority")
    if len({item["lease_sha256"] for item in normalized}) != len(normalized):
        raise ValueError("roster contains duplicate lease identities")
    if len({item["shard_spec_sha256"] for item in normalized}) != len(normalized):
        raise ValueError("roster contains duplicate shard identities")
    by_ordinal: dict[int, dict[str, Any]] = {}
    for lease in normalized:
        if lease["ordinal"] in by_ordinal:
            raise ValueError("roster contains duplicate lease ordinals")
        if lease["fleet_authority_sha256"] != expected_fleet:
            raise ValueError("lease names a foreign fleet authority")
        if lease["gpu_binding_sha256"] != expected_binding:
            raise ValueError("lease names a foreign GPU binding")
        if lease["execution_plan_sha256"] != expected_plan:
            raise ValueError("lease names a foreign execution plan")
        if lease["environment_authority_sha256"] != expected_environment:
            raise ValueError("lease names a foreign environment authority")
        if lease["max_attempts"] != expected_max_attempts:
            raise ValueError("lease retry ceiling differs from fleet authority")
        by_ordinal[lease["ordinal"]] = lease
    if set(by_ordinal) != set(range(expected_count)):
        raise ValueError("roster lease ordinals must be contiguous from zero")
    ordered = tuple(by_ordinal[index] for index in range(expected_count))
    lease_ids = tuple(item["lease_sha256"] for item in ordered)
    leaf_hashes = tuple(_leaf_hash(item) for item in lease_ids)
    ordered_commitment = _identity(
        ORDERED_LEAVES_SCHEMA,
        {
            "expected_lease_count": expected_count,
            "leaf_sha256s": list(leaf_hashes),
        },
    )
    authority = FleetRosterAuthority(
        expected_fleet,
        expected_count,
        ordered_commitment,
        _merkle_root(leaf_hashes),
        expected_binding,
        expected_plan,
        expected_environment,
        expected_max_attempts,
    )
    packages = tuple(
        ProofCarryingLease(
            lease,
            lease["lease_sha256"],
            authority.sha256,
            index,
            _proof_for_index(leaf_hashes, index),
        )
        for index, lease in enumerate(ordered)
    )
    return authority, packages


def verify_proof_carrying_lease(
    package: ProofCarryingLease,
    authority: FleetRosterAuthority,
) -> str:
    """Verify one roster inclusion proof without consulting scheduler state."""
    if not isinstance(package, ProofCarryingLease):
        raise TypeError("package must be ProofCarryingLease")
    if not isinstance(authority, FleetRosterAuthority):
        raise TypeError("authority must be FleetRosterAuthority")
    if package.roster_authority_sha256 != authority.sha256:
        raise ValueError("proof package names a foreign roster authority")
    lease = _normalized_lease(_thaw(package.lease))
    if package.lease_sha256 != lease["lease_sha256"]:
        raise ValueError("proof package lease identity mismatch")
    if lease["ordinal"] != package.leaf_index:
        raise ValueError("lease ordinal differs from Merkle leaf index")
    if package.leaf_index >= authority.expected_lease_count:
        raise ValueError("Merkle leaf index outside roster authority")
    if lease["fleet_authority_sha256"] != authority.fleet_authority_sha256:
        raise ValueError("lease names a foreign fleet authority")
    if lease["gpu_binding_sha256"] != authority.gpu_binding_sha256:
        raise ValueError("lease names a foreign GPU binding")
    if lease["execution_plan_sha256"] != authority.execution_plan_sha256:
        raise ValueError("lease names a foreign execution plan")
    if lease["environment_authority_sha256"] != authority.environment_authority_sha256:
        raise ValueError("lease names a foreign environment authority")
    if lease["max_attempts"] != authority.max_attempts_per_lease:
        raise ValueError("lease retry ceiling differs from roster authority")
    if lease["method_id"] != authority.method_id:
        raise ValueError("lease method differs from roster policy")
    if tuple(lease["budgets_per_class"]) != authority.budgets_per_class:
        raise ValueError("lease calibration frontier differs from roster policy")

    current = _leaf_hash(package.lease_sha256)
    index = package.leaf_index
    width = authority.expected_lease_count
    path = package.authentication_path
    expected_depth = 0
    probe_width = width
    while probe_width > 1:
        expected_depth += 1
        probe_width = (probe_width + 1) // 2
    if len(path) != expected_depth:
        raise ValueError("Merkle authentication path has the wrong depth")

    for entry in path:
        expected_side = "left" if index % 2 else "right"
        if entry.side != expected_side:
            raise ValueError("Merkle proof side contradicts the explicit leaf index")
        if index % 2 == 0 and index + 1 >= width:
            if entry.sibling_sha256 != current:
                raise ValueError("duplicate-last proof must duplicate the unpaired node")
        if entry.side == "left":
            current = _node_hash(entry.sibling_sha256, current)
        else:
            current = _node_hash(current, entry.sibling_sha256)
        index //= 2
        width = (width + 1) // 2
    if index != 0 or width != 1:
        raise ValueError("Merkle proof did not terminate at the roster root")
    if current != authority.merkle_root_sha256:
        raise ValueError("Merkle proof does not reconstruct the roster root")
    return package.lease_sha256


def serialize_roster_authority(authority: FleetRosterAuthority) -> bytes:
    """Return the canonical byte representation used for distribution and hashing."""
    if not isinstance(authority, FleetRosterAuthority):
        raise TypeError("authority must be FleetRosterAuthority")
    return _canonical(authority.to_dict())


def serialize_proof_carrying_lease(package: ProofCarryingLease) -> bytes:
    """Return a canonical provider-dispatch package without scheduler state."""
    if not isinstance(package, ProofCarryingLease):
        raise TypeError("package must be ProofCarryingLease")
    return _canonical(package.to_dict())
