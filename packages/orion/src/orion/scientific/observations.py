"""Observation-role, information-budget, and preprocessing-fit authority."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .common import (
    ALLOWED_ROLES,
    ObservationRole,
    OperationKind,
    TransformFitKind,
    canonical_sha256,
    display_fingerprint,
    freeze_json,
    nonempty,
    require_sha256,
    strings,
    thaw_json,
)


@dataclass(frozen=True, slots=True)
class ObservationSetAuthority:
    authority_id: str
    dataset_lineage_sha256: str
    role: ObservationRole
    observation_ids: tuple[str, ...]
    domain_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ObservationSetAuthority schema_version must be 2")
        if not isinstance(self.role, ObservationRole):
            raise TypeError("role must be ObservationRole")
        object.__setattr__(self, "authority_id", nonempty("authority_id", self.authority_id))
        object.__setattr__(
            self,
            "dataset_lineage_sha256",
            require_sha256("dataset_lineage_sha256", self.dataset_lineage_sha256),
        )
        object.__setattr__(self, "domain_id", nonempty("domain_id", self.domain_id))
        object.__setattr__(self, "observation_ids", strings("observation_ids", self.observation_ids))
        metadata = freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def authority_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.authority_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "authority_id": self.authority_id,
            "dataset_lineage_sha256": self.dataset_lineage_sha256,
            "role": self.role.value,
            "observation_ids": list(self.observation_ids),
            "domain_id": self.domain_id,
            "metadata": thaw_json(self.metadata),
        }
        if include_identity:
            payload["authority_sha256"] = self.authority_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class TargetObservationBudget:
    """Separate labeled and unlabeled target information budgets."""

    labeled_examples: int = 0
    labeled_examples_per_class: int | None = None
    unlabeled_examples: int = 0
    unlabeled_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("labeled_examples", self.labeled_examples),
            ("unlabeled_examples", self.unlabeled_examples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.labeled_examples_per_class is not None:
            value = self.labeled_examples_per_class
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("labeled_examples_per_class must be a non-negative integer or None")
            if value > 0 and self.labeled_examples == 0:
                raise ValueError(
                    "positive labeled_examples_per_class is inconsistent with zero labeled_examples"
                )
            if value > self.labeled_examples:
                raise ValueError(
                    "labeled_examples_per_class cannot exceed total labeled_examples"
                )
        if self.unlabeled_seconds is not None:
            value = self.unlabeled_seconds
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError("unlabeled_seconds must be numeric or None")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("unlabeled_seconds must be finite and non-negative")
            object.__setattr__(self, "unlabeled_seconds", number)

    @property
    def has_target_information(self) -> bool:
        return bool(
            self.labeled_examples
            or (self.labeled_examples_per_class or 0)
            or self.unlabeled_examples
            or (self.unlabeled_seconds or 0.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "labeled_examples": self.labeled_examples,
            "labeled_examples_per_class": self.labeled_examples_per_class,
            "unlabeled_examples": self.unlabeled_examples,
            "unlabeled_seconds": self.unlabeled_seconds,
        }


@dataclass(frozen=True, slots=True)
class ObservationConsumption:
    """Exact observation authorities consumed by one scientific operation."""

    operation_id: str
    operation: OperationKind
    observation_authority_sha256s: tuple[str, ...]
    roles: tuple[ObservationRole, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ObservationConsumption schema_version must be 2")
        if not isinstance(self.operation, OperationKind):
            raise TypeError("operation must be OperationKind")
        object.__setattr__(self, "operation_id", nonempty("operation_id", self.operation_id))
        shas = tuple(
            require_sha256("observation authority SHA-256", value)
            for value in self.observation_authority_sha256s
        )
        if not shas:
            raise ValueError("observation consumption must contain at least one observation authority")
        if len(set(shas)) != len(shas):
            raise ValueError("observation_authority_sha256s cannot contain duplicates")
        roles = tuple(self.roles)
        if any(not isinstance(role, ObservationRole) for role in roles):
            raise TypeError("roles must contain only ObservationRole values")
        if len(shas) != len(roles):
            raise ValueError("roles must align one-to-one with observation authorities")
        allowed = ALLOWED_ROLES[self.operation]
        unauthorized = sorted({role.value for role in roles if role not in allowed})
        if unauthorized:
            raise ValueError(
                f"{self.operation.value} cannot consume observation roles {unauthorized}; "
                f"allowed={sorted(role.value for role in allowed)}"
            )
        object.__setattr__(self, "observation_authority_sha256s", shas)
        object.__setattr__(self, "roles", roles)

    @classmethod
    def bind(
        cls,
        *,
        operation_id: str,
        operation: OperationKind,
        observations: Sequence[ObservationSetAuthority],
    ) -> "ObservationConsumption":
        observations_tuple = tuple(observations)
        if any(not isinstance(item, ObservationSetAuthority) for item in observations_tuple):
            raise TypeError("observations must contain only ObservationSetAuthority objects")
        return cls(
            operation_id=operation_id,
            operation=operation,
            observation_authority_sha256s=tuple(item.authority_sha256 for item in observations_tuple),
            roles=tuple(item.role for item in observations_tuple),
        )

    @property
    def consumption_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.consumption_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "operation": self.operation.value,
            "observation_authority_sha256s": list(self.observation_authority_sha256s),
            "roles": [role.value for role in self.roles],
        }
        if include_identity:
            payload["consumption_sha256"] = self.consumption_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class PreprocessingFitAuthority:
    transform_id: str
    fit_kind: TransformFitKind
    implementation: str
    implementation_version: str
    state_sha256: str
    consumption: ObservationConsumption | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("PreprocessingFitAuthority schema_version must be 2")
        object.__setattr__(self, "transform_id", nonempty("transform_id", self.transform_id))
        object.__setattr__(self, "implementation", nonempty("implementation", self.implementation))
        object.__setattr__(
            self,
            "implementation_version",
            nonempty("implementation_version", self.implementation_version),
        )
        object.__setattr__(self, "state_sha256", require_sha256("state_sha256", self.state_sha256))
        if self.fit_kind is TransformFitKind.PREDECLARED_FIXED:
            if self.consumption is not None:
                raise ValueError("predeclared fixed transforms cannot claim data-fitted consumption")
        elif self.fit_kind is TransformFitKind.DATA_FITTED:
            if self.consumption is None:
                raise ValueError("data-fitted transforms require observation consumption authority")
            if not isinstance(self.consumption, ObservationConsumption):
                raise TypeError("consumption must be ObservationConsumption")
            if self.consumption.operation is not OperationKind.PREPROCESSING_FIT:
                raise ValueError("data-fitted preprocessing must use preprocessing_fit consumption")
        else:
            raise ValueError("fit_kind must be a TransformFitKind")
        metadata = freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def authority_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.authority_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "transform_id": self.transform_id,
            "fit_kind": self.fit_kind.value,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "state_sha256": self.state_sha256,
            "consumption": (
                None
                if self.consumption is None
                else self.consumption.to_dict(include_identity=False)
            ),
            "metadata": thaw_json(self.metadata),
        }
        if include_identity:
            payload["authority_sha256"] = self.authority_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload
