"""Dataset/model lineage and pretraining-overlap authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .common import (
    IdentityAvailability,
    LineageCompleteness,
    OverlapStatus,
    canonical_sha256,
    display_fingerprint,
    freeze_json,
    nonempty,
    optional_sha256,
    require_sha256,
    strings,
    thaw_json,
)


@dataclass(frozen=True, slots=True)
class IdentitySet:
    """Identifiers available at one lineage level, or why they are unavailable."""

    level: str
    availability: IdentityAvailability
    identifiers: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        level = nonempty("identity level", self.level)
        identifiers = strings("identity identifiers", self.identifiers)
        reason = self.unavailable_reason
        if self.availability is IdentityAvailability.AVAILABLE:
            if not identifiers:
                raise ValueError("available identity sets require at least one identifier")
            if reason is not None:
                raise ValueError("available identity sets cannot carry unavailable_reason")
        elif self.availability is IdentityAvailability.UNAVAILABLE:
            if identifiers:
                raise ValueError("unavailable identity sets cannot carry identifiers")
            reason = nonempty("unavailable_reason", reason or "")
        else:
            raise ValueError("availability must be an IdentityAvailability")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "identifiers", identifiers)
        object.__setattr__(self, "unavailable_reason", reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "availability": self.availability.value,
            "identifiers": list(self.identifiers),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class DatasetLineage:
    """Immutable lineage for one downstream or pretraining dataset/domain."""

    dataset_id: str
    upstream_source: str
    version: str | None = None
    revision: str | None = None
    content_sha256: str | None = None
    parent_dataset_ids: tuple[str, ...] = ()
    identity_sets: tuple[IdentitySet, ...] = ()
    preprocessing_history: tuple[str, ...] = ()
    sampling_assumptions: Mapping[str, Any] = field(default_factory=dict)
    license: str | None = None
    citation: str | None = None
    lineage_completeness: LineageCompleteness = LineageCompleteness.UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("DatasetLineage schema_version must be 2")
        dataset_id = nonempty("dataset_id", self.dataset_id)
        upstream_source = nonempty("upstream_source", self.upstream_source)
        parents = strings("parent_dataset_ids", self.parent_dataset_ids)
        if dataset_id in parents:
            raise ValueError("dataset cannot list itself as a parent")
        if len({item.level for item in self.identity_sets}) != len(self.identity_sets):
            raise ValueError("identity_sets cannot repeat an identity level")
        history = strings("preprocessing_history", self.preprocessing_history)
        sampling = freeze_json(self.sampling_assumptions)
        metadata = freeze_json(self.metadata)
        if not isinstance(sampling, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("sampling_assumptions and metadata must be mappings")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "upstream_source", upstream_source)
        object.__setattr__(self, "content_sha256", optional_sha256("content_sha256", self.content_sha256))
        object.__setattr__(self, "parent_dataset_ids", parents)
        object.__setattr__(self, "preprocessing_history", history)
        object.__setattr__(self, "sampling_assumptions", sampling)
        object.__setattr__(self, "metadata", metadata)

    @property
    def lineage_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.lineage_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "upstream_source": self.upstream_source,
            "version": self.version,
            "revision": self.revision,
            "content_sha256": self.content_sha256,
            "parent_dataset_ids": list(self.parent_dataset_ids),
            "identity_sets": [item.to_dict() for item in self.identity_sets],
            "preprocessing_history": list(self.preprocessing_history),
            "sampling_assumptions": thaw_json(self.sampling_assumptions),
            "license": self.license,
            "citation": self.citation,
            "lineage_completeness": self.lineage_completeness.value,
            "metadata": thaw_json(self.metadata),
        }
        if include_identity:
            payload["lineage_sha256"] = self.lineage_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class ModelLineage:
    """Immutable lineage for a trained/pretrained model or representation artifact."""

    model_id: str
    upstream_source: str
    version: str | None = None
    revision: str | None = None
    checkpoint_sha256: str | None = None
    pretraining_dataset_ids: tuple[str, ...] = ()
    pretraining_identity_sets: tuple[IdentitySet, ...] = ()
    preprocessing_history: tuple[str, ...] = ()
    pretraining_lineage_completeness: LineageCompleteness = LineageCompleteness.UNKNOWN
    input_assumptions: Mapping[str, Any] = field(default_factory=dict)
    license: str | None = None
    citation: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ModelLineage schema_version must be 2")
        model_id = nonempty("model_id", self.model_id)
        source = nonempty("upstream_source", self.upstream_source)
        datasets = strings("pretraining_dataset_ids", self.pretraining_dataset_ids)
        history = strings("preprocessing_history", self.preprocessing_history)
        if len({item.level for item in self.pretraining_identity_sets}) != len(self.pretraining_identity_sets):
            raise ValueError("pretraining_identity_sets cannot repeat an identity level")
        assumptions = freeze_json(self.input_assumptions)
        metadata = freeze_json(self.metadata)
        if not isinstance(assumptions, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("input_assumptions and metadata must be mappings")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "upstream_source", source)
        object.__setattr__(self, "checkpoint_sha256", optional_sha256("checkpoint_sha256", self.checkpoint_sha256))
        object.__setattr__(self, "pretraining_dataset_ids", datasets)
        object.__setattr__(self, "preprocessing_history", history)
        object.__setattr__(self, "input_assumptions", assumptions)
        object.__setattr__(self, "metadata", metadata)

    @property
    def lineage_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.lineage_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "upstream_source": self.upstream_source,
            "version": self.version,
            "revision": self.revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "pretraining_dataset_ids": list(self.pretraining_dataset_ids),
            "pretraining_identity_sets": [item.to_dict() for item in self.pretraining_identity_sets],
            "preprocessing_history": list(self.preprocessing_history),
            "pretraining_lineage_completeness": self.pretraining_lineage_completeness.value,
            "input_assumptions": thaw_json(self.input_assumptions),
            "license": self.license,
            "citation": self.citation,
            "metadata": thaw_json(self.metadata),
        }
        if include_identity:
            payload["lineage_sha256"] = self.lineage_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class PretrainingOverlapAudit:
    status: OverlapStatus
    model_id: str
    evaluation_dataset_id: str
    model_lineage_sha256: str
    evaluation_dataset_lineage_sha256: str
    matched_dataset_ids: tuple[str, ...] = ()
    matched_identity_levels: tuple[str, ...] = ()
    unresolved_ancestor_ids: tuple[str, ...] = ()
    scope: str = "dataset_domain_and_declared_identity"
    reason: str = ""
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("PretrainingOverlapAudit schema_version must be 2")
        object.__setattr__(self, "model_id", nonempty("model_id", self.model_id))
        object.__setattr__(self, "evaluation_dataset_id", nonempty("evaluation_dataset_id", self.evaluation_dataset_id))
        object.__setattr__(self, "model_lineage_sha256", require_sha256("model_lineage_sha256", self.model_lineage_sha256))
        object.__setattr__(self, "evaluation_dataset_lineage_sha256", require_sha256("evaluation_dataset_lineage_sha256", self.evaluation_dataset_lineage_sha256))
        object.__setattr__(self, "matched_dataset_ids", strings("matched_dataset_ids", self.matched_dataset_ids))
        object.__setattr__(self, "matched_identity_levels", strings("matched_identity_levels", self.matched_identity_levels))
        object.__setattr__(self, "unresolved_ancestor_ids", strings("unresolved_ancestor_ids", self.unresolved_ancestor_ids))
        object.__setattr__(self, "scope", nonempty("scope", self.scope))
        object.__setattr__(self, "reason", nonempty("reason", self.reason))

    @property
    def audit_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.audit_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "model_id": self.model_id,
            "evaluation_dataset_id": self.evaluation_dataset_id,
            "model_lineage_sha256": self.model_lineage_sha256,
            "evaluation_dataset_lineage_sha256": self.evaluation_dataset_lineage_sha256,
            "matched_dataset_ids": list(self.matched_dataset_ids),
            "matched_identity_levels": list(self.matched_identity_levels),
            "unresolved_ancestor_ids": list(self.unresolved_ancestor_ids),
            "scope": self.scope,
            "reason": self.reason,
        }
        if include_identity:
            payload["audit_sha256"] = self.audit_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


def _dataset_domain_closure(
    dataset: DatasetLineage,
    known_datasets: Mapping[str, DatasetLineage],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    found: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()
    unresolved: set[str] = set()

    def add_found(dataset_id: str) -> None:
        if dataset_id not in found:
            found.append(dataset_id)

    def visit(current: DatasetLineage) -> None:
        if current.dataset_id in visiting:
            raise ValueError("dataset lineage contains a parent cycle")
        if current.dataset_id in visited:
            return
        visiting.add(current.dataset_id)
        add_found(current.dataset_id)
        for parent_id in current.parent_dataset_ids:
            add_found(parent_id)
            parent = known_datasets.get(parent_id)
            if parent is None:
                unresolved.add(parent_id)
            else:
                visit(parent)
        visiting.remove(current.dataset_id)
        visited.add(current.dataset_id)

    visit(dataset)
    return tuple(found), tuple(sorted(unresolved))


def _identity_overlap(model: ModelLineage, dataset: DatasetLineage) -> tuple[str, ...]:
    model_sets = {
        item.level: set(item.identifiers)
        for item in model.pretraining_identity_sets
        if item.availability is IdentityAvailability.AVAILABLE
    }
    dataset_sets = {
        item.level: set(item.identifiers)
        for item in dataset.identity_sets
        if item.availability is IdentityAvailability.AVAILABLE
    }
    return tuple(
        sorted(
            level
            for level in model_sets.keys() & dataset_sets.keys()
            if model_sets[level] & dataset_sets[level]
        )
    )


def audit_pretraining_overlap(
    model: ModelLineage,
    evaluation_dataset: DatasetLineage,
    *,
    known_datasets: Mapping[str, DatasetLineage] | None = None,
) -> PretrainingOverlapAudit:
    """Assess overlap without turning partial/unknown ancestry into disjointness."""

    known = dict(known_datasets or {})
    known.setdefault(evaluation_dataset.dataset_id, evaluation_dataset)
    evaluation_domains, unresolved = _dataset_domain_closure(evaluation_dataset, known)
    matched_datasets = tuple(sorted(set(evaluation_domains) & set(model.pretraining_dataset_ids)))
    matched_identity_levels = _identity_overlap(model, evaluation_dataset)

    if matched_datasets or matched_identity_levels:
        status = OverlapStatus.OVERLAP_DETECTED
        reason = "pretraining identity intersects evaluation dataset ancestry or declared entity identity"
    elif (
        model.pretraining_lineage_completeness is LineageCompleteness.UNKNOWN
        or evaluation_dataset.lineage_completeness is LineageCompleteness.UNKNOWN
    ):
        status = OverlapStatus.UNKNOWN_LINEAGE
        reason = "model or evaluation dataset lineage is unknown; disjointness cannot be established"
    elif unresolved:
        status = OverlapStatus.POSSIBLE_OVERLAP
        reason = "evaluation ancestry contains unresolved parent datasets; transitive disjointness is not established"
    elif (
        model.pretraining_lineage_completeness is LineageCompleteness.COMPLETE
        and evaluation_dataset.lineage_completeness is LineageCompleteness.COMPLETE
    ):
        status = OverlapStatus.DISJOINT_VERIFIED
        reason = "complete declared pretraining domains and identities are disjoint from evaluation lineage"
    else:
        status = OverlapStatus.POSSIBLE_OVERLAP
        reason = "declared domains do not overlap, but at least one lineage is partial"

    return PretrainingOverlapAudit(
        status=status,
        model_id=model.model_id,
        evaluation_dataset_id=evaluation_dataset.dataset_id,
        model_lineage_sha256=model.lineage_sha256,
        evaluation_dataset_lineage_sha256=evaluation_dataset.lineage_sha256,
        matched_dataset_ids=matched_datasets,
        matched_identity_levels=matched_identity_levels,
        unresolved_ancestor_ids=unresolved,
        reason=reason,
    )
