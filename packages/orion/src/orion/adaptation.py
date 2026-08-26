"""Auditable authority for state-changing ORION adaptation.

The adaptation plane is intentionally separate from an algorithm's update rule.
An optimizer, Hebbian rule, decoder, or personalization method may propose and
apply a change, but it does not get to choose the observations used to justify
that change. ``AdaptationAuthority`` freezes those semantics first.

This module is dependency-light and lives in ORION so both learned ORION
personalization and external biologically plausible learning systems can share
the same governance contract without becoming dependencies of one another.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .contracts import AdaptationProposal

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _jsonable(value: Any) -> Any:
    """Convert supported evidence values to deterministic JSON primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("adaptation evidence cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(
        "adaptation evidence must be composed of JSON-compatible primitives, "
        f"NumPy scalars/arrays, mappings, lists, or tuples; got {type(value).__name__}"
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(name: str, value: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return normalized


def _indices(name: str, values: Any) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype == np.bool_:
        raise ValueError(f"{name} must contain integer sample indices, not booleans")
    try:
        integer = array.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer sample indices") from exc
    if not np.array_equal(array, integer):
        raise ValueError(f"{name} must contain integer sample indices")
    result = tuple(int(value) for value in integer.tolist())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} cannot contain negative indices")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate indices")
    return result


def _numeric_evidence(name: str, values: Mapping[str, Any]) -> Mapping[str, float]:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        if not str(key).strip():
            raise ValueError(f"{name} keys must be non-empty")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name}[{key!r}] must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{key!r}] must be finite")
        normalized[str(key)] = number
    return MappingProxyType(normalized)


class AdaptationPhase(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    EVALUATED = "evaluated"
    RETAINED = "retained"
    ROLLED_BACK = "rolled-back"


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Immutable identity of model, representation, or optimizer state."""

    artifact_id: str
    sha256: str
    artifact_type: str = "model-state"
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        if not self.artifact_type.strip():
            raise ValueError("artifact_type must be non-empty")
        object.__setattr__(self, "sha256", _sha256("artifact sha256", self.sha256))
        object.__setattr__(self, "metadata", MappingProxyType(dict(_jsonable(self.metadata))))

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "sha256": self.sha256,
            "version": self.version,
            "metadata": dict(self.metadata),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class AdaptationAuthority:
    """Frozen sample and provenance authority for one adaptation evaluation.

    ``adaptation_indices`` are the only observations allowed to influence the
    proposed state change. ``evaluation_indices`` are immutable and must never
    be supplied to the adaptation rule. Final evaluation must use this complete
    frozen set rather than a favorable subset.
    """

    authority_id: str
    dataset_id: str
    split_unit: str
    adaptation_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    processed_data_sha256: str
    n_samples: int
    protocol_fingerprint: str | None = None
    source_authority_fingerprint: str | None = None
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.authority_id.strip() or not self.dataset_id.strip():
            raise ValueError("authority_id and dataset_id must be non-empty")
        if not self.split_unit.strip():
            raise ValueError("split_unit must be non-empty")
        if (
            isinstance(self.n_samples, bool)
            or not isinstance(self.n_samples, int)
            or self.n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        adaptation = _indices("adaptation_indices", self.adaptation_indices)
        evaluation = _indices("evaluation_indices", self.evaluation_indices)
        overlap = set(adaptation) & set(evaluation)
        if overlap:
            raise ValueError(
                "adaptation and evaluation indices must be disjoint; overlap="
                f"{sorted(overlap)[:8]}"
            )
        if max((*adaptation, *evaluation)) >= self.n_samples:
            raise ValueError("adaptation authority contains out-of-range sample indices")
        object.__setattr__(self, "adaptation_indices", adaptation)
        object.__setattr__(self, "evaluation_indices", evaluation)
        object.__setattr__(
            self,
            "processed_data_sha256",
            _sha256("processed_data_sha256", self.processed_data_sha256),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(_jsonable(self.metadata))))

    @property
    def authority_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "authority_id": self.authority_id,
            "dataset_id": self.dataset_id,
            "split_unit": self.split_unit,
            "adaptation_indices": list(self.adaptation_indices),
            "evaluation_indices": list(self.evaluation_indices),
            "processed_data_sha256": self.processed_data_sha256,
            "n_samples": self.n_samples,
            "protocol_fingerprint": self.protocol_fingerprint,
            "source_authority_fingerprint": self.source_authority_fingerprint,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }
        if include_fingerprint:
            payload["authority_fingerprint"] = self.authority_fingerprint
        return payload

    def require_adaptation_indices(self, values: Any) -> tuple[int, ...]:
        actual = _indices("applied adaptation indices", values)
        if actual != self.adaptation_indices:
            raise ValueError(
                "adaptation must use the exact authority adaptation indices in frozen order"
            )
        return actual

    def require_evaluation_indices(self, values: Any) -> tuple[int, ...]:
        actual = _indices("evaluation indices", values)
        if actual != self.evaluation_indices:
            raise ValueError(
                "evaluation must use the exact frozen evaluation indices; subsets, supersets, "
                "or reordered evaluation samples are not permitted"
            )
        return actual


@dataclass(frozen=True, slots=True)
class GovernedAdaptationProposal:
    """One ordinary ``AdaptationProposal`` bound to immutable authority/state."""

    proposal: AdaptationProposal
    authority_fingerprint: str
    before_artifact: ArtifactIdentity
    adaptation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.authority_fingerprint.strip():
            raise ValueError("authority_fingerprint must be non-empty")
        object.__setattr__(
            self,
            "adaptation_indices",
            _indices("adaptation_indices", self.adaptation_indices),
        )
        # Force deterministic serializability at the governance boundary. The
        # lightweight AdaptationProposal remains backward-compatible.
        _jsonable(dict(self.proposal.changes))
        _jsonable(dict(self.proposal.evidence))

    @classmethod
    def bind(
        cls,
        proposal: AdaptationProposal,
        *,
        authority: AdaptationAuthority,
        before_artifact: ArtifactIdentity,
        adaptation_indices: Any,
    ) -> "GovernedAdaptationProposal":
        indices = authority.require_adaptation_indices(adaptation_indices)
        return cls(
            proposal=proposal,
            authority_fingerprint=authority.authority_fingerprint,
            before_artifact=before_artifact,
            adaptation_indices=indices,
        )

    @property
    def proposal_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "phase": AdaptationPhase.PROPOSED.value,
            "authority_fingerprint": self.authority_fingerprint,
            "before_artifact": self.before_artifact.to_dict(),
            "adaptation_indices": list(self.adaptation_indices),
            "proposal": {
                "reason": self.proposal.reason,
                "changes": dict(self.proposal.changes),
                "evidence": dict(self.proposal.evidence),
                "requires_approval": self.proposal.requires_approval,
            },
        }
        if include_fingerprint:
            payload["proposal_fingerprint"] = self.proposal_fingerprint
        return _jsonable(payload)


@dataclass(frozen=True, slots=True)
class AdaptationDecision:
    """Approval or rejection of a governed proposal."""

    proposal_fingerprint: str
    authority_fingerprint: str
    phase: AdaptationPhase
    actor: str
    reason: str

    def __post_init__(self) -> None:
        if self.phase not in {AdaptationPhase.APPROVED, AdaptationPhase.REJECTED}:
            raise ValueError("decision phase must be approved or rejected")
        if not self.proposal_fingerprint.strip() or not self.authority_fingerprint.strip():
            raise ValueError("proposal_fingerprint and authority_fingerprint must be non-empty")
        if not self.actor.strip() or not self.reason.strip():
            raise ValueError("decision actor and reason must be non-empty")

    @classmethod
    def approve(
        cls,
        governed: GovernedAdaptationProposal,
        *,
        actor: str,
        reason: str,
    ) -> "AdaptationDecision":
        return cls(
            proposal_fingerprint=governed.proposal_fingerprint,
            authority_fingerprint=governed.authority_fingerprint,
            phase=AdaptationPhase.APPROVED,
            actor=actor,
            reason=reason,
        )

    @classmethod
    def reject(
        cls,
        governed: GovernedAdaptationProposal,
        *,
        actor: str,
        reason: str,
    ) -> "AdaptationDecision":
        return cls(
            proposal_fingerprint=governed.proposal_fingerprint,
            authority_fingerprint=governed.authority_fingerprint,
            phase=AdaptationPhase.REJECTED,
            actor=actor,
            reason=reason,
        )

    @property
    def decision_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "phase": self.phase.value,
            "proposal_fingerprint": self.proposal_fingerprint,
            "authority_fingerprint": self.authority_fingerprint,
            "actor": self.actor,
            "reason": self.reason,
        }
        if include_fingerprint:
            payload["decision_fingerprint"] = self.decision_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class AdaptationApplication:
    """Evidence that an approved mutation was applied to authorized data only."""

    proposal_fingerprint: str
    authority_fingerprint: str
    decision_fingerprint: str
    before_artifact: ArtifactIdentity
    after_artifact: ArtifactIdentity
    adaptation_indices: tuple[int, ...]
    update_evidence: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.before_artifact.sha256 == self.after_artifact.sha256:
            raise ValueError("an applied adaptation must change the artifact SHA-256 identity")
        object.__setattr__(
            self,
            "adaptation_indices",
            _indices("adaptation_indices", self.adaptation_indices),
        )
        object.__setattr__(
            self,
            "update_evidence",
            _numeric_evidence("update_evidence", self.update_evidence),
        )

    @classmethod
    def record(
        cls,
        governed: GovernedAdaptationProposal,
        decision: AdaptationDecision,
        *,
        authority: AdaptationAuthority,
        after_artifact: ArtifactIdentity,
        adaptation_indices: Any,
        update_evidence: Mapping[str, float] | None = None,
    ) -> "AdaptationApplication":
        if decision.phase is not AdaptationPhase.APPROVED:
            raise ValueError("a rejected adaptation cannot be applied")
        if governed.authority_fingerprint != authority.authority_fingerprint:
            raise ValueError("governed proposal authority does not match adaptation authority")
        if governed.adaptation_indices != authority.adaptation_indices:
            raise ValueError("governed proposal adaptation indices differ from frozen authority")
        if decision.proposal_fingerprint != governed.proposal_fingerprint:
            raise ValueError("decision does not belong to governed proposal")
        if decision.authority_fingerprint != governed.authority_fingerprint:
            raise ValueError("decision authority does not match governed proposal authority")
        if decision.authority_fingerprint != authority.authority_fingerprint:
            raise ValueError("decision authority does not match adaptation authority")
        indices = authority.require_adaptation_indices(adaptation_indices)
        return cls(
            proposal_fingerprint=governed.proposal_fingerprint,
            authority_fingerprint=authority.authority_fingerprint,
            decision_fingerprint=decision.decision_fingerprint,
            before_artifact=governed.before_artifact,
            after_artifact=after_artifact,
            adaptation_indices=indices,
            update_evidence={} if update_evidence is None else update_evidence,
        )

    @property
    def application_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "phase": AdaptationPhase.APPLIED.value,
            "proposal_fingerprint": self.proposal_fingerprint,
            "authority_fingerprint": self.authority_fingerprint,
            "decision_fingerprint": self.decision_fingerprint,
            "before_artifact": self.before_artifact.to_dict(),
            "after_artifact": self.after_artifact.to_dict(),
            "adaptation_indices": list(self.adaptation_indices),
            "update_evidence": dict(self.update_evidence),
        }
        if include_fingerprint:
            payload["application_fingerprint"] = self.application_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class AdaptationEvaluation:
    """Held-out evaluation of one applied adaptation under frozen authority."""

    application_fingerprint: str
    authority_fingerprint: str
    evaluation_indices: tuple[int, ...]
    metrics_before: Mapping[str, float]
    metrics_after: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_indices",
            _indices("evaluation_indices", self.evaluation_indices),
        )
        object.__setattr__(
            self,
            "metrics_before",
            _numeric_evidence("metrics_before", self.metrics_before),
        )
        object.__setattr__(
            self,
            "metrics_after",
            _numeric_evidence("metrics_after", self.metrics_after),
        )
        if set(self.metrics_before) != set(self.metrics_after):
            raise ValueError("metrics_before and metrics_after must contain the same metric names")
        if not self.metrics_before:
            raise ValueError("adaptation evaluation requires at least one metric")

    @classmethod
    def record(
        cls,
        application: AdaptationApplication,
        *,
        authority: AdaptationAuthority,
        evaluation_indices: Any,
        metrics_before: Mapping[str, float],
        metrics_after: Mapping[str, float],
    ) -> "AdaptationEvaluation":
        if application.authority_fingerprint != authority.authority_fingerprint:
            raise ValueError("application authority does not match evaluation authority")
        authority.require_adaptation_indices(application.adaptation_indices)
        indices = authority.require_evaluation_indices(evaluation_indices)
        return cls(
            application_fingerprint=application.application_fingerprint,
            authority_fingerprint=authority.authority_fingerprint,
            evaluation_indices=indices,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )

    @property
    def evaluation_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "phase": AdaptationPhase.EVALUATED.value,
            "application_fingerprint": self.application_fingerprint,
            "authority_fingerprint": self.authority_fingerprint,
            "evaluation_indices": list(self.evaluation_indices),
            "metrics_before": dict(self.metrics_before),
            "metrics_after": dict(self.metrics_after),
        }
        if include_fingerprint:
            payload["evaluation_fingerprint"] = self.evaluation_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class AdaptationOutcome:
    """Final retain-or-rollback decision after held-out evaluation."""

    evaluation_fingerprint: str
    authority_fingerprint: str
    phase: AdaptationPhase
    before_artifact: ArtifactIdentity
    after_artifact: ArtifactIdentity
    active_artifact: ArtifactIdentity
    actor: str
    reason: str

    def __post_init__(self) -> None:
        if self.phase not in {AdaptationPhase.RETAINED, AdaptationPhase.ROLLED_BACK}:
            raise ValueError("outcome phase must be retained or rolled-back")
        if not self.actor.strip() or not self.reason.strip():
            raise ValueError("outcome actor and reason must be non-empty")
        expected = (
            self.after_artifact
            if self.phase is AdaptationPhase.RETAINED
            else self.before_artifact
        )
        if self.active_artifact.sha256 != expected.sha256:
            if self.phase is AdaptationPhase.ROLLED_BACK:
                raise ValueError("rollback must restore the exact pre-adaptation artifact SHA-256")
            raise ValueError("retained outcome must keep the exact post-adaptation artifact SHA-256")

    @classmethod
    def retain(
        cls,
        application: AdaptationApplication,
        evaluation: AdaptationEvaluation,
        *,
        actor: str,
        reason: str,
    ) -> "AdaptationOutcome":
        if evaluation.application_fingerprint != application.application_fingerprint:
            raise ValueError("evaluation does not belong to adaptation application")
        if evaluation.authority_fingerprint != application.authority_fingerprint:
            raise ValueError("evaluation authority does not match adaptation application")
        return cls(
            evaluation_fingerprint=evaluation.evaluation_fingerprint,
            authority_fingerprint=application.authority_fingerprint,
            phase=AdaptationPhase.RETAINED,
            before_artifact=application.before_artifact,
            after_artifact=application.after_artifact,
            active_artifact=application.after_artifact,
            actor=actor,
            reason=reason,
        )

    @classmethod
    def rollback(
        cls,
        application: AdaptationApplication,
        evaluation: AdaptationEvaluation,
        *,
        restored_artifact: ArtifactIdentity,
        actor: str,
        reason: str,
    ) -> "AdaptationOutcome":
        if evaluation.application_fingerprint != application.application_fingerprint:
            raise ValueError("evaluation does not belong to adaptation application")
        if evaluation.authority_fingerprint != application.authority_fingerprint:
            raise ValueError("evaluation authority does not match adaptation application")
        return cls(
            evaluation_fingerprint=evaluation.evaluation_fingerprint,
            authority_fingerprint=application.authority_fingerprint,
            phase=AdaptationPhase.ROLLED_BACK,
            before_artifact=application.before_artifact,
            after_artifact=application.after_artifact,
            active_artifact=restored_artifact,
            actor=actor,
            reason=reason,
        )

    @property
    def outcome_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "phase": self.phase.value,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "authority_fingerprint": self.authority_fingerprint,
            "before_artifact": self.before_artifact.to_dict(),
            "after_artifact": self.after_artifact.to_dict(),
            "active_artifact": self.active_artifact.to_dict(),
            "actor": self.actor,
            "reason": self.reason,
        }
        if include_fingerprint:
            payload["outcome_fingerprint"] = self.outcome_fingerprint
        return payload
