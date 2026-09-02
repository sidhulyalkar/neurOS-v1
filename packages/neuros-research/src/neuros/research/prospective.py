"""Prospective geometry-to-decoding screening with a cryptographic reveal boundary."""

from __future__ import annotations

import itertools
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from ._canonical import canonical_sha256, require_nonempty, require_sha256

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MIN_CANDIDATES = 5
_EXACT_PERMUTATION_MAX_N = 8
_MONTE_CARLO_PERMUTATIONS = 10_000


def _require_git_sha(value: str, *, name: str) -> str:
    normalized = require_nonempty(value, name=name).lower()
    if not _GIT_SHA_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be an exact 40-character lowercase git SHA")
    return normalized


def _finite(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda row: (row[1], row[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        stop = start + 1
        value = indexed[start][1]
        while stop < len(indexed) and indexed[stop][1] == value:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for offset in range(start, stop):
            ranks[indexed[offset][0]] = average_rank
        start = stop
    return tuple(ranks)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation vectors must have equal length >= 2")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    denominator = left_norm * right_norm
    if denominator <= 0.0:
        raise ValueError("Spearman correlation is undefined for a constant vector")
    correlation = numerator / denominator
    return max(-1.0, min(1.0, correlation))


def _spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _verify_artifact_fingerprint(payload: dict[str, Any], *, name: str) -> dict[str, Any]:
    fingerprint = require_sha256(str(payload.get("fingerprint", "")), name=f"{name} fingerprint")
    unsigned = dict(payload)
    unsigned.pop("fingerprint", None)
    if canonical_sha256(unsigned) != fingerprint:
        raise ValueError(f"{name} fingerprint mismatch")
    return unsigned


def _permutation_evidence(
    geometry: tuple[float, ...],
    outcome: tuple[float, ...],
    *,
    observed: float,
    seed_material: str,
) -> dict[str, Any]:
    geometry_ranks = _average_ranks(geometry)
    outcome_ranks = _average_ranks(outcome)
    tolerance = 1e-12
    if len(outcome) <= _EXACT_PERMUTATION_MAX_N:
        extreme = 0
        total = 0
        for permutation in itertools.permutations(outcome_ranks):
            total += 1
            if _pearson(geometry_ranks, tuple(permutation)) >= observed - tolerance:
                extreme += 1
        return {
            "mode": "exact_label_permutation",
            "alternative": "positive_association",
            "permutations": total,
            "extreme_count": extreme,
            "p_value": extreme / total,
        }

    seed_sha256 = canonical_sha256({"seed_material": seed_material})
    rng = random.Random(int(seed_sha256[:16], 16))
    indices = list(range(len(outcome_ranks)))
    extreme = 0
    for _ in range(_MONTE_CARLO_PERMUTATIONS):
        rng.shuffle(indices)
        permuted = tuple(outcome_ranks[index] for index in indices)
        if _pearson(geometry_ranks, permuted) >= observed - tolerance:
            extreme += 1
    return {
        "mode": "deterministic_monte_carlo_label_permutation",
        "alternative": "positive_association",
        "permutations": _MONTE_CARLO_PERMUTATIONS,
        "extreme_count": extreme,
        "p_value": (extreme + 1) / (_MONTE_CARLO_PERMUTATIONS + 1),
        "seed_sha256": seed_sha256,
    }


@dataclass(frozen=True, slots=True)
class ProspectiveGeometryCandidate:
    """One candidate whose geometry score is frozen before decoding outcomes are revealed."""

    candidate_id: str
    geometry_score: float
    geometry_evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            require_nonempty(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self,
            "geometry_score",
            _finite(self.geometry_score, name="geometry_score"),
        )
        object.__setattr__(
            self,
            "geometry_evidence_sha256",
            require_sha256(
                self.geometry_evidence_sha256,
                name="geometry_evidence_sha256",
            ),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProspectiveGeometryCandidate:
        return cls(
            candidate_id=str(payload["candidate_id"]),
            geometry_score=float(payload["geometry_score"]),
            geometry_evidence_sha256=str(payload["geometry_evidence_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "geometry_score": self.geometry_score,
            "geometry_evidence_sha256": self.geometry_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProspectiveGeometryPlan:
    """Immutable pre-reveal screening plan for one declared representation candidate set."""

    plan_id: str
    candidates: tuple[ProspectiveGeometryCandidate, ...]
    geometry_metric: str
    evaluation_fingerprint: str
    split_fingerprint: str
    preprocessing_fingerprint: str
    source_revision: str
    outcome_metric: str = "validation_pearson_delta"

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", require_nonempty(self.plan_id, name="plan_id"))
        candidates = tuple(sorted(self.candidates, key=lambda row: row.candidate_id))
        if len(candidates) < _MIN_CANDIDATES:
            raise ValueError(
                f"prospective screening requires at least {_MIN_CANDIDATES} predeclared candidates"
            )
        ids = tuple(row.candidate_id for row in candidates)
        if len(set(ids)) != len(ids):
            raise ValueError("prospective candidate IDs must be unique")
        if len({row.geometry_score for row in candidates}) < 2:
            raise ValueError("prospective geometry scores must not be constant")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "geometry_metric",
            require_nonempty(self.geometry_metric, name="geometry_metric"),
        )
        if self.outcome_metric != "validation_pearson_delta":
            raise ValueError(
                "prospective geometry screening currently requires validation_pearson_delta outcomes"
            )
        for name in (
            "evaluation_fingerprint",
            "split_fingerprint",
            "preprocessing_fingerprint",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "source_revision",
            _require_git_sha(self.source_revision, name="source_revision"),
        )

    @property
    def candidate_set_sha256(self) -> str:
        return canonical_sha256([row.candidate_id for row in self.candidates])

    @property
    def geometry_projection_sha256(self) -> str:
        return canonical_sha256([row.to_dict() for row in self.candidates])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "neuros_prospective_geometry_plan",
            "schema_version": 1,
            "plan_id": self.plan_id,
            "candidates": [row.to_dict() for row in self.candidates],
            "candidate_set_sha256": self.candidate_set_sha256,
            "geometry_projection_sha256": self.geometry_projection_sha256,
            "geometry_metric": self.geometry_metric,
            "outcome_metric": self.outcome_metric,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "source_revision": self.source_revision,
            "reveal_boundary": (
                "Geometry scores, candidate membership, evaluator identity, split identity, and "
                "preprocessing identity are frozen by this fingerprint before outcome reveal."
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_artifact(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_artifact(cls, payload: dict[str, Any]) -> ProspectiveGeometryPlan:
        if not isinstance(payload, dict):
            raise TypeError("prospective geometry plan artifact must be an object")
        unsigned = _verify_artifact_fingerprint(payload, name="prospective geometry plan")
        if unsigned.get("kind") != "neuros_prospective_geometry_plan":
            raise ValueError("unexpected prospective geometry plan kind")
        if unsigned.get("schema_version") != 1:
            raise ValueError("unsupported prospective geometry plan schema version")
        rows = unsigned.get("candidates")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("prospective geometry plan candidates must be a list of objects")
        plan = cls(
            plan_id=str(unsigned["plan_id"]),
            candidates=tuple(ProspectiveGeometryCandidate.from_dict(row) for row in rows),
            geometry_metric=str(unsigned["geometry_metric"]),
            outcome_metric=str(unsigned["outcome_metric"]),
            evaluation_fingerprint=str(unsigned["evaluation_fingerprint"]),
            split_fingerprint=str(unsigned["split_fingerprint"]),
            preprocessing_fingerprint=str(unsigned["preprocessing_fingerprint"]),
            source_revision=str(unsigned["source_revision"]),
        )
        if unsigned != plan.to_dict():
            raise ValueError("prospective geometry plan contains non-canonical or altered fields")
        return plan


@dataclass(frozen=True, slots=True)
class ProspectiveOutcome:
    """One subsequently revealed development outcome bound to independent evidence."""

    candidate_id: str
    validation_pearson_delta: float
    outcome_evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            require_nonempty(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self,
            "validation_pearson_delta",
            _finite(self.validation_pearson_delta, name="validation_pearson_delta"),
        )
        object.__setattr__(
            self,
            "outcome_evidence_sha256",
            require_sha256(
                self.outcome_evidence_sha256,
                name="outcome_evidence_sha256",
            ),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProspectiveOutcome:
        return cls(
            candidate_id=str(payload["candidate_id"]),
            validation_pearson_delta=float(payload["validation_pearson_delta"]),
            outcome_evidence_sha256=str(payload["outcome_evidence_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "validation_pearson_delta": self.validation_pearson_delta,
            "outcome_evidence_sha256": self.outcome_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProspectiveGeometryReveal:
    """Outcome reveal that can be evaluated only against one exact frozen plan."""

    plan_fingerprint: str
    outcomes: tuple[ProspectiveOutcome, ...]
    source_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_fingerprint",
            require_sha256(self.plan_fingerprint, name="plan_fingerprint"),
        )
        outcomes = tuple(sorted(self.outcomes, key=lambda row: row.candidate_id))
        ids = tuple(row.candidate_id for row in outcomes)
        if not outcomes or len(set(ids)) != len(ids):
            raise ValueError("prospective reveal outcomes must be non-empty with unique candidate IDs")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(
            self,
            "source_revision",
            _require_git_sha(self.source_revision, name="source_revision"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "neuros_prospective_geometry_reveal",
            "schema_version": 1,
            "plan_fingerprint": self.plan_fingerprint,
            "outcomes": [row.to_dict() for row in self.outcomes],
            "source_revision": self.source_revision,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_artifact(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_artifact(cls, payload: dict[str, Any]) -> ProspectiveGeometryReveal:
        if not isinstance(payload, dict):
            raise TypeError("prospective geometry reveal artifact must be an object")
        unsigned = _verify_artifact_fingerprint(payload, name="prospective geometry reveal")
        if unsigned.get("kind") != "neuros_prospective_geometry_reveal":
            raise ValueError("unexpected prospective geometry reveal kind")
        if unsigned.get("schema_version") != 1:
            raise ValueError("unsupported prospective geometry reveal schema version")
        rows = unsigned.get("outcomes")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("prospective geometry reveal outcomes must be a list of objects")
        reveal = cls(
            plan_fingerprint=str(unsigned["plan_fingerprint"]),
            outcomes=tuple(ProspectiveOutcome.from_dict(row) for row in rows),
            source_revision=str(unsigned["source_revision"]),
        )
        if unsigned != reveal.to_dict():
            raise ValueError("prospective geometry reveal contains non-canonical or altered fields")
        return reveal


def evaluate_prospective_geometry_gain(
    plan: ProspectiveGeometryPlan,
    reveal: ProspectiveGeometryReveal,
) -> dict[str, Any]:
    """Evaluate the frozen geometry screen after an exact candidate-set outcome reveal."""

    if reveal.plan_fingerprint != plan.fingerprint:
        raise ValueError("prospective reveal does not reference the exact frozen plan fingerprint")
    plan_ids = tuple(row.candidate_id for row in plan.candidates)
    outcome_ids = tuple(row.candidate_id for row in reveal.outcomes)
    if outcome_ids != plan_ids:
        missing = sorted(set(plan_ids) - set(outcome_ids))
        extra = sorted(set(outcome_ids) - set(plan_ids))
        raise ValueError(
            "prospective reveal candidate set mismatch "
            f"(missing={missing}, extra={extra})"
        )

    geometry = tuple(row.geometry_score for row in plan.candidates)
    outcome = tuple(row.validation_pearson_delta for row in reveal.outcomes)
    observed = _spearman(geometry, outcome)
    permutation = _permutation_evidence(
        geometry,
        outcome,
        observed=observed,
        seed_material=f"{plan.fingerprint}:{reveal.fingerprint}",
    )

    leave_one_out: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(plan_ids):
        reduced_geometry = geometry[:index] + geometry[index + 1 :]
        reduced_outcome = outcome[:index] + outcome[index + 1 :]
        try:
            rho = _spearman(reduced_geometry, reduced_outcome)
        except ValueError:
            rho = None
        leave_one_out.append({"excluded_candidate_id": candidate_id, "rho": rho})
    defined_loo = [row["rho"] for row in leave_one_out if row["rho"] is not None]
    stability = {
        "defined_count": len(defined_loo),
        "all_positive": bool(defined_loo) and all(value > 0.0 for value in defined_loo),
        "minimum_rho": min(defined_loo) if defined_loo else None,
        "maximum_rho": max(defined_loo) if defined_loo else None,
        "leave_one_out": leave_one_out,
    }

    result: dict[str, Any] = {
        "kind": "neuros_prospective_geometry_gain_evidence",
        "schema_version": 1,
        "metric_name": "prospective_geometry_gain_spearman",
        "metric_value": observed,
        "candidate_count": len(plan.candidates),
        "plan_fingerprint": plan.fingerprint,
        "reveal_fingerprint": reveal.fingerprint,
        "candidate_set_sha256": plan.candidate_set_sha256,
        "geometry_projection_sha256": plan.geometry_projection_sha256,
        "evaluation_fingerprint": plan.evaluation_fingerprint,
        "split_fingerprint": plan.split_fingerprint,
        "preprocessing_fingerprint": plan.preprocessing_fingerprint,
        "permutation_evidence": permutation,
        "leave_one_out_stability": stability,
        "scientific_boundary": (
            "This result measures a development-only prospective association across the exact "
            "predeclared candidate set. The plan fingerprint cryptographically binds the geometry "
            "screen before reveal, but does not by itself prove wall-clock ordering or authorize "
            "G2/G3/G4 outcome access."
        ),
    }
    result["fingerprint"] = canonical_sha256(result)
    return result
