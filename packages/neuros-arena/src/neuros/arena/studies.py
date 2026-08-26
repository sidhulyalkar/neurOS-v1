"""Cross-domain studies for testing whether synthetic-world anchoring transfers.

These protocols are intentionally dataset-library agnostic. MOABB/BIDS/MNE can
supply de-identified EEG arrays upstream; Arena owns the leakage-resistant study
logic and evidence artifact.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from .reality import anchor_worlds_by_covariance
from .runner import ArenaRun


@dataclass(frozen=True)
class CohortAnchorFold:
    held_out_domain: str
    training_domains: tuple[str, ...]
    cohort_world_weights: dict[str, float]
    held_out_world_distances: dict[str, float]
    weighted_distance: float
    uniform_distance: float
    relative_improvement: float
    best_weighted_world: str
    closest_held_out_world: str


@dataclass(frozen=True)
class CohortAnchorStudy:
    folds: tuple[CohortAnchorFold, ...]
    mean_relative_improvement: float
    median_relative_improvement: float
    fraction_improved: float
    mean_weighted_distance: float
    mean_uniform_distance: float

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.synthetic_bci_arena.cohort_anchor_study.v1",
            "folds": [asdict(fold) for fold in self.folds],
            "summary": {
                "mean_relative_improvement": self.mean_relative_improvement,
                "median_relative_improvement": self.median_relative_improvement,
                "fraction_improved": self.fraction_improved,
                "mean_weighted_distance": self.mean_weighted_distance,
                "mean_uniform_distance": self.mean_uniform_distance,
            },
            "evidence_boundary": (
                "Leave-one-domain-out EEG similarity transfer only. Positive results do not establish human BCI accuracy, "
                "gameplay quality, prevalence, or subject-specific physiological truth."
            ),
        }


def _normalize(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("world weights must be finite non-negative values")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("world weights must have positive mass")
    return values / total


def run_leave_one_domain_out_covariance_study(
    worlds: Mapping[str, ArenaRun],
    domains: Mapping[str, np.ndarray],
    *,
    temperature: float = 1.0,
    shrinkage: float = 1e-3,
) -> CohortAnchorStudy:
    """Test whether world weighting learned on other domains transfers.

    Each fold holds one complete domain out. A domain should normally be a
    participant or participant-session, not a random window. World weights are
    estimated independently for every training domain, averaged, normalized,
    then frozen before the held-out domain is inspected.
    """
    if len(worlds) < 2:
        raise ValueError("study requires at least two synthetic worlds")
    if len(domains) < 3:
        raise ValueError("leave-one-domain-out study requires at least three real domains")
    world_ids = tuple(worlds)
    domain_ids = tuple(domains)
    folds: list[CohortAnchorFold] = []

    for held_out in domain_ids:
        training = tuple(domain_id for domain_id in domain_ids if domain_id != held_out)
        training_weights = []
        for domain_id in training:
            anchor = anchor_worlds_by_covariance(
                worlds,
                domains[domain_id],
                temperature=temperature,
                shrinkage=shrinkage,
            )
            by_world = anchor.by_world()
            training_weights.append([by_world[world_id] for world_id in world_ids])
        cohort = _normalize(np.mean(np.asarray(training_weights, dtype=float), axis=0))

        held_anchor = anchor_worlds_by_covariance(
            worlds,
            domains[held_out],
            temperature=temperature,
            shrinkage=shrinkage,
        )
        held_distances_by_world = {
            world_id: float(distance)
            for world_id, distance in zip(held_anchor.world_ids, held_anchor.distances, strict=True)
        }
        distances = np.asarray([held_distances_by_world[world_id] for world_id in world_ids], dtype=float)
        weighted = float(np.dot(cohort, distances))
        uniform = float(np.mean(distances))
        relative = float((uniform - weighted) / max(abs(uniform), 1e-12))
        folds.append(CohortAnchorFold(
            held_out_domain=held_out,
            training_domains=training,
            cohort_world_weights={world_id: float(value) for world_id, value in zip(world_ids, cohort, strict=True)},
            held_out_world_distances={world_id: held_distances_by_world[world_id] for world_id in world_ids},
            weighted_distance=weighted,
            uniform_distance=uniform,
            relative_improvement=relative,
            best_weighted_world=world_ids[int(np.argmax(cohort))],
            closest_held_out_world=min(world_ids, key=lambda world_id: held_distances_by_world[world_id]),
        ))

    improvements = np.asarray([fold.relative_improvement for fold in folds], dtype=float)
    weighted_distances = np.asarray([fold.weighted_distance for fold in folds], dtype=float)
    uniform_distances = np.asarray([fold.uniform_distance for fold in folds], dtype=float)
    return CohortAnchorStudy(
        folds=tuple(folds),
        mean_relative_improvement=float(np.mean(improvements)),
        median_relative_improvement=float(np.median(improvements)),
        fraction_improved=float(np.mean(improvements > 0)),
        mean_weighted_distance=float(np.mean(weighted_distances)),
        mean_uniform_distance=float(np.mean(uniform_distances)),
    )
