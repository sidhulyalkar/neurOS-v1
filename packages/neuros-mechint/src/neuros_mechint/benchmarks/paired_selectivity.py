"""Validation-only paired causal selectivity audit (experimental).

A selective effect is a difference-in-differences, not a raw logit margin
or a generic intervention-induced drop. No model weights are loaded here.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any

from neuros_mechint.core.manifest import stable_hash

SELECTIVITY_SCHEMA = "neuros-mechint.paired-selectivity.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _targets(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    items = tuple(values)
    if not items or any(not isinstance(x, str) or not x.strip() for x in items):
        raise ValueError(f"{name} must contain nonempty target names")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must not contain duplicate targets")
    return items


@dataclass(frozen=True, slots=True)
class PairedSelectivityDesign:
    """Frozen *declaration*, not proof of independent preregistration.

    Pair hashes bind exact semantic trial content, as computed before model
    inference. A separate, externally verifiable discovery commitment is
    necessary before using this diagnostic in a confirmatory study.
    """

    design_id: str
    model_revision: str
    discovery_commitment: str
    candidate_id: str
    candidate_targets: tuple[str, ...]
    magnitude_id: str
    magnitude_targets: tuple[str, ...]
    random_controls: tuple[tuple[str, tuple[str, ...]], ...]
    validation_pair_hashes: tuple[tuple[str, str], ...]
    control_condition: str = "reordered"
    min_validation_pairs: int = 8
    min_random_controls: int = 24
    min_positive_fraction: float = 0.75
    bootstrap_samples: int = 1000
    alpha: float = 0.05
    seed: int = 37

    def __post_init__(self) -> None:
        for name in ("design_id", "model_revision", "candidate_id", "magnitude_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64}|synthetic:[A-Za-z0-9][A-Za-z0-9._-]*)",
            self.model_revision,
        ) is None:
            raise ValueError("model_revision must be a pinned commit/hash or synthetic: fixture")
        if not isinstance(self.discovery_commitment, str) or not _SHA256.fullmatch(
            self.discovery_commitment
        ):
            raise ValueError("discovery_commitment must be a 64-hex SHA-256 digest")
        if self.control_condition not in ("reordered", "novel"):
            raise ValueError("control_condition must be reordered or novel")
        candidate = _targets(self.candidate_targets, "candidate_targets")
        magnitude = _targets(self.magnitude_targets, "magnitude_targets")
        if len(candidate) != len(magnitude) or set(candidate) == set(magnitude):
            raise ValueError("magnitude baseline must be a distinct same-size circuit")
        controls = tuple(
            (name, _targets(tuple(targets), f"random control {name}"))
            for name, targets in self.random_controls
        )
        circuits = ((self.candidate_id, candidate), (self.magnitude_id, magnitude), *controls)
        names = [name for name, _ in circuits]
        if len(names) != len(set(names)) or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError("circuit IDs must be unique and nonempty")
        signatures = [frozenset(targets) for _, targets in circuits]
        if len(signatures) != len(set(signatures)):
            raise ValueError("candidate, magnitude and random target sets must be distinct")
        if any(len(targets) != len(candidate) for _, targets in circuits):
            raise ValueError("every random control must have the same cardinality")
        pairs = tuple((pair_id, digest) for pair_id, digest in self.validation_pair_hashes)
        if not pairs:
            raise ValueError("validation_pair_hashes must not be empty")
        if any(
            not isinstance(pair_id, str) or not pair_id.strip()
            or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            for pair_id, digest in pairs
        ):
            raise ValueError("validation pair IDs and full-content hashes are required")
        if len({pair_id for pair_id, _ in pairs}) != len(pairs):
            raise ValueError("duplicate validation pair ID")
        if len({digest for _, digest in pairs}) != len(pairs):
            raise ValueError("duplicate validation pair content")
        if self.min_validation_pairs < 2 or self.min_random_controls < 1:
            raise ValueError("minimum pairs must be >=2 and controls >=1")
        if not 0 < self.min_positive_fraction <= 1 or not 0 < self.alpha < 1:
            raise ValueError("fractions and alpha must be inside (0,1]")
        if self.bootstrap_samples < 100 or isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("bootstrap_samples must be >=100 and seed an integer")
        object.__setattr__(self, "candidate_targets", candidate)
        object.__setattr__(self, "magnitude_targets", magnitude)
        object.__setattr__(self, "random_controls", controls)
        object.__setattr__(self, "validation_pair_hashes", pairs)


@dataclass(frozen=True, slots=True)
class PairedSelectivityObservation:
    pair_id: str
    pair_content_hash: str
    circuit_id: str
    intact_repeated: float
    intact_control: float
    intervened_repeated: float
    intervened_control: float
    split: str = "validation"

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("discovery observations cannot enter held-out selectivity evaluation")
        if not self.pair_id or not self.circuit_id:
            raise ValueError("pair_id and circuit_id must be nonempty")
        if not isinstance(self.pair_content_hash, str) or not _SHA256.fullmatch(
            self.pair_content_hash
        ):
            raise ValueError("pair_content_hash must be a full-content SHA-256 digest")
        for key in (
            "intact_repeated", "intact_control",
            "intervened_repeated", "intervened_control",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"nonfinite or nonnumeric intervention score: {key}")


def _finite_difference(left: float, right: float, *, label: str) -> float:
    """Reject derived overflow instead of serializing nonfinite causal effects."""
    result = left - right
    if not math.isfinite(result):
        raise ValueError(f"nonfinite derived selectivity: {label}")
    return result


def _finite_mean(values: list[float], *, label: str) -> float:
    result = mean(values)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite aggregate selectivity: {label}")
    return result


def _bootstrap(values: list[float], *, samples: int, alpha: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        _finite_mean(rng.choices(values, k=n), label="bootstrap")
        for _ in range(samples)
    )
    lower = int((alpha / 2) * (samples - 1))
    upper = int((1 - alpha / 2) * (samples - 1))
    return [estimates[lower], estimates[upper]]


def evaluate_paired_selectivity(
    design: PairedSelectivityDesign,
    observations: tuple[PairedSelectivityObservation, ...],
) -> dict[str, Any]:
    """Score a single preregistered candidate on complete matched held-out pairs.

    All candidate, magnitude and random circuits must be run against *exactly*
    the same validation pairs. Missing observations produce nonestimable
    diagnostics; extra, duplicated or mismatched observations fail closed.
    A supported result is diagnostic only, never a promoted causal claim.
    """
    obs = tuple(observations)
    expected_pairs = dict(design.validation_pair_hashes)
    control_names = [name for name, _ in design.random_controls]
    circuits = [design.candidate_id, design.magnitude_id, *control_names]
    expected_keys = {(pair_id, circuit_id) for pair_id in expected_pairs for circuit_id in circuits}
    actual: dict[tuple[str, str], PairedSelectivityObservation] = {}
    for row in obs:
        key = (row.pair_id, row.circuit_id)
        if key in actual:
            raise ValueError(f"duplicate matched intervention observation: {key}")
        if key not in expected_keys:
            raise ValueError(f"unplanned circuit/pair observation: {key}")
        if row.pair_content_hash != expected_pairs[row.pair_id]:
            raise ValueError(f"validation pair content hash mismatch: {row.pair_id}")
        actual[key] = row

    base = {
        "schema_version": SELECTIVITY_SCHEMA,
        "design_fingerprint": stable_hash(design),
        "control_condition": design.control_condition,
        "validation_only": True,
        "n_pairs_planned": len(expected_pairs),
        "n_random_controls_planned": len(control_names),
        "discovery_commitment": design.discovery_commitment,
        "preregistration_independently_verified": False,
        "real_model_evidence": False,
    }

    def finish(details: dict[str, Any]) -> dict[str, Any]:
        result = {**base, **details}
        result["result_hash"] = stable_hash(result)
        return result

    missing = sorted(expected_keys - actual.keys())
    if missing:
        return finish({
            "status": "NONESTIMABLE_MISSING_OBSERVATIONS",
            "n_observations_received": len(actual),
            "missing": [{"pair_id": p, "circuit_id": c} for p, c in missing],
            "supports_selective_effect": False,
        })

    for pair_id in expected_pairs:
        baseline = actual[(pair_id, design.candidate_id)]
        for circuit_id in circuits[1:]:
            row = actual[(pair_id, circuit_id)]
            for key in ("intact_repeated", "intact_control"):
                if not math.isclose(getattr(row, key), getattr(baseline, key), rel_tol=0, abs_tol=1e-8):
                    raise ValueError(f"intact baseline drift for pair {pair_id}: {circuit_id}")

    def did(row: PairedSelectivityObservation) -> float:
        intact = _finite_difference(
            row.intact_repeated, row.intact_control, label="intact contrast"
        )
        intervened = _finite_difference(
            row.intervened_repeated, row.intervened_control, label="intervened contrast"
        )
        return _finite_difference(intact, intervened, label="paired difference-in-differences")

    candidate_rows = [actual[(p, design.candidate_id)] for p in expected_pairs]
    magnitude_rows = [actual[(p, design.magnitude_id)] for p in expected_pairs]
    baseline_advantage = [
        _finite_difference(
            r.intact_repeated, r.intact_control, label="intact repetition advantage"
        )
        for r in candidate_rows
    ]
    candidate_effect = [did(r) for r in candidate_rows]
    magnitude_effect = [did(r) for r in magnitude_rows]
    magnitude_difference = [
        _finite_difference(a, b, label="candidate vs magnitude control")
        for a, b in zip(candidate_effect, magnitude_effect)
    ]
    random_effects = [
        _finite_mean(
            [did(actual[(p, circuit_id)]) for p in expected_pairs],
            label=f"random control {circuit_id}",
        )
        for circuit_id in control_names
    ]
    candidate_mean = _finite_mean(candidate_effect, label="candidate causal effect")
    # A descriptive, plus-one rank against declared circuits. This is NOT
    # a calibrated randomization p-value without an independently justified
    # exchangeable null/control-sampling procedure.
    rank_fraction = (1 + sum(value >= candidate_mean for value in random_effects)) / (
        1 + len(random_effects)
    )
    baseline_ci = _bootstrap(
        baseline_advantage, samples=design.bootstrap_samples, alpha=design.alpha, seed=design.seed
    )
    candidate_ci = _bootstrap(
        candidate_effect, samples=design.bootstrap_samples, alpha=design.alpha, seed=design.seed + 1
    )
    magnitude_ci = _bootstrap(
        magnitude_difference, samples=design.bootstrap_samples, alpha=design.alpha, seed=design.seed + 2
    )
    behavior_ok = (
        _finite_mean(baseline_advantage, label="behavioral advantage") > 0
        and baseline_ci[0] > 0
        and mean(float(r.intact_repeated > 0) for r in candidate_rows)
        >= design.min_positive_fraction
        and mean(float(x > 0) for x in baseline_advantage)
        >= design.min_positive_fraction
    )
    coverage_ok = (
        len(candidate_rows) >= design.min_validation_pairs
        and len(random_effects) >= design.min_random_controls
    )
    selective_ok = candidate_mean > 0 and candidate_ci[0] > 0
    controls_ok = rank_fraction <= design.alpha and magnitude_ci[0] > 0
    supports = (
        design.control_condition == "reordered"
        and coverage_ok and behavior_ok and selective_ok and controls_ok
    )
    if not coverage_ok:
        status = "INSUFFICIENT_COVERAGE"
    elif not behavior_ok:
        status = "REJECTED_BEHAVIORAL_COMPETENCE"
    elif not selective_ok:
        status = "REJECTED_NO_SELECTIVE_EFFECT"
    elif not controls_ok:
        status = "REJECTED_CONTROLS"
    elif design.control_condition != "reordered":
        status = "SENSITIVITY_ONLY_NOT_PRIMARY"
    else:
        status = "SUPPORTED_DIAGNOSTIC_ONLY"
    return finish({
        "status": status,
        "supports_selective_effect": supports,
        "n_pairs": len(candidate_rows),
        "n_random_controls": len(random_effects),
        "fraction_positive_repeated_margin": mean(
            float(r.intact_repeated > 0) for r in candidate_rows
        ),
        "fraction_positive_repetition_advantage": mean(
            float(x > 0) for x in baseline_advantage
        ),
        "mean_intact_repetition_advantage": _finite_mean(
            baseline_advantage, label="behavioral advantage"
        ),
        "ci_intact_repetition_advantage": baseline_ci,
        "mean_candidate_difference_in_differences": candidate_mean,
        "ci_candidate_difference_in_differences": candidate_ci,
        "mean_magnitude_difference_in_differences": _finite_mean(
            magnitude_effect, label="magnitude effect"
        ),
        "ci_candidate_advantage_vs_magnitude": magnitude_ci,
        "random_control_mean_effects": [
            {"circuit_id": circuit_id, "mean_effect": effect}
            for circuit_id, effect in zip(control_names, random_effects)
        ],
        "random_control_plus_one_rank_fraction": rank_fraction,
        "random_control_exchangeability_verified": False,
        "random_control_rank_is_calibrated_p_value": False,
        "pair_evidence": [
            {"pair_id": pair_id, "candidate_effect": a, "magnitude_effect": b}
            for pair_id, a, b in zip(expected_pairs, candidate_effect, magnitude_effect)
        ],
    })
