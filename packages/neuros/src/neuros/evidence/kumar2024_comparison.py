"""Preregistered comparison authority for promoted Kumar2024 NSQ studies.

This module is intentionally independent of model execution. It freezes the
stochastic/repeated-measure axes and the aggregation order that a promoted
comparison is allowed to use before any all-subject neural final-assessment
result is generated.

The independent population unit is always participant. Target session, split
seed, neural optimization seed, and calibration budget are repeated-measure or
algorithmic axes and never increase participant N.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .kumar2024 import (
    KUMAR2024_ALL_SUBJECTS,
    KUMAR2024_DEFAULT_BUDGETS,
    KUMAR2024_TARGET_SESSIONS,
)

_SHA256_HEX = frozenset("0123456789abcdef")

KUMAR2024_PROMOTED_SPLIT_SEEDS = (2026, 3407, 9109)
KUMAR2024_EEGNET_MODEL_SEEDS = (
    31415,
    384165836,
    3991196546,
)
KUMAR2024_PROMOTED_ANALYSIS_SEED = 160865088
KUMAR2024_PRIMARY_ENDPOINT = "paired_normalized_balanced_accuracy_frontier_auc"


def _canonical_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"schema": schema, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _nonempty(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _exact_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer without coercion")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _sha256(name: str, value: Any) -> str:
    text = _nonempty(name, value).lower()
    if len(text) != 64 or any(char not in _SHA256_HEX for char in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _unique_int_tuple(
    name: str,
    values: Sequence[Any],
    *,
    require_zero_start: bool = False,
) -> tuple[int, ...]:
    result = tuple(_exact_nonnegative_int(name, value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    if require_zero_start:
        if result != tuple(sorted(result)) or result[0] != 0:
            raise ValueError(f"{name} must be increasing and start at zero")
    return result


def _unique_strings(name: str, values: Sequence[Any]) -> tuple[str, ...]:
    result = tuple(_nonempty(name, value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _normalized_trapezoid(xs: Sequence[int], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("frontier AUC requires at least two aligned points")
    area = 0.0
    for index in range(len(xs) - 1):
        width = float(xs[index + 1] - xs[index])
        if width <= 0:
            raise ValueError("frontier budgets must be strictly increasing")
        area += width * 0.5 * (float(ys[index]) + float(ys[index + 1]))
    return float(area / float(xs[-1] - xs[0]))


def _stable_seed(base: int, *parts: Any) -> int:
    raw = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:4], "big")


def _bootstrap_mean_ci(
    participant_values: Mapping[int, float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    ids = sorted(participant_values)
    values = np.asarray([participant_values[item] for item in ids], dtype=np.float64)
    if len(values) == 0:
        return {"n_participants": 0, "mean": None, "ci95": [None, None]}
    center = float(np.mean(values))
    if len(values) == 1:
        return {"n_participants": 1, "mean": center, "ci95": [center, center]}
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(values), size=len(values))
        samples[index] = float(np.mean(values[selected]))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "n_participants": len(values),
        "mean": center,
        "ci95": [float(low), float(high)],
    }


@dataclass(frozen=True, slots=True)
class MethodOptimizationSeedPolicy:
    """Predeclared optimization-randomness policy for one method."""

    method_id: str
    stochastic: bool
    model_seeds: tuple[int, ...] = ()
    seed_source: str = "not_applicable"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("MethodOptimizationSeedPolicy schema_version must be 1")
        method = _nonempty("method_id", self.method_id)
        if not isinstance(self.stochastic, bool):
            raise TypeError("stochastic must be bool")
        seeds = tuple(
            sorted(
                _exact_nonnegative_int("model seed", value)
                for value in self.model_seeds
            )
        )
        if len(set(seeds)) != len(seeds):
            raise ValueError("model_seeds cannot contain duplicates")
        source = _nonempty("seed_source", self.seed_source)
        if self.stochastic and not seeds:
            raise ValueError("stochastic methods require predeclared model seeds")
        if not self.stochastic and seeds:
            raise ValueError("deterministic methods cannot invent a model-seed axis")
        if self.stochastic and source == "not_applicable":
            raise ValueError("stochastic model seeds require an explicit seed_source")
        if not self.stochastic and source != "not_applicable":
            raise ValueError("deterministic methods must use seed_source='not_applicable'")
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "model_seeds", seeds)
        object.__setattr__(self, "seed_source", source)

    @property
    def realization_model_seeds(self) -> tuple[int | None, ...]:
        return self.model_seeds if self.stochastic else (None,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "stochastic": self.stochastic,
            "model_seeds": list(self.model_seeds),
            "seed_source": self.seed_source,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            "neuros.kumar2024_method_optimization_seed_policy.v1",
            self.to_dict(),
        )


@dataclass(frozen=True, slots=True)
class Kumar2024ComparisonPlan:
    """Immutable promoted-comparison authority above single study realizations."""

    plan_id: str
    subjects: tuple[int, ...]
    target_sessions: tuple[str, ...]
    budgets_per_class: tuple[int, ...]
    split_seeds: tuple[int, ...]
    method_seed_policies: tuple[MethodOptimizationSeedPolicy, ...]
    analysis_seed: int
    bootstrap_replicates: int = 2000
    independent_unit: str = "participant"
    primary_metric: str = "balanced_accuracy"
    primary_endpoint: str = KUMAR2024_PRIMARY_ENDPOINT
    complete_frontier_required: bool = True
    aggregation_hierarchy: tuple[str, ...] = (
        "participant",
        "target_session",
        "split_seed",
        "model_seed",
        "calibration_budget",
    )
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("Kumar2024ComparisonPlan schema_version must be 1")
        plan_id = _nonempty("plan_id", self.plan_id)
        subjects = tuple(
            _exact_nonnegative_int("subject", value) for value in self.subjects
        )
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("subjects must be non-empty and unique")
        if any(subject not in KUMAR2024_ALL_SUBJECTS for subject in subjects):
            raise ValueError("Kumar2024 subjects must lie in 1..18")
        if subjects != tuple(sorted(subjects)):
            raise ValueError("subjects must be in increasing canonical order")
        sessions = _unique_strings("target session", self.target_sessions)
        if any(session not in KUMAR2024_TARGET_SESSIONS for session in sessions):
            raise ValueError("Kumar2024 target sessions must lie in 1..5")
        expected_session_order = tuple(
            value for value in KUMAR2024_TARGET_SESSIONS if value in set(sessions)
        )
        if sessions != expected_session_order:
            raise ValueError("target_sessions must preserve chronological order")
        budgets = _unique_int_tuple(
            "budgets_per_class",
            self.budgets_per_class,
            require_zero_start=True,
        )
        split_seeds = _unique_int_tuple("split_seeds", self.split_seeds)
        if split_seeds != tuple(sorted(split_seeds)):
            raise ValueError("split_seeds must be in increasing canonical order")
        policies = tuple(self.method_seed_policies)
        if not policies or any(
            not isinstance(item, MethodOptimizationSeedPolicy) for item in policies
        ):
            raise TypeError(
                "method_seed_policies must contain MethodOptimizationSeedPolicy objects"
            )
        methods = [item.method_id for item in policies]
        if len(set(methods)) != len(methods):
            raise ValueError("method_seed_policies cannot repeat a method")
        analysis_seed = _exact_nonnegative_int("analysis_seed", self.analysis_seed)
        if analysis_seed in split_seeds:
            raise ValueError("analysis seed must be distinct from split seeds")
        stochastic_seeds = {
            seed for policy in policies for seed in policy.model_seeds
        }
        if analysis_seed in stochastic_seeds:
            raise ValueError("analysis seed must be distinct from model seeds")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if self.independent_unit != "participant":
            raise ValueError(
                "promoted Kumar2024 inference requires participant as independent unit"
            )
        if self.primary_metric != "balanced_accuracy":
            raise ValueError("promoted Kumar2024 primary metric is balanced_accuracy")
        if self.primary_endpoint != KUMAR2024_PRIMARY_ENDPOINT:
            raise ValueError(
                f"promoted Kumar2024 primary endpoint is {KUMAR2024_PRIMARY_ENDPOINT}"
            )
        if self.complete_frontier_required is not True:
            raise ValueError("promoted comparison requires complete calibration frontiers")
        expected_hierarchy = (
            "participant",
            "target_session",
            "split_seed",
            "model_seed",
            "calibration_budget",
        )
        if tuple(self.aggregation_hierarchy) != expected_hierarchy:
            raise ValueError(
                "aggregation_hierarchy must preserve participant/session/split/model/budget order"
            )
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "target_sessions", sessions)
        object.__setattr__(self, "budgets_per_class", budgets)
        object.__setattr__(self, "split_seeds", split_seeds)
        object.__setattr__(self, "method_seed_policies", policies)
        object.__setattr__(self, "analysis_seed", analysis_seed)
        object.__setattr__(self, "aggregation_hierarchy", expected_hierarchy)

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(policy.method_id for policy in self.method_seed_policies)

    def policy_for(self, method_id: str) -> MethodOptimizationSeedPolicy:
        for policy in self.method_seed_policies:
            if policy.method_id == method_id:
                return policy
        raise KeyError(f"method {method_id!r} is not declared by comparison plan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "subjects": list(self.subjects),
            "target_sessions": list(self.target_sessions),
            "budgets_per_class": list(self.budgets_per_class),
            "split_seeds": list(self.split_seeds),
            "method_seed_policies": [
                item.to_dict() for item in self.method_seed_policies
            ],
            "analysis": {
                "independent_unit": self.independent_unit,
                "aggregation_hierarchy": list(self.aggregation_hierarchy),
                "primary_metric": self.primary_metric,
                "primary_endpoint": self.primary_endpoint,
                "complete_frontier_required": self.complete_frontier_required,
                "bootstrap_replicates": self.bootstrap_replicates,
                "analysis_seed": self.analysis_seed,
                "cohort_policy": (
                    "GR/PAR are preserved for stratified/descriptive reporting; "
                    "they are not treated as randomized interventions"
                ),
                "failure_policy": (
                    "all planned realization rows are retained; incomplete stochastic "
                    "or calibration frontiers do not borrow successful repeats"
                ),
            },
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            "neuros.kumar2024_comparison_plan.v1",
            self.to_dict(),
        )


def promoted_external_floor_plan() -> Kumar2024ComparisonPlan:
    """Return the preregistered external-floor comparison plan.

    The first EEGNet seed (31415) is the already-frozen authority anchor from
    PR #95. The two additional optimization seeds were fixed before promoted
    final-assessment execution by taking the first 32 bits of SHA-256 for:

      neuros.kumar2024.eegnet.optimization_seed.v1|1
      neuros.kumar2024.eegnet.optimization_seed.v1|2

    They are not selected from model performance.
    """

    return Kumar2024ComparisonPlan(
        plan_id="nsq-kumar2024-external-floor-v1",
        subjects=KUMAR2024_ALL_SUBJECTS,
        target_sessions=KUMAR2024_TARGET_SESSIONS,
        budgets_per_class=KUMAR2024_DEFAULT_BUDGETS,
        split_seeds=KUMAR2024_PROMOTED_SPLIT_SEEDS,
        method_seed_policies=(
            MethodOptimizationSeedPolicy(
                method_id="mne-csp-lda",
                stochastic=False,
            ),
            MethodOptimizationSeedPolicy(
                method_id="pyriemann-rg-lr",
                stochastic=False,
            ),
            MethodOptimizationSeedPolicy(
                method_id="braindecode-eegnet",
                stochastic=True,
                model_seeds=KUMAR2024_EEGNET_MODEL_SEEDS,
                seed_source=(
                    "31415 frozen by PR #95; additional seeds are SHA-256-derived "
                    "from neuros.kumar2024.eegnet.optimization_seed.v1 before "
                    "promoted final-assessment execution"
                ),
            ),
        ),
        analysis_seed=KUMAR2024_PROMOTED_ANALYSIS_SEED,
    )


def _cohort_for_subject(subject: int) -> str:
    return "GR" if subject <= 9 else "PAR"


def _model_seed_from_row(
    row: Mapping[str, Any],
    policy: MethodOptimizationSeedPolicy,
) -> int | None:
    raw = row.get("model_seed")
    if policy.stochastic:
        seed = _exact_nonnegative_int("model_seed", raw)
        if seed not in policy.model_seeds:
            raise ValueError(
                f"unplanned model seed {seed} for method {policy.method_id!r}"
            )
        return seed
    if raw not in {None, ""}:
        raise ValueError(
            f"deterministic method {policy.method_id!r} cannot carry model_seed={raw!r}"
        )
    return None


def validate_promoted_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Kumar2024ComparisonPlan,
) -> tuple[dict[str, Any], ...]:
    """Validate promoted comparison rows before any aggregation.

    Every competing method/model seed for one participant/session/split seed
    must bind the exact same case-authority SHA.
    """

    if not rows:
        raise ValueError("promoted comparison rows cannot be empty")
    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    case_authorities: dict[tuple[int, str, int], set[str]] = defaultdict(set)

    for raw in rows:
        method = _nonempty("method_id", raw.get("method_id"))
        policy = plan.policy_for(method)
        subject = _exact_nonnegative_int("subject", raw.get("subject"))
        if subject not in plan.subjects:
            raise ValueError(f"unplanned subject {subject}")
        session = _nonempty("held_out_session", raw.get("held_out_session"))
        if session not in plan.target_sessions:
            raise ValueError(f"unplanned target session {session!r}")
        split_seed = _exact_nonnegative_int("split_seed", raw.get("split_seed"))
        if split_seed not in plan.split_seeds:
            raise ValueError(f"unplanned split seed {split_seed}")
        budget = _exact_nonnegative_int(
            "calibration_per_class",
            raw.get("calibration_per_class"),
        )
        if budget not in plan.budgets_per_class:
            raise ValueError(f"unplanned calibration budget {budget}")
        model_seed = _model_seed_from_row(raw, policy)
        authority_sha = _sha256(
            "case_authority_sha256", raw.get("case_authority_sha256")
        )
        cohort = _nonempty("original_protocol", raw.get("original_protocol"))
        expected_cohort = _cohort_for_subject(subject)
        if cohort != expected_cohort:
            raise ValueError(
                f"subject {subject} cohort mismatch: expected {expected_cohort}, observed {cohort}"
            )
        status = _nonempty("status", raw.get("status"))
        balanced_accuracy = raw.get("balanced_accuracy")
        if balanced_accuracy is not None:
            if isinstance(balanced_accuracy, bool) or not isinstance(
                balanced_accuracy, (int, float, np.number)
            ):
                raise ValueError("balanced_accuracy must be numeric or None")
            balanced_accuracy = float(balanced_accuracy)
            if not math.isfinite(balanced_accuracy):
                raise ValueError("balanced_accuracy must be finite")
            if not 0.0 <= balanced_accuracy <= 1.0:
                raise ValueError("balanced_accuracy must lie in [0, 1]")
        if status == "success" and balanced_accuracy is None:
            raise ValueError("successful row requires balanced_accuracy")
        if status != "success" and balanced_accuracy is not None:
            raise ValueError("non-success row cannot carry balanced_accuracy")

        key = (method, subject, session, split_seed, model_seed, budget)
        if key in seen_keys:
            raise ValueError(f"duplicate promoted result row for key={key!r}")
        seen_keys.add(key)
        case_authorities[(subject, session, split_seed)].add(authority_sha)

        value = dict(raw)
        value.update(
            {
                "method_id": method,
                "subject": subject,
                "held_out_session": session,
                "split_seed": split_seed,
                "model_seed": model_seed,
                "calibration_per_class": budget,
                "case_authority_sha256": authority_sha,
                "original_protocol": cohort,
                "status": status,
                "balanced_accuracy": balanced_accuracy,
            }
        )
        normalized.append(value)

    mismatched = {
        key: sorted(values)
        for key, values in case_authorities.items()
        if len(values) != 1
    }
    if mismatched:
        raise ValueError(
            "competing methods/model seeds do not share exact case authority: "
            f"{mismatched}"
        )
    return tuple(normalized)


def _realization_frontiers(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Kumar2024ComparisonPlan,
) -> tuple[
    dict[tuple[str, int, str, int, int | None], float],
    dict[tuple[str, int, str, int, int | None], tuple[int, ...]],
]:
    by_realization: dict[
        tuple[str, int, str, int, int | None],
        dict[int, Mapping[str, Any]],
    ] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["method_id"]),
            int(row["subject"]),
            str(row["held_out_session"]),
            int(row["split_seed"]),
            row["model_seed"],
        )
        by_realization[key][int(row["calibration_per_class"])] = row

    complete: dict[tuple[str, int, str, int, int | None], float] = {}
    missing: dict[
        tuple[str, int, str, int, int | None],
        tuple[int, ...],
    ] = {}
    for method in plan.methods:
        policy = plan.policy_for(method)
        for subject in plan.subjects:
            for session in plan.target_sessions:
                for split_seed in plan.split_seeds:
                    for model_seed in policy.realization_model_seeds:
                        key = (method, subject, session, split_seed, model_seed)
                        by_budget = by_realization.get(key, {})
                        absent = tuple(
                            budget
                            for budget in plan.budgets_per_class
                            if budget not in by_budget
                            or by_budget[budget]["status"] != "success"
                            or by_budget[budget]["balanced_accuracy"] is None
                        )
                        if absent:
                            missing[key] = absent
                            continue
                        ys = [
                            float(by_budget[budget]["balanced_accuracy"])
                            for budget in plan.budgets_per_class
                        ]
                        complete[key] = _normalized_trapezoid(
                            plan.budgets_per_class,
                            ys,
                        )
    return complete, missing


def _collapse_frontier_participants(
    complete: Mapping[tuple[str, int, str, int, int | None], float],
    *,
    plan: Kumar2024ComparisonPlan,
) -> tuple[
    dict[str, dict[int, float]],
    dict[str, dict[int, dict[str, Any]]],
]:
    participant_values: dict[str, dict[int, float]] = {}
    diagnostics: dict[str, dict[int, dict[str, Any]]] = {}

    for method in plan.methods:
        policy = plan.policy_for(method)
        session_split_values: dict[tuple[int, str, int], float] = {}
        for subject in plan.subjects:
            for session in plan.target_sessions:
                for split_seed in plan.split_seeds:
                    values: list[float] = []
                    missing_model_seeds: list[int | None] = []
                    for model_seed in policy.realization_model_seeds:
                        key = (method, subject, session, split_seed, model_seed)
                        if key not in complete:
                            missing_model_seeds.append(model_seed)
                        else:
                            values.append(float(complete[key]))
                    if not missing_model_seeds:
                        session_split_values[(subject, session, split_seed)] = float(
                            np.mean(values)
                        )

        session_values: dict[tuple[int, str], float] = {}
        for subject in plan.subjects:
            for session in plan.target_sessions:
                values = [
                    session_split_values[(subject, session, split_seed)]
                    for split_seed in plan.split_seeds
                    if (subject, session, split_seed) in session_split_values
                ]
                if len(values) == len(plan.split_seeds):
                    session_values[(subject, session)] = float(np.mean(values))

        values_by_participant: dict[int, float] = {}
        method_diagnostics: dict[int, dict[str, Any]] = {}
        for subject in plan.subjects:
            complete_sessions = {
                session: session_values[(subject, session)]
                for session in plan.target_sessions
                if (subject, session) in session_values
            }
            if len(complete_sessions) == len(plan.target_sessions):
                values_by_participant[subject] = float(
                    np.mean(list(complete_sessions.values()))
                )
            method_diagnostics[subject] = {
                "complete_target_sessions": sorted(complete_sessions),
                "required_target_sessions": list(plan.target_sessions),
                "complete": subject in values_by_participant,
            }
        participant_values[method] = values_by_participant
        diagnostics[method] = method_diagnostics
    return participant_values, diagnostics



def _frontier_diagnostic_layers(
    complete: Mapping[tuple[str, int, str, int, int | None], float],
    *,
    plan: Kumar2024ComparisonPlan,
) -> dict[str, list[dict[str, Any]]]:
    """Expose repeated-measure estimates without treating them as population N.

    These records are descriptive traceability surfaces only. Population
    inference remains participant-level and is computed elsewhere.
    """

    model_realizations = [
        {
            "method_id": method,
            "subject": subject,
            "held_out_session": session,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "normalized_balanced_accuracy_frontier_auc": float(value),
        }
        for (method, subject, session, split_seed, model_seed), value in sorted(
            complete.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
                item[0][3],
                -1 if item[0][4] is None else item[0][4],
            ),
        )
    ]

    split_records: list[dict[str, Any]] = []
    split_values: dict[tuple[str, int, str, int], float] = {}
    for method in plan.methods:
        policy = plan.policy_for(method)
        for subject in plan.subjects:
            for session in plan.target_sessions:
                for split_seed in plan.split_seeds:
                    values = [
                        complete.get((method, subject, session, split_seed, model_seed))
                        for model_seed in policy.realization_model_seeds
                    ]
                    if any(value is None for value in values):
                        continue
                    value = float(np.mean([float(item) for item in values if item is not None]))
                    split_values[(method, subject, session, split_seed)] = value
                    split_records.append(
                        {
                            "method_id": method,
                            "subject": subject,
                            "held_out_session": session,
                            "split_seed": split_seed,
                            "model_seed_count": len(policy.realization_model_seeds),
                            "normalized_balanced_accuracy_frontier_auc": value,
                        }
                    )

    session_records: list[dict[str, Any]] = []
    for method in plan.methods:
        for subject in plan.subjects:
            for session in plan.target_sessions:
                values = [
                    split_values.get((method, subject, session, split_seed))
                    for split_seed in plan.split_seeds
                ]
                if any(value is None for value in values):
                    continue
                session_records.append(
                    {
                        "method_id": method,
                        "subject": subject,
                        "held_out_session": session,
                        "split_seed_count": len(plan.split_seeds),
                        "normalized_balanced_accuracy_frontier_auc": float(
                            np.mean([float(item) for item in values if item is not None])
                        ),
                    }
                )

    return {
        "model_realization_frontier_auc": model_realizations,
        "model_seed_averaged_split_frontier_auc": split_records,
        "split_seed_averaged_session_frontier_auc": session_records,
    }

def _pointwise_participant_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Kumar2024ComparisonPlan,
    method: str,
    budget: int,
) -> dict[int, float]:
    policy = plan.policy_for(method)
    lookup = {
        (
            int(row["subject"]),
            str(row["held_out_session"]),
            int(row["split_seed"]),
            row["model_seed"],
        ): row
        for row in rows
        if row["method_id"] == method
        and int(row["calibration_per_class"]) == budget
    }
    session_values: dict[tuple[int, str], float] = {}
    for subject in plan.subjects:
        for session in plan.target_sessions:
            split_values: list[float] = []
            for split_seed in plan.split_seeds:
                model_values: list[float] = []
                complete = True
                for model_seed in policy.realization_model_seeds:
                    row = lookup.get((subject, session, split_seed, model_seed))
                    if (
                        row is None
                        or row["status"] != "success"
                        or row["balanced_accuracy"] is None
                    ):
                        complete = False
                        break
                    model_values.append(float(row["balanced_accuracy"]))
                if complete:
                    split_values.append(float(np.mean(model_values)))
            if len(split_values) == len(plan.split_seeds):
                session_values[(subject, session)] = float(np.mean(split_values))

    participant: dict[int, float] = {}
    for subject in plan.subjects:
        values = [
            session_values[(subject, session)]
            for session in plan.target_sessions
            if (subject, session) in session_values
        ]
        if len(values) == len(plan.target_sessions):
            participant[subject] = float(np.mean(values))
    return participant


def summarize_promoted_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Kumar2024ComparisonPlan,
) -> dict[str, Any]:
    """Aggregate promoted rows without inflating N with sessions or random seeds."""

    normalized = validate_promoted_rows(rows, plan=plan)
    complete, incomplete = _realization_frontiers(normalized, plan=plan)
    participant_frontiers, diagnostics = _collapse_frontier_participants(
        complete,
        plan=plan,
    )
    frontier_diagnostics = _frontier_diagnostic_layers(complete, plan=plan)

    method_frontiers: list[dict[str, Any]] = []
    pointwise: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for method_index, method in enumerate(plan.methods):
        participant_values = participant_frontiers[method]
        method_frontiers.append(
            {
                "method_id": method,
                "optimization_seed_policy": plan.policy_for(method).to_dict(),
                "participant_frontier_auc": _bootstrap_mean_ci(
                    participant_values,
                    seed=_stable_seed(
                        plan.analysis_seed,
                        "method-frontier",
                        method_index,
                    ),
                    replicates=plan.bootstrap_replicates,
                ),
                "complete_frontier_participants": sorted(participant_values),
                "participant_completeness": diagnostics[method],
            }
        )
        for budget_index, budget in enumerate(plan.budgets_per_class):
            values = _pointwise_participant_values(
                normalized,
                plan=plan,
                method=method,
                budget=budget,
            )
            pointwise.append(
                {
                    "method_id": method,
                    "calibration_per_class": budget,
                    "participant_balanced_accuracy": _bootstrap_mean_ci(
                        values,
                        seed=_stable_seed(
                            plan.analysis_seed,
                            "pointwise",
                            method_index,
                            budget_index,
                        ),
                        replicates=plan.bootstrap_replicates,
                    ),
                    "complete_participants": sorted(values),
                }
            )
        method_rows = [row for row in normalized if row["method_id"] == method]
        counts = Counter(
            str(row["status"])
            for row in method_rows
            if row["status"] != "success"
        )
        failures.append(
            {
                "method_id": method,
                "attempted_rows": len(method_rows),
                "failure_status_counts": dict(sorted(counts.items())),
            }
        )

    paired_frontier: list[dict[str, Any]] = []
    for left_index, left in enumerate(plan.methods):
        for right_index in range(left_index + 1, len(plan.methods)):
            right = plan.methods[right_index]
            left_values = participant_frontiers[left]
            right_values = participant_frontiers[right]
            matched = sorted(set(left_values) & set(right_values))
            differences = {
                participant: left_values[participant] - right_values[participant]
                for participant in matched
            }
            paired_frontier.append(
                {
                    "left_method": left,
                    "right_method": right,
                    "matched_complete_frontier_participants": matched,
                    "left_minus_right_normalized_balanced_accuracy_frontier_auc": (
                        _bootstrap_mean_ci(
                            differences,
                            seed=_stable_seed(
                                plan.analysis_seed,
                                "paired-frontier",
                                left_index,
                                right_index,
                            ),
                            replicates=plan.bootstrap_replicates,
                        )
                    ),
                }
            )

    cohort_descriptive: list[dict[str, Any]] = []
    for method in plan.methods:
        values = participant_frontiers[method]
        for cohort in ("GR", "PAR"):
            cohort_values = {
                subject: value
                for subject, value in values.items()
                if _cohort_for_subject(subject) == cohort
            }
            cohort_descriptive.append(
                {
                    "method_id": method,
                    "original_protocol": cohort,
                    "complete_frontier_participants": sorted(cohort_values),
                    "mean_normalized_balanced_accuracy_frontier_auc": (
                        None
                        if not cohort_values
                        else float(np.mean(list(cohort_values.values())))
                    ),
                }
            )

    incomplete_records = [
        {
            "method_id": key[0],
            "subject": key[1],
            "held_out_session": key[2],
            "split_seed": key[3],
            "model_seed": key[4],
            "missing_or_failed_budgets": list(budgets),
        }
        for key, budgets in sorted(
            incomplete.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
                item[0][3],
                -1 if item[0][4] is None else item[0][4],
            ),
        )
    ]

    return {
        "schema_version": 1,
        "comparison_plan_sha256": plan.sha256,
        "independent_inferential_unit": "participant",
        "repeated_measure_axes": [
            "target_session",
            "split_seed",
            "model_seed",
            "calibration_budget",
        ],
        "aggregation_order": [
            "complete calibration frontier within model realization",
            "model seeds within participant/session/split seed",
            "split seeds within participant/session",
            "target sessions within participant",
            "participant-level population inference",
        ],
        "primary_metric": plan.primary_metric,
        "primary_study_endpoint": plan.primary_endpoint,
        "frontier_diagnostics_policy": (
            "model-realization, model-seed-averaged split, and split-seed-averaged "
            "session estimates are descriptive traceability only; they never increase "
            "the independent participant count"
        ),
        "frontier_diagnostics": frontier_diagnostics,
        "method_frontier_auc": method_frontiers,
        "paired_calibration_efficiency": paired_frontier,
        "secondary_pointwise_performance": pointwise,
        "cohort_descriptive": cohort_descriptive,
        "failure_summary": failures,
        "incomplete_realization_frontiers": incomplete_records,
        "failure_policy": (
            "missing or failed budget/model/split repeats invalidate that complete "
            "realization path; successful siblings are not used to increase effective N"
        ),
    }


__all__ = [
    "KUMAR2024_EEGNET_MODEL_SEEDS",
    "KUMAR2024_PRIMARY_ENDPOINT",
    "KUMAR2024_PROMOTED_ANALYSIS_SEED",
    "KUMAR2024_PROMOTED_SPLIT_SEEDS",
    "Kumar2024ComparisonPlan",
    "MethodOptimizationSeedPolicy",
    "promoted_external_floor_plan",
    "summarize_promoted_rows",
    "validate_promoted_rows",
]
