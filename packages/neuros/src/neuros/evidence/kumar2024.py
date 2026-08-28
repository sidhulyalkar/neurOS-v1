"""Canonical NSQ Kumar2024 v1 external-baseline study.

This module is deliberately a *composition* layer. It does not introduce a new
scientific split engine, model trainer, or metric implementation. Instead it
binds together the existing authorities owned by their appropriate packages:

- ORION Scientific Authority v2 for dataset lineage;
- neuros-foundation for MOABB loading and longitudinal sample authority;
- the production NSQ Runner v1 for external model execution;
- the production NSQ classification scorecard for metric semantics.

The study is a neurOS re-evaluation of the MOABB Kumar2024 bar-feedback subset.
It is not a reproduction of the original online GR/PAR intervention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

KUMAR2024_DATASET_ID = "moabb-kumar2024"
KUMAR2024_DATASET_KEY = "kumar2024"
KUMAR2024_EXPECTED_SESSIONS = ("0", "1", "2", "3", "4", "5")
KUMAR2024_TARGET_SESSIONS = ("1", "2", "3", "4", "5")
KUMAR2024_ALL_SUBJECTS = tuple(range(1, 19))
KUMAR2024_PAPER_DOI = "10.1093/pnasnexus/pgae076"
KUMAR2024_DATA_DOI = "10.5281/zenodo.10694880"
KUMAR2024_DEFAULT_BUDGETS = (0, 1, 2, 5, 10)
KUMAR2024_DEFAULT_METHODS = (
    "mne-csp-lda",
    "braindecode-eegnet",
    "braindecode-eegconformer",
)
_BUNDLE_FILES = (
    "study_manifest.json",
    "case_authorities.json",
    "case_results.json",
    "results.csv",
    "analysis.json",
    "report.md",
)


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Kumar2024 evidence identity cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Kumar2024 evidence identity cannot contain object arrays")
        return _canonical(value.tolist())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("Kumar2024 evidence mapping keys must be non-empty")
            if key in normalized:
                raise ValueError("Kumar2024 evidence mapping keys collide after normalization")
            normalized[key] = _canonical(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("unordered sets are not valid Kumar2024 evidence identity values")
    raise TypeError(
        "Kumar2024 evidence identity must use deterministic JSON-compatible values; "
        f"got {type(value).__name__}"
    )


def _identity_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"schema": schema, "payload": _canonical(payload)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_canonical(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _runtime_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in (
        "neuros",
        "neuros-core",
        "neuros-models",
        "neuros-foundation",
        "neuros-orion",
        "moabb",
        "mne",
        "scikit-learn",
        "braindecode",
        "torch",
        "numpy",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stable_seed(base: int, *parts: Any) -> int:
    raw = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:4], "big")


@dataclass(frozen=True, slots=True)
class Kumar2024PreprocessingSpec:
    """Fixed, non-data-fitted MOABB preprocessing requested by NSQ v1."""

    fmin_hz: float = 8.0
    fmax_hz: float = 30.0
    resample_hz: float | None = None
    additional_normalization: str = "none"
    return_epochs: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("Kumar2024PreprocessingSpec schema_version must be 1")
        low = float(self.fmin_hz)
        high = float(self.fmax_hz)
        if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high <= low:
            raise ValueError("preprocessing requires finite 0 < fmin_hz < fmax_hz")
        if self.resample_hz is not None:
            rate = float(self.resample_hz)
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("resample_hz must be finite and positive")
            object.__setattr__(self, "resample_hz", rate)
        normalization = str(self.additional_normalization).strip()
        if not normalization:
            raise ValueError("additional_normalization must be explicit")
        if self.return_epochs is not True:
            raise ValueError(
                "Kumar2024 NSQ v1 requires return_epochs=True to preserve processed channel authority"
            )
        object.__setattr__(self, "fmin_hz", low)
        object.__setattr__(self, "fmax_hz", high)
        object.__setattr__(self, "additional_normalization", normalization)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "moabb_paradigm": "LeftRightImagery",
            "fmin_hz": self.fmin_hz,
            "fmax_hz": self.fmax_hz,
            "resample_hz": self.resample_hz,
            "additional_normalization": self.additional_normalization,
            "return_epochs": self.return_epochs,
        }


@dataclass(frozen=True, slots=True)
class Kumar2024StudyConfig:
    """Frozen execution choices that are independent of loaded neural values."""

    subjects: tuple[int, ...] = (1, 10)
    target_sessions: tuple[str, ...] = KUMAR2024_TARGET_SESSIONS
    budgets_per_class: tuple[int, ...] = KUMAR2024_DEFAULT_BUDGETS
    methods: tuple[str, ...] = KUMAR2024_DEFAULT_METHODS
    split_seed: int = 2026
    evaluation_fraction: float = 0.5
    csp_components: int = 8
    braindecode_epochs: int = 1
    braindecode_batch_size: int = 32
    braindecode_learning_rate: float = 1e-3
    braindecode_weight_decay: float = 0.0
    braindecode_random_state: int = 2026
    device: str = "cpu"
    analysis_bootstrap_replicates: int = 2000
    analysis_seed: int = 9109
    profile: str = "pilot"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("Kumar2024StudyConfig schema_version must be 1")
        subjects = tuple(int(value) for value in self.subjects)
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("subjects must be non-empty and unique")
        if any(value not in KUMAR2024_ALL_SUBJECTS for value in subjects):
            raise ValueError("Kumar2024 subjects must lie in 1..18")
        sessions = tuple(str(value) for value in self.target_sessions)
        if not sessions or len(set(sessions)) != len(sessions):
            raise ValueError("target_sessions must be non-empty and unique")
        if any(value not in KUMAR2024_TARGET_SESSIONS for value in sessions):
            raise ValueError("Kumar2024 target sessions must lie in 1..5")
        budgets = tuple(int(value) for value in self.budgets_per_class)
        if budgets != tuple(sorted(set(budgets))) or not budgets or budgets[0] != 0:
            raise ValueError("budgets_per_class must be unique, increasing, and start at zero")
        methods = tuple(str(value).strip() for value in self.methods)
        unknown = sorted(set(methods) - set(KUMAR2024_DEFAULT_METHODS))
        if not methods or unknown:
            raise ValueError(f"unsupported Kumar2024 NSQ methods: {unknown}")
        if isinstance(self.split_seed, bool) or self.split_seed < 0:
            raise ValueError("split_seed must be a non-negative integer")
        fraction = float(self.evaluation_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("evaluation_fraction must lie strictly between zero and one")
        if self.csp_components <= 0:
            raise ValueError("csp_components must be positive")
        if self.braindecode_epochs <= 0 or self.braindecode_batch_size <= 0:
            raise ValueError("Braindecode epochs and batch size must be positive")
        if self.braindecode_learning_rate <= 0 or self.braindecode_weight_decay < 0:
            raise ValueError("Braindecode optimizer values are invalid")
        if self.analysis_bootstrap_replicates <= 0:
            raise ValueError("analysis_bootstrap_replicates must be positive")
        if self.analysis_seed < 0:
            raise ValueError("analysis_seed must be non-negative")
        profile = str(self.profile).strip()
        if profile not in {"pilot", "full"}:
            raise ValueError("profile must be 'pilot' or 'full'")
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "target_sessions", sessions)
        object.__setattr__(self, "budgets_per_class", budgets)
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "evaluation_fraction", fraction)
        object.__setattr__(self, "device", str(self.device).strip())
        object.__setattr__(self, "profile", profile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "subjects": list(self.subjects),
            "target_sessions": list(self.target_sessions),
            "budgets_per_class": list(self.budgets_per_class),
            "methods": list(self.methods),
            "history_policy": "prior",
            "split_seed": self.split_seed,
            "evaluation_fraction": self.evaluation_fraction,
            "csp_components": self.csp_components,
            "braindecode": {
                "epochs": self.braindecode_epochs,
                "batch_size": self.braindecode_batch_size,
                "learning_rate": self.braindecode_learning_rate,
                "weight_decay": self.braindecode_weight_decay,
                "random_state": self.braindecode_random_state,
                "device": self.device,
            },
            "analysis": {
                "independent_unit": "participant",
                "bootstrap_replicates": self.analysis_bootstrap_replicates,
                "seed": self.analysis_seed,
            },
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256("neuros.nsq_kumar2024_config.v1", self.to_dict())


def pilot_config() -> Kumar2024StudyConfig:
    return Kumar2024StudyConfig()


def full_config() -> Kumar2024StudyConfig:
    """Unpromoted all-subject reference configuration.

    This helper exists for feasibility work only. It is intentionally not exposed
    by the CLI because the promoted comparison in issue #27 still requires the
    shared split-seed ensemble, paired frontier-AUC endpoint, and predeclared
    neural-model seed authority. Twenty epochs is only a reference compute budget.
    """

    return Kumar2024StudyConfig(
        subjects=KUMAR2024_ALL_SUBJECTS,
        braindecode_epochs=20,
        profile="full",
    )


def _preprocessing_authority(
    preprocessing: Kumar2024PreprocessingSpec,
    epoch_descriptor: Any,
    versions: Mapping[str, str | None],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "fixed_preprocessing": preprocessing.to_dict(),
        "processed_signal_contract": epoch_descriptor.signal_contract_dict(),
        "processed_signal_contract_sha256": epoch_descriptor.signal_contract_sha256,
        "moabb_version": versions.get("moabb"),
        "mne_version": versions.get("mne"),
        "processed_value_units": {
            "get_data_units_argument": None,
            "unit_system": "MNE channel-type-specific default SI units",
            "eeg": "V",
        },
        "fit_kind": "fixed_upstream_transform_no_neuros_fit",
        "content_claim": (
            "identity covers the declared MOABB/MNE processing configuration and observed "
            "processed epoch geometry; it is not a raw-dataset checksum"
        ),
    }
    return {
        **payload,
        "sha256": _identity_sha256("neuros.kumar2024_preprocessing_authority.v1", payload),
    }


def build_dataset_lineage(
    *,
    config: Kumar2024StudyConfig,
    preprocessing_authority: Mapping[str, Any],
    versions: Mapping[str, str | None],
):
    """Construct honest partial lineage for the actual requested study corpus."""

    from orion.scientific import (
        DatasetLineage,
        IdentityAvailability,
        IdentitySet,
        LineageCompleteness,
    )

    moabb_version = versions.get("moabb")
    if not moabb_version:
        raise RuntimeError("Kumar2024 lineage requires an installed MOABB version")
    contract = preprocessing_authority["processed_signal_contract"]
    return DatasetLineage(
        dataset_id=KUMAR2024_DATASET_ID,
        upstream_source=(
            "MOABB Kumar2024 bar-feedback subset; Zenodo record " + KUMAR2024_DATA_DOI
        ),
        version=str(moabb_version),
        revision="moabb.datasets.Kumar2024",
        content_sha256=None,
        identity_sets=(
            IdentitySet(
                level="participant",
                availability=IdentityAvailability.AVAILABLE,
                identifiers=tuple(str(value) for value in config.subjects),
            ),
            IdentitySet(
                level="session",
                availability=IdentityAvailability.AVAILABLE,
                identifiers=KUMAR2024_EXPECTED_SESSIONS,
            ),
        ),
        preprocessing_history=(
            json.dumps(
                preprocessing_authority["fixed_preprocessing"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            f"processed_signal_contract_sha256={preprocessing_authority['processed_signal_contract_sha256']}",
        ),
        sampling_assumptions={
            "upstream_raw_eeg_channels": 22,
            "upstream_raw_sampling_rate_hz": 512.0,
            "upstream_reference": "CPz",
            "upstream_ground": "AFz",
            "processed_channel_names": contract["channel_names"],
            "processed_channel_types": contract["channel_types"],
            "processed_sampling_rate_hz": contract["sampling_rate_hz"],
            "processed_n_times": contract["n_times"],
            "processed_epoch_start_s": contract["epoch_start_s"],
            "processed_epoch_end_s": contract["epoch_end_s"],
            "processed_event_id": contract["event_id"],
            "processed_value_units": preprocessing_authority["processed_value_units"],
        },
        license="CC BY 4.0",
        citation=(
            f"paper doi:{KUMAR2024_PAPER_DOI}; data doi:{KUMAR2024_DATA_DOI}"
        ),
        lineage_completeness=LineageCompleteness.PARTIAL,
        metadata={
            "moabb_subset": "bar-feedback runs only",
            "excluded_upstream_runs": "car-racing runs",
            "original_protocol_context": {
                "subjects_1_9": "GR",
                "subjects_10_18": "PAR",
            },
            "requested_subjects": list(config.subjects),
            "expected_session_order": list(KUMAR2024_EXPECTED_SESSIONS),
            "not_a_reproduction_of_original_online_intervention": True,
            "content_sha256_reason": (
                "exact downloaded raw corpus bytes have not been hashed under a canonical rule"
            ),
        },
    )


def build_protocol(
    *,
    config: Kumar2024StudyConfig,
    dataset_lineage: Any,
    preprocessing_authority_sha256: str,
):
    from neuros.foundation_models.qualification import QualificationProtocolSpec
    from neuros.foundation_models.qualification_runner import (
        DEFAULT_CLASSIFICATION_SCORECARD,
    )

    return QualificationProtocolSpec(
        protocol_id="nsq-kumar2024-v1",
        dataset_id=KUMAR2024_DATASET_ID,
        dataset_lineage_sha256=dataset_lineage.lineage_sha256,
        task_id="left-vs-right-motor-imagery",
        independent_unit="participant",
        grouping_hierarchy=("participant", "session", "trial"),
        calibration_budgets_per_class=config.budgets_per_class,
        primary_metric="balanced_accuracy",
        secondary_metrics=(
            "accuracy",
            "roc_auc",
            "brier_score",
            "expected_calibration_error",
        ),
        metric_scorecard_sha256=DEFAULT_CLASSIFICATION_SCORECARD.sha256,
        robustness_axes=(
            "session",
            "subject",
        ),
        final_assessment_role="untouched_final_assessment",
        protocol_status="frozen",
        metadata={
            "history_policy": "prior",
            "expected_session_order": list(KUMAR2024_EXPECTED_SESSIONS),
            "preprocessing_authority_sha256": preprocessing_authority_sha256,
            "unlabeled_target_adaptation": False,
            "cohort_labels_are_contextual_metadata": True,
            "analysis_independent_unit": "participant",
            "planned_not_executed_robustness_axes": [
                "channel_drop",
                "artifact_sensitivity",
                "montage",
            ],
            "event_semantics": (
                "processed MNE event_id is retained verbatim; task labels are separately "
                "bound in GroupedEvaluationData targets and processed-data identity"
            ),
            "study_scope": (
                "neurOS re-evaluation of the MOABB bar-feedback subset; not a reproduction "
                "of the original online GR/PAR intervention"
            ),
        },
    )


def _make_case_authority(
    *,
    data: Any,
    dataset_spec: Any,
    subject: int,
    target_session: str,
    config: Kumar2024StudyConfig,
):
    from neuros.foundation_models.longitudinal import (
        chronological_partition,
        make_nested_calibration_split,
        ordered_group_values,
    )
    from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
    from neuros.foundation_models.moabb_longitudinal import validate_observed_sessions

    observed = validate_observed_sessions(
        dataset_spec,
        ordered_group_values(data, split_unit="session"),
    )
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value=target_session,
        order=observed,
    )
    # The preregistered split seed is shared literally across every case.
    # Subject/session identity is already bound by the case authority and must
    # not be mixed into the split RNG seed.
    case_seed = config.split_seed
    split = make_nested_calibration_split(
        partition,
        evaluation_fraction=config.evaluation_fraction,
        seed=case_seed,
    )
    largest_budget = config.budgets_per_class[-1]
    if largest_budget > split.max_budget_per_class:
        raise RuntimeError(
            "strict paired Kumar2024 frontier cannot be constructed: "
            f"subject={subject}, session={target_session}, requested={largest_budget}/class, "
            f"available={split.max_budget_per_class}/class"
        )
    metadata = dataset_spec.case_metadata(subject)
    metadata.update(
        {
            "subject": int(subject),
            "held_out_session": str(target_session),
            "split_seed": int(case_seed),
            "study": "nsq-kumar2024-v1",
        }
    )
    authority = LongitudinalCaseAuthority.from_split(
        split,
        case_id=(
            f"{KUMAR2024_DATASET_ID}/subject-{subject}/session-{target_session}/"
            f"split-{case_seed}"
        ),
        history_policy="prior",
        observed_group_order=observed,
        case_metadata=metadata,
    )
    authority.restore(data)
    return authority


def _method_factories(
    *,
    config: Kumar2024StudyConfig,
    sample_rate_hz: float,
) -> tuple[Any, ...]:
    from neuros.foundation_models.qualification_baselines import (
        MNECSPLDAFactory,
        UpstreamBraindecodeFactory,
    )

    factories: list[Any] = []
    if "mne-csp-lda" in config.methods:
        factories.append(MNECSPLDAFactory(n_components=config.csp_components))
    common = {
        "sample_rate_hz": float(sample_rate_hz),
        "learning_rate": config.braindecode_learning_rate,
        "weight_decay": config.braindecode_weight_decay,
        "n_epochs": config.braindecode_epochs,
        "batch_size": config.braindecode_batch_size,
        "device": config.device,
        "random_state": config.braindecode_random_state,
    }
    if "braindecode-eegnet" in config.methods:
        factories.append(UpstreamBraindecodeFactory(model_name="EEGNet", **common))
    if "braindecode-eegconformer" in config.methods:
        factories.append(UpstreamBraindecodeFactory(model_name="EEGConformer", **common))
    return tuple(factories)


def _flatten_result_row(row: Any, authority: Any) -> dict[str, Any]:
    payload = row.to_dict()
    score = payload.pop("score")
    metadata = dict(authority.case_metadata)
    result: dict[str, Any] = {
        "subject": int(metadata["subject"]),
        "original_protocol": metadata["original_protocol"],
        "held_out_session": metadata["held_out_session"],
        "source_sessions": json.dumps(list(authority.source_group_values), separators=(",", ":")),
        "history_policy": authority.history_policy,
        **payload,
    }
    if score is None:
        for metric in (
            "balanced_accuracy",
            "accuracy",
            "roc_auc",
            "brier_score",
            "expected_calibration_error",
        ):
            result[metric] = None
            result[f"{metric}_availability"] = "unavailable_run_failure"
    else:
        for metric, value in score["metrics"].items():
            result[metric] = value
            result[f"{metric}_availability"] = score["availability"][metric]
    if row.inference_s is None:
        result["inference_ms_per_trial"] = None
    else:
        result["inference_ms_per_trial"] = float(
            1000.0 * row.inference_s / max(row.evaluation_samples, 1)
        )
    return result


def _mean(values: Iterable[float]) -> float | None:
    numbers = np.asarray(list(values), dtype=np.float64)
    numbers = numbers[np.isfinite(numbers)]
    return None if len(numbers) == 0 else float(np.mean(numbers))


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


def _participant_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[int, float]:
    by_participant: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "success" or row.get(metric) is None:
            continue
        value = float(row[metric])
        if math.isfinite(value):
            by_participant[int(row["subject"])].append(value)
    return {
        participant: float(np.mean(values))
        for participant, values in by_participant.items()
        if values
    }


def _normalized_trapezoid(xs: Sequence[int], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("frontier AUC requires aligned x/y values with at least two points")
    area = 0.0
    for index in range(len(xs) - 1):
        width = float(xs[index + 1] - xs[index])
        area += width * 0.5 * (float(ys[index]) + float(ys[index + 1]))
    span = float(xs[-1] - xs[0])
    return float(area / span) if span > 0 else float(ys[0])


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Kumar2024StudyConfig,
) -> dict[str, Any]:
    """Participant-level deterministic analysis with failure preservation.

    Session rows are first aggregated within participant. Bootstrap resampling is
    therefore performed over participants, never trials or sessions as if they
    were independent people.
    """

    methods = tuple(config.methods)
    budgets = tuple(config.budgets_per_class)
    performance: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        for budget_index, budget in enumerate(budgets):
            group = [
                row
                for row in rows
                if row["method_id"] == method
                and int(row["calibration_per_class"]) == budget
            ]
            participants = _participant_metric(group, metric="balanced_accuracy")
            estimate = _bootstrap_mean_ci(
                participants,
                seed=_stable_seed(
                    config.analysis_seed,
                    "performance",
                    method_index,
                    budget_index,
                ),
                replicates=config.analysis_bootstrap_replicates,
            )
            failures = Counter(str(row["status"]) for row in group if row["status"] != "success")
            cohort_summary: dict[str, Any] = {}
            for cohort in ("GR", "PAR"):
                cohort_rows = [row for row in group if row["original_protocol"] == cohort]
                cohort_values = _participant_metric(cohort_rows, metric="balanced_accuracy")
                cohort_summary[cohort] = {
                    "n_participants": len(cohort_values),
                    "mean_balanced_accuracy": _mean(cohort_values.values()),
                }
            performance.append(
                {
                    "method_id": method,
                    "calibration_per_class": budget,
                    "attempted_rows": len(group),
                    "successful_rows": sum(row["status"] == "success" for row in group),
                    "failure_status_counts": dict(sorted(failures.items())),
                    "participant_level_balanced_accuracy": estimate,
                    "cohort_descriptive": cohort_summary,
                }
            )

    frontier_auc: list[dict[str, Any]] = []
    frontier_participant_values: dict[str, dict[int, float]] = {}
    expected_sessions = set(config.target_sessions)
    for method_index, method in enumerate(methods):
        by_case_budget: dict[tuple[int, str], dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            if row["method_id"] != method or row["status"] != "success":
                continue
            value = row.get("balanced_accuracy")
            if value is None:
                continue
            key = (int(row["subject"]), str(row["held_out_session"]))
            by_case_budget[key][int(row["calibration_per_class"])].append(float(value))

        complete_case_auc: dict[tuple[int, str], float] = {}
        for key, budget_values in by_case_budget.items():
            if any(budget not in budget_values for budget in budgets):
                continue
            ys = [float(np.mean(budget_values[budget])) for budget in budgets]
            complete_case_auc[key] = _normalized_trapezoid(budgets, ys)

        by_participant: dict[int, list[float]] = defaultdict(list)
        complete_sessions: dict[int, set[str]] = defaultdict(set)
        for (participant, session), value in complete_case_auc.items():
            by_participant[participant].append(value)
            complete_sessions[participant].add(session)

        auc_values = {
            participant: float(np.mean(values))
            for participant, values in by_participant.items()
            if complete_sessions[participant] == expected_sessions
        }
        frontier_participant_values[method] = auc_values
        frontier_auc.append(
            {
                "method_id": method,
                "normalized_balanced_accuracy_frontier_auc": _bootstrap_mean_ci(
                    auc_values,
                    seed=_stable_seed(config.analysis_seed, "auc", method_index),
                    replicates=config.analysis_bootstrap_replicates,
                ),
                "complete_frontier_participants": sorted(auc_values),
                "complete_frontier_subject_session_cases": [
                    [participant, session]
                    for participant, session in sorted(complete_case_auc)
                ],
                "pairing_rule": (
                    "a subject-session contributes only if every declared calibration budget "
                    "succeeds; a participant contributes only if every declared target session "
                    "has a complete subject-session frontier"
                ),
            }
        )

    paired: list[dict[str, Any]] = []
    for left_index, left in enumerate(methods):
        for right_index in range(left_index + 1, len(methods)):
            right = methods[right_index]
            for budget in budgets:
                left_rows = {
                    (int(row["subject"]), str(row["held_out_session"])): float(
                        row["balanced_accuracy"]
                    )
                    for row in rows
                    if row["method_id"] == left
                    and int(row["calibration_per_class"]) == budget
                    and row["status"] == "success"
                    and row.get("balanced_accuracy") is not None
                }
                right_rows = {
                    (int(row["subject"]), str(row["held_out_session"])): float(
                        row["balanced_accuracy"]
                    )
                    for row in rows
                    if row["method_id"] == right
                    and int(row["calibration_per_class"]) == budget
                    and row["status"] == "success"
                    and row.get("balanced_accuracy") is not None
                }
                matched = sorted(set(left_rows) & set(right_rows))
                by_participant: dict[int, list[float]] = defaultdict(list)
                for key in matched:
                    by_participant[key[0]].append(left_rows[key] - right_rows[key])
                participant_differences = {
                    participant: float(np.mean(values))
                    for participant, values in by_participant.items()
                }
                paired.append(
                    {
                        "left_method": left,
                        "right_method": right,
                        "calibration_per_class": budget,
                        "matched_subject_session_cases": len(matched),
                        "left_minus_right_balanced_accuracy": _bootstrap_mean_ci(
                            participant_differences,
                            seed=_stable_seed(
                                config.analysis_seed,
                                "paired",
                                left_index,
                                right_index,
                                budget,
                            ),
                            replicates=config.analysis_bootstrap_replicates,
                        ),
                    }
                )

    paired_frontier_auc: list[dict[str, Any]] = []
    for left_index, left in enumerate(methods):
        for right_index in range(left_index + 1, len(methods)):
            right = methods[right_index]
            left_values = frontier_participant_values.get(left, {})
            right_values = frontier_participant_values.get(right, {})
            matched_participants = sorted(set(left_values) & set(right_values))
            differences = {
                participant: left_values[participant] - right_values[participant]
                for participant in matched_participants
            }
            paired_frontier_auc.append(
                {
                    "left_method": left,
                    "right_method": right,
                    "matched_complete_frontier_participants": matched_participants,
                    "left_minus_right_normalized_balanced_accuracy_frontier_auc": (
                        _bootstrap_mean_ci(
                            differences,
                            seed=_stable_seed(
                                config.analysis_seed,
                                "paired-frontier-auc",
                                left_index,
                                right_index,
                            ),
                            replicates=config.analysis_bootstrap_replicates,
                        )
                    ),
                }
            )

    return {
        "schema_version": 1,
        "independent_inferential_unit": "participant",
        "session_handling": "aggregate within participant before participant-level inference",
        "uncertainty": {
            "method": "nonparametric participant bootstrap",
            "replicates": config.analysis_bootstrap_replicates,
            "seed": config.analysis_seed,
        },
        "primary_metric": "balanced_accuracy",
        "primary_study_endpoint": "paired_normalized_balanced_accuracy_frontier_auc",
        "performance": performance,
        "calibration_efficiency": frontier_auc,
        "paired_method_differences": paired,
        "paired_calibration_efficiency": paired_frontier_auc,
        "cohort_policy": (
            "GR/PAR summaries are descriptive context only; no causal treatment comparison is claimed"
        ),
        "failure_policy": "all attempted NSQ rows remain in results and failure counts",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty Kumar2024 NSQ result table")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            normalized: dict[str, Any] = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
                normalized[key] = value
            writer.writerow(normalized)


def _render_report(
    *,
    config: Kumar2024StudyConfig,
    lineage: Any,
    protocol: Any,
    preprocessing_authority: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> str:
    status_counts = Counter(str(row["status"]) for row in rows)
    lines = [
        "# NSQ Kumar2024 v1",
        "",
        "This is a **neurOS longitudinal re-evaluation of the MOABB Kumar2024 bar-feedback subset**.",
        "It is **not** a reproduction of the original online GR/PAR intervention.",
        "",
        f"- Profile: `{config.profile}`",
        f"- Dataset lineage SHA-256: `{lineage.lineage_sha256}`",
        f"- Protocol SHA-256: `{protocol.sha256}`",
        f"- Preprocessing authority SHA-256: `{preprocessing_authority['sha256']}`",
        f"- Participants: {', '.join(str(value) for value in config.subjects)}",
        f"- Target sessions: {', '.join(config.target_sessions)}",
        f"- Calibration budgets / class: {', '.join(str(value) for value in config.budgets_per_class)}",
        f"- Methods: {', '.join(config.methods)}",
        "",
        "## Scientific question",
        "",
        "> Under identical prospective longitudinal authority, how much task performance can each method achieve as a function of per-user labeled calibration cost?",
        "",
        "Prior sessions are the only source history. Target-session calibration rows are nested and labeled. The final assessment rows are frozen once and never enter fitting, adaptation, preprocessing fit, or state selection.",
        "",
        "## Result integrity",
        "",
        f"Attempt statuses: `{dict(sorted(status_counts.items()))}`.",
        "Failures, unavailable methods, OOMs, and nonconvergence remain in the raw frontier rather than disappearing from aggregates.",
        "",
        "Participant is the inferential unit. Session results are aggregated within participant before uncertainty is estimated by participant bootstrap.",
        "GR/PAR summaries are descriptive context and are not interpreted as a causal comparison of the original interventions.",
        "",
        "## Evidence boundary",
        "",
        "A successful bundle establishes a reproducible comparative result for the exact MOABB/neurOS protocol identified above. It does not reproduce the original online adaptive intervention, establish physiological mechanism, qualify hardware, prove online BCI efficacy, or establish clinical benefit.",
        "",
        "ORION is intentionally absent from this first superiority comparison. It may enter only after these external baselines are frozen and qualified.",
        "",
        f"Analysis artifact schema: `{analysis['schema_version']}`.",
    ]
    return "\n".join(lines) + "\n"


def _prepare_output(output: Path, *, overwrite: bool) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    managed = [output / name for name in (*_BUNDLE_FILES, "artifact_hashes.json")]
    existing = [path for path in managed if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing Kumar2024 evidence artifacts: "
            + ", ".join(path.name for path in existing)
        )
    return output


def _seal_bundle(output: Path) -> dict[str, Any]:
    files = {name: _file_sha256(output / name) for name in _BUNDLE_FILES}
    root = _identity_sha256("neuros.nsq_kumar2024_bundle.v1", {"files": files})
    payload = {
        "schema_version": 1,
        "files": files,
        "bundle_sha256": root,
    }
    _json_dump(output / "artifact_hashes.json", payload)
    return payload


def verify_bundle(output: str | Path) -> dict[str, Any]:
    """Verify a sealed study bundle without rerunning model training."""

    root = Path(output).resolve()
    manifest_path = root / "artifact_hashes.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError("invalid Kumar2024 artifact hash manifest")
    actual: dict[str, str] = {}
    for name, expected in payload["files"].items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing Kumar2024 bundle file: {name}")
        digest = _file_sha256(path)
        if digest != expected:
            raise ValueError(f"Kumar2024 bundle hash mismatch for {name}")
        actual[name] = digest
    expected_root = _identity_sha256(
        "neuros.nsq_kumar2024_bundle.v1", {"files": actual}
    )
    if payload.get("bundle_sha256") != expected_root:
        raise ValueError("Kumar2024 bundle root SHA-256 does not match file manifest")
    return {
        "verified": True,
        "bundle_sha256": expected_root,
        "files": actual,
    }


def run_study(
    output: str | Path,
    *,
    config: Kumar2024StudyConfig | None = None,
    preprocessing: Kumar2024PreprocessingSpec | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute and seal the canonical external-baseline Kumar2024 study."""

    from neuros.foundation_models.longitudinal import ordered_group_values
    from neuros.foundation_models.moabb_epochs import collect_moabb_epochs
    from neuros.foundation_models.moabb_longitudinal import (
        build_moabb_longitudinal_dataset,
        validate_observed_sessions,
    )
    from neuros.foundation_models.qualification_runner import (
        QualificationExecutionContext,
        run_external_qualification_case,
    )

    config = config or pilot_config()
    preprocessing = preprocessing or Kumar2024PreprocessingSpec()
    output_path = _prepare_output(Path(output), overwrite=overwrite)
    versions = _runtime_versions()
    dataset_spec, dataset, paradigm = build_moabb_longitudinal_dataset(
        KUMAR2024_DATASET_KEY,
        fmin=preprocessing.fmin_hz,
        fmax=preprocessing.fmax_hz,
        resample=preprocessing.resample_hz,
    )

    first_subject = config.subjects[0]
    first_data, first_descriptor = collect_moabb_epochs(
        dataset,
        paradigm,
        subjects=[first_subject],
        dataset_id=KUMAR2024_DATASET_ID,
    )
    validate_observed_sessions(
        dataset_spec,
        ordered_group_values(first_data, split_unit="session"),
    )
    preprocessing_authority = _preprocessing_authority(
        preprocessing,
        first_descriptor,
        versions,
    )
    lineage = build_dataset_lineage(
        config=config,
        preprocessing_authority=preprocessing_authority,
        versions=versions,
    )
    protocol = build_protocol(
        config=config,
        dataset_lineage=lineage,
        preprocessing_authority_sha256=preprocessing_authority["sha256"],
    )
    context = QualificationExecutionContext(
        observed_dataset_lineage_sha256=lineage.lineage_sha256,
        preprocessing_authority_sha256s=(preprocessing_authority["sha256"],),
        metadata={
            "study": "nsq-kumar2024-v1",
            "moabb_version": versions.get("moabb"),
            "mne_version": versions.get("mne"),
            "processed_signal_contract_sha256": first_descriptor.signal_contract_sha256,
        },
    )
    factories = _method_factories(
        config=config,
        sample_rate_hz=first_descriptor.sampling_rate_hz,
    )

    authorities: list[Any] = []
    case_results: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    subject_descriptors: dict[str, Any] = {}
    preloaded = {first_subject: (first_data, first_descriptor)}

    for subject in config.subjects:
        if subject in preloaded:
            data, descriptor = preloaded[subject]
        else:
            data, descriptor = collect_moabb_epochs(
                dataset,
                paradigm,
                subjects=[subject],
                dataset_id=KUMAR2024_DATASET_ID,
            )
        if descriptor.signal_contract_sha256 != first_descriptor.signal_contract_sha256:
            raise RuntimeError(
                "processed MOABB signal contract changed across participants: "
                f"subject={subject}, reference={first_descriptor.signal_contract_sha256}, "
                f"observed={descriptor.signal_contract_sha256}"
            )
        subject_descriptors[str(subject)] = {
            **descriptor.to_dict(),
            "descriptor_sha256": descriptor.sha256,
        }
        observed = validate_observed_sessions(
            dataset_spec,
            ordered_group_values(data, split_unit="session"),
        )
        if observed != KUMAR2024_EXPECTED_SESSIONS:
            raise RuntimeError(
                f"Kumar2024 chronology changed for subject {subject}: {observed}"
            )

        for target_session in config.target_sessions:
            authority = _make_case_authority(
                data=data,
                dataset_spec=dataset_spec,
                subject=subject,
                target_session=target_session,
                config=config,
            )
            authorities.append(authority)
            for factory in factories:
                result = run_external_qualification_case(
                    data,
                    authority,
                    protocol,
                    factory,
                    execution_context=context,
                )
                case_results.append(
                    {
                        "subject": subject,
                        "original_protocol": authority.case_metadata["original_protocol"],
                        "held_out_session": target_session,
                        "method_spec": {
                            **factory.method_spec.to_dict(),
                            "method_spec_sha256": factory.method_spec.sha256,
                        },
                        "result": result.to_dict(),
                    }
                )
                flat_rows.extend(_flatten_result_row(row, authority) for row in result.rows)

    analysis = summarize_rows(flat_rows, config=config)
    method_specs = []
    for factory in factories:
        spec = factory.method_spec
        method_specs.append({**spec.to_dict(), "method_spec_sha256": spec.sha256})
    study_identity_payload = {
        "config_sha256": config.sha256,
        "dataset_lineage_sha256": lineage.lineage_sha256,
        "protocol_sha256": protocol.sha256,
        "preprocessing_authority_sha256": preprocessing_authority["sha256"],
        "method_spec_sha256s": [item["method_spec_sha256"] for item in method_specs],
        "case_authority_sha256s": [item.authority_sha256 for item in authorities],
    }
    manifest = {
        "schema_version": 1,
        "study": "nsq-kumar2024-v1",
        "evidence_tier": "real_dataset",
        "study_sha256": _identity_sha256(
            "neuros.nsq_kumar2024_study.v1", study_identity_payload
        ),
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "preprocessing_authority": preprocessing_authority,
        "dataset_lineage": lineage.to_dict(),
        "protocol": {**protocol.to_dict(), "protocol_sha256": protocol.sha256},
        "execution_context": {**context.to_dict(), "execution_context_sha256": context.sha256},
        "method_specs": method_specs,
        "case_authority_sha256s": [item.authority_sha256 for item in authorities],
        "case_result_sha256s": [item["result"]["result_sha256"] for item in case_results],
        "subject_epoch_descriptors": subject_descriptors,
        "package_versions": versions,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "claim_boundary": (
            "offline comparative evidence for this exact MOABB bar-feedback subset and "
            "prospective longitudinal protocol only"
        ),
        "not_claimed": [
            "reproduction of the original online GR/PAR intervention",
            "physiological mechanism",
            "hardware qualification",
            "online BCI efficacy",
            "clinical benefit",
            "ORION superiority",
        ],
    }

    _json_dump(output_path / "study_manifest.json", manifest)
    _json_dump(
        output_path / "case_authorities.json",
        {
            "schema_version": 1,
            "authorities": [item.to_dict() for item in authorities],
        },
    )
    _json_dump(
        output_path / "case_results.json",
        {"schema_version": 1, "case_results": case_results},
    )
    _write_csv(output_path / "results.csv", flat_rows)
    _json_dump(output_path / "analysis.json", analysis)
    (output_path / "report.md").write_text(
        _render_report(
            config=config,
            lineage=lineage,
            protocol=protocol,
            preprocessing_authority=preprocessing_authority,
            rows=flat_rows,
            analysis=analysis,
        ),
        encoding="utf-8",
    )
    sealed = _seal_bundle(output_path)
    verified = verify_bundle(output_path)
    return {
        "study_sha256": manifest["study_sha256"],
        "bundle_sha256": sealed["bundle_sha256"],
        "protocol_sha256": protocol.sha256,
        "dataset_lineage_sha256": lineage.lineage_sha256,
        "cases": len(authorities),
        "result_rows": len(flat_rows),
        "verified": verified["verified"],
        "output": str(output_path),
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parse_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuros-nsq-kumar2024",
        description=(
            "Run or verify the canonical NSQ Kumar2024 external-baseline study. "
            "The default pilot is an execution/provenance gate, not a headline comparison."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("pilot",), default="pilot")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--subjects", type=_parse_ints, default=None)
    parser.add_argument("--sessions", type=_parse_strings, default=None)
    parser.add_argument("--methods", type=_parse_strings, default=None)
    parser.add_argument("--braindecode-epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument("--resample", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        print(json.dumps(verify_bundle(args.output), indent=2, sort_keys=True))
        return 0
    base = pilot_config()
    config = Kumar2024StudyConfig(
        subjects=base.subjects if args.subjects is None else args.subjects,
        target_sessions=base.target_sessions if args.sessions is None else args.sessions,
        budgets_per_class=base.budgets_per_class,
        methods=base.methods if args.methods is None else args.methods,
        split_seed=base.split_seed,
        evaluation_fraction=base.evaluation_fraction,
        csp_components=base.csp_components,
        braindecode_epochs=(
            base.braindecode_epochs
            if args.braindecode_epochs is None
            else args.braindecode_epochs
        ),
        braindecode_batch_size=base.braindecode_batch_size,
        braindecode_learning_rate=base.braindecode_learning_rate,
        braindecode_weight_decay=base.braindecode_weight_decay,
        braindecode_random_state=base.braindecode_random_state,
        device=base.device if args.device is None else args.device,
        analysis_bootstrap_replicates=base.analysis_bootstrap_replicates,
        analysis_seed=base.analysis_seed,
        profile=args.profile,
    )
    preprocessing = Kumar2024PreprocessingSpec(
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        resample_hz=args.resample,
    )
    result = run_study(
        args.output,
        config=config,
        preprocessing=preprocessing,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "KUMAR2024_ALL_SUBJECTS",
    "KUMAR2024_DATASET_ID",
    "KUMAR2024_DEFAULT_BUDGETS",
    "KUMAR2024_DEFAULT_METHODS",
    "KUMAR2024_EXPECTED_SESSIONS",
    "KUMAR2024_TARGET_SESSIONS",
    "Kumar2024PreprocessingSpec",
    "Kumar2024StudyConfig",
    "build_dataset_lineage",
    "build_parser",
    "build_protocol",
    "full_config",
    "main",
    "pilot_config",
    "run_study",
    "summarize_rows",
    "verify_bundle",
]
