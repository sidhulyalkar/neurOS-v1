"""Promoted Kumar2024 execution authority.

This module turns the preregistered comparison plan into a resumable, content-
addressed execution graph without running models or touching neural data.

There are deliberately two stages:

1. :class:`PromotedExecutionTemplate` expands the statistical authority into
   atomic worker shards. One shard owns exactly one
   participant/session/split/method/model-seed realization and *all* calibration
   budgets, so complete-frontier semantics cannot be weakened by scheduling.
2. :class:`PromotedExecutionPlan` binds that template to one exact materialized
   study, protocol, preprocessing authority, source revision, and method-spec
   identities before any promoted final-assessment execution is allowed.

Workers serialize :class:`PromotedShardResult` envelopes. Final assembly fails
closed on missing, duplicate, foreign, or authority-drifting shards before the
existing comparison authority is asked to summarize rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .kumar2024_comparison import (
    Kumar2024ComparisonPlan,
    promoted_external_floor_plan,
    summarize_promoted_rows,
    validate_promoted_rows,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("promoted execution identity cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not valid promoted execution identity values")
        return _canonical(value.tolist())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("promoted execution mapping keys must be non-empty")
            if key in normalized:
                raise ValueError("promoted execution mapping keys collide after normalization")
            normalized[key] = _canonical(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise TypeError(
        "promoted execution identity must use deterministic JSON-compatible values; "
        f"got {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    normalized = _canonical(value)
    if isinstance(normalized, dict):
        return MappingProxyType(
            {key: _freeze(item) for key, item in normalized.items()}
        )
    if isinstance(normalized, list):
        return tuple(_freeze(item) for item in normalized)
    return normalized


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _identity_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"schema": schema, "payload": _canonical(payload)},
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


def _sha256(name: str, value: Any) -> str:
    text = _nonempty(name, value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _git_sha(value: Any) -> str:
    text = _nonempty("source_revision", value).lower()
    if not _GIT_SHA_RE.fullmatch(text):
        raise ValueError("source_revision must be a 40-character lowercase Git SHA")
    return text


def _exact_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer without coercion")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True, slots=True)
class PromotedExecutionShardSpec:
    """One atomic promoted worker assignment.

    Every declared calibration budget belongs to the same shard. A scheduler
    therefore cannot turn a partial calibration frontier into five independent
    "successful" jobs.
    """

    comparison_plan_sha256: str
    subject: int
    target_session: str
    split_seed: int
    method_id: str
    model_seed: int | None
    budgets_per_class: tuple[int, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedExecutionShardSpec schema_version must be 1")
        object.__setattr__(
            self,
            "comparison_plan_sha256",
            _sha256("comparison_plan_sha256", self.comparison_plan_sha256),
        )
        object.__setattr__(self, "subject", _exact_nonnegative_int("subject", self.subject))
        object.__setattr__(self, "target_session", _nonempty("target_session", self.target_session))
        object.__setattr__(self, "split_seed", _exact_nonnegative_int("split_seed", self.split_seed))
        object.__setattr__(self, "method_id", _nonempty("method_id", self.method_id))
        if self.model_seed is not None:
            object.__setattr__(
                self,
                "model_seed",
                _exact_nonnegative_int("model_seed", self.model_seed),
            )
        budgets = tuple(
            _exact_nonnegative_int("calibration budget", value)
            for value in self.budgets_per_class
        )
        if not budgets or budgets != tuple(sorted(set(budgets))) or budgets[0] != 0:
            raise ValueError(
                "budgets_per_class must be unique, increasing, and start at zero"
            )
        object.__setattr__(self, "budgets_per_class", budgets)

    @property
    def method_realization_key(self) -> str:
        if self.model_seed is None:
            return f"{self.method_id}/deterministic"
        return f"{self.method_id}/model-seed-{self.model_seed}"

    @property
    def shard_id(self) -> str:
        seed = "deterministic" if self.model_seed is None else f"seed-{self.model_seed}"
        return (
            f"subject-{self.subject:02d}/session-{self.target_session}/"
            f"split-{self.split_seed}/{self.method_id}/{seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparison_plan_sha256": self.comparison_plan_sha256,
            "shard_id": self.shard_id,
            "subject": self.subject,
            "target_session": self.target_session,
            "split_seed": self.split_seed,
            "method_id": self.method_id,
            "model_seed": self.model_seed,
            "method_realization_key": self.method_realization_key,
            "budgets_per_class": list(self.budgets_per_class),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_execution_shard_spec.v1",
            self.to_dict(),
        )


@dataclass(frozen=True, slots=True)
class PromotedExecutionTemplate:
    """Data-independent expansion of one comparison authority into worker shards."""

    comparison_plan_sha256: str
    shards: tuple[PromotedExecutionShardSpec, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedExecutionTemplate schema_version must be 1")
        plan_sha = _sha256("comparison_plan_sha256", self.comparison_plan_sha256)
        shards = tuple(self.shards)
        if not shards:
            raise ValueError("promoted execution template requires at least one shard")
        if any(not isinstance(item, PromotedExecutionShardSpec) for item in shards):
            raise TypeError("shards must contain PromotedExecutionShardSpec objects")
        if any(item.comparison_plan_sha256 != plan_sha for item in shards):
            raise ValueError("every shard must bind the template comparison plan")
        ids = [item.shard_id for item in shards]
        hashes = [item.sha256 for item in shards]
        if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
            raise ValueError("promoted execution shards must be unique")
        canonical = tuple(
            sorted(
                shards,
                key=lambda item: (
                    item.subject,
                    item.target_session,
                    item.split_seed,
                    item.method_id,
                    -1 if item.model_seed is None else item.model_seed,
                ),
            )
        )
        object.__setattr__(self, "comparison_plan_sha256", plan_sha)
        object.__setattr__(self, "shards", canonical)

    @property
    def method_realization_keys(self) -> tuple[str, ...]:
        return tuple(sorted({item.method_realization_key for item in self.shards}))

    @property
    def expected_fit_attempts(self) -> int:
        return sum(len(item.budgets_per_class) for item in self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparison_plan_sha256": self.comparison_plan_sha256,
            "expected_shards": len(self.shards),
            "expected_fit_attempts": self.expected_fit_attempts,
            "method_realization_keys": list(self.method_realization_keys),
            "shards": [
                {**item.to_dict(), "shard_spec_sha256": item.sha256}
                for item in self.shards
            ],
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_execution_template.v1",
            self.to_dict(),
        )


def build_promoted_execution_template(
    plan: Kumar2024ComparisonPlan | None = None,
) -> PromotedExecutionTemplate:
    """Expand the comparison plan without loading data or instantiating a model."""

    plan = plan or promoted_external_floor_plan()
    shards: list[PromotedExecutionShardSpec] = []
    for subject in plan.subjects:
        for session in plan.target_sessions:
            for split_seed in plan.split_seeds:
                for policy in plan.method_seed_policies:
                    for model_seed in policy.realization_model_seeds:
                        shards.append(
                            PromotedExecutionShardSpec(
                                comparison_plan_sha256=plan.sha256,
                                subject=subject,
                                target_session=session,
                                split_seed=split_seed,
                                method_id=policy.method_id,
                                model_seed=model_seed,
                                budgets_per_class=plan.budgets_per_class,
                            )
                        )
    return PromotedExecutionTemplate(
        comparison_plan_sha256=plan.sha256,
        shards=tuple(shards),
    )


@dataclass(frozen=True, slots=True)
class PromotedExecutionBinding:
    """Materialized scientific authorities required before workers may execute."""

    comparison_plan_sha256: str
    template_sha256: str
    study_materialization_sha256: str
    environment_authority_sha256: str
    raw_materialization_sha256: str
    dataset_lineage_sha256: str
    protocol_sha256: str
    preprocessing_authority_sha256: str
    source_revision: str
    case_authority_sha256_by_case: tuple[tuple[int, str, int, str], ...]
    method_spec_sha256_by_realization: tuple[tuple[str, str], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedExecutionBinding schema_version must be 1")
        for name in (
            "comparison_plan_sha256",
            "template_sha256",
            "study_materialization_sha256",
            "environment_authority_sha256",
            "raw_materialization_sha256",
            "dataset_lineage_sha256",
            "protocol_sha256",
            "preprocessing_authority_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        object.__setattr__(self, "source_revision", _git_sha(self.source_revision))

        cases: list[tuple[int, str, int, str]] = []
        for raw_case in self.case_authority_sha256_by_case:
            if not isinstance(raw_case, (tuple, list)) or len(raw_case) != 4:
                raise ValueError(
                    "case authority entries must be (subject, session, split_seed, sha256)"
                )
            raw_subject, raw_session, raw_split_seed, raw_sha = raw_case
            cases.append(
                (
                    _exact_nonnegative_int("case subject", raw_subject),
                    _nonempty("case target session", raw_session),
                    _exact_nonnegative_int("case split seed", raw_split_seed),
                    _sha256("case_authority_sha256", raw_sha),
                )
            )
        cases.sort(key=lambda item: (item[0], item[1], item[2]))
        case_keys = [(item[0], item[1], item[2]) for item in cases]
        if not cases or len(set(case_keys)) != len(case_keys):
            raise ValueError("case authority map must be non-empty and unique")
        object.__setattr__(
            self,
            "case_authority_sha256_by_case",
            tuple(cases),
        )

        pairs: list[tuple[str, str]] = []
        for raw_key, raw_sha in self.method_spec_sha256_by_realization:
            pairs.append(
                (
                    _nonempty("method realization key", raw_key),
                    _sha256("method_spec_sha256", raw_sha),
                )
            )
        pairs.sort(key=lambda item: item[0])
        keys = [item[0] for item in pairs]
        if not pairs or len(set(keys)) != len(keys):
            raise ValueError("method spec authorities must be non-empty and unique")
        object.__setattr__(
            self,
            "method_spec_sha256_by_realization",
            tuple(pairs),
        )

    @property
    def case_authority_map(self) -> dict[tuple[int, str, int], str]:
        return {
            (subject, session, split_seed): sha
            for subject, session, split_seed, sha in self.case_authority_sha256_by_case
        }

    @property
    def method_spec_map(self) -> dict[str, str]:
        return dict(self.method_spec_sha256_by_realization)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "comparison_plan_sha256": self.comparison_plan_sha256,
            "template_sha256": self.template_sha256,
            "study_materialization_sha256": self.study_materialization_sha256,
            "environment_authority_sha256": self.environment_authority_sha256,
            "raw_materialization_sha256": self.raw_materialization_sha256,
            "dataset_lineage_sha256": self.dataset_lineage_sha256,
            "protocol_sha256": self.protocol_sha256,
            "preprocessing_authority_sha256": self.preprocessing_authority_sha256,
            "source_revision": self.source_revision,
            "case_authority_sha256_by_case": [
                {
                    "subject": subject,
                    "target_session": session,
                    "split_seed": split_seed,
                    "case_authority_sha256": sha,
                }
                for subject, session, split_seed, sha in self.case_authority_sha256_by_case
            ],
            "method_spec_sha256_by_realization": {
                key: value for key, value in self.method_spec_sha256_by_realization
            },
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_execution_binding.v1",
            self.to_dict(),
        )


@dataclass(frozen=True, slots=True)
class PromotedExecutionPlan:
    """Bound execution graph. Creating this object authorizes scheduling, not claims."""

    template: PromotedExecutionTemplate
    binding: PromotedExecutionBinding
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedExecutionPlan schema_version must be 1")
        if not isinstance(self.template, PromotedExecutionTemplate):
            raise TypeError("template must be PromotedExecutionTemplate")
        if not isinstance(self.binding, PromotedExecutionBinding):
            raise TypeError("binding must be PromotedExecutionBinding")
        if self.binding.comparison_plan_sha256 != self.template.comparison_plan_sha256:
            raise ValueError("binding and template comparison-plan SHA differ")
        if self.binding.template_sha256 != self.template.sha256:
            raise ValueError("binding does not name the exact execution template")
        expected = set(self.template.method_realization_keys)
        observed = set(self.binding.method_spec_map)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                "method-spec authority map does not match template realizations: "
                f"missing={missing}, extra={extra}"
            )

        expected_cases = {
            (item.subject, item.target_session, item.split_seed)
            for item in self.template.shards
        }
        observed_cases = set(self.binding.case_authority_map)
        if observed_cases != expected_cases:
            missing = sorted(expected_cases - observed_cases)
            extra = sorted(observed_cases - expected_cases)
            raise ValueError(
                "case-authority map does not match template cases: "
                f"missing={missing}, extra={extra}"
            )

    @property
    def shard_by_sha256(self) -> dict[str, PromotedExecutionShardSpec]:
        return {item.sha256: item for item in self.template.shards}

    def expected_case_authority_sha256(
        self,
        shard: PromotedExecutionShardSpec,
    ) -> str:
        key = (shard.subject, shard.target_session, shard.split_seed)
        return self.binding.case_authority_map[key]

    def expected_method_spec_sha256(
        self,
        shard: PromotedExecutionShardSpec,
    ) -> str:
        return self.binding.method_spec_map[shard.method_realization_key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "template": {
                **self.template.to_dict(),
                "template_sha256": self.template.sha256,
            },
            "binding": {
                **self.binding.to_dict(),
                "binding_sha256": self.binding.sha256,
            },
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_execution_plan.v1",
            self.to_dict(),
        )


def bind_promoted_execution_template(
    template: PromotedExecutionTemplate,
    *,
    study_materialization_sha256: str,
    environment_authority_sha256: str,
    raw_materialization_sha256: str,
    dataset_lineage_sha256: str,
    protocol_sha256: str,
    preprocessing_authority_sha256: str,
    source_revision: str,
    case_authority_sha256_by_case: Mapping[tuple[int, str, int], str],
    method_spec_sha256_by_realization: Mapping[str, str],
) -> PromotedExecutionPlan:
    """Bind a data-independent template to exact archived scientific authorities."""

    binding = PromotedExecutionBinding(
        comparison_plan_sha256=template.comparison_plan_sha256,
        template_sha256=template.sha256,
        study_materialization_sha256=study_materialization_sha256,
        environment_authority_sha256=environment_authority_sha256,
        raw_materialization_sha256=raw_materialization_sha256,
        dataset_lineage_sha256=dataset_lineage_sha256,
        protocol_sha256=protocol_sha256,
        preprocessing_authority_sha256=preprocessing_authority_sha256,
        source_revision=source_revision,
        case_authority_sha256_by_case=tuple(
            (subject, session, split_seed, sha)
            for (subject, session, split_seed), sha in case_authority_sha256_by_case.items()
        ),
        method_spec_sha256_by_realization=tuple(
            method_spec_sha256_by_realization.items()
        ),
    )
    return PromotedExecutionPlan(template=template, binding=binding)


@dataclass(frozen=True, slots=True)
class PromotedShardResult:
    """Content-addressed result envelope produced by exactly one worker shard."""

    execution_plan_sha256: str
    shard_spec_sha256: str
    comparison_plan_sha256: str
    study_materialization_sha256: str
    environment_authority_sha256: str
    raw_materialization_sha256: str
    dataset_lineage_sha256: str
    protocol_sha256: str
    preprocessing_authority_sha256: str
    case_authority_sha256: str
    method_spec_sha256: str
    rows: tuple[Mapping[str, Any], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedShardResult schema_version must be 1")
        for name in (
            "execution_plan_sha256",
            "shard_spec_sha256",
            "comparison_plan_sha256",
            "study_materialization_sha256",
            "environment_authority_sha256",
            "raw_materialization_sha256",
            "dataset_lineage_sha256",
            "protocol_sha256",
            "preprocessing_authority_sha256",
            "case_authority_sha256",
            "method_spec_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        if any(not isinstance(item, Mapping) for item in self.rows):
            raise TypeError("promoted shard result rows must be mappings")
        rows = tuple(_freeze(dict(item)) for item in self.rows)
        if not rows:
            raise ValueError("promoted shard result must contain attempted budget rows")
        object.__setattr__(self, "rows", rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_plan_sha256": self.execution_plan_sha256,
            "shard_spec_sha256": self.shard_spec_sha256,
            "comparison_plan_sha256": self.comparison_plan_sha256,
            "study_materialization_sha256": self.study_materialization_sha256,
            "environment_authority_sha256": self.environment_authority_sha256,
            "raw_materialization_sha256": self.raw_materialization_sha256,
            "dataset_lineage_sha256": self.dataset_lineage_sha256,
            "protocol_sha256": self.protocol_sha256,
            "preprocessing_authority_sha256": self.preprocessing_authority_sha256,
            "case_authority_sha256": self.case_authority_sha256,
            "method_spec_sha256": self.method_spec_sha256,
            "rows": [_thaw(item) for item in self.rows],
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_shard_result.v1",
            self.to_dict(),
        )


def validate_promoted_shard_result(
    result: PromotedShardResult,
    *,
    execution_plan: PromotedExecutionPlan,
    comparison_plan: Kumar2024ComparisonPlan,
) -> tuple[dict[str, Any], ...]:
    """Validate one worker envelope against every preregistered identity."""

    if result.execution_plan_sha256 != execution_plan.sha256:
        raise ValueError("shard result names a different promoted execution plan")
    if result.comparison_plan_sha256 != comparison_plan.sha256:
        raise ValueError("shard result names a different comparison plan")
    if execution_plan.template.comparison_plan_sha256 != comparison_plan.sha256:
        raise ValueError("execution plan and supplied comparison plan differ")
    shard = execution_plan.shard_by_sha256.get(result.shard_spec_sha256)
    if shard is None:
        raise ValueError("shard result names an unknown shard specification")
    binding = execution_plan.binding
    for name in (
        "study_materialization_sha256",
        "environment_authority_sha256",
        "raw_materialization_sha256",
        "dataset_lineage_sha256",
        "protocol_sha256",
        "preprocessing_authority_sha256",
    ):
        if getattr(result, name) != getattr(binding, name):
            raise ValueError(f"shard result {name} differs from execution authority")
    expected_method_sha = execution_plan.expected_method_spec_sha256(shard)
    if result.method_spec_sha256 != expected_method_sha:
        raise ValueError("shard result method-spec SHA differs from execution authority")
    expected_case_sha = execution_plan.expected_case_authority_sha256(shard)
    if result.case_authority_sha256 != expected_case_sha:
        raise ValueError("shard result case-authority SHA differs from execution authority")

    rows = tuple(_thaw(item) for item in result.rows)
    budgets = sorted(int(item.get("calibration_per_class", -1)) for item in rows)
    if budgets != list(shard.budgets_per_class):
        raise ValueError(
            "shard must preserve exactly one attempted row for every calibration budget"
        )
    for row in rows:
        if str(row.get("method_id")) != shard.method_id:
            raise ValueError("row method_id differs from shard specification")
        if int(row.get("subject")) != shard.subject:
            raise ValueError("row subject differs from shard specification")
        if str(row.get("held_out_session")) != shard.target_session:
            raise ValueError("row target session differs from shard specification")
        if int(row.get("split_seed")) != shard.split_seed:
            raise ValueError("row split seed differs from shard specification")
        raw_model_seed = row.get("model_seed")
        if raw_model_seed != shard.model_seed:
            raise ValueError("row model seed differs from shard specification")
        if row.get("case_authority_sha256") != result.case_authority_sha256:
            raise ValueError("row case authority differs from shard envelope")

    # Reuse the preregistered semantic validator even at shard granularity.
    return validate_promoted_rows(rows, plan=comparison_plan)


def assemble_promoted_execution(
    shard_results: Sequence[PromotedShardResult],
    *,
    execution_plan: PromotedExecutionPlan,
    comparison_plan: Kumar2024ComparisonPlan,
) -> dict[str, Any]:
    """Fail closed on orchestration drift, then run participant-level analysis."""

    if execution_plan.template.comparison_plan_sha256 != comparison_plan.sha256:
        raise ValueError("execution plan and comparison plan differ")
    expected = execution_plan.shard_by_sha256
    observed: dict[str, PromotedShardResult] = {}
    for result in shard_results:
        if not isinstance(result, PromotedShardResult):
            raise TypeError("shard_results must contain PromotedShardResult objects")
        key = result.shard_spec_sha256
        if key in observed:
            raise ValueError(f"duplicate promoted shard result for {key}")
        if key not in expected:
            raise ValueError(f"unknown promoted shard result for {key}")
        observed[key] = result

    missing = sorted(set(expected) - set(observed))
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"promoted execution is missing {len(missing)} expected shard artifacts; "
            f"first_missing={preview}"
        )

    rows: list[dict[str, Any]] = []
    shard_hashes: list[str] = []
    for shard in execution_plan.template.shards:
        result = observed[shard.sha256]
        validated = validate_promoted_shard_result(
            result,
            execution_plan=execution_plan,
            comparison_plan=comparison_plan,
        )
        rows.extend(validated)
        shard_hashes.append(result.sha256)

    analysis = summarize_promoted_rows(rows, plan=comparison_plan)
    return {
        "schema_version": 1,
        "comparison_plan_sha256": comparison_plan.sha256,
        "execution_plan_sha256": execution_plan.sha256,
        "study_materialization_sha256": execution_plan.binding.study_materialization_sha256,
        "expected_shards": len(execution_plan.template.shards),
        "received_shards": len(shard_results),
        "attempted_rows": len(rows),
        "shard_result_sha256s": shard_hashes,
        "analysis": analysis,
    }


__all__ = [
    "PromotedExecutionBinding",
    "PromotedExecutionPlan",
    "PromotedExecutionShardSpec",
    "PromotedExecutionTemplate",
    "PromotedShardResult",
    "assemble_promoted_execution",
    "bind_promoted_execution_template",
    "build_promoted_execution_template",
    "validate_promoted_shard_result",
]
