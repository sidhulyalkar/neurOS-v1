"""Top-level Scientific Authority v2 study and claim ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .common import (
    ClaimQualification,
    EvidenceDomain,
    OverlapStatus,
    canonical_sha256,
    display_fingerprint,
    freeze_json,
    nonempty,
    require_sha256,
    thaw_json,
)
from .evaluation import FailurePreservingResultSet, MetricSpec, RepeatedMeasuresAuthority
from .lineage import DatasetLineage, ModelLineage, PretrainingOverlapAudit
from .observations import ObservationSetAuthority, PreprocessingFitAuthority, TargetObservationBudget


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    claim_id: str
    domain: EvidenceDomain
    scope: str
    qualification: ClaimQualification
    evidence_sha256s: tuple[str, ...]
    model_id: str | None = None
    evaluation_dataset_id: str | None = None
    target_budget_id: str | None = None
    zero_shot_claim: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("EvidenceClaim schema_version must be 2")
        object.__setattr__(self, "claim_id", nonempty("claim_id", self.claim_id))
        object.__setattr__(self, "scope", nonempty("scope", self.scope))
        shas = tuple(require_sha256("evidence SHA-256", value) for value in self.evidence_sha256s)
        if not shas:
            raise ValueError("evidence_sha256s must be non-empty")
        if len(set(shas)) != len(shas):
            raise ValueError("evidence_sha256s cannot contain duplicates")
        if not isinstance(self.zero_shot_claim, bool):
            raise ValueError("zero_shot_claim must be boolean")
        if self.model_id is not None:
            object.__setattr__(self, "model_id", nonempty("model_id", self.model_id))
        if self.evaluation_dataset_id is not None:
            object.__setattr__(
                self,
                "evaluation_dataset_id",
                nonempty("evaluation_dataset_id", self.evaluation_dataset_id),
            )
        if self.target_budget_id is not None:
            object.__setattr__(self, "target_budget_id", nonempty("target_budget_id", self.target_budget_id))
        object.__setattr__(self, "evidence_sha256s", shas)
        metadata = freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "domain": self.domain.value,
            "scope": self.scope,
            "qualification": self.qualification.value,
            "evidence_sha256s": list(self.evidence_sha256s),
            "model_id": self.model_id,
            "evaluation_dataset_id": self.evaluation_dataset_id,
            "target_budget_id": self.target_budget_id,
            "zero_shot_claim": self.zero_shot_claim,
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ScientificStudyAuthority:
    """Immutable study ledger for promoted neural comparisons."""

    study_id: str
    protocol_sha256: str
    datasets: tuple[DatasetLineage, ...]
    models: tuple[ModelLineage, ...]
    observations: tuple[ObservationSetAuthority, ...]
    preprocessing: tuple[PreprocessingFitAuthority, ...]
    metrics: tuple[MetricSpec, ...]
    repeated_measures: RepeatedMeasuresAuthority
    overlap_audits: tuple[PretrainingOverlapAudit, ...] = ()
    result_sets: tuple[FailurePreservingResultSet, ...] = ()
    target_budgets: Mapping[str, TargetObservationBudget] = field(default_factory=dict)
    claims: tuple[EvidenceClaim, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ScientificStudyAuthority schema_version must be 2")
        object.__setattr__(self, "study_id", nonempty("study_id", self.study_id))
        object.__setattr__(self, "protocol_sha256", require_sha256("protocol_sha256", self.protocol_sha256))

        datasets_tuple = tuple(self.datasets)
        models_tuple = tuple(self.models)
        observations_tuple = tuple(self.observations)
        preprocessing_tuple = tuple(self.preprocessing)
        metrics_tuple = tuple(self.metrics)
        audits_tuple = tuple(self.overlap_audits)
        result_sets_tuple = tuple(self.result_sets)
        claims_tuple = tuple(self.claims)
        object.__setattr__(self, "datasets", datasets_tuple)
        object.__setattr__(self, "models", models_tuple)
        object.__setattr__(self, "observations", observations_tuple)
        object.__setattr__(self, "preprocessing", preprocessing_tuple)
        object.__setattr__(self, "metrics", metrics_tuple)
        object.__setattr__(self, "overlap_audits", audits_tuple)
        object.__setattr__(self, "result_sets", result_sets_tuple)
        object.__setattr__(self, "claims", claims_tuple)

        for name, values, expected_type, key in (
            ("datasets", datasets_tuple, DatasetLineage, lambda item: item.dataset_id),
            ("models", models_tuple, ModelLineage, lambda item: item.model_id),
            ("observations", observations_tuple, ObservationSetAuthority, lambda item: item.authority_id),
            ("preprocessing", preprocessing_tuple, PreprocessingFitAuthority, lambda item: item.transform_id),
            ("metrics", metrics_tuple, MetricSpec, lambda item: item.metric_id),
            ("claims", claims_tuple, EvidenceClaim, lambda item: item.claim_id),
        ):
            if any(not isinstance(item, expected_type) for item in values):
                raise TypeError(f"{name} contains an invalid authority type")
            keys = [key(item) for item in values]
            if len(set(keys)) != len(keys):
                raise ValueError(f"{name} cannot contain duplicate identities")
        if any(not isinstance(item, PretrainingOverlapAudit) for item in audits_tuple):
            raise TypeError("overlap_audits contains an invalid authority type")
        if any(not isinstance(item, FailurePreservingResultSet) for item in result_sets_tuple):
            raise TypeError("result_sets contains an invalid authority type")
        if not datasets_tuple:
            raise ValueError("a scientific study requires at least one dataset lineage")
        if not metrics_tuple:
            raise ValueError("a scientific study requires at least one metric spec")
        if sum(1 for metric in metrics_tuple if metric.primary) != 1:
            raise ValueError("a promoted scientific study requires exactly one primary metric")
        if not isinstance(self.repeated_measures, RepeatedMeasuresAuthority):
            raise TypeError("repeated_measures must be RepeatedMeasuresAuthority")

        datasets = {item.dataset_id: item for item in datasets_tuple}
        dataset_shas = {item.lineage_sha256 for item in datasets_tuple}
        models = {item.model_id: item for item in models_tuple}
        observation_shas = {item.authority_sha256 for item in observations_tuple}

        for observation in observations_tuple:
            if observation.dataset_lineage_sha256 not in dataset_shas:
                raise ValueError(
                    f"observation {observation.authority_id!r} references a dataset lineage "
                    "that is not part of the study"
                )
        for transform in preprocessing_tuple:
            if transform.consumption is None:
                continue
            unknown = set(transform.consumption.observation_authority_sha256s) - observation_shas
            if unknown:
                raise ValueError(
                    f"preprocessing transform {transform.transform_id!r} consumes observation "
                    "authority outside the declared study universe"
                )

        audits = {(item.model_id, item.evaluation_dataset_id): item for item in audits_tuple}
        if len(audits) != len(audits_tuple):
            raise ValueError("overlap_audits cannot repeat a model/evaluation-dataset pair")
        for audit in audits_tuple:
            model = models.get(audit.model_id)
            dataset = datasets.get(audit.evaluation_dataset_id)
            if model is None or dataset is None:
                raise ValueError("overlap audit references a model or dataset outside the study")
            if audit.model_lineage_sha256 != model.lineage_sha256:
                raise ValueError("overlap audit model lineage SHA-256 is stale or forged")
            if audit.evaluation_dataset_lineage_sha256 != dataset.lineage_sha256:
                raise ValueError("overlap audit dataset lineage SHA-256 is stale or forged")

        for result in result_sets_tuple:
            result.require_metric_specs(metrics_tuple)
        result_shas = {item.result_sha256 for item in result_sets_tuple}

        if not isinstance(self.target_budgets, Mapping):
            raise TypeError("target_budgets must be a mapping")
        budgets: dict[str, TargetObservationBudget] = {}
        for key, budget in self.target_budgets.items():
            normalized = nonempty("target budget id", str(key))
            if normalized in budgets:
                raise ValueError("target_budgets cannot contain duplicate normalized IDs")
            if not isinstance(budget, TargetObservationBudget):
                raise TypeError("target_budgets values must be TargetObservationBudget objects")
            budgets[normalized] = budget
        frozen_budgets = MappingProxyType(budgets)
        object.__setattr__(self, "target_budgets", frozen_budgets)

        for claim in claims_tuple:
            if claim.domain is EvidenceDomain.TASK_UTILITY:
                if not result_shas.intersection(claim.evidence_sha256s):
                    raise ValueError(
                        "task-utility claims must cite at least one embedded failure-preserving result-set SHA-256"
                    )
            if claim.zero_shot_claim:
                if claim.target_budget_id is None:
                    raise ValueError("zero-shot claims require an explicit target_budget_id")
                budget = budgets.get(claim.target_budget_id)
                if budget is None:
                    raise ValueError("zero-shot claim references an unknown target budget")
                if (
                    budget.labeled_examples != 0
                    or budget.unlabeled_examples != 0
                    or (budget.unlabeled_seconds or 0.0) != 0.0
                ):
                    raise ValueError(
                        "zero-shot claim is invalid because target information budget is nonzero"
                    )

            if claim.model_id is None or claim.evaluation_dataset_id is None:
                if claim.qualification is ClaimQualification.CLEAN:
                    raise ValueError(
                        "clean claims require model_id and evaluation_dataset_id so independence can be audited"
                    )
                continue
            audit = audits.get((claim.model_id, claim.evaluation_dataset_id))
            if audit is None:
                if claim.qualification is ClaimQualification.CLEAN:
                    raise ValueError("clean model/evaluation claims require an explicit overlap audit")
                continue
            if audit.status is OverlapStatus.OVERLAP_DETECTED:
                if claim.qualification is not ClaimQualification.CONTAMINATED_PRETRAINING_OVERLAP:
                    raise ValueError(
                        "overlap-detected claims must be labeled contaminated_pretraining_overlap"
                    )
            elif audit.status in {OverlapStatus.UNKNOWN_LINEAGE, OverlapStatus.POSSIBLE_OVERLAP}:
                if claim.qualification is not ClaimQualification.UNKNOWN_PRETRAINING_LINEAGE:
                    raise ValueError(
                        "unknown/possible-overlap claims must be labeled unknown_pretraining_lineage"
                    )
            elif audit.status is OverlapStatus.DISJOINT_VERIFIED:
                if claim.qualification is ClaimQualification.CLEAN:
                    pass
                elif claim.qualification not in {
                    ClaimQualification.DESCRIPTIVE_ONLY,
                    ClaimQualification.NOT_APPLICABLE,
                }:
                    raise ValueError(
                        "disjoint verified claims cannot be labeled as overlap-contaminated"
                    )

        metadata = freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def study_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.study_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "protocol_sha256": self.protocol_sha256,
            "datasets": [item.to_dict() for item in self.datasets],
            "models": [item.to_dict() for item in self.models],
            "observations": [item.to_dict() for item in self.observations],
            "preprocessing": [item.to_dict() for item in self.preprocessing],
            "metrics": [item.to_dict() for item in self.metrics],
            "repeated_measures": self.repeated_measures.to_dict(),
            "overlap_audits": [item.to_dict() for item in self.overlap_audits],
            "result_sets": [item.to_dict() for item in self.result_sets],
            "target_budgets": {
                key: budget.to_dict() for key, budget in sorted(self.target_budgets.items())
            },
            "claims": [item.to_dict() for item in self.claims],
            "metadata": thaw_json(self.metadata),
        }
        if include_identity:
            payload["study_sha256"] = self.study_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload

    def report(self) -> dict[str, Any]:
        grouped = {domain.value: [] for domain in EvidenceDomain}
        for claim in self.claims:
            grouped[claim.domain.value].append(claim.to_dict())
        return {
            "schema": "orion.scientific_authority.v2",
            "study_id": self.study_id,
            "study_sha256": self.study_sha256,
            "display_fingerprint": self.display_fingerprint,
            "protocol_sha256": self.protocol_sha256,
            "pretraining_overlap": [item.to_dict() for item in self.overlap_audits],
            "target_observation_budgets": {
                key: budget.to_dict() for key, budget in sorted(self.target_budgets.items())
            },
            "metric_specs": [item.to_dict() for item in self.metrics],
            "repeated_measures": self.repeated_measures.to_dict(),
            "result_sets": [
                {
                    "result_sha256": item.result_sha256,
                    "display_fingerprint": item.display_fingerprint,
                    "status_counts": item.status_counts(),
                    "n_cases": len(item.declared_case_ids),
                    "n_methods": len(item.method_ids),
                }
                for item in self.result_sets
            ],
            "evidence_domains": grouped,
            "claim_scope": [item.to_dict() for item in self.claims],
        }
