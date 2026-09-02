"""Typed semantics and fail-closed packet materialization for model-proposed research."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from ._canonical import canonical_sha256, require_nonempty, require_sha256
from .contracts import (
    DatasetAuthority,
    EvaluationAuthority,
    ExperimentPacket,
    ExternalDispatchPolicy,
    Hypothesis,
    ResearchAgent,
)
from .nim import ResearchProposal

CriterionOperator = Literal[">", ">=", "<", "<="]
MetricDirection = Literal["higher_is_better", "lower_is_better", "neutrality"]
ClaimRelation = Literal[
    "absolute",
    "matched_control",
    "temporal_null",
    "complementarity",
    "stability",
    "control_sweep",
    "prospective_prediction",
]

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_OPERATORS = frozenset({">", ">=", "<", "<="})
_ALLOWED_CLAIM_RELATIONS = frozenset(
    {
        "absolute",
        "matched_control",
        "temporal_null",
        "complementarity",
        "stability",
        "control_sweep",
        "prospective_prediction",
    }
)
_COMPARATIVE_CLAIM_RE = re.compile(
    r"\b(?:than|versus|vs\.?|compared(?:\s+to|\s+with)?|relative\s+to|"
    r"matched[-\s]?control|baseline|improvement\s+over|gain\s+over|reduction\s+versus)\b",
    re.IGNORECASE,
)
_PROSPECTIVE_PREDICTION_RE = re.compile(
    r"\b(?:prospectiv(?:e|ely)|predicts?|forecast(?:s|ing|ed)?|later\s+validation|"
    r"subsequent\s+validation|future\s+validation|pre[-\s]?reveal\s+screen(?:s|ing)?)\b",
    re.IGNORECASE,
)

INDEPENDENT_CANDIDATE_STOPPING_RULE = (
    "Evaluate every queued candidate independently against its own typed support and rejection "
    "criteria. Rejection or failure of one candidate does not terminate unrelated candidates. "
    "Global execution stops only when frozen safety, data-authority, evaluator-authority, or "
    "compute-budget constraints prevent further authorized execution."
)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Machine-readable meaning for one development-only research metric."""

    name: str
    direction: MetricDirection
    definition: str
    unit: str = "unitless"
    claim_relation: ClaimRelation = "absolute"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_nonempty(self.name, name="metric name"))
        object.__setattr__(
            self,
            "definition",
            require_nonempty(self.definition, name="metric definition"),
        )
        object.__setattr__(self, "unit", require_nonempty(self.unit, name="metric unit"))
        if self.direction not in {"higher_is_better", "lower_is_better", "neutrality"}:
            raise ValueError(f"unsupported metric direction {self.direction!r}")
        if self.claim_relation not in _ALLOWED_CLAIM_RELATIONS:
            raise ValueError(f"unsupported metric claim relation {self.claim_relation!r}")

    @property
    def support_operators(self) -> frozenset[str]:
        if self.direction == "higher_is_better":
            return frozenset({">", ">="})
        return frozenset({"<", "<="})

    @property
    def rejection_operators(self) -> frozenset[str]:
        if self.direction == "higher_is_better":
            return frozenset({"<", "<="})
        return frozenset({">", ">="})

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "direction": self.direction,
            "definition": self.definition,
            "unit": self.unit,
            "claim_relation": self.claim_relation,
        }


ALGORITHMIC_METRIC_REGISTRY: dict[str, MetricSpec] = {
    "validation_pearson": MetricSpec(
        "validation_pearson",
        "higher_is_better",
        "Mean development-validation Pearson correlation under the frozen split.",
    ),
    "validation_mse": MetricSpec(
        "validation_mse",
        "lower_is_better",
        "Mean development-validation mean squared prediction error under the frozen split.",
    ),
    "validation_pearson_delta": MetricSpec(
        "validation_pearson_delta",
        "higher_is_better",
        "Candidate minus matched-control development-validation Pearson correlation.",
        claim_relation="matched_control",
    ),
    "validation_mse_reduction": MetricSpec(
        "validation_mse_reduction",
        "higher_is_better",
        "Matched-control MSE minus candidate MSE on the same development-validation examples.",
        claim_relation="matched_control",
    ),
    "rsa_spearman": MetricSpec(
        "rsa_spearman",
        "higher_is_better",
        "Development-only Spearman RSA alignment between frozen representation and neural geometry.",
    ),
    "prospective_geometry_gain_spearman": MetricSpec(
        "prospective_geometry_gain_spearman",
        "higher_is_better",
        "Spearman correlation across a predeclared candidate set between a geometry score frozen before outcome reveal and the subsequently revealed matched-control validation Pearson delta.",
        claim_relation="prospective_prediction",
    ),
    "temporal_shift_drop": MetricSpec(
        "temporal_shift_drop",
        "higher_is_better",
        "Validation score at zero shift minus the best matched temporal-shift null score.",
        claim_relation="temporal_null",
    ),
    "validation_stability": MetricSpec(
        "validation_stability",
        "higher_is_better",
        "Predeclared stability score across development folds, seeds, or segments.",
        claim_relation="stability",
    ),
    "runtime_seconds": MetricSpec(
        "runtime_seconds",
        "lower_is_better",
        "Wall-clock runtime for the frozen development workload.",
        "seconds",
    ),
    "cache_gb": MetricSpec(
        "cache_gb",
        "lower_is_better",
        "Peak or materialized cache footprint for the frozen development workload.",
        "GB",
    ),
    "complementarity_score": MetricSpec(
        "complementarity_score",
        "higher_is_better",
        "One minus matched validation residual-error correlation; larger means more complementary errors.",
        claim_relation="complementarity",
    ),
    "validation_pearson_span": MetricSpec(
        "validation_pearson_span",
        "neutrality",
        "Maximum minus minimum validation Pearson correlation across a predeclared control sweep.",
        claim_relation="control_sweep",
    ),
    "matched_geometry_rsa_delta": MetricSpec(
        "matched_geometry_rsa_delta",
        "higher_is_better",
        "RSA Spearman delta versus a capacity- and dimensionality-matched geometry control.",
        claim_relation="matched_control",
    ),
}


def metric_registry_payload() -> dict[str, dict[str, str]]:
    """Return a stable JSON-friendly metric registry for model prompts and artifacts."""

    return {
        name: ALGORITHMIC_METRIC_REGISTRY[name].to_dict()
        for name in sorted(ALGORITHMIC_METRIC_REGISTRY)
    }


def enforce_independent_synthesis_stopping_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Demote model-authored stopping prose to an advisory note and install fixed authority."""

    if not isinstance(payload, dict):
        raise TypeError("synthesis payload must be a dictionary")
    model_note = require_nonempty(str(payload.get("stopping_rule", "")), name="model stopping note")
    normalized = dict(payload)
    normalized["model_stopping_note"] = model_note
    normalized["stopping_rule"] = INDEPENDENT_CANDIDATE_STOPPING_RULE
    normalized["stopping_rule_authority"] = "deterministic_independent_candidate_policy"
    return normalized


@dataclass(frozen=True, slots=True)
class DecisionCriterion:
    """One directional, machine-checkable development-only decision predicate."""

    metric: str
    operator: CriterionOperator
    threshold: float
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", require_nonempty(self.metric, name="criterion metric"))
        if self.operator not in _ALLOWED_OPERATORS:
            raise ValueError(f"unsupported criterion operator {self.operator!r}")
        threshold = float(self.threshold)
        if not math.isfinite(threshold):
            raise ValueError("criterion threshold must be finite")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(
            self,
            "rationale",
            require_nonempty(self.rationale, name="criterion rationale"),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DecisionCriterion:
        return cls(
            metric=str(payload["metric"]),
            operator=str(payload["operator"]),  # type: ignore[arg-type]
            threshold=float(payload["threshold"]),
            rationale=str(payload["rationale"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "rationale": self.rationale,
        }


def _validate_criterion_direction(
    criterion: DecisionCriterion,
    *,
    purpose: Literal["support", "reject"],
) -> None:
    spec = ALGORITHMIC_METRIC_REGISTRY.get(criterion.metric)
    if spec is None:
        raise ValueError(f"unknown semantic metric {criterion.metric!r}")
    allowed = spec.support_operators if purpose == "support" else spec.rejection_operators
    if criterion.operator not in allowed:
        raise ValueError(
            f"{purpose} criterion for {criterion.metric!r} contradicts "
            f"registered direction {spec.direction!r}"
        )


def _validate_nonoverlap(
    support: tuple[DecisionCriterion, ...],
    rejection: tuple[DecisionCriterion, ...],
) -> None:
    for supporting in support:
        spec = ALGORITHMIC_METRIC_REGISTRY[supporting.metric]
        for rejecting in rejection:
            if supporting.metric != rejecting.metric:
                continue
            if spec.direction == "higher_is_better":
                if supporting.threshold <= rejecting.threshold:
                    raise ValueError(
                        f"support/rejection regions overlap for metric {supporting.metric!r}"
                    )
            elif supporting.threshold >= rejecting.threshold:
                raise ValueError(
                    f"support/rejection regions overlap for metric {supporting.metric!r}"
                )


@dataclass(frozen=True, slots=True)
class SemanticResearchProposal:
    """A model proposal whose metric semantics are explicit enough to review deterministically."""

    proposal: ResearchProposal
    primary_metric: str
    claim_relation: ClaimRelation
    control_description: str
    supports_if: tuple[DecisionCriterion, ...]
    rejects_if: tuple[DecisionCriterion, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "primary_metric",
            require_nonempty(self.primary_metric, name="primary_metric"),
        )
        if self.primary_metric not in ALGORITHMIC_METRIC_REGISTRY:
            raise ValueError(f"unknown primary metric {self.primary_metric!r}")
        if self.primary_metric not in self.proposal.development_metrics:
            raise ValueError("primary_metric must appear in development_metrics")
        if not self.supports_if or not self.rejects_if:
            raise ValueError("supports_if and rejects_if must both contain criteria")

        relation = str(self.claim_relation).strip()
        if relation not in _ALLOWED_CLAIM_RELATIONS:
            raise ValueError(f"unsupported proposal claim relation {relation!r}")
        object.__setattr__(self, "claim_relation", relation)
        control = str(self.control_description).strip()
        primary_spec = ALGORITHMIC_METRIC_REGISTRY[self.primary_metric]
        if relation != primary_spec.claim_relation:
            raise ValueError(
                f"claim_relation {relation!r} contradicts primary metric "
                f"{self.primary_metric!r} relation {primary_spec.claim_relation!r}"
            )
        if relation == "absolute":
            if control:
                raise ValueError("absolute claims cannot carry a comparison control")
        elif not control:
            raise ValueError(f"{relation} claims require a non-empty control_description")
        object.__setattr__(self, "control_description", control)

        explicit_claim_text = f"{self.proposal.statement} {self.proposal.falsification_test}"
        if relation == "absolute" and _COMPARATIVE_CLAIM_RE.search(explicit_claim_text):
            raise ValueError(
                "explicit comparative claim requires a comparative primary metric, "
                "not an absolute metric"
            )
        if relation != "prospective_prediction" and _PROSPECTIVE_PREDICTION_RE.search(
            explicit_claim_text
        ):
            raise ValueError(
                "prospective prediction claim requires the prospective_prediction relation "
                "and prospective_geometry_gain_spearman primary metric"
            )

        known_metrics = set(self.proposal.development_metrics)
        if not known_metrics.issubset(ALGORITHMIC_METRIC_REGISTRY):
            unknown = sorted(known_metrics - set(ALGORITHMIC_METRIC_REGISTRY))
            raise ValueError(f"proposal contains metrics outside semantic registry: {unknown}")

        for criterion in self.supports_if:
            if criterion.metric not in known_metrics:
                raise ValueError("support criterion metric must appear in development_metrics")
            _validate_criterion_direction(criterion, purpose="support")
        for criterion in self.rejects_if:
            if criterion.metric not in known_metrics:
                raise ValueError("rejection criterion metric must appear in development_metrics")
            _validate_criterion_direction(criterion, purpose="reject")
        if self.primary_metric not in {criterion.metric for criterion in self.supports_if}:
            raise ValueError("primary_metric must have a support criterion")
        if self.primary_metric not in {criterion.metric for criterion in self.rejects_if}:
            raise ValueError("primary_metric must have a rejection criterion")

        support_keys = {
            (criterion.metric, criterion.operator, criterion.threshold)
            for criterion in self.supports_if
        }
        reject_keys = {
            (criterion.metric, criterion.operator, criterion.threshold)
            for criterion in self.rejects_if
        }
        if support_keys & reject_keys:
            raise ValueError("support and rejection criteria cannot contain the same predicate")
        _validate_nonoverlap(self.supports_if, self.rejects_if)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        allowed_payload_classes: tuple[str, ...],
        allowed_development_metrics: tuple[str, ...],
    ) -> SemanticResearchProposal:
        proposal = ResearchProposal.from_dict(
            payload,
            allowed_payload_classes=allowed_payload_classes,
            allowed_development_metrics=allowed_development_metrics,
        )
        supports = payload.get("supports_if")
        rejects = payload.get("rejects_if")
        if not isinstance(supports, list) or not isinstance(rejects, list):
            raise ValueError("semantic proposal requires supports_if and rejects_if lists")
        if not all(isinstance(row, dict) for row in (*supports, *rejects)):
            raise ValueError("all semantic criteria must be JSON objects")
        return cls(
            proposal=proposal,
            primary_metric=str(payload["primary_metric"]),
            claim_relation=str(payload["claim_relation"]),  # type: ignore[arg-type]
            control_description=str(payload.get("control_description", "")),
            supports_if=tuple(DecisionCriterion.from_dict(row) for row in supports),
            rejects_if=tuple(DecisionCriterion.from_dict(row) for row in rejects),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.proposal.to_dict(),
            "primary_metric": self.primary_metric,
            "claim_relation": self.claim_relation,
            "control_description": self.control_description,
            "supports_if": [criterion.to_dict() for criterion in self.supports_if],
            "rejects_if": [criterion.to_dict() for criterion in self.rejects_if],
        }

    @property
    def candidate_id(self) -> str:
        return self.proposal.candidate_id

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


def parse_semantic_proposals(
    payload: dict[str, Any],
    *,
    allowed_payload_classes: tuple[str, ...],
    allowed_development_metrics: tuple[str, ...],
    min_candidates: int = 3,
    max_candidates: int = 8,
) -> tuple[SemanticResearchProposal, ...]:
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("proposal response must contain a candidates list")
    if len(rows) < min_candidates or len(rows) > max_candidates:
        raise ValueError(
            f"proposal response must contain {min_candidates}..{max_candidates} candidates"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every candidate must be a JSON object")
    proposals = tuple(
        SemanticResearchProposal.from_dict(
            row,
            allowed_payload_classes=allowed_payload_classes,
            allowed_development_metrics=allowed_development_metrics,
        )
        for row in rows
    )
    ids = [proposal.candidate_id for proposal in proposals]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique")
    return proposals


@dataclass(frozen=True, slots=True)
class ExecutionBinding:
    """Real-world identities required before an untrusted proposal can become executable."""

    dataset_id: str
    dataset_source_fingerprint: str
    dataset_source_revision: str
    evaluator_id: str
    split_fingerprint: str
    preprocessing_fingerprint: str
    evaluation_domains: tuple[str, ...]
    runner_entrypoint: str
    code_revision: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "dataset_source_revision",
            "evaluator_id",
            "runner_entrypoint",
        ):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), name=name))
        for name in (
            "dataset_source_fingerprint",
            "split_fingerprint",
            "preprocessing_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        code_revision = require_nonempty(self.code_revision, name="code_revision")
        if not _GIT_SHA_RE.fullmatch(code_revision):
            raise ValueError("code_revision must be an exact 40-character lowercase git SHA")
        object.__setattr__(self, "code_revision", code_revision)
        domains = tuple(
            require_nonempty(value, name="evaluation_domains")
            for value in self.evaluation_domains
        )
        if not domains or len(set(domains)) != len(domains):
            raise ValueError("evaluation_domains must be non-empty and unique")
        object.__setattr__(self, "evaluation_domains", domains)
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be non-empty and unique")
        object.__setattr__(self, "seeds", seeds)


def materialize_g1_packet(
    proposal: SemanticResearchProposal,
    binding: ExecutionBinding,
    *,
    proposer_model: str,
    proposer_prompt_sha256: str,
) -> ExperimentPacket:
    """Mint a G1 development packet only after real execution identities are supplied."""

    proposer_model = require_nonempty(proposer_model, name="proposer_model")
    proposer_prompt_sha256 = require_sha256(
        proposer_prompt_sha256,
        name="proposer_prompt_sha256",
    )
    dataset = DatasetAuthority(
        dataset_id=binding.dataset_id,
        source_fingerprint=binding.dataset_source_fingerprint,
        access="authorized_restricted",
        source_revision=binding.dataset_source_revision,
        metadata={"preprocessing_fingerprint": binding.preprocessing_fingerprint},
    )
    evaluation = EvaluationAuthority(
        evaluator_id=binding.evaluator_id,
        split_fingerprint=binding.split_fingerprint,
        metric_names=proposal.proposal.development_metrics,
        evaluation_domains=binding.evaluation_domains,
        optimization_boundary="train_validation",
    )
    agent = ResearchAgent(
        agent_id=f"nvidia-nim:{proposal.candidate_id}",
        kind="frontier_model",
        provider="nvidia_nim",
        model=proposer_model,
        prompt_sha256=proposer_prompt_sha256,
        role="proposal_only",
        metadata={"proposal_fingerprint": proposal.fingerprint},
    )
    hypothesis = Hypothesis(
        hypothesis_id=proposal.candidate_id,
        statement=proposal.proposal.statement,
        changed_variables=proposal.proposal.changed_variables,
        rationale=proposal.proposal.rationale,
    )
    return ExperimentPacket(
        experiment_id=f"g1-{proposal.candidate_id}-{proposal.fingerprint[:12]}",
        dataset=dataset,
        evaluation=evaluation,
        agent=agent,
        hypothesis=hypothesis,
        code_revision=binding.code_revision,
        seeds=binding.seeds,
        information_regimes=("train_only_inductive",),
        claim_ceiling="predictive_id",
        compute_budget={"tier": proposal.proposal.estimated_compute_tier},
        dispatch_policy=ExternalDispatchPolicy(),
        metadata={
            "gate": "G1",
            "runner_entrypoint": binding.runner_entrypoint,
            "preprocessing_fingerprint": binding.preprocessing_fingerprint,
            "primary_metric": proposal.primary_metric,
            "claim_relation": proposal.claim_relation,
            "control_description": proposal.control_description,
            "supports_if": [criterion.to_dict() for criterion in proposal.supports_if],
            "rejects_if": [criterion.to_dict() for criterion in proposal.rejects_if],
            "scientific_boundary": (
                "This packet authorizes deterministic development-only execution. "
                "It does not authorize G2/G3/G4 truth access or scientific promotion."
            ),
        },
    )
