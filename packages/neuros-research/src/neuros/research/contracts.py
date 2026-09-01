"""Immutable authority contracts for evidence-bound research search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ._canonical import canonical_sha256, freeze_json, require_nonempty, require_sha256, thaw_json

DataAccess = Literal["public", "authorized_restricted", "synthetic"]
OptimizationBoundary = Literal["none", "train_only", "train_validation"]
InformationRegime = Literal[
    "train_only_inductive",
    "external_pretrained",
    "transductive_unlabeled",
    "evaluation_only",
    "simulation_only",
]
AgentKind = Literal["human", "frontier_model", "local_model", "deterministic_program"]
ClaimCeiling = Literal[
    "software_only",
    "descriptive",
    "predictive_id",
    "predictive_ood",
    "cross_domain_transfer",
    "causal_hypothesis",
]


def _unique_strings(values: tuple[str, ...], *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    normalized = tuple(require_nonempty(value, name=name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be unique")
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must contain at least one value")
    return normalized


@dataclass(frozen=True, slots=True)
class DatasetAuthority:
    """Identity and access boundary for one research dataset view."""

    dataset_id: str
    source_fingerprint: str
    access: DataAccess
    source_revision: str = "unspecified"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", require_nonempty(self.dataset_id, name="dataset_id"))
        object.__setattr__(
            self,
            "source_fingerprint",
            require_sha256(self.source_fingerprint, name="source_fingerprint"),
        )
        object.__setattr__(
            self,
            "source_revision",
            require_nonempty(self.source_revision, name="source_revision"),
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata, path="dataset.metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_fingerprint": self.source_fingerprint,
            "access": self.access,
            "source_revision": self.source_revision,
            "metadata": thaw_json(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class EvaluationAuthority:
    """The referee an experiment is not allowed to rewrite."""

    evaluator_id: str
    split_fingerprint: str
    metric_names: tuple[str, ...]
    evaluation_domains: tuple[str, ...]
    optimization_boundary: OptimizationBoundary = "train_validation"
    forbidden_feedback: tuple[str, ...] = (
        "hidden_test_targets",
        "private_leaderboard",
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluator_id", require_nonempty(self.evaluator_id, name="evaluator_id"))
        object.__setattr__(
            self,
            "split_fingerprint",
            require_sha256(self.split_fingerprint, name="split_fingerprint"),
        )
        object.__setattr__(
            self,
            "metric_names",
            _unique_strings(self.metric_names, name="metric_names"),
        )
        object.__setattr__(
            self,
            "evaluation_domains",
            _unique_strings(self.evaluation_domains, name="evaluation_domains"),
        )
        object.__setattr__(
            self,
            "forbidden_feedback",
            _unique_strings(self.forbidden_feedback, name="forbidden_feedback", allow_empty=True),
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata, path="evaluation.metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "split_fingerprint": self.split_fingerprint,
            "metric_names": list(self.metric_names),
            "evaluation_domains": list(self.evaluation_domains),
            "optimization_boundary": self.optimization_boundary,
            "forbidden_feedback": list(self.forbidden_feedback),
            "metadata": thaw_json(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ResearchAgent:
    """Identity of a proposer or implementer, never evaluation authority."""

    agent_id: str
    kind: AgentKind
    provider: str
    model: str
    version: str = "unspecified"
    prompt_sha256: str | None = None
    role: str = "researcher"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("agent_id", "provider", "model", "version", "role"):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), name=name))
        if self.prompt_sha256 is not None:
            object.__setattr__(
                self,
                "prompt_sha256",
                require_sha256(self.prompt_sha256, name="prompt_sha256"),
            )
        object.__setattr__(self, "metadata", freeze_json(self.metadata, path="agent.metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "kind": self.kind,
            "provider": self.provider,
            "model": self.model,
            "version": self.version,
            "prompt_sha256": self.prompt_sha256,
            "role": self.role,
            "metadata": thaw_json(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """Falsifiable experiment intent and explicit lineage."""

    hypothesis_id: str
    statement: str
    changed_variables: tuple[str, ...]
    parent_experiment_ids: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hypothesis_id", require_nonempty(self.hypothesis_id, name="hypothesis_id")
        )
        object.__setattr__(self, "statement", require_nonempty(self.statement, name="statement"))
        object.__setattr__(
            self,
            "changed_variables",
            _unique_strings(self.changed_variables, name="changed_variables"),
        )
        object.__setattr__(
            self,
            "parent_experiment_ids",
            _unique_strings(
                self.parent_experiment_ids,
                name="parent_experiment_ids",
                allow_empty=True,
            ),
        )
        object.__setattr__(self, "rationale", str(self.rationale).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "changed_variables": list(self.changed_variables),
            "parent_experiment_ids": list(self.parent_experiment_ids),
            "rationale": self.rationale,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ExternalDispatchPolicy:
    """Classes of material that may or may not leave the trusted execution boundary."""

    allowed_payload_classes: tuple[str, ...] = (
        "source_code",
        "schemas",
        "aggregate_metrics",
        "deidentified_plots",
        "public_metadata",
    )
    prohibited_payload_classes: tuple[str, ...] = (
        "raw_participant_data",
        "participant_identifiers",
        "hidden_test_targets",
        "credentials",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_payload_classes",
            _unique_strings(
                self.allowed_payload_classes,
                name="allowed_payload_classes",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "prohibited_payload_classes",
            _unique_strings(
                self.prohibited_payload_classes,
                name="prohibited_payload_classes",
                allow_empty=True,
            ),
        )
        overlap = set(self.allowed_payload_classes) & set(self.prohibited_payload_classes)
        if overlap:
            raise ValueError(f"dispatch payload classes cannot be both allowed and prohibited: {sorted(overlap)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_payload_classes": list(self.allowed_payload_classes),
            "prohibited_payload_classes": list(self.prohibited_payload_classes),
        }


@dataclass(frozen=True, slots=True)
class ExperimentPacket:
    """Complete immutable contract handed to an experiment executor."""

    experiment_id: str
    dataset: DatasetAuthority
    evaluation: EvaluationAuthority
    agent: ResearchAgent
    hypothesis: Hypothesis
    code_revision: str
    seeds: tuple[int, ...]
    information_regimes: tuple[InformationRegime, ...]
    claim_ceiling: ClaimCeiling
    compute_budget: Mapping[str, Any] = field(default_factory=dict)
    representation_fingerprint: str | None = None
    dispatch_policy: ExternalDispatchPolicy = field(default_factory=ExternalDispatchPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_id", require_nonempty(self.experiment_id, name="experiment_id")
        )
        object.__setattr__(
            self,
            "code_revision",
            require_nonempty(self.code_revision, name="code_revision"),
        )
        if not self.seeds:
            raise ValueError("seeds must contain at least one deterministic seed")
        normalized_seeds = tuple(int(seed) for seed in self.seeds)
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("seeds must be unique")
        object.__setattr__(self, "seeds", normalized_seeds)
        regimes = tuple(self.information_regimes)
        if not regimes or len(set(regimes)) != len(regimes):
            raise ValueError("information_regimes must be non-empty and unique")
        object.__setattr__(self, "information_regimes", regimes)
        if self.representation_fingerprint is not None:
            object.__setattr__(
                self,
                "representation_fingerprint",
                require_sha256(
                    self.representation_fingerprint,
                    name="representation_fingerprint",
                ),
            )
        object.__setattr__(
            self,
            "compute_budget",
            freeze_json(self.compute_budget, path="experiment.compute_budget"),
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata, path="experiment.metadata"))
        object.__setattr__(
            self, "schema_version", require_nonempty(self.schema_version, name="schema_version")
        )
        if self.experiment_id in self.hypothesis.parent_experiment_ids:
            raise ValueError("an experiment cannot declare itself as a parent")

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "dataset": self.dataset.to_dict(),
            "dataset_fingerprint": self.dataset.fingerprint,
            "evaluation": self.evaluation.to_dict(),
            "evaluation_fingerprint": self.evaluation.fingerprint,
            "agent": self.agent.to_dict(),
            "agent_fingerprint": self.agent.fingerprint,
            "hypothesis": self.hypothesis.to_dict(),
            "hypothesis_fingerprint": self.hypothesis.fingerprint,
            "code_revision": self.code_revision,
            "seeds": list(self.seeds),
            "information_regimes": list(self.information_regimes),
            "claim_ceiling": self.claim_ceiling,
            "compute_budget": thaw_json(self.compute_budget),
            "representation_fingerprint": self.representation_fingerprint,
            "dispatch_policy": self.dispatch_policy.to_dict(),
            "metadata": thaw_json(self.metadata),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict(include_fingerprint=False))
