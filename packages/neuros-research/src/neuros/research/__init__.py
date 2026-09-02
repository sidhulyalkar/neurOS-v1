"""Provider-agnostic autonomous research authority for neurOS."""

from .algonauts import AlgonautsAuthoritySpec
from .arbiter import EvidenceArbiter, MetricGate, PromotionDecision, PromotionPolicy
from .contracts import (
    DatasetAuthority,
    EvaluationAuthority,
    ExperimentPacket,
    ExternalDispatchPolicy,
    Hypothesis,
    ResearchAgent,
)
from .evidence import AdversarialCheck, ExperimentEvidence, MetricObservation
from .insight import InsightCard
from .ledger import EvidenceLedger, LedgerEvent
from .prospective import (
    ProspectiveGeometryCandidate,
    ProspectiveGeometryPlan,
    ProspectiveGeometryReveal,
    ProspectiveOutcome,
    evaluate_prospective_geometry_gain,
)
from .registry import ResearchRegistry
from .semantics import (
    ALGORITHMIC_METRIC_REGISTRY,
    DecisionCriterion,
    ExecutionBinding,
    MetricSpec,
    SemanticResearchProposal,
    materialize_g1_packet,
    metric_registry_payload,
    parse_semantic_proposals,
)

__all__ = [
    "ALGORITHMIC_METRIC_REGISTRY",
    "AdversarialCheck",
    "AlgonautsAuthoritySpec",
    "DatasetAuthority",
    "DecisionCriterion",
    "EvaluationAuthority",
    "EvidenceArbiter",
    "EvidenceLedger",
    "ExecutionBinding",
    "ExperimentEvidence",
    "ExperimentPacket",
    "ExternalDispatchPolicy",
    "Hypothesis",
    "InsightCard",
    "LedgerEvent",
    "MetricGate",
    "MetricObservation",
    "MetricSpec",
    "PromotionDecision",
    "PromotionPolicy",
    "ProspectiveGeometryCandidate",
    "ProspectiveGeometryPlan",
    "ProspectiveGeometryReveal",
    "ProspectiveOutcome",
    "ResearchAgent",
    "ResearchRegistry",
    "SemanticResearchProposal",
    "evaluate_prospective_geometry_gain",
    "materialize_g1_packet",
    "metric_registry_payload",
    "parse_semantic_proposals",
]
