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
from .registry import ResearchRegistry

__all__ = [
    "AdversarialCheck",
    "AlgonautsAuthoritySpec",
    "DatasetAuthority",
    "EvaluationAuthority",
    "EvidenceArbiter",
    "EvidenceLedger",
    "ExperimentEvidence",
    "ExperimentPacket",
    "ExternalDispatchPolicy",
    "Hypothesis",
    "InsightCard",
    "LedgerEvent",
    "MetricGate",
    "MetricObservation",
    "PromotionDecision",
    "PromotionPolicy",
    "ResearchAgent",
    "ResearchRegistry",
]
