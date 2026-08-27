"""Compatibility facade for ORION Scientific Authority v2.

New code may import from :mod:`orion.scientific`. This module remains the stable
flat import surface for callers introduced before the authority subsystem was
split into focused lineage, observation, evaluation, study, and longitudinal
modules.
"""

from .scientific import (
    CaseOutcome,
    CaseStatus,
    ClaimQualification,
    DatasetLineage,
    EvidenceClaim,
    EvidenceDomain,
    FailureAggregationPolicy,
    FailurePreservingResultSet,
    IdentityAvailability,
    IdentitySet,
    LineageCompleteness,
    MetricDirection,
    MetricSpec,
    ModelLineage,
    ObservationConsumption,
    ObservationRole,
    ObservationSetAuthority,
    OperationKind,
    OverlapStatus,
    PreprocessingFitAuthority,
    PretrainingOverlapAudit,
    ProbabilityRequirement,
    RepeatedMeasuresAuthority,
    ScientificStudyAuthority,
    TargetObservationBudget,
    TransformFitKind,
    audit_pretraining_overlap,
    bind_longitudinal_case_authority,
)

__all__ = [
    "CaseOutcome",
    "CaseStatus",
    "ClaimQualification",
    "DatasetLineage",
    "EvidenceClaim",
    "EvidenceDomain",
    "FailureAggregationPolicy",
    "FailurePreservingResultSet",
    "IdentityAvailability",
    "IdentitySet",
    "LineageCompleteness",
    "MetricDirection",
    "MetricSpec",
    "ModelLineage",
    "ObservationConsumption",
    "ObservationRole",
    "ObservationSetAuthority",
    "OperationKind",
    "OverlapStatus",
    "PreprocessingFitAuthority",
    "PretrainingOverlapAudit",
    "ProbabilityRequirement",
    "RepeatedMeasuresAuthority",
    "ScientificStudyAuthority",
    "TargetObservationBudget",
    "TransformFitKind",
    "audit_pretraining_overlap",
    "bind_longitudinal_case_authority",
]
