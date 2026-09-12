"""Boundary-preserving neural representation comparison and evidence tools."""

from .autoencoder import AutoencoderRepresentation
from .benchmark import RepresentationBenchmark
from .contracts import (
    FitRegime,
    MethodOutcome,
    MethodStatus,
    RepresentationBenchmarkResult,
    RepresentationEmbedding,
    RepresentationError,
    RepresentationMethod,
    RepresentationUnavailableError,
    SequenceBatch,
)
from .controlled import build_controlled_temporal_manifold, latent_trajectory, observations
from .evidence import (
    CaseStatus,
    EvaluationScope,
    MethodEvidenceSummary,
    RepresentationCaseEvidence,
    RepresentationEvidenceGrid,
)
from .external import PrecomputedTemporalSSLRepresentation
from .metrics import (
    aggregate_geometry_metrics,
    aggregate_reference_metrics,
    local_neighborhood_preservation,
    pairwise_distance_rank_preservation,
    temporal_continuity_ratio,
)
from .pca import PCARepresentation
from .predictive import (
    LagPredictiveAutoencoderRepresentation,
    build_lagged_training_pairs,
)
from .sequence_authority import (
    SequenceMethodOutcome,
    SequenceRepresentationBenchmarkResult,
    run_sequencewise_representation_benchmark,
)
from .sweep import (
    CaseMethodEvidence,
    MethodSweepSummary,
    NoiseMethodSummary,
    RepresentationSweepResult,
    SweepCase,
    build_representation_sweep,
)
from .tphate import (
    UPSTREAM_LICENSE_NOTICE,
    UPSTREAM_REPOSITORY,
    TPHATEEmbeddingError,
    TPHATERepresentation,
    TPHATEUnavailableError,
)

__all__ = [
    "UPSTREAM_LICENSE_NOTICE",
    "UPSTREAM_REPOSITORY",
    "AutoencoderRepresentation",
    "CaseMethodEvidence",
    "CaseStatus",
    "EvaluationScope",
    "FitRegime",
    "LagPredictiveAutoencoderRepresentation",
    "MethodEvidenceSummary",
    "MethodOutcome",
    "MethodStatus",
    "MethodSweepSummary",
    "NoiseMethodSummary",
    "PCARepresentation",
    "PrecomputedTemporalSSLRepresentation",
    "RepresentationBenchmark",
    "RepresentationBenchmarkResult",
    "RepresentationCaseEvidence",
    "RepresentationEmbedding",
    "RepresentationError",
    "RepresentationEvidenceGrid",
    "RepresentationMethod",
    "RepresentationSweepResult",
    "RepresentationUnavailableError",
    "SequenceBatch",
    "SequenceMethodOutcome",
    "SequenceRepresentationBenchmarkResult",
    "SweepCase",
    "TPHATEEmbeddingError",
    "TPHATERepresentation",
    "TPHATEUnavailableError",
    "aggregate_geometry_metrics",
    "aggregate_reference_metrics",
    "build_controlled_temporal_manifold",
    "build_lagged_training_pairs",
    "build_representation_sweep",
    "latent_trajectory",
    "local_neighborhood_preservation",
    "observations",
    "pairwise_distance_rank_preservation",
    "run_sequencewise_representation_benchmark",
    "temporal_continuity_ratio",
]
