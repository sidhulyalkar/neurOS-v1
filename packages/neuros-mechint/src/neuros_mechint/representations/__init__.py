"""Boundary-preserving neural representation comparison tools."""

from .autoencoder import AutoencoderRepresentation
from .benchmark import RepresentationBenchmark
from .cases import (
    CasePreservingRepresentationBenchmark,
    CasePreservingRepresentationResult,
    CaseStatus,
    MethodCaseSummary,
    RepresentationCaseOutcome,
    RepresentationNonconvergenceError,
)
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
from .external import PrecomputedTemporalSSLRepresentation
from .metrics import (
    aggregate_geometry_metrics,
    aggregate_reference_metrics,
    local_neighborhood_preservation,
    pairwise_distance_rank_preservation,
    temporal_continuity_ratio,
)
from .pca import PCARepresentation
from .tphate import (
    TPHATEEmbeddingError,
    TPHATERepresentation,
    TPHATEUnavailableError,
    UPSTREAM_LICENSE_NOTICE,
    UPSTREAM_REPOSITORY,
)

__all__ = [
    "AutoencoderRepresentation",
    "CasePreservingRepresentationBenchmark",
    "CasePreservingRepresentationResult",
    "CaseStatus",
    "FitRegime",
    "MethodCaseSummary",
    "MethodOutcome",
    "MethodStatus",
    "PCARepresentation",
    "PrecomputedTemporalSSLRepresentation",
    "RepresentationBenchmark",
    "RepresentationBenchmarkResult",
    "RepresentationCaseOutcome",
    "RepresentationEmbedding",
    "RepresentationError",
    "RepresentationMethod",
    "RepresentationNonconvergenceError",
    "RepresentationUnavailableError",
    "SequenceBatch",
    "TPHATEEmbeddingError",
    "TPHATERepresentation",
    "TPHATEUnavailableError",
    "UPSTREAM_LICENSE_NOTICE",
    "UPSTREAM_REPOSITORY",
    "aggregate_geometry_metrics",
    "aggregate_reference_metrics",
    "local_neighborhood_preservation",
    "pairwise_distance_rank_preservation",
    "temporal_continuity_ratio",
]
