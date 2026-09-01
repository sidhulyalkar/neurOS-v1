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
    EvaluationScope,
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
from .corruptions import (
    TemporalCorruption,
    make_controlled_corruption_manifold,
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
from .sweep import (
    ControlledNoiseSweepResult,
    NoiseLevelSummary,
    SweepCaseRecord,
    run_controlled_noise_sweep,
)
from .synthetic import (
    ControlledTemporalManifold,
    latent_trajectory,
    make_controlled_temporal_manifold,
)
from .temporal_ablation import TemporalOrderInterventionRepresentation
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
    "ControlledNoiseSweepResult",
    "ControlledTemporalManifold",
    "EvaluationScope",
    "FitRegime",
    "MethodCaseSummary",
    "MethodOutcome",
    "MethodStatus",
    "NoiseLevelSummary",
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
    "SweepCaseRecord",
    "TPHATEEmbeddingError",
    "TPHATERepresentation",
    "TPHATEUnavailableError",
    "TemporalCorruption",
    "TemporalOrderInterventionRepresentation",
    "UPSTREAM_LICENSE_NOTICE",
    "UPSTREAM_REPOSITORY",
    "aggregate_geometry_metrics",
    "aggregate_reference_metrics",
    "latent_trajectory",
    "local_neighborhood_preservation",
    "make_controlled_corruption_manifold",
    "make_controlled_temporal_manifold",
    "pairwise_distance_rank_preservation",
    "run_controlled_noise_sweep",
    "temporal_continuity_ratio",
]
