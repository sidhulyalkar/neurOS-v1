"""neurOS SourceWeigher: reliability-aware source selection and fusion.

The numerical core is NumPy-only. FastAPI is an optional deployment boundary,
not a requirement for in-process training or runtime use.
"""
from __future__ import annotations

from .client import SourceWeightClient
from .diagnostics import (
    PerturbationReport,
    SourceStabilityReport,
    effective_sample_size,
    jensen_shannon_weight_shift,
    leave_one_source_out_stability,
    target_perturbation_sensitivity,
)
from .distribution import (
    MMDSourceWeigher,
    RiemannianCovarianceWeigher,
    rbf_mmd2,
    spd_affine_invariant_distance,
)
from .integration import ReliabilityWeightedFusion, RepresentationSourceWeigher
from .strategies import DistanceWeigher, GibbsRiskWeigher, OnlineSourceWeigher
from .summaries import RunningFeatureSummary, summarize_features
from .weigher import (
    SourceWeigher,
    WeightingDiagnostics,
    WeightingResult,
    project_to_simplex,
)

__version__ = "0.2.0"

__all__ = [
    "DistanceWeigher",
    "GibbsRiskWeigher",
    "OnlineSourceWeigher",
    "MMDSourceWeigher",
    "RiemannianCovarianceWeigher",
    "PerturbationReport",
    "ReliabilityWeightedFusion",
    "RepresentationSourceWeigher",
    "RunningFeatureSummary",
    "SourceStabilityReport",
    "SourceWeightClient",
    "SourceWeigher",
    "WeightingDiagnostics",
    "WeightingResult",
    "effective_sample_size",
    "jensen_shannon_weight_shift",
    "leave_one_source_out_stability",
    "project_to_simplex",
    "rbf_mmd2",
    "spd_affine_invariant_distance",
    "summarize_features",
    "target_perturbation_sensitivity",
]


def __getattr__(name: str):
    # Preserve the historical ``from neuros_sourceweigher import app`` API
    # without making FastAPI a mandatory package dependency.
    if name in {"app", "create_app"}:
        try:
            from .service import app, create_app
        except ImportError as exc:
            raise ImportError(
                "FastAPI service support is optional. "
                "Install `neuros-sourceweigher[service]`."
            ) from exc
        return app if name == "app" else create_app
    raise AttributeError(name)
