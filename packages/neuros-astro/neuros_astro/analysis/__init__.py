"""
Advanced analysis tools for neuros-astro.

This package provides statistical analysis, comparative methods,
and publication-ready analysis workflows.
"""

from neuros_astro.analysis.statistics import (
    compute_event_statistics,
    compare_event_distributions,
    permutation_test,
    bootstrap_confidence_interval,
    effect_size_cohens_d,
)

from neuros_astro.analysis.network_metrics import (
    compute_network_stability,
    compute_temporal_network_metrics,
    detect_network_communities,
    compute_network_motifs,
)

from neuros_astro.analysis.cross_session import (
    align_sessions,
    compute_cross_session_similarity,
    identify_stable_patterns,
)

__all__ = [
    # Statistics
    "compute_event_statistics",
    "compare_event_distributions",
    "permutation_test",
    "bootstrap_confidence_interval",
    "effect_size_cohens_d",
    # Network metrics
    "compute_network_stability",
    "compute_temporal_network_metrics",
    "detect_network_communities",
    "compute_network_motifs",
    # Cross-session
    "align_sessions",
    "compute_cross_session_similarity",
    "identify_stable_patterns",
]
