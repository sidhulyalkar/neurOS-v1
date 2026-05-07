"""
ENGRAM-FMx Diagnostics and Visualization.

Provides tools for analyzing and visualizing ENGRAM model behavior:
- Memory retrieval patterns
- Latent workspace trajectories
- Gate activations
- Sparse anchor attention patterns
"""

from neuros_neurofm.backbones.engram_fmx.diagnostics.memory_metrics import (
    compute_memory_entropy,
    compute_memory_usage,
    analyze_memory_retrieval,
    MemoryTracker,
)
from neuros_neurofm.backbones.engram_fmx.diagnostics.latent_metrics import (
    compute_latent_pca,
    compute_latent_similarity,
    track_latent_trajectory,
    LatentTracker,
)
from neuros_neurofm.backbones.engram_fmx.diagnostics.visualize import (
    plot_memory_heatmap,
    plot_memory_entropy_over_time,
    plot_latent_pca,
    plot_latent_trajectory_3d,
    plot_gate_activations,
    plot_sparse_anchor_indices,
    create_diagnostic_dashboard,
)

__all__ = [
    # Memory metrics
    "compute_memory_entropy",
    "compute_memory_usage",
    "analyze_memory_retrieval",
    "MemoryTracker",
    # Latent metrics
    "compute_latent_pca",
    "compute_latent_similarity",
    "track_latent_trajectory",
    "LatentTracker",
    # Visualization
    "plot_memory_heatmap",
    "plot_memory_entropy_over_time",
    "plot_latent_pca",
    "plot_latent_trajectory_3d",
    "plot_gate_activations",
    "plot_sparse_anchor_indices",
    "create_diagnostic_dashboard",
]
