"""
Visualization for ENGRAM-FMx.

Creates diagnostic plots for model behavior analysis:
- Memory heatmaps
- Latent PCA trajectories
- Gate activations
- Sparse anchor patterns
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import torch

from neuros_neurofm.backbones.engram_fmx.diagnostics.memory_metrics import (
    compute_memory_entropy,
    compute_memory_usage,
)
from neuros_neurofm.backbones.engram_fmx.diagnostics.latent_metrics import (
    compute_latent_pca,
)


def _check_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for visualization: pip install matplotlib")


def plot_memory_heatmap(
    memory_weights: torch.Tensor,
    ax: Optional[Any] = None,
    title: str = "Memory Retrieval Weights",
    cmap: str = "viridis",
    show_colorbar: bool = True,
) -> Any:
    """Plot heatmap of memory retrieval weights.

    Parameters
    ----------
    memory_weights : torch.Tensor
        Memory weights [K, M] or [B, K, M] (will average over B).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.
    title : str
        Plot title.
    cmap : str
        Colormap name.
    show_colorbar : bool
        Whether to show colorbar.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    _check_matplotlib()

    # Handle batch dimension
    if memory_weights.dim() == 3:
        weights = memory_weights.mean(dim=0)  # Average over batch
    else:
        weights = memory_weights

    weights = weights.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(weights, aspect="auto", cmap=cmap)

    ax.set_xlabel("Memory Slot")
    ax.set_ylabel("Latent Query")
    ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Weight")

    return ax


def plot_memory_entropy_over_time(
    entropy_history: List[float],
    steps: List[int],
    ax: Optional[Any] = None,
    title: str = "Memory Entropy Over Training",
) -> Any:
    """Plot memory entropy over training steps.

    Parameters
    ----------
    entropy_history : List[float]
        Entropy values at each step.
    steps : List[int]
        Training step numbers.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(steps, entropy_history, linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Memory Entropy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_latent_pca(
    latents: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    ax: Optional[Any] = None,
    title: str = "Latent PCA Projection",
    dims: Tuple[int, int] = (0, 1),
    cmap: str = "tab10",
) -> Any:
    """Plot PCA projection of latent states.

    Parameters
    ----------
    latents : torch.Tensor
        Latent states [B, K, D] or [N, D].
    labels : torch.Tensor, optional
        Labels for coloring [B, K] or [N].
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str
        Plot title.
    dims : Tuple[int, int]
        Which PCA dimensions to plot.
    cmap : str
        Colormap for labels.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    _check_matplotlib()

    # Compute PCA
    pca_result = compute_latent_pca(latents, n_components=max(dims) + 1)
    projected = pca_result["projected"]

    # Flatten if needed
    if projected.dim() == 3:
        projected = projected.reshape(-1, projected.shape[-1])

    projected = projected.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    x = projected[:, dims[0]]
    y = projected[:, dims[1]]

    if labels is not None:
        labels_flat = labels.reshape(-1).detach().cpu().numpy()
        scatter = ax.scatter(x, y, c=labels_flat, cmap=cmap, alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label="Label")
    else:
        ax.scatter(x, y, alpha=0.6, s=20)

    # Add variance explained
    var_x = pca_result["explained_variance_ratio"][dims[0]].item()
    var_y = pca_result["explained_variance_ratio"][dims[1]].item()

    ax.set_xlabel(f"PC{dims[0]+1} ({var_x:.1%} var)")
    ax.set_ylabel(f"PC{dims[1]+1} ({var_y:.1%} var)")
    ax.set_title(title)

    return ax


def plot_latent_trajectory_3d(
    latents_over_time: List[torch.Tensor],
    ax: Optional[Any] = None,
    title: str = "Latent Trajectory (3D PCA)",
    cmap: str = "viridis",
) -> Any:
    """Plot 3D PCA trajectory of latents over time/layers.

    Parameters
    ----------
    latents_over_time : List[torch.Tensor]
        List of latent states at each timestep.
    ax : matplotlib.axes.Axes, optional
        3D axes to plot on.
    title : str
        Plot title.
    cmap : str
        Colormap for time.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    _check_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D

    T = len(latents_over_time)

    # Stack and compute PCA
    stacked = torch.stack(latents_over_time, dim=0)  # [T, B, K, D]
    T, B, K, D = stacked.shape
    flat = stacked.reshape(-1, D)  # [T*B*K, D]

    pca_result = compute_latent_pca(flat, n_components=3)
    projected = pca_result["projected"].reshape(T, B, K, 3)

    # Average over batch and latent slots
    trajectory = projected.mean(dim=(1, 2))  # [T, 3]
    trajectory = trajectory.detach().cpu().numpy()

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Color by time
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, T))

    for i in range(T - 1):
        ax.plot(
            trajectory[i:i+2, 0],
            trajectory[i:i+2, 1],
            trajectory[i:i+2, 2],
            color=colors[i],
            linewidth=2,
        )

    # Mark start and end
    ax.scatter(*trajectory[0], c="green", s=100, marker="o", label="Start")
    ax.scatter(*trajectory[-1], c="red", s=100, marker="s", label="End")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend()

    return ax


def plot_gate_activations(
    gate_values: Dict[str, List[float]],
    steps: List[int],
    ax: Optional[Any] = None,
    title: str = "Fusion Gate Activations",
) -> Any:
    """Plot gate activation values over training.

    Parameters
    ----------
    gate_values : Dict[str, List[float]]
        Dictionary mapping gate names to activation histories.
    steps : List[int]
        Training step numbers.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    for name, values in gate_values.items():
        ax.plot(steps[:len(values)], values, label=name, linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gate Value")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    return ax


def plot_sparse_anchor_indices(
    anchor_indices: torch.Tensor,
    sequence_length: int,
    ax: Optional[Any] = None,
    title: str = "Sparse Anchor Selection",
) -> Any:
    """Plot histogram of sparse anchor selections.

    Parameters
    ----------
    anchor_indices : torch.Tensor
        Selected anchor indices [B, P] or [P].
    sequence_length : int
        Total sequence length.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    indices = anchor_indices.reshape(-1).detach().cpu().numpy()

    ax.hist(indices, bins=min(50, sequence_length), range=(0, sequence_length),
            alpha=0.7, edgecolor="black")

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Selection Count")
    ax.set_title(title)
    ax.set_xlim(0, sequence_length)

    return ax


def create_diagnostic_dashboard(
    diagnostics: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> Any:
    """Create a comprehensive diagnostic dashboard.

    Parameters
    ----------
    diagnostics : Dict[str, Any]
        Diagnostics dictionary from ENGRAM backbone.
    save_path : str, optional
        Path to save figure.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Memory heatmap (if available)
    if "memory_weights" in diagnostics:
        ax1 = fig.add_subplot(gs[0, 0])
        plot_memory_heatmap(diagnostics["memory_weights"], ax=ax1)

    # 2. Memory entropy
    if "memory_entropy" in diagnostics:
        ax2 = fig.add_subplot(gs[0, 1])
        entropy = diagnostics["memory_entropy"]
        if isinstance(entropy, list):
            ax2.plot(entropy)
            ax2.set_title("Memory Entropy")
        else:
            ax2.text(0.5, 0.5, f"Entropy: {entropy:.3f}",
                    ha="center", va="center", fontsize=14)
            ax2.set_title("Memory Entropy (single value)")

    # 3. Gate activations (if available)
    gate_keys = [k for k in diagnostics if k.startswith("gate_")]
    if gate_keys:
        ax3 = fig.add_subplot(gs[0, 2])
        gate_values = {k: diagnostics[k] for k in gate_keys}
        for name, val in gate_values.items():
            if isinstance(val, (int, float)):
                ax3.bar(name.replace("gate_", ""), val)
        ax3.set_ylabel("Gate Value")
        ax3.set_title("Fusion Gate Values")
        ax3.set_ylim(0, 1)

    # 4. Sparse anchor indices (if available)
    if "sparse_anchor_indices" in diagnostics:
        ax4 = fig.add_subplot(gs[1, 0])
        indices = diagnostics["sparse_anchor_indices"]
        seq_len = diagnostics.get("sequence_length", 256)
        plot_sparse_anchor_indices(indices, seq_len, ax=ax4)

    # 5. Summary statistics
    ax5 = fig.add_subplot(gs[1, 1:])
    stats_text = "ENGRAM Diagnostics Summary\n" + "=" * 40 + "\n"

    for key, value in diagnostics.items():
        if isinstance(value, (int, float)):
            stats_text += f"{key}: {value:.4f}\n"

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace")
    ax5.axis("off")

    # Add title
    fig.suptitle("ENGRAM-FMx Diagnostic Dashboard", fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
