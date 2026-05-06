"""
Publication-quality visualization utilities for astrocyte events and networks.

This module provides plotting functions for generating figures suitable for
scientific publications, presentations, and exploratory data analysis.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn(
        "matplotlib not installed. Install with: pip install neuros-astro[viz]"
    )

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from neuros_astro.metadata.schema import AstroEvent, AstroGraph


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install neuros-astro[viz]"
        )


def plot_event_raster(
    events: list[AstroEvent],
    frame_rate_hz: float,
    figsize: tuple[float, float] = (12, 6),
    save_path: Optional[Path | str] = None,
    ax: Optional[Axes] = None,
    show_confidence: bool = True,
    title: str = "Astrocyte Calcium Event Raster",
) -> Tuple[Figure, Axes]:
    """
    Plot event raster showing when and where events occur.

    Args:
        events: List of AstroEvent objects
        frame_rate_hz: Frame rate for time conversion
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save figure
        ax: Optional existing axes to plot on
        show_confidence: Whether to color events by confidence
        title: Plot title

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from neuros_astro.events.event_detection import detect_events_from_traces
        >>> events = detect_events_from_traces(traces, frame_rate_hz=10.0, session_id="demo")
        >>> fig, ax = plot_event_raster(events, frame_rate_hz=10.0)
        >>> plt.show()
    """
    _check_matplotlib()

    if len(events) == 0:
        print("⚠️  No events to plot")
        return None, None

    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get unique regions and create mapping
    region_ids = [e.region_id for e in events]
    unique_regions = sorted(set(region_ids))
    region_to_y = {r: i for i, r in enumerate(unique_regions)}

    # Plot each event
    for event in events:
        y = region_to_y[event.region_id]
        onset_s = event.onset_frame / frame_rate_hz
        offset_s = event.offset_frame / frame_rate_hz
        peak_s = event.peak_frame / frame_rate_hz

        # Color by confidence if requested
        if show_confidence:
            color = plt.cm.viridis(event.confidence)
            alpha = 0.7
        else:
            color = 'steelblue'
            alpha = 0.6

        # Event duration as horizontal line
        ax.plot([onset_s, offset_s], [y, y], '-', linewidth=3,
                color=color, alpha=alpha)

        # Peak marker
        ax.plot(peak_s, y, 'o', markersize=5, color='darkred',
                alpha=0.8, zorder=10)

    # Styling
    ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Astrocyte Region', fontsize=13, fontweight='bold')
    ax.set_yticks(range(len(unique_regions)))
    ax.set_yticklabels(unique_regions, fontsize=10)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add colorbar if showing confidence
    if show_confidence:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Confidence', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig, ax


def plot_event_distributions(
    events: list[AstroEvent],
    figsize: tuple[float, float] = (14, 10),
    save_path: Optional[Path | str] = None,
    bins: int = 25,
) -> Figure:
    """
    Plot distributions of event features in a multi-panel figure.

    Args:
        events: List of AstroEvent objects
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        bins: Number of histogram bins

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_event_distributions(events)
        >>> plt.show()
    """
    _check_matplotlib()

    if len(events) == 0:
        print("⚠️  No events to plot")
        return None

    # Extract features
    durations = np.array([e.duration_s for e in events])
    amplitudes = np.array([e.peak_dff for e in events])
    confidences = np.array([e.confidence for e in events])

    # Spatial features (if available)
    has_spatial = any(e.area_px is not None for e in events)
    if has_spatial:
        areas = np.array([e.area_px if e.area_px is not None else np.nan
                         for e in events])
        areas = areas[~np.isnan(areas)]

    # Create figure
    fig = plt.figure(figsize=figsize)

    if has_spatial:
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    else:
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Duration distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(durations, bins=bins, color='skyblue', edgecolor='black',
             alpha=0.7, linewidth=1.5)
    ax1.axvline(np.median(durations), color='red', linestyle='--',
                linewidth=2, label=f'Median: {np.median(durations):.2f}s')
    ax1.set_xlabel('Duration (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Event Duration Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Amplitude distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(amplitudes, bins=bins, color='lightcoral', edgecolor='black',
             alpha=0.7, linewidth=1.5)
    ax2.axvline(np.median(amplitudes), color='red', linestyle='--',
                linewidth=2, label=f'Median: {np.median(amplitudes):.3f}')
    ax2.set_xlabel('Peak ΔF/F', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Event Amplitude Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Confidence distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(confidences, bins=bins, color='lightgreen', edgecolor='black',
             alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Event Confidence Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Duration vs Amplitude scatter
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(durations, amplitudes, c=confidences, cmap='viridis',
                         s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Duration (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Peak ΔF/F', fontsize=12, fontweight='bold')
    ax4.set_title('Duration vs Amplitude', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Confidence', fontsize=11, fontweight='bold')

    # Spatial area distribution (if available)
    if has_spatial:
        ax5 = fig.add_subplot(gs[2, :])
        ax5.hist(areas, bins=bins, color='plum', edgecolor='black',
                alpha=0.7, linewidth=1.5)
        ax5.axvline(np.median(areas), color='red', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(areas):.1f} px')
        ax5.set_xlabel('Spatial Area (pixels)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax5.set_title('Event Spatial Area Distribution', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

    plt.suptitle('Astrocyte Event Feature Distributions',
                fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig


def plot_event_statistics_summary(events: list[AstroEvent]) -> None:
    """
    Print comprehensive statistical summary of events.

    Args:
        events: List of AstroEvent objects

    Example:
        >>> plot_event_statistics_summary(events)
    """
    if len(events) == 0:
        print("⚠️  No events to summarize")
        return

    durations = np.array([e.duration_s for e in events])
    amplitudes = np.array([e.peak_dff for e in events])
    confidences = np.array([e.confidence for e in events])

    print("=" * 70)
    print("  ASTROCYTE EVENT STATISTICS SUMMARY")
    print("=" * 70)
    print(f"\n📊 Total events: {len(events)}")
    print(f"📊 Unique regions: {len(set(e.region_id for e in events))}")
    print()

    print("⏱️  Duration (seconds):")
    print(f"   Mean ± SD:      {np.mean(durations):.2f} ± {np.std(durations):.2f}")
    print(f"   Median [IQR]:   {np.median(durations):.2f} "
          f"[{np.percentile(durations, 25):.2f}-{np.percentile(durations, 75):.2f}]")
    print(f"   Range:          {np.min(durations):.2f} - {np.max(durations):.2f}")
    print()

    print("📈 Amplitude (ΔF/F):")
    print(f"   Mean ± SD:      {np.mean(amplitudes):.3f} ± {np.std(amplitudes):.3f}")
    print(f"   Median [IQR]:   {np.median(amplitudes):.3f} "
          f"[{np.percentile(amplitudes, 25):.3f}-{np.percentile(amplitudes, 75):.3f}]")
    print(f"   Range:          {np.min(amplitudes):.3f} - {np.max(amplitudes):.3f}")
    print()

    print("✓ Confidence:")
    print(f"   Mean ± SD:      {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
    print(f"   Range:          {np.min(confidences):.3f} - {np.max(confidences):.3f}")

    # Check if spatial features available
    has_spatial = any(e.area_px is not None for e in events)
    if has_spatial:
        areas = np.array([e.area_px for e in events if e.area_px is not None])
        print()
        print("🗺️  Spatial Area (pixels):")
        print(f"   Mean ± SD:      {np.mean(areas):.1f} ± {np.std(areas):.1f}")
        print(f"   Median [IQR]:   {np.median(areas):.1f} "
              f"[{np.percentile(areas, 25):.1f}-{np.percentile(areas, 75):.1f}]")

    print("=" * 70)


def plot_network_graph(
    graph: AstroGraph,
    figsize: tuple[float, float] = (10, 10),
    save_path: Optional[Path | str] = None,
    ax: Optional[Axes] = None,
    layout: str = "spring",
    node_color: str = "lightblue",
    show_weights: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot astrocyte functional connectivity graph.

    Args:
        graph: AstroGraph object
        figsize: Figure size
        save_path: Optional save path
        ax: Optional existing axes
        layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')
        node_color: Node color
        show_weights: Whether to show edge weights as line widths

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from neuros_astro.networks.functional_connectivity import build_event_coactivation_graph
        >>> graphs = build_event_coactivation_graph(events, session_id="demo", frame_rate_hz=10.0)
        >>> fig, ax = plot_network_graph(graphs[0])
        >>> plt.show()
    """
    _check_matplotlib()

    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for network visualization. "
            "Install with: pip install networkx"
        )

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)

    # Add edges with weights
    for (source, target), weight in zip(graph.edges, graph.edge_weights):
        G.add_edge(source, target, weight=weight)

    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_color,
        node_size=800,
        edgecolors='black',
        linewidths=2,
        alpha=0.9
    )

    # Draw edges with weights as line width
    if show_weights and len(graph.edge_weights) > 0:
        # Normalize weights for visualization
        weights = np.array(graph.edge_weights)
        normalized_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        edge_widths = 0.5 + normalized_weights * 4  # Range: 0.5 to 4.5

        for (source, target), width in zip(graph.edges, edge_widths):
            nx.draw_networkx_edges(
                G, pos, [(source, target)], ax=ax,
                width=width,
                alpha=0.6,
                edge_color='gray'
            )
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')

    # Title
    ax.set_title(
        f'Astrocyte Functional Network\n'
        f't = [{graph.window_start_s:.1f}, {graph.window_end_s:.1f}]s',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig, ax


def plot_network_evolution(
    graphs: list[AstroGraph],
    figsize: tuple[float, float] = (16, 4),
    save_path: Optional[Path | str] = None,
    max_graphs: int = 5,
) -> Figure:
    """
    Plot temporal evolution of network structure across multiple windows.

    Args:
        graphs: List of AstroGraph objects
        figsize: Figure size
        save_path: Optional save path
        max_graphs: Maximum number of graphs to show

    Returns:
        matplotlib Figure

    Example:
        >>> graphs = build_event_coactivation_graph(events, session_id="demo", frame_rate_hz=10.0)
        >>> fig = plot_network_evolution(graphs)
        >>> plt.show()
    """
    _check_matplotlib()

    n_graphs = min(len(graphs), max_graphs)
    if n_graphs == 0:
        print("⚠️  No graphs to plot")
        return None

    fig, axes = plt.subplots(1, n_graphs, figsize=figsize)
    if n_graphs == 1:
        axes = [axes]

    # Plot each graph
    for i, graph in enumerate(graphs[:n_graphs]):
        plot_network_graph(graph, ax=axes[i], show_weights=True, layout="spring")

    plt.suptitle('Astrocyte Network Temporal Evolution',
                fontsize=16, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig


def plot_trace_with_events(
    trace: np.ndarray,
    events: list[AstroEvent],
    region_id: str,
    frame_rate_hz: float,
    figsize: tuple[float, float] = (14, 5),
    save_path: Optional[Path | str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot fluorescence trace with detected events overlaid.

    Args:
        trace: 1D array of fluorescence values
        events: List of events for this region
        region_id: Region identifier
        frame_rate_hz: Frame rate
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> region_events = [e for e in events if e.region_id == 'roi_05']
        >>> fig, ax = plot_trace_with_events(traces[5], region_events, 'roi_05', 10.0)
        >>> plt.show()
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Time axis
    time = np.arange(len(trace)) / frame_rate_hz

    # Plot trace
    ax.plot(time, trace, 'k-', linewidth=1, alpha=0.7, label='ΔF/F trace')

    # Overlay events
    region_events = [e for e in events if e.region_id == region_id]

    for event in region_events:
        onset_s = event.onset_frame / frame_rate_hz
        offset_s = event.offset_frame / frame_rate_hz
        peak_s = event.peak_frame / frame_rate_hz

        # Event span
        ax.axvspan(onset_s, offset_s, alpha=0.3, color='lightcoral',
                  label='Event' if event == region_events[0] else '')

        # Peak marker
        ax.plot(peak_s, event.peak_dff, 'ro', markersize=8,
               label='Peak' if event == region_events[0] else '', zorder=10)

    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ΔF/F', fontsize=12, fontweight='bold')
    ax.set_title(f'Astrocyte Trace with Events: {region_id}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig, ax
