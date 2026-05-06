"""
Publication-quality figure generation pipeline.

Provides high-level functions to generate complete figure panels
for manuscripts, posters, and presentations.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neuros_astro.metadata.schema import AstroEvent, AstroGraph
from neuros_astro.analysis.statistics import compute_event_statistics, EventStatistics
from neuros_astro.analysis.network_metrics import compute_temporal_network_metrics

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not installed")

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def generate_figure_1_pipeline_overview(
    events: List[AstroEvent],
    graphs: List[AstroGraph],
    trace_sample: Optional[np.ndarray] = None,
    frame_rate_hz: float = 10.0,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """
    Generate Figure 1: neuros-astro pipeline overview.

    Four panels:
    A) Example trace with detected events
    B) Event raster across regions
    C) Network graph snapshot
    D) Event feature distributions

    Args:
        events: List of detected events
        graphs: List of network graphs
        trace_sample: Optional sample trace to show
        frame_rate_hz: Frame rate
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure

    Example:
        >>> fig = generate_figure_1_pipeline_overview(
        ...     events, graphs, trace_sample=traces[0], frame_rate_hz=10.0,
        ...     save_path="figures/figure1_pipeline.png"
        ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    from neuros_astro.visualization.event_plots import (
        plot_event_raster,
        plot_event_distributions,
        plot_trace_with_events,
        plot_network_graph,
    )

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Trace with events
    ax_a = fig.add_subplot(gs[0, 0])
    if trace_sample is not None:
        region_id = events[0].region_id if events else "roi_00"
        region_events = [e for e in events if e.region_id == region_id]
        plot_trace_with_events(trace_sample, region_events, region_id,
                               frame_rate_hz, save_path=None)
        ax_a.set_title("A) Example Trace with Detected Events",
                      fontsize=14, fontweight='bold', loc='left')

    # Panel B: Event raster
    ax_b = fig.add_subplot(gs[0, 1])
    plot_event_raster(events, frame_rate_hz, ax=ax_b, show_confidence=True)
    ax_b.set_title("B) Event Raster Across Regions",
                   fontsize=14, fontweight='bold', loc='left')

    # Panel C: Network graph
    ax_c = fig.add_subplot(gs[1, 0])
    if graphs:
        plot_network_graph(graphs[0], ax=ax_c, layout='spring', show_weights=True)
        ax_c.set_title("C) Functional Connectivity Network",
                       fontsize=14, fontweight='bold', loc='left')

    # Panel D: Event distributions (create subplot for inset)
    ax_d = fig.add_subplot(gs[1, 1])
    # Plot distributions manually to fit in single panel
    durations = [e.duration_s for e in events]
    amplitudes = [e.peak_dff for e in events]

    ax_d_inset = ax_d.inset_axes([0, 0.5, 1, 0.45])
    ax_d_inset.hist(durations, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax_d_inset.set_xlabel('Duration (s)', fontsize=10)
    ax_d_inset.set_ylabel('Count', fontsize=10)
    ax_d_inset.set_title('Event Durations', fontsize=11, fontweight='bold')

    ax_d.hist(amplitudes, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax_d.set_xlabel('Peak ΔF/F', fontsize=11, fontweight='bold')
    ax_d.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax_d.set_title("D) Event Feature Distributions",
                   fontsize=14, fontweight='bold', loc='left')

    plt.suptitle('neuros-astro Pipeline Overview',
                fontsize=18, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 1 to {save_path}")

    return fig


def generate_figure_2_validation(
    events_synthetic: List[AstroEvent],
    events_real: List[AstroEvent],
    save_path: Optional[str | Path] = None,
) -> Figure:
    """
    Generate Figure 2: Validation on synthetic and real data.

    Compares event detection performance on synthetic ground truth
    vs real data characteristics.

    Args:
        events_synthetic: Events from synthetic data
        events_real: Events from real data
        save_path: Optional path to save

    Returns:
        matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Synthetic data statistics
    stats_syn = compute_event_statistics(events_synthetic)
    stats_real = compute_event_statistics(events_real)

    # Plot comparisons
    metrics = ['duration_mean', 'amplitude_mean', 'confidence_mean']
    metric_labels = ['Mean Duration (s)', 'Mean Amplitude (ΔF/F)', 'Mean Confidence']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[0, idx]
        values = [getattr(stats_syn, metric), getattr(stats_real, metric)]
        colors = ['skyblue', 'lightcoral']
        ax.bar(['Synthetic', 'Real'], values, color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Distribution comparisons
    ax = axes[1, 0]
    ax.hist([e.duration_s for e in events_synthetic], bins=20, alpha=0.6,
           label='Synthetic', color='skyblue', edgecolor='black')
    ax.hist([e.duration_s for e in events_real], bins=20, alpha=0.6,
           label='Real', color='lightcoral', edgecolor='black')
    ax.set_xlabel('Duration (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.legend()
    ax.set_title('Duration Distributions', fontsize=12, fontweight='bold')

    ax = axes[1, 1]
    ax.hist([e.peak_dff for e in events_synthetic], bins=20, alpha=0.6,
           label='Synthetic', color='skyblue', edgecolor='black')
    ax.hist([e.peak_dff for e in events_real], bins=20, alpha=0.6,
           label='Real', color='lightcoral', edgecolor='black')
    ax.set_xlabel('Peak ΔF/F', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.legend()
    ax.set_title('Amplitude Distributions', fontsize=12, fontweight='bold')

    # Event rate comparison
    ax = axes[1, 2]
    event_counts_syn = {}
    event_counts_real = {}

    for e in events_synthetic:
        event_counts_syn[e.region_id] = event_counts_syn.get(e.region_id, 0) + 1

    for e in events_real:
        event_counts_real[e.region_id] = event_counts_real.get(e.region_id, 0) + 1

    ax.bar(range(len(event_counts_syn)), list(event_counts_syn.values()),
          alpha=0.6, label='Synthetic', color='skyblue', edgecolor='black')
    ax.bar(range(len(event_counts_real)), list(event_counts_real.values()),
          alpha=0.6, label='Real', color='lightcoral', edgecolor='black')
    ax.set_xlabel('Region Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Event Count', fontsize=11, fontweight='bold')
    ax.legend()
    ax.set_title('Events per Region', fontsize=12, fontweight='bold')

    plt.suptitle('Validation: Synthetic vs Real Data',
                fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 2 to {save_path}")

    return fig


def generate_figure_3_network_analysis(
    graphs: List[AstroGraph],
    save_path: Optional[str | Path] = None,
) -> Figure:
    """
    Generate Figure 3: Network dynamics and stability.

    Shows temporal evolution of network metrics and stability.

    Args:
        graphs: List of temporal network graphs
        save_path: Optional path to save

    Returns:
        matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    from neuros_astro.visualization.event_plots import plot_network_evolution
    from neuros_astro.analysis.network_metrics import compute_network_stability

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Network evolution (top row, full width)
    ax_evolution = fig.add_subplot(gs[0, :])
    plot_network_evolution(graphs, max_graphs=5, save_path=None)
    ax_evolution.set_title("A) Temporal Network Evolution",
                           fontsize=14, fontweight='bold', loc='left')

    # Compute network metrics
    metrics = compute_temporal_network_metrics(graphs)

    # Panel B: Network density over time
    ax_b = fig.add_subplot(gs[1, 0])
    window_centers = [(m.window_start_s + m.window_end_s) / 2 for m in metrics]
    densities = [m.density for m in metrics]

    ax_b.plot(window_centers, densities, 'o-', linewidth=2, markersize=8,
             color='steelblue', markerfacecolor='lightblue', markeredgecolor='black',
             markeredgewidth=1.5)
    ax_b.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Network Density', fontsize=11, fontweight='bold')
    ax_b.set_title('B) Network Density Over Time', fontsize=12, fontweight='bold')
    ax_b.grid(True, alpha=0.3)

    # Panel C: Network stability
    ax_c = fig.add_subplot(gs[1, 1])
    stability = compute_network_stability(graphs, method='jaccard')

    if len(stability) > 0:
        ax_c.plot(range(len(stability)), stability, 'o-', linewidth=2, markersize=8,
                 color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='black',
                 markeredgewidth=1.5)
        ax_c.axhline(y=np.mean(stability), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(stability):.3f}')
        ax_c.set_xlabel('Time Window Index', fontsize=11, fontweight='bold')
        ax_c.set_ylabel('Jaccard Similarity', fontsize=11, fontweight='bold')
        ax_c.set_title('C) Network Stability (Consecutive Windows)',
                       fontsize=12, fontweight='bold')
        ax_c.legend()
        ax_c.grid(True, alpha=0.3)

    plt.suptitle('Network Dynamics and Stability',
                fontsize=18, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 3 to {save_path}")

    return fig


def generate_figure_4_ablation_results(
    ablation_data: Dict[str, Dict[str, float]],
    save_path: Optional[str | Path] = None,
) -> Figure:
    """
    Generate Figure 4: Ablation study results.

    Shows model performance with/without astrocyte modality.

    Args:
        ablation_data: Dict of condition_name -> {metric_name: value}
        save_path: Optional path to save

    Returns:
        matplotlib Figure

    Example:
        >>> ablation_data = {
        ...     'neural_only': {'prediction_loss': 0.25, 'decoding_acc': 0.72},
        ...     'neural+astro': {'prediction_loss': 0.18, 'decoding_acc': 0.83},
        ... }
        >>> fig = generate_figure_4_ablation_results(ablation_data)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    conditions = list(ablation_data.keys())
    metrics = list(ablation_data[conditions[0]].keys())

    # Panel A: Grouped bar chart
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.35
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']

    for idx, condition in enumerate(conditions):
        values = [ablation_data[condition][m] for m in metrics]
        offset = width * (idx - len(conditions) / 2 + 0.5)
        ax.bar(x + offset, values, width, label=condition,
              color=colors[idx % len(colors)], edgecolor='black', alpha=0.7)

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('A) Model Performance Comparison',
                fontsize=14, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Improvement percentages
    ax = axes[1]
    if len(conditions) >= 2:
        baseline = conditions[0]
        improvements = {}

        for metric in metrics:
            baseline_val = ablation_data[baseline][metric]

            for condition in conditions[1:]:
                test_val = ablation_data[condition][metric]

                if baseline_val != 0:
                    improvement = ((test_val - baseline_val) / abs(baseline_val)) * 100
                else:
                    improvement = 0.0

                key = f"{condition} vs {baseline}"
                if key not in improvements:
                    improvements[key] = {}

                improvements[key][metric] = improvement

        x = np.arange(len(metrics))
        width = 0.35

        for idx, (comp_name, imp_data) in enumerate(improvements.items()):
            values = [imp_data[m] for m in metrics]
            offset = width * (idx - len(improvements) / 2 + 0.5)
            ax.bar(x + offset, values, width, label=comp_name,
                  color=colors[(idx + 1) % len(colors)], edgecolor='black', alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percent Change (%)', fontsize=12, fontweight='bold')
        ax.set_title('B) Performance Improvement',
                    fontsize=14, fontweight='bold', loc='left')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Ablation Study Results',
                fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 4 to {save_path}")

    return fig
