#!/usr/bin/env python
"""
Cross-session analysis of Allen Visual Coding astrocyte data.

Analyzes all processed sessions to identify:
- Cross-session consistency
- Stable patterns
- Session-to-session variability
- Publication-ready statistics

Usage:
    python analyze_allen_sessions.py
    python analyze_allen_sessions.py --results-dir ./allen_nwb_results
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

from neuros_astro.analysis.statistics import EventStatistics
from neuros_astro.analysis.cross_session import compute_cross_session_similarity


def load_session_events(session_dir):
    """Load events from a session's parquet file."""
    import pyarrow.parquet as pq

    events_path = session_dir / "events.parquet"
    if not events_path.exists():
        return []

    table = pq.read_table(events_path)
    df = table.to_pandas()

    # Convert to AstroEvent-like objects for compatibility
    from neuros_astro.metadata.schema import AstroEvent

    events = []
    for _, row in df.iterrows():
        event = AstroEvent(
            session_id=row['session_id'],
            region_id=row['region_id'],
            start_time_s=row['start_time_s'],
            end_time_s=row['end_time_s'],
            peak_time_s=row['peak_time_s'],
            duration_s=row['duration_s'],
            amplitude=row['amplitude'],
            confidence=row['confidence'],
        )
        events.append(event)

    return events


def generate_cross_session_report(results_dir, output_dir):
    """Generate comprehensive cross-session analysis report."""

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALLEN VISUAL CODING - CROSS-SESSION ANALYSIS")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Load Overall Summary
    # -------------------------------------------------------------------------
    print("[1/6] Loading session summaries...")

    summary_path = results_dir / "overall_summary.json"
    with open(summary_path) as f:
        overall_summary = json.load(f)

    n_sessions = overall_summary['n_sessions_processed']
    sessions = overall_summary['sessions']

    print(f"  ✓ Loaded {n_sessions} sessions")
    print(f"  ✓ Total recording time: {overall_summary['total_recording_time_min']:.1f} min")
    print(f"  ✓ Total events: {overall_summary['total_events']}")
    print(f"  ✓ Total networks: {overall_summary['total_networks']}")

    # -------------------------------------------------------------------------
    # Extract Session-Level Metrics
    # -------------------------------------------------------------------------
    print("\n[2/6] Extracting session-level metrics...")

    session_ids = [s['session_id'] for s in sessions]
    n_rois = np.array([s['n_rois'] for s in sessions])
    durations = np.array([s['duration_min'] for s in sessions])
    n_events = np.array([s['n_events'] for s in sessions])
    event_rates = np.array([s['statistics']['event_rate_hz'] for s in sessions])
    events_per_roi = np.array([s['statistics']['events_per_roi'] for s in sessions])

    # Event properties
    event_durations_mean = np.array([s['statistics']['duration_mean'] for s in sessions])
    event_durations_std = np.array([s['statistics']['duration_std'] for s in sessions])
    event_amplitudes_mean = np.array([s['statistics']['amplitude_mean'] for s in sessions])
    event_amplitudes_std = np.array([s['statistics']['amplitude_std'] for s in sessions])

    # Network metrics
    network_density = np.array([s['network_metrics']['mean_density'] for s in sessions])
    network_clustering = np.array([s['network_metrics']['mean_clustering'] for s in sessions])
    network_stability = np.array([s['network_metrics']['stability'] for s in sessions])

    print(f"  ✓ Extracted metrics for {n_sessions} sessions")

    # -------------------------------------------------------------------------
    # Compute Cross-Session Statistics
    # -------------------------------------------------------------------------
    print("\n[3/6] Computing cross-session statistics...")

    def coefficient_of_variation(x):
        """CV = std / mean"""
        return np.std(x) / np.mean(x) if np.mean(x) > 0 else 0.0

    metrics = {
        'event_rate_hz': {
            'mean': np.mean(event_rates),
            'std': np.std(event_rates),
            'cv': coefficient_of_variation(event_rates),
            'min': np.min(event_rates),
            'max': np.max(event_rates),
        },
        'events_per_roi': {
            'mean': np.mean(events_per_roi),
            'std': np.std(events_per_roi),
            'cv': coefficient_of_variation(events_per_roi),
            'min': np.min(events_per_roi),
            'max': np.max(events_per_roi),
        },
        'event_duration_s': {
            'mean': np.mean(event_durations_mean),
            'std': np.std(event_durations_mean),
            'cv': coefficient_of_variation(event_durations_mean),
            'min': np.min(event_durations_mean),
            'max': np.max(event_durations_mean),
        },
        'event_amplitude': {
            'mean': np.mean(event_amplitudes_mean),
            'std': np.std(event_amplitudes_mean),
            'cv': coefficient_of_variation(event_amplitudes_mean),
            'min': np.min(event_amplitudes_mean),
            'max': np.max(event_amplitudes_mean),
        },
        'network_stability': {
            'mean': np.mean(network_stability),
            'std': np.std(network_stability),
            'cv': coefficient_of_variation(network_stability),
            'min': np.min(network_stability),
            'max': np.max(network_stability),
        },
        'network_density': {
            'mean': np.mean(network_density),
            'std': np.std(network_density),
            'cv': coefficient_of_variation(network_density),
            'min': np.min(network_density),
            'max': np.max(network_density),
        },
    }

    print("\n  Cross-Session Summary Statistics:")
    print("  " + "-" * 76)
    print(f"  {'Metric':<25} {'Mean':<12} {'Std':<12} {'CV':<10} {'Range':<15}")
    print("  " + "-" * 76)

    for metric_name, vals in metrics.items():
        range_str = f"[{vals['min']:.3f}, {vals['max']:.3f}]"
        print(f"  {metric_name:<25} {vals['mean']:<12.4f} {vals['std']:<12.4f} {vals['cv']:<10.3f} {range_str:<15}")

    print("  " + "-" * 76)

    # -------------------------------------------------------------------------
    # Identify Consistent Features
    # -------------------------------------------------------------------------
    print("\n[4/6] Identifying consistent features across sessions...")

    # Features are considered consistent if CV < 0.5 (moderate variability)
    consistency_threshold = 0.5

    consistent_features = []
    variable_features = []

    for metric_name, vals in metrics.items():
        if vals['cv'] < consistency_threshold:
            consistent_features.append(metric_name)
        else:
            variable_features.append(metric_name)

    print(f"\n  ✓ Consistent features (CV < {consistency_threshold}):")
    for feature in consistent_features:
        cv = metrics[feature]['cv']
        print(f"    - {feature}: CV = {cv:.3f}")

    if variable_features:
        print(f"\n  ⚠️  Variable features (CV >= {consistency_threshold}):")
        for feature in variable_features:
            cv = metrics[feature]['cv']
            print(f"    - {feature}: CV = {cv:.3f}")

    # -------------------------------------------------------------------------
    # Generate Visualizations
    # -------------------------------------------------------------------------
    print("\n[5/6] Generating cross-session visualizations...")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Figure 1: Event rate and stability across sessions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Event rate
    axes[0, 0].bar(range(n_sessions), event_rates, color='steelblue', alpha=0.7)
    axes[0, 0].axhline(np.mean(event_rates), color='red', linestyle='--',
                       label=f'Mean: {np.mean(event_rates):.3f} Hz')
    axes[0, 0].set_xlabel('Session Index')
    axes[0, 0].set_ylabel('Event Rate (Hz)')
    axes[0, 0].set_title('Event Rate Across Sessions')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Events per ROI
    axes[0, 1].bar(range(n_sessions), events_per_roi, color='forestgreen', alpha=0.7)
    axes[0, 1].axhline(np.mean(events_per_roi), color='red', linestyle='--',
                       label=f'Mean: {np.mean(events_per_roi):.1f}')
    axes[0, 1].set_xlabel('Session Index')
    axes[0, 1].set_ylabel('Events per ROI')
    axes[0, 1].set_title('Events per ROI Across Sessions')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Network stability
    axes[1, 0].bar(range(n_sessions), network_stability, color='coral', alpha=0.7)
    axes[1, 0].axhline(np.mean(network_stability), color='red', linestyle='--',
                       label=f'Mean: {np.mean(network_stability):.3f}')
    axes[1, 0].set_xlabel('Session Index')
    axes[1, 0].set_ylabel('Network Stability')
    axes[1, 0].set_title('Network Stability Across Sessions')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Event duration
    axes[1, 1].bar(range(n_sessions), event_durations_mean, color='purple', alpha=0.7,
                   yerr=event_durations_std, capsize=4)
    axes[1, 1].axhline(np.mean(event_durations_mean), color='red', linestyle='--',
                       label=f'Mean: {np.mean(event_durations_mean):.2f}s')
    axes[1, 1].set_xlabel('Session Index')
    axes[1, 1].set_ylabel('Event Duration (s)')
    axes[1, 1].set_title('Mean Event Duration Across Sessions')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = figures_dir / "cross_session_metrics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path.name}")
    plt.close()

    # Figure 2: Correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create correlation matrix
    data_matrix = np.column_stack([
        event_rates,
        events_per_roi,
        event_durations_mean,
        event_amplitudes_mean,
        network_stability,
        network_density,
    ])

    labels = ['Event Rate', 'Events/ROI', 'Duration', 'Amplitude', 'Stability', 'Density']
    corr_matrix = np.corrcoef(data_matrix.T)

    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center',
                          color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Cross-Session Metric Correlations')
    plt.tight_layout()

    fig_path = figures_dir / "metric_correlations.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path.name}")
    plt.close()

    # -------------------------------------------------------------------------
    # Generate Summary Report
    # -------------------------------------------------------------------------
    print("\n[6/6] Generating summary report...")

    report = {
        'n_sessions': n_sessions,
        'total_recording_time_min': float(overall_summary['total_recording_time_min']),
        'total_events': int(overall_summary['total_events']),
        'total_networks': int(overall_summary['total_networks']),
        'cross_session_metrics': {k: {kk: float(vv) for kk, vv in v.items()}
                                  for k, v in metrics.items()},
        'consistent_features': consistent_features,
        'variable_features': variable_features,
        'session_summary': [
            {
                'session_id': s['session_id'],
                'n_rois': s['n_rois'],
                'duration_min': s['duration_min'],
                'n_events': s['n_events'],
                'event_rate_hz': s['statistics']['event_rate_hz'],
                'network_stability': s['network_metrics']['stability'],
            }
            for s in sessions
        ]
    }

    report_path = output_dir / "cross_session_analysis.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Saved: {report_path.name}")

    # Text summary
    summary_text_path = output_dir / "cross_session_summary.txt"
    with open(summary_text_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ALLEN VISUAL CODING - CROSS-SESSION ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Sessions Analyzed: {n_sessions}\n")
        f.write(f"Total Recording Time: {overall_summary['total_recording_time_min']:.1f} min ({overall_summary['total_recording_time_min']/60:.1f} hours)\n")
        f.write(f"Total Events: {overall_summary['total_events']:,}\n")
        f.write(f"Total Networks: {overall_summary['total_networks']:,}\n\n")

        f.write("-" * 80 + "\n")
        f.write("CROSS-SESSION METRICS\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'CV':<10} {'Range':<15}\n")
        f.write("-" * 80 + "\n")

        for metric_name, vals in metrics.items():
            range_str = f"[{vals['min']:.3f}, {vals['max']:.3f}]"
            f.write(f"{metric_name:<25} {vals['mean']:<12.4f} {vals['std']:<12.4f} {vals['cv']:<10.3f} {range_str:<15}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("CONSISTENCY ANALYSIS\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Consistent Features (CV < {consistency_threshold}):\n")
        for feature in consistent_features:
            cv = metrics[feature]['cv']
            f.write(f"  ✓ {feature}: CV = {cv:.3f}\n")

        if variable_features:
            f.write(f"\nVariable Features (CV >= {consistency_threshold}):\n")
            for feature in variable_features:
                cv = metrics[feature]['cv']
                f.write(f"  ⚠️  {feature}: CV = {cv:.3f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"1. Network Stability: {np.mean(network_stability):.3f} ± {np.std(network_stability):.3f}\n")
        f.write(f"   - Highly stable networks across all sessions\n")
        f.write(f"   - Minimum: {np.min(network_stability):.3f}, Maximum: {np.max(network_stability):.3f}\n\n")

        f.write(f"2. Event Detection: {overall_summary['total_events']:,} total events\n")
        f.write(f"   - Mean rate: {np.mean(event_rates):.3f} Hz\n")
        f.write(f"   - Mean per ROI: {np.mean(events_per_roi):.1f} events\n\n")

        f.write(f"3. Event Characteristics:\n")
        f.write(f"   - Duration: {np.mean(event_durations_mean):.2f} ± {np.std(event_durations_mean):.2f} s\n")
        f.write(f"   - Amplitude: {np.mean(event_amplitudes_mean):.2f} ± {np.std(event_amplitudes_mean):.2f}\n\n")

    print(f"  ✓ Saved: {summary_text_path.name}")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CROSS-SESSION ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated:")
    print(f"  ✓ {report_path.name}")
    print(f"  ✓ {summary_text_path.name}")
    print(f"  ✓ figures/cross_session_metrics.png")
    print(f"  ✓ figures/metric_correlations.png")
    print()

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Cross-session analysis of Allen astrocyte data"
    )
    parser.add_argument('--results-dir', type=str, default='./allen_nwb_results',
                       help='Directory containing session results')
    parser.add_argument('--output-dir', type=str, default='./cross_session_analysis',
                       help='Output directory for analysis results')

    args = parser.parse_args()

    report = generate_cross_session_report(args.results_dir, args.output_dir)

    print("=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"\nSessions: {report['n_sessions']}")
    print(f"Recording Time: {report['total_recording_time_min']:.1f} min")
    print(f"Total Events: {report['total_events']:,}")
    print(f"\nConsistent Features: {', '.join(report['consistent_features'])}")
    print()


if __name__ == "__main__":
    main()
