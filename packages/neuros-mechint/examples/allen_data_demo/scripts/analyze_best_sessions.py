#!/usr/bin/env python3
"""
Analyze multi-session validation results and identify best sessions for SAE training.

This script:
1. Loads the multi-session validation JSON results
2. Ranks sessions by orientation selectivity and other metrics
3. Recommends top sessions for SAE validation
4. Generates visualizations comparing sessions
5. Creates a session configuration file for easy use

Usage:
    python scripts/analyze_best_sessions.py --results multi_session_results_FULL.json
    python scripts/analyze_best_sessions.py --results multi_session_results_FULL.json --top-n 10
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_path: str) -> dict:
    """Load multi-session validation results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_sessions(results: dict) -> pd.DataFrame:
    """
    Analyze session results and create ranked DataFrame.

    Returns DataFrame with columns:
    - session_id
    - n_units
    - n_stimuli
    - max_correlation
    - mean_correlation
    - n_significant
    - fraction_selective
    - orientation_max_corr
    - direction_max_corr
    - ori_dir_ratio (orientation/direction selectivity ratio)
    """
    successful = [r for r in results['results'] if r['success']]

    rows = []
    for r in successful:
        ot = r['orientation_tuning']
        dt = r['direction_tuning']

        # Ratio of orientation to direction selectivity
        # Higher ratio means neurons are orientation-selective but not direction-selective
        ori_dir_ratio = ot['max_correlation'] / dt['max_correlation'] if dt['max_correlation'] > 0 else np.inf

        row = {
            'session_id': r['session_id'],
            'n_units': r['n_units'],
            'n_stimuli': r['n_stimuli'],
            'max_correlation': ot['max_correlation'],
            'mean_correlation': ot['mean_correlation'],
            'n_significant': ot['n_significant'],
            'fraction_selective': ot['fraction_selective'],
            'orientation_max_corr': ot['max_correlation'],
            'direction_max_corr': dt['max_correlation'],
            'ori_dir_ratio': ori_dir_ratio
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by fraction_selective (primary) and max_correlation (secondary)
    df = df.sort_values(['fraction_selective', 'max_correlation'], ascending=[False, False])
    df = df.reset_index(drop=True)

    return df


def select_best_sessions(df: pd.DataFrame, top_n: int = 10, min_units: int = 20) -> pd.DataFrame:
    """
    Select best sessions for SAE training based on multiple criteria.

    Criteria:
    1. High fraction of selective units (>30%)
    2. High max correlation (>0.6)
    3. Sufficient number of units (>min_units)
    4. High orientation/direction ratio (orientation-selective, not direction-selective)
    """
    # Filter criteria
    filtered = df[
        (df['fraction_selective'] > 0.3) &
        (df['max_correlation'] > 0.6) &
        (df['n_units'] >= min_units)
    ]

    # Take top N
    top_sessions = filtered.head(top_n)

    return top_sessions


def create_session_config(top_sessions: pd.DataFrame, output_path: str):
    """Create a configuration file for using top sessions."""
    config = {
        'recommended_sessions': {
            'best_overall': int(top_sessions.iloc[0]['session_id']),
            'highest_selectivity': int(top_sessions.iloc[0]['session_id']),
            'most_units': int(top_sessions.loc[top_sessions['n_units'].idxmax(), 'session_id']),
            'top_5': top_sessions.head(5)['session_id'].astype(int).tolist(),
            'top_10': top_sessions.head(10)['session_id'].astype(int).tolist() if len(top_sessions) >= 10 else top_sessions['session_id'].astype(int).tolist(),
        },
        'session_details': []
    }

    for _, row in top_sessions.iterrows():
        config['session_details'].append({
            'session_id': int(row['session_id']),
            'n_units': int(row['n_units']),
            'n_stimuli': int(row['n_stimuli']),
            'fraction_selective': float(row['fraction_selective']),
            'max_correlation': float(row['max_correlation']),
            'mean_correlation': float(row['mean_correlation']),
            'n_significant': int(row['n_significant']),
            'recommendation': 'Excellent' if row['fraction_selective'] > 0.4 else 'Good'
        })

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Session configuration saved to: {output_path}")
    return config


def plot_session_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate visualizations comparing sessions."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Fraction selective vs max correlation scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df['fraction_selective'] * 100,
        df['max_correlation'],
        s=df['n_units'] * 2,
        c=df['mean_correlation'],
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    # Annotate top sessions
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        ax.annotate(
            str(row['session_id']),
            (row['fraction_selective'] * 100, row['max_correlation']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
        )

    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Min correlation threshold')
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Min selectivity threshold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Correlation', rotation=270, labelpad=15)

    ax.set_xlabel('Fraction of Selective Units (%)', fontsize=12)
    ax.set_ylabel('Max Orientation Correlation', fontsize=12)
    ax.set_title('Allen Session Quality for SAE Validation\n(Size = number of units)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'session_quality_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'session_quality_scatter.png'}")
    plt.close()

    # 2. Orientation vs Direction selectivity
    fig, ax = plt.subplots(figsize=(10, 6))

    top_n = min(15, len(df))
    top_df = df.head(top_n)

    x = np.arange(len(top_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, top_df['orientation_max_corr'], width, label='Orientation', alpha=0.8)
    bars2 = ax.bar(x + width/2, top_df['direction_max_corr'], width, label='Direction', alpha=0.8)

    ax.set_xlabel('Session ID', fontsize=12)
    ax.set_ylabel('Max Correlation', fontsize=12)
    ax.set_title(f'Orientation vs Direction Selectivity (Top {top_n} Sessions)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(top_df['session_id'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'orientation_vs_direction.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'orientation_vs_direction.png'}")
    plt.close()

    # 3. Distribution of selective units
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['fraction_selective'] * 100, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=30, color='red', linestyle='--', linewidth=2, label='30% threshold')
    ax.axvline(x=df['fraction_selective'].median() * 100, color='green', linestyle='--', linewidth=2, label=f'Median: {df["fraction_selective"].median()*100:.1f}%')

    ax.set_xlabel('Fraction of Selective Units (%)', fontsize=12)
    ax.set_ylabel('Number of Sessions', fontsize=12)
    ax.set_title('Distribution of Orientation Selectivity Across Sessions', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'selectivity_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'selectivity_distribution.png'}")
    plt.close()


def print_summary_report(df: pd.DataFrame, top_sessions: pd.DataFrame, results: dict):
    """Print comprehensive summary report."""
    print("\n" + "="*80)
    print("MULTI-SESSION ANALYSIS SUMMARY")
    print("="*80)

    # Overall stats
    successful = [r for r in results['results'] if r['success']]
    failed = [r for r in results['results'] if not r['success']]

    print(f"\nTotal sessions processed: {len(results['results'])}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print(f"\nFailed sessions:")
        for r in failed[:5]:  # Show first 5
            print(f"  {r['session_id']}: {r['error']}")

    # Quality metrics
    print(f"\n" + "-"*80)
    print("QUALITY METRICS ACROSS ALL SESSIONS")
    print("-"*80)
    print(f"Fraction selective:")
    print(f"  Best: {df['fraction_selective'].max()*100:.1f}%")
    print(f"  Median: {df['fraction_selective'].median()*100:.1f}%")
    print(f"  Worst: {df['fraction_selective'].min()*100:.1f}%")

    print(f"\nMax correlation:")
    print(f"  Best: {df['max_correlation'].max():.3f}")
    print(f"  Median: {df['max_correlation'].median():.3f}")
    print(f"  Worst: {df['max_correlation'].min():.3f}")

    print(f"\nNumber of units:")
    print(f"  Most: {df['n_units'].max()}")
    print(f"  Median: {df['n_units'].median():.0f}")
    print(f"  Least: {df['n_units'].min()}")

    # Sessions meeting criteria
    good_sessions = df[
        (df['fraction_selective'] > 0.3) &
        (df['max_correlation'] > 0.6)
    ]

    print(f"\n" + "-"*80)
    print(f"SESSIONS MEETING VALIDATION CRITERIA")
    print("-"*80)
    print(f"  >30% selective AND >0.6 max corr: {len(good_sessions)} sessions")
    print(f"  >40% selective: {len(df[df['fraction_selective'] > 0.4])} sessions")
    print(f"  >50% selective: {len(df[df['fraction_selective'] > 0.5])} sessions")

    # Top sessions
    print(f"\n" + "-"*80)
    print(f"TOP {len(top_sessions)} RECOMMENDED SESSIONS FOR SAE VALIDATION")
    print("-"*80)
    print(f"{'Rank':<6} {'Session ID':<12} {'Units':<8} {'% Select':<12} {'Max Corr':<12} {'Mean Corr':<12} {'Recommendation':<15}")
    print("-"*80)

    for idx, (_, row) in enumerate(top_sessions.iterrows(), 1):
        recommendation = "⭐ EXCELLENT" if row['fraction_selective'] > 0.4 else "✓ GOOD"
        print(
            f"{idx:<6} {int(row['session_id']):<12} {int(row['n_units']):<8} "
            f"{row['fraction_selective']*100:<11.1f}% {row['max_correlation']:<12.3f} "
            f"{row['mean_correlation']:<12.3f} {recommendation:<15}"
        )

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze multi-session validation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--results', type=str, required=True, help='Path to multi_session_results JSON file')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top sessions to select')
    parser.add_argument('--min-units', type=int, default=20, help='Minimum units per session')
    parser.add_argument('--output-dir', type=str, default='session_analysis', help='Output directory for plots and config')
    args = parser.parse_args()

    print("="*80)
    print("ANALYZING MULTI-SESSION VALIDATION RESULTS")
    print("="*80)

    # Load results
    print(f"\nLoading results from: {args.results}")
    results = load_results(args.results)

    # Analyze sessions
    print("Analyzing sessions...")
    df = analyze_sessions(results)

    # Select best sessions
    print(f"Selecting top {args.top_n} sessions...")
    top_sessions = select_best_sessions(df, top_n=args.top_n, min_units=args.min_units)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_session_comparison(df, output_dir)

    # Create session configuration
    config_path = output_dir / 'recommended_sessions.json'
    config = create_session_config(top_sessions, str(config_path))

    # Save full analysis to CSV
    csv_path = output_dir / 'all_sessions_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")

    # Print summary report
    print_summary_report(df, top_sessions, results)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - recommended_sessions.json (session configuration)")
    print(f"  - all_sessions_analysis.csv (full analysis)")
    print(f"  - session_quality_scatter.png")
    print(f"  - orientation_vs_direction.png")
    print(f"  - selectivity_distribution.png")

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the recommended sessions in recommended_sessions.json")
    print("2. Use these sessions for SAE training with AllenVisualCodingValidator")
    print("3. Example usage:")
    print(f"\n   from neuros.datasets.allen_datasets import AllenVisualCodingValidator")
    print(f"   validator = AllenVisualCodingValidator(")
    print(f"       session_id={int(top_sessions.iloc[0]['session_id'])},")
    print(f"       cache_dir='allen_validation_cache',")
    print(f"       brain_areas=['VISp'],")
    print(f"       use_all_units=True")
    print(f"   )")
    print(f"   windows = validator.get_neural_windows()")
    print(f"   # Train your SAE on these windows!")
    print("="*80)


if __name__ == "__main__":
    main()
