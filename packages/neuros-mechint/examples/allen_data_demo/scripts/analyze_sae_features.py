#!/usr/bin/env python3
"""
Comprehensive SAE Feature Analysis

This script performs deep analysis of trained SAE features:
1. Compares SAE features vs raw neurons for orientation selectivity
2. Analyzes feature sparsity and activation patterns
3. Identifies feature types (orientation, spatial frequency, etc.)
4. Generates publication-quality comparison figures

Usage:
    python scripts/analyze_sae_features.py \
        --sae-results sae_models/training_results.json \
        --session-id 754829445 \
        --allen-cache allen_validation_cache
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sae_model(model_path: str, input_dim: int, hidden_dim: int):
    """Load trained SAE model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    from sae_training_top_sessions import SimpleSAE

    sae = SimpleSAE(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(torch.load(model_path))
    sae.eval()

    return sae


def analyze_orientation_selectivity(
    features: np.ndarray,
    orientations: np.ndarray,
    feature_type: str = "SAE"
) -> Dict:
    """
    Analyze orientation selectivity of features (SAE or raw neurons).

    Returns detailed statistics including:
    - Circular correlation for each feature
    - Preferred orientations
    - Tuning curve widths
    - Feature clustering by preferred orientation
    """
    from scipy.stats import pearsonr, circmean

    # Convert orientations to radians for circular statistics
    ori_sin = np.sin(np.deg2rad(orientations * 2))
    ori_cos = np.cos(np.deg2rad(orientations * 2))

    n_features = features.shape[1]
    correlations = []
    p_values = []
    preferred_oris = []
    tuning_widths = []

    for feat_idx in range(n_features):
        feat_response = features[:, feat_idx]

        # Circular correlation
        corr_sin, p_sin = pearsonr(feat_response, ori_sin)
        corr_cos, p_cos = pearsonr(feat_response, ori_cos)

        # Take max correlation
        if abs(corr_sin) > abs(corr_cos):
            corr = abs(corr_sin)
            p_val = p_sin
            phase = np.arctan2(np.mean(feat_response * ori_sin), np.mean(feat_response * ori_cos))
        else:
            corr = abs(corr_cos)
            p_val = p_cos
            phase = np.arctan2(np.mean(feat_response * ori_sin), np.mean(feat_response * ori_cos))

        correlations.append(corr)
        p_values.append(p_val)

        # Preferred orientation (convert back from doubled angle)
        pref_ori = (np.rad2deg(phase) / 2) % 180
        preferred_oris.append(pref_ori)

        # Compute tuning width (half-width at half-max)
        unique_oris = np.unique(orientations)
        tuning_curve = np.array([np.mean(feat_response[orientations == ori]) for ori in unique_oris])

        if np.max(tuning_curve) > 0:
            half_max = np.max(tuning_curve) / 2
            above_half_max = unique_oris[tuning_curve >= half_max]
            if len(above_half_max) > 0:
                width = np.max(above_half_max) - np.min(above_half_max)
            else:
                width = 0
        else:
            width = 0

        tuning_widths.append(width)

    correlations = np.array(correlations)
    p_values = np.array(p_values)
    preferred_oris = np.array(preferred_oris)
    tuning_widths = np.array(tuning_widths)

    # Statistics
    significant_mask = (correlations > 0.3) & (p_values < 0.05)
    n_significant = np.sum(significant_mask)

    results = {
        'feature_type': feature_type,
        'n_features': n_features,
        'correlations': correlations,
        'p_values': p_values,
        'preferred_orientations': preferred_oris,
        'tuning_widths': tuning_widths,
        'max_correlation': float(np.max(correlations)),
        'mean_correlation': float(np.mean(correlations)),
        'median_correlation': float(np.median(correlations)),
        'n_significant': int(n_significant),
        'fraction_selective': float(n_significant / n_features),
        'significant_mask': significant_mask
    }

    logger.info(f"{feature_type} Features:")
    logger.info(f"  Max correlation: {results['max_correlation']:.3f}")
    logger.info(f"  Mean correlation: {results['mean_correlation']:.3f}")
    logger.info(f"  Selective features: {n_significant}/{n_features} ({results['fraction_selective']*100:.1f}%)")

    return results


def analyze_feature_sparsity(features: np.ndarray) -> Dict:
    """
    Analyze sparsity of feature activations.

    Returns:
    - L0 sparsity (% of zeros)
    - L1 sparsity (mean absolute activation)
    - Lifetime sparsity (% of samples each feature is active)
    - Population sparsity (% of features active per sample)
    """
    # L0 sparsity (fraction of activations that are zero)
    l0_sparsity = np.mean(features == 0)

    # L1 norm
    l1_norm = np.mean(np.abs(features))

    # Lifetime sparsity (per feature)
    lifetime_sparsity = np.mean(features > 0, axis=0)

    # Population sparsity (per sample)
    population_sparsity = np.mean(features > 0, axis=1)

    results = {
        'l0_sparsity': float(l0_sparsity),
        'l1_norm': float(l1_norm),
        'mean_lifetime_sparsity': float(np.mean(lifetime_sparsity)),
        'mean_population_sparsity': float(np.mean(population_sparsity)),
        'lifetime_sparsity_per_feature': lifetime_sparsity,
        'population_sparsity_per_sample': population_sparsity
    }

    logger.info(f"Feature Sparsity:")
    logger.info(f"  L0 (% zeros): {results['l0_sparsity']*100:.1f}%")
    logger.info(f"  Mean lifetime sparsity: {results['mean_lifetime_sparsity']*100:.1f}%")
    logger.info(f"  Mean population sparsity: {results['mean_population_sparsity']*100:.1f}%")

    return results


def plot_comparison_figures(
    raw_results: Dict,
    sae_results: Dict,
    sparsity_results: Dict,
    output_dir: Path
):
    """Generate comprehensive comparison figures."""

    output_dir.mkdir(exist_ok=True, parents=True)

    # Figure 1: Correlation distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 30)
    ax.hist(raw_results['correlations'], bins=bins, alpha=0.6, label='Raw Neurons', color='blue', edgecolor='black')
    ax.hist(sae_results['correlations'], bins=bins, alpha=0.6, label='SAE Features', color='orange', edgecolor='black')
    ax.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Threshold (0.3)')
    ax.set_xlabel('Orientation Correlation', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Orientation Selectivity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative distribution
    ax = axes[1]
    sorted_raw = np.sort(raw_results['correlations'])[::-1]
    sorted_sae = np.sort(sae_results['correlations'])[::-1]
    ax.plot(np.arange(len(sorted_raw)), sorted_raw, label='Raw Neurons', linewidth=2)
    ax.plot(np.arange(len(sorted_sae)), sorted_sae, label='SAE Features', linewidth=2)
    ax.axhline(0.3, color='red', linestyle='--', linewidth=2, label='Threshold (0.3)')
    ax.set_xlabel('Feature Rank', fontsize=12)
    ax.set_ylabel('Orientation Correlation', fontsize=12)
    ax.set_title('Sorted Feature Selectivity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved: {output_dir / 'correlation_comparison.png'}")
    plt.close()

    # Figure 2: Preferred orientation distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': 'polar'})

    # Raw neurons
    ax = axes[0]
    raw_selective = raw_results['preferred_orientations'][raw_results['significant_mask']]
    if len(raw_selective) > 0:
        raw_selective_rad = np.deg2rad(raw_selective * 2)  # Double for 180° period
        ax.hist(raw_selective_rad, bins=16, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title(f'Raw Neurons (n={raw_results["n_significant"]})', fontsize=14, pad=20)

    # SAE features
    ax = axes[1]
    sae_selective = sae_results['preferred_orientations'][sae_results['significant_mask']]
    if len(sae_selective) > 0:
        sae_selective_rad = np.deg2rad(sae_selective * 2)
        ax.hist(sae_selective_rad, bins=16, alpha=0.7, color='orange', edgecolor='black')
    ax.set_title(f'SAE Features (n={sae_results["n_significant"]})', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'preferred_orientation_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved: {output_dir / 'preferred_orientation_distribution.png'}")
    plt.close()

    # Figure 3: Sparsity analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Lifetime sparsity distribution
    ax = axes[0, 0]
    ax.hist(sparsity_results['lifetime_sparsity_per_feature'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(sparsity_results['mean_lifetime_sparsity'], color='red', linestyle='--', linewidth=2, label=f'Mean: {sparsity_results["mean_lifetime_sparsity"]:.2f}')
    ax.set_xlabel('Lifetime Sparsity (% active)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Feature Lifetime Sparsity Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Population sparsity distribution
    ax = axes[0, 1]
    ax.hist(sparsity_results['population_sparsity_per_sample'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(sparsity_results['mean_population_sparsity'], color='red', linestyle='--', linewidth=2, label=f'Mean: {sparsity_results["mean_population_sparsity"]:.2f}')
    ax.set_xlabel('Population Sparsity (% active)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Sample Population Sparsity Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation vs lifetime sparsity
    ax = axes[1, 0]
    scatter = ax.scatter(
        sparsity_results['lifetime_sparsity_per_feature'],
        sae_results['correlations'],
        c=sae_results['significant_mask'],
        cmap='RdYlGn',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel('Lifetime Sparsity', fontsize=12)
    ax.set_ylabel('Orientation Correlation', fontsize=12)
    ax.set_title('Selectivity vs Sparsity', fontsize=14)
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Selective')

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
SAE Feature Analysis Summary

Orientation Selectivity:
  Raw Neurons: {raw_results['fraction_selective']*100:.1f}% selective
  SAE Features: {sae_results['fraction_selective']*100:.1f}% selective

  Raw max corr: {raw_results['max_correlation']:.3f}
  SAE max corr: {sae_results['max_correlation']:.3f}

Sparsity:
  L0 (% zeros): {sparsity_results['l0_sparsity']*100:.1f}%
  L1 norm: {sparsity_results['l1_norm']:.3f}

  Lifetime sparsity: {sparsity_results['mean_lifetime_sparsity']*100:.1f}%
  Population sparsity: {sparsity_results['mean_population_sparsity']*100:.1f}%

Interpretation:
  {'✅ SAE improves selectivity' if sae_results['fraction_selective'] > raw_results['fraction_selective'] else '⚠️ SAE reduces selectivity'}
  {'✅ Good sparsity (<50%)' if sparsity_results['mean_lifetime_sparsity'] < 0.5 else '⚠️ Low sparsity (>50%)'}
"""

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'sparsity_analysis.png', dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved: {output_dir / 'sparsity_analysis.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze SAE features')
    parser.add_argument('--sae-results', type=str, required=True, help='Path to training_results.json')
    parser.add_argument('--session-id', type=int, required=True, help='Session ID to analyze')
    parser.add_argument('--allen-cache', type=str, default='allen_validation_cache', help='Allen cache directory')
    parser.add_argument('--output-dir', type=str, default='sae_analysis', help='Output directory')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("SAE FEATURE ANALYSIS")
    logger.info("="*80)

    # Load SAE results
    logger.info(f"\nLoading SAE results from: {args.sae_results}")
    with open(args.sae_results, 'r') as f:
        all_results = json.load(f)

    # Find results for this session
    session_results = None
    for r in all_results:
        if r['session_id'] == args.session_id:
            session_results = r
            break

    if session_results is None:
        logger.error(f"No results found for session {args.session_id}")
        return

    logger.info(f"Found results for session {args.session_id}")

    # Load neural data
    logger.info("\nLoading neural data...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "neuros-foundation" / "src"))
    from neuros.datasets.allen_datasets import AllenVisualCodingValidator

    validator = AllenVisualCodingValidator(
        session_id=args.session_id,
        cache_dir=args.allen_cache,
        brain_areas=['VISp'],
        use_all_units=True
    )

    windows = validator.get_neural_windows()
    labels = validator.get_task_labels()

    # Extract raw neural data - filter out null orientations
    valid_data = []
    valid_orientations = []

    for w in windows:
        ori = w.metadata['orientation']

        # Skip null/NaN orientations
        if ori == 'null' or ori is None:
            continue

        # Try to convert to float
        try:
            ori_float = float(ori)
            ori_180 = ori_float % 180  # Convert to 0-180°

            valid_data.append(w.data.mean(axis=0))
            valid_orientations.append(ori_180)
        except (ValueError, TypeError):
            continue

    raw_data = np.array(valid_data)
    orientations = np.array(valid_orientations)

    n_skipped = len(windows) - len(valid_data)
    if n_skipped > 0:
        logger.info(f"  Skipped {n_skipped} windows with invalid orientations")

    logger.info(f"  ✓ Loaded {len(valid_data)} windows with {raw_data.shape[1]} neurons")

    # Load trained SAE
    logger.info("\nLoading trained SAE...")
    sae_path = Path('sae_models') / f"sae_session_{args.session_id}.pt"

    if not sae_path.exists():
        logger.error(f"SAE model not found: {sae_path}")
        return

    input_dim = raw_data.shape[1]
    hidden_dim = session_results['data_stats']['n_sae_features']

    sae = load_sae_model(str(sae_path), input_dim, hidden_dim)

    # Normalize data (same as training)
    raw_mean = raw_data.mean(axis=0, keepdims=True)
    raw_std = raw_data.std(axis=0, keepdims=True) + 1e-8
    raw_norm = (raw_data - raw_mean) / raw_std

    # Extract SAE features
    raw_tensor = torch.FloatTensor(raw_norm)
    sae_features = sae.encode(raw_tensor).numpy()

    logger.info(f"  ✓ Extracted {sae_features.shape[1]} SAE features")

    # Analyze raw neurons
    logger.info("\n" + "-"*80)
    logger.info("Analyzing Raw Neurons...")
    logger.info("-"*80)
    raw_results = analyze_orientation_selectivity(raw_data, orientations, "Raw Neurons")

    # Analyze SAE features
    logger.info("\n" + "-"*80)
    logger.info("Analyzing SAE Features...")
    logger.info("-"*80)
    sae_results = analyze_orientation_selectivity(sae_features, orientations, "SAE Features")

    # Analyze sparsity
    logger.info("\n" + "-"*80)
    logger.info("Analyzing Feature Sparsity...")
    logger.info("-"*80)
    sparsity_results = analyze_feature_sparsity(sae_features)

    # Generate figures
    logger.info("\n" + "-"*80)
    logger.info("Generating Comparison Figures...")
    logger.info("-"*80)
    output_dir = Path(args.output_dir)
    plot_comparison_figures(raw_results, sae_results, sparsity_results, output_dir)

    # Save detailed results
    detailed_results = {
        'session_id': args.session_id,
        'raw_neurons': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in raw_results.items()},
        'sae_features': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sae_results.items()},
        'sparsity': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sparsity_results.items()}
    }

    results_path = output_dir / f'detailed_analysis_session_{args.session_id}.json'
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    logger.info(f"  ✓ Saved: {results_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nSession {args.session_id} Summary:")
    logger.info(f"  Raw Neurons: {raw_results['fraction_selective']*100:.1f}% selective")
    logger.info(f"  SAE Features: {sae_results['fraction_selective']*100:.1f}% selective")
    logger.info(f"  Sparsity: {sparsity_results['mean_lifetime_sparsity']*100:.1f}% active")
    logger.info(f"\nOutputs saved to: {output_dir}/")
    logger.info("="*80)


if __name__ == "__main__":
    main()
