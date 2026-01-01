#!/usr/bin/env python3
"""
SAE Feature Visualization and Interpretation

Creates comprehensive visualizations to understand what SAE features represent:
1. Tuning curves for top selective features
2. Feature activation heatmaps
3. Feature-neuron connection weights
4. Preferred orientation map
5. Feature clustering analysis

Usage:
    python scripts/visualize_sae_features.py \
        --session-id 754829445 \
        --allen-cache allen_validation_cache \
        --sae-model sae_models/sae_session_754829445.pt \
        --output-dir sae_visualizations
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import torch
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_tuning_curves(
    features: np.ndarray,
    orientations: np.ndarray,
    feature_indices: List[int],
    preferred_oris: np.ndarray,
    output_path: Path
):
    """Plot orientation tuning curves for selected features."""

    unique_oris = np.unique(orientations)
    n_features = len(feature_indices)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat_idx in enumerate(feature_indices[:6]):  # Plot top 6
        ax = axes[i]

        # Compute tuning curve
        tuning_curve = np.array([
            np.mean(features[:, feat_idx][orientations == ori])
            for ori in unique_oris
        ])

        # Normalize
        if np.max(tuning_curve) > 0:
            tuning_curve = tuning_curve / np.max(tuning_curve)

        # Plot
        ax.plot(unique_oris, tuning_curve, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax.axvline(preferred_oris[feat_idx], color='red', linestyle='--', linewidth=2, label=f'Preferred: {preferred_oris[feat_idx]:.1f}°')

        ax.set_xlabel('Orientation (degrees)', fontsize=11)
        ax.set_ylabel('Normalized Response', fontsize=11)
        ax.set_title(f'Feature {feat_idx} Tuning Curve', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved tuning curves: {output_path}")
    plt.close()


def plot_feature_activation_heatmap(
    features: np.ndarray,
    orientations: np.ndarray,
    correlations: np.ndarray,
    output_path: Path
):
    """Plot heatmap of feature activations sorted by orientation."""

    # Sort by orientation
    sort_idx = np.argsort(orientations)
    features_sorted = features[sort_idx]
    orientations_sorted = orientations[sort_idx]

    # Sort features by correlation
    feat_sort_idx = np.argsort(correlations)[::-1][:50]  # Top 50 features
    features_sorted = features_sorted[:, feat_sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [20, 1]})

    # Heatmap
    ax = axes[0]
    im = ax.imshow(features_sorted.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Samples (sorted by orientation)', fontsize=12)
    ax.set_ylabel('SAE Features (sorted by selectivity)', fontsize=12)
    ax.set_title('Feature Activation Heatmap', fontsize=14, fontweight='bold')

    # Add orientation colorbar on side
    ax_ori = axes[1]
    ori_gradient = orientations_sorted.reshape(-1, 1)
    im_ori = ax_ori.imshow(ori_gradient.T, aspect='auto', cmap='hsv', interpolation='nearest')
    ax_ori.set_yticks([])
    ax_ori.set_xticks([])
    ax_ori.set_ylabel('Orientation', fontsize=12, rotation=270, labelpad=15)

    plt.colorbar(im, ax=ax, label='Activation')
    plt.colorbar(im_ori, ax=ax_ori, label='Orientation (°)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved activation heatmap: {output_path}")
    plt.close()


def plot_weight_visualization(
    sae_model,
    raw_data: np.ndarray,
    feature_indices: List[int],
    output_path: Path
):
    """Visualize encoder weights for selected features."""

    encoder_weights = sae_model.encoder.weight.detach().cpu().numpy()

    n_features = len(feature_indices)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat_idx in enumerate(feature_indices[:6]):
        ax = axes[i]

        # Get weights for this feature
        weights = encoder_weights[feat_idx, :]

        # Plot as bar chart
        neuron_indices = np.arange(len(weights))
        colors = ['red' if w > 0 else 'blue' for w in weights]
        ax.bar(neuron_indices, weights, color=colors, alpha=0.6)

        ax.set_xlabel('Neuron Index', fontsize=11)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(f'Feature {feat_idx} Encoder Weights', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved weight visualization: {output_path}")
    plt.close()


def plot_preferred_orientation_map(
    preferred_oris: np.ndarray,
    correlations: np.ndarray,
    output_path: Path
):
    """Create polar plot showing distribution of preferred orientations."""

    # Filter to significant features
    significant_mask = correlations > 0.3
    pref_oris_sig = preferred_oris[significant_mask]
    corrs_sig = correlations[significant_mask]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')

    # Convert to radians (double for 180° period)
    angles = np.deg2rad(pref_oris_sig * 2)

    # Scatter plot with size based on correlation
    scatter = ax.scatter(angles, corrs_sig, s=corrs_sig*500, c=corrs_sig, cmap='hot', alpha=0.6, edgecolors='black', linewidth=0.5)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Preferred Orientation Distribution\n(size and color = correlation strength)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1)

    plt.colorbar(scatter, ax=ax, label='Correlation', pad=0.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved orientation map: {output_path}")
    plt.close()


def plot_feature_clustering(
    features: np.ndarray,
    correlations: np.ndarray,
    preferred_oris: np.ndarray,
    output_path: Path
):
    """Cluster features and visualize in PCA space."""

    # Use only significant features
    significant_mask = correlations > 0.3
    features_sig = features[:, significant_mask]
    pref_oris_sig = preferred_oris[significant_mask]

    if features_sig.shape[1] < 5:
        logger.warning("  Too few significant features for clustering")
        return

    # Transpose to cluster features (not samples)
    feature_vectors = features_sig.T

    # PCA for visualization
    pca = PCA(n_components=2)
    feature_pca = pca.fit_transform(feature_vectors)

    # K-means clustering (try to find ~4 orientation clusters)
    n_clusters = min(4, len(feature_vectors))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feature_vectors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Colored by cluster
    ax = axes[0]
    scatter = ax.scatter(feature_pca[:, 0], feature_pca[:, 1], c=clusters, cmap='tab10', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Feature Clustering (K-means)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cluster')

    # Plot 2: Colored by preferred orientation
    ax = axes[1]
    scatter = ax.scatter(feature_pca[:, 0], feature_pca[:, 1], c=pref_oris_sig, cmap='hsv', s=100, alpha=0.7, edgecolors='black', linewidth=0.5, vmin=0, vmax=180)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Feature Clustering (by preferred orientation)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Preferred Orientation (°)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved feature clustering: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize SAE features')
    parser.add_argument('--session-id', type=int, required=True, help='Session ID')
    parser.add_argument('--allen-cache', type=str, default='allen_validation_cache', help='Allen cache directory')
    parser.add_argument('--sae-model', type=str, required=True, help='Path to trained SAE model')
    parser.add_argument('--output-dir', type=str, default='sae_visualizations', help='Output directory')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("SAE FEATURE VISUALIZATION")
    logger.info("="*80)

    # Load data
    logger.info(f"\nLoading session {args.session_id}...")
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

    # Extract data - filter out null orientations
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

    # Load SAE
    logger.info(f"\nLoading SAE model from: {args.sae_model}")
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    from sae_training_top_sessions import SimpleSAE

    # Infer dimensions from model file
    checkpoint = torch.load(args.sae_model, map_location='cpu')
    input_dim = checkpoint['encoder.weight'].shape[1]
    hidden_dim = checkpoint['encoder.weight'].shape[0]

    sae = SimpleSAE(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(checkpoint)
    sae.eval()

    logger.info(f"  Input dim: {input_dim}, Hidden dim: {hidden_dim}")

    # Normalize and extract features
    raw_mean = raw_data.mean(axis=0, keepdims=True)
    raw_std = raw_data.std(axis=0, keepdims=True) + 1e-8
    raw_norm = (raw_data - raw_mean) / raw_std

    raw_tensor = torch.FloatTensor(raw_norm)
    sae_features = sae.encode(raw_tensor).numpy()

    logger.info(f"  Extracted {sae_features.shape[1]} features from {len(sae_features)} samples")

    # Compute orientation selectivity
    logger.info("\nComputing orientation selectivity...")
    from scipy.stats import pearsonr

    ori_sin = np.sin(np.deg2rad(orientations * 2))
    ori_cos = np.cos(np.deg2rad(orientations * 2))

    correlations = []
    preferred_oris = []

    for feat_idx in range(sae_features.shape[1]):
        feat_response = sae_features[:, feat_idx]

        corr_sin, _ = pearsonr(feat_response, ori_sin)
        corr_cos, _ = pearsonr(feat_response, ori_cos)

        corr = max(abs(corr_sin), abs(corr_cos))
        correlations.append(corr)

        # Preferred orientation
        phase = np.arctan2(np.mean(feat_response * ori_sin), np.mean(feat_response * ori_cos))
        pref_ori = (np.rad2deg(phase) / 2) % 180
        preferred_oris.append(pref_ori)

    correlations = np.array(correlations)
    preferred_oris = np.array(preferred_oris)

    # Select top features for visualization
    top_indices = np.argsort(correlations)[::-1][:6]

    logger.info(f"  Top 6 features by correlation:")
    for i, idx in enumerate(top_indices, 1):
        logger.info(f"    {i}. Feature {idx}: corr={correlations[idx]:.3f}, pref_ori={preferred_oris[idx]:.1f}°")

    # Generate visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("\nGenerating visualizations...")

    # 1. Tuning curves
    plot_tuning_curves(
        sae_features, orientations, top_indices, preferred_oris,
        output_dir / f'tuning_curves_session_{args.session_id}.png'
    )

    # 2. Activation heatmap
    plot_feature_activation_heatmap(
        sae_features, orientations, correlations,
        output_dir / f'activation_heatmap_session_{args.session_id}.png'
    )

    # 3. Weight visualization
    plot_weight_visualization(
        sae, raw_data, top_indices,
        output_dir / f'weights_session_{args.session_id}.png'
    )

    # 4. Preferred orientation map
    plot_preferred_orientation_map(
        preferred_oris, correlations,
        output_dir / f'orientation_map_session_{args.session_id}.png'
    )

    # 5. Feature clustering
    plot_feature_clustering(
        sae_features, correlations, preferred_oris,
        output_dir / f'feature_clustering_session_{args.session_id}.png'
    )

    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nAll visualizations saved to: {output_dir}/")
    logger.info("="*80)


if __name__ == "__main__":
    main()
