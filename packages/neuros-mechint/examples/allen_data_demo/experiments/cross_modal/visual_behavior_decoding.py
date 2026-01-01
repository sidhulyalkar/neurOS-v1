#!/usr/bin/env python3
"""
Visual→Behavior Cross-Modal Decoding
======================================

Implements Experiment 2.1 from ADVANCED_RESEARCH_ROADMAP.md:
- Decode running speed from SAE features vs raw neurons
- Decode pupil size from SAE features
- Test if SAE features capture behaviorally relevant information
- Identify which features are behavior-selective

This script answers:
1. Do SAE features predict motor behavior better than raw neurons?
2. Are orientation-selective features also behavior-selective?
3. Which SAE features encode task-relevant information?

Usage:
    python experiments/cross_modal/visual_behavior_decoding.py \
        --sae-model sae_models/sae_session_754829445.pt \
        --session-id 754829445 \
        --allen-cache allen_validation_cache \
        --output-dir results/cross_modal
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleSAE(nn.Module):
    """Simple SAE matching training script."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.encoder(x))
        x_recon = self.decoder(h)
        return x_recon, h

    def encode(self, x):
        with torch.no_grad():
            h = self.relu(self.encoder(x))
        return h


class CrossModalDecoder:
    """
    Decode behavioral variables from neural representations.

    Compares decoding performance of SAE features vs raw neurons.
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Args:
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state

    def decode_behavior(
        self,
        features: np.ndarray,
        behavior: np.ndarray,
        feature_type: str = "SAE"
    ) -> Dict:
        """
        Decode behavioral variable from neural features using Ridge regression.

        Args:
            features: Neural features [n_samples, n_features]
            behavior: Behavioral variable [n_samples]
            feature_type: "SAE" or "Raw" for labeling

        Returns:
            results: Dict with R², CV scores, model weights
        """
        # Remove NaN/inf
        valid_idx = np.isfinite(behavior) & np.all(np.isfinite(features), axis=1)
        features_clean = features[valid_idx]
        behavior_clean = behavior[valid_idx]

        if len(features_clean) < 10:
            logger.warning(f"  Not enough valid samples: {len(features_clean)}")
            return {
                'r2': 0.0,
                'cv_scores': [],
                'mean_cv_r2': 0.0,
                'std_cv_r2': 0.0,
                'feature_importance': np.zeros(features.shape[1])
            }

        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)

        # Ridge regression with cross-validation for alpha selection
        decoder = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=self.n_folds)
        decoder.fit(features_scaled, behavior_clean)

        # Performance on full data
        predictions = decoder.predict(features_scaled)
        r2 = np.corrcoef(predictions, behavior_clean)[0, 1] ** 2

        # Cross-validated performance
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            decoder, features_scaled, behavior_clean,
            cv=kfold, scoring='r2'
        )

        # Feature importance (absolute coefficients)
        feature_importance = np.abs(decoder.coef_)

        results = {
            'r2': float(r2),
            'cv_scores': cv_scores.tolist(),
            'mean_cv_r2': float(cv_scores.mean()),
            'std_cv_r2': float(cv_scores.std()),
            'feature_importance': feature_importance,
            'best_alpha': float(decoder.alpha_),
            'n_samples': len(features_clean)
        }

        logger.info(f"  {feature_type} decoding: R²={r2:.3f}, CV R²={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

        return results

    def compare_representations(
        self,
        sae_features: np.ndarray,
        raw_neurons: np.ndarray,
        behavior: np.ndarray,
        behavior_name: str
    ) -> Dict:
        """
        Compare SAE features vs raw neurons for behavior decoding.

        Args:
            sae_features: SAE features [n_samples, n_sae_features]
            raw_neurons: Raw neural data [n_samples, n_neurons]
            behavior: Behavioral variable [n_samples]
            behavior_name: Name of behavior (e.g., "running_speed")

        Returns:
            comparison: Dict with both results and improvement metrics
        """
        logger.info(f"\nDecoding {behavior_name}...")

        # Decode from SAE features
        sae_results = self.decode_behavior(sae_features, behavior, feature_type="SAE")

        # Decode from raw neurons
        raw_results = self.decode_behavior(raw_neurons, behavior, feature_type="Raw")

        # Compute improvement
        if raw_results['r2'] > 0:
            improvement = (sae_results['r2'] - raw_results['r2']) / raw_results['r2']
        else:
            improvement = 0.0

        comparison = {
            'behavior': behavior_name,
            'sae': sae_results,
            'raw': raw_results,
            'improvement_pct': float(improvement * 100),
            'sae_better': sae_results['r2'] > raw_results['r2']
        }

        logger.info(f"  SAE improvement: {improvement*100:+.1f}%")

        return comparison


def load_data_with_behavior(
    sae_model_path: str,
    session_id: int,
    cache_dir: str
) -> Tuple[nn.Module, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load SAE, neural data, AND behavioral variables.

    Returns:
        sae: Trained SAE model
        neural_data: Raw neural activity [n_windows, neurons]
        sae_features: SAE features [n_windows, features]
        behavior: Dict with behavioral variables (running_speed, pupil_area, etc.)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "neuros-foundation" / "src"))
    from neuros.datasets.allen_datasets import AllenVisualCodingValidator

    # Load Allen data
    logger.info(f"Loading Allen session {session_id}...")
    validator = AllenVisualCodingValidator(
        session_id=session_id,
        cache_dir=cache_dir,
        brain_areas=['VISp'],
        use_all_units=True
    )

    windows = validator.get_neural_windows(window_length=1.0, stride=0.5, bin_size=0.02)

    # Extract data AND behavior
    valid_data = []
    running_speeds = []
    pupil_areas = []
    lick_times = []

    for w in windows:
        # Skip null orientations
        ori = w.metadata.get('orientation')
        if ori == 'null' or ori is None:
            continue

        # Neural data
        valid_data.append(w.data.mean(axis=0))

        # Behavioral variables (if available)
        running_speeds.append(w.metadata.get('running_speed', np.nan))
        pupil_areas.append(w.metadata.get('pupil_area', np.nan))
        lick_times.append(w.metadata.get('lick_times', 0))

    neural_data = np.array(valid_data)

    # Normalize neural data
    neural_data = (neural_data - neural_data.mean(axis=0)) / (neural_data.std(axis=0) + 1e-8)

    logger.info(f"  Neural data: {neural_data.shape}")

    # Load SAE
    logger.info(f"Loading SAE from {sae_model_path}...")
    input_dim = neural_data.shape[1]

    results_path = Path(sae_model_path).parent / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
            hidden_dim = results[0]['data_stats']['n_sae_features']
    else:
        hidden_dim = 128

    sae = SimpleSAE(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(torch.load(sae_model_path))
    sae.eval()

    # Get SAE features
    logger.info("Computing SAE features...")
    X_tensor = torch.FloatTensor(neural_data)
    sae_features = sae.encode(X_tensor).numpy()

    # Behavioral variables
    behavior = {
        'running_speed': np.array(running_speeds),
        'pupil_area': np.array(pupil_areas),
        'lick_count': np.array(lick_times)
    }

    # Report behavioral data availability
    logger.info("\nBehavioral data availability:")
    for name, values in behavior.items():
        n_valid = np.sum(np.isfinite(values))
        pct = n_valid / len(values) * 100
        logger.info(f"  {name}: {n_valid}/{len(values)} ({pct:.1f}%) valid samples")

    return sae, neural_data, sae_features, behavior


def visualize_decoding_comparison(
    comparisons: List[Dict],
    save_path: Optional[str] = None
):
    """
    Visualize SAE vs Raw neuron decoding performance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    behaviors = [c['behavior'] for c in comparisons]
    sae_r2 = [c['sae']['r2'] for c in comparisons]
    raw_r2 = [c['raw']['r2'] for c in comparisons]
    improvements = [c['improvement_pct'] for c in comparisons]

    # Plot 1: R² comparison
    ax = axes[0, 0]
    x = np.arange(len(behaviors))
    width = 0.35
    ax.bar(x - width/2, sae_r2, width, label='SAE Features', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, raw_r2, width, label='Raw Neurons', color='coral', alpha=0.8)
    ax.set_xlabel('Behavioral Variable')
    ax.set_ylabel('R² Score')
    ax.set_title('Decoding Performance: SAE vs Raw')
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Improvement
    ax = axes[0, 1]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.barh(behaviors, improvements, color=colors, alpha=0.7)
    ax.set_xlabel('Improvement (%)')
    ax.set_title('SAE Improvement over Raw Neurons')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 3: Cross-validation scores
    ax = axes[1, 0]
    for i, comp in enumerate(comparisons):
        cv_scores_sae = comp['sae']['cv_scores']
        cv_scores_raw = comp['raw']['cv_scores']

        ax.scatter([i]*len(cv_scores_sae), cv_scores_sae,
                  color='steelblue', alpha=0.6, s=50, label='SAE' if i == 0 else '')
        ax.scatter([i]*len(cv_scores_raw), cv_scores_raw,
                  color='coral', alpha=0.6, s=50, label='Raw' if i == 0 else '')

    ax.set_xticks(range(len(behaviors)))
    ax.set_xticklabels(behaviors, rotation=45, ha='right')
    ax.set_ylabel('Cross-Validation R²')
    ax.set_title('CV Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary stats
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    Cross-Modal Decoding Summary
    {'='*40}

    Behaviors tested: {len(behaviors)}

    SAE Features:
      Mean R²: {np.mean(sae_r2):.3f}
      Best: {max(sae_r2):.3f} ({behaviors[np.argmax(sae_r2)]})

    Raw Neurons:
      Mean R²: {np.mean(raw_r2):.3f}
      Best: {max(raw_r2):.3f} ({behaviors[np.argmax(raw_r2)]})

    SAE outperforms Raw: {sum(1 for c in comparisons if c['sae_better'])}/{len(comparisons)}
    Mean improvement: {np.mean(improvements):.1f}%
    """

    ax.text(0.1, 0.5, summary_text, fontfamily='monospace',
            fontsize=10, verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved visualization to {save_path}")

    plt.close()


def analyze_feature_behavior_selectivity(
    sae_features: np.ndarray,
    behavior: np.ndarray,
    feature_selectivity: np.ndarray,
    behavior_name: str,
    save_path: Optional[str] = None
) -> Dict:
    """
    Analyze which SAE features are both orientation-selective AND behavior-selective.

    Args:
        sae_features: SAE features [n_samples, n_features]
        behavior: Behavioral variable [n_samples]
        feature_selectivity: Orientation selectivity for each feature [n_features]
        behavior_name: Name of behavior

    Returns:
        analysis: Dict with overlap statistics
    """
    logger.info(f"\nAnalyzing feature-behavior selectivity for {behavior_name}...")

    # Decode to get feature importance
    decoder = CrossModalDecoder()
    results = decoder.decode_behavior(sae_features, behavior, feature_type="SAE")
    feature_importance = results['feature_importance']

    # Threshold for "selective"
    ori_selective = feature_selectivity > 0.3
    behavior_selective = feature_importance > np.median(feature_importance)

    # Overlap
    both_selective = ori_selective & behavior_selective
    n_both = np.sum(both_selective)
    n_ori_only = np.sum(ori_selective & ~behavior_selective)
    n_behavior_only = np.sum(~ori_selective & behavior_selective)
    n_neither = np.sum(~ori_selective & ~behavior_selective)

    logger.info(f"  Both selective: {n_both}")
    logger.info(f"  Orientation only: {n_ori_only}")
    logger.info(f"  Behavior only: {n_behavior_only}")
    logger.info(f"  Neither: {n_neither}")

    # Visualize
    if save_path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Scatter
        ax = axes[0]
        scatter = ax.scatter(
            feature_selectivity,
            feature_importance,
            c=both_selective,
            cmap='RdYlGn',
            alpha=0.6,
            s=50
        )
        ax.axhline(np.median(feature_importance), color='red', linestyle='--', alpha=0.3)
        ax.axvline(0.3, color='blue', linestyle='--', alpha=0.3)
        ax.set_xlabel('Orientation Selectivity')
        ax.set_ylabel(f'{behavior_name} Importance')
        ax.set_title('Feature Selectivity: Orientation vs Behavior')
        ax.grid(True, alpha=0.3)

        # Plot 2: Venn diagram-style counts
        ax = axes[1]
        categories = ['Both', 'Ori Only', 'Behavior Only', 'Neither']
        counts = [n_both, n_ori_only, n_behavior_only, n_neither]
        colors = ['green', 'blue', 'orange', 'gray']

        ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Selectivity Categories')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved selectivity analysis to {save_path}")
        plt.close()

    analysis = {
        'behavior': behavior_name,
        'n_both_selective': int(n_both),
        'n_orientation_only': int(n_ori_only),
        'n_behavior_only': int(n_behavior_only),
        'n_neither': int(n_neither),
        'overlap_ratio': float(n_both / (n_both + n_ori_only + n_behavior_only + 1e-8))
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description='Visual→Behavior Cross-Modal Decoding'
    )
    parser.add_argument('--sae-model', type=str, required=True,
                       help='Path to trained SAE model')
    parser.add_argument('--session-id', type=int, required=True,
                       help='Allen session ID')
    parser.add_argument('--allen-cache', type=str, required=True,
                       help='Allen cache directory')
    parser.add_argument('--output-dir', type=str, default='results/cross_modal',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Cross-Modal Decoding Analysis")
    logger.info("="*60)

    # Load data with behavior
    sae, neural_data, sae_features, behavior = load_data_with_behavior(
        args.sae_model, args.session_id, args.allen_cache
    )

    # Compute feature selectivity
    logger.info("\nComputing SAE feature orientation selectivity...")
    # Load orientations
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "neuros-foundation" / "src"))
    from neuros.datasets.allen_datasets import AllenVisualCodingValidator

    validator = AllenVisualCodingValidator(
        session_id=args.session_id,
        cache_dir=args.allen_cache,
        use_all_units=True
    )
    windows = validator.get_neural_windows(window_length=1.0, stride=0.5, bin_size=0.02)

    orientations = []
    for w in windows:
        ori = w.metadata.get('orientation')
        if ori != 'null' and ori is not None:
            try:
                orientations.append(float(ori) % 180)
            except:
                pass

    orientations = np.array(orientations)

    # Compute selectivity
    from scipy.stats import pearsonr
    ori_sin = np.sin(np.deg2rad(orientations * 2))
    ori_cos = np.cos(np.deg2rad(orientations * 2))

    feature_selectivity = []
    for feat_idx in range(sae_features.shape[1]):
        feat_response = sae_features[:, feat_idx]
        corr_sin, _ = pearsonr(feat_response, ori_sin)
        corr_cos, _ = pearsonr(feat_response, ori_cos)
        corr = max(abs(corr_sin), abs(corr_cos))
        feature_selectivity.append(corr)

    feature_selectivity = np.array(feature_selectivity)

    # Cross-modal decoding
    decoder = CrossModalDecoder(n_folds=5)
    comparisons = []

    for behavior_name, behavior_values in behavior.items():
        comparison = decoder.compare_representations(
            sae_features, neural_data, behavior_values, behavior_name
        )
        comparisons.append(comparison)

    # Visualize comparisons
    vis_path = output_dir / 'decoding_comparison.png'
    visualize_decoding_comparison(comparisons, save_path=vis_path)

    # Analyze feature selectivity overlap
    selectivity_analyses = []

    for behavior_name, behavior_values in behavior.items():
        if np.sum(np.isfinite(behavior_values)) > 10:
            analysis_path = output_dir / f'selectivity_{behavior_name}.png'
            analysis = analyze_feature_behavior_selectivity(
                sae_features, behavior_values, feature_selectivity,
                behavior_name, save_path=analysis_path
            )
            selectivity_analyses.append(analysis)

    # Save results
    results = {
        'session_id': args.session_id,
        'decoding_comparisons': comparisons,
        'selectivity_analyses': selectivity_analyses,
        'summary': {
            'mean_sae_r2': float(np.mean([c['sae']['r2'] for c in comparisons])),
            'mean_raw_r2': float(np.mean([c['raw']['r2'] for c in comparisons])),
            'mean_improvement_pct': float(np.mean([c['improvement_pct'] for c in comparisons])),
            'sae_wins': sum(1 for c in comparisons if c['sae_better'])
        }
    }

    results_path = output_dir / f'cross_modal_results_session_{args.session_id}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Saved results to {results_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"\nSAE features: Mean R² = {results['summary']['mean_sae_r2']:.3f}")
    logger.info(f"Raw neurons: Mean R² = {results['summary']['mean_raw_r2']:.3f}")
    logger.info(f"Improvement: {results['summary']['mean_improvement_pct']:+.1f}%")
    logger.info(f"SAE better in {results['summary']['sae_wins']}/{len(comparisons)} cases")

    logger.info("\n✓ Cross-Modal Analysis Complete!")


if __name__ == '__main__':
    main()
