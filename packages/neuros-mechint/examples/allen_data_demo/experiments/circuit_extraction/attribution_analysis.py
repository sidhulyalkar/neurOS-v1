#!/usr/bin/env python3
"""
Feature Attribution Analysis for SAE Circuit Extraction
========================================================

Implements Experiment 1.1 from ADVANCED_RESEARCH_ROADMAP.md:
- For each SAE feature, identify which neurons contribute
- Compute Integrated Gradients attribution
- Extract interpretable circuits
- Visualize neuron-to-feature connectivity

This script answers:
1. Which neurons contribute to each SAE feature?
2. How do neurons combine to create orientation selectivity?
3. What computational motifs emerge (feedforward, recurrent, lateral)?

Usage:
    python experiments/circuit_extraction/attribution_analysis.py \
        --sae-model sae_models/sae_session_754829445.pt \
        --session-id 754829445 \
        --allen-cache allen_validation_cache \
        --output-dir results/circuits \
        --top-features 20
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
from tqdm import tqdm
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
        """Get SAE features."""
        with torch.no_grad():
            h = self.relu(self.encoder(x))
        return h


class FeatureAttributor:
    """
    Compute attribution of neurons to SAE features using Integrated Gradients.

    This identifies which input neurons causally contribute to each SAE feature.
    """

    def __init__(self, sae_model: nn.Module, device: str = 'cpu'):
        """
        Args:
            sae_model: Trained SAE
            device: Computation device
        """
        self.sae = sae_model
        self.device = device
        self.sae.to(device)
        self.sae.eval()

    def compute_attribution(
        self,
        neural_data: torch.Tensor,
        feature_idx: int,
        num_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attribution for a specific SAE feature.

        Args:
            neural_data: Input neural activity [batch, neurons]
            feature_idx: Index of SAE feature to attribute
            num_steps: Number of integration steps
            baseline: Baseline input (default: zeros)

        Returns:
            attributions: [neurons] tensor showing contribution of each neuron
        """
        neural_data = neural_data.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(neural_data)
        else:
            baseline = baseline.to(self.device)

        # Create path from baseline to input
        alphas = torch.linspace(0, 1, num_steps, device=self.device)

        # Compute gradients along path
        gradients = []

        for alpha in alphas:
            # Interpolated input
            interpolated = baseline + alpha * (neural_data - baseline)
            interpolated.requires_grad_(True)

            # Forward pass
            _, features = self.sae(interpolated)

            # Target: specific feature activation
            target = features[:, feature_idx].sum()

            # Backward pass
            target.backward()

            # Store gradient
            gradients.append(interpolated.grad.detach().clone())

            # Clear gradients
            interpolated.grad.zero_()

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated gradients: (input - baseline) * avg_gradients
        attributions = (neural_data - baseline) * avg_gradients

        # Average over batch
        attributions = attributions.mean(dim=0).abs()

        return attributions

    def compute_all_attributions(
        self,
        neural_data: torch.Tensor,
        n_features: int,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute attributions for all SAE features.

        Args:
            neural_data: Input neural data [batch, neurons]
            n_features: Number of SAE features
            num_steps: Integration steps

        Returns:
            attribution_matrix: [n_features, neurons] showing contribution of each neuron to each feature
        """
        logger.info(f"Computing attributions for {n_features} features...")

        attributions = []

        for feature_idx in tqdm(range(n_features), desc="Computing attributions"):
            attr = self.compute_attribution(
                neural_data, feature_idx, num_steps=num_steps
            )
            attributions.append(attr.cpu().numpy())

        attribution_matrix = np.array(attributions)

        return attribution_matrix


class CircuitExtractor:
    """
    Extract computational circuits from attribution scores.

    Identifies groups of neurons that work together to create SAE features.
    """

    def __init__(self, attribution_matrix: np.ndarray, threshold: float = 0.1):
        """
        Args:
            attribution_matrix: [n_features, neurons] attribution scores
            threshold: Minimum attribution to include neuron in circuit
        """
        self.attributions = attribution_matrix
        self.threshold = threshold

    def extract_circuit(self, feature_idx: int, top_k: int = 10) -> Dict:
        """
        Extract circuit for a specific SAE feature.

        Args:
            feature_idx: SAE feature index
            top_k: Number of top contributing neurons

        Returns:
            circuit: Dict with neuron indices, weights, and statistics
        """
        # Get attribution scores for this feature
        scores = self.attributions[feature_idx]

        # Normalize to 0-1
        scores_norm = scores / (scores.max() + 1e-8)

        # Find neurons above threshold
        above_threshold = np.where(scores_norm > self.threshold)[0]

        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]

        circuit = {
            'feature_idx': feature_idx,
            'top_neurons': top_indices.tolist(),
            'attribution_scores': scores[top_indices].tolist(),
            'normalized_scores': scores_norm[top_indices].tolist(),
            'n_neurons_above_threshold': len(above_threshold),
            'circuit_sparsity': 1.0 - (len(above_threshold) / len(scores))
        }

        return circuit

    def analyze_circuit_motifs(self, circuits: List[Dict]) -> Dict:
        """
        Analyze circuit patterns across multiple features.

        Identifies:
        - Neuron reuse (neurons contributing to multiple features)
        - Circuit sparsity patterns
        - Hierarchical structure

        Args:
            circuits: List of circuit dicts

        Returns:
            motif_analysis: Dict with motif statistics
        """
        # Track neuron participation
        neuron_participation = {}

        for circuit in circuits:
            for neuron_idx in circuit['top_neurons']:
                if neuron_idx not in neuron_participation:
                    neuron_participation[neuron_idx] = []
                neuron_participation[neuron_idx].append(circuit['feature_idx'])

        # Find highly reused neurons
        reused_neurons = {
            neuron: features
            for neuron, features in neuron_participation.items()
            if len(features) > 1
        }

        # Sparsity statistics
        sparsities = [c['circuit_sparsity'] for c in circuits]

        motif_analysis = {
            'total_neurons_used': len(neuron_participation),
            'reused_neurons': len(reused_neurons),
            'max_reuse': max([len(f) for f in neuron_participation.values()]),
            'mean_sparsity': np.mean(sparsities),
            'std_sparsity': np.std(sparsities),
            'neuron_participation': neuron_participation,
            'highly_reused': {
                n: features for n, features in reused_neurons.items()
                if len(features) >= 3
            }
        }

        return motif_analysis


def load_sae_and_data(
    sae_model_path: str,
    session_id: int,
    cache_dir: str
) -> Tuple[nn.Module, np.ndarray, np.ndarray, Dict]:
    """
    Load trained SAE and Allen session data.

    Returns:
        sae: Trained SAE model
        neural_data: Raw neural activity [n_windows, neurons]
        orientations: Orientation labels [n_windows]
        metadata: Session metadata
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

    windows = validator.get_neural_windows(
        window_length=1.0,
        stride=0.5,
        bin_size=0.02
    )

    # Extract data
    valid_data = []
    valid_orientations = []

    for w in windows:
        ori = w.metadata['orientation']
        if ori == 'null' or ori is None:
            continue

        try:
            ori_float = float(ori)
            ori_180 = ori_float % 180
            valid_data.append(w.data.mean(axis=0))
            valid_orientations.append(ori_180)
        except (ValueError, TypeError):
            continue

    neural_data = np.array(valid_data)
    orientations = np.array(valid_orientations)

    # Normalize
    neural_data = (neural_data - neural_data.mean(axis=0)) / (neural_data.std(axis=0) + 1e-8)

    logger.info(f"  Neural data: {neural_data.shape}")
    logger.info(f"  Orientations: {len(np.unique(orientations))} unique")

    # Load SAE
    logger.info(f"Loading SAE from {sae_model_path}...")
    input_dim = neural_data.shape[1]

    # Infer hidden_dim from training results
    results_path = Path(sae_model_path).parent / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
            hidden_dim = results[0]['data_stats']['n_sae_features']
    else:
        hidden_dim = 128  # Default

    sae = SimpleSAE(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(torch.load(sae_model_path))
    sae.eval()

    logger.info(f"  SAE: {input_dim} → {hidden_dim}")

    metadata = {
        'n_neurons': input_dim,
        'n_features': hidden_dim,
        'n_windows': len(neural_data)
    }

    return sae, neural_data, orientations, metadata


def compute_feature_selectivity(
    sae: nn.Module,
    neural_data: np.ndarray,
    orientations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orientation selectivity for each SAE feature.

    Returns:
        correlations: [n_features] orientation selectivity
        preferred_orientations: [n_features] preferred orientation in degrees
    """
    # Get SAE features
    X_tensor = torch.FloatTensor(neural_data)
    features = sae.encode(X_tensor).numpy()

    # Compute selectivity
    ori_sin = np.sin(np.deg2rad(orientations * 2))
    ori_cos = np.cos(np.deg2rad(orientations * 2))

    n_features = features.shape[1]
    correlations = []
    preferred_oris = []

    for feat_idx in range(n_features):
        feat_response = features[:, feat_idx]

        # Circular correlation
        corr_sin, _ = pearsonr(feat_response, ori_sin)
        corr_cos, _ = pearsonr(feat_response, ori_cos)

        corr = max(abs(corr_sin), abs(corr_cos))
        correlations.append(corr)

        # Preferred orientation
        phase = np.arctan2(
            np.mean(feat_response * ori_sin),
            np.mean(feat_response * ori_cos)
        )
        pref_ori = (np.rad2deg(phase) / 2) % 180
        preferred_oris.append(pref_ori)

    return np.array(correlations), np.array(preferred_oris)


def visualize_circuit(
    circuit: Dict,
    attribution_matrix: np.ndarray,
    neuron_selectivity: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize circuit for a single SAE feature.

    Shows:
    - Neuron contribution (bar plot)
    - Connection weights (heatmap)
    - Neuron properties (color-coded by selectivity)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    feature_idx = circuit['feature_idx']
    top_neurons = circuit['top_neurons']
    scores = circuit['attribution_scores']

    # Plot 1: Attribution scores
    ax = axes[0]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_neurons)))
    ax.barh(range(len(top_neurons)), scores, color=colors)
    ax.set_yticks(range(len(top_neurons)))
    ax.set_yticklabels([f'N{n}' for n in top_neurons])
    ax.set_xlabel('Attribution Score')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Feature {feature_idx}: Top Contributing Neurons')
    ax.invert_yaxis()

    # Plot 2: Connection weights
    ax = axes[1]
    weights = attribution_matrix[feature_idx, top_neurons].reshape(-1, 1)
    sns.heatmap(
        weights,
        cmap='RdBu_r',
        center=0,
        yticklabels=[f'N{n}' for n in top_neurons],
        xticklabels=['Feature'],
        cbar_kws={'label': 'Attribution'},
        ax=ax
    )
    ax.set_title('Attribution Heatmap')

    # Plot 3: Neuron properties
    ax = axes[2]
    if neuron_selectivity is not None:
        neuron_sel = neuron_selectivity[top_neurons]
        scatter = ax.scatter(
            range(len(top_neurons)),
            scores,
            c=neuron_sel,
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        ax.set_xlabel('Neuron Rank')
        ax.set_ylabel('Attribution Score')
        ax.set_title('Attribution vs Neuron Selectivity')
        plt.colorbar(scatter, ax=ax, label='Neuron Selectivity')
    else:
        ax.scatter(range(len(top_neurons)), scores, s=100, alpha=0.7)
        ax.set_xlabel('Neuron Rank')
        ax.set_ylabel('Attribution Score')
        ax.set_title('Attribution by Rank')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved circuit visualization to {save_path}")

    plt.close()


def visualize_motif_analysis(
    motif_analysis: Dict,
    save_path: Optional[str] = None
):
    """
    Visualize circuit motif patterns.

    Shows:
    - Neuron reuse statistics
    - Circuit sparsity distribution
    - Highly reused neurons
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Neuron participation histogram
    ax = axes[0, 0]
    participation_counts = [
        len(features)
        for features in motif_analysis['neuron_participation'].values()
    ]
    ax.hist(participation_counts, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Features per Neuron')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Neuron Reuse Distribution')
    ax.axvline(np.mean(participation_counts), color='red', linestyle='--',
               label=f'Mean: {np.mean(participation_counts):.1f}')
    ax.legend()

    # Plot 2: Top reused neurons
    ax = axes[0, 1]
    if motif_analysis['highly_reused']:
        reused = motif_analysis['highly_reused']
        top_10 = sorted(reused.items(), key=lambda x: len(x[1]), reverse=True)[:10]

        neurons = [f"N{n}" for n, _ in top_10]
        counts = [len(features) for _, features in top_10]

        ax.barh(range(len(neurons)), counts, color='coral')
        ax.set_yticks(range(len(neurons)))
        ax.set_yticklabels(neurons)
        ax.set_xlabel('Number of Features')
        ax.set_title('Most Reused Neurons')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'No highly reused neurons',
                ha='center', va='center', transform=ax.transAxes)

    # Plot 3: Summary statistics
    ax = axes[1, 0]
    ax.axis('off')

    stats_text = f"""
    Circuit Motif Analysis Summary
    {'='*40}

    Total neurons used: {motif_analysis['total_neurons_used']}
    Reused neurons: {motif_analysis['reused_neurons']}
    Max reuse: {motif_analysis['max_reuse']} features

    Mean circuit sparsity: {motif_analysis['mean_sparsity']:.2%}
    Std circuit sparsity: {motif_analysis['std_sparsity']:.2%}

    Highly reused neurons (≥3 features): {len(motif_analysis['highly_reused'])}
    """

    ax.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
            verticalalignment='center')

    # Plot 4: Sparsity distribution
    ax = axes[1, 1]
    # Placeholder - would need all circuits to compute this
    ax.text(0.5, 0.5, 'Sparsity analysis\n(requires all circuits)',
            ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved motif analysis to {save_path}")

    plt.close()


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)

    Returns:
        JSON-serializable version of obj
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(
        description='Feature Attribution Analysis for SAE Circuit Extraction'
    )
    parser.add_argument('--sae-model', type=str, required=True,
                       help='Path to trained SAE model (.pt file)')
    parser.add_argument('--session-id', type=int, required=True,
                       help='Allen session ID')
    parser.add_argument('--allen-cache', type=str, required=True,
                       help='Path to Allen cache directory')
    parser.add_argument('--output-dir', type=str, default='results/circuits',
                       help='Output directory for results')
    parser.add_argument('--top-features', type=int, default=20,
                       help='Number of top features to analyze')
    parser.add_argument('--top-neurons', type=int, default=10,
                       help='Number of top neurons per circuit')
    parser.add_argument('--num-steps', type=int, default=50,
                       help='Integration steps for Integrated Gradients')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Feature Attribution Analysis")
    logger.info("="*60)

    # Load SAE and data
    sae, neural_data, orientations, metadata = load_sae_and_data(
        args.sae_model, args.session_id, args.allen_cache
    )

    # Compute feature selectivity
    logger.info("\nComputing SAE feature selectivity...")
    correlations, preferred_oris = compute_feature_selectivity(
        sae, neural_data, orientations
    )

    # Identify top selective features
    top_feature_indices = np.argsort(correlations)[-args.top_features:][::-1]

    logger.info(f"\nTop {args.top_features} features:")
    for i, feat_idx in enumerate(top_feature_indices[:10]):
        logger.info(f"  {i+1}. Feature {feat_idx}: corr={correlations[feat_idx]:.3f}, "
                   f"pref_ori={preferred_oris[feat_idx]:.1f}°")

    # Compute attributions
    logger.info("\nComputing feature attributions...")
    attributor = FeatureAttributor(sae, device=args.device)

    X_tensor = torch.FloatTensor(neural_data)
    attribution_matrix = attributor.compute_all_attributions(
        X_tensor, metadata['n_features'], num_steps=args.num_steps
    )

    logger.info(f"  Attribution matrix shape: {attribution_matrix.shape}")

    # Save attribution matrix
    np.save(output_dir / f'attribution_matrix_session_{args.session_id}.npy',
            attribution_matrix)
    logger.info(f"  Saved attribution matrix")

    # Extract circuits
    logger.info("\nExtracting circuits...")
    extractor = CircuitExtractor(attribution_matrix, threshold=0.1)

    circuits = []
    for feat_idx in top_feature_indices:
        circuit = extractor.extract_circuit(feat_idx, top_k=args.top_neurons)
        circuit['selectivity'] = correlations[feat_idx]
        circuit['preferred_orientation'] = preferred_oris[feat_idx]
        circuits.append(circuit)

    # Analyze motifs
    logger.info("\nAnalyzing circuit motifs...")
    motif_analysis = extractor.analyze_circuit_motifs(circuits)

    logger.info(f"  Total neurons used: {motif_analysis['total_neurons_used']}")
    logger.info(f"  Reused neurons: {motif_analysis['reused_neurons']}")
    logger.info(f"  Max reuse: {motif_analysis['max_reuse']} features")
    logger.info(f"  Mean circuit sparsity: {motif_analysis['mean_sparsity']:.2%}")

    # Compute raw neuron selectivity for comparison
    logger.info("\nComputing raw neuron selectivity...")
    raw_selectivity = []
    for neuron_idx in range(neural_data.shape[1]):
        neuron_response = neural_data[:, neuron_idx]
        ori_sin = np.sin(np.deg2rad(orientations * 2))
        ori_cos = np.cos(np.deg2rad(orientations * 2))

        corr_sin, _ = pearsonr(neuron_response, ori_sin)
        corr_cos, _ = pearsonr(neuron_response, ori_cos)
        corr = max(abs(corr_sin), abs(corr_cos))
        raw_selectivity.append(corr)

    raw_selectivity = np.array(raw_selectivity)

    # Visualize top circuits
    logger.info(f"\nGenerating visualizations for top {min(10, len(circuits))} circuits...")
    for i, circuit in enumerate(circuits[:10]):
        feat_idx = circuit['feature_idx']
        save_path = output_dir / f'circuit_feature_{feat_idx}.png'

        visualize_circuit(
            circuit,
            attribution_matrix,
            neuron_selectivity=raw_selectivity,
            save_path=save_path
        )

    # Visualize motif analysis
    motif_save_path = output_dir / 'circuit_motif_analysis.png'
    visualize_motif_analysis(motif_analysis, save_path=motif_save_path)

    # Save results - convert all numpy types to Python native types
    results = {
        'session_id': args.session_id,
        'metadata': metadata,
        'top_features': top_feature_indices.tolist(),
        'feature_selectivity': correlations[top_feature_indices].tolist(),
        'feature_preferred_orientations': preferred_oris[top_feature_indices].tolist(),
        'circuits': convert_to_json_serializable(circuits),
        'motif_analysis': {
            'total_neurons_used': int(motif_analysis['total_neurons_used']),
            'reused_neurons': int(motif_analysis['reused_neurons']),
            'max_reuse': int(motif_analysis['max_reuse']),
            'mean_sparsity': float(motif_analysis['mean_sparsity']),
            'std_sparsity': float(motif_analysis['std_sparsity']),
            'highly_reused_count': len(motif_analysis['highly_reused'])
        }
    }

    results_path = output_dir / f'attribution_results_session_{args.session_id}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved results to {results_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Analyzed {args.top_features} top SAE features")
    logger.info(f"Extracted {len(circuits)} circuits")
    logger.info(f"Visualizations saved to {output_dir}")
    logger.info(f"\nCircuit Statistics:")
    logger.info(f"  - Mean neurons per circuit: {np.mean([c['n_neurons_above_threshold'] for c in circuits]):.1f}")
    logger.info(f"  - Mean circuit sparsity: {motif_analysis['mean_sparsity']:.2%}")
    logger.info(f"  - Highly reused neurons: {len(motif_analysis['highly_reused'])}")

    logger.info("\n✓ Feature Attribution Analysis Complete!")


if __name__ == '__main__':
    main()
