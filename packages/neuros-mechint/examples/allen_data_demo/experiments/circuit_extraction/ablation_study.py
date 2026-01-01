#!/usr/bin/env python3
"""
Causal Circuit Perturbation via Ablation
==========================================

Implements Experiment 1.2 from ADVANCED_RESEARCH_ROADMAP.md:
- Ablate individual neurons to measure causal impact on SAE features
- Identify minimal circuits (which neurons are necessary vs sufficient)
- Measure circuit redundancy and robustness
- Validate attribution results with causal interventions

This script answers:
1. Which neurons are causally necessary for each SAE feature?
2. How robust are features to single neuron damage?
3. What is the minimal circuit for orientation selectivity?

Usage:
    python experiments/circuit_extraction/ablation_study.py \
        --sae-model sae_models/sae_session_754829445.pt \
        --attribution-results results/circuits/attribution_results_session_754829445.json \
        --session-id 754829445 \
        --allen-cache allen_validation_cache \
        --output-dir results/circuits/ablation
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


class NeuronAblator:
    """
    Systematically ablate neurons and measure impact on SAE features.

    Uses intervention methods from neuros-mechint to causally test
    which neurons are necessary for feature activation.
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

    def ablate_single_neuron(
        self,
        neural_data: torch.Tensor,
        neuron_idx: int,
        ablation_type: str = 'zero'
    ) -> torch.Tensor:
        """
        Ablate a single input neuron and return SAE features.

        Args:
            neural_data: Input neural activity [batch, neurons]
            neuron_idx: Index of neuron to ablate
            ablation_type: 'zero' (set to 0) or 'mean' (set to mean)

        Returns:
            ablated_features: SAE features with neuron ablated [batch, features]
        """
        neural_data = neural_data.to(self.device)
        ablated_data = neural_data.clone()

        if ablation_type == 'zero':
            ablated_data[:, neuron_idx] = 0
        elif ablation_type == 'mean':
            ablated_data[:, neuron_idx] = neural_data[:, neuron_idx].mean()
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

        # Get SAE features with ablation
        with torch.no_grad():
            ablated_features = self.sae.encode(ablated_data)

        return ablated_features

    def measure_feature_disruption(
        self,
        baseline_features: torch.Tensor,
        ablated_features: torch.Tensor,
        feature_idx: int
    ) -> Dict[str, float]:
        """
        Measure how much ablation disrupted a specific SAE feature.

        Args:
            baseline_features: Original SAE features [batch, features]
            ablated_features: SAE features after ablation [batch, features]
            feature_idx: Index of feature to measure

        Returns:
            disruption_metrics: Dict with various disruption measures
        """
        baseline = baseline_features[:, feature_idx].cpu().numpy()
        ablated = ablated_features[:, feature_idx].cpu().numpy()

        # Absolute change
        abs_change = np.abs(ablated - baseline).mean()

        # Relative change
        baseline_mean = baseline.mean()
        if baseline_mean > 1e-8:
            relative_change = abs_change / baseline_mean
        else:
            relative_change = 0.0

        # Correlation (preserved structure)
        if len(baseline) > 1 and baseline.std() > 1e-8 and ablated.std() > 1e-8:
            correlation, _ = pearsonr(baseline, ablated)
        else:
            correlation = 1.0

        # Peak response change
        peak_change = np.abs(baseline.max() - ablated.max()) / (baseline.max() + 1e-8)

        metrics = {
            'mean_abs_change': float(abs_change),
            'relative_change': float(relative_change),
            'correlation': float(correlation),
            'peak_response_change': float(peak_change),
            'disruption_score': float(abs_change * (1 - correlation))  # Combined metric
        }

        return metrics

    def ablate_circuit(
        self,
        neural_data: torch.Tensor,
        circuit_neurons: List[int],
        feature_idx: int,
        ablation_type: str = 'zero'
    ) -> Dict[str, float]:
        """
        Ablate an entire circuit (multiple neurons) and measure impact.

        Args:
            neural_data: Input neural data
            circuit_neurons: List of neuron indices to ablate
            feature_idx: SAE feature to measure
            ablation_type: Ablation method

        Returns:
            disruption_metrics: Impact of ablating entire circuit
        """
        # Baseline features
        with torch.no_grad():
            baseline_features = self.sae.encode(neural_data.to(self.device))

        # Ablate all circuit neurons
        ablated_data = neural_data.clone()
        for neuron_idx in circuit_neurons:
            if ablation_type == 'zero':
                ablated_data[:, neuron_idx] = 0
            elif ablation_type == 'mean':
                ablated_data[:, neuron_idx] = neural_data[:, neuron_idx].mean()

        # Get ablated features
        with torch.no_grad():
            ablated_features = self.sae.encode(ablated_data.to(self.device))

        # Measure disruption
        metrics = self.measure_feature_disruption(
            baseline_features, ablated_features, feature_idx
        )

        return metrics

    def find_minimal_circuit(
        self,
        neural_data: torch.Tensor,
        circuit_neurons: List[int],
        feature_idx: int,
        threshold: float = 0.9,
        ablation_type: str = 'zero'
    ) -> Dict:
        """
        Find minimal set of neurons needed to preserve feature (≥threshold performance).

        Uses greedy ablation: removes neurons one by one until feature degrades.

        Args:
            neural_data: Input data
            circuit_neurons: Starting circuit (from attribution)
            feature_idx: SAE feature index
            threshold: Minimum correlation to maintain
            ablation_type: Ablation method

        Returns:
            minimal_circuit: Dict with minimal neuron set and statistics
        """
        logger.info(f"  Finding minimal circuit for feature {feature_idx}...")

        # Baseline
        with torch.no_grad():
            baseline_features = self.sae.encode(neural_data.to(self.device))
        baseline_activation = baseline_features[:, feature_idx].cpu().numpy()

        # Start with full circuit
        active_neurons = set(circuit_neurons)
        removed_neurons = []

        # Greedy removal
        for neuron_idx in circuit_neurons:
            # Try removing this neuron
            test_neurons = active_neurons - {neuron_idx}

            # Ablate all neurons NOT in test set
            all_neurons = set(range(neural_data.shape[1]))
            ablate_set = all_neurons - test_neurons

            ablated_data = neural_data.clone()
            for ablate_idx in ablate_set:
                if ablation_type == 'zero':
                    ablated_data[:, ablate_idx] = 0

            # Get features
            with torch.no_grad():
                ablated_features = self.sae.encode(ablated_data.to(self.device))
            ablated_activation = ablated_features[:, feature_idx].cpu().numpy()

            # Check if correlation maintained
            if baseline_activation.std() > 1e-8 and ablated_activation.std() > 1e-8:
                corr, _ = pearsonr(baseline_activation, ablated_activation)
            else:
                corr = 0.0

            # If performance OK, keep neuron removed
            if corr >= threshold:
                active_neurons = test_neurons
                removed_neurons.append(neuron_idx)
                logger.info(f"    Removed neuron {neuron_idx}, corr={corr:.3f}")

        minimal_circuit = {
            'feature_idx': feature_idx,
            'original_circuit_size': len(circuit_neurons),
            'minimal_circuit_size': len(active_neurons),
            'minimal_neurons': sorted(list(active_neurons)),
            'removed_neurons': removed_neurons,
            'compression_ratio': len(active_neurons) / len(circuit_neurons)
        }

        logger.info(f"    Minimal circuit: {len(active_neurons)}/{len(circuit_neurons)} neurons")

        return minimal_circuit


def load_data(
    sae_model_path: str,
    session_id: int,
    cache_dir: str
) -> Tuple[nn.Module, torch.Tensor, np.ndarray]:
    """Load SAE and neural data."""
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

    # Extract data
    valid_data = []
    valid_orientations = []

    for w in windows:
        ori = w.metadata['orientation']
        if ori == 'null' or ori is None:
            continue
        try:
            ori_float = float(ori)
            valid_data.append(w.data.mean(axis=0))
            valid_orientations.append(ori_float % 180)
        except (ValueError, TypeError):
            continue

    neural_data = np.array(valid_data)
    orientations = np.array(valid_orientations)

    # Normalize
    neural_data = (neural_data - neural_data.mean(axis=0)) / (neural_data.std(axis=0) + 1e-8)

    logger.info(f"  Neural data: {neural_data.shape}")

    # Load SAE
    logger.info(f"Loading SAE from {sae_model_path}...")
    input_dim = neural_data.shape[1]

    # Infer hidden_dim
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

    logger.info(f"  SAE: {input_dim} → {hidden_dim}")

    return sae, torch.FloatTensor(neural_data), orientations


def visualize_ablation_results(
    ablation_results: List[Dict],
    feature_idx: int,
    save_path: Optional[str] = None
):
    """
    Visualize ablation results for a single feature.

    Shows impact of ablating each neuron on feature activation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    neuron_indices = [r['neuron_idx'] for r in ablation_results]
    disruption_scores = [r['disruption_score'] for r in ablation_results]
    correlations = [r['correlation'] for r in ablation_results]
    relative_changes = [r['relative_change'] for r in ablation_results]

    # Plot 1: Disruption scores
    ax = axes[0, 0]
    colors = ['red' if d > 0.2 else 'orange' if d > 0.1 else 'green' for d in disruption_scores]
    ax.bar(range(len(neuron_indices)), disruption_scores, color=colors, alpha=0.7)
    ax.set_xlabel('Neuron Rank')
    ax.set_ylabel('Disruption Score')
    ax.set_title(f'Feature {feature_idx}: Ablation Impact')
    ax.axhline(0.2, color='red', linestyle='--', alpha=0.3, label='High impact')
    ax.axhline(0.1, color='orange', linestyle='--', alpha=0.3, label='Medium impact')
    ax.legend()

    # Plot 2: Correlation preservation
    ax = axes[0, 1]
    ax.plot(range(len(correlations)), correlations, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Neuron Rank')
    ax.set_ylabel('Correlation (baseline vs ablated)')
    ax.set_title('Feature Structure Preservation')
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.3, label='90% preserved')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Relative change
    ax = axes[1, 0]
    ax.scatter(range(len(relative_changes)), relative_changes, c=disruption_scores,
               cmap='Reds', s=100, alpha=0.7)
    ax.set_xlabel('Neuron Rank')
    ax.set_ylabel('Relative Activation Change')
    ax.set_title('Activation Magnitude Change')
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    n_critical = sum(1 for d in disruption_scores if d > 0.2)
    n_important = sum(1 for d in disruption_scores if 0.1 < d <= 0.2)
    n_negligible = sum(1 for d in disruption_scores if d <= 0.1)

    summary_text = f"""
    Ablation Study Summary
    {'='*40}

    Feature: {feature_idx}
    Neurons tested: {len(ablation_results)}

    Impact categories:
      Critical (>0.2): {n_critical} neurons
      Important (0.1-0.2): {n_important} neurons
      Negligible (<0.1): {n_negligible} neurons

    Mean disruption: {np.mean(disruption_scores):.3f}
    Max disruption: {np.max(disruption_scores):.3f}

    Most critical neuron: {neuron_indices[np.argmax(disruption_scores)]}
    (disruption score: {np.max(disruption_scores):.3f})
    """

    ax.text(0.1, 0.5, summary_text, fontfamily='monospace',
            fontsize=10, verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved ablation visualization to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Causal Circuit Perturbation via Ablation'
    )
    parser.add_argument('--sae-model', type=str, required=True,
                       help='Path to trained SAE model')
    parser.add_argument('--attribution-results', type=str, required=True,
                       help='Path to attribution results JSON')
    parser.add_argument('--session-id', type=int, required=True,
                       help='Allen session ID')
    parser.add_argument('--allen-cache', type=str, required=True,
                       help='Allen cache directory')
    parser.add_argument('--output-dir', type=str, default='results/circuits/ablation',
                       help='Output directory')
    parser.add_argument('--top-features', type=int, default=10,
                       help='Number of top features to ablate')
    parser.add_argument('--ablation-type', type=str, default='zero',
                       choices=['zero', 'mean'],
                       help='Ablation method')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Causal Circuit Perturbation Analysis")
    logger.info("="*60)

    # Load attribution results
    logger.info(f"\nLoading attribution results from {args.attribution_results}...")
    with open(args.attribution_results) as f:
        attribution_data = json.load(f)

    circuits = attribution_data['circuits'][:args.top_features]
    logger.info(f"  Loaded {len(circuits)} circuits to test")

    # Load SAE and data
    sae, neural_data, orientations = load_data(
        args.sae_model, args.session_id, args.allen_cache
    )

    # Initialize ablator
    ablator = NeuronAblator(sae, device=args.device)

    # Get baseline features
    logger.info("\nComputing baseline SAE features...")
    with torch.no_grad():
        baseline_features = sae.encode(neural_data.to(args.device))

    # Ablation study for each circuit
    all_ablation_results = {}
    minimal_circuits = []

    for circuit in tqdm(circuits, desc="Ablating circuits"):
        feature_idx = circuit['feature_idx']
        circuit_neurons = circuit['top_neurons']

        logger.info(f"\n--- Feature {feature_idx} (selectivity={circuit['selectivity']:.3f}) ---")

        # Ablate each neuron individually
        neuron_ablation_results = []

        for neuron_idx in circuit_neurons:
            # Ablate single neuron
            ablated_features = ablator.ablate_single_neuron(
                neural_data, neuron_idx, ablation_type=args.ablation_type
            )

            # Measure disruption
            metrics = ablator.measure_feature_disruption(
                baseline_features, ablated_features, feature_idx
            )

            metrics['neuron_idx'] = neuron_idx
            neuron_ablation_results.append(metrics)

            logger.info(f"  Neuron {neuron_idx}: disruption={metrics['disruption_score']:.3f}, "
                       f"corr={metrics['correlation']:.3f}")

        all_ablation_results[feature_idx] = neuron_ablation_results

        # Find minimal circuit
        minimal_circuit = ablator.find_minimal_circuit(
            neural_data, circuit_neurons, feature_idx,
            threshold=0.9, ablation_type=args.ablation_type
        )
        minimal_circuits.append(minimal_circuit)

        # Visualize
        vis_path = output_dir / f'ablation_feature_{feature_idx}.png'
        visualize_ablation_results(neuron_ablation_results, feature_idx, save_path=vis_path)

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    compression_ratios = [mc['compression_ratio'] for mc in minimal_circuits]
    logger.info(f"\nMinimal Circuit Statistics:")
    logger.info(f"  Mean compression: {np.mean(compression_ratios):.2%}")
    logger.info(f"  Median compression: {np.median(compression_ratios):.2%}")

    # Count critical neurons
    critical_neurons = set()
    for feature_idx, results in all_ablation_results.items():
        for r in results:
            if r['disruption_score'] > 0.2:
                critical_neurons.add(r['neuron_idx'])

    logger.info(f"\nCritical neurons (high impact): {len(critical_neurons)}")

    # Save results
    results = {
        'session_id': args.session_id,
        'ablation_type': args.ablation_type,
        'ablation_results': {
            str(k): v for k, v in all_ablation_results.items()
        },
        'minimal_circuits': minimal_circuits,
        'summary': {
            'mean_compression_ratio': float(np.mean(compression_ratios)),
            'n_critical_neurons': len(critical_neurons),
            'critical_neurons': list(critical_neurons)
        }
    }

    results_path = output_dir / f'ablation_results_session_{args.session_id}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Saved results to {results_path}")
    logger.info("✓ Ablation Study Complete!")


if __name__ == '__main__':
    main()
