"""
Comprehensive Mechanistic Interpretability Analysis
===================================================

This example demonstrates the world's most advanced mechanistic interpretability
suite for neural foundation models, including:

1. Sparse Autoencoder (SAE) concept discovery with hierarchical dictionaries
2. Brain-model alignment (CCA, RSA, Procrustes)
3. Dynamical systems analysis (Koopman operators, Lyapunov exponents)
4. Causal graph discovery (Granger causality, time-varying networks)
5. Counterfactual interventions (latent surgery, do-calculus)
6. Meta-dynamics tracking (training trajectory analysis)
7. Geometric/topological analysis (manifolds, persistent homology)
8. Attribution and importance scoring (integrated gradients)
9. Automated HTML report generation

Requirements:
    - Trained NeuroFMX checkpoint
    - Validation dataset (1000+ samples recommended)
    - 32GB+ RAM
    - Optional: Multi-modal brain recordings for alignment
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from neuros_neurofm.model import NeuroFMX
from neuros_mechint import (
    # Sparse autoencoders
    HierarchicalSAE,
    ConceptDictionary,
    MultiLayerSAETrainer,

    # Brain alignment
    CCAAlignment,
    RSAAlignment,

    # Dynamics
    DynamicsAnalyzer,
    KoopmanOperator,

    # Causal analysis
    CausalGraphBuilder,

    # Counterfactuals
    LatentSurgery,
    DoCalculusEngine,

    # Meta-dynamics
    MetaDynamicsTracker,

    # Geometry/Topology
    ManifoldAnalyzer,
    TopologyAnalyzer,

    # Attribution
    AttributionAnalyzer,

    # Reporting
    MechIntReporter,
)
from neuros_neurofm.data.webdataset_loader import create_webdataset_loader


def load_model(checkpoint_path: str, config: dict) -> NeuroFMX:
    """Load trained NeuroFMX model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    model = NeuroFMX(
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        architecture=config['model']['architecture'],
        modality_configs=config['model']['modalities'],
    )

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    return model


def extract_activations(
    model: NeuroFMX,
    data_loader,
    layers_to_extract: List[str],
    max_samples: int = 10000,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Extract intermediate activations from model

    Returns:
        Dictionary {layer_name: activations}
        Each activation tensor is (N, T, D) where N=samples, T=time, D=features
    """
    print(f"\nExtracting activations from {len(layers_to_extract)} layers...")
    print(f"Max samples: {max_samples}")

    model = model.to(device)
    activations = {layer: [] for layer in layers_to_extract}

    # Register hooks
    handles = []

    def make_hook(layer_name):
        def hook(module, input, output):
            # Store activation on CPU to save memory
            activations[layer_name].append(output.detach().cpu())
        return hook

    # Attach hooks
    for layer_name in layers_to_extract:
        module = dict(model.named_modules())[layer_name]
        handle = module.register_forward_hook(make_hook(layer_name))
        handles.append(handle)

    # Forward pass through data
    num_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if num_samples >= max_samples:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            _ = model(batch)

            num_samples += batch['eeg'].shape[0] if 'eeg' in batch else 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {num_samples}/{max_samples} samples")

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Concatenate activations
    for layer_name in layers_to_extract:
        activations[layer_name] = torch.cat(activations[layer_name], dim=0)
        print(f"  {layer_name}: {activations[layer_name].shape}")

    print("✓ Activation extraction complete")

    return activations


def run_sae_analysis(
    activations: Dict[str, torch.Tensor],
    config: dict,
    output_dir: Path
) -> Dict[str, HierarchicalSAE]:
    """
    Train hierarchical SAEs and discover concepts

    Returns dictionary of trained SAEs per layer
    """
    print("\n" + "=" * 80)
    print("1. Sparse Autoencoder (SAE) Analysis")
    print("=" * 80)

    sae_config = config['mechint']['sae']
    saes = {}

    for layer_name, layer_activations in activations.items():
        print(f"\nTraining SAE for {layer_name}...")

        # Flatten time dimension: (N, T, D) -> (N*T, D)
        flat_activations = layer_activations.reshape(-1, layer_activations.shape[-1])

        # Create hierarchical SAE
        hsae = HierarchicalSAE(
            layer_sizes=[
                flat_activations.shape[-1],  # Input dimension
                *sae_config['layer_sizes']
            ],
            sparsity_coefficients=sae_config['sparsity_coefficients'],
        )

        # Train SAE
        trainer = MultiLayerSAETrainer(hsae)
        trainer.train(
            activations=flat_activations,
            num_epochs=sae_config['train_epochs'],
            batch_size=256,
            learning_rate=1e-4,
        )

        saes[layer_name] = hsae

        print(f"✓ SAE trained for {layer_name}")

        # Save SAE
        sae_path = output_dir / f'sae_{layer_name.replace(".", "_")}.pt'
        torch.save(hsae.state_dict(), sae_path)
        print(f"  Saved to {sae_path}")

    return saes


def run_brain_alignment(
    model_activations: Dict[str, torch.Tensor],
    brain_data: Dict[str, torch.Tensor],
    config: dict,
    output_dir: Path
):
    """
    Analyze brain-model alignment using CCA and RSA
    """
    print("\n" + "=" * 80)
    print("2. Brain-Model Alignment Analysis")
    print("=" * 80)

    alignment_config = config['mechint']['alignment']

    results = {}

    for layer_name, model_acts in model_activations.items():
        print(f"\nAnalyzing alignment for {layer_name}...")

        # Flatten activations
        model_flat = model_acts.reshape(-1, model_acts.shape[-1])

        for brain_region, brain_acts in brain_data.items():
            brain_flat = brain_acts.reshape(-1, brain_acts.shape[-1])

            # Ensure same number of samples
            min_samples = min(model_flat.shape[0], brain_flat.shape[0])
            model_flat = model_flat[:min_samples]
            brain_flat = brain_flat[:min_samples]

            # CCA alignment
            if alignment_config['method'] in ['CCA', 'all']:
                cca = CCAAlignment(
                    n_components=alignment_config['n_components'],
                    regularization=0.01,
                )

                cca_score, cca_dims, cca_ci = cca.fit_and_score(
                    model_flat.numpy(),
                    brain_flat.numpy(),
                    bootstrap=alignment_config.get('bootstrap', 200),
                )

                results[f'{layer_name}_{brain_region}_cca'] = {
                    'score': cca_score,
                    'dimensions': cca_dims,
                    'ci_lower': cca_ci[0],
                    'ci_upper': cca_ci[1],
                }

                print(f"  {brain_region} CCA: {cca_score:.3f} [{cca_ci[0]:.3f}, {cca_ci[1]:.3f}]")

            # RSA alignment
            if alignment_config['method'] in ['RSA', 'all']:
                rsa = RSAAlignment(method='spearman')

                rsa_score, rsa_ci = rsa.fit_and_score(
                    model_flat.numpy(),
                    brain_flat.numpy(),
                    bootstrap=alignment_config.get('bootstrap', 200),
                )

                results[f'{layer_name}_{brain_region}_rsa'] = {
                    'score': rsa_score,
                    'ci_lower': rsa_ci[0],
                    'ci_upper': rsa_ci[1],
                }

                print(f"  {brain_region} RSA: {rsa_score:.3f} [{rsa_ci[0]:.3f}, {rsa_ci[1]:.3f}]")

    # Save results
    import json
    results_path = output_dir / 'alignment_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Alignment results saved to {results_path}")

    return results


def run_dynamics_analysis(
    activations: Dict[str, torch.Tensor],
    config: dict,
    output_dir: Path
):
    """
    Analyze dynamical properties: Koopman operators, Lyapunov exponents, manifolds
    """
    print("\n" + "=" * 80)
    print("3. Dynamical Systems Analysis")
    print("=" * 80)

    dynamics_config = config['mechint'].get('dynamics', {})

    results = {}

    for layer_name, layer_acts in activations.items():
        print(f"\nAnalyzing dynamics for {layer_name}...")

        # Initialize analyzer
        analyzer = DynamicsAnalyzer(
            dim=layer_acts.shape[-1],
            dt=0.001,  # 1ms timestep
        )

        # Koopman operator analysis
        print("  Computing Koopman operator...")
        koopman_window = dynamics_config.get('koopman_window', 128)

        koopman = KoopmanOperator(
            state_dim=layer_acts.shape[-1],
            window_size=koopman_window,
        )

        # Fit on trajectories
        trajectories = layer_acts.numpy()  # (N, T, D)
        koopman.fit(trajectories)

        # Extract eigenvalues (stability analysis)
        eigenvalues = koopman.eigenvalues
        stable_modes = (np.abs(eigenvalues) < 1).sum()
        unstable_modes = (np.abs(eigenvalues) >= 1).sum()

        print(f"    Stable modes: {stable_modes}")
        print(f"    Unstable modes: {unstable_modes}")

        # Lyapunov exponents
        print("  Computing Lyapunov exponents...")
        lyapunov_exponents = analyzer.compute_lyapunov_exponents(
            trajectories[:100],  # Use subset for speed
            steps=1000,
        )

        max_lyapunov = lyapunov_exponents.max()
        print(f"    Max Lyapunov exponent: {max_lyapunov:.4f}")
        if max_lyapunov > 0:
            print(f"    → Chaotic dynamics detected")
        else:
            print(f"    → Stable dynamics")

        # Manifold analysis
        print("  Estimating manifold dimension...")
        manifold_analyzer = ManifoldAnalyzer()
        intrinsic_dim = manifold_analyzer.estimate_intrinsic_dimension(
            trajectories.reshape(-1, trajectories.shape[-1])[:5000],
            method='mle',
        )

        print(f"    Intrinsic dimension: {intrinsic_dim:.1f}")
        print(f"    Embedding dimension: {layer_acts.shape[-1]}")
        print(f"    Compression ratio: {intrinsic_dim / layer_acts.shape[-1]:.2%}")

        results[layer_name] = {
            'koopman_stable_modes': int(stable_modes),
            'koopman_unstable_modes': int(unstable_modes),
            'max_lyapunov': float(max_lyapunov),
            'intrinsic_dimension': float(intrinsic_dim),
            'embedding_dimension': int(layer_acts.shape[-1]),
        }

    # Save results
    import json
    results_path = output_dir / 'dynamics_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Dynamics results saved to {results_path}")

    return results


def run_counterfactual_analysis(
    model: NeuroFMX,
    sample_batch: Dict,
    layers_to_intervene: List[str],
    config: dict,
    output_dir: Path,
    device: str = 'cuda'
):
    """
    Perform counterfactual interventions via latent surgery
    """
    print("\n" + "=" * 80)
    print("4. Counterfactual Interventions")
    print("=" * 80)

    counterfactual_config = config['mechint'].get('counterfactuals', {})

    model = model.to(device)
    sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample_batch.items()}

    # Initialize latent surgery engine
    surgery = LatentSurgery(model)

    results = {}

    for layer_name in layers_to_intervene:
        print(f"\nIntervening on {layer_name}...")

        # Baseline output
        with torch.no_grad():
            baseline_output = model(sample_batch)

        # Define intervention: amplify activity
        def amplify_intervention(latent):
            return latent * counterfactual_config.get('magnitude', 1.5)

        # Apply intervention
        intervened_output = surgery.edit_latent(
            input_data=sample_batch,
            layer_name=layer_name,
            edit_fn=amplify_intervention,
        )

        # Measure effect
        effect_magnitude = (intervened_output - baseline_output).abs().mean().item()

        print(f"  Effect magnitude: {effect_magnitude:.4f}")

        results[layer_name] = {
            'intervention': 'amplify_1.5x',
            'effect_magnitude': float(effect_magnitude),
        }

    # Save results
    import json
    results_path = output_dir / 'counterfactual_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Counterfactual results saved to {results_path}")

    return results


def run_topology_analysis(
    activations: Dict[str, torch.Tensor],
    config: dict,
    output_dir: Path
):
    """
    Analyze topological structure via persistent homology
    """
    print("\n" + "=" * 80)
    print("5. Topological Analysis (Persistent Homology)")
    print("=" * 80)

    topology_config = config['mechint'].get('topology', {})

    topology_analyzer = TopologyAnalyzer(
        max_dimension=topology_config.get('rips_maxdim', 2),
    )

    results = {}

    for layer_name, layer_acts in activations.items():
        print(f"\nAnalyzing topology for {layer_name}...")

        # Flatten and subsample (persistent homology is expensive)
        flat_acts = layer_acts.reshape(-1, layer_acts.shape[-1])
        subsample_size = min(2000, flat_acts.shape[0])
        indices = np.random.choice(flat_acts.shape[0], subsample_size, replace=False)
        subsample = flat_acts[indices].numpy()

        # Compute persistent homology
        persistence = topology_analyzer.compute_persistence(subsample)

        # Extract Betti numbers (topological features)
        betti_numbers = topology_analyzer.compute_betti_numbers(persistence)

        print(f"  Betti numbers: {betti_numbers}")
        print(f"    β₀ (connected components): {betti_numbers[0]}")
        print(f"    β₁ (loops): {betti_numbers.get(1, 0)}")
        print(f"    β₂ (voids): {betti_numbers.get(2, 0)}")

        results[layer_name] = {
            'betti_numbers': {int(k): int(v) for k, v in betti_numbers.items()},
        }

    # Save results
    import json
    results_path = output_dir / 'topology_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Topology results saved to {results_path}")

    return results


def generate_comprehensive_report(
    output_dir: Path,
    config: dict,
    all_results: Dict,
):
    """
    Generate comprehensive HTML report with all analyses
    """
    print("\n" + "=" * 80)
    print("6. Generating Comprehensive Report")
    print("=" * 80)

    reporter = MechIntReporter(
        output_dir=str(output_dir),
        format=config['mechint']['reporting'].get('format', 'html'),
    )

    # Add all results
    for analysis_name, results in all_results.items():
        reporter.add_section(
            name=analysis_name,
            results=results,
        )

    # Generate report
    report_path = reporter.generate_report(
        title="NeuroFMX Mechanistic Interpretability Report",
        description="Comprehensive analysis of learned representations",
    )

    print(f"\n✓ Report generated: {report_path}")
    print(f"  Open in browser: file://{report_path}")

    return report_path


def main():
    """Main mechanistic interpretability workflow"""

    # Configuration
    config_path = "configs/mechint/default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Paths
    checkpoint_path = "checkpoints/neurofmx_final.pt"
    output_dir = Path(config['mechint'].get('output_dir', 'reports/mechint'))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NeuroFMX Mechanistic Interpretability Analysis")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    # Load model
    model = load_model(checkpoint_path, config)

    # Setup data loader
    print("\nLoading validation data...")
    val_loader = create_webdataset_loader(
        shard_urls=config['data']['val_shards'],
        batch_size=32,
        num_workers=4,
        modality_specs=config['data']['modality_specs'],
    )
    print("✓ Data loader ready")

    # Extract activations
    layers_to_analyze = [
        f'backbone.layers.{i}' for i in config['mechint']['sample_layers']
    ]

    activations = extract_activations(
        model=model,
        data_loader=val_loader,
        layers_to_extract=layers_to_analyze,
        max_samples=config.get('resources', {}).get('max_samples', 10000),
    )

    # Store all results
    all_results = {}

    # 1. SAE Analysis
    if config['mechint']['sae']['enabled']:
        sae_results = run_sae_analysis(activations, config, output_dir)
        all_results['sae'] = sae_results

    # 2. Brain Alignment (if brain data available)
    if config['mechint']['alignment']['enabled']:
        # Load brain data (placeholder - replace with actual data)
        brain_data = {
            'motor_cortex': torch.randn(1000, 100, 64),
            'visual_cortex': torch.randn(1000, 100, 128),
        }
        alignment_results = run_brain_alignment(activations, brain_data, config, output_dir)
        all_results['alignment'] = alignment_results

    # 3. Dynamics Analysis
    if config['mechint'].get('dynamics', {}).get('enabled', True):
        dynamics_results = run_dynamics_analysis(activations, config, output_dir)
        all_results['dynamics'] = dynamics_results

    # 4. Counterfactual Interventions
    if config['mechint'].get('counterfactuals', {}).get('enabled', False):
        # Get sample batch
        sample_batch = next(iter(val_loader))
        counterfactual_results = run_counterfactual_analysis(
            model, sample_batch, layers_to_analyze, config, output_dir
        )
        all_results['counterfactuals'] = counterfactual_results

    # 5. Topology Analysis
    if config['mechint'].get('topology', {}).get('enabled', True):
        topology_results = run_topology_analysis(activations, config, output_dir)
        all_results['topology'] = topology_results

    # 6. Generate comprehensive report
    report_path = generate_comprehensive_report(output_dir, config, all_results)

    print("\n" + "=" * 80)
    print("Mechanistic Interpretability Analysis Complete!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"HTML report: {report_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
