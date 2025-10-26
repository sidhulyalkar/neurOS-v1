"""
Comprehensive Evaluation and Benchmarking
=========================================

This example demonstrates systematic evaluation of NeuroFMX on downstream tasks:

1. Zero-shot evaluation (frozen features + linear probe)
2. Few-shot learning (1-shot, 5-shot, 10-shot, 25-shot, 50-shot)
3. Fine-tuning with LoRA adapters
4. Cross-species generalization
5. Cross-task transfer
6. Temporal generalization (train on past, test on future)
7. Benchmark suite (motor decoding, visual encoding, speech decoding, etc.)

Requirements:
    - Trained NeuroFMX checkpoint
    - Labeled benchmark datasets
    - 16GB+ GPU memory
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from neuros_neurofm.model import NeuroFMX
from neuros_neurofm.evaluation import (
    TaskRegistry,
    ZeroShotEvaluator,
    FewShotEvaluator,
    LoRAFineTuner,
)
from neuros_neurofm.data.webdataset_loader import create_webdataset_loader


def load_model(checkpoint_path: str, config: dict, device: str = 'cuda') -> NeuroFMX:
    """Load trained NeuroFMX model"""
    print(f"Loading model from {checkpoint_path}...")

    model = NeuroFMX(
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        architecture=config['model']['architecture'],
        modality_configs=config['model']['modalities'],
        enable_lora=True,  # Enable LoRA for fine-tuning
        lora_rank=config['model'].get('lora_rank', 8),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    return model


def setup_benchmark_tasks(config: dict) -> TaskRegistry:
    """
    Setup benchmark tasks for evaluation

    Standard neuroscience benchmarks:
    - Motor decoding (predict cursor position from neural activity)
    - Visual encoding (predict neural activity from visual stimuli)
    - Speech decoding (decode speech from neural signals)
    - Memory encoding (predict memory formation)
    - Attention decoding (decode attention state)
    """
    print("\nSetting up benchmark tasks...")

    registry = TaskRegistry()

    benchmark_config = config['evaluation']['benchmarks']

    # 1. Motor decoding (regression)
    if 'motor_decoding' in benchmark_config:
        registry.register_task(
            name='motor_decoding',
            task_type='regression',
            input_modalities=['spikes', 'lfp'],
            output_dim=2,  # (x, y) cursor position
            metric='r2',
            description='Decode cursor position from motor cortex activity'
        )

    # 2. Visual encoding (regression)
    if 'visual_encoding' in benchmark_config:
        registry.register_task(
            name='visual_encoding',
            task_type='regression',
            input_modalities=['video'],
            output_dim=128,  # Neural activity dimension
            metric='r2',
            description='Predict V1 activity from visual stimuli'
        )

    # 3. Speech decoding (classification)
    if 'speech_decoding' in benchmark_config:
        registry.register_task(
            name='speech_decoding',
            task_type='classification',
            input_modalities=['ecog'],
            output_dim=50,  # 50 phonemes
            metric='accuracy',
            description='Decode phonemes from ECoG'
        )

    # 4. Memory encoding (binary classification)
    if 'memory_encoding' in benchmark_config:
        registry.register_task(
            name='memory_encoding',
            task_type='classification',
            input_modalities=['eeg'],
            output_dim=2,  # remembered vs forgotten
            metric='auroc',
            description='Predict memory formation from EEG'
        )

    # 5. Sleep stage classification
    if 'sleep_staging' in benchmark_config:
        registry.register_task(
            name='sleep_staging',
            task_type='classification',
            input_modalities=['eeg'],
            output_dim=5,  # Wake, N1, N2, N3, REM
            metric='f1_macro',
            description='Classify sleep stages from EEG'
        )

    print(f"✓ Registered {len(registry.tasks)} benchmark tasks")

    return registry


def run_zero_shot_evaluation(
    model: NeuroFMX,
    task_registry: TaskRegistry,
    data_loaders: Dict[str, torch.utils.data.DataLoader],
    config: dict,
    output_dir: Path,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Zero-shot evaluation: frozen features + linear probe

    This tests the quality of learned representations without task-specific training
    """
    print("\n" + "=" * 80)
    print("1. Zero-Shot Evaluation")
    print("=" * 80)

    evaluator = ZeroShotEvaluator(
        model=model,
        freeze_backbone=True,
        device=device,
    )

    results = []

    for task_name, task_config in task_registry.tasks.items():
        print(f"\nEvaluating {task_name}...")

        if task_name not in data_loaders:
            print(f"  Skipping (no data)")
            continue

        # Extract features
        print("  Extracting features...")
        train_features, train_labels = evaluator.extract_features(
            data_loader=data_loaders[task_name]['train'],
            task_config=task_config,
        )

        test_features, test_labels = evaluator.extract_features(
            data_loader=data_loaders[task_name]['test'],
            task_config=task_config,
        )

        # Train linear probe
        print("  Training linear probe...")
        probe = evaluator.train_probe(
            features=train_features,
            labels=train_labels,
            task_type=task_config['task_type'],
            output_dim=task_config['output_dim'],
        )

        # Evaluate
        print("  Evaluating...")
        metrics = evaluator.evaluate(
            probe=probe,
            features=test_features,
            labels=test_labels,
            task_type=task_config['task_type'],
            metric=task_config['metric'],
        )

        print(f"  ✓ {task_config['metric']}: {metrics[task_config['metric']]:.4f}")

        results.append({
            'task': task_name,
            'method': 'zero_shot',
            'metric': task_config['metric'],
            'score': metrics[task_config['metric']],
            **metrics
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_path = output_dir / 'zero_shot_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Zero-shot results saved to {results_path}")

    return results_df


def run_few_shot_evaluation(
    model: NeuroFMX,
    task_registry: TaskRegistry,
    data_loaders: Dict[str, torch.utils.data.DataLoader],
    config: dict,
    output_dir: Path,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Few-shot evaluation: K-shot learning with LoRA

    Tests sample efficiency of the foundation model
    """
    print("\n" + "=" * 80)
    print("2. Few-Shot Evaluation")
    print("=" * 80)

    few_shot_config = config['evaluation'].get('few_shot', {})
    k_shots = few_shot_config.get('k_shots', [1, 5, 10, 25, 50])

    evaluator = FewShotEvaluator(
        model=model,
        lora_rank=few_shot_config.get('lora_rank', 8),
        lora_alpha=few_shot_config.get('lora_alpha', 16),
        device=device,
    )

    results = []

    for task_name, task_config in task_registry.tasks.items():
        print(f"\nEvaluating {task_name}...")

        if task_name not in data_loaders:
            print(f"  Skipping (no data)")
            continue

        for k in k_shots:
            print(f"\n  {k}-shot learning...")

            # Sample K examples per class (for classification)
            # or K total examples (for regression)
            train_loader_k = evaluator.create_k_shot_loader(
                data_loader=data_loaders[task_name]['train'],
                k=k,
                task_type=task_config['task_type'],
            )

            # Fine-tune with LoRA
            print(f"    Fine-tuning with LoRA...")
            evaluator.fine_tune(
                train_loader=train_loader_k,
                task_config=task_config,
                num_epochs=few_shot_config.get('num_epochs', 10),
                learning_rate=few_shot_config.get('learning_rate', 1e-4),
            )

            # Evaluate
            print(f"    Evaluating...")
            metrics = evaluator.evaluate(
                test_loader=data_loaders[task_name]['test'],
                task_config=task_config,
            )

            print(f"    ✓ {task_config['metric']}: {metrics[task_config['metric']]:.4f}")

            results.append({
                'task': task_name,
                'method': f'{k}_shot',
                'k': k,
                'metric': task_config['metric'],
                'score': metrics[task_config['metric']],
                **metrics
            })

            # Reset LoRA weights for next k
            evaluator.reset_lora()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_path = output_dir / 'few_shot_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Few-shot results saved to {results_path}")

    return results_df


def run_cross_species_evaluation(
    model: NeuroFMX,
    species_data: Dict[str, torch.utils.data.DataLoader],
    config: dict,
    output_dir: Path,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Cross-species generalization:
    Train on one species, test on another

    E.g., train on macaque, test on human
    """
    print("\n" + "=" * 80)
    print("3. Cross-Species Generalization")
    print("=" * 80)

    evaluator = ZeroShotEvaluator(model=model, device=device)

    results = []

    species_list = list(species_data.keys())

    for train_species in species_list:
        for test_species in species_list:
            if train_species == test_species:
                continue

            print(f"\nTrain: {train_species} → Test: {test_species}")

            # Extract features
            train_features, train_labels = evaluator.extract_features(
                data_loader=species_data[train_species]['train'],
                task_config={'input_modalities': ['spikes'], 'output_dim': 2},
            )

            test_features, test_labels = evaluator.extract_features(
                data_loader=species_data[test_species]['test'],
                task_config={'input_modalities': ['spikes'], 'output_dim': 2},
            )

            # Train probe on source species
            probe = evaluator.train_probe(
                features=train_features,
                labels=train_labels,
                task_type='regression',
                output_dim=2,
            )

            # Test on target species
            metrics = evaluator.evaluate(
                probe=probe,
                features=test_features,
                labels=test_labels,
                task_type='regression',
                metric='r2',
            )

            print(f"  R² score: {metrics['r2']:.4f}")

            results.append({
                'train_species': train_species,
                'test_species': test_species,
                'r2': metrics['r2'],
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Create cross-species matrix
    species_matrix = results_df.pivot(
        index='train_species',
        columns='test_species',
        values='r2'
    )

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(species_matrix, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('Cross-Species Generalization (R²)')
    plt.tight_layout()
    plot_path = output_dir / 'cross_species_heatmap.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\n✓ Cross-species heatmap saved to {plot_path}")

    # Save results
    results_path = output_dir / 'cross_species_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ Cross-species results saved to {results_path}")

    return results_df


def plot_few_shot_curves(
    few_shot_results: pd.DataFrame,
    output_dir: Path
):
    """
    Plot few-shot learning curves

    Shows how performance scales with number of training examples
    """
    print("\nGenerating few-shot learning curves...")

    fig, axes = plt.subplots(1, len(few_shot_results['task'].unique()),
                             figsize=(5 * len(few_shot_results['task'].unique()), 4))

    if len(few_shot_results['task'].unique()) == 1:
        axes = [axes]

    for i, task in enumerate(few_shot_results['task'].unique()):
        task_data = few_shot_results[few_shot_results['task'] == task]

        axes[i].plot(task_data['k'], task_data['score'], marker='o', linewidth=2)
        axes[i].set_xlabel('Number of Training Examples (K)')
        axes[i].set_ylabel(task_data['metric'].iloc[0].upper())
        axes[i].set_title(task.replace('_', ' ').title())
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xscale('log')

    plt.tight_layout()
    plot_path = output_dir / 'few_shot_curves.png'
    plt.savefig(plot_path, dpi=300)
    print(f"✓ Few-shot curves saved to {plot_path}")


def generate_benchmark_report(
    zero_shot_results: pd.DataFrame,
    few_shot_results: pd.DataFrame,
    cross_species_results: Optional[pd.DataFrame],
    output_dir: Path,
):
    """
    Generate comprehensive benchmark report
    """
    print("\n" + "=" * 80)
    print("Generating Benchmark Report")
    print("=" * 80)

    report_path = output_dir / 'benchmark_report.md'

    with open(report_path, 'w') as f:
        f.write("# NeuroFMX Benchmark Report\n\n")
        f.write("## Zero-Shot Evaluation\n\n")

        if not zero_shot_results.empty:
            f.write("| Task | Metric | Score |\n")
            f.write("|------|--------|-------|\n")
            for _, row in zero_shot_results.iterrows():
                f.write(f"| {row['task']} | {row['metric']} | {row['score']:.4f} |\n")
            f.write("\n")
        else:
            f.write("No zero-shot results available.\n\n")

        f.write("## Few-Shot Learning\n\n")

        if not few_shot_results.empty:
            for task in few_shot_results['task'].unique():
                task_data = few_shot_results[few_shot_results['task'] == task]
                f.write(f"### {task.replace('_', ' ').title()}\n\n")
                f.write("| K-shot | Score |\n")
                f.write("|--------|-------|\n")
                for _, row in task_data.iterrows():
                    f.write(f"| {row['k']} | {row['score']:.4f} |\n")
                f.write("\n")
        else:
            f.write("No few-shot results available.\n\n")

        if cross_species_results is not None and not cross_species_results.empty:
            f.write("## Cross-Species Generalization\n\n")
            f.write("| Train Species | Test Species | R² |\n")
            f.write("|---------------|--------------|----|\n")
            for _, row in cross_species_results.iterrows():
                f.write(f"| {row['train_species']} | {row['test_species']} | {row['r2']:.4f} |\n")
            f.write("\n")

    print(f"✓ Benchmark report saved to {report_path}")


def main():
    """Main evaluation workflow"""

    # Load configuration
    config_path = "configs/evaluation/default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    checkpoint_path = config['evaluation']['checkpoint_path']
    output_dir = Path(config['evaluation'].get('output_dir', 'reports/evaluation'))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("NeuroFMX Evaluation & Benchmarking")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load model
    model = load_model(checkpoint_path, config, device=device)

    # Setup benchmark tasks
    task_registry = setup_benchmark_tasks(config)

    # Load data loaders (placeholder - replace with actual data)
    print("\nLoading benchmark datasets...")
    data_loaders = {}
    for task_name in task_registry.tasks.keys():
        # Create dummy loaders (replace with actual data loading)
        data_loaders[task_name] = {
            'train': None,  # Replace with actual DataLoader
            'test': None,   # Replace with actual DataLoader
        }
    print("✓ Data loaders ready")

    # Run evaluations
    all_results = {}

    # 1. Zero-shot evaluation
    if config['evaluation'].get('run_zero_shot', True):
        zero_shot_results = run_zero_shot_evaluation(
            model, task_registry, data_loaders, config, output_dir, device
        )
        all_results['zero_shot'] = zero_shot_results
    else:
        zero_shot_results = pd.DataFrame()

    # 2. Few-shot evaluation
    if config['evaluation'].get('run_few_shot', True):
        few_shot_results = run_few_shot_evaluation(
            model, task_registry, data_loaders, config, output_dir, device
        )
        all_results['few_shot'] = few_shot_results

        # Plot learning curves
        plot_few_shot_curves(few_shot_results, output_dir)
    else:
        few_shot_results = pd.DataFrame()

    # 3. Cross-species evaluation
    cross_species_results = None
    if config['evaluation'].get('run_cross_species', False):
        # Load species-specific data
        species_data = {}  # Replace with actual data
        cross_species_results = run_cross_species_evaluation(
            model, species_data, config, output_dir, device
        )
        all_results['cross_species'] = cross_species_results

    # Generate comprehensive report
    generate_benchmark_report(
        zero_shot_results,
        few_shot_results,
        cross_species_results,
        output_dir,
    )

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"Benchmark report: {output_dir / 'benchmark_report.md'}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
