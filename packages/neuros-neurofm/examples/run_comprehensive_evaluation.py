"""
Comprehensive Evaluation Example for NeuroFMX.

Demonstrates how to use the task registry, zero-shot evaluation, and few-shot
evaluation systems to benchmark a pretrained NeuroFMX model.

Usage:
    python run_comprehensive_evaluation.py --model_path checkpoints/neurofmx_pretrained.pt \
                                           --config configs/eval/eval_tasks.yaml \
                                           --output_dir results/evaluation/
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from neuros_neurofm.models.neurofmx import NeuroFMX
from neuros_neurofm.evaluation import (
    # Task Registry
    TaskRegistry,
    get_global_registry,
    TaskType,
    Species,
    Modality,
    # Zero-Shot Evaluation
    ZeroShotEvaluator,
    ZeroShotConfig,
    run_zero_shot_suite,
    # Few-Shot Evaluation
    FewShotEvaluator,
    FewShotConfig,
    run_few_shot_suite,
)


def load_model(model_path: str, device: str = "cuda") -> nn.Module:
    """Load pretrained NeuroFMX model.

    Parameters
    ----------
    model_path : str
        Path to model checkpoint.
    device : str
        Device to load model on.

    Returns
    -------
    model : nn.Module
        Loaded model.
    """
    print(f"Loading model from {model_path}...")

    # Initialize model (adjust hyperparameters as needed)
    model = NeuroFMX(
        d_model=768,
        n_mamba_blocks=16,
        n_latents=128,
        latent_dim=512,
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model loaded successfully!")

    return model


def setup_task_registry(config_path: str) -> TaskRegistry:
    """Setup task registry from YAML config.

    Parameters
    ----------
    config_path : str
        Path to eval_tasks.yaml.

    Returns
    -------
    registry : TaskRegistry
        Loaded task registry.
    """
    print(f"Loading task registry from {config_path}...")
    registry = TaskRegistry(config_path=config_path)
    print(f"Loaded {len(registry)} evaluation tasks")

    # Print summary
    print("\nTask Summary:")
    print(f"  Species: {set(t.species.value for t in registry._metadata_cache.values())}")
    print(f"  Modalities: {set(t.modality.value for t in registry._metadata_cache.values())}")
    print(f"  Task Types: {set(t.task_type.value for t in registry._metadata_cache.values())}")

    return registry


def run_zero_shot_benchmark(
    model: nn.Module,
    registry: TaskRegistry,
    output_dir: Path,
    task_filter: Optional[dict] = None,
):
    """Run zero-shot evaluation benchmark.

    Parameters
    ----------
    model : nn.Module
        Pretrained model.
    registry : TaskRegistry
        Task registry.
    output_dir : Path
        Output directory.
    task_filter : dict, optional
        Filter tasks by species, modality, etc.
    """
    print("\n" + "="*80)
    print("ZERO-SHOT EVALUATION")
    print("="*80)

    # Configure zero-shot evaluation
    config = ZeroShotConfig(
        probe_lr=1e-3,
        probe_epochs=100,
        probe_batch_size=128,
        early_stopping_patience=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Get tasks to evaluate
    if task_filter:
        task_names = registry.list_tasks(**task_filter)
    else:
        task_names = list(registry._tasks.keys())

    tasks = [registry.get(name) for name in task_names]

    print(f"Evaluating {len(tasks)} tasks in zero-shot setting...")

    # Run evaluation
    zero_shot_dir = output_dir / "zero_shot"
    results = run_zero_shot_suite(
        model=model,
        tasks=tasks,
        config=config,
        output_dir=zero_shot_dir,
    )

    # Print summary
    print("\nZero-Shot Results Summary:")
    print("-" * 80)
    for task_name, result in results.items():
        if 'error' in result:
            print(f"  {task_name}: ERROR - {result['error']}")
        else:
            metric_name = result['metric_name']
            metric_value = result['test_metric']
            print(f"  {task_name}: {metric_name} = {metric_value:.4f}")

    return results


def run_few_shot_benchmark(
    model: nn.Module,
    registry: TaskRegistry,
    output_dir: Path,
    task_filter: Optional[dict] = None,
    k_shots: Optional[List[int]] = None,
):
    """Run few-shot evaluation benchmark.

    Parameters
    ----------
    model : nn.Module
        Pretrained model.
    registry : TaskRegistry
        Task registry.
    output_dir : Path
        Output directory.
    task_filter : dict, optional
        Filter tasks by species, modality, etc.
    k_shots : list, optional
        K-shot values to evaluate.
    """
    print("\n" + "="*80)
    print("FEW-SHOT EVALUATION")
    print("="*80)

    # Configure few-shot evaluation
    config = FewShotConfig(
        k_shots=k_shots or [1, 5, 10, 25, 50],
        n_episodes=100,
        n_query_samples=100,
        adaptation_method="lora",
        lora_rank=8,
        lora_alpha=16.0,
        adaptation_lr=1e-4,
        adaptation_steps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Get tasks to evaluate
    if task_filter:
        task_names = registry.list_tasks(**task_filter)
    else:
        task_names = list(registry._tasks.keys())

    tasks = [registry.get(name) for name in task_names]

    print(f"Evaluating {len(tasks)} tasks with K-shot learning...")
    print(f"K values: {config.k_shots}")
    print(f"Episodes per K: {config.n_episodes}")

    # Run evaluation
    few_shot_dir = output_dir / "few_shot"
    results = run_few_shot_suite(
        model=model,
        tasks=tasks,
        config=config,
        output_dir=few_shot_dir,
    )

    # Print summary
    print("\nFew-Shot Results Summary:")
    print("-" * 80)
    for task_name, result in results.items():
        if 'error' in result:
            print(f"  {task_name}: ERROR - {result['error']}")
        else:
            print(f"  {task_name}:")
            for k in config.k_shots:
                if k in result['k_shot_results']:
                    k_results = result['k_shot_results'][k]
                    # Get primary metric
                    metric_name = list(k_results.keys())[0]
                    metric_stats = k_results[metric_name]
                    print(f"    K={k}: {metric_name} = {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}")

    return results


def run_targeted_evaluation(
    model: nn.Module,
    registry: TaskRegistry,
    output_dir: Path,
):
    """Run targeted evaluation on specific task categories.

    Parameters
    ----------
    model : nn.Module
        Pretrained model.
    registry : TaskRegistry
        Task registry.
    output_dir : Path
        Output directory.
    """
    print("\n" + "="*80)
    print("TARGETED EVALUATIONS")
    print("="*80)

    # 1. Cross-species transfer tasks
    print("\n1. Cross-Species Transfer Tasks")
    cross_species_tasks = registry.list_tasks(species=Species.MULTI_SPECIES)
    if cross_species_tasks:
        print(f"   Found {len(cross_species_tasks)} cross-species tasks")
        for task_name in cross_species_tasks:
            print(f"   - {task_name}")

    # 2. Motor decoding tasks
    print("\n2. Motor Decoding Tasks")
    motor_tasks = registry.list_tasks(tags=["motor", "bci"])
    if motor_tasks:
        print(f"   Found {len(motor_tasks)} motor tasks")
        for task_name in motor_tasks:
            print(f"   - {task_name}")

    # 3. Clinical applications
    print("\n3. Clinical Application Tasks")
    clinical_tasks = registry.list_tasks(tags=["clinical"])
    if clinical_tasks:
        print(f"   Found {len(clinical_tasks)} clinical tasks")
        for task_name in clinical_tasks:
            print(f"   - {task_name}")

    # 4. Encoding models
    print("\n4. Neural Encoding Tasks")
    encoding_tasks = registry.list_tasks(task_type=TaskType.ENCODING)
    if encoding_tasks:
        print(f"   Found {len(encoding_tasks)} encoding tasks")
        for task_name in encoding_tasks:
            print(f"   - {task_name}")


def generate_final_report(
    zero_shot_results: dict,
    few_shot_results: dict,
    output_dir: Path,
):
    """Generate comprehensive evaluation report.

    Parameters
    ----------
    zero_shot_results : dict
        Zero-shot evaluation results.
    few_shot_results : dict
        Few-shot evaluation results.
    output_dir : Path
        Output directory.
    """
    report_path = output_dir / "comprehensive_evaluation_report.md"

    with open(report_path, 'w') as f:
        f.write("# NeuroFMX Comprehensive Evaluation Report\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Tasks Evaluated**: {len(zero_shot_results)}\n")
        f.write(f"- **Zero-Shot Tasks**: {sum(1 for r in zero_shot_results.values() if 'error' not in r)}\n")
        f.write(f"- **Few-Shot Tasks**: {sum(1 for r in few_shot_results.values() if 'error' not in r)}\n\n")

        # Zero-Shot Results
        f.write("## Zero-Shot Evaluation Results\n\n")
        f.write("| Task | Species | Modality | Metric | Score |\n")
        f.write("|------|---------|----------|--------|-------|\n")

        for task_name, result in sorted(zero_shot_results.items()):
            if 'error' not in result:
                f.write(f"| {task_name} | {result['species']} | {result['modality']} | ")
                f.write(f"{result['metric_name']} | {result['test_metric']:.4f} |\n")

        # Few-Shot Results
        f.write("\n## Few-Shot Evaluation Results\n\n")
        f.write("| Task | 1-Shot | 5-Shot | 10-Shot | 25-Shot | 50-Shot |\n")
        f.write("|------|--------|--------|---------|---------|--------|\n")

        for task_name, result in sorted(few_shot_results.items()):
            if 'error' not in result and 'k_shot_results' in result:
                f.write(f"| {task_name} |")
                for k in [1, 5, 10, 25, 50]:
                    if k in result['k_shot_results']:
                        k_res = result['k_shot_results'][k]
                        metric_name = list(k_res.keys())[0]
                        mean = k_res[metric_name]['mean']
                        f.write(f" {mean:.4f} |")
                    else:
                        f.write(" - |")
                f.write("\n")

        f.write("\n## Conclusion\n\n")
        f.write("This comprehensive evaluation demonstrates NeuroFMX's capabilities across ")
        f.write("multiple species, modalities, and task types in both zero-shot and few-shot settings.\n")

    print(f"\nFinal report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation of NeuroFMX model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval/eval_tasks.yaml",
        help="Path to evaluation tasks config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="Run zero-shot evaluation",
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Run few-shot evaluation",
    )
    parser.add_argument(
        "--k_shots",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25, 50],
        help="K-shot values for few-shot evaluation",
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["mouse", "monkey", "human", "rat", "zebrafish", "multi_species"],
        help="Filter tasks by species",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["spikes", "lfp", "eeg", "ecog", "fmri", "calcium"],
        help="Filter tasks by modality",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path, device=args.device)

    # Setup task registry
    registry = setup_task_registry(args.config)

    # Build task filter
    task_filter = {}
    if args.species:
        task_filter['species'] = Species(args.species)
    if args.modality:
        task_filter['modality'] = Modality(args.modality)

    # Run targeted evaluation overview
    run_targeted_evaluation(model, registry, output_dir)

    # Run evaluations
    zero_shot_results = {}
    few_shot_results = {}

    if args.zero_shot or (not args.zero_shot and not args.few_shot):
        zero_shot_results = run_zero_shot_benchmark(
            model, registry, output_dir, task_filter
        )

    if args.few_shot or (not args.zero_shot and not args.few_shot):
        few_shot_results = run_few_shot_benchmark(
            model, registry, output_dir, task_filter, args.k_shots
        )

    # Generate final report
    if zero_shot_results or few_shot_results:
        generate_final_report(zero_shot_results, few_shot_results, output_dir)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
