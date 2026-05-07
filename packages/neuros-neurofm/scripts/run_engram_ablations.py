#!/usr/bin/env python
"""
ENGRAM-FMx Ablation Runner.

Runs all ablation experiments and generates a summary comparison.

Usage:
    python scripts/run_engram_ablations.py
    python scripts/run_engram_ablations.py --configs configs/engram_fmx/ablations/
    python scripts/run_engram_ablations.py --tasks associative_recall delayed_copy
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuros_neurofm.training.train_engram_synthetic import train, TrainingConfig

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


ABLATION_CONFIGS = {
    "full_model": {
        "use_local_processing": True,
        "use_ssm": True,
        "use_latent_workspace": True,
        "use_attractor_memory": True,
        "use_operator_dynamics": True,
        "use_sparse_anchor_attention": True,
    },
    "no_memory": {
        "use_local_processing": True,
        "use_ssm": True,
        "use_latent_workspace": True,
        "use_attractor_memory": False,
        "use_operator_dynamics": True,
        "use_sparse_anchor_attention": True,
    },
    "no_operator": {
        "use_local_processing": True,
        "use_ssm": True,
        "use_latent_workspace": True,
        "use_attractor_memory": True,
        "use_operator_dynamics": False,
        "use_sparse_anchor_attention": True,
    },
    "no_sparse_attention": {
        "use_local_processing": True,
        "use_ssm": True,
        "use_latent_workspace": True,
        "use_attractor_memory": True,
        "use_operator_dynamics": True,
        "use_sparse_anchor_attention": False,
    },
    "no_workspace": {
        "use_local_processing": True,
        "use_ssm": True,
        "use_latent_workspace": False,
        "use_attractor_memory": False,  # Requires workspace
        "use_operator_dynamics": False,  # Requires workspace
        "use_sparse_anchor_attention": False,  # Requires workspace
    },
    "ssm_only": {
        "use_local_processing": True,
        "use_ssm": True,
        "use_latent_workspace": False,
        "use_attractor_memory": False,
        "use_operator_dynamics": False,
        "use_sparse_anchor_attention": False,
    },
}


def run_single_ablation(
    ablation_name: str,
    ablation_config: Dict[str, bool],
    task: str,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    print(f"\n{'='*60}")
    print(f"Running ablation: {ablation_name} on task: {task}")
    print(f"{'='*60}")

    # Create config
    config = TrainingConfig(
        task=task,
        experiment_name=f"{task}_{ablation_name}",
        **base_config,
        **ablation_config,
    )

    # Train
    results = train(config)

    return {
        "ablation": ablation_name,
        "task": task,
        "final_train_loss": results["final_train_loss"],
        "final_val_loss": results["final_val_loss"],
        "best_val_loss": results["best_val_loss"],
        "num_params": results["num_params"],
        **ablation_config,
    }


def run_all_ablations(
    tasks: List[str],
    ablations: List[str],
    base_config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run all ablation experiments."""
    all_results = {}

    for task in tasks:
        task_results = []

        for ablation_name in ablations:
            if ablation_name not in ABLATION_CONFIGS:
                print(f"Warning: Unknown ablation {ablation_name}, skipping")
                continue

            result = run_single_ablation(
                ablation_name=ablation_name,
                ablation_config=ABLATION_CONFIGS[ablation_name],
                task=task,
                base_config=base_config,
            )
            task_results.append(result)

        all_results[task] = task_results

    return all_results


def generate_summary(results: Dict[str, List[Dict[str, Any]]], output_path: str):
    """Generate summary comparison."""
    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)

    for task, task_results in results.items():
        print(f"\n### Task: {task}")
        print("-" * 60)
        print(f"{'Ablation':<25} {'Val Loss':<12} {'Params':<12}")
        print("-" * 60)

        # Sort by val loss
        sorted_results = sorted(task_results, key=lambda x: x["best_val_loss"])

        for r in sorted_results:
            print(f"{r['ablation']:<25} {r['best_val_loss']:<12.4f} {r['num_params']:<12,}")

        # Compute relative differences
        if sorted_results:
            best = sorted_results[0]
            print(f"\nBest: {best['ablation']} (val_loss={best['best_val_loss']:.4f})")

            if len(sorted_results) > 1:
                print("\nRelative to best:")
                for r in sorted_results[1:]:
                    diff = ((r["best_val_loss"] - best["best_val_loss"]) / best["best_val_loss"]) * 100
                    print(f"  {r['ablation']}: +{diff:.1f}%")

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ENGRAM-FMx ablation experiments")

    parser.add_argument("--tasks", nargs="+", default=["associative_recall"],
                        choices=["associative_recall", "delayed_copy", "neural_dynamics"],
                        help="Tasks to run ablations on")
    parser.add_argument("--ablations", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                        help="Ablations to run")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max training steps per ablation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_size", type=str, default="tiny")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="runs/engram_fmx/ablations")

    args = parser.parse_args()

    # Base config (shared across ablations)
    base_config = {
        "model_size": args.model_size,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "output_dir": args.output_dir,
        "num_train_samples": 5000,
        "num_val_samples": 500,
        "log_interval": 100,
        "eval_interval": 100,
        "save_interval": 1000,  # Don't save intermediate checkpoints
    }

    # Run ablations
    results = run_all_ablations(
        tasks=args.tasks,
        ablations=args.ablations,
        base_config=base_config,
        output_dir=args.output_dir,
    )

    # Generate summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"ablation_summary_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_summary(results, str(output_path))


if __name__ == "__main__":
    main()
