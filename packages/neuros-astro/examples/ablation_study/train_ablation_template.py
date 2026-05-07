#!/usr/bin/env python
"""
Training script for allen_astro_ablation ablation study.

This script trains both baseline and test conditions and tracks results.

Usage:
    # Train baseline (neural only)
    python train_ablation.py --condition neural_only

    # Train test (neural + astro)
    python train_ablation.py --condition neural_astro

    # Train both
    python train_ablation.py --condition all
"""

import argparse
from pathlib import Path
from neuros_astro.experiments.tracker import ExperimentTracker, ExperimentResult
from neuros_astro.experiments.ablation import AblationStudy

# TODO: Import your neuroFMx training code
# from neuros_neurofm.train import train_model


def train_condition(condition_name, study_dir):
    """Train a single ablation condition."""

    # Load study
    study = AblationStudy.load(study_dir / "ablation_summary.json")
    condition = study.conditions[condition_name]

    print(f"Training condition: {condition_name}")
    print(f"Modalities: {condition.modalities}")

    # TODO: Implement your training logic here
    # Example:
    # model = train_model(
    #     config=condition.config,
    #     modalities=condition.modalities,
    # )
    # metrics = evaluate_model(model)

    # Mock results for demonstration
    # Replace with actual training results
    if condition_name == "neural_only":
        mock_metrics = {
            'prediction_loss': 0.250,
            'decoding_accuracy': 0.720,
            'cross_session_transfer': 0.650,
            'r2_score': 0.680,
        }
    else:  # neural_astro
        mock_metrics = {
            'prediction_loss': 0.185,  # Lower is better
            'decoding_accuracy': 0.830,  # Higher is better
            'cross_session_transfer': 0.780,  # Higher is better
            'r2_score': 0.810,  # Higher is better
        }

    # Create result
    result = ExperimentResult(
        experiment_id=condition.config.experiment_id,
        config=condition.config,
        model_metrics=mock_metrics,
        processing_time_s=1200.0,  # Replace with actual time
    )

    # Save result to study
    study.set_result(condition_name, result)
    study.save_summary()

    print(f"  ✓ Training complete for {condition_name}")
    print(f"  ✓ Results saved")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True,
                       choices=['neural_only', 'neural_astro', 'all'],
                       help='Condition to train')
    parser.add_argument('--study-dir', type=str, default='ablation_study',
                       help='Ablation study directory')

    args = parser.parse_args()

    study_dir = Path(args.study_dir)

    if args.condition == 'all':
        conditions = ['neural_only', 'neural_astro']
    else:
        conditions = [args.condition]

    for condition in conditions:
        train_condition(condition, study_dir)

    # Generate comparison report
    study = AblationStudy.load(study_dir / "ablation_summary.json")

    if study.conditions['neural_only'].result and study.conditions['neural_astro'].result:
        print("\n" + "=" * 80)
        print("ABLATION COMPARISON")
        print("=" * 80)

        for metric in ['prediction_loss', 'decoding_accuracy', 'cross_session_transfer']:
            comparison = study.compare_conditions(
                baseline_name='neural_only',
                test_name='neural_astro',
                metric=metric,
            )

            print(f"\n{metric}:")
            print(f"  Baseline: {comparison.baseline_value:.4f}")
            print(f"  Test: {comparison.test_value:.4f}")
            print(f"  Change: {comparison.percent_change:+.2f}%")
            print(f"  {comparison.interpretation}")

        print("\n" + study.generate_comparison_table())


if __name__ == "__main__":
    main()
