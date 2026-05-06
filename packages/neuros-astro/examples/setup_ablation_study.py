#!/usr/bin/env python
"""
Setup ablation study framework for neural vs neural+astro comparison.

Creates experimental configurations and tracking infrastructure for
systematic ablation experiments comparing:
- Baseline: Neural data only
- Test: Neural + Astrocyte data

Usage:
    python setup_ablation_study.py
    python setup_ablation_study.py --study-name allen_astro_ablation
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from neuros_astro.experiments.ablation import AblationStudy, AblationCondition
from neuros_astro.experiments.tracker import ExperimentTracker, ExperimentConfig


def setup_ablation_framework(study_name, output_dir, results_dir):
    """
    Setup complete ablation study framework.

    Creates:
    - Ablation study configuration
    - Experiment tracker
    - Baseline and test condition configs
    - Ready-to-use templates
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"SETTING UP ABLATION STUDY: {study_name}")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # 1. Create Ablation Study
    # -------------------------------------------------------------------------
    print("[1/5] Creating ablation study structure...")

    study = AblationStudy(
        study_name=study_name,
        output_dir=output_dir,
    )

    print(f"  ✓ Study initialized: {study_name}")
    print(f"  ✓ Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # 2. Load Session Information
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading session information...")

    results_dir = Path(results_dir)
    summary_path = results_dir / "overall_summary.json"

    with open(summary_path) as f:
        overall_summary = json.load(f)

    session_ids = [s['session_id'] for s in overall_summary['sessions']]
    n_sessions = len(session_ids)

    print(f"  ✓ Found {n_sessions} sessions")
    print(f"  ✓ Session IDs: {', '.join(session_ids[:3])}...")

    # -------------------------------------------------------------------------
    # 3. Create Baseline Condition (Neural Only)
    # -------------------------------------------------------------------------
    print("\n[3/5] Creating baseline condition (neural-only)...")

    baseline_config = ExperimentConfig(
        experiment_id=f"{study_name}_baseline_neural",
        experiment_name="Baseline: Neural Data Only",
        description="Baseline model trained on neural data without astrocyte signals",
        dataset_name="allen_visual_coding",
        session_ids=session_ids,
        modalities_enabled=['neural'],
        frame_rate_hz=30.0,
        random_seed=42,
        model_architecture='neurofmx',
        model_parameters={
            'hidden_dim': 512,
            'n_layers': 6,
            'n_heads': 8,
            'dropout': 0.1,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'n_epochs': 100,
            'early_stopping_patience': 10,
        },
    )

    study.add_condition(
        condition_name="neural_only",
        description="Baseline with neural data only (no astrocyte signals)",
        modalities=['neural'],
        config=baseline_config,
    )

    print("  ✓ Baseline condition created")
    print(f"    - Modalities: neural only")
    print(f"    - Sessions: {n_sessions}")

    # -------------------------------------------------------------------------
    # 4. Create Test Condition (Neural + Astro)
    # -------------------------------------------------------------------------
    print("\n[4/5] Creating test condition (neural + astrocyte)...")

    test_config = ExperimentConfig(
        experiment_id=f"{study_name}_test_neural_astro",
        experiment_name="Test: Neural + Astrocyte Data",
        description="Test model trained on neural data augmented with astrocyte signals",
        dataset_name="allen_visual_coding_multimodal",
        session_ids=session_ids,
        modalities_enabled=['neural', 'astro'],
        frame_rate_hz=30.0,
        random_seed=42,
        model_architecture='neurofmx_multimodal',
        model_parameters={
            'hidden_dim': 512,
            'n_layers': 6,
            'n_heads': 8,
            'dropout': 0.1,
            'astro_encoder_dim': 128,  # Additional for astro modality
            'batch_size': 32,
            'learning_rate': 1e-4,
            'n_epochs': 100,
            'early_stopping_patience': 10,
        },
    )

    study.add_condition(
        condition_name="neural_astro",
        description="Test with neural and astrocyte data",
        modalities=['neural', 'astro'],
        config=test_config,
    )

    print("  ✓ Test condition created")
    print(f"    - Modalities: neural + astro")
    print(f"    - Sessions: {n_sessions}")
    print(f"    - Astro tokens available: {overall_summary['total_events']} events")

    # -------------------------------------------------------------------------
    # 5. Generate Templates and Instructions
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating experiment templates...")

    # Save study configuration
    study.save_summary()

    # Create experiment tracker
    tracker_dir = output_dir / "experiment_tracking"
    tracker = ExperimentTracker(tracker_dir=tracker_dir)

    # Register experiments
    tracker.register_experiment(baseline_config)
    tracker.register_experiment(test_config)

    print(f"  ✓ Experiment tracker created: {tracker_dir}")

    # Create training script template
    training_template = f"""#!/usr/bin/env python
\"\"\"
Training script for {study_name} ablation study.

This script trains both baseline and test conditions and tracks results.

Usage:
    # Train baseline (neural only)
    python train_ablation.py --condition neural_only

    # Train test (neural + astro)
    python train_ablation.py --condition neural_astro

    # Train both
    python train_ablation.py --condition all
\"\"\"

import argparse
from pathlib import Path
from neuros_astro.experiments.tracker import ExperimentTracker, ExperimentResult
from neuros_astro.experiments.ablation import AblationStudy

# TODO: Import your neuroFMx training code
# from neuros_neurofm.train import train_model


def train_condition(condition_name, study_dir):
    \"\"\"Train a single ablation condition.\"\"\"

    # Load study
    study = AblationStudy.load(study_dir / "ablation_summary.json")
    condition = study.conditions[condition_name]

    print(f"Training condition: {{condition_name}}")
    print(f"Modalities: {{condition.modalities}}")

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
        mock_metrics = {{
            'prediction_loss': 0.250,
            'decoding_accuracy': 0.720,
            'cross_session_transfer': 0.650,
            'r2_score': 0.680,
        }}
    else:  # neural_astro
        mock_metrics = {{
            'prediction_loss': 0.185,  # Lower is better
            'decoding_accuracy': 0.830,  # Higher is better
            'cross_session_transfer': 0.780,  # Higher is better
            'r2_score': 0.810,  # Higher is better
        }}

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

    print(f"  ✓ Training complete for {{condition_name}}")
    print(f"  ✓ Results saved")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True,
                       choices=['neural_only', 'neural_astro', 'all'],
                       help='Condition to train')
    parser.add_argument('--study-dir', type=str, default='{output_dir}',
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
        print("\\n" + "=" * 80)
        print("ABLATION COMPARISON")
        print("=" * 80)

        for metric in ['prediction_loss', 'decoding_accuracy', 'cross_session_transfer']:
            comparison = study.compare_conditions(
                baseline_name='neural_only',
                test_name='neural_astro',
                metric=metric,
            )

            print(f"\\n{{metric}}:")
            print(f"  Baseline: {{comparison.baseline_value:.4f}}")
            print(f"  Test: {{comparison.test_value:.4f}}")
            print(f"  Change: {{comparison.percent_change:+.2f}}%")
            print(f"  {{comparison.interpretation}}")

        print("\\n" + study.generate_comparison_table())


if __name__ == "__main__":
    main()
"""

    template_path = output_dir / "train_ablation_template.py"
    with open(template_path, 'w') as f:
        f.write(training_template)

    print(f"  ✓ Training template: {template_path.name}")

    # Create README with instructions
    readme = f"""# Ablation Study: {study_name}

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sessions:** {n_sessions}
**Status:** Ready for training

---

## Overview

This ablation study compares:
- **Baseline:** Neural data only
- **Test:** Neural + Astrocyte data

**Goal:** Quantify the contribution of astrocyte signals to model performance.

---

## Files Created

```
{output_dir}/
├── ablation_summary.json          # Study configuration
├── train_ablation_template.py     # Training script template
├── experiment_tracking/            # Experiment tracker
│   ├── neural_only/               # Baseline experiments
│   └── neural_astro/              # Test experiments
└── README.md                       # This file
```

---

## Quick Start

### 1. Train Baseline (Neural Only)

```bash
# TODO: Implement your training code in train_ablation_template.py
python train_ablation_template.py --condition neural_only
```

Expected to train on {n_sessions} sessions with neural data only.

### 2. Train Test (Neural + Astro)

```bash
python train_ablation_template.py --condition neural_astro
```

Will load astrocyte tokens from:
- `{results_dir}/session_*/astro_tokens.npz`
- Total events available: {overall_summary['total_events']}

### 3. Compare Results

```bash
python train_ablation_template.py --condition all
```

Automatically generates comparison table and statistics.

---

## Integration with neuroFMx

### Step 1: Create Multimodal Dataset Loader

Location: `packages/neuros-neurofm/src/neuros_neurofm/datasets/allen_multimodal_dataset.py`

```python
class AllenMultimodalDataset:
    def __init__(self, session_ids, modalities=['neural', 'astro']):
        self.session_ids = session_ids
        self.modalities = modalities

        # Load neural data (existing)
        self.neural_data = load_neural_data(session_ids)

        # Load astro tokens (new)
        if 'astro' in modalities:
            self.astro_tokens = load_astro_tokens(session_ids)

    def __getitem__(self, idx):
        batch = {{}}

        # Neural modality
        batch['neural'] = self.neural_data[idx]

        # Astro modality (if enabled)
        if 'astro' in self.modalities:
            batch['astro'] = self.astro_tokens[idx]

        return batch
```

### Step 2: Update Training Loop

```python
# In your training script
if 'astro' in config.modalities_enabled:
    # Use multimodal model
    model = MultimodalNeuroFMx(
        neural_dim=512,
        astro_dim=128,
        fusion_strategy='cross_attention'
    )
else:
    # Use baseline model
    model = NeuroFMx(hidden_dim=512)
```

---

## Metrics to Track

### Primary Metrics (Lower is Better)
- `prediction_loss`: Cross-entropy or MSE loss

### Primary Metrics (Higher is Better)
- `decoding_accuracy`: Stimulus decoding accuracy
- `cross_session_transfer`: Generalization to held-out sessions
- `r2_score`: Explained variance

### Secondary Metrics
- `training_time_s`: Computational cost
- `model_size_mb`: Memory footprint
- `inference_latency_ms`: Speed

---

## Expected Results

Based on astrocyte literature, we expect:
- **Improved prediction:** 5-15% reduction in loss
- **Better decoding:** 5-10% increase in accuracy
- **Enhanced generalization:** Improved cross-session transfer

---

## Analysis Pipeline

Once training is complete:

```bash
# Generate comprehensive comparison
python -c "
from neuros_astro.experiments.ablation import AblationStudy

study = AblationStudy.load('{output_dir}/ablation_summary.json')

# Compare all metrics
for metric in ['prediction_loss', 'decoding_accuracy', 'cross_session_transfer']:
    comparison = study.compare_conditions(
        baseline_name='neural_only',
        test_name='neural_astro',
        metric=metric
    )
    print(f'{{metric}}: {{comparison.percent_change:+.2f}}%')

# Generate publication table
print(study.generate_comparison_table())
"
```

---

## Publication Figures

Generate ablation figures:

```python
from neuros_astro.visualization.publication_figures import generate_figure_4_ablation_results

ablation_data = {{
    'neural_only': study.conditions['neural_only'].result.model_metrics,
    'neural_astro': study.conditions['neural_astro'].result.model_metrics,
}}

generate_figure_4_ablation_results(
    ablation_data,
    save_path='./figures/ablation_results.png'
)
```

---

## Next Steps

1. ✅ **DONE:** Ablation framework setup
2. 📝 **TODO:** Implement multimodal dataset loader
3. 📝 **TODO:** Train baseline (neural-only) model
4. 📝 **TODO:** Train test (neural+astro) model
5. 📝 **TODO:** Generate ablation comparison figures
6. 📝 **TODO:** Write Methods section for manuscript

---

## Questions?

See documentation:
- `packages/neuros-astro/ADVANCED_FEATURES.md`
- `packages/neuros-astro/examples/ALLEN_NWB_PROCESSING_SUMMARY.md`

---

**Ready to start training! 🚀**
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme)

    print(f"  ✓ Instructions: {readme_path.name}")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ABLATION STUDY SETUP COMPLETE!")
    print("=" * 80)
    print(f"\nStudy: {study_name}")
    print(f"Output: {output_dir}")
    print("\nConditions created:")
    print("  ✓ neural_only (baseline)")
    print("  ✓ neural_astro (test)")
    print(f"\nData available:")
    print(f"  ✓ {n_sessions} sessions")
    print(f"  ✓ {overall_summary['total_events']} astrocyte events")
    print(f"  ✓ {overall_summary['total_networks']} functional networks")
    print("\nNext steps:")
    print(f"  1. Review {readme_path.name}")
    print(f"  2. Implement training in {template_path.name}")
    print("  3. Run experiments")
    print("  4. Generate comparison figures")
    print()

    return study


def main():
    parser = argparse.ArgumentParser(
        description="Setup ablation study framework"
    )
    parser.add_argument('--study-name', type=str, default='allen_astro_ablation',
                       help='Name for the ablation study')
    parser.add_argument('--output-dir', type=str, default='./ablation_study',
                       help='Output directory')
    parser.add_argument('--results-dir', type=str, default='./allen_nwb_results',
                       help='Directory containing Allen processing results')

    args = parser.parse_args()

    study = setup_ablation_framework(
        study_name=args.study_name,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
