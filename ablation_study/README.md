# Ablation Study: allen_astro_ablation

**Created:** 2026-05-05 21:17:42
**Sessions:** 10
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
ablation_study/
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

Expected to train on 10 sessions with neural data only.

### 2. Train Test (Neural + Astro)

```bash
python train_ablation_template.py --condition neural_astro
```

Will load astrocyte tokens from:
- `allen_nwb_results/session_*/astro_tokens.npz`
- Total events available: 9153

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
        batch = {}

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

study = AblationStudy.load('ablation_study/ablation_summary.json')

# Compare all metrics
for metric in ['prediction_loss', 'decoding_accuracy', 'cross_session_transfer']:
    comparison = study.compare_conditions(
        baseline_name='neural_only',
        test_name='neural_astro',
        metric=metric
    )
    print(f'{metric}: {comparison.percent_change:+.2f}%')

# Generate publication table
print(study.generate_comparison_table())
"
```

---

## Publication Figures

Generate ablation figures:

```python
from neuros_astro.visualization.publication_figures import generate_figure_4_ablation_results

ablation_data = {
    'neural_only': study.conditions['neural_only'].result.model_metrics,
    'neural_astro': study.conditions['neural_astro'].result.model_metrics,
}

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
