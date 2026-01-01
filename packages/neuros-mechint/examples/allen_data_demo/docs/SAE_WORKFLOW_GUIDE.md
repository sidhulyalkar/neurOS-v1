# SAE Training and Validation Workflow with Allen Data

This guide shows you how to use your 32 downloaded Allen sessions for SAE training and validation.

## 🎯 Complete Workflow

### Step 1: Run Multi-Session Validation (Currently Running)

```bash
conda activate mechint_playground

python examples/multi_session_validation.py \
    --allen-cache allen_validation_cache \
    --output multi_session_results_FULL.json
```

**What this does:**
- Analyzes all 32 Allen sessions for orientation tuning quality
- Computes orientation vs direction selectivity
- Identifies sessions with strong orientation-selective neurons
- Generates comprehensive JSON report

**Expected results:**
- ~25-30 successful sessions (some may lack V1 units or drifting gratings)
- Top sessions will have 35-50% orientation-selective units
- Max correlations of 0.7-0.85 for the best sessions

---

### Step 2: Analyze Results and Select Best Sessions

Once the validation completes, run:

```bash
python scripts/analyze_best_sessions.py \
    --results multi_session_results_FULL.json \
    --top-n 10 \
    --output-dir session_analysis
```

**What this does:**
- Ranks sessions by orientation selectivity and quality metrics
- Generates visualizations comparing sessions
- Creates `recommended_sessions.json` with top sessions
- Saves full analysis to CSV

**Outputs:**
- `session_analysis/recommended_sessions.json` - Top session IDs
- `session_analysis/all_sessions_analysis.csv` - Full spreadsheet
- `session_analysis/session_quality_scatter.png` - Quality visualization
- `session_analysis/orientation_vs_direction.png` - Selectivity comparison
- `session_analysis/selectivity_distribution.png` - Distribution plot

---

### Step 3: Train SAE on Top Sessions

#### Option A: Train on Single Best Session

```bash
python examples/sae_training_top_sessions.py \
    --session-config session_analysis/recommended_sessions.json \
    --allen-cache allen_validation_cache \
    --sae-dim 256 \
    --sparsity 0.01 \
    --epochs 100 \
    --output-dir sae_models
```

#### Option B: Train on Top 5 Sessions

```bash
python examples/sae_training_top_sessions.py \
    --session-config session_analysis/recommended_sessions.json \
    --allen-cache allen_validation_cache \
    --use-top-n 5 \
    --sae-dim 256 \
    --sparsity 0.01 \
    --epochs 100 \
    --output-dir sae_models
```

**What this does:**
- Loads neural data from recommended sessions
- Trains sparse autoencoder on neural activity
- Validates that SAE features discover orientation selectivity
- Saves trained models and results

**Outputs:**
- `sae_models/sae_session_<ID>.pt` - Trained SAE weights
- `sae_models/training_results.json` - Validation metrics

---

### Step 4: Use SAEs for Mechanistic Interpretability

Once you have trained SAEs, you can use them with the mechint package:

```python
from neuros.datasets.allen_datasets import AllenVisualCodingValidator
from neuros_mechint.multimodal_sae_analysis import MultiModalSAEAnalyzer
import torch
import json

# Load recommended session
with open('session_analysis/recommended_sessions.json', 'r') as f:
    config = json.load(f)

best_session_id = config['recommended_sessions']['best_overall']

# Load Allen data
validator = AllenVisualCodingValidator(
    session_id=best_session_id,
    cache_dir='allen_validation_cache',
    brain_areas=['VISp'],
    use_all_units=True
)

windows = validator.get_neural_windows()
labels = validator.get_task_labels()

# Load trained SAE
from examples.sae_training_top_sessions import SimpleSAE
sae = SimpleSAE(input_dim=..., hidden_dim=256)
sae.load_state_dict(torch.load(f'sae_models/sae_session_{best_session_id}.pt'))

# Extract SAE features
X = np.array([w.data.mean(axis=0) for w in windows])
X_tensor = torch.FloatTensor(X)
sae_features = sae.encode(X_tensor).numpy()

# Analyze feature selectivity
analyzer = MultiModalSAEAnalyzer()
results = analyzer.analyze_orientation_features(
    activations=sae_features,
    orientations=labels['orientation'][:len(sae_features)],
    return_controls=True
)

# Visualize results
fig = analyzer.visualize_feature_analysis(
    results=results,
    analysis_type='orientation',
    save_path='sae_orientation_selectivity.png'
)

print(f"Found {results['n_significant']} orientation-selective SAE features")
print(f"Max correlation: {results['max_correlation']:.3f}")
```

---

## 📊 Expected Results

### Multi-Session Validation
Based on your first 5 sessions:
- **30-42%** of neurons are orientation-selective (excellent!)
- **Max correlations of 0.68-0.80** (strong tuning)
- **All units** perform better than "good quality" only

### SAE Training
- SAE features should discover similar orientation selectivity
- Expect **20-40%** of SAE features to be orientation-selective
- This validates that SAEs learn meaningful neural representations

---

## 🎯 Selection Criteria for Best Sessions

The analysis script ranks sessions by:

1. **Fraction of selective units** (>30% is good, >40% is excellent)
2. **Max correlation strength** (>0.6 required, >0.7 preferred)
3. **Number of units** (more is better for robust statistics)
4. **Orientation/Direction ratio** (higher = more orientation-selective)

**Recommended for SAE training:**
- Use sessions with >35% selective units
- Prefer sessions with >50 units for robustness
- Top 5 sessions will give you diverse, high-quality data

---

## 🔧 Customization Options

### Multi-Session Validation
```bash
--max-sessions N          # Process only first N sessions (for testing)
--use-all-units          # Use all units (default: True)
--output PATH            # Custom output path
```

### Session Analysis
```bash
--top-n N               # Select top N sessions (default: 10)
--min-units N           # Minimum units per session (default: 20)
--output-dir PATH       # Custom output directory
```

### SAE Training
```bash
--use-top-n N           # Train on top N sessions (default: 1)
--sae-dim DIM           # SAE hidden dimension (default: 128)
--sparsity LAMBDA       # Sparsity penalty (default: 0.01)
--epochs N              # Training epochs (default: 50)
--lr RATE               # Learning rate (default: 0.001)
```

---

## 🚨 Troubleshooting

### Issue: "No VISp units found"
**Solution:** Some sessions may not have V1 (VISp) recordings. The script automatically skips these.

### Issue: "Too few units"
**Solution:** By default, sessions with <10 units are skipped. Adjust with `--min-units` if needed.

### Issue: "No drifting gratings stimulus"
**Solution:** Some sessions may not include drifting gratings. They're automatically skipped.

### Issue: SAE features have low selectivity
**Solutions:**
1. Increase `--sae-dim` (try 256 or 512)
2. Adjust `--sparsity` (try 0.005 or 0.02)
3. Train longer (`--epochs 200`)
4. Use top 5 sessions instead of just 1

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `multi_session_validation.py` | Validate all Allen sessions |
| `analyze_best_sessions.py` | Analyze results, select top sessions |
| `sae_training_top_sessions.py` | Train SAEs on recommended sessions |
| `recommended_sessions.json` | Top session IDs for easy use |
| `all_sessions_analysis.csv` | Full spreadsheet of all sessions |
| `sae_models/sae_session_*.pt` | Trained SAE weights |

---

## 🎉 Success Metrics

Your SAE validation is successful if:

✅ **Raw neurons** show 30-40% orientation-selective (matches literature)
✅ **SAE features** show 20-40% orientation-selective (SAE learns it!)
✅ **Max correlations** are >0.6 for both (strong tuning)
✅ **Above shuffle baseline** (statistical significance)

This demonstrates that your SAE framework can:
1. Learn interpretable features from neural data
2. Discover known neural coding properties (orientation)
3. Generalize across multiple recording sessions

Perfect foundation for mechanistic interpretability research! 🚀
