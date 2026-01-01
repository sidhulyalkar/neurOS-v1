# SAE Analysis and Optimization Workflow

**Complete guide for analyzing, optimizing, and validating your SAE features**

After training your SAE, use these scripts to deeply understand its performance and optimize it for maximum interpretability.

---

## 📋 Quick Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `analyze_sae_features.py` | Compare SAE vs raw neurons | After initial training |
| `sae_hyperparameter_search.py` | Find optimal architecture | Before final training |
| `visualize_sae_features.py` | Understand what features learned | After training good SAE |

---

## 🎯 Workflow Overview

```
1. Train SAE (basic config)
         ↓
2. Analyze features vs raw neurons
         ↓
3. If results suboptimal → Hyperparameter search
         ↓
4. Retrain with optimal config
         ↓
5. Deep visualization and interpretation
         ↓
6. Publication figures
```

---

## Step 1: Analyze Initial SAE Results

**After running** `sae_training_top_sessions.py`, analyze how well your SAE features compare to raw neurons.

### Run Analysis

```bash
python scripts/analyze_sae_features.py \
    --sae-results sae_models/training_results.json \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir sae_analysis
```

### What This Does

1. **Loads your trained SAE** and extracts features
2. **Compares orientation selectivity**:
   - Raw neurons vs SAE features
   - Distribution of correlations
   - Number of selective features
3. **Analyzes sparsity**:
   - L0 sparsity (% zeros)
   - Lifetime sparsity (% active)
   - Population sparsity
4. **Generates comparison figures**:
   - `correlation_comparison.png` - Histogram and cumulative plot
   - `preferred_orientation_distribution.png` - Polar plots showing coverage
   - `sparsity_analysis.png` - Sparsity metrics and relationships

### Expected Output

```
SAE Features:
  Max correlation: 0.XXX
  Mean correlation: 0.XXX
  Selective features: XX/256 (XX.X%)

Feature Sparsity:
  L0 (% zeros): XX.X%
  Mean lifetime sparsity: XX.X%
  Mean population sparsity: XX.X%
```

### Interpretation Guide

**Orientation Selectivity:**
- ✅ **Good**: SAE features ≥ raw neurons selectivity
- ⚠️ **Warning**: SAE features < raw neurons (SAE may be underparameterized)
- 🎯 **Target**: 20-40% selective features

**Sparsity:**
- ✅ **Good**: 20-40% lifetime sparsity (features are specialized)
- ⚠️ **Warning**: >60% (too sparse, features underused)
- ⚠️ **Warning**: <10% (too dense, features not specialized)

**Key Metrics:**
```
Interpretation Chart:
┌─────────────────────┬──────────────┬─────────────────────┐
│ Metric              │ Good Range   │ Action if Outside   │
├─────────────────────┼──────────────┼─────────────────────┤
│ % Selective         │ 20-40%       │ Adjust hidden_dim   │
│ Lifetime Sparsity   │ 20-40%       │ Adjust sparsity λ   │
│ Reconstruction Loss │ <1.0         │ Increase epochs     │
│ Max Correlation     │ >0.6         │ Check data quality  │
└─────────────────────┴──────────────┴─────────────────────┘
```

---

## Step 2: Hyperparameter Search (Optional but Recommended)

If your initial results are suboptimal, run systematic hyperparameter search to find the best architecture.

### Quick Search (5-10 min)

```bash
python scripts/sae_hyperparameter_search.py \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --quick \
    --output-dir hyperparameter_search
```

Tests 8 configurations (2 hidden dims × 2 sparsity × 2 epochs).

### Full Search (1-2 hours)

```bash
python scripts/sae_hyperparameter_search.py \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir hyperparameter_search
```

Tests 96 configurations (4 hidden dims × 4 sparsity × 3 learning rates × 2 activations).

### What This Does

1. **Grid search** over hyperparameter space:
   - Hidden dimensions: [64, 128, 256, 512, 1024]
   - Sparsity penalty: [0.001, 0.005, 0.01, 0.02, 0.05]
   - Learning rates: [0.0001, 0.0005, 0.001, 0.005]
   - Activations: [ReLU, LeakyReLU, GELU]

2. **Evaluates each config** on:
   - Reconstruction loss
   - Orientation selectivity
   - Feature sparsity
   - Feature diversity (low inter-feature correlation)

3. **Computes composite score**:
   ```
   Score = selectivity - 0.5×recon_loss - 0.2×|sparsity-0.3| - 0.1×feature_corr
   ```

4. **Ranks configs** and saves top 10

### Expected Output

```
TOP 10 CONFIGURATIONS

[1] Score: 0.XXX
  Config: {'hidden_dim': 256, 'sparsity': 0.01, 'lr': 0.001, 'activation': 'relu'}
  Selectivity: XX.X% (max_corr: 0.XXX)
  Recon loss: 0.XXX
  Sparsity: XX.X%
  Feature diversity: 0.XXX
...

RECOMMENDED CONFIGURATION
Hidden dim: 256
Sparsity: 0.01
Learning rate: 0.001
Activation: relu
```

### Using Results

After search completes:
1. Check `hyperparameter_search_session_<ID>.json` for full results
2. Check `hyperparameter_search_session_<ID>.csv` for table view
3. Use top config for final training:

```bash
python examples/sae_training_top_sessions.py \
    --session-config recommended_sessions.json \
    --allen-cache allen_validation_cache \
    --sae-dim 256 \        # Use recommended hidden_dim
    --sparsity 0.01 \      # Use recommended sparsity
    --lr 0.001 \           # Use recommended lr
    --epochs 100
```

---

## Step 3: Deep Feature Visualization

Once you have a well-trained SAE, visualize what features it learned.

### Run Visualization

```bash
python scripts/visualize_sae_features.py \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --sae-model sae_models/sae_session_754829445.pt \
    --output-dir sae_visualizations
```

### What This Generates

**1. Tuning Curves** (`tuning_curves_session_<ID>.png`)
- Shows response vs orientation for top 6 features
- Reveals feature selectivity width and preferred orientation
- **Use for**: Understanding individual feature properties

**2. Activation Heatmap** (`activation_heatmap_session_<ID>.png`)
- 2D map of feature activations across samples
- Samples sorted by orientation
- Features sorted by selectivity
- **Use for**: Understanding population-level patterns

**3. Weight Visualization** (`weights_session_<ID>.png`)
- Shows encoder weights from neurons to features
- Positive (red) vs negative (blue) connections
- **Use for**: Understanding which neurons contribute to each feature

**4. Preferred Orientation Map** (`orientation_map_session_<ID>.png`)
- Polar plot showing distribution of preferred orientations
- Size and color encode correlation strength
- **Use for**: Checking uniform coverage of orientation space

**5. Feature Clustering** (`feature_clustering_session_<ID>.png`)
- PCA visualization of feature similarity
- Colored by K-means clusters and preferred orientation
- **Use for**: Understanding feature organization

### Interpretation Guide

**Tuning Curves:**
- ✅ **Sharp peak**: Feature is highly selective
- ✅ **Consistent preferred orientation**: Feature is reliable
- ⚠️ **Flat**: Feature is not orientation-selective (may encode other properties)
- ⚠️ **Multiple peaks**: Feature may be polysemantic

**Activation Heatmap:**
- ✅ **Vertical stripes**: Features activate for specific orientations
- ✅ **Diverse patterns**: Different features encode different information
- ⚠️ **All features active**: Sparsity too low
- ⚠️ **Few active features**: Hidden dim too small

**Weight Visualization:**
- ✅ **Sparse weights**: Feature reads from subset of neurons
- ✅ **Clear structure**: Feature has interpretable input selectivity
- ⚠️ **Uniform weights**: Feature averages over all neurons

**Orientation Map:**
- ✅ **Uniform coverage**: Features cover all orientations
- ⚠️ **Clustering**: Some orientations over-represented
- 🎯 **Target**: ~Equal number of features for each 45° bin

**Feature Clustering:**
- ✅ **Clear clusters**: Features organize by preferred orientation
- ✅ **Gradual color gradient**: Smooth transition between orientations
- ⚠️ **Random scatter**: Features not organized (increase hidden_dim)

---

## 🎯 Optimization Decision Tree

Use this flowchart to optimize your SAE:

```
Start: Analyze initial SAE
         ↓
   Selectivity < 20%?
    ├─ Yes → Increase hidden_dim (e.g., 128→256)
    └─ No ↓
   Selectivity << Raw neurons?
    ├─ Yes → Decrease sparsity λ (e.g., 0.02→0.01)
    └─ No ↓
   Sparsity > 60%?
    ├─ Yes → Decrease sparsity λ (e.g., 0.02→0.01)
    └─ No ↓
   Sparsity < 10%?
    ├─ Yes → Increase sparsity λ (e.g., 0.01→0.02)
    └─ No ↓
   Recon loss > 1.0?
    ├─ Yes → Increase epochs or decrease LR
    └─ No ↓
   All metrics good?
    └─ Yes → Use for validation! ✅
```

---

## 📊 Publication-Quality Figures

### Figure 1: Raw vs SAE Comparison
**File**: `sae_analysis/correlation_comparison.png`

**Use for**: Demonstrating that SAE features discover orientation selectivity

**Caption**:
> "Comparison of orientation selectivity between raw neurons and SAE features. (A) Distribution of circular correlations with orientation. (B) Cumulative distribution showing rank-ordered selectivity. SAE features achieve comparable selectivity to raw neurons (XX% vs XX% with correlation >0.3), demonstrating successful feature learning."

### Figure 2: Feature Tuning Curves
**File**: `sae_visualizations/tuning_curves_session_<ID>.png`

**Use for**: Showing that SAE features have interpretable tuning

**Caption**:
> "Orientation tuning curves for top 6 SAE features. Each feature shows selective response to specific orientations (red dashed line indicates preferred orientation). Tuning curves match expected profiles of V1 orientation-selective neurons."

### Figure 3: Orientation Coverage
**File**: `sae_visualizations/orientation_map_session_<ID>.png`

**Use for**: Demonstrating comprehensive orientation coverage

**Caption**:
> "Polar distribution of preferred orientations for significant SAE features (circular correlation >0.3). Size and color indicate correlation strength. Features uniformly cover orientation space, suggesting complete representation of visual input."

### Figure 4: Sparsity Analysis
**File**: `sae_analysis/sparsity_analysis.png`

**Use for**: Showing SAE learns sparse, interpretable representations

**Caption**:
> "Sparsity analysis of SAE features. (A,B) Distributions of lifetime and population sparsity. (C) Relationship between sparsity and orientation selectivity. (D) Summary statistics. Features achieve XX% lifetime sparsity, indicating specialized representations."

---

## 🔬 Advanced Analysis Tips

### 1. Cross-Session Generalization

Test if features generalize across sessions:

```bash
# Train on session 1
python examples/sae_training_top_sessions.py \
    --session-config recommended_sessions.json \
    --allen-cache allen_validation_cache

# Test on session 2
python scripts/analyze_sae_features.py \
    --sae-results sae_models/training_results.json \
    --session-id 760345702 \  # Different session!
    --allen-cache allen_validation_cache
```

**What to expect:**
- ✅ Features should still show orientation selectivity (though slightly reduced)
- ✅ Preferred orientations should remain stable
- ⚠️ If selectivity drops dramatically, features may be overfitting

### 2. Compare Different Hidden Dimensions

Train SAEs with different sizes and compare:

```bash
for dim in 64 128 256 512; do
    python examples/sae_training_top_sessions.py \
        --session-config recommended_sessions.json \
        --sae-dim $dim \
        --output-dir sae_models_dim${dim}
done

# Analyze each
for dim in 64 128 256 512; do
    python scripts/analyze_sae_features.py \
        --sae-results sae_models_dim${dim}/training_results.json \
        --session-id 754829445 \
        --output-dir sae_analysis_dim${dim}
done
```

**Plot results:**
```python
import pandas as pd
import matplotlib.pyplot as plt

results = {
    64: {...},
    128: {...},
    256: {...},
    512: {...}
}

dims = list(results.keys())
selectivity = [results[d]['fraction_selective'] for d in dims]
sparsity = [results[d]['lifetime_sparsity'] for d in dims]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(dims, selectivity, 'o-')
ax[0].set_xlabel('Hidden Dimension')
ax[0].set_ylabel('% Selective Features')
ax[1].plot(dims, sparsity, 'o-')
ax[1].set_xlabel('Hidden Dimension')
ax[1].set_ylabel('Lifetime Sparsity')
```

### 3. Ablation Studies

Test importance of different components:

```bash
# No sparsity penalty
python examples/sae_training_top_sessions.py \
    --sparsity 0.0 \
    --output-dir sae_models_no_sparsity

# Different activations
python examples/sae_training_top_sessions.py \
    --activation leaky_relu \
    --output-dir sae_models_leaky_relu
```

---

## 💡 Troubleshooting

### Problem: Low selectivity (<20%)

**Possible causes:**
1. Hidden dim too small → Increase to 256 or 512
2. Sparsity too high → Decrease λ from 0.02 to 0.01
3. Training too short → Increase epochs to 100+
4. Learning rate too high → Decrease to 0.0005

**Diagnostic:**
```bash
python scripts/visualize_sae_features.py ...
# Check tuning curves - if flat, features aren't specialized
```

### Problem: High sparsity (>60%)

**Cause**: Sparsity penalty too strong

**Fix**:
```bash
python examples/sae_training_top_sessions.py --sparsity 0.005  # Reduce from 0.01
```

### Problem: Features look random

**Possible causes:**
1. Bad initialization → Retrain with different seed
2. Poor data quality → Check raw neuron selectivity first
3. Wrong normalization → Check data preprocessing

**Diagnostic:**
```bash
python scripts/analyze_sae_features.py ...
# Compare with raw neurons - if raw neurons have low selectivity, data may be issue
```

### Problem: SAE worse than raw neurons

**Possible causes:**
1. Overconstrained (too much sparsity) → Reduce λ
2. Underparameterized (too few features) → Increase hidden_dim
3. Poor training → Increase epochs, tune LR

**Fix**: Run hyperparameter search
```bash
python scripts/sae_hyperparameter_search.py --session-id ... --quick
```

---

## ✅ Success Criteria Checklist

Before using your SAE for publication:

- [ ] **Selectivity**: ≥20% features with correlation >0.3
- [ ] **Comparison**: SAE selectivity ≥ 80% of raw neuron selectivity
- [ ] **Sparsity**: 20-40% lifetime sparsity
- [ ] **Reconstruction**: Test loss <1.0
- [ ] **Coverage**: Uniform distribution of preferred orientations
- [ ] **Diversity**: Low inter-feature correlation (<0.3 mean)
- [ ] **Generalization**: Performance maintained on held-out sessions
- [ ] **Visualization**: Clear tuning curves for top features

---

## 📚 Quick Command Reference

```bash
# 1. Analyze trained SAE
python scripts/analyze_sae_features.py \
    --sae-results sae_models/training_results.json \
    --session-id 754829445 \
    --output-dir sae_analysis

# 2. Hyperparameter search (quick)
python scripts/sae_hyperparameter_search.py \
    --session-id 754829445 \
    --quick

# 3. Hyperparameter search (full)
python scripts/sae_hyperparameter_search.py \
    --session-id 754829445

# 4. Visualize features
python scripts/visualize_sae_features.py \
    --session-id 754829445 \
    --sae-model sae_models/sae_session_754829445.pt

# 5. Retrain with optimal config
python examples/sae_training_top_sessions.py \
    --session-config recommended_sessions.json \
    --sae-dim 256 \
    --sparsity 0.01 \
    --epochs 100
```

---

**Next**: After optimization, proceed to cross-modal comparison and mechanistic interpretability experiments in [MECHINT_VALIDATION_PROGRESS.md](MECHINT_VALIDATION_PROGRESS.md)
