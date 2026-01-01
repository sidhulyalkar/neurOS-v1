# Next Steps - SAE Analysis Pipeline

**Current Status**: SAE training complete ✅ (Session 754829445, 60.9% selective features)

Run these commands in order to deeply analyze and optimize your SAE:

---

## Step 1: Feature Analysis (Run First - 15 min)

**Compare SAE features vs raw neurons, analyze sparsity**

```bash
python scripts/analyze_sae_features.py \
    --sae-results sae_models/training_results.json \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir sae_analysis
```

**Output files**:
- `sae_analysis/correlation_comparison.png` - Histogram comparing selectivity
- `sae_analysis/preferred_orientation_distribution.png` - Polar plots
- `sae_analysis/sparsity_analysis.png` - L0/lifetime/population sparsity
- `sae_analysis/feature_analysis_session_754829445.json` - Raw data

**What to check**:
- Are SAE features ≥ raw neurons in selectivity?
- Is sparsity in 20-40% range?
- Do features cover all orientations uniformly?

---

## Step 2: Feature Visualization (10 min)

**Understand what each feature learned**

```bash
python scripts/visualize_sae_features.py \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --sae-model sae_models/sae_session_754829445.pt \
    --output-dir sae_visualizations
```

**Output files** (5 publication-quality figures):
1. `tuning_curves_session_754829445.png` - Top 6 features
2. `activation_heatmap_session_754829445.png` - Population activity
3. `weights_session_754829445.png` - Encoder weights
4. `orientation_map_session_754829445.png` - Polar coverage
5. `feature_clustering_session_754829445.png` - PCA analysis

**What to check**:
- Do tuning curves show sharp peaks?
- Is orientation coverage uniform?
- Do features cluster by orientation?

---

## Step 3: Hyperparameter Search (Optional - 5 min quick / 1-2 hr full)

**Find optimal architecture for maximum interpretability**

### Quick Search (Recommended First)

```bash
python scripts/sae_hyperparameter_search.py \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --quick \
    --output-dir hyperparameter_search
```

Tests: 2 hidden dims × 2 sparsity = 4 configs

### Full Search (If quick search shows promise)

```bash
python scripts/sae_hyperparameter_search.py \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir hyperparameter_search
```

Tests: 4 hidden dims × 4 sparsity × 3 LRs × 2 activations = 96 configs

**Output files**:
- `hyperparameter_search/hyperparameter_search_session_754829445.json`
- `hyperparameter_search/hyperparameter_search_session_754829445.csv`

**What to check**:
- Recommended configuration (hidden_dim, sparsity, lr)
- Expected selectivity improvement
- Composite score ranking

---

## Step 4: Retrain with Optimal Config (15 min)

**After hyperparameter search, retrain with best settings**

```bash
# Use recommended values from hyperparameter search
python examples/sae_training_top_sessions.py \
    --session-config session_analysis/recommended_sessions.json \
    --allen-cache allen_validation_cache \
    --sae-dim 128 \        # Example: use recommended value
    --sparsity 0.01 \      # Example: use recommended value
    --lr 0.001 \           # Example: use recommended value
    --epochs 500           # More epochs for final model
```

---

## Step 5: Multi-Session Validation (30 min)

**Train on top 5 sessions to test generalization**

```bash
python examples/sae_training_top_sessions.py \
    --session-config session_analysis/recommended_sessions.json \
    --allen-cache allen_validation_cache \
    --use-top-n 5 \
    --sae-dim 256 \
    --epochs 100 \
    --output-dir sae_models_multi_session
```

**Then analyze each**:
```bash
for session_id in 754829445 766640955 760345702 768515987 771160300; do
    python scripts/analyze_sae_features.py \
        --sae-results sae_models_multi_session/training_results.json \
        --session-id $session_id \
        --output-dir sae_analysis_session_${session_id}
done
```

---

## Decision Tree

```
Start: Training complete (60.9% selective)
         ↓
Run analyze_sae_features.py
         ↓
Check results:
  ├─ Selectivity good? → Run visualize_sae_features.py → Publication ready!
  ├─ Selectivity <50%? → Run hyperparameter_search.py → Retrain
  ├─ Sparsity >60%?    → Decrease sparsity λ → Retrain
  └─ Sparsity <10%?    → Increase sparsity λ → Retrain
```

---

## Expected Timeline

| Task | Time | Priority |
|------|------|----------|
| analyze_sae_features.py | 15 min | 🔴 Critical |
| visualize_sae_features.py | 10 min | 🔴 Critical |
| hyperparameter_search.py (quick) | 5 min | 🟡 Recommended |
| hyperparameter_search.py (full) | 1-2 hr | 🟢 Optional |
| Retrain with optimal config | 15 min | 🟡 If needed |
| Multi-session training | 30 min | 🟢 For publication |

---

## Troubleshooting

### Analysis script fails
- Check that SAE model exists: `sae_models/sae_session_754829445.pt`
- Check training results: `sae_models/training_results.json`
- Verify session cache: `allen_validation_cache/session_754829445/`

### Low selectivity in analysis
- Normal for first run
- Run hyperparameter search to optimize
- Try higher hidden_dim (256, 512)
- Try lower sparsity (0.005, 0.01)

### Visualization crashes
- Check available memory (visualizations are memory-intensive)
- Try reducing number of features plotted (edit script)

---

## Success Metrics

After analysis, check these:

| Metric | Target | Your Result |
|--------|--------|-------------|
| % Selective features | >20% | 60.9% ✅ |
| Max correlation | >0.6 | 0.707 ✅ |
| Mean correlation | >0.25 | 0.365 ✅ |
| Lifetime sparsity | 20-40% | TBD |
| Reconstruction loss | <1.0 | 0.467 ✅ |
| Orientation coverage | Uniform | TBD |

---

## Files You'll Generate

```
neurOS-v1/
├── sae_analysis/
│   ├── correlation_comparison.png              # SAE vs raw comparison
│   ├── preferred_orientation_distribution.png  # Coverage analysis
│   ├── sparsity_analysis.png                   # Sparsity metrics
│   └── feature_analysis_session_754829445.json # Raw data
├── sae_visualizations/
│   ├── tuning_curves_session_754829445.png
│   ├── activation_heatmap_session_754829445.png
│   ├── weights_session_754829445.png
│   ├── orientation_map_session_754829445.png
│   └── feature_clustering_session_754829445.png
└── hyperparameter_search/
    ├── hyperparameter_search_session_754829445.json
    └── hyperparameter_search_session_754829445.csv
```

---

## Quick Copy-Paste

**Full analysis pipeline (run all at once)**:

```bash
# Step 1: Analyze
python scripts/analyze_sae_features.py \
    --sae-results sae_models/training_results.json \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir sae_analysis

# Step 2: Visualize
python scripts/visualize_sae_features.py \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --sae-model sae_models/sae_session_754829445.pt \
    --output-dir sae_visualizations

# Step 3: Hyperparameter search (quick)
python scripts/sae_hyperparameter_search.py \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --quick \
    --output-dir hyperparameter_search
```

---

**You're ready to go! Start with Step 1 (analyze_sae_features.py) to understand your SAE's performance.** 🚀
