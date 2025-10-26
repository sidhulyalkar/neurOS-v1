# Quick Start Guide: Model-to-Brain Alignment

## Which Method Should I Use?

### Decision Tree

```
Do you want to ALIGN representations or COMPARE them?
│
├─ ALIGN (find shared space)
│  │
│  ├─ Linear alignment? → Use CCA
│  │  ├─ Simple case → CCA
│  │  ├─ High-dimensional → RegularizedCCA
│  │  ├─ Non-linear → KernelCCA
│  │  └─ Time series → TimeVaryingCCA
│  │
│  └─ Predictive alignment? → Use PLS
│     ├─ Know #components → PLS
│     └─ Auto-select → CrossValidatedPLS
│
└─ COMPARE (test similarity)
   └─ Compare geometries → Use RSA
      ├─ Single comparison → RSA
      ├─ Multiple layers → compare_multiple_rdms
      ├─ Clustering → HierarchicalRSA
      └─ Visualization → MDSVisualization
```

## 30-Second Examples

### CCA: Find Shared Space

```python
from neuros_neurofm.interpretability.alignment import CCA

cca = CCA(n_components=10)
cca.fit(model_acts, brain_acts)
corrs = cca.canonical_correlations_  # How well aligned?
```

### RSA: Compare Geometries

```python
from neuros_neurofm.interpretability.alignment import RSA

rsa = RSA(metric='correlation', comparison='spearman')
similarity = rsa.compare(model_acts, brain_acts)  # 0-1 score
```

### PLS: Predict Brain Activity

```python
from neuros_neurofm.interpretability.alignment import PLS

pls = PLS(n_components=20)
pls.fit(model_acts, brain_acts)
predicted = pls.predict(new_model_acts)
```

### Statistical Testing

```python
from neuros_neurofm.interpretability.alignment import (
    NoiseCeiling, PermutationTest
)

# What's the ceiling?
nc = NoiseCeiling(method='split-half')
ceiling = nc.estimate(brain_with_reps)

# Is it significant?
perm = PermutationTest(n_permutations=1000)
result = perm.test(your_metric, model_acts, brain_acts)
p_value = result['p_value']
```

## Common Workflows

### Workflow 1: Basic Alignment

```python
from neuros_neurofm.interpretability.alignment import CCA

# 1. Fit
cca = CCA(n_components=10)
cca.fit(model_acts, brain_acts)

# 2. Check correlations
print(f"Top 5 correlations: {cca.canonical_correlations_[:5]}")

# 3. Transform to shared space
model_c, brain_c = cca.transform(model_acts, brain_acts)

# 4. Evaluate on new data
score = cca.score(model_acts_test, brain_acts_test)
```

### Workflow 2: Layer-by-Layer Analysis

```python
from neuros_neurofm.interpretability.alignment import RSA

rsa = RSA(metric='correlation', comparison='spearman')

# Compare each layer to brain
layer_scores = []
for layer_acts in model_layers:
    score = rsa.compare(layer_acts, brain_acts)
    layer_scores.append(score)

# Find best layer
best_layer = np.argmax(layer_scores)
print(f"Layer {best_layer} most similar to brain")
```

### Workflow 3: Predictive Modeling

```python
from neuros_neurofm.interpretability.alignment import (
    CrossValidatedPLS, PLSVisualization
)

# Auto-select components
pls = CrossValidatedPLS(max_components=50, cv=5)
pls.fit(model_acts, brain_acts)

print(f"Selected {pls.best_n_components_} components")
print(f"CV score: {max(pls.cv_scores_.values()):.4f}")

# Visualize
vis = PLSVisualization(pls.pls_)
fig = vis.plot_explained_variance()
```

### Workflow 4: Complete Statistical Evaluation

```python
from neuros_neurofm.interpretability.alignment import (
    PLS, NoiseCeiling, NormalizedScore,
    BootstrapCI, PermutationTest
)

# Fit model
pls = PLS(n_components=20)
pls.fit(X_train, Y_train)

# Evaluate
raw_score = pls.score(X_test, Y_test)

# Noise ceiling
nc = NoiseCeiling()
ceiling = nc.estimate(Y_with_reps)
normalized = NormalizedScore().normalize(raw_score, ceiling)

# Confidence interval
bootstrap = BootstrapCI(n_bootstrap=1000)
ci = bootstrap.compute(lambda x, y: pls.score(x, y), X_test, Y_test)

# Significance
perm = PermutationTest(n_permutations=1000)
sig = perm.test(lambda x, y: pls.score(x, y), X_test, Y_test)

# Report
print(f"Raw score: {raw_score:.4f}")
print(f"Normalized: {normalized:.2%} of explainable variance")
print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
print(f"p-value: {sig['p_value']:.4f}")
```

## Parameter Cheat Sheet

### CCA Parameters

```python
CCA(
    n_components=10,        # How many components? (1-min(dim))
    reg=0.0,               # Regularization (0=none, >0=ridge)
    device='cuda'          # 'cuda' or 'cpu'
)
```

**Tuning tips:**
- Start with n_components = 10-20
- Use RegularizedCCA if results are unstable
- Increase reg if you get warnings about singular matrices

### RSA Parameters

```python
RSA(
    metric='correlation',      # Distance: 'correlation', 'euclidean', 'cosine'
    comparison='spearman',     # Comparison: 'spearman', 'kendall', 'pearson'
    device='cuda'
)
```

**Tuning tips:**
- 'correlation' metric most common for neural data
- 'spearman' robust to outliers
- 'pearson' for normally distributed RDMs

### PLS Parameters

```python
PLS(
    n_components=20,      # Number of latent components
    scale=True,          # Standardize data? (usually True)
    device='cuda'
)
```

**Tuning tips:**
- Use CrossValidatedPLS to auto-select n_components
- scale=True recommended for different feature scales
- Start with n_components = 10-30

### Metrics Parameters

```python
# Noise ceiling
NoiseCeiling(
    method='split-half',    # 'split-half', 'leave-one-out', 'ncsnr'
    n_splits=100           # More = more accurate (slower)
)

# Bootstrap
BootstrapCI(
    n_bootstrap=1000,      # More = more accurate (slower)
    confidence=0.95       # CI level (0.95 = 95%)
)

# Permutation
PermutationTest(
    n_permutations=1000   # More = more accurate (slower)
)
```

**Tuning tips:**
- 'split-half' fastest, requires 2+ repetitions
- 'leave-one-out' more accurate, slower
- 1000 bootstrap/permutation iterations usually sufficient

## Common Errors and Solutions

### Error: "Covariance matrix is singular"
**Solution:** Use RegularizedCCA instead of CCA
```python
rcca = RegularizedCCA(n_components=10, reg_params=[1e-3, 1e-2, 1e-1])
```

### Error: "Need at least 2 repetitions"
**Solution:** Use different noise ceiling method
```python
nc = NoiseCeiling(method='ncsnr')  # Doesn't need repetitions
```

### Warning: "Some bootstrap iterations failed"
**Solution:** Reduce n_bootstrap or check data quality
```python
bootstrap = BootstrapCI(n_bootstrap=500)  # Use fewer iterations
```

### Poor performance
**Solutions:**
1. Increase n_components
2. Try regularization (RegularizedCCA, PLS with scale=True)
3. Check for NaN/Inf in data
4. Verify data alignment (same stimuli in same order)

## Performance Tips

### For Large Datasets

```python
# Use GPU
cca = CCA(n_components=10, device='cuda')

# Reduce bootstrap/permutation iterations
bootstrap = BootstrapCI(n_bootstrap=500)
perm = PermutationTest(n_permutations=500)

# Use subset for cross-validation
cv_pls = CrossValidatedPLS(max_components=30, cv=3)  # 3-fold instead of 5
```

### For Small Datasets

```python
# Use more robust methods
rcca = RegularizedCCA(n_components=10)  # Regularization helps
rsa = RSA(comparison='spearman')        # Spearman more robust

# More CV folds
cv_pls = CrossValidatedPLS(cv=10)       # Leave-one-out-like
```

## Minimum Working Examples

### Example 1: "Does my model align with brain?"

```python
from neuros_neurofm.interpretability.alignment import CCA

cca = CCA(n_components=10)
cca.fit(model_acts, brain_acts)

# If mean correlation > 0.3, decent alignment
print(f"Mean correlation: {cca.score(model_acts, brain_acts):.4f}")
```

### Example 2: "Which layer matches brain best?"

```python
from neuros_neurofm.interpretability.alignment import RSA

rsa = RSA()
scores = [rsa.compare(layer, brain_acts) for layer in layers]
best = np.argmax(scores)
print(f"Layer {best}: {scores[best]:.4f}")
```

### Example 3: "Can I predict brain from model?"

```python
from neuros_neurofm.interpretability.alignment import CrossValidatedPLS

pls = CrossValidatedPLS(max_components=50)
pls.fit(model_acts, brain_acts)

print(f"R² = {pls.score(model_acts, brain_acts):.4f}")
```

### Example 4: "Is this significant?"

```python
from neuros_neurofm.interpretability.alignment import PermutationTest

perm = PermutationTest(n_permutations=1000)
result = perm.test(your_metric, model_acts, brain_acts)

if result['p_value'] < 0.05:
    print("Significant!")
else:
    print("Not significant")
```

## Next Steps

- **Full examples:** See `README.md`
- **API details:** Check docstrings (`help(CCA)`)
- **Theory:** Read references in main README
- **Questions:** File an issue on GitHub

## Quick Reference Card

| Task | Method | One-liner |
|------|--------|-----------|
| Linear alignment | CCA | `CCA(10).fit(X, Y).score(X, Y)` |
| Geometry comparison | RSA | `RSA().compare(X, Y)` |
| Predict brain | PLS | `PLS(20).fit(X, Y).predict(X)` |
| Noise ceiling | NoiseCeiling | `NoiseCeiling().estimate(Y_reps)` |
| Confidence interval | Bootstrap | `BootstrapCI().compute(fn, X, Y)` |
| Significance | Permutation | `PermutationTest().test(fn, X, Y)` |

---

**Remember:**
1. Always cross-validate
2. Estimate noise ceiling when possible
3. Test significance
4. Report confidence intervals
