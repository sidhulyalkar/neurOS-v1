# Model-to-Brain Alignment Tools for NeuroFMX

This module provides comprehensive tools for aligning neural network representations with brain recordings. It implements state-of-the-art methods from computational neuroscience for comparing and aligning model activations with neural data.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Methods](#methods)
   - [CCA - Canonical Correlation Analysis](#cca---canonical-correlation-analysis)
   - [RSA - Representational Similarity Analysis](#rsa---representational-similarity-analysis)
   - [PLS - Partial Least Squares](#pls---partial-least-squares)
   - [Metrics - Evaluation Tools](#metrics---evaluation-tools)
4. [Quick Start](#quick-start)
5. [Examples](#examples)
6. [API Reference](#api-reference)

## Overview

The alignment module provides four main categories of tools:

1. **CCA (Canonical Correlation Analysis)**: Linear methods for finding shared representational spaces
2. **RSA (Representational Similarity Analysis)**: Geometry-based comparison of representational structures
3. **PLS (Partial Least Squares)**: Predictive alignment with latent variable analysis
4. **Metrics**: Statistical evaluation tools including noise ceiling estimation and significance testing

## Installation

```bash
# Required dependencies
pip install torch numpy scipy scikit-learn matplotlib seaborn tqdm

# Or install the full neuros-neurofm package
cd packages/neuros-neurofm
pip install -e .
```

## Methods

### CCA - Canonical Correlation Analysis

CCA finds linear transformations that maximize the correlation between two sets of variables. Useful for:
- Finding shared representational spaces between model and brain
- Linear alignment of multi-dimensional representations
- Time-varying alignment analysis

**Available Methods:**
- `CCA`: Standard canonical correlation analysis
- `RegularizedCCA`: CCA with automatic regularization selection
- `KernelCCA`: Non-linear CCA using kernel methods
- `TimeVaryingCCA`: Sliding window CCA for time series

**Key Features:**
- Multiple CCA variants (standard, regularized, kernel)
- Cross-validated dimension selection
- Time-varying analysis for temporal data
- GPU acceleration support

### RSA - Representational Similarity Analysis

RSA compares the geometry of representational spaces without requiring alignment. Useful for:
- Testing if model and brain represent information similarly
- Comparing multiple layers or models simultaneously
- Visualizing representational structure

**Available Methods:**
- `RepresentationalDissimilarityMatrix`: Compute RDMs with various distance metrics
- `RSA`: Compare RDMs using correlation measures
- `HierarchicalRSA`: Cluster stimuli based on representations
- `MDSVisualization`: 2D/3D visualization of representational spaces

**Key Features:**
- Multiple distance metrics (correlation, euclidean, cosine, mahalanobis)
- RDM comparison methods (Spearman, Kendall, Pearson)
- Hierarchical clustering with dendrograms
- MDS visualization

### PLS - Partial Least Squares

PLS finds latent variables that maximize covariance between predictors and targets. Useful for:
- Predicting brain activity from model activations
- Finding latent dimensions that explain both spaces
- Analyzing component contributions

**Available Methods:**
- `PLS`: Standard partial least squares regression
- `CrossValidatedPLS`: PLS with automatic component selection
- `PLSVisualization`: Visualization tools for PLS results

**Key Features:**
- Multi-output prediction
- Cross-validated component selection
- Explained variance analysis
- Comprehensive visualizations

### Metrics - Evaluation Tools

Statistical tools for evaluating alignment quality:

**Available Methods:**
- `NoiseCeiling`: Estimate maximum achievable performance given measurement noise
- `BootstrapCI`: Compute bootstrap confidence intervals
- `PermutationTest`: Test statistical significance
- `NormalizedScore`: Normalize scores by noise ceiling
- `CrossValidatedMetric`: Robust cross-validated evaluation

**Key Features:**
- Multiple noise ceiling estimation methods
- Bootstrap resampling for confidence intervals
- Permutation testing for significance
- Cross-validation for robust estimates

## Quick Start

### Basic CCA Example

```python
from neuros_neurofm.interpretability.alignment import CCA
import torch

# Model activations: (n_samples, n_features_model)
model_acts = torch.randn(200, 512)

# Brain recordings: (n_samples, n_features_brain)
brain_acts = torch.randn(200, 100)

# Fit CCA
cca = CCA(n_components=10)
cca.fit(model_acts, brain_acts)

# Transform to shared space
model_canonical, brain_canonical = cca.transform(model_acts, brain_acts)

# Get canonical correlations
print(f"Correlations: {cca.canonical_correlations_}")
```

### Basic RSA Example

```python
from neuros_neurofm.interpretability.alignment import RSA

# Compare representational geometries
rsa = RSA(metric='correlation', comparison='spearman')
similarity = rsa.compare(model_acts, brain_acts)

print(f"RSA similarity: {similarity:.4f}")
```

### Basic PLS Example

```python
from neuros_neurofm.interpretability.alignment import PLS

# Fit PLS
pls = PLS(n_components=20)
pls.fit(model_acts, brain_acts)

# Predict brain activity
brain_predicted = pls.predict(model_acts)

# Evaluate
r2_score = pls.score(model_acts, brain_acts)
print(f"R² score: {r2_score:.4f}")
```

### Statistical Evaluation

```python
from neuros_neurofm.interpretability.alignment import (
    NoiseCeiling, BootstrapCI, PermutationTest
)

# Noise ceiling (requires repeated measurements)
brain_with_reps = torch.randn(10, 100, 200)  # 10 reps, 100 stimuli, 200 voxels
nc = NoiseCeiling(method='split-half')
ceiling = nc.estimate(brain_with_reps)
print(f"Noise ceiling: {ceiling:.4f}")

# Bootstrap confidence intervals
def my_metric(x, y):
    return torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()

bootstrap = BootstrapCI(n_bootstrap=1000)
ci = bootstrap.compute(my_metric, model_acts, brain_acts)
print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

# Permutation test
perm = PermutationTest(n_permutations=1000)
result = perm.test(my_metric, model_acts, brain_acts)
print(f"p-value: {result['p_value']:.4f}")
```

## Examples

### Example 1: Complete CCA Analysis

```python
from neuros_neurofm.interpretability.alignment import (
    RegularizedCCA, select_cca_dimensions
)

# Select optimal number of components
result = select_cca_dimensions(model_acts, brain_acts, max_components=50, cv=5)
best_n = result['best_n_components']
print(f"Best n_components: {best_n}")

# Fit regularized CCA
rcca = RegularizedCCA(n_components=best_n)
rcca.fit(model_acts, brain_acts)

print(f"Best regularization: {rcca.best_reg_}")
print(f"Mean correlation: {rcca.score(model_acts, brain_acts):.4f}")
```

### Example 2: Multi-Layer RSA Comparison

```python
from neuros_neurofm.interpretability.alignment import (
    RepresentationalDissimilarityMatrix, compare_multiple_rdms
)

# Get activations from multiple layers
layer1_acts = torch.randn(100, 256)
layer2_acts = torch.randn(100, 512)
layer3_acts = torch.randn(100, 1024)
brain_acts = torch.randn(100, 200)

# Compute RDMs
rdm_computer = RepresentationalDissimilarityMatrix(metric='correlation')
rdm1 = rdm_computer.compute(layer1_acts)
rdm2 = rdm_computer.compute(layer2_acts)
rdm3 = rdm_computer.compute(layer3_acts)
rdm_brain = rdm_computer.compute(brain_acts)

# Compare all RDMs
rdms = [rdm1, rdm2, rdm3, rdm_brain]
labels = ['Layer 1', 'Layer 2', 'Layer 3', 'Brain']
sim_matrix, fig = compare_multiple_rdms(rdms, labels)

print("Similarity matrix:")
print(sim_matrix)
```

### Example 3: Cross-Validated PLS with Visualization

```python
from neuros_neurofm.interpretability.alignment import (
    CrossValidatedPLS, PLSVisualization
)

# Fit with automatic component selection
cv_pls = CrossValidatedPLS(max_components=50, cv=5)
cv_pls.fit(model_acts, brain_acts)

print(f"Best n_components: {cv_pls.best_n_components_}")

# Visualize results
pls_vis = PLSVisualization(cv_pls.pls_)

# Plot latent variables
fig1 = pls_vis.plot_latent_variables(components=[0, 1])

# Plot explained variance
fig2 = pls_vis.plot_explained_variance()

# Plot predictions
fig3 = pls_vis.plot_predictions(model_acts, brain_acts)
```

### Example 4: Complete Evaluation Pipeline

```python
from neuros_neurofm.interpretability.alignment import (
    PLS, NoiseCeiling, NormalizedScore, BootstrapCI, PermutationTest
)

# 1. Fit alignment model
pls = PLS(n_components=20)
pls.fit(model_acts_train, brain_acts_train)

# 2. Compute raw score
raw_score = pls.score(model_acts_test, brain_acts_test)
print(f"Raw R² score: {raw_score:.4f}")

# 3. Estimate noise ceiling
brain_with_reps = torch.randn(10, 100, 200)  # Multiple measurements
nc = NoiseCeiling(method='split-half')
ceiling = nc.estimate(brain_with_reps)
print(f"Noise ceiling: {ceiling:.4f}")

# 4. Normalize by ceiling
normalizer = NormalizedScore()
normalized_score = normalizer.normalize(raw_score, ceiling)
print(f"Normalized score: {normalized_score:.2%} of explainable variance")

# 5. Compute confidence interval
def pls_score_fn(x, y):
    return pls.score(x, y)

bootstrap = BootstrapCI(n_bootstrap=1000)
ci = bootstrap.compute(pls_score_fn, model_acts_test, brain_acts_test)
print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

# 6. Test significance
perm = PermutationTest(n_permutations=1000)
perm_result = perm.test(pls_score_fn, model_acts_test, brain_acts_test)
print(f"p-value: {perm_result['p_value']:.4f}")
```

### Example 5: Time-Varying CCA

```python
from neuros_neurofm.interpretability.alignment import TimeVaryingCCA

# Time series data: (n_samples, n_timepoints, n_features)
model_time = torch.randn(50, 200, 512)
brain_time = torch.randn(50, 200, 100)

# Fit time-varying CCA
tvcca = TimeVaryingCCA(
    n_components=10,
    window_size=50,
    stride=10
)

results = tvcca.fit_transform(model_time, brain_time)

# Plot correlations over time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i in range(5):  # Plot first 5 components
    plt.plot(
        results['window_times'].cpu().numpy(),
        results['correlations'][:, i].cpu().numpy(),
        label=f'Component {i+1}'
    )
plt.xlabel('Time')
plt.ylabel('Canonical Correlation')
plt.legend()
plt.title('Time-Varying Canonical Correlations')
plt.show()
```

## API Reference

### CCA Module

#### `CCA(n_components, reg=0.0, device=None)`
Standard canonical correlation analysis.

**Parameters:**
- `n_components` (int): Number of canonical components
- `reg` (float): Regularization parameter
- `device` (str): 'cuda' or 'cpu'

**Methods:**
- `fit(X, Y)`: Fit CCA model
- `transform(X, Y=None)`: Transform to canonical space
- `score(X, Y)`: Compute mean canonical correlation

**Attributes:**
- `canonical_correlations_`: Canonical correlation values
- `weights_x_`, `weights_y_`: Canonical weights

---

#### `RegularizedCCA(n_components, reg_params, cv=5, device=None)`
CCA with cross-validated regularization selection.

**Additional Attributes:**
- `best_reg_`: Best regularization parameter
- `cv_scores_`: Cross-validation scores for each reg parameter

---

#### `KernelCCA(n_components, kernel='rbf', gamma=0.1, reg=1e-3, device=None)`
Non-linear CCA using kernel methods.

**Parameters:**
- `kernel` (str): Kernel type ('rbf', 'linear', 'poly')
- `gamma` (float): Kernel coefficient

---

### RSA Module

#### `RepresentationalDissimilarityMatrix(metric='correlation', device=None)`
Compute representational dissimilarity matrices.

**Parameters:**
- `metric` (str): Distance metric ('correlation', 'euclidean', 'cosine', 'mahalanobis')

**Methods:**
- `compute(representations)`: Compute RDM
- `visualize(rdm, labels)`: Visualize RDM as heatmap

---

#### `RSA(metric='correlation', comparison='spearman', device=None)`
Representational similarity analysis.

**Parameters:**
- `comparison` (str): RDM comparison method ('spearman', 'kendall', 'pearson')

**Methods:**
- `compare(reps1, reps2)`: Compare two sets of representations
- `compare_rdms(rdm1, rdm2)`: Compare pre-computed RDMs

---

### PLS Module

#### `PLS(n_components, scale=True, device=None)`
Partial least squares regression.

**Methods:**
- `fit(X, Y)`: Fit PLS model
- `transform(X, Y=None)`: Transform to latent space
- `predict(X)`: Predict target values
- `score(X, Y)`: Compute R² score
- `explained_variance()`: Get variance explained per component

**Attributes:**
- `weights_x_`, `weights_y_`: PLS weights
- `loadings_x_`, `loadings_y_`: PLS loadings
- `x_scores_`, `y_scores_`: Latent variable scores

---

#### `CrossValidatedPLS(max_components, cv=5, scale=True, device=None)`
PLS with automatic component selection.

**Attributes:**
- `best_n_components_`: Optimal number of components
- `cv_scores_`: Cross-validation scores

---

### Metrics Module

#### `NoiseCeiling(method='split-half', n_splits=100, device=None)`
Estimate noise ceiling for brain measurements.

**Parameters:**
- `method` (str): Estimation method ('split-half', 'leave-one-out', 'ncsnr')

**Methods:**
- `estimate(brain_data, per_feature=False)`: Estimate ceiling

---

#### `BootstrapCI(n_bootstrap=1000, confidence=0.95, random_state=42)`
Bootstrap confidence intervals.

**Methods:**
- `compute(metric_fn, *args, **kwargs)`: Compute CI for a metric

**Returns:** Dictionary with mean, std, lower, upper bounds

---

#### `PermutationTest(n_permutations=1000, random_state=42)`
Permutation testing for significance.

**Methods:**
- `test(metric_fn, X, Y)`: Test if alignment is significant

**Returns:** Dictionary with observed value, p-value, null distribution stats

---

## Best Practices

### Choosing an Alignment Method

1. **Use CCA when:**
   - You want linear alignment between spaces
   - You need to project both representations to a shared space
   - You're interested in the strength of linear relationships

2. **Use RSA when:**
   - You want to compare representational geometries without alignment
   - You're comparing multiple models or layers simultaneously
   - You're interested in whether information is represented similarly

3. **Use PLS when:**
   - You want to predict brain activity from model activations
   - You need to handle high-dimensional outputs
   - You're interested in latent variable analysis

### Statistical Evaluation

Always include:
1. **Noise ceiling estimation** - Know the maximum achievable performance
2. **Cross-validation** - Get robust performance estimates
3. **Confidence intervals** - Quantify uncertainty
4. **Significance testing** - Verify results aren't due to chance

### Common Pitfalls

1. **Overfitting**: Use cross-validation, especially with high-dimensional data
2. **Multiple comparisons**: Correct p-values when testing multiple hypotheses
3. **Data leakage**: Never use test data for model selection
4. **Ignoring noise ceiling**: Raw scores can be misleading without ceiling normalization

## References

- Canonical Correlation Analysis: Hotelling (1936)
- RSA: Kriegeskorte et al. (2008) Frontiers in Systems Neuroscience
- PLS: Wold (1966)
- Noise Ceiling: Schoppe et al. (2016) PLOS Computational Biology

## License

Part of the NeuroFMX project. See main repository for license details.
