# Implementation Guide for Notebooks 03-10

This guide provides detailed outlines, code examples, and concepts for notebooks 03-10. Use this as a reference to:
- Understand what each notebook covers
- Get started with example code
- Build your own analyses

## Status

✅ **Complete & Ready to Use:**
- [README.md](README.md) - Learning path guide
- [01_introduction_and_quickstart.ipynb](01_introduction_and_quickstart.ipynb) - Full introduction
- [02_sparse_autoencoders.ipynb](02_sparse_autoencoders.ipynb) - Complete SAE tutorial

📖 **Detailed Guides Below (Ready to Implement):**
- 03: Causal Interventions
- 04: Fractal Analysis
- 05: Brain Alignment
- 06: Dynamical Systems
- 07: Circuit Extraction
- 08: Biophysical Modeling
- 09: Information Theory
- 10: Advanced Topics

---

## Notebook 03: Causal Interventions & Circuit Discovery

### Core Concepts

**Goal**: Discover computational circuits through causal interventions

**Key Techniques**:
1. Activation Patching
2. Ablation Studies
3. Path Tracing
4. Causal Graph Construction

### Structure

#### Part 1: Activation Patching Basics
```python
from neuros_mechint.interventions import ActivationPatcher

# Setup: Clean vs Corrupted inputs
clean_input = torch.randn(seq_len, batch, d_model)
corrupted_input = clean_input + noise

# Patch a component
patcher = ActivationPatcher(model, layer_name='self_attn')
result = patcher.patch(
    clean_input=clean_input,
    corrupted_input=corrupted_input,
    loss_fn=your_loss_function
)

# Recovery score: How much did patching help?
print(f"Recovery: {result['recovery_score']:.2%}")
# High score = component is causally important!
```

**Key Insight**: Recovery score tells you if a component is *causally necessary*
- Score near 1.0 → Critical component
- Score near 0.5 → Moderately important
- Score near 0.0 → Not important

#### Part 2: Component-Specific Patching
```python
from neuros_mechint.interventions import AttentionPatcher, MLPPatcher

# Patch individual attention heads
attn_patcher = AttentionPatcher(model, 'self_attn')
for head_idx in range(num_heads):
    result = attn_patcher.patch_head(
        clean_input, corrupted_input, head_idx, loss_fn
    )
    print(f"Head {head_idx}: {result['recovery_score']:.2%}")

# Patch MLP neurons
mlp_patcher = MLPPatcher(model, 'linear1')
neuron_importance = mlp_patcher.patch_neurons(
    clean_input, corrupted_input, loss_fn,
    neuron_indices=range(0, 100, 10)  # Sample neurons
)
```

#### Part 3: Ablation Studies
```python
from neuros_mechint.interventions import AblationStudy

# Systematic ablation of components
study = AblationStudy(model=model)
results = study.run_hierarchical_ablation(
    test_data=test_inputs,
    loss_fn=loss_fn,
    components=['self_attn', 'linear1', 'linear2']
)

# Visualize component importance
study.plot_ablation_results(results)
```

**What to ablate**:
- Layers (coarse)
- Sublayers (medium: attention, MLP)
- Individual neurons/heads (fine)

#### Part 4: Path Analysis
```python
from neuros_mechint.interventions import PathAnalyzer

# Find information flow paths
analyzer = PathAnalyzer(model)
paths = analyzer.find_paths(
    input_data=test_input,
    source_layer='input',
    target_layer='output',
    num_paths=10
)

# Visualize top paths
for i, path in enumerate(paths[:5]):
    print(f"Path {i}: {' → '.join(path['layers'])}")
    print(f"  Strength: {path['strength']:.3f}")
```

#### Part 5: Causal Graph Construction
```python
from neuros_mechint.interventions import CausalGraph

# Build graph from interventions
graph = CausalGraph()
graph.build_from_perturbations(
    model=model,
    test_data=test_inputs,
    components=['self_attn', 'norm1', 'linear1', 'linear2']
)

# Visualize
graph.visualize(layout='hierarchical')

# Find critical paths
critical_circuit = graph.find_minimal_circuit(
    source='input',
    target='output',
    min_strength=0.5
)
```

### Exercises

1. **Find Induction Heads**: Use attention patching to find heads that do pattern completion
2. **Minimal Circuit**: Extract the smallest circuit that performs a task
3. **Compare Architectures**: Build causal graphs for different model architectures

### Key Papers
- Elhage et al. (2021): "A Mathematical Framework for Transformer Circuits"
- Wang et al. (2022): "Interpretability in the Wild"

---

## Notebook 04: Fractal Analysis & Biological Realism

### Core Concepts

**Goal**: Measure and enforce biological-like complexity in neural networks

**Why Fractals?** The brain exhibits scale-free, fractal dynamics across time and space. Models with similar properties may be more brain-like and interpretable.

### Structure

#### Part 1: Temporal Fractal Metrics
```python
from neuros_mechint.fractals import (
    HiguchiFractalDimension,
    DetrendedFluctuationAnalysis,
    HurstExponent,
    SpectralSlope
)

# Collect neural timeseries
activations = []  # Shape: (time, neurons)
with torch.no_grad():
    for t in range(1000):
        x = model(input[t])
        activations.append(x)
timeseries = torch.stack(activations)

# Compute Higuchi Fractal Dimension
hfd = HiguchiFractalDimension(kmax=10)
fd = hfd(timeseries)
print(f"Fractal Dimension: {fd:.3f}")
# FD ∈ [1, 2]: 1=simple, 2=complex/space-filling

# Detrended Fluctuation Analysis
dfa = DetrendedFluctuationAnalysis(min_scale=4, max_scale=100)
alpha = dfa(timeseries)
print(f"DFA exponent α: {alpha:.3f}")
# α=0.5 (white noise), α=1.0 (pink noise), α=1.5 (brown noise)

# Spectral slope (1/f^β scaling)
spectral = SpectralSlope(freq_range=(0.1, 50))
beta = spectral(timeseries)
print(f"Spectral exponent β: {beta:.3f}")
# β=0 (white), β=1 (pink/1/f), β=2 (brown)
```

**Interpretation**:
- **Healthy brain**: FD ≈ 1.4-1.7, β ≈ 0.8-1.2 (pink noise)
- **Too simple**: FD < 1.3, β < 0.5
- **Too complex**: FD > 1.8, β > 1.5

#### Part 2: Fractal Regularization During Training
```python
from neuros_mechint.fractals import SpectralPrior, MultifractalSmoothness

# Add fractal loss to training
spectral_prior = SpectralPrior(target_exponent=1.0, weight=0.1)
multifractal_loss = MultifractalSmoothness(weight=0.05)

# Training loop
for batch in dataloader:
    # Regular task loss
    output = model(batch)
    task_loss = criterion(output, labels)

    # Fractal regularization
    hidden_states = get_hidden_states(model, batch)
    frac_loss = spectral_prior(hidden_states) + multifractal_loss(hidden_states)

    # Total loss
    total_loss = task_loss + frac_loss
    total_loss.backward()
    optimizer.step()
```

**Benefits**:
- More brain-like dynamics
- Better generalization
- Improved robustness

#### Part 3: Generating Fractal Stimuli
```python
from neuros_mechint.fractals import (
    FractionalBrownianMotion,
    ColoredNoise,
    FractalPatterns
)

# Fractional Brownian Motion with Hurst exponent H
fbm = FractionalBrownianMotion(hurst=0.7, length=1000)
stimulus = fbm.generate()

# Colored noise (1/f^β)
pink_noise = ColoredNoise(exponent=1.0, length=1000)
stimulus = pink_noise.generate()

# 2D Fractal patterns
fractal_img = FractalPatterns.generate_2d(
    size=(128, 128),
    fractal_dimension=1.5
)

# Test model on fractal stimuli
response = model(stimulus)
```

**Use cases**:
- Test if models prefer naturalistic (fractal) vs random stimuli
- Measure scale-dependent processing
- Generate biologically-realistic test data

#### Part 4: Real-Time Fractal Tracking
```python
from neuros_mechint.fractals import LatentFDTracker

# Track fractal dimension during training
fd_tracker = LatentFDTracker(
    model=model,
    layer_names=['hidden1', 'hidden2', 'hidden3'],
    track_every_n_steps=100
)

# Training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # Training step
        ...

        # Track FD
        if step % 100 == 0:
            fd_tracker.step(model, batch)

# Visualize evolution
fd_tracker.plot_fd_evolution()
```

#### Part 5: Multifractal Analysis
```python
from neuros_mechint.fractals import MultifractalSpectrum

# Compute full multifractal spectrum
mf = MultifractalSpectrum(q_range=(-5, 5), num_q=50)
spectrum = mf(timeseries)

# Plot f(α) curve
plt.plot(spectrum['alpha'], spectrum['f_alpha'])
plt.xlabel('Hölder exponent α')
plt.ylabel('f(α)')
plt.title('Multifractal Spectrum')
plt.show()

# Multifractal width
width = spectrum['alpha'].max() - spectrum['alpha'].min()
print(f"Multifractality: {width:.3f}")
# Larger width = more heterogeneous scaling
```

### Exercises

1. **Measure Brain-Likeness**: Compare fractal properties of your model to real EEG/fMRI
2. **Regularization Study**: Train models with/without fractal regularization, compare performance
3. **Scale Analysis**: Use colored noise to test frequency-specific processing

### Key Papers
- Higuchi (1988): "Approach to an irregular time series"
- Peng et al. (1994): "Mosaic organization of DNA nucleotides"
- He (2014): "Scale-free brain activity"

---

## Notebook 05: Brain Alignment - Comparing Models to Brains

### Core Concepts

**Goal**: Quantify how well model representations match brain activity

**Three Main Techniques**:
1. **CCA** - Find shared representational subspaces
2. **RSA** - Compare representational geometries
3. **PLS** - Predict brain activity from model

### Structure

#### Part 1: Canonical Correlation Analysis (CCA)
```python
from neuros_mechint.alignment import CCA, RegularizedCCA, select_cca_dimensions

# Load model and brain data
model_activations = ...  # Shape: (samples, model_features)
brain_data = ...  # Shape: (samples, brain_voxels)

# Standard CCA
cca = CCA(n_components=20)
cca.fit(model_activations, brain_data)

# Transform to shared space
model_proj = cca.transform_X(model_activations_test)
brain_proj = cca.transform_Y(brain_data_test)

# Compute alignment score
alignment = cca.score(model_activations_test, brain_data_test)
print(f"CCA alignment: {alignment:.3f}")

# Regularized CCA for high-dimensional data
reg_cca = RegularizedCCA(n_components=50, regularization='auto')
reg_cca.fit(model_activations, brain_data)

# Select optimal dimensions via cross-validation
optimal_dims = select_cca_dimensions(
    model_activations, brain_data,
    max_components=100,
    cv_folds=5
)
print(f"Optimal dimensions: {optimal_dims}")
```

**Interpretation**:
- **CCA score > 0.7**: Strong alignment
- **CCA score 0.4-0.7**: Moderate alignment
- **CCA score < 0.4**: Weak alignment

**When to use**: When you want to find *shared* representational spaces

#### Part 2: Representational Similarity Analysis (RSA)
```python
from neuros_mechint.alignment import RSA, RepresentationalDissimilarityMatrix

# Compute RDMs
model_rdm = RepresentationalDissimilarityMatrix(metric='correlation')
model_rdm.compute(model_activations)

brain_rdm = RepresentationalDissimilarityMatrix(metric='correlation')
brain_rdm.compute(brain_data)

# Compare RDMs
rsa = RSA(comparison_metric='spearman')
similarity = rsa.compare(model_rdm, brain_rdm)
print(f"RSA similarity: {similarity:.3f}")

# Visualize RDMs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
model_rdm.plot(ax=axes[0], title='Model RDM')
brain_rdm.plot(ax=axes[1], title='Brain RDM')
rsa.plot_comparison(model_rdm, brain_rdm, ax=axes[2])
plt.show()

# Second-order RSA: compare across many stimuli
from neuros_mechint.alignment import compare_multiple_rdms
rdm_similarities = compare_multiple_rdms(
    [model_rdm_1, model_rdm_2, ...],
    [brain_rdm_1, brain_rdm_2, ...]
)
```

**Interpretation**:
- **Spearman ρ > 0.5**: Good geometric match
- **Spearman ρ < 0.3**: Poor match

**When to use**: When you care about *geometry* not absolute values

#### Part 3: Partial Least Squares (PLS)
```python
from neuros_mechint.alignment import PLS, CrossValidatedPLS

# Fit PLS model
pls = PLS(n_components=30)
pls.fit(model_activations, brain_data)

# Predict brain activity from model
predicted_brain = pls.predict(model_activations_test)

# Evaluate prediction
r2_score = pls.score(model_activations_test, brain_data_test)
print(f"PLS R²: {r2_score:.3f}")

# Cross-validated PLS
cv_pls = CrossValidatedPLS(max_components=100, cv_folds=5)
cv_pls.fit(model_activations, brain_data)

# Get explained variance per component
explained_var = cv_pls.explained_variance()
plt.plot(explained_var)
plt.xlabel('Component')
plt.ylabel('Explained Variance')
plt.title('PLS Explained Variance')
plt.show()
```

**Interpretation**:
- **R² > 0.5**: Model strongly predicts brain
- **R² < 0.2**: Weak prediction

**When to use**: When you want to *predict* brain from model

#### Part 4: Statistical Evaluation
```python
from neuros_mechint.alignment import (
    NoiseCeiling,
    BootstrapCI,
    PermutationTest,
    NormalizedScore
)

# Compute noise ceiling (max achievable performance)
noise_ceiling = NoiseCeiling(method='split-half')
ceiling = noise_ceiling.estimate(brain_data)
print(f"Noise ceiling: {ceiling:.3f}")

# Bootstrap confidence intervals
bootstrap = BootstrapCI(n_bootstrap=1000)
ci = bootstrap.compute(
    metric_fn=lambda: cca.score(model_activations_test, brain_data_test)
)
print(f"95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")

# Permutation test for significance
perm_test = PermutationTest(n_permutations=1000)
p_value = perm_test.test(
    score=alignment,
    null_fn=lambda: cca.score(
        model_activations_test,
        permute(brain_data_test)
    )
)
print(f"p-value: {p_value:.4f}")

# Normalize by noise ceiling
normalized = NormalizedScore(noise_ceiling=ceiling)
norm_score = normalized.compute(alignment)
print(f"Normalized score: {norm_score:.2%} of explainable variance")
```

#### Part 5: Multi-Layer Alignment
```python
# Compare alignment across model layers
layers = ['layer1', 'layer2', 'layer3', 'layer4']
alignment_scores = {}

for layer in layers:
    layer_acts = get_layer_activations(model, layer, stimuli)
    cca = CCA(n_components=50)
    cca.fit(layer_acts, brain_data)
    alignment_scores[layer] = cca.score(layer_acts_test, brain_data_test)

# Visualize layer progression
plt.bar(range(len(layers)), [alignment_scores[l] for l in layers])
plt.xticks(range(len(layers)), layers)
plt.ylabel('Alignment Score')
plt.title('Alignment Across Model Layers')
plt.show()

# Find best layer
best_layer = max(alignment_scores, key=alignment_scores.get)
print(f"Best aligned layer: {best_layer}")
```

### Complete Workflow Example

```python
# Full brain alignment pipeline
class BrainAlignmentPipeline:
    def __init__(self, model, brain_data):
        self.model = model
        self.brain_data = brain_data
        self.results = {}

    def run_full_analysis(self, stimuli):
        # 1. Extract model activations
        model_acts = self.get_activations(stimuli)

        # 2. CCA alignment
        cca = CCA(n_components=50)
        cca.fit(model_acts, self.brain_data)
        self.results['cca_score'] = cca.score(model_acts, self.brain_data)

        # 3. RSA geometry comparison
        rsa = RSA()
        self.results['rsa_similarity'] = rsa.compare(model_acts, self.brain_data)

        # 4. PLS prediction
        pls = PLS(n_components=30)
        pls.fit(model_acts, self.brain_data)
        self.results['pls_r2'] = pls.score(model_acts, self.brain_data)

        # 5. Statistical testing
        noise_ceiling = NoiseCeiling().estimate(self.brain_data)
        self.results['noise_ceiling'] = noise_ceiling

        # 6. Generate report
        self.generate_report()

    def generate_report(self):
        print("=" * 50)
        print("BRAIN ALIGNMENT REPORT")
        print("=" * 50)
        print(f"CCA Alignment:    {self.results['cca_score']:.3f}")
        print(f"RSA Similarity:   {self.results['rsa_similarity']:.3f}")
        print(f"PLS R²:           {self.results['pls_r2']:.3f}")
        print(f"Noise Ceiling:    {self.results['noise_ceiling']:.3f}")
        print(f"Normalized Score: {self.results['cca_score']/self.results['noise_ceiling']:.2%}")
```

### Exercises

1. **Layer Sweep**: Find which model layer best aligns with different brain regions
2. **Time-Varying Alignment**: Use TimeVaryingCCA to track how alignment changes over time
3. **Cross-Modal**: Align visual model to auditory cortex (should be low!)

### Key Papers
- Kriegeskorte et al. (2008): "Representational Similarity Analysis"
- Hotelling (1936): "Relations Between Two Sets of Variates"
- Yamins & DiCarlo (2016): "Using goal-driven deep learning models to understand sensory cortex"

---

## Notebook 06: Dynamical Systems Analysis

### Core Concepts

**Goal**: Understand neural trajectories using nonlinear dynamics and control theory

**Key Tools**:
1. Koopman operator theory
2. Lyapunov exponents
3. Fixed point analysis
4. Controllability

### Structure

#### Part 1: Koopman Operator Theory
```python
from neuros_mechint.dynamics import DynamicsAnalyzer

# Collect trajectory data from RNN
rnn = nn.RNN(input_size=10, hidden_size=50, num_layers=1)
trajectories = []

h = torch.zeros(1, 1, 50)
for t in range(1000):
    x_t = torch.randn(1, 1, 10)
    output, h = rnn(x_t, h)
    trajectories.append(h.squeeze().detach())

trajectories = torch.stack(trajectories)  # Shape: (time, hidden_dim)

# Estimate Koopman operator via DMD
analyzer = DynamicsAnalyzer()
koopman_result = analyzer.estimate_koopman_operator(
    trajectories,
    rank=20  # Low-rank approximation
)

# Extract eigenvalues and modes
eigenvalues = koopman_result['eigenvalues']
eigenvectors = koopman_result['eigenvectors']
modes = koopman_result['modes']

# Visualize eigenvalues in complex plane
plt.scatter(eigenvalues.real, eigenvalues.imag)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
unit_circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--')
plt.gca().add_patch(unit_circle)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Koopman Eigenvalues')
plt.axis('equal')
plt.show()

# Find dominant modes
dominant_modes = analyzer.identify_dominant_modes(
    koopman_result,
    criterion='growth_rate'
)
print(f"Found {len(dominant_modes)} dominant modes")
```

**Interpretation**:
- **|λ| > 1**: Unstable mode (grows)
- **|λ| < 1**: Stable mode (decays)
- **|λ| ≈ 1**: Slow mode (important!)

#### Part 2: Lyapunov Exponents - Measuring Chaos
```python
# Compute Lyapunov exponents
lyapunov = analyzer.compute_lyapunov_exponents(
    trajectories,
    n_steps_forward=100
)

print(f"Largest Lyapunov exponent: {lyapunov[0]:.4f}")

# Interpret
if lyapunov[0] > 0:
    print("System is chaotic!")
elif lyapunov[0] < 0:
    print("System is stable")
else:
    print("System is on edge of chaos")

# Plot all exponents
plt.bar(range(len(lyapunov)), lyapunov)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Exponent Index')
plt.ylabel('Lyapunov Exponent')
plt.title('Lyapunov Spectrum')
plt.show()

# Lyapunov dimension
lyap_dim = analyzer.estimate_lyapunov_dimension(lyapunov)
print(f"Lyapunov dimension: {lyap_dim:.2f}")
```

#### Part 3: Fixed Point Analysis
```python
# Find fixed points
fixed_points = analyzer.find_fixed_points(
    model=rnn,
    n_inits=100,  # Try 100 random initializations
    tolerance=1e-4
)

print(f"Found {len(fixed_points)} fixed points")

# Classify stability
for i, fp in enumerate(fixed_points):
    stability = analyzer.classify_fixed_point_stability(
        model=rnn,
        fixed_point=fp['state']
    )
    print(f"FP {i}: {stability} (eigenvalue max: {fp['max_eigenvalue']:.3f})")

# Visualize in PCA space
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
traj_2d = pca.fit_transform(trajectories.numpy())
fp_2d = pca.transform(torch.stack([fp['state'] for fp in fixed_points]).numpy())

plt.plot(traj_2d[:, 0], traj_2d[:, 1], alpha=0.5, label='Trajectory')
plt.scatter(fp_2d[:, 0], fp_2d[:, 1], c='red', s=100, marker='*', label='Fixed Points')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Neural Trajectory & Fixed Points')
plt.legend()
plt.show()
```

#### Part 4: Intrinsic Dimensionality
```python
# Estimate effective dimensionality
dim_pca = analyzer.estimate_intrinsic_dimensionality(
    trajectories,
    method='pca',
    variance_threshold=0.9
)

dim_pr = analyzer.estimate_intrinsic_dimensionality(
    trajectories,
    method='participation_ratio'
)

dim_mle = analyzer.estimate_intrinsic_dimensionality(
    trajectories,
    method='mle'
)

print(f"Intrinsic dimensionality estimates:")
print(f"  PCA (90% var):        {dim_pca}")
print(f"  Participation Ratio:  {dim_pr:.1f}")
print(f"  MLE:                  {dim_mle:.1f}")
```

#### Part 5: Controllability Analysis
```python
# Compute controllability matrix
controllability = analyzer.compute_controllability(
    model=rnn,
    input_dim=10,
    rank_threshold=0.99
)

print(f"System is controllable: {controllability['is_controllable']}")
print(f"Controllability rank: {controllability['rank']}")

# Find control neurons (most controllable dimensions)
control_neurons = controllability['most_controllable_neurons'][:10]
print(f"Most controllable neurons: {control_neurons}")

# Visualize controllability Gramian
plt.imshow(controllability['gramian'], cmap='viridis')
plt.colorbar(label='Controllability')
plt.title('Controllability Gramian')
plt.xlabel('State Dimension')
plt.ylabel('State Dimension')
plt.show()
```

### Complete Analysis Pipeline

```python
class DynamicsAnalysisPipeline:
    def __init__(self, model):
        self.model = model
        self.analyzer = DynamicsAnalyzer()

    def run_full_analysis(self, trajectories):
        results = {}

        # 1. Koopman analysis
        print("1. Koopman operator analysis...")
        koopman = self.analyzer.estimate_koopman_operator(trajectories)
        results['koopman'] = koopman

        # 2. Chaos analysis
        print("2. Lyapunov exponents...")
        lyapunov = self.analyzer.compute_lyapunov_exponents(trajectories)
        results['lyapunov'] = lyapunov
        results['is_chaotic'] = lyapunov[0] > 0

        # 3. Fixed points
        print("3. Finding fixed points...")
        fixed_points = self.analyzer.find_fixed_points(self.model)
        results['fixed_points'] = fixed_points

        # 4. Dimensionality
        print("4. Intrinsic dimensionality...")
        dim = self.analyzer.estimate_intrinsic_dimensionality(trajectories)
        results['dimensionality'] = dim

        # 5. Controllability
        print("5. Controllability...")
        control = self.analyzer.compute_controllability(self.model)
        results['controllability'] = control

        # Generate report
        self.print_report(results)

        return results

    def print_report(self, results):
        print("\n" + "="*50)
        print("DYNAMICAL SYSTEMS ANALYSIS REPORT")
        print("="*50)
        print(f"Largest Lyapunov:     {results['lyapunov'][0]:.4f}")
        print(f"Is Chaotic:           {results['is_chaotic']}")
        print(f"# Fixed Points:       {len(results['fixed_points'])}")
        print(f"Intrinsic Dimension:  {results['dimensionality']:.1f}")
        print(f"Is Controllable:      {results['controllability']['is_controllable']}")
        print("="*50)
```

### Exercises

1. **Compare RNN Architectures**: Analyze GRU vs LSTM vs vanilla RNN dynamics
2. **Task-Dependent Dynamics**: How do fixed points change for different tasks?
3. **Dimensionality vs Performance**: Relate intrinsic dimensionality to task performance

### Key Papers
- Sussillo & Barak (2013): "Opening the Black Box"
- Brunton et al. (2016): "Discovering governing equations from data"
- Koopman (1931): "Hamiltonian Systems and Transformation in Hilbert Space"

---

## Quick Reference for Notebooks 07-10

Due to space, here are condensed outlines for the remaining notebooks. Request full implementations as needed!

### Notebook 07: Circuit Extraction

**Topics**:
- Latent RNN models (Langdon & Engel 2025)
- DUNL sparse coding for mixed selectivity
- Feature visualization via activation maximization
- Circuit motif detection

**Key Code**:
```python
from neuros_mechint.circuits import LatentCircuitModel, CircuitFitter

# Extract minimal RNN circuit
circuit = LatentCircuitModel(latent_dim=10, enforce_dales_law=True)
fitter = CircuitFitter(circuit)
fitter.fit(neural_data)

# Analyze circuit
circuit.find_fixed_points()
circuit.visualize_ei_diagram()
```

### Notebook 08: Biophysical Modeling

**Topics**:
- Spiking neural networks (LIF, Izhikevich, Hodgkin-Huxley)
- Dale's law enforcement (E/I separation)
- STDP and synaptic plasticity
- Training biophysically-constrained networks

**Key Code**:
```python
from neuros_mechint.biophysical import SpikingNeuralNetwork, DalesLinear, STDP

# Spiking network
snn = SpikingNeuralNetwork(
    input_dim=100,
    hidden_dims=[200, 200],
    output_dim=10,
    neuron_type='lif'
)

# Dale's law layer
ei_layer = DalesLinear(in_features=100, out_features=200, ei_ratio=0.8)
```

### Notebook 09: Information Theory

**Topics**:
- Mutual information estimation (MINE)
- Tishby's information plane
- Energy landscape mapping
- Entropy production

**Key Code**:
```python
from neuros_mechint.energy_flow import InformationFlowAnalyzer, EnergyLandscape

# Information plane
analyzer = InformationFlowAnalyzer()
trajectory = analyzer.compute_information_plane_trajectory(
    model, train_data, epochs=100
)

# Energy landscape
landscape = EnergyLandscape()
landscape.estimate_energy_function(neural_states)
basins = landscape.detect_basins()
```

### Notebook 10: Advanced Topics

**Topics**:
- Meta-dynamics during training
- Manifold geometry and topology
- Counterfactual interventions
- Feature attribution methods
- Automated reporting

**Key Code**:
```python
from neuros_mechint import MetaDynamicsTracker, ManifoldAnalyzer

# Track training
tracker = MetaDynamicsTracker()
tracker.track_epoch(model, epoch_data)
phases = tracker.detect_phases()

# Topology
topology = TopologyAnalyzer()
betti_numbers = topology.compute_persistent_homology(representations)
```

---

## Getting Started

1. **Start with notebooks 01-02** (already complete!)
2. **Use this guide** to understand what's in notebooks 03-10
3. **Request specific notebooks** if you want full detailed implementations
4. **Explore the library** using the code examples above

## Need More Detail?

If you want full implementations of any notebooks 03-10, just ask! I can create them with:
- Complete explanations
- Extensive code examples
- Visualizations
- Exercises
- References

Happy learning! 🚀
