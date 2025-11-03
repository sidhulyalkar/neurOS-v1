# Phase 2 Features: Comprehensive Guide

Welcome to the Phase 2 features guide! This document provides a comprehensive overview of the advanced mechanistic interpretability techniques introduced in **neuros-mechint** Phase 2.

## Table of Contents

1. [Overview](#overview)
2. [Circuit Discovery & Comparison](#circuit-discovery--comparison)
3. [Thermodynamic Analysis](#thermodynamic-analysis)
4. [Continuous-Time Dynamics](#continuous-time-dynamics)
5. [Infrastructure & Workflows](#infrastructure--workflows)
6. [Integration Patterns](#integration-patterns)
7. [Best Practices](#best-practices)
8. [Research Applications](#research-applications)

---

## Overview

### What's New in Phase 2?

Phase 2 extends **neuros-mechint** with cutting-edge research methods for automated circuit discovery, thermodynamic analysis, continuous dynamics, and large-scale experiment management.

**Key Additions**:
- **15 new analyzer classes** across 4 major categories
- **Automated circuit discovery** (ACDC algorithm)
- **Thermodynamic principles** (Landauer, fluctuation theorems)
- **Neural ODEs** for continuous-time dynamics
- **Database & pipeline infrastructure** for experiments at scale

### Phase 2 Components Summary

| Component | Purpose | Key Classes | Notebook |
|-----------|---------|-------------|----------|
| Path Patching | Direct/indirect effect decomposition | `PathPatcher` | 11 |
| ACDC | Automated circuit pruning | `AutomatedCircuitDiscovery` | 11 |
| Circuit Comparison | Cross-model similarity | `CircuitComparator` | 13 |
| Motif Detection | Structural pattern finding | `MotifDetector` | 13 |
| Landauer Analysis | Minimum energy bounds | `LandauerAnalyzer` | 12 |
| NESS Analysis | Non-equilibrium steady states | `NESSAnalyzer` | 12 |
| Fluctuation Theorems | Thermodynamic symmetries | `FluctuationTheoremAnalyzer` | 12 |
| Energy Cascades | Hierarchical energy flow | `EnergyCascadeAnalyzer` | 15 |
| Hamiltonian Decomposition | Conservative/dissipative split | `HamiltonianDecomposer` | 15 |
| Neural ODE Integration | Continuous dynamics | `NeuralODEIntegrator` | 14 |
| Slow Feature Analysis | Temporal hierarchies | `SlowFeatureAnalyzer` | 14 |
| MechIntDatabase | Result storage & queries | `MechIntDatabase` | 16 |
| MechIntPipeline | Multi-stage workflows | `MechIntPipeline` | 16 |

---

## Circuit Discovery & Comparison

### Path Patching (`PathPatcher`)

**Purpose**: Decompose information flow into direct and indirect effects.

**Key Concept**: Replace activations at a specific layer with "clean" vs "corrupted" inputs to measure causal impact.

**Basic Usage**:
```python
from neuros_mechint.circuits import PathPatcher

patcher = PathPatcher(model=model, device=device)

result = patcher.patch_all_paths(
    clean_input=clean_input,
    corrupted_input=corrupted_input
)

print(f"Direct effects: {result.direct_effects}")
print(f"Indirect effects: {result.indirect_effects}")
print(f"Total effects: {result.total_effects}")
```

**Interpretation**:
- **Direct effect**: Activation change when only this layer receives clean input
- **Indirect effect**: Change due to upstream layers
- **Total effect**: Combined impact on output

**Notebook**: [11_path_patching_and_acdc.ipynb](11_path_patching_and_acdc.ipynb)

### Automated Circuit Discovery (ACDC)

**Purpose**: Automatically find minimal circuits through iterative edge pruning.

**Algorithm**:
1. Start with full computational graph
2. Ablate each edge and measure impact on target output
3. Prune edges below importance threshold
4. Iterate until convergence

**Basic Usage**:
```python
from neuros_mechint.circuits import AutomatedCircuitDiscovery

acdc = AutomatedCircuitDiscovery(
    model=model,
    importance_threshold=0.01,  # Edges below this are pruned
    device=device
)

circuit = acdc.discover_circuit(
    inputs=sample_inputs,
    targets=sample_labels,
    max_iterations=20
)

print(f"Circuit has {len(circuit.nodes)} nodes, {len(circuit.edges)} edges")
print(f"Converged in {circuit.n_iterations} iterations")
```

**Parameters**:
- `importance_threshold`: Lower = more edges retained (more complete circuit)
- `max_iterations`: Maximum pruning iterations
- `edge_ablation_strategy`: 'zero', 'mean', or 'resample'

**Notebook**: [11_path_patching_and_acdc.ipynb](11_path_patching_and_acdc.ipynb)

### Circuit Comparator

**Purpose**: Quantify similarity between circuits from different models.

**Metrics**:
- **Node overlap**: Fraction of shared computational units
- **Edge overlap**: Fraction of shared connections
- **Structural similarity**: Graph edit distance
- **Overall similarity score**: Weighted combination

**Basic Usage**:
```python
from neuros_mechint.circuits import CircuitComparator
from neuros_mechint.database import MechIntDatabase

# Store circuits in database
db = MechIntDatabase(root_dir='./results')
circuit_a_id = db.store(circuit_a, tags=['model_a', 'task_1'])
circuit_b_id = db.store(circuit_b, tags=['model_b', 'task_1'])

# Compare
comparator = CircuitComparator(database=db)
comparison = comparator.compare_circuits(circuit_a_id, circuit_b_id)

print(f"Similarity score: {comparison.similarity_score:.3f}")
print(f"Node overlap: {comparison.node_overlap:.3f}")
print(f"Edge overlap: {comparison.edge_overlap:.3f}")
```

**Notebook**: [13_circuit_comparison_and_motifs.ipynb](13_circuit_comparison_and_motifs.ipynb)

### Motif Detector

**Purpose**: Find recurring structural patterns (motifs) in circuits.

**Motif Types**:
- **Feedforward**: A → B → C (sequential processing)
- **Recurrent**: A → B → A (feedback loops)
- **Skip**: A → C (bypassing intermediate layers)
- **Convergent**: A → C ← B (multi-input integration)
- **Divergent**: B ← A → C (broadcast)
- **Triangle**: A ↔ B ↔ C ↔ A (fully connected trio)

**Basic Usage**:
```python
from neuros_mechint.circuits import MotifDetector

detector = MotifDetector(
    circuit=circuit,
    n_random_samples=100  # For significance testing
)

motifs = detector.detect_all_motifs(compute_significance=True)

for motif_type, count in motifs.motif_counts.items():
    z_score = motifs.z_scores[motif_type]
    p_value = motifs.p_values[motif_type]
    print(f"{motif_type}: {count} instances (Z={z_score:.2f}, p={p_value:.4f})")
```

**Interpretation**:
- **Z-score > 2**: Motif is significantly overrepresented
- **Z-score < -2**: Motif is significantly underrepresented
- **p-value < 0.05**: Statistically significant

**Notebook**: [13_circuit_comparison_and_motifs.ipynb](13_circuit_comparison_and_motifs.ipynb)

---

## Thermodynamic Analysis

### Landauer's Principle

**Physical Law**: Every irreversible bit erasure requires at minimum **kT ln(2)** energy dissipation.

**Minimum Energy**: E_min = kT ln(2) ≈ 2.87 × 10⁻²¹ J at T=300K

**Basic Usage**:
```python
from neuros_mechint.energy_flow import LandauerAnalyzer

landauer = LandauerAnalyzer(
    model=model,
    temperature=300.0,  # Kelvin
    device=device
)

result = landauer.analyze_forward_pass(inputs=sample_inputs)

print(f"Total bits erased: {result.total_bits_erased:.2f}")
print(f"Minimum energy: {result.total_min_energy:.2e} J")
print(f"Reversibility score: {result.reversibility_score:.3f}")

# Per-layer analysis
for layer_name, bits in result.layer_bits_erased.items():
    energy = result.layer_min_energy[layer_name]
    print(f"{layer_name}: {bits:.2f} bits → {energy:.2e} J")
```

**Interpretation**:
- **Bits erased**: Measure of information loss
- **Reversibility score**: 1.0 = fully reversible, 0.0 = maximally irreversible
- **Per-layer costs**: Identify expensive operations

**Notebook**: [12_thermodynamic_analysis.ipynb](12_thermodynamic_analysis.ipynb)

### Non-Equilibrium Steady States (NESS)

**Purpose**: Analyze thermodynamics of recurrent networks in steady state.

**Key Concepts**:
- **Entropy production rate**: How fast system creates entropy
- **Fluctuation-dissipation ratio**: Deviation from equilibrium
- **Effective temperature**: Internal "heat" of dynamics

**Basic Usage**:
```python
from neuros_mechint.energy_flow import NESSAnalyzer

ness = NESSAnalyzer(
    model=rnn_model,
    temperature=300.0,
    device=device
)

result = ness.analyze_steady_state(
    inputs=sequential_inputs,
    n_samples=100  # Trajectory samples
)

print(f"Entropy production rate: {result.entropy_production_rate:.3e} J/K/s")
print(f"FD ratio: {result.FD_ratio:.3f}")
print(f"Effective temperature: {result.effective_temperature:.2f} K")
```

**Interpretation**:
- **Entropy production > 0**: System is out of equilibrium
- **FD ratio ≈ 1**: Near-equilibrium behavior
- **Effective T > physical T**: System is "hotter" than environment

**Notebook**: [12_thermodynamic_analysis.ipynb](12_thermodynamic_analysis.ipynb)

### Fluctuation Theorems

**Purpose**: Test fundamental thermodynamic symmetries in neural networks.

**Four Theorems**:
1. **Crooks Theorem**: P(σ=A) / P(σ=-A) = exp(A)
2. **Jarzynski Equality**: ⟨exp(-W/kT)⟩ = exp(-ΔF/kT)
3. **Fluctuation-Dissipation Theorem**: Response = susceptibility × fluctuations
4. **Integral Fluctuation Theorem**: ⟨exp(-Δs_tot)⟩ = 1

**Basic Usage**:
```python
from neuros_mechint.energy_flow import FluctuationTheoremAnalyzer

ft_analyzer = FluctuationTheoremAnalyzer(
    model=model,
    temperature=300.0,
    device=device
)

# Test all four theorems
result = ft_analyzer.test_all_theorems(
    forward_data=forward_trajectories,
    reverse_data=reverse_trajectories,
    perturbation=small_perturbation
)

print(f"Crooks satisfied: {result.crooks_satisfied}")
print(f"Jarzynski error: {result.jarzynski_error:.3e}")
print(f"FDT R²: {result.fdt_r2:.3f}")
print(f"IFT violation: {result.ift_violation:.3e}")
```

**Notebook**: [12_thermodynamic_analysis.ipynb](12_thermodynamic_analysis.ipynb)

### Energy Cascades

**Purpose**: Track energy flow through network layers, analogous to turbulent cascades.

**Key Concepts**:
- **Input/Output energy**: Energy (variance) entering and leaving each layer
- **Dissipation**: Energy lost (information erasure)
- **Transfer efficiency**: Fraction of energy passed to next layer
- **Cascade exponent**: Power-law scaling (Kolmogorov: -5/3)

**Basic Usage**:
```python
from neuros_mechint.energy_flow import EnergyCascadeAnalyzer

cascade = EnergyCascadeAnalyzer(
    model=model,
    energy_metric='variance',  # or 'frobenius', 'nuclear'
    track_spectrum=True
)

result = cascade.analyze_cascade(inputs=sample_inputs)

for layer_name in result.layer_input_energy.keys():
    e_in = result.layer_input_energy[layer_name]
    e_out = result.layer_output_energy[layer_name]
    dissipation = result.layer_dissipation[layer_name]
    efficiency = result.transfer_efficiency[layer_name]

    print(f"{layer_name}:")
    print(f"  E_in={e_in:.2f}, E_out={e_out:.2f}, Dissipation={dissipation:.2f}")
    print(f"  Efficiency={efficiency:.2%}")

print(f"Cascade exponent: {result.cascade_exponent:.3f}")
```

**Interpretation**:
- **Dissipation > 0**: Layer is lossy (typical for ReLU)
- **Efficiency < 1**: Energy is being lost
- **Cascade exponent ≈ -5/3**: Kolmogorov-like turbulent cascade

**Notebook**: [15_energy_cascades_and_hamiltonian.ipynb](15_energy_cascades_and_hamiltonian.ipynb)

### Hamiltonian Decomposition

**Purpose**: Separate dynamics into conservative (energy-preserving) and dissipative (energy-losing) components.

**Mathematical Framework**: Helmholtz decomposition
- **F_total** = **F_conservative** + **F_dissipative**
- Conservative: ∇ × F_cons = 0 (curl-free)
- Dissipative: ∇ · F_diss > 0 (divergence)

**Basic Usage**:
```python
from neuros_mechint.energy_flow import HamiltonianDecomposer

decomposer = HamiltonianDecomposer(
    model=dynamics_model,
    dt=0.01,
    method='helmholtz'  # or 'finite_difference'
)

result = decomposer.decompose_dynamics(
    initial_states=initial_conditions,
    n_timesteps=500
)

print(f"Hamiltonian fraction: {result.hamiltonian_fraction:.2%}")
print(f"Dissipation fraction: {result.dissipation_fraction:.2%}")
print(f"Phase space volume change: {result.phase_space_volume:.3f}")
```

**Interpretation**:
- **Hamiltonian fraction → 1**: Dynamics are mostly conservative (reversible)
- **Dissipation fraction → 1**: Dynamics are mostly dissipative (irreversible)
- **Volume < 1**: Phase space is contracting (attractor dynamics)

**Notebook**: [15_energy_cascades_and_hamiltonian.ipynb](15_energy_cascades_and_hamiltonian.ipynb)

---

## Continuous-Time Dynamics

### Neural ODE Integration

**Purpose**: Treat neural networks as continuous dynamical systems: dx/dt = f_θ(x, t)

**Integration Methods**:
- **Euler**: Fast, first-order accurate
- **RK4**: Runge-Kutta 4th order, good balance
- **dopri5**: Adaptive Dormand-Prince, high accuracy

**Basic Usage**:
```python
from neuros_mechint.dynamics import NeuralODEIntegrator

integrator = NeuralODEIntegrator(
    model=dynamics_model,
    dt=0.01,
    method='rk4'
)

# Integrate single trajectory
result = integrator.integrate_trajectory(
    initial_state=x0,
    time_span=(0, 10),
    n_points=1000
)

print(f"Trajectory shape: {result.trajectory.shape}")
print(f"Integration method: {result.method}")

# Analyze flow field
flow_result = integrator.analyze_flow_field(
    state_space_bounds=(-3, 3),
    n_points=20
)

print(f"Fixed points found: {len(flow_result.fixed_points)}")
```

**Applications**:
- Continuous-time RNN analysis
- Phase portrait visualization
- Fixed point detection
- Stability analysis

**Notebook**: [14_neural_ode_and_slow_features.ipynb](14_neural_ode_and_slow_features.ipynb)

### Slow Feature Analysis (SFA)

**Purpose**: Discover slow-varying features in temporal data.

**Principle**: min Δ(y) = min ⟨ẏ²⟩ subject to constraints
- Slower features capture higher-level abstractions
- Fast features encode transient details

**Basic Usage**:
```python
from neuros_mechint.dynamics import SlowFeatureAnalyzer

sfa = SlowFeatureAnalyzer(
    expansion_degree=2,  # Polynomial expansion
    whitening=True       # Decorrelate features
)

result = sfa.analyze_timeseries(
    activations=temporal_activations,
    n_slow_features=10
)

print(f"Delta values (slowness): {result.delta_values}")
print(f"Characteristic times: {result.characteristic_times}")

# Slowest feature
slowest_feature = result.slow_features[:, 0]
slowest_time = result.characteristic_times[0]
print(f"Slowest feature has timescale: {slowest_time:.2f} timesteps")
```

**Interpretation**:
- **Small δ**: Feature changes slowly (high-level)
- **Large δ**: Feature changes quickly (low-level)
- **Characteristic time**: How long feature persists

**Notebook**: [14_neural_ode_and_slow_features.ipynb](14_neural_ode_and_slow_features.ipynb)

---

## Infrastructure & Workflows

### MechIntDatabase

**Purpose**: Efficient storage and retrieval of mechanistic interpretability results.

**Architecture**:
- **HDF5**: Large arrays (activations, trajectories)
- **SQLite**: Metadata, tags, timestamps
- **Content hashing**: SHA256 for automatic deduplication

**Basic Usage**:
```python
from neuros_mechint.database import MechIntDatabase

# Create database
db = MechIntDatabase(root_dir='./mechint_results')

# Store result
result_id = db.store(
    result=analysis_result,
    tags=['experiment_1', 'transformer', 'acdc']
)

print(f"Stored with ID: {result_id}")

# Query by tags
all_acdc = db.query(tags=['acdc'])
print(f"Found {len(all_acdc)} ACDC results")

# Load result
loaded_result = db.load(result_id)

# Get tags
tags = db.get_tags(result_id)
print(f"Tags: {tags}")

# List all results
all_results = db.list_all()
print(f"Total results: {len(all_results)}")
```

**Best Practices**:
- Use hierarchical tags: `[experiment_name, model_type, analysis_type]`
- Query before computing to check for existing results
- Regularly archive old experiments

**Notebook**: [16_pipeline_and_database.ipynb](16_pipeline_and_database.ipynb)

### MechIntPipeline

**Purpose**: Orchestrate multi-stage analysis workflows with checkpointing.

**Pipeline Modes**:
- **Quick** (~5 min): Basic circuits + energy
- **Standard** (~15 min): + Thermodynamics + Dynamics
- **Comprehensive** (~30+ min): All analyses + full visualization
- **Custom**: User-defined stages

**Basic Usage**:
```python
from neuros_mechint.pipeline import MechIntPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    mode='standard',
    enable_circuits=True,
    enable_energy=True,
    enable_thermodynamics=True,
    enable_dynamics=True,
    enable_visualization=True,
    save_checkpoints=True
)

pipeline = MechIntPipeline(
    config=config,
    database=db,
    device=device
)

# Run pipeline
results = pipeline.run(
    model=model,
    inputs=inputs,
    targets=targets,
    experiment_name='my_experiment'
)

print(f"Completed stages: {list(results.keys())}")

# Resume from checkpoint (if interrupted)
if pipeline.has_checkpoint('my_experiment'):
    results = pipeline.resume_from_checkpoint('my_experiment')

# Generate report
report = pipeline.generate_report(
    results=results,
    experiment_name='my_experiment'
)
print(report)
```

**Notebook**: [16_pipeline_and_database.ipynb](16_pipeline_and_database.ipynb)

---

## Integration Patterns

### Pattern 1: Automated Circuit Discovery → Motif Analysis

```python
# Discover circuit
acdc = AutomatedCircuitDiscovery(model=model, device=device)
circuit = acdc.discover_circuit(inputs, targets, max_iterations=15)

# Detect motifs
detector = MotifDetector(circuit=circuit, n_random_samples=100)
motifs = detector.detect_all_motifs(compute_significance=True)

# Identify significant patterns
for motif_type, z_score in motifs.z_scores.items():
    if abs(z_score) > 2:
        count = motifs.motif_counts[motif_type]
        print(f"Significant {motif_type}: {count} instances (Z={z_score:.2f})")
```

### Pattern 2: Cross-Model Circuit Comparison

```python
# Discover circuits for multiple models
db = MechIntDatabase(root_dir='./results')
circuit_ids = []

for model_name, model in models.items():
    acdc = AutomatedCircuitDiscovery(model=model, device=device)
    circuit = acdc.discover_circuit(inputs, targets)
    circuit_id = db.store(circuit, tags=[model_name, 'task_1'])
    circuit_ids.append((model_name, circuit_id))

# Compare all pairs
comparator = CircuitComparator(database=db)
for i, (name_a, id_a) in enumerate(circuit_ids):
    for name_b, id_b in circuit_ids[i+1:]:
        comparison = comparator.compare_circuits(id_a, id_b)
        print(f"{name_a} vs {name_b}: similarity={comparison.similarity_score:.3f}")
```

### Pattern 3: Thermodynamic Efficiency Analysis

```python
# Analyze energy costs
landauer = LandauerAnalyzer(model=model, temperature=300, device=device)
landauer_result = landauer.analyze_forward_pass(inputs)

# Analyze energy cascades
cascade = EnergyCascadeAnalyzer(model=model, energy_metric='variance')
cascade_result = cascade.analyze_cascade(inputs)

# Combine insights
total_cost = landauer_result.total_min_energy
efficiency = np.mean(list(cascade_result.transfer_efficiency.values()))
reversibility = landauer_result.reversibility_score

print(f"Thermodynamic profile:")
print(f"  Total cost: {total_cost:.2e} J")
print(f"  Avg efficiency: {efficiency:.2%}")
print(f"  Reversibility: {reversibility:.3f}")
```

### Pattern 4: Complete Phase 2 Pipeline

```python
# Use pipeline for comprehensive analysis
config = PipelineConfig(mode='comprehensive', save_checkpoints=True)
pipeline = MechIntPipeline(config=config, database=db, device=device)

# Run on multiple model variants
for model_name, model in models.items():
    print(f"Analyzing {model_name}...")
    results = pipeline.run(
        model=model,
        inputs=inputs,
        targets=targets,
        experiment_name=f'comparison_{model_name}'
    )
    print(f"  Completed {len(results)} stages")

# Query and compare results
all_landauer = db.query(tags=['landauer'])
for result_id in all_landauer:
    result = db.load(result_id)
    tags = db.get_tags(result_id)
    model_name = [t for t in tags if t.startswith('comparison_')][0]
    print(f"{model_name}: {result.total_min_energy:.2e} J")
```

---

## Best Practices

### Circuit Discovery

1. **Start with high threshold**: Begin with `importance_threshold=0.1`, then decrease
2. **Use appropriate ablation**: `'zero'` for ReLU, `'mean'` for LayerNorm
3. **Validate with path patching**: Cross-check ACDC results with direct path analysis
4. **Visualize circuits**: Use EnhancedVisualizer to inspect discovered circuits
5. **Compare across seeds**: Run multiple times to assess stability

### Thermodynamic Analysis

1. **Use correct temperature**: Match experimental conditions (300K typical)
2. **Check energy conservation**: Verify input energy ≈ output energy + dissipation
3. **Analyze per-layer costs**: Identify inefficient operations
4. **Compare to Landauer bound**: Assess reversibility potential
5. **Consider batch effects**: Average over multiple batches for stability

### Dynamics Analysis

1. **Choose appropriate dt**: Smaller for stiff systems, larger for smooth dynamics
2. **Use adaptive methods**: `dopri5` for high accuracy, `rk4` for balance
3. **Validate fixed points**: Verify ∥f(x*)∥ ≈ 0
4. **Check stability**: Compute eigenvalues of Jacobian at fixed points
5. **Visualize phase space**: 2D/3D projections reveal structure

### Database & Pipeline

1. **Tag strategically**: Use hierarchical tags for easy querying
2. **Enable checkpoints**: Essential for long-running analyses
3. **Query before compute**: Check for existing results
4. **Archive regularly**: Move old experiments to separate database
5. **Use pipeline modes**: Start with `quick`, scale to `comprehensive`

---

## Research Applications

### Application 1: Discovering Universal Circuits

**Goal**: Identify circuits that emerge across different architectures trained on the same task.

**Workflow**:
1. Train multiple architectures (CNN, Transformer, MLP) on same task
2. Run ACDC on each model
3. Store circuits in database with consistent tags
4. Use CircuitComparator to find overlapping structures
5. Use MotifDetector to identify common patterns

**Analysis**:
```python
# Compare circuits
similarities = []
for i, model_a in enumerate(models):
    for model_b in models[i+1:]:
        comparison = comparator.compare_circuits(id_a, id_b)
        similarities.append(comparison.similarity_score)

avg_similarity = np.mean(similarities)
print(f"Average cross-architecture similarity: {avg_similarity:.3f}")

# Universal circuits have high similarity across all models
```

### Application 2: Energy-Efficient Architecture Search

**Goal**: Design architectures that minimize thermodynamic cost while maintaining performance.

**Workflow**:
1. Generate candidate architectures
2. Train and evaluate accuracy
3. Run Landauer + Energy Cascade analysis
4. Compute efficiency metric: accuracy / energy_cost
5. Identify Pareto-optimal architectures

**Analysis**:
```python
# Efficiency frontier
efficiencies = []
for model_name, model in candidates.items():
    accuracy = evaluate_accuracy(model, test_data)
    landauer_result = landauer.analyze_forward_pass(inputs)
    cost = landauer_result.total_min_energy
    efficiency = accuracy / cost
    efficiencies.append((model_name, accuracy, cost, efficiency))

# Sort by efficiency
efficiencies.sort(key=lambda x: x[3], reverse=True)
print("Top 5 efficient architectures:")
for name, acc, cost, eff in efficiencies[:5]:
    print(f"  {name}: acc={acc:.3f}, cost={cost:.2e}J, eff={eff:.2e}")
```

### Application 3: Dynamics-Based Model Selection

**Goal**: Choose RNN architecture based on desired dynamical properties.

**Workflow**:
1. Train candidate RNNs (GRU, LSTM, Vanilla)
2. Integrate dynamics with Neural ODE
3. Compute fixed points and stability
4. Analyze slow features for temporal hierarchies
5. Select model with desired phase space structure

**Analysis**:
```python
# Compare dynamical properties
for model_name, rnn in rnns.items():
    integrator = NeuralODEIntegrator(model=rnn, dt=0.01, method='rk4')
    flow_result = integrator.analyze_flow_field((-2, 2), n_points=15)

    sfa = SlowFeatureAnalyzer(expansion_degree=2)
    sfa_result = sfa.analyze_timeseries(activations, n_slow_features=5)

    print(f"{model_name}:")
    print(f"  Fixed points: {len(flow_result.fixed_points)}")
    print(f"  Slowest timescale: {sfa_result.characteristic_times[0]:.2f}")
```

### Application 4: Interpretable Training Dynamics

**Goal**: Track how circuits and thermodynamics evolve during training.

**Workflow**:
1. Set up training loop with periodic checkpoints
2. At each checkpoint: run ACDC + Landauer analysis
3. Store results in database with epoch tags
4. Query and visualize evolution over time

**Analysis**:
```python
# Training loop
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader)

    if epoch % checkpoint_freq == 0:
        # Discover circuit
        circuit = acdc.discover_circuit(inputs, targets)
        circuit_id = db.store(circuit, tags=['training', f'epoch_{epoch}'])

        # Analyze thermodynamics
        landauer_result = landauer.analyze_forward_pass(inputs)
        landauer_id = db.store(landauer_result, tags=['training', f'epoch_{epoch}'])

# Analyze evolution
circuit_complexity = []
energy_costs = []

for epoch in range(0, num_epochs, checkpoint_freq):
    circuit_ids = db.query(tags=['training', f'epoch_{epoch}', 'circuit'])
    landauer_ids = db.query(tags=['training', f'epoch_{epoch}', 'landauer'])

    circuit = db.load(circuit_ids[0])
    landauer = db.load(landauer_ids[0])

    circuit_complexity.append(len(circuit.edges))
    energy_costs.append(landauer.total_min_energy)

# Plot evolution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(0, num_epochs, checkpoint_freq), circuit_complexity)
plt.xlabel('Epoch')
plt.ylabel('Circuit Complexity (# edges)')
plt.title('Circuit Evolution During Training')

plt.subplot(1, 2, 2)
plt.plot(range(0, num_epochs, checkpoint_freq), energy_costs)
plt.xlabel('Epoch')
plt.ylabel('Energy Cost (J)')
plt.title('Thermodynamic Cost Evolution')
plt.tight_layout()
plt.show()
```

---

## Conclusion

Phase 2 features provide cutting-edge tools for:
- **Automated discovery**: Find circuits without manual specification
- **Cross-model comparison**: Identify universal vs specific structures
- **Physical constraints**: Analyze energy costs and thermodynamic limits
- **Continuous dynamics**: Treat networks as differential equations
- **Scalable workflows**: Manage experiments with database and pipelines

### Next Steps

1. **Explore notebooks 11-16**: Hands-on learning with working examples
2. **Apply to your models**: Use Phase 2 tools on your research
3. **Combine techniques**: Integrate Phase 1 + Phase 2 for comprehensive analysis
4. **Contribute**: Share discoveries, report issues, improve documentation

### Resources

- **Notebooks**: [11](11_path_patching_and_acdc.ipynb), [12](12_thermodynamic_analysis.ipynb), [13](13_circuit_comparison_and_motifs.ipynb), [14](14_neural_ode_and_slow_features.ipynb), [15](15_energy_cascades_and_hamiltonian.ipynb), [16](16_pipeline_and_database.ipynb)
- **Tests**: `tests/test_integration_phase2.py`
- **Documentation**: [README.md](README.md), [START_HERE.md](START_HERE.md)

**Happy discovering! 🔬🧠**
