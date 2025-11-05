# neuros-mechint: Mechanistic Interpretability Toolkit

**The world's most comprehensive mechanistic interpretability suite for neural networks.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Development](https://img.shields.io/badge/status-active%20development-brightgreen.svg)]()

## 🎯 Overview

`neuros-mechint` provides a unified framework for understanding how neural networks compute. From sparse autoencoders to thermodynamic analysis, from circuit discovery to biophysical modeling—everything you need for mechanistic interpretability research.

### Key Features

- **🔬 Sparse Autoencoders**: Decompose polysemantic neurons into interpretable features
- **⚡ Circuit Discovery**: Extract minimal computational circuits via ACDC and path patching
- **🌊 Energy Flow Analysis**: Thermodynamics of computation, Landauer's principle, NESS
- **🧠 Brain Alignment**: Compare model representations to neural recordings (CCA, RSA, Procrustes)
- **📊 Fractal Analysis**: Scale-free dynamics, 1/f noise, temporal fractals
- **🔄 Dynamical Systems**: Koopman operators, Lyapunov exponents, Neural ODEs
- **🎭 Counterfactuals**: Causal interventions, latent surgery, do-calculus
- **🏗️ Topology & Geometry**: Manifold analysis, persistent homology, curvature

## 📦 Installation

```bash
# From source (recommended for development)
git clone https://github.com/your-org/neurOS-v1.git
cd neurOS-v1/packages/neuros-mechint
pip install -e .

# Or via pip (when published)
pip install neuros-mechint
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn
- Matplotlib, seaborn (visualization)
- Optional: ripser (topology), persim (persistence diagrams)

## 🚀 Quick Start

```python
import torch
from neuros_mechint import SparseAutoencoder
from neuros_mechint.circuits import AutomatedCircuitDiscovery
from neuros_mechint.energy_flow import LandauerAnalyzer

# 1. Sparse Autoencoder Analysis
model = ... # Your PyTorch model
activations = ... # Collect activations

sae = SparseAutoencoder(input_dim=512, latent_dim=2048, sparsity=0.05)
sae_result = sae.train(activations, epochs=100)
features = sae.encode(activations)  # Interpretable features

# 2. Circuit Discovery
acdc = AutomatedCircuitDiscovery(model, importance_threshold=0.01)
circuit = acdc.discover_circuit(inputs, targets)
print(f"Discovered circuit: {len(circuit.edges)} edges")

# 3. Thermodynamic Analysis
landauer = LandauerAnalyzer(model, temperature=300)
result = landauer.analyze_forward_pass(inputs)
print(f"Bits erased: {result.total_bits_erased:.0f}")
print(f"Minimum energy: {result.minimum_energy_joules:.2e} J")
```

## 📚 Documentation & Examples

### 22 Comprehensive Jupyter Notebooks

We provide extensive tutorial notebooks covering all aspects of mechanistic interpretability:

**Foundation (Notebooks 1-6)**
- 01: Introduction and Quickstart
- 02: Sparse Autoencoders
- 03: Causal Interventions
- 04: Fractal Analysis
- 05: Brain Alignment
- 06: Dynamical Systems

**Advanced Analysis (Notebooks 7-16)**
- 07: Circuit Extraction & Latent Models
- 08: Biophysical Modeling
- 09: Information Theory
- 10: Advanced Topics (Meta-dynamics, Geometry)
- 11: Path Patching & ACDC
- 12: Thermodynamic Analysis
- 13: Circuit Comparison & Motifs
- 14: Neural ODEs & Slow Features
- 15: Energy Cascades & Hamiltonian
- 16: Pipeline & Database Integration

**Specialized Topics (Notebooks 17-22)**
- 17: Advanced Biophysical Modeling
- 18: Intervention Strategies
- 19: Cross-Species Alignment
- 20: Temporal Dynamics
- 21: Criticality Analysis
- 22: Multifractal Analysis

### Running the Notebooks

```bash
cd examples/
jupyter lab
# Open 01_introduction_and_quickstart.ipynb to begin
```

## 🧩 Module Overview

### Core Modules

```python
neuros_mechint/
├── sae/                          # Sparse autoencoder analysis
├── circuits/                     # Circuit discovery & analysis
├── energy_flow/                  # Thermodynamics & energy analysis
├── dynamics/                     # Dynamical systems analysis
├── alignment/                    # Brain-model alignment
├── fractals/                     # Fractal & scale-free analysis
├── biophysical/                  # Biophysical neuron models
├── interventions/                # Causal interventions
├── counterfactuals/              # Counterfactual analysis
├── meta_dynamics/                # Training trajectory analysis
├── geometry_topology/            # Manifold & topology analysis
└── visualization/                # Visualization tools
```

### Key Classes

**Sparse Autoencoders**
```python
from neuros_mechint.sae import SparseAutoencoder, SAETrainer, SAEVisualizer
```

**Circuit Discovery**
```python
from neuros_mechint.circuits import (
    AutomatedCircuitDiscovery,  # ACDC algorithm
    PathPatcher,                 # Activation patching
    CircuitComparator,          # Compare circuits
    MotifDetector              # Find recurring patterns
)
```

**Thermodynamics**
```python
from neuros_mechint.energy_flow import (
    LandauerAnalyzer,           # Landauer's principle
    NESSAnalyzer,               # Non-equilibrium steady states
    FluctuationTheoremAnalyzer, # Fluctuation theorems
    EnergyCascadeAnalyzer,      # Energy flow through layers
    HamiltonianDecomposer       # Conservative vs dissipative
)
```

**Dynamics**
```python
from neuros_mechint.dynamics import (
    NeuralODEIntegrator,        # Continuous-time dynamics
    SlowFeatureAnalyzer,        # Temporal hierarchies
    KoopmanAnalyzer,           # Koopman operator
    LyapunovAnalyzer           # Stability analysis
)
```

**Brain Alignment**
```python
from neuros_mechint.alignment import (
    CCAAlignment,               # Canonical correlation
    RSAAnalyzer,               # Representational similarity
    ProcrustesAlignment        # Orthogonal alignment
)
```

## 🎓 Research Applications

### Interpretability Research
- Understand how transformers process language
- Discover computational circuits in vision models
- Map features to human-interpretable concepts

### Neuroscience
- Compare artificial and biological neural networks
- Design better brain-computer interfaces
- Test theories of neural computation

### Safety & Alignment
- Detect deceptive or unintended behaviors
- Verify model reasoning
- Build more transparent AI systems

### Architecture Design
- Design networks with desired computational properties
- Optimize for energy efficiency
- Create biologically-inspired architectures

## 📊 Results & Database

Store and compare analysis results:

```python
from neuros_mechint.database import MechIntDatabase

db = MechIntDatabase(root_dir="./results")

# Store analysis
result_id = db.store(circuit_result, tags=["gpt2", "layer3", "acdc"])

# Query results
results = db.query(tags=["acdc"], method="ACDC")

# Compare across experiments
comparison = db.compare_results([id1, id2, id3])
```

## 🔬 Advanced Features

### Thermodynamics of Computation
```python
# Landauer's Principle: minimum energy per bit erased
landauer = LandauerAnalyzer(model)
result = landauer.analyze_forward_pass(inputs)

# Non-Equilibrium Steady States
ness = NESSAnalyzer(model)
steady_state = ness.analyze_steady_state(inputs)

# Fluctuation Theorems
ft = FluctuationTheoremAnalyzer(model)
crooks = ft.test_crooks_theorem(forward_data, reverse_data)
```

### Counterfactual Interventions
```python
from neuros_mechint.counterfactuals import LatentSurgery, SyntheticLesions

# Modify specific neurons
surgery = LatentSurgery(model)
original, intervened = surgery.intervene(x, "layer2", neuron_idx=42, value=1.0)

# Identify critical components
lesion = SyntheticLesions(model)
critical_neurons = lesion.identify_critical_neurons(x, "layer3")
```

### Meta-Dynamics
```python
from neuros_mechint.meta_dynamics import RepresentationalTrajectory

# Track representations during training
trajectory = RepresentationalTrajectory(checkpoints, data)
phases = trajectory.detect_phases()  # Fitting vs compression
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/neurOS-v1.git
cd neurOS-v1/packages/neuros-mechint

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check code quality
black src/
flake8 src/
mypy src/
```

## 📖 Citation

If you use `neuros-mechint` in your research, please cite:

```bibtex
@software{neuros_mechint2025,
  title = {neuros-mechint: Comprehensive Mechanistic Interpretability Toolkit},
  author = {neurOS Team},
  year = {2025},
  url = {https://github.com/your-org/neurOS-v1}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

Built on foundational work from:
- Anthropic's mechanistic interpretability research
- OpenAI's sparse autoencoder work
- The Distill.pub team's visualization techniques
- Computational neuroscience community

## 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/your-org/neurOS-v1/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/neurOS-v1/discussions)
- **Email**: neuros@example.com

---

## 🗺️ Roadmap

See [CHANGELOG.md](CHANGELOG.md) for version history and [docs/planning/](docs/planning/) for future plans.

### Current Status: Phase 2 Complete ✅

- ✅ Core SAE, circuits, alignment, fractals
- ✅ Thermodynamics & energy flow analysis
- ✅ Advanced dynamics (Neural ODEs, slow features)
- ✅ Counterfactuals & causal interventions
- ✅ 22 comprehensive tutorial notebooks
- ✅ Complete API documentation
- ⏳ Package reorganization (planned)
- ⏳ Performance optimization (planned)
- ⏳ Extended test coverage (planned)

### Future Enhancements

- Real-time analysis dashboard
- Integration with popular frameworks (HuggingFace, PyTorch Lightning)
- Cloud-based analysis pipeline
- Multi-GPU support for large models
- Extended model zoo with pre-trained analyzers

---

**Built with ❤️ by the neurOS team**

*Making neural networks interpretable, one circuit at a time.*
