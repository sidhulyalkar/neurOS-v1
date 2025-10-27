# neuros-mechint: Mechanistic Interpretability Toolbox

**World's most comprehensive mechanistic interpretability suite for neural networks.**

neuros-mechint is a standalone Python package providing cutting-edge tools for understanding, analyzing, and interpreting neural networks through the lens of mechanistic interpretability, fractal geometry, circuit discovery, biophysical modeling, and causal interventions.

## 🎯 Key Features

### 🔬 Core Interpretability
- **Neuron Activation Analysis**: Track and visualize neuron firing patterns
- **Circuit Discovery**: Find computational circuits via interventions
- **Sparse Autoencoders**: Decompose features with SAEs and hierarchical SAEs
- **Feature Attribution**: Advanced attribution methods (Integrated Gradients, DeepLIFT, SHAP)

### 📐 Fractal Geometry Suite
- **Temporal Fractals**: Higuchi FD, DFA, Hurst exponent, spectral slope
- **Graph Fractals**: Box-covering algorithm for network fractality
- **Multifractal Analysis**: Complete multifractal spectrum
- **Fractal Regularizers**: Train with biologically-plausible fractal priors
- **Fractal Stimuli**: Generate fBm, colored noise, cascades
- **Real-time Probes**: Track fractal dimension during training

### 🔍 Circuit Inference
- **Latent RNN Extraction**: Minimal computational circuits (Langdon & Engel 2025)
- **DUNL Sparse Coding**: Mixed selectivity decomposition
- **Feature Visualization**: Activation maximization with biological constraints
- **Dynamics Analysis**: Fixed points, stability, dimensionality

### 🧠 Biophysical Modeling
- **Spiking Networks**: LIF, Izhikevich, Hodgkin-Huxley neurons
- **Surrogate Gradients**: Differentiable spike backpropagation
- **Dale's Law**: E/I neuron separation enforcement
- **Biological Constraints**: Train with biophysical realism

### ⚡ Causal Interventions
- **Activation Patching**: Causal tracing at any layer/position
- **Ablation Studies**: Hierarchical neuron/layer/component ablation
- **Path Analysis**: Information flow and computational paths
- **Causal Graphs**: Build and visualize causal computation graphs

### 🧬 Brain Alignment
- **CCA**: Canonical Correlation Analysis
- **RSA**: Representational Similarity Analysis
- **Procrustes Alignment**: Align representations across models/brains

### 🌊 Dynamics & Geometry
- **Koopman Operators**: Linearize nonlinear dynamics
- **Lyapunov Exponents**: Chaos and stability analysis
- **Manifold Geometry**: Curvature, dimensionality, topology
- **Persistent Homology**: Topological data analysis

### 📊 Information Theory
- **Information Flow**: Mutual information estimation (MINE)
- **Energy Landscapes**: Basin detection and transitions
- **Entropy Production**: Measure dissipation during computation

### 📈 Meta-Dynamics
- **Training Trajectories**: Track feature emergence and drift
- **Phase Transitions**: Detect critical points in training
- **Representational Drift**: Monitor stability over time

## 🚀 Installation

```bash
# Basic installation
pip install neuros-mechint

# With visualization tools
pip install neuros-mechint[viz]

# With notebook support
pip install neuros-mechint[notebooks]

# Full installation (all extras)
pip install neuros-mechint[all]

# Development installation
git clone https://github.com/neuros-ai/neuros-mechint
cd neuros-mechint
pip install -e ".[dev]"
```

## 📚 Quick Start

### Fractal Analysis

```python
from neuros_mechint.fractals import HiguchiFractalDimension, SpectralPrior
import torch

# Compute fractal dimension
fractal = HiguchiFractalDimension(k_max=10)
signal = torch.randn(32, 1000)  # Batch of signals
fd = fractal.compute(signal)
print(f"Fractal Dimension: {fd.mean():.3f} ± {fd.std():.3f}")

# Add fractal regularization to training
regularizer = SpectralPrior(target_beta=1.0, weight=0.01)
activations = model.get_activations(inputs)
fractal_loss = regularizer(activations)
total_loss = task_loss + fractal_loss
```

### Circuit Discovery

```python
from neuros_mechint.circuits import LatentCircuitModel, CircuitFitter

# Extract minimal circuit from neural responses
circuit = LatentCircuitModel(n_latent=10, n_observed=100, enforce_dales=True)
fitter = CircuitFitter(circuit, learning_rate=1e-3)

# Fit circuit to data
results = fitter.fit(neural_responses, stimuli, n_epochs=1000)

# Analyze dynamics
from neuros_mechint.circuits import RecurrentDynamicsAnalyzer
analyzer = RecurrentDynamicsAnalyzer()
dynamics = analyzer.analyze(results)
print(f"Fixed points: {len(dynamics['fixed_points'])}")
```

### Causal Interventions

```python
from neuros_mechint.interventions import ActivationPatcher, AblationStudy

# Activation patching
patcher = ActivationPatcher(model)
result = patcher.patch(
    clean_input, corrupted_input,
    patches=[PatchSpec(layer_name='layer_6', component='residual')],
    metric_fn=accuracy_fn
)
print(f"Recovery: {result['recovery_score']:.3f}")

# Hierarchical ablation study
study = AblationStudy(model)
results = study.hierarchical_ablation(
    input_data, layer_names=['layer_4', 'layer_6', 'layer_8'],
    metric_fn=accuracy_fn, ablate_components=True
)
print(study.summarize(results))
```

### Spiking Networks

```python
from neuros_mechint.biophysical import LeakyIntegrateFireNeuron, DalesLinear

# LIF neuron layer
lif = LeakyIntegrateFireNeuron(n_neurons=256, tau=10.0, threshold=1.0)
spikes, voltages = lif(input_current, dt=1.0)

# Linear layer with Dale's law
ei_layer = DalesLinear(in_features=256, out_features=128, ei_ratio=0.8)
output = ei_layer(input)  # Automatically enforces E/I separation
```

### Sparse Autoencoders

```python
from neuros_mechint.sae import SparseAutoencoder, MultiLayerSAETrainer

# Single SAE
sae = SparseAutoencoder(n_inputs=768, n_features=4096, sparsity=0.05)

# Train SAEs on multiple layers
trainer = MultiLayerSAETrainer(
    model=model,
    layer_names=['layer_6', 'layer_8', 'layer_10'],
    n_features=4096,
)
sae_results = trainer.train(dataloader, n_epochs=10)
```

## 🎓 Tutorials

Comprehensive tutorials are available in the `tutorials/` directory:

1. **Fractal Analysis**: Measure and regularize fractal properties
2. **Circuit Discovery**: Extract interpretable circuits
3. **Causal Interventions**: Patch and ablate for causal understanding
4. **Biophysical Constraints**: Train with Dale's law and spiking neurons
5. **End-to-End**: Combine all techniques

See [tutorials/README.md](tutorials/README.md) for details.

## 📖 Documentation

Full documentation is available at: https://neuros-mechint.readthedocs.io

- **API Reference**: Complete API documentation
- **User Guide**: In-depth explanations of concepts
- **Examples**: Real-world usage examples
- **Theory**: Scientific foundations

## 🧪 Examples

Check out the `examples/` directory for:
- Analyzing transformer attention patterns
- Discovering circuits in language models
- Training with fractal regularization
- Biophysical constraints in CNNs
- Multi-model brain alignment

## 🔬 Scientific Foundations

neuros-mechint implements methods from cutting-edge research:

- **Langdon & Engel (2025)**: Latent circuit extraction
- **Higuchi (1988)**: Fractal dimension estimation
- **Peng et al. (1994)**: Detrended fluctuation analysis
- **Izhikevich (2003)**: Simple spiking neuron model
- **Elhage et al. (2021)**: Transformer circuits
- **Nanda et al. (2023)**: Sparse autoencoders
- And many more...

## 🤝 Use Cases

### Research
- Understand neural coding in foundation models
- Test neuroscience hypotheses with biophysical constraints
- Discover computational motifs in deep networks
- Align model representations with brain activity

### Production
- Interpretable AI for safety-critical applications
- Real-time monitoring of model behavior
- Debugging and validation of neural networks
- Feature engineering guided by mechanistic insights

### Education
- Teaching deep learning interpretability
- Demonstrating fractal properties in neural systems
- Hands-on circuit discovery exercises
- Biophysical modeling workshops

## 🔌 Integration

neuros-mechint is designed to work with:
- **Any PyTorch model**: Just pass your `nn.Module`
- **Hugging Face Transformers**: Seamless integration
- **neuros-foundation**: Built-in support for neurOS models
- **Custom architectures**: Flexible hooks and callbacks

## 🌟 Why neuros-mechint?

1. **Comprehensive**: Most complete interpretability toolkit available
2. **Cutting-edge**: Implements latest research (2024-2025)
3. **Production-ready**: 100% type hints, tested, documented
4. **Flexible**: Works with any PyTorch model
5. **Novel**: Unique fractal and biophysical tools
6. **Open source**: MIT license, community-driven

## 🛣️ Roadmap

- [ ] Additional neuron models (AdEx, QuadraticIF)
- [ ] Synaptic plasticity (STDP, STP)
- [ ] Multi-scale analysis (LFP generation, Virtual Brain)
- [ ] Distributed training support
- [ ] JAX backend
- [ ] Browser-based visualization tools

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

- **Website**: https://neuros.ai
- **Email**: team@neuros.ai
- **Twitter**: @neuros_ai
- **Discord**: https://discord.gg/neuros

## 🙏 Acknowledgments

Built with support from the mechanistic interpretability community and neuroscience researchers worldwide.

## 📚 Citation

If you use neuros-mechint in your research, please cite:

```bibtex
@software{neuros_mechint2025,
  title = {neuros-mechint: Mechanistic Interpretability Toolbox for Neural Networks},
  author = {neurOS Team},
  year = {2025},
  url = {https://github.com/neuros-ai/neuros-mechint},
  version = {0.1.0}
}
```

---

**Created with Claude Code**
https://claude.com/claude-code
