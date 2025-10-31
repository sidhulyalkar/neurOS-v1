# Mechanistic Interpretability Learning Path

Welcome to the **neuros-mechint** educational notebook series! This comprehensive collection of Jupyter notebooks will guide you through the most advanced mechanistic interpretability techniques for understanding neural networks, with a special focus on neuroscience applications.

## About This Library

**neuros-mechint** is the world's most comprehensive mechanistic interpretability suite for neural networks. It combines cutting-edge techniques from AI interpretability research with neuroscience-inspired analysis methods to help you understand how neural networks—both artificial and biological—process information.

## Why Mechanistic Interpretability Matters

Understanding how neural networks work internally is crucial for:
- **Neuroscience**: Comparing artificial and biological neural networks to understand brain computation
- **AI Safety**: Ensuring neural networks are aligned and behave as expected
- **Model Debugging**: Identifying and fixing issues in network behavior
- **Scientific Discovery**: Uncovering computational principles that govern intelligent behavior
- **Trust & Transparency**: Building interpretable AI systems for high-stakes applications

## Learning Path Overview

The notebooks are designed to be followed sequentially, building from foundational concepts to advanced techniques:

### 🌟 Foundational (Start Here!)

#### [01_introduction_and_quickstart.ipynb](01_introduction_and_quickstart.ipynb)
**What you'll learn**: Library overview, basic workflow, first interpretability analysis
- Installation and setup
- Core concepts and terminology
- Your first circuit discovery experiment
- Overview of all major techniques
- **Prerequisites**: Basic Python and PyTorch knowledge
- **Time**: 30-45 minutes

#### [02_sparse_autoencoders.ipynb](02_sparse_autoencoders.ipynb)
**What you'll learn**: Decomposing polysemantic neurons into interpretable features
- The problem of polysemanticity
- How Sparse Autoencoders (SAEs) work
- Training SAEs on neural activations
- Hierarchical concept extraction
- Causal feature importance
- **Prerequisites**: Notebook 01
- **Time**: 60-90 minutes

#### [03_causal_interventions.ipynb](03_causal_interventions.ipynb)
**What you'll learn**: Discovering computational circuits through causal interventions
- Activation patching methodology
- Ablation studies for component importance
- Path tracing and information flow
- Finding circuits for specific behaviors
- Building causal graphs
- **Prerequisites**: Notebooks 01-02
- **Time**: 60-90 minutes

### 🧬 Biological Realism

#### [04_fractal_analysis.ipynb](04_fractal_analysis.ipynb)
**What you'll learn**: Measuring and enforcing biological-like scale-free dynamics
- Why fractals matter in neuroscience
- Computing fractal dimensions and scaling exponents
- Fractal regularization for training
- Generating fractal stimuli
- Comparing model and brain complexity
- **Prerequisites**: Notebooks 01-03
- **Time**: 75-120 minutes

#### [08_biophysical_modeling.ipynb](08_biophysical_modeling.ipynb)
**What you'll learn**: Incorporating realistic neural dynamics and constraints
- Spiking neural networks (LIF, Izhikevich, Hodgkin-Huxley)
- Dale's law and excitatory/inhibitory separation
- Synaptic plasticity (STDP, short-term plasticity)
- Training biologically-constrained networks
- **Prerequisites**: Notebooks 01-04
- **Time**: 90-120 minutes

### 🧠 Brain Alignment

#### [05_brain_alignment.ipynb](05_brain_alignment.ipynb)
**What you'll learn**: Comparing model representations with brain recordings
- Canonical Correlation Analysis (CCA)
- Representational Similarity Analysis (RSA)
- Partial Least Squares (PLS)
- Statistical evaluation and noise ceilings
- Multi-modal alignment strategies
- **Prerequisites**: Notebooks 01-04
- **Time**: 90-120 minutes

### 🔬 Advanced Analysis

#### [06_dynamical_systems.ipynb](06_dynamical_systems.ipynb)
**What you'll learn**: Analyzing neural trajectories using nonlinear dynamics
- Koopman operator theory for linearization
- Lyapunov exponents and chaos
- Fixed point analysis and attractors
- Intrinsic dimensionality estimation
- Controllability analysis
- **Prerequisites**: Notebooks 01-05
- **Time**: 90-120 minutes

#### [07_circuit_extraction.ipynb](07_circuit_extraction.ipynb)
**What you'll learn**: Extracting minimal interpretable circuits
- Latent RNN circuit models
- DUNL sparse coding for mixed selectivity
- Feature visualization and activation maximization
- Circuit motif detection
- E/I circuit diagrams
- **Prerequisites**: Notebooks 01-06
- **Time**: 75-120 minutes

#### [09_information_theory.ipynb](09_information_theory.ipynb)
**What you'll learn**: Information flow and energy landscape analysis
- Mutual information estimation (MINE)
- Tishby's information plane
- Energy landscape mapping
- Basin detection and stability
- Entropy production and thermodynamics
- **Prerequisites**: Notebooks 01-07
- **Time**: 90-120 minutes

### 🚀 Expert Topics

#### [10_advanced_topics.ipynb](10_advanced_topics.ipynb)
**What you'll learn**: Cutting-edge techniques and integrations
- Meta-dynamics and training trajectory analysis
- Manifold geometry and topology
- Counterfactual interventions
- Feature attribution methods
- Integration with training pipelines
- Automated reporting
- **Prerequisites**: All previous notebooks
- **Time**: 120+ minutes

## How to Use These Notebooks

### Installation

First, install the library with visualization support:

```bash
pip install neuros-mechint[viz]
```

Or for the full experience:

```bash
pip install neuros-mechint[all]
```

### Running the Notebooks

1. Clone or download this repository
2. Navigate to the examples folder
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Start with `01_introduction_and_quickstart.ipynb`

### Learning Recommendations

**For beginners**: Follow notebooks 01-03 sequentially to build a solid foundation.

**For neuroscientists**: After the foundation (01-03), focus on notebooks 04, 05, and 08 for biological realism and brain alignment.

**For AI researchers**: After the foundation, dive into notebooks 06, 07, and 09 for advanced dynamical systems and information theory.

**For practitioners**: Notebooks 02, 03, and 10 will help you integrate these techniques into your workflow.

**For researchers**: Complete the entire series to master all techniques.

### Interactive Learning

Each notebook includes:
- **Conceptual explanations**: Understanding the "why" before the "how"
- **Mathematical foundations**: Key equations with intuitive explanations
- **Hands-on code**: Practical examples you can modify and experiment with
- **Visualizations**: Rich plots to build intuition
- **Real-world applications**: Connections to neuroscience and AI research
- **Exercises**: Optional challenges to deepen understanding
- **References**: Links to seminal papers and resources

### Getting Help

- Check the [main documentation](../README.md) for API references
- Review the [STATUS.md](../STATUS.md) for implementation details
- Join discussions and ask questions in Issues
- Contribute your own examples and improvements!

## Scientific Foundation

This library implements techniques from leading research:

- **Sparse Autoencoders**: Anthropic's "Towards Monosemanticity" (2023)
- **Circuit Discovery**: Elhage et al. "A Mathematical Framework" (2021)
- **Fractal Analysis**: Higuchi (1988), Peng et al. (1994)
- **Brain Alignment**: Kriegeskorte et al. RSA (2008), Hotelling CCA (1936)
- **Circuit Extraction**: Langdon & Engel (2025)
- **Dynamical Systems**: Sussillo & Barak (2013)
- **Information Theory**: Tishby & Zaslavsky (2015)
- **Biophysical Models**: Izhikevich (2003), Hodgkin & Huxley (1952)

## Contributing

We welcome contributions to this educational resource! If you:
- Find errors or unclear explanations
- Have ideas for new notebooks or examples
- Want to add exercises or visualizations
- Have applied these techniques to interesting problems

Please open an issue or submit a pull request. Together, we can make mechanistic interpretability accessible to everyone.

## Vision: A Standard for Neuroscience

Our goal is to make **neuros-mechint** the standard toolkit for:
- Analyzing neural network models in neuroscience
- Comparing artificial and biological intelligence
- Standardizing interpretability experiments across labs
- Enabling collaborative research and reproducibility
- Accelerating discoveries about brain computation

By learning and using these tools, you're joining a community working to understand the principles of intelligence—artificial and biological.

## Quick Reference

### Core Techniques Summary

| Technique | Purpose | Key Classes | Notebook |
|-----------|---------|-------------|----------|
| Sparse Autoencoders | Feature discovery | `SparseAutoencoder`, `HierarchicalSAE` | 02 |
| Activation Patching | Causal circuit discovery | `ActivationPatcher`, `PathAnalyzer` | 03 |
| Fractal Analysis | Biological complexity | `HiguchiFractalDimension`, `SpectralPrior` | 04 |
| Brain Alignment | Model-brain comparison | `CCA`, `RSA`, `PLS` | 05 |
| Dynamical Systems | Trajectory analysis | `DynamicsAnalyzer`, Koopman operators | 06 |
| Circuit Extraction | Minimal circuits | `LatentCircuitModel`, `DUNL` | 07 |
| Biophysical Models | Realistic neurons | `SpikingNeuralNetwork`, `DalesLinear` | 08 |
| Information Theory | Information flow | `InformationFlowAnalyzer`, `EnergyLandscape` | 09 |
| Meta-dynamics | Training analysis | `MetaDynamicsTracker` | 10 |

### Common Workflows

**Understand a trained transformer**:
1. Train SAEs (Notebook 02)
2. Run activation patching (Notebook 03)
3. Extract circuits (Notebook 07)

**Compare model to brain data**:
1. Extract activations from model and brain
2. Apply CCA/RSA (Notebook 05)
3. Analyze fractal properties (Notebook 04)

**Train biologically-realistic network**:
1. Apply Dale's law constraints (Notebook 08)
2. Add fractal regularization (Notebook 04)
3. Track dynamics during training (Notebook 06, 10)

**Analyze RNN dynamics**:
1. Find fixed points (Notebook 06)
2. Extract latent circuits (Notebook 07)
3. Map energy landscape (Notebook 09)

## Let's Begin!

Ready to start your journey into mechanistic interpretability? Open [01_introduction_and_quickstart.ipynb](01_introduction_and_quickstart.ipynb) and let's understand how neural networks really work!

---

*"The goal is to understand not just what neural networks compute, but how they compute it—to read the 'source code' written in weights and activations."*

