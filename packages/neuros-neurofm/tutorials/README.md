# NeuroFMX Tutorials

Comprehensive tutorials for using NeuroFMX's revolutionary capabilities.

## Available Tutorials

### 1. SourceWeigher: Domain Adaptation for Neural Data
**File**: [sourceweigher_tutorial.ipynb](./sourceweigher_tutorial.ipynb)

Learn how to use SourceWeigher for domain adaptation when training on multi-subject neural data.

**Topics Covered**:
- Loading multi-subject neural recordings
- Computing domain statistics
- Estimating mixture weights
- Three-phase training (pretrain → weighted → fine-tune)
- Comparing with baseline (no adaptation)
- Visualizing domain adaptation benefits

**Dataset**: Allen Neuropixels
**Duration**: ~30 minutes
**Level**: Intermediate

---

## Coming Soon

### 2. Fractal Analysis of Neural Representations
- Measuring fractal dimension of latent representations
- Using fractal regularizers during training
- Analyzing multifractal spectra
- Real-time fractal probes

### 3. Circuit Discovery with Latent RNNs
- Extracting minimal computational circuits
- Analyzing recurrent dynamics
- Finding fixed points and stability
- Visualizing circuit structure

### 4. Biophysically-Constrained Learning
- Training with Dale's law (E/I separation)
- Differentiable spiking networks
- Surrogate gradient methods
- Comparing LIF, Izhikevich, and HH neurons

### 5. Causal Interventions and Mechanistic Interpretability
- Activation patching for causal tracing
- Systematic ablation studies
- Path analysis and information flow
- Building causal graphs

### 6. End-to-End: Training a Revolutionary NeuroFMX Model
- Combining all techniques
- Fractal priors + circuit extraction + Dale's law
- Multi-subject domain adaptation
- Comprehensive interpretability analysis

---

## Running the Tutorials

### Prerequisites

```bash
# Install NeuroFMX
cd packages/neuros-neurofm
pip install -e .

# Install SourceWeigher
cd ../neuros-sourceweigher
pip install -e .

# Install Jupyter
pip install jupyter notebook ipykernel matplotlib
```

### Launch Jupyter

```bash
cd packages/neuros-neurofm/tutorials
jupyter notebook
```

### Download Data

The tutorials use the Allen Neuropixels dataset. Download it using:

```bash
python ../scripts/download_allen_data.py
```

---

## Tutorial Structure

Each tutorial follows this structure:

1. **Introduction**: Problem statement and key concepts
2. **Setup**: Imports and configuration
3. **Data Loading**: Load and prepare datasets
4. **Implementation**: Step-by-step implementation
5. **Visualization**: Plot results and insights
6. **Comparison**: Compare with baselines
7. **Summary**: Key takeaways and next steps

---

## Contributing

Have ideas for new tutorials? Please:
1. Open an issue describing the tutorial
2. Follow the existing tutorial structure
3. Include clear explanations and visualizations
4. Test on the provided datasets

---

## Support

For questions or issues:
- Check the [main documentation](../docs/)
- Review the [integration plans](../)
- Open an issue on GitHub

---

**Created with Claude Code**
https://claude.com/claude-code
