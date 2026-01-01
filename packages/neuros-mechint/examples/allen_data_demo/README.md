# Allen Visual Coding SAE Demo

**Sparse Autoencoders for Neural Circuit Discovery in Visual Cortex**

This demo showcases the application of mechanistic interpretability techniques (Sparse Autoencoders) to real neural electrophysiology data from the Allen Brain Observatory. This work represents the first comprehensive validation of SAE methods on multi-session neural recordings.

---

## 🎯 Overview

This project demonstrates that Sparse Autoencoders (SAEs) can discover interpretable neural features that **exceed the selectivity of individual biological neurons**. Applied to mouse primary visual cortex (V1) recordings, we achieved:

- **71.1% orientation-selective SAE features** vs. 37.0% in raw neurons (+92% improvement)
- **Sparse, distributed circuits** with 65% neuron reuse across features
- **Robust validation** across 10 independent recording sessions
- **Complete circuit analysis** via attribution and ablation studies

---

## 📊 Key Results

| Metric | Raw Neurons | SAE Features | Improvement |
|--------|-------------|--------------|-------------|
| % Selective (>0.3 correlation) | 37.0% | 71.1% | +92% |
| Max Correlation | 0.821 | 0.780 | Maintained |
| Mean Correlation | 0.274 | 0.372 | +36% |
| Reconstruction Loss | N/A | 0.438 | Excellent |
| Sparsity | N/A | 48.1% | Optimal |

**Multi-Session Validation**: All 10 top sessions showed >60% selective features, confirming biological robustness.

---

## 📁 Repository Structure

```
allen_data_demo/
├── README.md                           # This file
├── RUN_EXPERIMENTS.sh                  # Quick run script for all experiments
├── scripts/                            # Analysis and training scripts
│   ├── multi_session_validation.py    # Validate across 32 sessions
│   ├── sae_training_top_sessions.py   # Train SAEs on recommended sessions
│   ├── analyze_sae_features.py        # Compare SAE vs raw neuron selectivity
│   ├── visualize_sae_features.py      # Generate publication-quality figures
│   ├── sae_hyperparameter_search.py   # Optimize SAE architecture
│   ├── analyze_best_sessions.py       # Rank sessions by quality
│   ├── download_validation_data.py    # Download Allen data
│   └── sae_validation_example.py      # Simple validation example
├── experiments/                        # Advanced mechanistic interpretability
│   ├── circuit_extraction/
│   │   ├── attribution_analysis.py    # Feature→neuron attribution (Integrated Gradients)
│   │   └── ablation_study.py          # Causal validation via neuron silencing
│   ├── cross_modal/
│   │   └── visual_behavior_decoding.py # Decode behavior from SAE features
│   └── dynamics/
│       └── feature_dynamics.py        # Temporal response analysis
├── results/                            # Generated analysis outputs
│   ├── circuits/                       # Circuit diagrams and attribution results
│   └── dynamics/                       # Temporal dynamics visualizations
├── sae_models/                         # Trained SAE models
│   └── training_results.json          # Training metrics (*.pt files gitignored)
├── config/
│   └── recommended_sessions.json      # Top 10 session IDs
└── docs/                               # Comprehensive documentation
    ├── COMPREHENSIVE_SCIENTIFIC_MANUSCRIPT.md  # Full manuscript (14,500 words)
    ├── SAE_TRAINING_SUCCESS.md        # Training results summary
    ├── COMPREHENSIVE_SAE_ANALYSIS.md  # Detailed analysis
    ├── CIRCUIT_EXTRACTION_GUIDE.md    # Circuit analysis guide
    ├── DIRECTION_VS_ORIENTATION.md    # Circular statistics methodology
    ├── TOP_SESSIONS_SUMMARY.md        # Session quality rankings
    ├── SAE_WORKFLOW_GUIDE.md          # Complete workflow
    ├── SAE_ANALYSIS_WORKFLOW.md       # Analysis pipeline
    ├── MECHINT_VALIDATION_PROGRESS.md # Validation report
    ├── NEXT_STEPS.md                  # Advanced experiments
    ├── PLOT_INTERPRETATION_GUIDE.md   # Figure interpretation
    ├── DATA_DOWNLOAD_GUIDE.md         # Data acquisition
    ├── QUICK_START.md                 # Quick start guide
    └── VALIDATION_QUICKSTART.md       # Fast validation

```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch allensdk scikit-learn scipy matplotlib seaborn captum

# OR install neuros-mechint package
cd ../../../  # Navigate to package root
pip install -e .
```

### 1. Download Data (One-Time Setup)

```bash
python scripts/download_validation_data.py \
    --output-dir allen_validation_cache \
    --n-sessions 10 \
    --brain-areas VISp
```

**Note**: This downloads ~5-10GB of data. The `allen_validation_cache/` directory is gitignored.

### 2. Train Your First SAE

```bash
python scripts/sae_training_top_sessions.py \
    --session-config config/recommended_sessions.json \
    --allen-cache allen_validation_cache \
    --sae-dim 128 \
    --sparsity 0.01 \
    --epochs 50
```

**Output**: Trained SAE model and orientation selectivity metrics.

### 3. Analyze SAE Features

```bash
python scripts/analyze_sae_features.py \
    --sae-results sae_models/training_results.json \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir results/sae_analysis
```

**Generates**:
- Selectivity comparison plots (SAE vs raw neurons)
- Orientation coverage analysis
- Sparsity metrics

### 4. Visualize Features

```bash
python scripts/visualize_sae_features.py \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --sae-model sae_models/sae_session_754829445.pt \
    --output-dir results/sae_visualizations
```

**Generates 5 publication-quality figures**:
1. Tuning curves (top features)
2. Activation heatmaps
3. Weight structure
4. Orientation coverage map
5. Feature clustering (PCA)

---

## 🔬 Advanced Experiments

### Circuit Extraction

Identify which neurons contribute to each SAE feature using Integrated Gradients:

```bash
python experiments/circuit_extraction/attribution_analysis.py \
    --sae-model sae_models/sae_session_754829445.pt \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir results/circuits \
    --top-features 20
```

**Output**: Circuit diagrams, attribution scores, neuron reuse analysis.

### Ablation Studies

Validate circuits via computational neuron silencing:

```bash
python experiments/circuit_extraction/ablation_study.py \
    --sae-model sae_models/sae_session_754829445.pt \
    --attribution-results results/circuits/attribution_results_session_754829445.json \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir results/circuits/ablation
```

**Output**: Disruption scores, minimal circuits, causal validation.

### Temporal Dynamics

Analyze how features evolve during stimulus presentation:

```bash
python experiments/dynamics/feature_dynamics.py \
    --sae-model sae_models/sae_session_754829445.pt \
    --session-id 754829445 \
    --allen-cache allen_validation_cache \
    --output-dir results/dynamics \
    --top-features 10
```

**Output**: Response latencies, decay constants, temporal profiles.

### Run All Experiments

```bash
bash RUN_EXPERIMENTS.sh
```

**Total Runtime**: ~60-90 minutes for complete pipeline.

---

## 📖 Documentation

### For Quick Start

1. **[QUICK_START.md](docs/QUICK_START.md)** - Get started in 15 minutes
2. **[SAE_WORKFLOW_GUIDE.md](docs/SAE_WORKFLOW_GUIDE.md)** - Complete training workflow

### For Deep Dive

1. **[COMPREHENSIVE_SCIENTIFIC_MANUSCRIPT.md](docs/COMPREHENSIVE_SCIENTIFIC_MANUSCRIPT.md)** - Full manuscript (14,500 words)
   - Complete methods, results, discussion
   - Publication-ready format
   - Extensive references

2. **[COMPREHENSIVE_SAE_ANALYSIS.md](docs/COMPREHENSIVE_SAE_ANALYSIS.md)** - Detailed analysis results
   - Hyperparameter optimization
   - Multi-session validation
   - Circuit motif analysis

3. **[CIRCUIT_EXTRACTION_GUIDE.md](docs/CIRCUIT_EXTRACTION_GUIDE.md)** - Circuit analysis guide
   - Attribution methods
   - Ablation procedures
   - Interpretation guidelines

### Methodological Contributions

1. **[DIRECTION_VS_ORIENTATION.md](docs/DIRECTION_VS_ORIENTATION.md)** - Circular statistics for orientation analysis
   - **Key Result**: 2.5-3× improvement in measured selectivity
   - Proper handling of 180° periodicity
   - Critical for accurate tuning analysis

---

## 🧬 Key Findings

### 1. SAE Features Exceed Biological Neuron Selectivity

**Discovery**: SAEs decomposed mixed-selectivity neural responses into highly tuned orientation detectors, with 71% of features showing strong selectivity compared to 37% of raw neurons.

**Interpretation**: Population-level representations are more orientation-tuned than individual neurons, supporting distributed coding models.

### 2. Sparse, Distributed Circuits with Neuron Reuse

**Discovery**: 71/92 neurons (77%) participated in feature computations, with 65% contributing to multiple features (mean: 3.2 features/neuron).

**Interpretation**: Neurons function as **reusable computational building blocks** rather than dedicated "grandmother cells."

### 3. Robust Redundancy via Ablation Validation

**Discovery**: Ablating individual neurons caused <20% feature disruption (max: 18%), with most disruptions <5%.

**Interpretation**: Features are **robust to single neuron loss** through distributed redundancy.

### 4. Sustained Temporal Dynamics

**Discovery**: All analyzed features exhibited sustained responses (decay τ: 1.7-8.9s).

**Interpretation**: Features encode **stimulus identity** rather than transient events, resembling complex cell responses.

### 5. Multi-Session Generalization

**Discovery**: Results replicated across 10 independent sessions (all >60% selective features).

**Interpretation**: Method is **robust to biological variability** and generalizes across animals/recordings.

---

## 🔧 Methodological Contributions

### Circular Statistics for Orientation Analysis

**Problem**: Standard linear correlation underestimates orientation selectivity by 2.5-3× when applied to 8-direction stimulus data.

**Solution**: Proper circular statistics using sin(2θ) and cos(2θ) transformations (180° period).

**Impact**: This methodological correction is **critical for accurate tuning analysis** and should be adopted by the field.

See [DIRECTION_VS_ORIENTATION.md](docs/DIRECTION_VS_ORIENTATION.md) for full derivation.

### SAE Hyperparameter Guidance

**Optimal Configuration** (validated on neural data):
- Hidden dim: 128 (1.39× overcomplete)
- Sparsity (λ): 0.01
- Activation: LeakyReLU
- Learning rate: 0.001
- Epochs: 50-100

See [COMPREHENSIVE_SAE_ANALYSIS.md](docs/COMPREHENSIVE_SAE_ANALYSIS.md) for hyperparameter search results.

### Attribution + Ablation Framework

**Pipeline**:
1. **Attribution** (Integrated Gradients) → Identify candidate circuits
2. **Ablation** (neuron silencing) → Validate causal necessity
3. **Minimal circuits** → Determine sufficiency

**Value**: Provides stronger evidence than attribution alone (which may reflect correlation).

See [CIRCUIT_EXTRACTION_GUIDE.md](docs/CIRCUIT_EXTRACTION_GUIDE.md) for implementation details.

---

## 🎓 Citation

If you use this work, please cite:

```bibtex
@article{neurOS_SAE_2026,
  title={Sparse Autoencoders Reveal Interpretable Neural Circuits in Visual Cortex: A Mechanistic Interpretability Approach to Neuroscience},
  author={[TO BE FILLED]},
  journal={[TO BE DETERMINED]},
  year={2026},
  note={Code available at https://github.com/[YOUR_REPO]/neurOS-v1}
}
```

---

## 📚 References

**Key Papers**:

- **Sparse Autoencoders**: Anthropic (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning."
- **Allen Brain Observatory**: de Vries et al. (2020). "A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex." *Nature Neuroscience*.
- **Orientation Selectivity**: Hubel & Wiesel (1962). "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex." *J. Physiology*.
- **Circular Statistics**: Fisher (1993). *Statistical Analysis of Circular Data*. Cambridge University Press.
- **Integrated Gradients**: Sundararajan et al. (2017). "Axiomatic attribution for deep networks." *ICML*.

**Full Bibliography**: See [COMPREHENSIVE_SCIENTIFIC_MANUSCRIPT.md](docs/COMPREHENSIVE_SCIENTIFIC_MANUSCRIPT.md#6-references)

---

## 🤝 Contributing

This demo is part of the neurOS mechanistic interpretability toolkit. Contributions welcome!

See main repository [CONTRIBUTING.md](../../../../CONTRIBUTING.md) for guidelines.

---

## 📧 Contact

For questions about this demo:
- Open an issue on GitHub
- See [COMPREHENSIVE_SCIENTIFIC_MANUSCRIPT.md](docs/COMPREHENSIVE_SCIENTIFIC_MANUSCRIPT.md) for detailed methods
- Check [FAQ section in SAE_WORKFLOW_GUIDE.md](docs/SAE_WORKFLOW_GUIDE.md)

---

## 🔓 License

This project is licensed under the [LICENSE](../../../../LICENSE) of the main neurOS repository.

Allen Brain Observatory data are available under the Allen Institute Terms of Use.

---

## 🙏 Acknowledgments

- **Allen Institute for Brain Science** for open-access Visual Coding dataset
- **Anthropic** for pioneering SAE methods in mechanistic interpretability
- **neurOS contributors** for the foundational toolkit

---

## 📈 Project Status

- ✅ SAE Training Complete
- ✅ Multi-Session Validation (10/10 sessions)
- ✅ Circuit Extraction (20 features analyzed)
- ✅ Ablation Studies (10 features validated)
- ✅ Temporal Dynamics (10 features characterized)
- ✅ Manuscript Complete (14,500 words)
- 🔄 Cross-Modal Behavioral Decoding (in progress)
- 📝 Preparing for publication submission

**Last Updated**: January 1, 2026

---

**Ready to discover interpretable neural circuits? Start with [QUICK_START.md](docs/QUICK_START.md)!** 🧠✨
