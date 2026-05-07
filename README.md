# neurOS-v1 – A Modular Operating System for Brain‑Computer Interfaces

**neurOS-v1** is a reimagined version of the original neurOS codebase. It focuses on clarity, extensibility and measurable performance while delivering a production‑ready platform for brain–computer interface (BCI) development and research. This repository contains modular Python packages for real‑time neural data streaming, processing, classification, and advanced neural foundation models.

## Highlights

* **Modular Package Architecture** – Self-contained packages for different aspects of neural computing: core BCI, foundation models, interpretability, astrocyte processing, and more.
* **Neural Foundation Models (NeuroFM-X)** – State-of-the-art backbone architectures including the new **ENGRAM-FMx** for neural sequence modeling.
* **Mechanistic Interpretability** – Comprehensive toolkit for understanding neural network internals with Sparse Autoencoders (SAEs).
* **Astrocyte Integration** – First-of-its-kind glial signal processing for multimodal neural modeling.
* **Streamlined Drivers** – Unified API for reading data from real or simulated BCI hardware.
* **Agent‑based Orchestration** – Asynchronous orchestrator coordinates multiple agents for real‑time BCI pipelines.

---

## Package Overview

### Core Packages

| Package | Description |
|---------|-------------|
| **[neuros](packages/neuros/)** | Main neurOS package - BCI operating system core |
| **[neuros-core](packages/neuros-core/)** | Core functionality, data structures, and utilities |
| **[neuros-drivers](packages/neuros-drivers/)** | Hardware drivers for EEG, video, audio, multi-modal acquisition |
| **[neuros-models](packages/neuros-models/)** | Deep learning models - EEGNet, Transformers, LSTM |
| **[neuros-ui](packages/neuros-ui/)** | User interfaces - Streamlit dashboard, FastAPI server |
| **[neuros-cloud](packages/neuros-cloud/)** | Cloud infrastructure - Kafka, AWS SageMaker, WebDataset |

### Research Packages

| Package | Description |
|---------|-------------|
| **[neuros-neurofm](packages/neuros-neurofm/)** | **Neural Foundation Model (NeuroFM-X)** - Advanced backbone architectures |
| **[neuros-astro](packages/neuros-astro/)** | **Astrocyte Processing** - Glial signal extraction and tokenization |
| **[neuros-mechint](packages/neuros-mechint/)** | **Mechanistic Interpretability** - SAEs, circuit analysis, probing |
| **[neuros-foundation](packages/neuros-foundation/)** | Pre-trained models - POYO, NDT, CEBRA, Neuroformer |
| **[neuros-neuroviz](packages/neuros-neuroviz/)** | GPU-accelerated visualization for neural data |
| **[neuros-sourceweigher](packages/neuros-sourceweigher/)** | Domain adaptation weight estimation |

---

## Featured: NeuroFM-X with ENGRAM-FMx Backbone

**NeuroFM-X** (`packages/neuros-neurofm/`) is our neural foundation model package featuring the new **ENGRAM-FMx** (Energy-guided Neural Generative Recurrent Attractor Model) backbone architecture.

### ENGRAM-FMx Architecture

ENGRAM-FMx combines cutting-edge components for neural sequence modeling:

- **Selective State-Space Models (SSM)** – Efficient O(T) temporal propagation
- **Perceiver-style Latent Workspace** – Compression to fixed-size latent slots
- **Hopfield-style Attractor Memory** – Energy-guided associative retrieval
- **Neural Operators (Spectral/FFT)** – Latent field dynamics evolution
- **Sparse Anchor Attention** – Exact grounding at selected positions
- **Gated Fusion** – Adaptive integration of processing streams

### Quick Start with ENGRAM-FMx

```bash
cd packages/neuros-neurofm
pip install -e .

# Run synthetic training (RTX 3070 Ti compatible)
python -m neuros_neurofm.training.train_engram_synthetic \
    --config configs/engram_fmx/tiny_synthetic.yaml

# Run ablation experiments
python scripts/run_engram_ablations.py --tasks associative_recall
```

### ENGRAM-FMx Usage

```python
from neuros_neurofm.models import ENGRAMFMxConfig, ENGRAMBackbone
import torch

# Create model
config = ENGRAMFMxConfig(
    input_dim=256,
    hidden_dim=256,
    num_layers=4,
    num_latents=64,
    memory_slots=256,
)
backbone = ENGRAMBackbone(config)

# Forward pass
x = torch.randn(2, 512, 256)  # [batch, seq_len, features]
output = backbone(x)

print(output.sequence_output.shape)  # [2, 512, 256]
print(output.latent_output.shape)    # [2, 64, 256]
print(output.diagnostics.keys())     # Memory entropy, gate values, etc.
```

---

## Featured: Astrocyte Integration (neuros-astro)

**neuros-astro** (`packages/neuros-astro/`) extracts astrocyte calcium events and functional network states from optical physiology data, converting them into model-ready tokens for multimodal neural modeling.

### Key Features

- **Calcium Event Detection** – ML-based astrocyte signal extraction
- **Functional Network Analysis** – Graph-based astrocyte connectivity
- **Tokenization** – Convert glial signals to model inputs
- **Allen Data Integration** – Process Allen Brain Observatory NWB files

### Quick Start

```bash
cd packages/neuros-astro
pip install -e .

# Process Allen Brain Observatory data
python examples/06_process_allen_data.py

# Run ablation study (neural vs neural+astro)
python examples/ablation_study/train_ablation.py --condition all
```

---

## Featured: Mechanistic Interpretability (neuros-mechint)

**neuros-mechint** (`packages/neuros-mechint/`) provides comprehensive tools for understanding neural network internals.

### Key Features

- **Sparse Autoencoders (SAEs)** – Learn interpretable features from activations
- **Circuit Analysis** – Extract and visualize computational circuits
- **Probing** – Linear and nonlinear probes for representations
- **Ablation Studies** – Systematic component importance analysis
- **Cross-Modal Analysis** – Compare features across modalities

### Quick Start

```bash
cd packages/neuros-mechint
pip install -e .

# Run SAE analysis on Allen data
cd examples/allen_data_demo
python run_advanced_mechint.py
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/shulyalk/neuros-v1.git
cd neuros-v1

# Install core package
pip install -e .

# Install specific research packages
pip install -e packages/neuros-neurofm
pip install -e packages/neuros-astro
pip install -e packages/neuros-mechint

# Run diagnostics
neuros --help
```

### Running a Pipeline

```bash
# Run a real-time pipeline with simulated driver
neuros run --duration 5

# Run benchmarks
neuros benchmark --duration 10 --report benchmarks/report.json

# Launch dashboard (requires streamlit)
pip install streamlit
neuros dashboard
```

### Constellation Demo

Multi-modal ingestion demo supporting EEG, audio, video, EDA, fNIRS/HD-DOT, respiration, ECG, and phone sensors:

```bash
neuros constellation \
  --duration 10 \
  --output-dir /tmp/constellation_demo \
  --subject-id demo \
  --session-id session1
```

See `docs/runbook_constellation.md` for detailed setup instructions.

---

## Repository Structure

```
neurOS-v1/
├── packages/
│   ├── neuros/                 # Main BCI OS package
│   ├── neuros-core/            # Core functionality
│   ├── neuros-drivers/         # Hardware drivers
│   ├── neuros-models/          # ML models (EEGNet, etc.)
│   ├── neuros-neurofm/         # Neural Foundation Model + ENGRAM-FMx
│   │   ├── src/neuros_neurofm/
│   │   │   ├── backbones/
│   │   │   │   └── engram_fmx/ # ENGRAM-FMx backbone
│   │   │   ├── models/         # Model components
│   │   │   ├── training/       # Training scripts
│   │   │   └── data/           # Datasets
│   │   ├── configs/            # Training configs
│   │   └── tests/              # Unit tests (108 tests)
│   ├── neuros-astro/           # Astrocyte processing
│   │   ├── neuros_astro/       # Source code
│   │   └── examples/           # Usage examples + ablation study
│   ├── neuros-mechint/         # Mechanistic interpretability
│   │   ├── src/neuros_mechint/ # Source code
│   │   └── examples/           # SAE demos, Allen data analysis
│   ├── neuros-foundation/      # Pre-trained models
│   ├── neuros-neuroviz/        # Visualization
│   ├── neuros-cloud/           # Cloud infrastructure
│   ├── neuros-ui/              # User interfaces
│   └── neuros-sourceweigher/   # Domain adaptation
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
└── tests/                      # Integration tests
```

---

## Project Status

**Version:** 2.0.0 (Beta)

| Component | Status |
|-----------|--------|
| Core BCI Pipeline | ✅ Complete |
| ENGRAM-FMx Backbone | ✅ Complete (108 tests passing) |
| Astrocyte Integration | ✅ Complete |
| Mechanistic Interpretability | ✅ Complete |
| Multi-modal Support | ✅ 15+ drivers |
| Model Library | ✅ 10+ models |
| Cloud Infrastructure | ✅ Kafka, SageMaker, Zarr |
| Test Coverage | ⚠️ 40% (target: >90%) |
| Hardware Testing | ⚠️ In progress |

---

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** – Get up and running in 5 minutes
- **[CONTRIBUTING.md](CONTRIBUTING.md)** – Developer guide and contribution guidelines
- **[docs/](docs/)** – Technical documentation and white papers
- **Package READMEs** – Each package has its own README with detailed usage

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. The modular architecture makes it easy to add new:

- Drivers for hardware devices
- Processing algorithms
- Model architectures
- Analysis tools

---

## Support

- **Issues:** Report bugs or request features on [GitHub Issues](https://github.com/shulyalk/neuros-v1/issues)
- **Discussions:** Ask questions and share ideas
- **Documentation:** Check the [docs/](docs/) folder for technical details

---

## License

This project is licensed under the MIT license – see [LICENSE](LICENSE) for details.

---

**neurOS v1** – Building the future of brain-computer interfaces 🧠
