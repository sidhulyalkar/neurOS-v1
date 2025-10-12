# Installation Guide

This guide will help you install NeurOS and its dependencies on your system.

---

## Requirements

- **Python**: 3.9 or higher (3.13 recommended)
- **OS**: Linux, macOS, or Windows
- **RAM**: 8GB minimum (16GB recommended for foundation models)
- **GPU**: Optional (NVIDIA GPU with CUDA for faster inference)

---

## Quick Install

### Basic Installation

For most users, install from PyPI:

```bash
pip install neuros
```

This installs the core package with essential dependencies.

### Full Installation

To install with all optional features:

```bash
pip install neuros[all]
```

This includes:
- Foundation model support (PyTorch)
- Dashboard (Streamlit)
- Hardware drivers (BrainFlow, LSL)
- Advanced datasets (Allen SDK, MOABB)
- Visualization tools
- Distributed computing (Ray, Dask)

---

## Installation Options

### By Feature

Install only the features you need:

```bash
# Foundation models
pip install neuros[foundation]

# Dashboard
pip install neuros[dashboard]

# Hardware support
pip install neuros[hardware]

# Dataset loaders
pip install neuros[datasets]

# Visualization
pip install neuros[viz]

# Distributed computing
pip install neuros[distributed]

# Documentation building
pip install neuros[docs]
```

### Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/yourusername/neuros-v1.git
cd neuros-v1

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev,all]"

# Run tests to verify installation
pytest
```

---

## Dependency Details

### Core Dependencies

Always installed:
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning
- `pandas` - Data manipulation
- `pyyaml` - Configuration files

### Optional Dependencies

#### Foundation Models (`neuros[foundation]`)
- `torch` - Deep learning framework
- `torch_brain` - Neuroscience-specific models
- `temporaldata` - Neural time series handling

#### Dashboard (`neuros[dashboard]`)
- `streamlit` - Interactive dashboard
- `plotly` - Interactive plots
- `dash` - Web apps

#### Hardware (`neuros[hardware]`)
- `brainflow` - EEG/BCI hardware interface
- `pylsl` - Lab Streaming Layer
- `psychopy` - Stimulus presentation

#### Datasets (`neuros[datasets]`)
- `allensdk` - Allen Institute data
- `moabb` - BCI datasets
- `pynwb` - NWB format support
- `mne` - MEG/EEG analysis

#### Visualization (`neuros[viz]`)
- `matplotlib` - Plotting
- `seaborn` - Statistical plots
- `umap-learn` - Dimensionality reduction
- `plotly` - Interactive plots

#### Distributed (`neuros[distributed]`)
- `ray` - Distributed computing
- `dask` - Parallel processing
- `redis` - Caching
- `kafka-python` - Stream processing

---

## Platform-Specific Notes

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.13 hdf5 liblsl

# Install NeurOS
pip3 install neuros[all]
```

### Linux (Ubuntu/Debian)

```bash
# Update package lists
sudo apt update

# Install Python and dependencies
sudo apt install python3.13 python3-pip python3-venv \
                 libhdf5-dev liblsl-dev

# Install NeurOS
pip3 install neuros[all]
```

### Windows

```powershell
# Install Python 3.13 from python.org
# Then in PowerShell:

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install NeurOS
pip install neuros[all]
```

---

## GPU Support

For GPU-accelerated inference with foundation models:

### NVIDIA GPU (CUDA)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install NeurOS
pip install neuros[foundation]
```

### Apple Silicon (M1/M2/M3)

PyTorch automatically uses Metal Performance Shaders (MPS):

```bash
pip install neuros[foundation]
# MPS will be used automatically if available
```

### AMD GPU (ROCm)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install neuros[foundation]
```

---

## Verification

After installation, verify everything works:

```python
import neuros

# Check version
print(neuros.__version__)

# Check available modules
print("Foundation models available:", neuros.foundation_models.POYO_AVAILABLE)
print("NWB support available:", neuros.io.NWB_AVAILABLE)

# Run a quick test
from neuros.pipeline import Pipeline
pipeline = Pipeline(driver='simulated_eeg', model='svm')
print("✅ NeurOS installed successfully!")
```

Or run the test suite:

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_models.py -v

# Check coverage
pytest --cov=neuros --cov-report=html
```

---

## Common Issues

### Issue: PyTorch installation fails

**Solution**: Install PyTorch separately first:
```bash
pip install torch torchvision torchaudio
pip install neuros[foundation]
```

### Issue: LSL library not found

**Solution**: Install liblsl system package:
```bash
# macOS
brew install labstreaminglayer/tap/lsl

# Linux
sudo apt install liblsl-dev

# Or skip hardware features
pip install neuros  # without [hardware]
```

### Issue: Allen SDK installation fails

**Solution**: Use conda for AllenSDK:
```bash
conda install -c conda-forge allensdk
pip install neuros
```

### Issue: Out of memory with foundation models

**Solution**:
- Use smaller batch sizes
- Enable gradient checkpointing
- Use CPU inference for smaller datasets
- Close other applications

---

## Docker Installation

For reproducible environments:

```bash
# Pull pre-built image
docker pull neuros/neuros:latest

# Or build from source
docker build -t neuros:local .

# Run container
docker run -it --gpus all neuros/neuros:latest
```

### Docker Compose

For multi-service setup (with Kafka, Redis, etc.):

```bash
# Clone repository
git clone https://github.com/yourusername/neuros-v1.git
cd neuros-v1

# Start all services
docker-compose up -d

# Run demo
docker-compose exec neuros python scripts/run_local_demo.py
```

---

## Conda Installation

Using Conda/Mamba:

```bash
# Create environment
conda create -n neuros python=3.13

# Activate environment
conda activate neuros

# Install dependencies
conda install numpy scipy scikit-learn pandas pytorch torchvision torchaudio -c pytorch

# Install NeurOS
pip install neuros[all]
```

---

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade neuros

# Or with all extras
pip install --upgrade neuros[all]
```

---

## Uninstallation

To remove NeurOS:

```bash
pip uninstall neuros

# Remove configuration files (optional)
rm -rf ~/.neuros
```

---

## Next Steps

- **[Quickstart Guide](quickstart.md)** - Run your first pipeline in 5 minutes
- **[First Pipeline](first-pipeline.md)** - Build a complete BCI system
- **[Foundation Models](../user-guide/foundation-models.md)** - Use state-of-the-art models
- **[API Reference](../api/pipeline.md)** - Detailed API documentation

---

## Getting Help

- **Documentation Issues**: [GitHub Issues](https://github.com/yourusername/neuros-v1/issues)
- **Installation Problems**: [Discussions](https://github.com/yourusername/neuros-v1/discussions)
- **Feature Requests**: [Roadmap](../development/roadmap.md)
