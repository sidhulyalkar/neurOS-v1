# NeurOS Modularization Plan

**Version**: 1.0
**Date**: 2025-10-16
**Status**: Planning Phase
**Based On**: ChatGPT Evaluation Round 2 (chatGPT-eval2.pdf)

---

## Executive Summary

This document outlines the strategic plan to transition NeurOS from a monolithic package structure to a modular, multi-package architecture. This modularization is the **highest priority architectural recommendation** from the comprehensive external evaluation.

### Key Motivation (from Evaluation)

> "It likely makes sense to modularize it into a suite of packages as the project matures. The current monolithic design made sense to quickly develop a cohesive platform, but as features grow, **modular design will improve manageability and user experience**."

### Benefits of Modularization

1. **Reduced Dependencies**: Users can install only what they need
2. **Faster Installation**: Core functionality without heavy ML/hardware dependencies
3. **Easier Maintenance**: Clear separation of concerns
4. **Better Testing**: Isolated component testing
5. **Flexible Deployment**: Deploy only necessary components to edge devices
6. **Clearer Documentation**: Package-specific docs for different user types

---

## Current Monolithic Structure

### Package Analysis

**Current Installation**: `pip install neuros` installs everything:
- Core pipeline and orchestration
- All hardware drivers (BrainFlow, LSL, camera, microphone)
- All deep learning models (PyTorch, TensorFlow, JAX)
- All foundation models (POYO, NDT, CEBRA, Neuroformer)
- UI components (Streamlit, FastAPI)
- Cloud infrastructure (Kafka, boto3 for SageMaker)
- Export formats (WebDataset, Zarr)

**Dependency Count**: ~50+ dependencies installed unconditionally

**Pain Points**:
- Heavy installation for simple use cases
- Conda environment conflicts (especially with hardware drivers)
- Difficult to deploy minimal versions to embedded systems
- Testing complexity due to tangled dependencies

---

## Proposed Modular Architecture

### Package Breakdown

```
neuros-ecosystem/
├── neuros-core/          # Minimal core (always required)
├── neuros-drivers/       # Hardware integration
├── neuros-models/        # Deep learning models
├── neuros-foundation/    # Foundation models (POYO, NDT, etc.)
├── neuros-ui/           # User interfaces
├── neuros-cloud/        # Cloud infrastructure
└── neuros/              # Meta-package (installs common subset)
```

---

## 1. neuros-core

**Purpose**: Minimal core functionality with zero heavy dependencies

### Components
- `neuros.core.pipeline`: Pipeline base classes
- `neuros.core.orchestrator`: MultiModalOrchestrator
- `neuros.core.agents`: BaseAgent, DeviceAgent, ProcessingAgent, ModelAgent
- `neuros.core.config`: Configuration management
- `neuros.core.utils`: Common utilities
- `neuros.io.base`: Abstract I/O interfaces
- `neuros.processing.base`: BaseProcessor, signal processing utilities

### Dependencies (Minimal)
```python
numpy>=1.24.0
scipy>=1.11.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

### Installation
```bash
pip install neuros-core
```

### Use Cases
- Building custom pipelines without heavy dependencies
- Embedded/edge deployment
- Teaching fundamental concepts
- CI/CD testing of core logic

---

## 2. neuros-drivers

**Purpose**: Hardware integration and device drivers

### Components
- `neuros.drivers.brainflow_driver`: BrainAmp, OpenBCI, etc.
- `neuros.drivers.lsl_driver`: Lab Streaming Layer
- `neuros.drivers.camera_driver`: Video capture
- `neuros.drivers.microphone_driver`: Audio capture
- `neuros.drivers.mock_driver`: Testing/simulation
- `neuros.io.bids_loader`: BIDS dataset loader
- `neuros.io.nwb_loader`: NWB file I/O

### Dependencies
```python
neuros-core>=1.0.0
brainflow>=5.10.0
pylsl>=1.16.0
opencv-python>=4.8.0
pyaudio>=0.2.13
pynwb>=2.5.0
hdmf>=3.10.0
```

### Installation
```bash
pip install neuros-drivers
# Or specific subsets:
pip install neuros-drivers[eeg]        # BrainFlow + LSL only
pip install neuros-drivers[video]      # Camera only
pip install neuros-drivers[nwb]        # NWB I/O only
```

### Use Cases
- Real-time data acquisition
- BCI/neurofeedback applications
- Multi-modal recording sessions

---

## 3. neuros-models

**Purpose**: Deep learning models and training utilities

### Components
- `neuros.models.eegnet`: EEGNet architecture
- `neuros.models.transformer`: Transformer models
- `neuros.models.lstm`: LSTM models
- `neuros.models.simple_models`: Linear, SVM, RF classifiers
- `neuros.training`: Training loops and utilities
- `neuros.evaluation`: Metrics and evaluation

### Dependencies
```python
neuros-core>=1.0.0
torch>=2.0.0
scikit-learn>=1.3.0
```

### Installation
```bash
pip install neuros-models
# Or specific frameworks:
pip install neuros-models[pytorch]     # PyTorch models only
pip install neuros-models[sklearn]     # Sklearn models only
```

### Use Cases
- Training custom neural decoders
- Fine-tuning models on new datasets
- Benchmarking different architectures

---

## 4. neuros-foundation

**Purpose**: Pre-trained foundation models (POYO, NDT, CEBRA, Neuroformer)

### Components
- `neuros.foundation_models.base_foundation_model`: Base interface
- `neuros.foundation_models.poyo_model`: POYO/POYO+ models
- `neuros.foundation_models.ndt_model`: NDT2/NDT3 models
- `neuros.foundation_models.cebra_model`: CEBRA model
- `neuros.foundation_models.neuroformer_model`: Neuroformer model
- `neuros.datasets.allen_datasets`: Allen Institute data loaders

### Dependencies
```python
neuros-core>=1.0.0
neuros-models>=1.0.0  # Requires PyTorch
# Model-specific dependencies (optional imports):
# - POYO: custom repo
# - NDT: nlb_tools
# - CEBRA: cebra package
# - Neuroformer: custom repo
```

### Installation
```bash
pip install neuros-foundation
# Or specific models:
pip install neuros-foundation[poyo]
pip install neuros-foundation[ndt]
pip install neuros-foundation[cebra]
pip install neuros-foundation[neuroformer]
```

### Use Cases
- Zero-shot neural decoding
- Transfer learning from large-scale datasets
- Research on foundation models

---

## 5. neuros-ui

**Purpose**: User interfaces and visualization

### Components
- `neuros.ui.dashboard`: Streamlit dashboard
- `neuros.ui.api`: FastAPI server
- `neuros.viz`: Visualization utilities

### Dependencies
```python
neuros-core>=1.0.0
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0
plotly>=5.17.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

### Installation
```bash
pip install neuros-ui
pip install neuros-ui[dashboard]  # Streamlit only
pip install neuros-ui[api]        # FastAPI only
```

### Use Cases
- Real-time monitoring dashboards
- REST API deployment
- Interactive visualization

---

## 6. neuros-cloud

**Purpose**: Cloud infrastructure and distributed processing

### Components
- `neuros.cloud.kafka_writer`: Kafka integration
- `neuros.cloud.pipeline`: Cloud pipeline orchestration
- `neuros.cloud.sagemaker`: AWS SageMaker integration
- `neuros.export.webdataset_export`: WebDataset format
- `neuros.export.zarr_export`: Zarr cloud-native format

### Dependencies
```python
neuros-core>=1.0.0
kafka-python>=2.0.2
boto3>=1.28.0
webdataset>=0.2.0
zarr>=2.16.0
```

### Installation
```bash
pip install neuros-cloud
pip install neuros-cloud[kafka]      # Kafka only
pip install neuros-cloud[aws]        # AWS integration
pip install neuros-cloud[export]     # Export formats
```

### Use Cases
- Large-scale data ingestion
- Cloud training/inference
- Multi-site data collection

---

## 7. neuros (Meta-Package)

**Purpose**: Convenient installation of common components

### Installation Options

```bash
# Minimal installation (core only)
pip install neuros-core

# Standard installation (most common use cases)
pip install neuros
# Equivalent to:
# pip install neuros-core neuros-drivers neuros-models

# Full installation (everything)
pip install neuros[all]
# Equivalent to:
# pip install neuros-core neuros-drivers neuros-models \
#             neuros-foundation neuros-ui neuros-cloud

# Custom combinations
pip install neuros[bci]              # core + drivers + models
pip install neuros[research]         # core + models + foundation
pip install neuros[deployment]       # core + ui + cloud
```

### pyproject.toml (Meta-Package)
```toml
[project]
name = "neuros"
version = "1.0.0"
dependencies = [
    "neuros-core>=1.0.0",
    "neuros-drivers>=1.0.0",
    "neuros-models>=1.0.0",
]

[project.optional-dependencies]
all = [
    "neuros-foundation>=1.0.0",
    "neuros-ui>=1.0.0",
    "neuros-cloud>=1.0.0",
]
bci = [
    "neuros-drivers[eeg]>=1.0.0",
    "neuros-models[pytorch]>=1.0.0",
]
research = [
    "neuros-models>=1.0.0",
    "neuros-foundation>=1.0.0",
]
deployment = [
    "neuros-ui>=1.0.0",
    "neuros-cloud>=1.0.0",
]
```

---

## Migration Strategy

### Phase 1: Repository Restructuring (Weeks 1-2)

**Goal**: Reorganize current monolithic repo without breaking changes

1. **Create Package Directories**
   ```
   neuros-v1/
   ├── packages/
   │   ├── neuros-core/
   │   │   ├── pyproject.toml
   │   │   └── src/neuros/core/
   │   ├── neuros-drivers/
   │   │   ├── pyproject.toml
   │   │   └── src/neuros/drivers/
   │   ├── neuros-models/
   │   ├── neuros-foundation/
   │   ├── neuros-ui/
   │   ├── neuros-cloud/
   │   └── neuros/  # Meta-package
   ├── tests/  # Shared test suite
   └── docs/   # Shared documentation
   ```

2. **Move Code to Packages**
   - Move files to appropriate package directories
   - Update imports to use absolute paths
   - Maintain namespace packages (`neuros.*`)

3. **Update Dependencies**
   - Each package gets its own `pyproject.toml`
   - Define minimal dependencies per package
   - Use optional dependency groups

4. **Verify Tests Pass**
   - Run full test suite after each move
   - Update import paths in tests
   - Ensure 100% test pass rate maintained

### Phase 2: Namespace Package Setup (Weeks 3-4)

**Goal**: Enable independent package installation while maintaining unified `neuros.*` namespace

1. **Configure Namespace Packages**
   ```python
   # Each package's src/neuros/__init__.py
   __path__ = __import__('pkgutil').extend_path(__path__, __name__)
   ```

2. **Test Independent Installation**
   ```bash
   # Test minimal installation
   pip install packages/neuros-core
   python -c "from neuros.core.pipeline import Pipeline"

   # Test combinations
   pip install packages/neuros-core packages/neuros-drivers
   python -c "from neuros.drivers.mock_driver import MockDriver"
   ```

3. **Verify Backward Compatibility**
   ```python
   # Old imports should still work:
   from neuros.models import EEGNet  # Should work if neuros-models installed
   from neuros.drivers import MockDriver  # Should work if neuros-drivers installed
   ```

### Phase 3: PyPI Publication (Weeks 5-6)

**Goal**: Publish packages to PyPI for public consumption

1. **Prepare Packages**
   - Add README.md for each package
   - Create package-specific documentation
   - Set version to 1.0.0 for all packages
   - Add LICENSE files

2. **Test Publishing (TestPyPI)**
   ```bash
   # Publish to TestPyPI first
   cd packages/neuros-core
   python -m build
   twine upload --repository testpypi dist/*

   # Test installation
   pip install --index-url https://test.pypi.org/simple/ neuros-core
   ```

3. **Production Publishing**
   ```bash
   # Publish all packages in dependency order:
   # 1. neuros-core (no deps)
   # 2. neuros-drivers, neuros-models (depend on core)
   # 3. neuros-foundation (depends on models)
   # 4. neuros-ui, neuros-cloud (depend on core)
   # 5. neuros (meta-package, depends on all)
   ```

4. **Update Installation Instructions**
   - Update README.md with new installation options
   - Create migration guide for existing users
   - Update all documentation and tutorials

### Phase 4: Documentation & Testing (Weeks 7-8)

**Goal**: Comprehensive docs and 80%+ test coverage

1. **Package-Specific Docs**
   - Create docs/packages/core.md
   - Create docs/packages/drivers.md
   - Create docs/packages/models.md
   - Create docs/packages/foundation.md
   - Create docs/packages/ui.md
   - Create docs/packages/cloud.md

2. **Update Tutorials**
   - Update Tutorial 1-3 with new installation instructions
   - Add "Minimal Installation" sections
   - Show different installation scenarios

3. **Increase Test Coverage**
   - Target 80%+ for neuros-core
   - Target 70%+ for other packages
   - Add integration tests for package combinations

4. **CI/CD Updates**
   - Test each package independently
   - Test package combinations
   - Automated publishing on release tags

---

## API Boundary Definitions

### Core Package Exports

```python
# neuros-core: Public API
from neuros.core.pipeline import Pipeline
from neuros.core.orchestrator import MultiModalOrchestrator
from neuros.core.agents import (
    BaseAgent,
    DeviceAgent,
    ProcessingAgent,
    ModelAgent,
    FusionAgent,
)
from neuros.core.config import Config
from neuros.processing import SignalProcessor, FeatureExtractor
```

### Driver Package Exports

```python
# neuros-drivers: Public API
from neuros.drivers import (
    MockDriver,
    BrainFlowDriver,
    LSLDriver,
    CameraDriver,
    MicrophoneDriver,
)
from neuros.io import (
    BIDSLoader,
    NWBLoader,
)
```

### Models Package Exports

```python
# neuros-models: Public API
from neuros.models import (
    EEGNet,
    TransformerModel,
    LSTMModel,
    SimpleClassifier,
)
from neuros.training import Trainer
from neuros.evaluation import evaluate_model, classification_metrics
```

### Foundation Package Exports

```python
# neuros-foundation: Public API
from neuros.foundation_models import (
    POYOModel,
    POYOPlusModel,
    NDT2Model,
    NDT3Model,
    CEBRAModel,
    NeuroformerModel,
)
from neuros.datasets import (
    load_allen_visual_coding,
    load_allen_neuropixels,
)
```

---

## Backward Compatibility Strategy

### Commitment

**Zero Breaking Changes for v1.0 → v2.0 Transition**

Users should be able to:
```bash
# Old way (still works)
pip install neuros

# Imports unchanged
from neuros.models import EEGNet
from neuros.drivers import MockDriver
from neuros.core.pipeline import Pipeline
```

### Transition Package

Create `neuros` meta-package that:
1. Installs common dependencies (core, drivers, models)
2. Re-exports all public APIs for backward compatibility
3. Shows deprecation warnings for old import paths (optional)

### Migration Guide

Provide clear guide for users:

```markdown
# Migration Guide: Monolithic → Modular

## Quick Start

**Old Installation** (still works):
```bash
pip install neuros  # Installs core + drivers + models
```

**New Installation Options**:
```bash
# Minimal (if you only need pipelines)
pip install neuros-core

# Custom (mix and match)
pip install neuros-core neuros-drivers neuros-ui

# Full (everything, like old way)
pip install neuros[all]
```

## Import Paths (Unchanged)

All import paths remain the same:
```python
from neuros.core.pipeline import Pipeline  # ✓ Still works
from neuros.models import EEGNet          # ✓ Still works
from neuros.drivers import MockDriver     # ✓ Still works
```

## Benefits of Modular Installation

- **Faster installs**: Only install what you need
- **Smaller Docker images**: ~500MB → ~50MB for core-only
- **Fewer dependency conflicts**: Avoid PyTorch if you only need drivers
```

---

## Timeline & Milestones

### Week 1-2: Restructuring
- [ ] Create `packages/` directory structure
- [ ] Move code to appropriate packages
- [ ] Create package-specific `pyproject.toml` files
- [ ] Update imports
- [ ] Verify all 303 tests pass

### Week 3-4: Namespace & Testing
- [ ] Configure namespace packages
- [ ] Test independent package installation
- [ ] Test package combinations
- [ ] Verify backward compatibility
- [ ] Increase test coverage to 80%

### Week 5-6: Publishing
- [ ] Create package READMEs
- [ ] Publish to TestPyPI
- [ ] Test installation from TestPyPI
- [ ] Publish to production PyPI
- [ ] Update main README

### Week 7-8: Documentation
- [ ] Create package-specific docs
- [ ] Update tutorials with new installation options
- [ ] Write migration guide
- [ ] Update CI/CD
- [ ] Announce v2.0 release

**Target Release Date**: 8 weeks from start (~December 2025)

---

## Success Metrics

### Technical Metrics
- ✅ All 303 tests pass after migration
- ✅ Test coverage ≥80% for core, ≥70% for others
- ✅ Zero breaking changes in public API
- ✅ neuros-core installs in <30s (vs. 5+ min for full package)
- ✅ neuros-core Docker image <100MB (vs. 500MB+ currently)

### User Experience Metrics
- ✅ Clear installation documentation for 5+ common scenarios
- ✅ Migration guide with examples
- ✅ Package-specific docs published
- ✅ Tutorials updated with installation options

### Adoption Metrics (Post-Release)
- Track downloads per package (PyPI stats)
- Measure which combinations are most popular
- Gather user feedback on migration experience

---

## Risk Mitigation

### Risk 1: Import Breakage
**Mitigation**: Extensive testing of import paths, maintain meta-package with re-exports

### Risk 2: Version Conflicts
**Mitigation**: Pin compatible versions across packages (neuros-core==1.0.0 required by all)

### Risk 3: Documentation Gaps
**Mitigation**: Create comprehensive migration guide, update all tutorials before release

### Risk 4: CI/CD Complexity
**Mitigation**: Use monorepo tools (e.g., `nx`, `turborepo`) or GitHub Actions matrix

### Risk 5: User Confusion
**Mitigation**: Clear communication, maintain "standard" installation option that works like before

---

## Open Questions

1. **Monorepo vs. Multi-Repo?**
   - **Recommendation**: Start with monorepo (`packages/` structure) for easier coordination
   - Can split to separate repos later if needed

2. **Version Synchronization?**
   - **Recommendation**: All packages versioned together (1.0.0, 1.1.0, etc.)
   - Simpler than independent versioning initially

3. **Entry Points for CLI?**
   - CLI currently in neuros-core, but uses models/drivers
   - **Recommendation**: Move CLI to meta-package or create neuros-cli

4. **Foundation Models Licensing?**
   - Some foundation models have specific licenses
   - **Recommendation**: Clearly document licenses per model, optional installation

---

## Conclusion

This modularization plan directly addresses the evaluation's strongest recommendation:

> "NeurOS can remain integrated in vision but modular in implementation, which is likely the ideal balance."

By splitting into focused packages while maintaining a unified namespace and backward compatibility, we achieve:
- **Flexibility**: Users install only what they need
- **Maintainability**: Clear separation of concerns
- **Scalability**: Can grow packages independently
- **Accessibility**: Lower barrier to entry for simple use cases

**Next Steps**: Review this plan, then begin Phase 1 (Repository Restructuring).

---

## References

1. ChatGPT Evaluation Round 2 (chatGPT-eval2.pdf)
2. Current NeurOS Architecture (neuros/*)
3. Python Packaging Guide: https://packaging.python.org/en/latest/guides/packaging-namespace-packages/
4. Example Multi-Package Projects: PyTorch Lightning, HuggingFace Transformers
