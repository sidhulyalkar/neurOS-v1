# Week 1 Sprint Plan: Foundation Models & Core Quality
**Date:** October 11, 2025
**Duration:** This Week (5-7 days)
**Team:** Claude Code Agent (Parallel Execution Mode)

---

## Sprint Goals

1. ✅ Rename Constellation → Foundation Models Demo
2. 📊 Complete coverage analysis
3. 🧪 Fix all failing tests
4. 📚 Setup documentation infrastructure
5. 🧬 Implement NWB support (DIAMOND completion)
6. 📓 Create first 2 tutorial notebooks

---

## Task 1: Rename Constellation → Foundation Models Demo

**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** IN PROGRESS

### Files to Rename/Update:
- [ ] `docs/runbook_constellation.md` → `docs/runbook_foundation_models_demo.md`
- [ ] `notebooks/constellation_demo.ipynb` → `notebooks/foundation_models_demo.ipynb`
- [ ] Update all references in:
  - [ ] `neuros/cli.py`
  - [ ] `scripts/run_local_demo.py`
  - [ ] `README.md`
  - [ ] `QUICKSTART.md`
  - [ ] `AUDIT.md`
  - [ ] `setup.py`

### New Focus:
**Old:** Multi-modal synchronization (EEG, video, audio, Kafka streaming)
**New:** Foundation Models showcase (POYO+, NDT3, CEBRA, Neuroformer with real data)

### Demo Content:
1. Load Allen Visual Coding dataset
2. Demonstrate each foundation model
3. Show transfer learning (cross-session)
4. Compare with classical models
5. Real-time inference dashboard
6. Zero-shot and few-shot examples

---

## Task 2: Coverage Analysis

**Priority:** CRITICAL | **Effort:** 1 hour | **Status:** PENDING

### Commands:
```bash
# Install coverage if needed
pip install coverage pytest-cov

# Run coverage analysis
pytest --cov=neuros --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

### Analysis Steps:
1. Identify overall coverage percentage
2. Find uncovered modules
3. Prioritize critical paths (pipeline, agents, models)
4. Create test plan for gaps

### Goal:
- Current: Unknown
- Target: >90% coverage
- Critical modules must be >95%

---

## Task 3: Fix Failing Tests

**Priority:** CRITICAL | **Effort:** 2-4 hours | **Status:** PENDING

### Known Issues:
1. **LSL Sync Tests** (`tests/test_sync.py`)
   - Error: LSL binary library not found
   - Solution: Mock LSL or mark as optional

2. **Async Test Markers**
   - Some async tests may need proper markers

3. **Optional Dependencies**
   - Tests should handle missing deps gracefully

### Fix Strategy:
```python
# For LSL tests
@pytest.mark.skipif(not LSL_AVAILABLE, reason="LSL not installed")
def test_lsl_sync():
    ...

# Or mock it
@patch('neuros.sync.lsl_sync.resolve_stream')
def test_lsl_sync_mocked(mock_resolve):
    ...
```

---

## Task 4: Setup MkDocs Documentation

**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** PENDING

### Structure:
```
docs/
├── index.md (landing page)
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── first-pipeline.md
├── user-guide/
│   ├── pipelines.md
│   ├── models.md
│   ├── foundation-models.md
│   ├── agents.md
│   └── cli.md
├── tutorials/
│   ├── basic-pipeline.md
│   ├── foundation-models.md
│   ├── multi-modal.md
│   └── custom-models.md
├── api-reference/
│   ├── pipeline.md
│   ├── models.md
│   ├── agents.md
│   └── foundation-models.md
├── development/
│   ├── contributing.md
│   ├── architecture.md
│   └── testing.md
└── about/
    ├── comparison.md
    └── roadmap.md
```

### Setup Commands:
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs new .
# Configure mkdocs.yml
mkdocs serve  # Test locally
```

---

## Task 5: NWB Support Implementation

**Priority:** HIGH | **Effort:** 6-8 hours | **Status:** PENDING

### Implementation Plan:

**File:** `neuros/io/nwb_loader.py`

```python
"""
NWB (Neurodata Without Borders) file format support.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

try:
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import LFP, ElectricalSeries
    NWB_AVAILABLE = True
except ImportError:
    NWB_AVAILABLE = False

class NWBLoader:
    """Load neural data from NWB files."""

    def __init__(self, file_path: str):
        if not NWB_AVAILABLE:
            raise ImportError("pynwb not installed")
        self.file_path = Path(file_path)

    def load_spikes(self) -> Dict[str, np.ndarray]:
        """Load spike times from NWB file."""
        pass

    def load_lfp(self) -> np.ndarray:
        """Load LFP data from NWB file."""
        pass

    def load_behavior(self) -> Dict[str, np.ndarray]:
        """Load behavioral data from NWB file."""
        pass

class NWBWriter:
    """Write neural data to NWB files."""

    def __init__(self, file_path: str, session_description: str):
        pass

    def add_spikes(self, spike_times: Dict[str, np.ndarray]):
        """Add spike data to NWB file."""
        pass

    def add_lfp(self, lfp_data: np.ndarray, rate: float):
        """Add LFP data to NWB file."""
        pass
```

### Testing:
```python
# tests/test_nwb_io.py
def test_nwb_loader():
    # Test with mock NWB file
    pass

def test_nwb_writer():
    # Test writing to NWB
    pass
```

### Documentation:
- Tutorial: "Working with NWB Files"
- Integration with Allen datasets

---

## Task 6: Tutorial Notebooks

**Priority:** HIGH | **Effort:** 4-6 hours each | **Status:** PENDING

### Tutorial 1: Basic Pipeline Setup
**File:** `notebooks/tutorial_01_basic_pipeline.ipynb`

**Content:**
1. Installation and setup
2. Creating a simple pipeline
3. Loading data (simulated EEG)
4. Adding preprocessing
5. Training a classifier (SVM)
6. Making predictions
7. Evaluating results

**Code Outline:**
```python
# 1. Imports
from neuros.pipeline import Pipeline
from neuros.models import SVMModel

# 2. Create pipeline
pipeline = Pipeline(
    driver='simulated_eeg',
    processor='bandpass_filter',
    model='svm'
)

# 3. Run pipeline
results = pipeline.run(duration=60)

# 4. Visualize
pipeline.plot_results()
```

### Tutorial 2: Foundation Models Showcase
**File:** `notebooks/tutorial_02_foundation_models.ipynb`

**Content:**
1. Introduction to foundation models
2. Loading Allen dataset
3. Using POYO+ for multi-task decoding
4. Using NDT3 for motor decoding
5. Using CEBRA for latent embeddings
6. Using Neuroformer for zero-shot prediction
7. Comparing performance
8. Transfer learning example

**Code Outline:**
```python
# Load Allen data
from neuros.datasets import load_allen_visual_coding
data = load_allen_visual_coding()

# POYO+ multi-task
from neuros.foundation_models import POYOPlusModel
poyo = POYOPlusModel.from_pretrained('poyo-plus-base')
predictions = poyo.predict(data['spikes'])

# NDT3 motor decoding
from neuros.foundation_models import NDT3Model
ndt3 = NDT3Model.from_pretrained('ndt3-motor')
motor_output = ndt3.predict(data['spikes'])

# CEBRA embeddings
from neuros.foundation_models import CEBRAModel
cebra = CEBRAModel(input_dim=96, output_dim=3)
embeddings = cebra.fit_transform(data['spikes'])

# Visualize embeddings
import matplotlib.pyplot as plt
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=data['labels'])
```

---

## Task 7: TorchBrain Interoperability Test

**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** PENDING

### Investigation:
1. Install torch_brain (optional dependency)
2. Test loading a TorchBrain model
3. Create wrapper class
4. Document integration pattern

### Wrapper Implementation:
```python
# neuros/foundation_models/torchbrain_wrapper.py
try:
    import torch_brain
    TORCHBRAIN_AVAILABLE = True
except ImportError:
    TORCHBRAIN_AVAILABLE = False

class TorchBrainModelWrapper(BaseFoundationModel):
    """Wrapper for TorchBrain models."""

    def __init__(self, model_name: str):
        if not TORCHBRAIN_AVAILABLE:
            raise ImportError("torch_brain not installed")
        self.model = torch_brain.models.load(model_name)

    def predict(self, X):
        return self.model(X)
```

---

## Parallel Execution Strategy

### Stream 1: Documentation (2-3 hours)
1. Setup MkDocs structure
2. Write index.md and installation.md
3. Auto-generate API docs
4. Deploy to GitHub Pages

### Stream 2: Testing (3-4 hours)
1. Run coverage analysis
2. Fix LSL test failures
3. Add missing tests for critical paths
4. Verify all 239+ tests pass

### Stream 3: NWB Implementation (6-8 hours)
1. Implement NWBLoader class
2. Implement NWBWriter class
3. Add tests
4. Create tutorial example
5. Update documentation

### Stream 4: Tutorial Notebooks (4-6 hours)
1. Create Tutorial 1 (Basic Pipeline)
2. Create Tutorial 2 (Foundation Models)
3. Test notebooks end-to-end
4. Add to documentation

### Stream 5: Demo Refactoring (2-3 hours)
1. Rename Constellation files
2. Update all references
3. Refactor demo to focus on foundation models
4. Test demo runs successfully

---

## Success Criteria

By end of Week 1:
- [ ] All 239+ tests passing (no failures)
- [ ] Coverage >80% (goal: 90%)
- [ ] MkDocs site live with 5+ pages
- [ ] NWB support implemented and tested
- [ ] 2 tutorial notebooks complete
- [ ] Foundation Models demo renamed and functional
- [ ] TorchBrain interoperability tested

---

## Risks and Mitigation

**Risk 1:** NWB implementation takes longer than expected
**Mitigation:** Implement loader first (read-only), defer writer to Week 2

**Risk 2:** Coverage analysis reveals <50% coverage
**Mitigation:** Focus on critical path tests first, defer comprehensive coverage to Week 2

**Risk 3:** Documentation setup complex
**Mitigation:** Use mkdocs-material template, copy structure from established projects

---

## Week 2 Preview

Next week priorities:
1. Tutorial 3-5 (Multi-modal, Custom Models, Benchmarking)
2. TemporalData interoperability
3. Foundation model pretrained weights integration
4. Advanced visualizations
5. BIDS support
6. Online learning implementation

---

**Let's execute this systematically and get NeurOS conference-ready! 🚀**
