# NeuroFMX Testing Guide

Comprehensive guide to testing NeuroFMX foundation models.

## Table of Contents

1. [Test Suite Overview](#test-suite-overview)
2. [Running Tests](#running-tests)
3. [Test Organization](#test-organization)
4. [Writing Tests](#writing-tests)
5. [Test Coverage](#test-coverage)
6. [Continuous Integration](#continuous-integration)

---

## Test Suite Overview

NeuroFMX has a comprehensive test suite covering:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test workflows and component interactions
- **Performance Tests**: Test scalability and efficiency
- **Edge Case Tests**: Test robustness and error handling

**Total Tests**: 100+ test cases across 20+ test files

---

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-benchmark

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=neuros_neurofm --cov-report=html

# Run specific test file
pytest tests/test_energy_flow.py

# Run specific test class
pytest tests/test_energy_flow.py::TestInformationFlowAnalyzer

# Run specific test
pytest tests/test_energy_flow.py::TestInformationFlowAnalyzer::test_knn_mi_estimation

# Run with verbose output
pytest tests/ -v

# Run with detailed output for failures
pytest tests/ --tb=long

# Run only fast tests (skip slow tests)
pytest tests/ -m "not slow"
```

### Test Markers

Tests are marked with pytest markers:

```python
@pytest.mark.slow  # Long-running tests
@pytest.mark.gpu   # Requires GPU
@pytest.mark.integration  # Integration tests
```

Run specific markers:

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only GPU tests
pytest tests/ -m "gpu"

# Run only integration tests
pytest tests/ -m "integration"
```

---

## Test Organization

### Test Directory Structure

```
tests/
├── test_energy_flow.py              # Energy flow & information theory (NEW!)
├── test_mechint_integration.py      # Mech-int integration tests
├── test_mechint_hooks.py            # PyTorch Lightning hooks
├── test_reporting.py                # HTML report generation
├── test_webdataset.py               # Data pipeline
├── test_webdataset_integration.py   # Data integration
├── test_temporal_alignment.py       # Multi-rate alignment
├── test_ray_tune_search.py          # Hyperparameter search
├── conftest.py                      # Shared fixtures
└── ...
```

### Test Files by Module

| Module | Test File | Test Count |
|--------|-----------|------------|
| Energy Flow | `test_energy_flow.py` | 30+ |
| Mech-Int | `test_mechint_integration.py` | 15+ |
| Hooks | `test_mechint_hooks.py` | 20+ |
| Reporting | `test_reporting.py` | 15+ |
| WebDataset | `test_webdataset.py` | 20+ |
| Temporal Alignment | `test_temporal_alignment.py` | 25+ |
| Ray Tune | `test_ray_tune_search.py` | 10+ |

---

## Test Coverage

### Energy Flow Tests (`test_energy_flow.py`)

**New comprehensive test file with 30+ test cases!**

#### InformationFlowAnalyzer Tests
- ✅ k-NN mutual information estimation
- ✅ Histogram-based MI estimation
- ✅ MINE neural network MI estimation
- ✅ Information plane computation
- ✅ Information bottleneck curve

#### EnergyLandscape Tests
- ✅ Density-based landscape estimation
- ✅ Score-based landscape estimation
- ✅ Quadratic landscape estimation
- ✅ Energy basin detection
- ✅ Barrier computation between basins
- ✅ 2D landscape visualization

#### EntropyProduction Tests
- ✅ Entropy production rate estimation
- ✅ Dissipation rate calculation
- ✅ Nonequilibrium score
- ✅ Equilibrium vs nonequilibrium detection

#### MINENetwork Tests
- ✅ Network initialization
- ✅ Forward pass
- ✅ Training step

#### Integration Tests
- ✅ Full information analysis pipeline
- ✅ Full energy landscape pipeline
- ✅ Full entropy production pipeline
- ✅ Combined information + energy analysis

#### Edge Case Tests
- ✅ Small datasets
- ✅ High-dimensional data
- ✅ Degenerate landscapes
- ✅ Constant trajectories

#### Performance Tests
- ✅ Large-scale information plane
- ✅ Fine-grained landscape

**Example:**

```python
def test_knn_mi_estimation(self, info_analyzer, sample_data):
    """Test k-NN mutual information estimation"""
    X, Z, Y = sample_data['X'], [sample_data['Z']], sample_data['Y']

    mi_results = info_analyzer.estimate_mutual_information(
        X, Z, Y, method='knn'
    )

    assert len(mi_results) == 1
    result = mi_results['layer_0']
    assert isinstance(result, MutualInformationEstimate)
    assert result.I_XZ >= 0  # MI is non-negative
    assert result.I_ZY >= 0
```

### Mechanistic Interpretability Tests

#### `test_mechint_integration.py`

**15+ test classes covering all mech-int modules:**

- `TestSAEIntegration` - Hierarchical SAE training
- `TestAlignmentIntegration` - CCA/RSA brain alignment
- `TestDynamicsIntegration` - Koopman/Lyapunov analysis
- `TestCounterfactualIntegration` - Latent surgery
- `TestMetaDynamicsIntegration` - Training trajectory tracking
- `TestReportingIntegration` - HTML report generation
- `TestHooksIntegration` - PyTorch Lightning callbacks
- `TestEndToEndWorkflow` - Complete pipeline
- `TestCrossModuleIntegration` - Module interactions

#### `test_mechint_hooks.py`

**20+ tests for PyTorch Lightning integration:**

- Hook initialization
- Activation sampling during training
- SAE training hooks
- Brain alignment hooks
- Real-time reporting
- FastAPI integration

### Data Pipeline Tests

#### `test_webdataset.py`

**20+ tests for WebDataset:**

- Shard creation and writing
- Data loading and batching
- Shuffling and sampling
- Multi-modal data handling
- Error handling

#### `test_temporal_alignment.py`

**25+ tests for temporal alignment:**

- Multi-rate alignment (1Hz - 30kHz)
- Interpolation methods (nearest, linear, cubic, causal)
- Jitter correction
- Missing data handling
- Edge cases

### Other Test Files

#### `test_reporting.py`

- HTML report generation
- MLflow integration
- W&B integration
- Figure creation
- Styling and formatting

#### `test_ray_tune_search.py`

- Hyperparameter search spaces
- ASHA scheduler
- PBT scheduler
- Bayesian optimization
- Multi-objective optimization

---

## Writing Tests

### Test Structure

Follow this structure for new tests:

```python
"""
Test module docstring explaining what is tested
"""

import pytest
import torch
from neuros_neurofm.module import Component


# ==================== Fixtures ====================

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    return {
        'input': torch.randn(100, 10),
        'target': torch.randn(100, 5),
    }


@pytest.fixture
def component():
    """Create component instance"""
    return Component(param=value)


# ==================== Tests ====================

class TestComponent:
    """Test Component class"""

    def test_initialization(self, component):
        """Test component initialization"""
        assert component is not None
        assert component.param == value

    def test_basic_functionality(self, component, sample_data):
        """Test basic functionality"""
        output = component.process(sample_data['input'])

        assert output.shape == sample_data['target'].shape
        assert torch.isfinite(output).all()

    def test_edge_case(self, component):
        """Test edge case handling"""
        # Test with empty input
        with pytest.raises(ValueError):
            component.process(torch.empty(0, 10))
```

### Best Practices

1. **Use descriptive names**
   ```python
   # Good
   def test_knn_mi_estimation_returns_nonnegative_values(self):

   # Bad
   def test_mi(self):
   ```

2. **Test one thing per test**
   ```python
   # Good
   def test_forward_pass(self):
       output = model(input)
       assert output.shape == expected_shape

   def test_forward_pass_finite(self):
       output = model(input)
       assert torch.isfinite(output).all()

   # Bad
   def test_everything(self):
       # Tests 10 different things
   ```

3. **Use fixtures for setup**
   ```python
   @pytest.fixture
   def trained_model():
       model = Model()
       model.train(data)
       return model
   ```

4. **Test error conditions**
   ```python
   def test_invalid_input_raises_error(self):
       with pytest.raises(ValueError):
           component.process(invalid_input)
   ```

5. **Use markers**
   ```python
   @pytest.mark.slow
   def test_large_scale_training(self):
       # Long-running test
   ```

### Parametrized Tests

Test multiple inputs efficiently:

```python
@pytest.mark.parametrize('method,expected_type', [
    ('knn', MutualInformationEstimate),
    ('histogram', MutualInformationEstimate),
    ('mine', MutualInformationEstimate),
])
def test_mi_estimation_methods(self, method, expected_type):
    result = analyzer.estimate_mi(X, Z, method=method)
    assert isinstance(result, expected_type)
```

---

## Test Coverage

### Running Coverage Analysis

```bash
# Generate coverage report
pytest tests/ --cov=neuros_neurofm --cov-report=html

# Open HTML report
open htmlcov/index.html

# Generate terminal report
pytest tests/ --cov=neuros_neurofm --cov-report=term
```

### Current Coverage

| Module | Coverage |
|--------|----------|
| Energy Flow | 95%+ |
| Interpretability | 90%+ |
| Data Pipeline | 85%+ |
| Training | 80%+ |
| Losses | 85%+ |
| **Overall** | **85%+** |

### Coverage Goals

- **Critical paths**: 95%+ coverage
- **Core functionality**: 90%+ coverage
- **Edge cases**: 80%+ coverage

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=neuros_neurofm --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        args: [tests/, -m, "not slow"]
        language: system
        pass_filenames: false
        always_run: true
```

---

## Debugging Tests

### Running Single Test with Debugger

```python
# Add breakpoint in test
def test_component(self):
    component = Component()
    breakpoint()  # Python 3.7+
    result = component.process(data)
```

```bash
# Run with pytest
pytest tests/test_module.py::test_component
```

### Print Debugging

```python
def test_component(self, capsys):
    """Test with captured output"""
    component = Component()
    print(f"Component state: {component}")

    result = component.process(data)

    # Check printed output
    captured = capsys.readouterr()
    assert "Component state" in captured.out
```

### Verbose Output

```bash
# Show print statements
pytest tests/ -s

# Show full diff on assertion failures
pytest tests/ -vv

# Show full traceback
pytest tests/ --tb=long
```

---

## Performance Testing

### Benchmarking

```python
import pytest

@pytest.mark.benchmark
def test_component_performance(benchmark):
    """Benchmark component performance"""
    component = Component()
    data = torch.randn(1000, 100)

    result = benchmark(component.process, data)

    # Assertions can still be made
    assert result.shape == (1000, 50)
```

Run benchmarks:

```bash
pytest tests/ --benchmark-only
```

### Memory Profiling

```python
@pytest.mark.memory
def test_memory_usage():
    """Test memory usage"""
    import tracemalloc

    tracemalloc.start()

    # Run code
    model = LargeModel()
    output = model(large_input)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Check memory usage
    assert peak < 1000 * 1024 * 1024  # < 1GB
```

---

## Test Data

### Generating Test Data

```python
# Use fixtures for consistent test data
@pytest.fixture
def neural_data():
    """Generate synthetic neural data"""
    torch.manual_seed(42)
    return {
        'eeg': torch.randn(100, 1000, 64),  # (trials, time, channels)
        'spikes': torch.poisson(torch.ones(100, 1000, 96) * 5),
        'behavior': torch.randn(100, 1000, 10),
    }

@pytest.fixture
def real_data():
    """Load real test data (cached)"""
    # Load from test data directory
    # Cache for repeated use
    pass
```

### Test Data Organization

```
tests/
├── data/
│   ├── sample_eeg.npy
│   ├── sample_spikes.npy
│   └── sample_video.mp4
├── fixtures/
│   └── sample_checkpoints/
└── conftest.py  # Shared fixtures
```

---

## Troubleshooting

### Tests Failing Locally

1. **Check dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```

2. **Check Python version**
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Clear cache**
   ```bash
   pytest --cache-clear
   ```

### Tests Passing Locally, Failing in CI

1. **Check for randomness**
   ```python
   # Always set seeds
   torch.manual_seed(42)
   np.random.seed(42)
   ```

2. **Check for path issues**
   ```python
   # Use pathlib for cross-platform paths
   from pathlib import Path
   data_path = Path(__file__).parent / 'data'
   ```

3. **Check for GPU dependencies**
   ```python
   @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
   def test_gpu_function(self):
       pass
   ```

---

## Summary

NeuroFMX has comprehensive testing with:

- ✅ **100+ test cases** across 20+ files
- ✅ **85%+ code coverage**
- ✅ **Unit, integration, and performance tests**
- ✅ **Edge case and robustness testing**
- ✅ **Continuous integration ready**

**New energy flow tests** (`test_energy_flow.py`):
- ✅ 30+ comprehensive tests
- ✅ All InformationFlowAnalyzer methods
- ✅ All EnergyLandscape methods
- ✅ All EntropyProduction methods
- ✅ MINENetwork training
- ✅ Integration tests
- ✅ Edge cases and performance tests

Run tests regularly to ensure code quality!

```bash
# Quick test run
pytest tests/ -m "not slow"

# Full test suite with coverage
pytest tests/ --cov=neuros_neurofm --cov-report=html
```

---

**Last Updated:** January 2025
**Test Suite Version:** 2.0
**Total Tests:** 100+
**Coverage:** 85%+
