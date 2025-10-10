# Contributing to NeurOS

Thank you for your interest in contributing to NeurOS! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Adding New Features](#adding-new-features)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

### Prerequisites

- **Python 3.10 or higher** (recommended: 3.11+)
- **Git** for version control
- **Virtual environment** (venv, conda, or similar)
- Optional: **BrainFlow** for hardware support
- Optional: **Lab Streaming Layer (LSL)** for real-time synchronization

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/neuros-v1.git
cd neuros-v1

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install -e ".[test,dashboard,notebook]"

# Verify installation
neuros --help
pytest tests/
```

---

## Development Setup

### Recommended Development Tools

- **IDE:** VS Code, PyCharm, or similar with Python support
- **Linter:** `ruff` or `flake8`
- **Formatter:** `black`
- **Type checker:** `mypy`
- **Git hooks:** `pre-commit` (optional but recommended)

### Setting Up Pre-Commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  - repo: https://github.com/psf/black
    rev: 23.10.0
    hooks:
      - id: black
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

---

## Project Structure

```
neuros-v1/
├── neuros/                 # Main package
│   ├── agents/            # Agent implementations (orchestrator, device, processing, model)
│   ├── api/               # FastAPI REST API
│   ├── drivers/           # Hardware drivers (BrainFlow, mock, video, motion, etc.)
│   ├── models/            # ML models (EEGNet, CNN, Transformer, etc.)
│   ├── processing/        # Signal processing (filters, features, adaptation)
│   ├── db/                # Database utilities
│   ├── export/            # Data export (WebDataset, Petastorm)
│   ├── ingest/            # Data ingestion (Kafka, Redis, ZMQ)
│   ├── cloud/             # Cloud deployment (Constellation pipeline)
│   ├── cli.py             # Command-line interface
│   ├── pipeline.py        # Pipeline wrapper classes
│   ├── autoconfig.py      # Automatic pipeline configuration
│   ├── dashboard.py       # Streamlit dashboard
│   └── security.py        # Authentication and security
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── notebooks/             # Example Jupyter notebooks
├── setup.py               # Package configuration
├── requirements.txt       # Dependencies
├── pytest.ini             # Pytest configuration
└── README.md              # Project overview
```

---

## Coding Standards

### General Principles

1. **Clarity over cleverness** - Write code that is easy to understand
2. **Modularity** - Each module should have a single, well-defined purpose
3. **Documentation** - All public APIs must have docstrings
4. **Type hints** - Use type annotations for function signatures
5. **Async-first** - Prefer async/await for I/O operations

### Python Style

We follow **PEP 8** with some modifications:

- **Line length:** 120 characters (not 79)
- **String quotes:** Prefer double quotes `"` for strings
- **Imports:** Group in order: stdlib, third-party, local
- **Type hints:** Required for all public functions
- **Docstrings:** Google-style or NumPy-style (be consistent)

### Example Function

```python
"""Good example of a well-documented function."""
from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np


async def process_signal(
    data: np.ndarray,
    fs: float,
    *,
    window_size: Optional[int] = None,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """Process a signal into overlapping windows.

    Parameters
    ----------
    data : np.ndarray
        Input signal of shape (n_samples,) or (n_channels, n_samples).
    fs : float
        Sampling frequency in Hz.
    window_size : int, optional
        Window size in samples. If None, uses 1 second of data.
    overlap : float, optional
        Overlap fraction between 0 and 1. Default is 0.5 (50% overlap).

    Returns
    -------
    list[np.ndarray]
        List of windowed segments.

    Raises
    ------
    ValueError
        If overlap is not between 0 and 1.

    Examples
    --------
    >>> signal = np.random.randn(1000)
    >>> windows = await process_signal(signal, fs=250.0, window_size=100)
    >>> len(windows)
    19
    """
    if not 0 <= overlap < 1:
        raise ValueError(f"Overlap must be between 0 and 1, got {overlap}")

    window_size = window_size or int(fs)
    step = int(window_size * (1 - overlap))

    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i : i + window_size])

    return windows
```

### Docstring Format

Use **NumPy-style** docstrings for consistency:

```python
def function_name(param1: int, param2: str) -> bool:
    """One-line summary.

    Optional longer description that provides more context.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.

    Raises
    ------
    ValueError
        When param1 is negative.

    Examples
    --------
    >>> function_name(42, "hello")
    True
    """
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_pipeline.py

# Run tests matching pattern
pytest -k "test_model"

# Run with coverage
pytest --cov=neuros --cov-report=html

# Run only fast tests (skip slow/hardware tests)
pytest -m "not slow and not hardware"
```

### Writing Tests

#### Test Structure

```python
"""Test module docstring."""
import pytest
import numpy as np

from neuros.models import SimpleClassifier


class TestSimpleClassifier:
    """Tests for SimpleClassifier."""

    def test_train_and_predict(self):
        """Test that model can train and predict."""
        # Arrange
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)

        model = SimpleClassifier()

        # Act
        model.train(X_train, y_train)
        predictions = model.predict(X_test)

        # Assert
        assert len(predictions) == 20
        assert all(p in [0, 1] for p in predictions)

    def test_predict_before_train_raises(self):
        """Test that predicting before training raises error."""
        model = SimpleClassifier()
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="not trained"):
            model.predict(X)
```

#### Async Tests

```python
import pytest


@pytest.mark.asyncio
async def test_pipeline_run():
    """Test that pipeline runs successfully."""
    from neuros.pipeline import Pipeline
    from neuros.drivers import MockDriver

    pipeline = Pipeline(driver=MockDriver())
    metrics = await pipeline.run(duration=0.5)

    assert "throughput" in metrics
    assert metrics["throughput"] > 0
```

#### Test Markers

Use markers to organize tests:

```python
@pytest.mark.unit
def test_something():
    """Unit test."""

@pytest.mark.integration
async def test_full_pipeline():
    """Integration test."""

@pytest.mark.slow
def test_long_running():
    """Slow test."""

@pytest.mark.hardware
def test_openbci():
    """Test requiring real hardware."""
```

### Test Coverage

Aim for **>80% code coverage** for all new features. Check coverage with:

```bash
pytest --cov=neuros --cov-report=term-missing
```

---

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest main:
   ```bash
   git checkout main
   git pull origin main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Run tests** and ensure they pass:
   ```bash
   pytest
   ```

3. **Check code style**:
   ```bash
   black neuros/ tests/
   ruff check neuros/ tests/
   mypy neuros/
   ```

4. **Update documentation** if you changed public APIs

5. **Add tests** for new functionality

### PR Title Format

Use conventional commits format:

```
type(scope): brief description

Examples:
feat(models): add LSTM model for sequence classification
fix(drivers): resolve BrainFlow connection timeout
docs(readme): update installation instructions
test(pipeline): add integration test for multi-modal pipeline
refactor(agents): simplify orchestrator initialization
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring without changing functionality
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- Bullet list of specific changes
- Each change on its own line

## Testing
How has this been tested?
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Tests added and passing
- [ ] No breaking changes (or documented if unavoidable)
- [ ] CHANGELOG updated (if applicable)
```

### Review Process

1. **Submit PR** with clear title and description
2. **CI checks** must pass (tests, linting)
3. **Code review** by at least one maintainer
4. **Address feedback** - make requested changes
5. **Approval** - maintainer approves PR
6. **Merge** - squash commits and merge to main

---

## Adding New Features

### Adding a New Driver

Drivers provide data from hardware or simulated sources.

1. **Create driver file**: `neuros/drivers/my_device_driver.py`

2. **Inherit from BaseDriver**:

```python
from neuros.drivers.base_driver import BaseDriver


class MyDeviceDriver(BaseDriver):
    """Driver for MyDevice hardware.

    Parameters
    ----------
    device_id : str
        Identifier for the device.
    sampling_rate : float, optional
        Sampling rate in Hz. Default is 250.0.
    channels : int, optional
        Number of channels. Default is 8.
    """

    def __init__(
        self,
        device_id: str,
        sampling_rate: float = 250.0,
        channels: int = 8,
    ):
        super().__init__(sampling_rate=sampling_rate, channels=channels)
        self.device_id = device_id
        self._connection = None

    async def start(self) -> None:
        """Start the device and begin streaming."""
        # Initialize hardware connection
        self._connection = await self._connect_to_device()

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        """Stream data from device.

        Yields
        ------
        timestamp : float
            Unix timestamp in seconds.
        data : np.ndarray
            Data array of shape (n_channels,).
        """
        while self._running:
            timestamp, data = await self._connection.read_sample()
            yield timestamp, data

    async def stop(self) -> None:
        """Stop streaming and disconnect."""
        if self._connection:
            await self._connection.close()
```

3. **Add tests**: `tests/test_my_device_driver.py`

4. **Register in autoconfig** (optional): Add to `neuros/autoconfig.py`

### Adding a New Model

Models encapsulate training and inference logic.

1. **Create model file**: `neuros/models/my_model.py`

2. **Inherit from BaseModel**:

```python
from neuros.models.base_model import BaseModel
import numpy as np


class MyModel(BaseModel):
    """Custom model implementation.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_classes : int, optional
        Number of output classes. Default is 2.
    """

    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        # Initialize model parameters

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).
        y : np.ndarray
            Training labels of shape (n_samples,).
        """
        # Training logic
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input features.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (n_samples,).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        # Prediction logic
        return predictions
```

3. **Add to model registry**: `neuros/models/__init__.py`

4. **Add tests**: `tests/test_models.py`

### Adding a New Processing Agent

Processing agents handle modality-specific data transformations.

1. **Create agent file**: `neuros/agents/my_agent.py`

2. **Inherit from BaseAgent**:

```python
from neuros.agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    """Agent for processing MyModality data.

    Parameters
    ----------
    input_queue : asyncio.Queue
        Queue receiving raw data.
    output_queue : asyncio.Queue
        Queue for processed features.
    """

    def __init__(self, input_queue, output_queue, **kwargs):
        super().__init__(name="MyAgent")
        self.input_queue = input_queue
        self.output_queue = output_queue
        # Initialize agent-specific parameters

    async def run(self) -> None:
        """Main processing loop."""
        while True:
            # Get data from input queue
            timestamp, data = await self.input_queue.get()

            # Process data
            features = self._extract_features(data)

            # Send to output queue
            await self.output_queue.put((timestamp, features))

    def _extract_features(self, data):
        """Extract features from raw data."""
        # Feature extraction logic
        return features
```

---

## Reporting Issues

### Bug Reports

Use the issue template:

**Title:** Clear, descriptive title

**Description:**
- What happened?
- What did you expect to happen?
- Steps to reproduce
- System information (OS, Python version, NeurOS version)
- Relevant code snippets or error messages

**Labels:** bug, priority (if urgent)

### Feature Requests

**Title:** Clear description of the feature

**Description:**
- What problem does this solve?
- Proposed solution
- Alternatives considered
- Would you be willing to implement it?

**Labels:** enhancement

---

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them get started
- Focus on the best outcome for the project
- Assume good faith
- Follow the community guidelines

---

## Questions?

- **Documentation:** Check the [README](README.md) and [docs/](docs/) folder
- **Discussions:** Use GitHub Discussions for questions
- **Issues:** For bugs or feature requests
- **Email:** contact@neuros.ai (if available)

---

## License

By contributing to NeurOS, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to NeurOS! 🧠✨
