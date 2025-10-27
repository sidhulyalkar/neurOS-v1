# Contributing to neuros-mechint

Thank you for your interest in contributing to neuros-mechint! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/neuros-ai/neuros-mechint/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Code snippet if applicable
   - Environment details (OS, Python version, package versions)

### Suggesting Features

1. Check [Issues](https://github.com/neuros-ai/neuros-mechint/issues) for existing requests
2. Create a new issue with:
   - Clear use case and motivation
   - Proposed API or interface
   - Examples of how it would be used
   - Any relevant research papers or references

### Pull Requests

1. **Fork the repository** and create a branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** including docstrings and README if needed
5. **Run tests** to ensure everything passes
6. **Submit a pull request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/neuros-mechint
cd neuros-mechint

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy src/neuros_mechint

# Format code
black src/neuros_mechint
isort src/neuros_mechint
```

## Coding Standards

### Style Guide

- **PEP 8**: Follow Python's official style guide
- **Line length**: Maximum 100 characters
- **Formatting**: Use `black` and `isort`
- **Type hints**: Required for all functions
- **Docstrings**: Google-style docstrings for all public APIs

### Example

```python
from typing import Optional
import torch
from torch import Tensor


def compute_metric(
    signal: Tensor,
    k_max: int = 10,
    device: Optional[str] = None,
) -> Tensor:
    """
    Compute some metric from a signal.

    Args:
        signal: Input signal tensor of shape (batch, time)
        k_max: Maximum window size
        device: Torch device ('cuda' or 'cpu')

    Returns:
        Computed metric tensor of shape (batch,)

    Example:
        >>> signal = torch.randn(32, 1000)
        >>> metric = compute_metric(signal, k_max=10)
        >>> print(f"Metric: {metric.mean():.3f}")
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    signal = signal.to(device)
    # Implementation...
    return result
```

### Documentation Requirements

All public classes and functions must have:

1. **One-line summary**
2. **Detailed description** (if needed)
3. **Args** section with types and descriptions
4. **Returns** section with type and description
5. **Example** section with working code

### Testing Requirements

- **Unit tests** for all new functions
- **Integration tests** for new features
- **Coverage**: Aim for >80%
- **Fixtures**: Use pytest fixtures for common setup

Example test:

```python
import pytest
import torch
from neuros_mechint.fractals import HiguchiFractalDimension


def test_higuchi_fd_basic():
    """Test Higuchi FD computation on simple signal."""
    fd = HiguchiFractalDimension(k_max=10)
    signal = torch.randn(32, 1000)
    result = fd.compute(signal)

    assert result.shape == (32,)
    assert torch.all(result > 1.0)  # FD should be > 1
    assert torch.all(result < 2.0)  # FD should be < 2


def test_higuchi_fd_pure_noise():
    """Test that pure white noise gives FD ≈ 1.5."""
    fd = HiguchiFractalDimension(k_max=10)
    signal = torch.randn(100, 10000)  # Large for stability
    result = fd.compute(signal)

    # White noise should have FD ≈ 1.5
    assert torch.abs(result.mean() - 1.5) < 0.1
```

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(fractals): add graph fractal dimension metric

Implement box-covering algorithm for computing fractal dimension
of graphs/networks. Includes batched computation and GPU support.

Closes #42
```

```
fix(sae): correct sparsity penalty computation

The L1 penalty was not being normalized by batch size,
leading to incorrect scaling.

Fixes #156
```

## Review Process

1. **Automated checks** must pass (tests, linting, type checking)
2. **Code review** by at least one maintainer
3. **Documentation** review if docs changed
4. **Approval** from maintainer before merge

## Areas for Contribution

We especially welcome contributions in:

- **New interpretability methods**: Implement recent research
- **Optimizations**: GPU/memory/speed improvements
- **Documentation**: Tutorials, examples, docstrings
- **Testing**: More comprehensive test coverage
- **Visualizations**: Better plotting and visualization tools
- **Integration examples**: With popular frameworks (HF, Lightning, etc.)

## Questions?

- Open a [Discussion](https://github.com/neuros-ai/neuros-mechint/discussions)
- Join our [Discord](https://discord.gg/neuros)
- Email: team@neuros.ai

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to neuros-mechint!** 🎉
