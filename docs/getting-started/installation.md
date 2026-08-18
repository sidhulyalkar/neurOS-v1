# Installation

neurOS is a multi-package Python workspace. Package metadata lives under `packages/*/pyproject.toml`; the repository root is for workspace/tooling configuration and is not itself an installable distribution.

## Supported Python

The neurOS kernel, SDK, drivers, and model packages target Python 3.10-3.12. Some research packages currently retain Python 3.9 compatibility while they are migrated.

## Recommended development install

Clone the repository and use the workspace bootstrap script:

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python scripts/bootstrap.py --profile bci --test-tools
```

The BCI profile installs, in dependency order:

```text
neuros-core
neuros-drivers
neuros-models
neuros
```

Then verify the standard path:

```bash
pytest -q tests/test_pipeline.py tests/test_kernel_contracts.py
neuros --help
```

## Installation profiles

The bootstrap script provides explicit profiles instead of one enormous dependency bundle:

```bash
# Kernel contracts/runtime only
python scripts/bootstrap.py --profile kernel

# Standard BCI runtime + sources + decoders + SDK
python scripts/bootstrap.py --profile bci

# Kernel + ORION contract layer
python scripts/bootstrap.py --profile orion

# Foundation-model and interpretability research stack
python scripts/bootstrap.py --profile research

# Every workspace package
python scripts/bootstrap.py --profile all
```

Use `--test-tools` to add shared pytest tooling, or `--dry-run` to inspect the commands before installation.

## Installing individual packages

Each distribution can also be installed independently:

```bash
pip install -e packages/neuros-core
pip install -e packages/neuros-drivers
pip install -e packages/neuros-models
pip install -e packages/neuros
```

Because this repository contains unpublished workspace dependencies and active development packages, installing editable local packages is the source-of-truth contributor workflow until a coordinated PyPI release is published and validated.

## Optional hardware dependencies

Hardware and heavyweight formats are intentionally optional.

### EEG / BrainFlow / LSL

```bash
pip install -e "packages/neuros-drivers[eeg]"
```

### Video

```bash
pip install -e "packages/neuros-drivers[video]"
```

### Audio

```bash
pip install -e "packages/neuros-drivers[audio]"
```

### NWB

```bash
pip install -e "packages/neuros-drivers[nwb]"
```

### PyTorch decoders

```bash
pip install -e "packages/neuros-models[pytorch]"
```

This keeps a mock/synthetic runtime lightweight and avoids forcing hardware SDKs, cloud libraries, or foundation-model dependencies onto every installation.

## ORION

Install the ORION contracts alongside the kernel with:

```bash
python scripts/bootstrap.py --profile orion --test-tools
pytest -q tests/test_orion_contracts.py
```

ORION currently defines stable neural tokenization, representation, adaptive-decoder, and adaptation interfaces. Experimental NeuroFM/tokenizer implementations remain separate research packages until they are validated behind those contracts.

## Workspace tooling

The root `pyproject.toml` defines repository-wide tooling and the workspace membership. It deliberately does not contain `[project]` package metadata.

The old root `setup.py` has been removed. Do not use:

```bash
pip install -e .
```

from the repository root.

## GPU environments

GPU stacks are intentionally not pinned by the kernel. Install the appropriate PyTorch/CUDA/ROCm build for the target machine first, then install the neurOS package that requires PyTorch. This avoids silently replacing a working accelerator stack.

## Common validation commands

```bash
# Kernel contracts
pytest -q \
  tests/test_kernel_contracts.py \
  tests/test_runtime_queues.py \
  tests/test_clock_sync.py \
  tests/test_replay.py

# BCI smoke path
pytest -q tests/test_pipeline.py

# ORION contracts
pytest -q tests/test_orion_contracts.py
```

GitHub Actions runs the corresponding kernel, BCI, and ORION jobs automatically.

## Troubleshooting

### A hardware library will not install

Install the minimal BCI profile first and add only the relevant driver extra. Hardware SDK failures should not prevent the mock/replay runtime from working.

### Imports resolve differently between packages

Ensure all workspace packages were installed into the same virtual environment. The `neuros` namespace is intentionally distributed across multiple installations.

### `pip install neuros` does not match the repository

The repository is ahead of a coordinated package release during this refactor. Use the editable workspace profile for development until published versions are explicitly tagged and validated.

### Need deterministic debugging without hardware

Use `neuros.recording.ReplaySource` with recorded `SignalFrame` objects. Replay uses the same canonical frame contract as live sources.
