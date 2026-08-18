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
neuros doctor --json
neuros validate configs/examples/mock_bci.yaml --json
```

## Installation profiles

The bootstrap script provides explicit profiles instead of one enormous dependency bundle:

```bash
# Kernel contracts/runtime only
python scripts/bootstrap.py --profile kernel

# Standard BCI runtime + sources + decoders + SDK
python scripts/bootstrap.py --profile bci

# Kernel + ORION representation/tokenization layer
python scripts/bootstrap.py --profile orion

# Foundation-model and interpretability research stack
python scripts/bootstrap.py --profile research

# Every workspace package
python scripts/bootstrap.py --profile all
```

Use `--test-tools` to add shared pytest tooling, or `--dry-run` to inspect the commands before installation.

## Installing individual packages

Each distribution can also be installed independently from its package directory:

```bash
pip install -e packages/neuros-core
pip install -e packages/neuros-drivers
pip install -e packages/neuros-models
pip install -e packages/neuros
```

Because this repository contains unpublished workspace dependencies and active development packages, installing editable local packages is the source-of-truth contributor workflow until a coordinated PyPI release is published and validated.

## Optional hardware and storage dependencies

Hardware, deep-learning frameworks, and interoperability formats are intentionally optional.

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

### Persistent NWB/Zarr interoperability

The canonical neurOS archive has no additional storage dependency. To add NWB and Zarr exports:

```bash
pip install -e "packages/neuros-core[recording]"
```

### PyTorch decoders

```bash
pip install -e "packages/neuros-models[pytorch]"
```

This keeps a mock/synthetic runtime lightweight and avoids forcing hardware SDKs, cloud libraries, storage backends, or foundation-model dependencies onto every installation.

## ORION

Install ORION alongside the kernel with:

```bash
python scripts/bootstrap.py --profile orion --test-tools
pytest -q tests/test_orion_contracts.py tests/test_orion_tokenization.py
```

ORION provides the stable neural tokenization/representation contracts plus the controlled initial tokenizer benchmark. Experimental NeuroFM components remain research candidates until they satisfy ORION contracts and comparative evidence gates.

## Workspace tooling

The root `pyproject.toml` defines repository-wide tooling and workspace membership. It deliberately does not contain `[project]` package metadata.

The historical root packaging path has been removed. **Do not install the repository root as though it were a single Python distribution.** Use `scripts/bootstrap.py` or install the individual package directories shown above.

## GPU environments

GPU stacks are intentionally not pinned by the kernel. Install the appropriate PyTorch/CUDA/ROCm build for the target machine first, then install the neurOS package that requires PyTorch. This avoids silently replacing a working accelerator stack.

## Common validation commands

```bash
# Kernel/runtime contracts
pytest -q \
  tests/test_kernel_contracts.py \
  tests/test_runtime_queues.py \
  tests/test_runtime_executor.py \
  tests/test_clock_sync.py \
  tests/test_replay.py

# BCI/config/CLI smoke path
neuros run configs/examples/mock_bci.yaml --duration 0.1 --json

# Persistent recording/replay
neuros record configs/examples/mock_bci.yaml \
  --output /tmp/neuros-session \
  --session-id smoke \
  --duration 0.1
neuros inspect /tmp/neuros-session --verify --json
neuros replay /tmp/neuros-session \
  --config configs/examples/mock_bci.yaml \
  --json

# Scientific/runtime quality gate
python scripts/run_quality_gate.py \
  configs/examples/mock_bci.yaml \
  --thresholds configs/quality/ci.yaml

# ORION tokenization benchmark
python scripts/orion/run_tokenizer_benchmark.py \
  configs/orion/tokenization_smoke.yaml \
  --output /tmp/orion-tokenization
```

GitHub Actions runs the corresponding kernel, BCI, recording, scientific-quality, ORION, and repository-hygiene jobs automatically.

## Troubleshooting

### A hardware library will not install

Install the minimal BCI profile first and add only the relevant driver extra. Hardware SDK failures should not prevent the mock/replay runtime from working.

### Imports resolve differently between packages

Ensure all workspace packages were installed into the same virtual environment. The `neuros` namespace is intentionally distributed across multiple installations.

### A published package does not match the repository

The repository may be ahead of a coordinated package release during active refactors. Use the workspace profile for development until published versions are explicitly tagged and validated.

### Need deterministic debugging without hardware

Use a canonical neurOS session archive and `ArchiveReplaySource`, or the `neuros replay` command. Replay uses the same `SignalFrame` and RuntimeGraph path as live sources.
