# neurOS NeuroFM

`neuros-neurofm` is the **alpha research package** for native neural foundation-model experiments inside neurOS.

It contains exploratory architectures and utilities for neural population modeling, including state-space backbones, population-level aggregation, multimodal fusion, adaptation, continual-learning, evaluation, and mechanistic-analysis integration.

> **Maturity boundary:** this package is research software. It is not a promoted ORION implementation, a hardware-qualified decoder, or a source of production/clinical claims. Model quality, latency, transfer, and mechanism claims require an immutable model artifact plus a leakage-controlled evaluation manifest on named data and hardware where applicable.

See [`../../docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md) for the repository-wide maturity map.

## Why this package exists

neurOS deliberately separates three concerns:

- `neuros-foundation` catalogs and evaluates external neural foundation-model ecosystems behind common protocols.
- `neuros-neurofm` is the sandbox for **native** neurOS foundation-model R&D.
- `neuros-orion` is the representation/adaptation plane that receives research components only after comparative evidence justifies promotion.

That separation lets the project explore ambitious architectures without making experimental code part of the deployment contract by accident.

## Current research surface

The source tree includes work on:

- neural population and state-space models;
- modality-specific tokenization and dataset adapters;
- multimodal fusion;
- continual/adaptive learning;
- augmentation and diffusion experiments;
- evaluation utilities;
- inference and integration helpers;
- mechanistic-interpretability integration.

The presence of a module means the implementation is available for research. It does **not** imply that the method has beaten a baseline, generalized across subjects/sessions/devices, or passed a hardware qualification profile.

## Installation from this monorepo

From the repository root:

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python -m pip install -e "packages/neuros-neurofm[dev]"
```

Optional research profiles are declared in `pyproject.toml`:

```bash
# state-space / Mamba experiments
python -m pip install -e "packages/neuros-neurofm[mamba]"

# training stack
python -m pip install -e "packages/neuros-neurofm[training]"

# DANDI / NWB / Allen / IBL dataset tooling
python -m pip install -e "packages/neuros-neurofm[datasets]"

# mechanistic-analysis integration
python -m pip install -e "packages/neuros-neurofm[mechint]"
```

Install only the extras required by the experiment. GPU/CUDA compatibility depends on the selected PyTorch and optional backend versions.

## Evidence requirements

A NeuroFM experiment should not be promoted based on a single training curve or favorable held-out fold. At minimum, a comparative result should record:

1. dataset identity and version;
2. subject/session/site/device split unit;
3. preprocessing and tokenization fingerprint;
4. architecture and parameter-count identity;
5. training/calibration budget;
6. random seeds and repeated-run uncertainty;
7. task utility metrics;
8. representation diagnostics such as geometry and domain leakage;
9. robustness to missing/noisy channels or units where relevant;
10. artifact/checkpoint hashes and exact code/package versions.

For transfer/adaptation studies, fit, adaptation, calibration, and evaluation partitions must be disjoint according to the declared protocol.

For mechanistic studies, attribution alone is insufficient. Intervention, held-out faithfulness, replication, and cross-session/subject stability should be reported separately.

## Recommended real-data progression

The current roadmap is to test native NeuroFM ideas through shared neurOS benchmark protocols rather than maintain a private scoreboard inside this package:

```text
synthetic falsification
  -> one real multi-session dataset
  -> subject/session-disjoint comparison
  -> cross-dataset/device transfer
  -> few-shot adaptation/calibration-cost study
  -> mechanism stability study
  -> promoted ORION component only if evidence survives
```

Useful external sources include MOABB/MNE for EEG benchmarking and DANDI/FALCON/NLB-style NWB datasets for population/spiking studies. The exact supported benchmark bridge should live in maintained neurOS evaluation code, not be implied by this README.

## Development

From the repository root:

```bash
python -m pip install -e "packages/neuros-neurofm[dev]"
pytest -q packages/neuros-neurofm/tests
```

Dataset- or GPU-dependent tests should remain explicitly marked so the default software contract suite does not silently depend on large downloads or specialized hardware.

## Relationship to ORION

ORION should consume a NeuroFM idea only when the experiment answers a practical question such as:

- Does this representation reduce calibration cost on a new session or subject?
- Does it improve robustness to channel/unit dropout or device drift?
- Does it preserve task information while reducing subject/session leakage?
- Does a proposed mechanism remain causally important under held-out interventions?
- Does the benefit survive matched downstream capacity and matched training budgets?

Until then, the implementation remains research material by design.

## Documentation

- [Project maturity map](../../docs/PROJECT_STATUS.md)
- [Architecture](../../docs/ARCHITECTURE.md)
- [Roadmap](../../ROADMAP.md)
- [Contributing](../../CONTRIBUTING.md)
- [Package documentation](docs/)

## License

MIT. See the repository [`LICENSE`](../../LICENSE).
