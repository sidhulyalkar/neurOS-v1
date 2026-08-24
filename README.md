# neurOS

**neurOS is a modular runtime and SDK for building reliable brain-computer interface systems.**

The project is organized around a deliberately small kernel: explicit neural-data contracts, device-independent streaming, timing and synchronization, typed runtime graphs, processing, decoder execution, persistent recording/replay, observability, scientific quality gates, and extension points. Research packages can innovate quickly without owning the runtime architecture.

**ORION** is the complementary neural-intelligence layer for neural tokenization, learned representations, adaptive decoding, personalization, and neural foundation-model research.

> **Status:** active research and engineering platform. APIs are being stabilized and tested, but hardware qualification, clinical validation, and safety certification are separate workstreams. This repository is not a medically validated BCI system.

For the current package-by-package maturity and evidence map, see [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Architecture

```text
hardware / datasets / replay
          |
          v
      Sources
          |
          v
SignalFrame + StreamDescriptor       stable neural data ABI
          |
          v
     RuntimeGraph
 source -> transform -> fusion -> decoder -> sink
          |                    |
          |                    +-> DecoderOutput + embeddings
          |
 timing / queues / monitoring / recording / quality
          |
          +-----------------------------+
          |                             |
          v                             v
   model ecosystem                    ORION
 models / foundation /         neural tokens / representations /
 mechint / source trust        adaptive neural intelligence
```

The architectural boundary is intentional:

- **neurOS** answers: _How should neural systems execute reliably and reproducibly?_
- **ORION** answers: _How should neural systems represent, understand, and adapt to neural activity?_

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the detailed design.

## Repository map

```text
packages/
  neuros-core/          kernel contracts, runtime, processing, timing, recording, quality
  neuros-drivers/       hardware, simulated, and dataset sources
  neuros-models/        task-specific decoders and mechanistic analysis surfaces
  neuros-foundation/    neural foundation-model discovery, adapters, probes, benchmarks
  neuros-sourceweigher/ reliability-aware source/domain weighting and fusion
  neuros-ui/            dashboard/API/visualization surfaces
  neuros-cloud/         optional distributed/cloud integrations
  neuros/               user-facing SDK and CLI
  orion/                ORION tokenization and neural-intelligence interfaces
  neuros-neurofm/       experimental native neural foundation-model research
  neuros-mechint/       causal mechanism and reproducible evidence framework

configs/                versioned runtime, quality, and ORION experiment configs
examples/               supported executable examples; CI is required for promotion here
tutorials/              maintained learning material
notebooks/              transitional BCI notebooks; not automatically a stable API surface
experiments/            research code, notebooks, papers, and exploratory evaluations
docs/                   current architecture, maturity map, and guides
docs/archive/           historical plans, migration reports, and session notes
scripts/                current bootstrap, benchmark, quality, and developer utilities
scripts/archive/        historical migration utilities
tests/                  repository-wide contract, integration, replay, and scientific tests
```

Research may depend on stable runtime contracts. **Stable runtime contracts must not depend on research implementations.**

## Installation

This is a multi-package Python workspace. Package metadata lives under each package's `pyproject.toml`; the removed root `setup.py` is not a supported installation path.

For standard BCI development:

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python scripts/bootstrap.py --profile bci --test-tools
```

Other profiles:

```bash
python scripts/bootstrap.py --profile kernel
python scripts/bootstrap.py --profile orion
python scripts/bootstrap.py --profile research
python scripts/bootstrap.py --profile all
```

## Config-first workflow

The primary execution surface is a versioned YAML configuration resolved through installed plugins.

```bash
neuros doctor --json
neuros plugins --json
neuros devices --json
neuros validate configs/examples/mock_bci.yaml --json
neuros run configs/examples/mock_bci.yaml --duration 2 --json
```

The checked-in mock configuration is deliberately training-free and uses a deterministic threshold decoder so installation/runtime smoke tests never hide model training in the CLI.

Python APIs remain available:

```python
import asyncio
import numpy as np

from neuros.drivers.mock_driver import MockDriver
from neuros.models.simple_classifier import SimpleClassifier
from neuros.pipeline import Pipeline

model = SimpleClassifier()
model.train(np.random.randn(100, 40), np.random.randint(0, 2, 100))

pipeline = Pipeline(
    driver=MockDriver(sampling_rate=250, channels=8),
    model=model,
)

metrics = asyncio.run(pipeline.run(duration=2.0))
print(metrics)
```

Standard `Pipeline` and `MultiModalPipeline` paths compile to the same native `RuntimeGraph` executor used by config execution.

## Canonical neural data and real-time semantics

`SignalFrame` is the neural interchange contract. A frame carries:

- stream and sequence identity,
- sample rate and array payload,
- device, host-receive, and synchronized nanosecond clocks,
- clock domain and synchronization uncertainty,
- composable quality flags,
- immutable metadata/provenance.

Runtime edges have explicit capacity and overflow policy (`block`, `drop_oldest`, `drop_newest`, or `fail`). Runtime snapshots report accepted/dropped counts, queue high-water marks, node failures, and bounded latency summaries including p50/p95/p99.

## Persistent recording and replay

neurOS can persist exact `SignalFrame` semantics in a canonical session archive and replay those recordings through the same runtime graph without constructing the original hardware SDK.

```bash
neuros record configs/examples/mock_bci.yaml \
  --output /tmp/session \
  --session-id demo \
  --duration 10

neuros inspect /tmp/session --verify --json
neuros replay /tmp/session \
  --config configs/examples/mock_bci.yaml \
  --json
```

The canonical archive records sequence/timing/quality metadata, stream descriptors, config hash, Git SHA, package versions, runtime metrics, and per-frame SHA-256 integrity. NWB and Zarr are optional interoperability exports rather than replacements for neurOS' lossless replay semantics.

## Plugins

External packages can register implementations through Python entry points:

```text
neuros.sources
neuros.transforms
neuros.tokenizers
neuros.encoders
neuros.decoders
neuros.sinks
neuros.monitors
```

This keeps the kernel independent of concrete hardware, decoder, storage, and ORION implementations.

## Task-specific decoders and mechanistic analysis

`neuros-models` provides classical baselines plus faithful PyTorch EEGNet, EEG Conformer, Transformer, CNN, LSTM, and multimodal fusion decoders. Maintained deep models expose structured probabilities/logits, pooled embeddings, and stable mechanistic-analysis manifests rather than changing algorithm identity based on which optional framework happens to be installed.

```bash
neuros-models list
neuros-models list --mechint-ready
neuros-models show eeg-conformer
neuros-models doctor
```

An inspectable model can opt into the common `neuros-mechint` causal experiment layer through a manifest-validated adapter. A declared analysis surface identifies where an intervention can be performed; it does not itself establish biological or causal meaning.

## Neural foundation-model interoperability

`neuros-foundation` is the discovery, adapter, representation-probe, and benchmark boundary for external neural foundation-model ecosystems. It separates catalog metadata from locally runnable integrations and fails closed when an upstream implementation is unavailable.

The package includes representation diagnostics such as effective rank, anisotropy, linear CKA, invariance, domain leakage, linear probing, pairwise similarity, sample-efficiency curves, and protocol fingerprints that make subject/session/site/device split semantics explicit.

This package is for fair interoperability and evaluation. It does not turn third-party marketing claims into neurOS benchmark evidence.

## Reliability-aware source transfer

`neuros-sourceweigher` estimates how much a target should trust candidate subjects, sessions, sites, devices, models, or sensor streams. It includes constrained source-mixture estimation, transparent distance and Gibbs-risk baselines, online drift adaptation, representation-space weighting, MMD/Riemannian distribution methods, stability diagnostics, and a native runtime fusion operator.

The core package is local-first and NumPy-only; its HTTP service is optional.

## Mechanistic evidence

`neuros-mechint` is a causal experiment and evidence framework for studying artificial and neural-data model mechanisms. It maintains intervention, faithfulness, held-out evidence, factorial comparison, correspondence, replication, dose-response, and artifact-provenance contracts.

Its release status deliberately distinguishes **software contract readiness** from **empirical evidence completion**. Passing a software gate does not establish a real neural mechanism.

## ORION tokenization

ORION begins at the representation boundary:

```text
explicit spike SignalFrames
   -> NeuroTokenizer
   -> NeuroTokenBatch
   -> NeuralEncoder
   -> RepresentationBatch
   -> AdaptiveDecoder
```

The initial tokenization layer includes exact-event, binned-count, relative-ISI, burst/pause/rebound, synchrony-packet, vector-quantized motif, and assembly tokenizers. A controlled benchmark compares them using separately seeded train/test synthetic sessions with labeled motifs, jitter and unit-dropout perturbations, compression, entropy, motif decoding, robustness, and runtime metrics.

```bash
python scripts/orion/run_tokenizer_benchmark.py \
  configs/orion/tokenization_smoke.yaml \
  --output /tmp/orion-tokenization
```

The benchmark emits `metrics.json`, `comparison_table.csv`, and `tokenizer_cards.md`. Existing NeuroFM research is promoted behind ORION interfaces only when comparative evidence justifies it.

## Quality and evidence

GitHub Actions currently separates several software evidence layers:

1. kernel contracts on Python 3.10, 3.11, and 3.12,
2. installed BCI/config/CLI/runtime smoke execution,
3. exact recording/replay plus NWB/Zarr interoperability,
4. deterministic scientific and latency quality gates,
5. task-model/mechanistic-analysis contracts,
6. foundation-model interoperability regressions and dependency-light examples,
7. SourceWeigher regression tests across Python 3.10, 3.11, and 3.12,
8. ORION contracts and the controlled tokenizer benchmark,
9. mech-int scientific software gates, ecosystem compatibility, and executed CPU tutorials,
10. repository hygiene.

Generic CI quality thresholds are intentionally distinct from hardware-specific qualification. Device/model/application latency and reliability claims require a recorded qualification profile with the relevant hardware, firmware, data, and artifact metadata.

Run the generic evidence gate with:

```bash
python scripts/run_quality_gate.py \
  configs/examples/mock_bci.yaml \
  --thresholds configs/quality/ci.yaml
```

## Research and historical material

Broad research remains in the monorepo, including NeuroFM, mechanistic interpretability, DINO-based neuroscience experiments, and ORION representation work. Research artifacts live under `experiments/` or research packages and are not automatically stable APIs.

Historical cleanup reports, migration plans, implementation notes, and session summaries live under [`docs/archive/`](docs/archive/). They are retained for provenance but are **not** current documentation.

## Roadmap and contributing

- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) is the current capability/maturity map.
- [`ROADMAP.md`](ROADMAP.md) tracks the next qualification and productization sequence.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) defines package boundaries, test expectations, plugin rules, and research-promotion requirements.

## License

MIT. See [`LICENSE`](LICENSE).
