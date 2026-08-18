# neurOS

**neurOS is a modular runtime and SDK for building reliable brain-computer interface systems.**

The project is being organized around a deliberately small kernel: explicit neural data contracts, device-independent streaming, timing and synchronization, processing, decoder execution, recording/replay, observability, and extension points. Research packages can innovate quickly without owning the runtime architecture.

ORION is the complementary neural-intelligence layer for tokenization, learned representations, adaptive decoding, personalization, and future neural foundation-model research.

> **Status:** active research and engineering platform. The core APIs are being stabilized; hardware validation and production qualification remain ongoing. The repository should not be interpreted as a medically validated or production-certified BCI system.

## Architecture

```text
hardware / datasets
       |
       v
neuros-drivers
       |
       v
SignalFrame + StreamDescriptor        <- stable neural data contracts
       |
       v
neurOS runtime
  timing | processing | queues | monitoring | recording/replay
       |
       +--------------------------+
       |                          |
       v                          v
neuros-models                    ORION
conventional decoders       tokenization / representations /
                            adaptive neural intelligence
       |                          |
       +------------+-------------+
                    v
               applications
```

The important boundary is simple:

- **neurOS** answers: _How should neural systems execute reliably?_
- **ORION** answers: _How should neural systems represent, understand, and adapt to neural activity?_

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the detailed design.

## Repository layout

```text
packages/
  neuros-core/          kernel contracts, runtime, processing, timing, replay
  neuros-drivers/       hardware and dataset sources
  neuros-models/        task-specific decoders and model adapters
  neuros-foundation/    adapters for external neural foundation models
  neuros-ui/            dashboard/API/visualization surfaces
  neuros-cloud/         optional distributed/cloud integrations
  neuros/               user-facing SDK and CLI
  orion/                ORION neural-intelligence contracts
  neuros-neurofm/       experimental neural foundation-model research
  neuros-mechint/       mechanistic-interpretability research toolkit
  neuros-sourceweigher/ focused service component

examples/ and notebooks/  learning and research material
docs/                    architecture, guides, and research plans
scripts/                 development/bootstrap/benchmark utilities
tests/                   repository-wide contract and integration tests
```

## Installation

The repository is a multi-package workspace. **The old root `setup.py` installation path is intentionally gone.** Package metadata lives with each package under `packages/*/pyproject.toml`.

For the standard BCI development profile:

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python scripts/bootstrap.py --profile bci --test-tools
```

Other profiles:

```bash
python scripts/bootstrap.py --profile kernel     # runtime only
python scripts/bootstrap.py --profile orion      # kernel + ORION contracts
python scripts/bootstrap.py --profile research   # research stack
python scripts/bootstrap.py --profile all        # full workspace
```

The root `pyproject.toml` also declares the workspace for tools that support Python monorepos.

## First pipeline

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

The runtime report includes throughput and latency as well as queue-drop/high-water telemetry so overload is visible rather than silently ignored.

## Canonical neural data

Hardware-specific code can still expose legacy streaming tuples for compatibility, but the long-term interchange format is `SignalFrame`:

```python
from neuros.contracts import SignalFrame, StreamDescriptor
```

A frame carries stream identity, sequence number, sampling rate, explicit device/host/synchronized clocks, quality flags, metadata, and the neural array itself. This is the contract that recording, synchronization, ORION, and future runtime operators build on.

## Configuration and plugins

neurOS now has a versioned configuration schema and a real plugin registry. External packages can expose entry points in categories such as:

```text
neuros.sources
neuros.transforms
neuros.tokenizers
neuros.encoders
neuros.decoders
neuros.sinks
neuros.monitors
```

Configuration is represented by `neuros.config.PipelineConfig`, with explicit stream IDs and runtime backpressure policy.

## ORION

ORION begins at the neural representation boundary:

```text
SignalFrame
   -> NeuroTokenizer
   -> NeuroTokenBatch
   -> NeuralEncoder
   -> RepresentationBatch
   -> AdaptiveDecoder
```

Existing NeuroFM and neurotokenization experiments should migrate behind these interfaces only when their scientific value is demonstrated. The ORION package is intentionally a stable contract surface, not a claim that a universal neural foundation model is already solved.

## Timing and replay

The kernel includes:

- bounded device-to-host clock-drift estimation,
- explicit synchronization uncertainty,
- canonical synchronized timestamps,
- deterministic `SignalFrame` recording/replay primitives.

These capabilities are intended to make multimodal experiments reproducible and to support hardware-independent regression testing.

## Quality gates

GitHub Actions validates three layers:

1. kernel contracts across Python 3.10-3.12,
2. an end-to-end mock BCI pipeline,
3. ORION representation contracts.

Run locally with:

```bash
pytest tests/
```

Research packages maintain additional package-specific test suites.

## Research packages

The repository contains broad experimental work, including NeuroFM, mechanistic interpretability, DINO-based neuroscience experiments, and neurotokenization plans. These are valuable research surfaces but are not automatically part of the neurOS kernel API.

The architectural rule going forward is:

> Research may depend on stable runtime contracts. Stable runtime contracts should not depend on research implementations.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). New foundational functionality should prefer stable contracts and plugin interfaces over new cross-package imports.

## License

MIT. See [`LICENSE`](LICENSE).
