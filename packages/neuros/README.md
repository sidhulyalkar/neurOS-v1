# neurOS

`neuros` is the user-facing SDK and CLI composition package for the neurOS brain-computer interface runtime.

> **Project status:** active research and engineering platform. Software contract tests do not imply hardware qualification, clinical validation, or safety certification. See the repository's `docs/PROJECT_STATUS.md` for the current package-by-package maturity map.

## Installation

```bash
pip install neuros
```

Optional profiles declared by this distribution include:

```bash
pip install "neuros[bci]"        # EEG driver dependencies, PyTorch models, dashboard
pip install "neuros[recording]"  # NWB/Zarr recording interoperability
pip install "neuros[research]"   # model/foundation/SourceWeigher research stack
pip install "neuros[orion]"      # ORION contracts/tokenization package
pip install "neuros[deployment]" # optional UI/cloud integrations
pip install "neuros[all]"
```

For monorepo development, prefer the version-controlled workspace profiles in `scripts/bootstrap.py` rather than treating the repository root as one Python package.

## Start with the CLI

```bash
neuros doctor --json
neuros plugins --json
neuros devices --json
neuros validate configs/examples/mock_bci.yaml --json
neuros run configs/examples/mock_bci.yaml --duration 2 --json
```

The checked-in mock configuration is deterministic and training-free, making it suitable for installation/runtime smoke testing.

## Record and replay

```bash
neuros record configs/examples/mock_bci.yaml \
  --output /tmp/neuros-session \
  --session-id demo \
  --duration 2

neuros inspect /tmp/neuros-session --verify --json
neuros replay /tmp/neuros-session \
  --config configs/examples/mock_bci.yaml \
  --json
```

The canonical neurOS archive preserves sequence, timing, quality, stream metadata, provenance, and integrity semantics. NWB and Zarr are interoperability exports rather than replacements for the lossless replay boundary.

## Python API

The compatibility/convenience pipeline compiles standard paths to the native runtime graph:

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

For new architecture work, prefer explicit neurOS contracts, `RuntimeGraph`, plugins, and persistent replay over adding another orchestration abstraction.

## Package ecosystem

The meta-package composes several independently versioned distributions:

- `neuros-core`: runtime, contracts, config, processing, recording/replay, quality;
- `neuros-drivers`: hardware/simulated/dataset sources;
- `neuros-models`: task-specific decoders and model-side interpretability manifests;
- `neuros-foundation`: foundation-model catalog/adapters/representation probes;
- `neuros-sourceweigher`: source/domain reliability and transfer-aware fusion;
- `neuros-mechint`: causal mechanism/evidence tooling;
- `neuros-orion`: ORION tokenization and neural-intelligence contracts;
- `neuros-ui` and `neuros-cloud`: optional integration surfaces with separate maturity boundaries.

The architectural rule is simple: research and product integrations may depend on stable runtime contracts, but the runtime kernel must not depend on research implementations.

## Documentation

Current documentation lives in the repository:

- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `ROADMAP.md`
- `CONTRIBUTING.md`

Repository: https://github.com/sidhulyalkar/neurOS-v1

## License

MIT License.
