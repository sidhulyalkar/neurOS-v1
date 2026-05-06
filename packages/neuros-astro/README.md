# neuros-astro

**A glial signal processing layer for neural foundation models**

`neuros-astro` extracts astrocyte calcium events and astrocyte functional network states from optical physiology data, then converts them into model-ready tokens for multimodal neural modeling. It is designed to integrate with the broader `neurOS` and `neuroFMx` ecosystem as an `astro` modality.

## Why astrocytes?

Most neural foundation-model pipelines focus on spikes, LFP, calcium traces, behavior, video, and task variables. Astrocytes may provide a slower, spatially structured context signal that helps explain neural state, arousal, plasticity, behavioral context, and cross-session drift.

## Installation

### Basic installation

```bash
pip install -e packages/neuros-astro
```

### With optional dependencies

```bash
# For NWB file support
pip install -e packages/neuros-astro[nwb]

# For DANDI dataset scanning
pip install -e packages/neuros-astro[dandi]

# For imaging utilities
pip install -e packages/neuros-astro[imaging]

# For visualization
pip install -e packages/neuros-astro[viz]

# Install everything
pip install -e packages/neuros-astro[all]
```

## Quick Start

### 1. Generate synthetic astrocyte data

```bash
neuros-astro generate-synthetic --out-dir examples/data --frame-rate 10
```

### 2. Scan a dataset for astrocyte potential

```bash
neuros-astro scan examples/metadata/demo.json --out scan_report.json
```

### 3. Detect astrocyte events from traces

```bash
neuros-astro detect-trace-events examples/data/synthetic_traces.npy \
  --frame-rate 10 \
  --session-id demo \
  --out examples/data/events.parquet
```

### 4. Build astrocyte functional networks

```bash
neuros-astro build-network examples/data/events.parquet \
  --frame-rate 10 \
  --session-id demo \
  --out examples/data/graphs.json
```

### 5. Tokenize events for foundation models

```bash
neuros-astro tokenize-events examples/data/events.parquet \
  --frame-rate 10 \
  --session-id demo \
  --out examples/data/astro_tokens.npz
```

## Main Features

- **Dataset Triage**: Score NWB/DANDI/local datasets for astrocyte reanalysis potential
- **Synthetic Data**: Generate testable astrocyte traces and movies
- **Event Detection**: Detect calcium events from traces (1D) or movies (3D)
- **Network Construction**: Build astrocyte coactivation graphs
- **Tokenization**: Convert events to model-ready tokens (irregular or binned)
- **Export Formats**: Parquet tables, NPZ arrays, neuroFMx manifests
- **Visualization**: Event rasters, network plots, spatial overlays

## Package Structure

```
neuros-astro/
├── neuros_astro/
│   ├── metadata/          # Schemas and dataset scoring
│   ├── io/                # Data loaders and synthetic generators
│   ├── events/            # Event detection algorithms
│   ├── segmentation/      # Spatial segmentation utilities
│   ├── networks/          # Graph construction
│   ├── tokenization/      # Model-ready token generation
│   ├── export/            # Format converters
│   ├── visualization/     # Plotting utilities
│   └── cli/               # Command-line interface
├── tests/                 # Comprehensive test suite
├── examples/              # Example scripts
└── configs/               # Configuration files
```

## Python API

```python
from neuros_astro.io.synthetic import generate_synthetic_astro_traces
from neuros_astro.events.event_detection import detect_events_from_traces
from neuros_astro.networks.functional_connectivity import build_event_coactivation_graph
from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer

# Generate synthetic data
traces, gt_events = generate_synthetic_astro_traces(
    n_regions=10,
    duration_s=60.0,
    frame_rate_hz=10.0,
    seed=42
)

# Detect events
events = detect_events_from_traces(
    traces=traces,
    frame_rate_hz=10.0,
    session_id="demo"
)

# Build networks
graphs = build_event_coactivation_graph(
    events=events,
    session_id="demo",
    frame_rate_hz=10.0
)

# Tokenize for models
tokenizer = AstroEventTokenizer()
tokens = tokenizer.tokenize(events)
```

## Integration with neuroFMx

Add astro tokens as a modality in your neuroFMx config:

```yaml
modalities:
  neural:
    enabled: true
  behavior:
    enabled: true
  astro:
    enabled: true
    token_path: examples/data/astro_tokens.npz
    sampling: irregular
    timestamp_key: timestamps_s
```

## Documentation

- [Whitepaper](../../docs/neuros_astro_whitepaper.md) - Scientific motivation and design
- [Implementation Plan](../../docs/neuros_astro_implementation_plan.md) - Development roadmap
- [Development Plan](../../NEUROS_ASTRO_DEVELOPMENT_PLAN.md) - Structured milestones

## Testing

```bash
# Run all tests
pytest packages/neuros-astro/tests

# Run with coverage
pytest packages/neuros-astro/tests --cov=neuros_astro --cov-report=html
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use `neuros-astro` in your research, please cite:

```bibtex
@software{neuros_astro,
  title={neuros-astro: A glial signal processing layer for neural foundation models},
  author={neurOS Contributors},
  year={2024},
  url={https://github.com/your-org/neurOS-v1}
}
```

## Contact

For questions, issues, or discussions, please use GitHub Issues or Discussions.
