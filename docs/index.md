# neurOS Documentation

neurOS is a modular runtime and SDK for building reproducible brain-computer interface systems. The current architecture centers on explicit neural-data contracts, typed runtime graphs, deterministic recording/replay, real plugin discovery, measurable backpressure/latency, and a separate ORION neural-intelligence layer.

> neurOS is an active research and engineering platform. Generic software tests are not hardware qualification, medical validation, or clinical evidence.

## Start here

- [Installation](getting-started/installation.md)
- [Architecture](ARCHITECTURE.md)
- [API surface](API_REFERENCE.md)
- [Repository roadmap](../ROADMAP.md)
- [Contributing](../CONTRIBUTING.md)

## The execution model

```text
Source plugin
   -> SignalFrame
   -> RuntimeGraph
      source -> transform -> fusion -> decoder -> sink
   -> DecoderOutput
```

Every edge has explicit capacity and overflow policy. Runtime snapshots expose queue loss/high-water metrics and node latency/failure statistics. Live sources and replay sources use the same executor.

## Config-first operation

```bash
neuros doctor --json
neuros plugins --json
neuros devices --json
neuros validate configs/examples/mock_bci.yaml --json
neuros run configs/examples/mock_bci.yaml --duration 2 --json
```

Configuration is versioned and resolved through Python entry-point plugins. Kernel code does not need to import concrete hardware or decoder packages.

## Recording and replay

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

The canonical neurOS archive preserves exact sequence/timing/quality semantics and provenance. NWB and Zarr are optional interoperability exports.

## Scientific quality

The repository separates several evidence layers rather than collapsing them into a single test count:

- kernel contract tests,
- end-to-end BCI/config execution,
- persistent replay and storage interoperability,
- deterministic scientific/latency gates,
- ORION representation benchmarks,
- future hardware qualification profiles.

Current generic quality thresholds are version-controlled under `configs/quality/`. Performance claims should always identify the evidence tier that produced them.

## ORION

ORION is the neural-intelligence layer. Its initial tokenization work uses explicit spike provenance and a controlled benchmark across:

- event tokens,
- binned counts,
- relative-ISI WAIT/SPIKE tokens,
- burst/pause/rebound tokens,
- synchrony packets,
- vector-quantized motifs,
- population assemblies.

The synthetic benchmark uses separately seeded train/test sessions, labeled neural motifs, timing jitter, and unit dropout. Fit-requiring tokenizers never auto-fit during evaluation.

## Repository organization

Current product documentation stays under `docs/`. Historical plans and session notes live under `docs/archive/` and should not be treated as current API or performance documentation.

Research notebooks, papers, and exploratory code live under `experiments/` or research packages. A research artifact becomes a supported example or stable ORION component only after it satisfies the relevant contract and evidence gates.
