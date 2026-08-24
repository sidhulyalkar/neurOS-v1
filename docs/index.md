# neurOS Documentation

neurOS is a modular runtime and SDK for building reproducible brain-computer interface systems. The architecture centers on explicit neural-data contracts, typed runtime graphs, deterministic recording/replay, plugin discovery, measurable timing/backpressure, task-specific decoder contracts, and a separate ORION neural-intelligence layer.

> neurOS is an active research and engineering platform. Generic software tests are not hardware qualification, medical validation, biological evidence, or clinical certification.

## Start here

- [Project status and maturity](PROJECT_STATUS.md)
- [Installation](getting-started/installation.md)
- [Architecture](ARCHITECTURE.md)
- [API surface](API_REFERENCE.md)
- [Repository roadmap](../ROADMAP.md)
- [Contributing](../CONTRIBUTING.md)

## Platform at a glance

```text
hardware / datasets / replay
          |
          v
      Source plugins
          |
          v
       SignalFrame
          |
          v
      RuntimeGraph
 source -> transform -> fusion -> decoder -> sink
          |                       |
          |                       +-> DecoderOutput + embedding
          |
 timing / quality / recording / replay / provenance
          |
          +----------------------------+
          |                            |
          v                            v
 model + transfer ecosystem           ORION
 models | foundation | mechint     tokenization | representation
 source reliability                adaptation | personalization
```

Every runtime edge has explicit capacity and overflow policy. Runtime snapshots expose queue loss/high-water metrics and node latency/failure statistics. Live and replay sources use the same executor.

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

The canonical neurOS archive preserves exact sequence, timing, quality, and provenance semantics. NWB and Zarr are optional interoperability exports.

## Decoder and representation ecosystem

`neuros-models` owns faithful task-specific decoders, logits/embeddings, stable analysis surfaces, and model-side metadata. Its deep decoders use an explicit PyTorch implementation rather than environment-dependent algorithm substitution.

```bash
neuros-models list
neuros-models list --mechint-ready
neuros-models show eeg-conformer
```

`neuros-foundation` owns discovery, adapter capability metadata, representation probes, and split/protocol-aware comparisons across neural foundation-model ecosystems. Catalog presence is not treated as proof that a model is locally runnable or independently reproduced.

`neuros-sourceweigher` owns source/domain reliability estimation and transfer-risk-aware fusion. It can consume representations without making model packages depend on its algorithms.

`neuros-mechint` owns causal intervention, faithfulness, held-out evidence, replication, and evidence-artifact contracts. Its v1 release explicitly separates software readiness from empirical evidence completion.

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

## Quality and evidence

The repository separates evidence rather than collapsing it into one green badge:

- kernel contracts across supported Python versions,
- end-to-end BCI/config execution,
- persistent replay and storage interoperability,
- deterministic scientific/latency gates,
- model/mechanistic-analysis contracts,
- foundation-model interoperability regressions,
- SourceWeigher regressions,
- ORION tokenization contracts and synthetic benchmark evidence,
- mech-int scientific software gates and executed tutorials.

Hardware qualification, real-dataset evidence, closed-loop qualification, and clinical evidence remain stronger and separate tiers.

## Repository organization

Current product documentation stays under `docs/`. Historical plans and session notes live under `docs/archive/` and should not be treated as current API or performance documentation.

Research notebooks, papers, and exploratory code live under `experiments/` or research packages. A research artifact becomes a supported example or stable ORION component only after it satisfies the relevant contract and evidence gates.
