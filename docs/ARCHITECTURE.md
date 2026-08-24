# neurOS + ORION Architecture

This document is the architectural source of truth for the maintained neurOS runtime and its boundaries with ORION and the research/model ecosystem.

## 1. Design objective

neurOS should provide the smallest stable set of abstractions needed to construct reliable, replayable, measurable brain-computer interface systems across changing hardware, signal modalities, models, and research ideas.

The kernel is intentionally narrower than the repository. Research packages may move quickly. Runtime contracts should move slowly and accumulate evidence before they are broadened.

The governing separation is:

- **neurOS runtime plane:** reliable acquisition, timing, execution, recording, replay, configuration, quality, and observability;
- **model and evidence plane:** task decoders, representation interoperability, source reliability, and causal/mechanistic analysis;
- **ORION intelligence plane:** neural tokenization, learned representations, adaptation, and personalization;
- **application/safety plane:** task behavior and, eventually, constrained closed-loop actions.

## 2. System boundary

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                              Applications                                  │
│ communication | prosthetics | research | adaptive UX | experiments        │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
                            user-facing SDK
                                   │
                  ┌────────────────┴─────────────────┐
                  │                                  │
                  ▼                                  ▼
┌────────────────────────────┐        ┌────────────────────────────────────┐
│       neurOS runtime       │        │               ORION                │
│                            │        │                                    │
│ acquisition                │◀──────▶│ tokenization                       │
│ data contracts             │        │ learned representations            │
│ synchronization            │        │ adaptive decoding                  │
│ graph execution            │        │ personalization                    │
│ recording / replay         │        │ neural-intelligence research       │
│ quality / observability    │        │                                    │
└──────────────┬─────────────┘        └────────────────┬───────────────────┘
               │                                       │
               │              ┌────────────────────────┘
               ▼              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      Model + evidence ecosystem                            │
│ task decoders | foundation adapters | source trust | mechanistic evidence │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
                                   ▼
                        stable neural contracts
```

neurOS owns reliable execution. ORION owns neural intelligence. Model/evidence packages may consume the same contracts without becoming hidden dependencies of the runtime kernel.

## 3. Package responsibilities

### `neuros-core`

The kernel. It contains:

- canonical neural data contracts;
- runtime graph and executor semantics;
- queue/backpressure and lifecycle behavior;
- processing primitives;
- clock synchronization;
- recording/replay primitives and persistent archives;
- versioned configuration schemas;
- plugin discovery;
- generic scientific/runtime quality infrastructure;
- compatibility orchestration APIs.

**Dependency rule:** `neuros-core` must not import concrete driver, task-model, UI/cloud, ORION implementation, NeuroFM, or mechanistic-interpretability packages.

### `neuros-drivers`

Concrete hardware, simulated, and dataset sources. Drivers may depend on `neuros-core`; the reverse dependency is forbidden.

Heavy or hardware-specific dependencies belong in extras or external plugin distributions. A driver being importable does not mean a named hardware/firmware combination is qualified.

### `neuros-models`

Task-specific decoders and model-side analysis contracts.

Maintained deep decoders use an explicit PyTorch backend and expose:

- `DecoderOutput` compatible inference;
- probabilities/logits when genuinely supported;
- pooled embeddings through `encode(...)`;
- stable model identity and metadata;
- an `InterpretabilityManifest` describing named analysis surfaces;
- an optional manifest-validated bridge into `neuros-mechint`.

A model name must describe the algorithm actually executed. Missing optional dependencies must fail clearly rather than silently substitute another model family.

### `neuros-foundation`

The interoperability/evaluation boundary for external neural foundation-model ecosystems.

It owns:

- model/capability catalog metadata;
- locally runnable adapters where integrations are verified;
- fail-closed availability semantics;
- representation probes;
- split/protocol-aware benchmark metadata and fingerprints;
- bridges from frozen encoders into neurOS decoder contracts.

Catalog presence is not equivalent to local execution, reproduced performance, or endorsement of an upstream claim.

### `neuros-sourceweigher`

The source/domain reliability layer. It owns algorithms for estimating which subjects, sessions, sites, devices, models, or streams a target should trust and by how much.

It remains dependency-light and can integrate with neurOS runtime fusion or consume embeddings from model/foundation layers without forcing a reverse dependency into those packages.

### `neuros-mechint`

The causal mechanism/evidence framework. It owns intervention, faithfulness, held-out evidence, comparison, correspondence, replication, dose-response, provenance, and evidence-artifact contracts.

Its software release status deliberately separates **software contract readiness** from **empirical evidence completion**. Adapter availability or passing software gates never establishes biological homology or a real neural mechanism.

### `neuros-neurofm`

Experimental native neural foundation-model research. NeuroFM work is a candidate source of ORION implementations, not an automatic dependency or promoted representation layer.

### `neuros`

The user-facing meta-package and CLI. This composition layer may depend on core, drivers, and models and may expose optional profiles for research, recording, deployment, and ORION.

### `neuros-ui` and `neuros-cloud`

Optional presentation, API, distributed, export, and observability integrations. They must consume stable runtime/config/event contracts instead of becoming alternate orchestration systems.

Until dedicated release/qualification lanes exist for these packages, their package versions should not be interpreted as equal evidence maturity with the kernel.

### `orion` / distribution `neuros-orion`

The stable ORION contract and current tokenization surface. ORION defines token, representation, encoder, adaptive-decoder, and adaptation-proposal interfaces. Research implementations move behind these contracts only after comparative evidence justifies promotion.

## 4. Canonical neural data

The primary runtime data unit is `SignalFrame`.

```text
SignalFrame
  stream_id
  sequence_id
  data
  sample_rate_hz
  device_time_ns          optional
  host_receive_time_ns
  synchronized_time_ns    optional
  clock_domain
  quality flags
  metadata
```

A separate `StreamDescriptor` describes relatively static stream metadata such as modality, channels, units, device identity, sampling rate, and clock domain.

### Timing rule

A single floating-point timestamp is insufficient as a long-term BCI interchange contract. The frame explicitly represents different clocks so latency, synchronization, and provenance are not conflated.

`SignalFrame.timestamp_ns` resolves in this order:

1. synchronized time;
2. device time;
3. host receive time.

Legacy tuple-producing drivers can be adapted into frames at the source boundary while they migrate.

## 5. Clock synchronization

`ClockSynchronizer` estimates an affine map:

```text
host_time = scale * device_time + offset
```

from a bounded window of timestamp pairs. It reports:

- offset;
- scale;
- drift in parts per million;
- residual uncertainty;
- observation count.

The synchronizer can annotate a `SignalFrame` with synchronized time and `CLOCK_UNCERTAIN` quality state when residual timing error exceeds the configured threshold.

This software estimator is an alignment primitive, not a substitute for hardware synchronization when tighter guarantees are required.

## 6. Decoder semantics

The canonical task-model response is `DecoderOutput`:

```text
prediction
confidence        optional
uncertainty       optional
probabilities     optional
logits            optional
embedding         optional
model_id/version
inference_time_ns
metadata
```

**Critical rule:** absence of calibrated probability is represented as `confidence=None`. It must never be converted to artificial certainty such as `1.0`.

Task-model embeddings create a shared but explicit seam for representation probes, source/domain weighting, and mechanistic analyses. Embedding availability does not imply that the representation is invariant, causal, or transferable until those properties are measured.

## 7. Native runtime graph

`RuntimeGraph` is the maintained execution representation:

```text
SOURCE -> TRANSFORM -> ... -> FUSION -> DECODER -> SINK
                  \                    /
                   ------ MONITOR -----
```

`RuntimeNode` records node identity, kind, operator, execution policy, optional latency budget, and metadata. `RuntimeEdge` records source/destination, bounded capacity, and overflow policy.

`RuntimeExecutor` natively executes this graph for finite replay and live streaming sources. It provides supervised failure propagation, draining/cancellation semantics, execution classes, queue telemetry, and bounded per-node latency statistics.

`Pipeline` and `MultiModalPipeline` are compatibility/convenience facades. Standard paths compile to `RuntimeGraph` and execute through `RuntimeExecutor`. Historical custom processing-agent classes remain an explicit migration exception rather than a second preferred runtime architecture.

## 8. Backpressure

Every bounded runtime edge has explicit overload behavior:

- `BLOCK`
- `DROP_OLDEST`
- `DROP_NEWEST`
- `FAIL`

`DROP_OLDEST` is useful for many real-time paths because stale neural data can be less useful than fresh data, but applications should choose policy deliberately.

Queue telemetry includes accepted items, dropped items, and high-water mark. Dropped samples are therefore measurable system behavior, not a hidden implementation detail.

## 9. Lifecycle and failures

Runtime lifecycle states are explicit and failures are supervised rather than allowed to disappear inside worker tasks.

The maintained execution pattern is:

```python
await executor.start()
async for output in executor.outputs():
    ...
await executor.stop()
```

Finite and timed convenience execution use the same engine. Runtime snapshots retain failures, per-node activity/latency, and per-edge queue evidence.

## 10. Persistent recording and replay

neurOS has a canonical dependency-light session archive for exact `SignalFrame` persistence.

The archive preserves:

- sequence identity;
- device, host, and synchronized timestamps;
- clock domain and quality state;
- stream descriptors;
- frame metadata/provenance;
- per-frame payload integrity hashes;
- config hash;
- Git/package/environment provenance;
- model-artifact references and runtime metrics where supplied.

`ArchiveReplaySource` exposes archived streams through the same source contract used by live hardware. `RecordingSource` can wrap a source so received frames are persisted before forwarding.

NWB and Zarr are maintained optional interoperability exports. They do not replace neurOS' canonical lossless replay semantics.

Replay is a first-class requirement because it supports deterministic regression, hardware-independent debugging, latency experiments, ORION/model evaluation, fault injection, and reproducibility.

## 11. Configuration

The versioned `PipelineConfig` schema describes streams, source plugins, transforms, decoder, sinks, monitors, queue capacity, overflow policy, and metadata.

`resolve_config(...)` compiles the validated configuration into a `RuntimeGraph`. Source overrides allow archived sessions to replace live hardware plugins without instantiating the original device SDK.

Configuration schema changes require explicit migration. Old experiment files must not silently acquire new semantics.

## 12. Plugin architecture

neurOS discovers extension packages through Python entry points:

```text
neuros.sources
neuros.transforms
neuros.tokenizers
neuros.encoders
neuros.decoders
neuros.sinks
neuros.monitors
```

This allows hardware/model integrations to evolve independently from the kernel. External plugins should be able to satisfy these contracts without modifying `neuros-core`.

## 13. ORION boundary

ORION starts where provenance-rich neural data becomes a machine-native neural representation:

```text
SignalFrame(s)
    |
    v
NeuroTokenizer
    |
    v
NeuroTokenBatch
    |
    v
NeuralEncoder
    |
    v
RepresentationBatch
    |
    v
AdaptiveDecoder
    |
    v
DecoderOutput
```

Online change is represented by `AdaptationProposal` containing a reason, requested changes, supporting evidence, and whether approval is required. Adaptation should be observable and auditable rather than an invisible side effect.

## 14. ORION tokenization evidence

The maintained initial ORION tokenizer layer includes:

- exact event tokens;
- binned counts;
- relative-ISI WAIT/SPIKE tokens;
- burst/pause/rebound tokens;
- synchrony packets;
- vector-quantized motifs;
- population assemblies.

The controlled synthetic benchmark uses known motifs, separately seeded train/test sessions, timing jitter, unit dropout, compression, entropy, motif decoding, robustness, and runtime metrics. Fit-requiring tokenizers are fit on the training side rather than auto-fitting during evaluation.

This is **scientific synthetic evidence**, not proof that one tokenizer is superior on real human BCI data. Real-data promotion requires deployment-unit-disjoint evaluation.

## 15. Model and evidence interoperability

The maintained package direction allows one model representation to be studied through several orthogonal lenses:

```text
neuros-models / external verified encoder
                |
                +-> DecoderOutput / embedding
                |
                +-> neuros-foundation representation probes
                |
                +-> neuros-sourceweigher transfer reliability
                |
                +-> neuros-mechint causal interventions
```

These tools answer different questions and must not be collapsed into a single score:

- representation similarity is not task utility;
- domain similarity is not signal quality;
- attention or attribution is not mechanism;
- a causal effect in one model/session is not cross-subject stability;
- synthetic robustness is not hardware qualification.

## 16. Dependency direction

Allowed:

```text
neuros-drivers --------> neuros-core
neuros-models ---------> neuros-core
neuros-ui -------------> neuros-core
neuros-cloud ----------> neuros-core
orion -----------------> neuros-core
neuros ----------------> core + drivers + models
foundation ------------> core + model contracts
sourceweigher ---------> dependency-light / optional runtime integration
mechint ---------------> core + optional model/ORION/research adapters
research --------------> stable packages
```

Forbidden without an explicit architectural proposal:

```text
neuros-core -> concrete drivers
neuros-core -> concrete task models
neuros-core -> UI/cloud
neuros-core -> NeuroFM/mechint/ORION implementations
```

Convenient imports are not sufficient justification for reversing dependency direction.

## 17. Quality and evidence gates

Current repository CI separates multiple software evidence surfaces:

1. repository hygiene;
2. kernel contracts across Python 3.10, 3.11, and 3.12;
3. installed BCI/config/CLI/runtime smoke execution;
4. scientific and latency quality gates;
5. recording/replay plus NWB/Zarr interoperability;
6. task-model/mechanistic-analysis contracts;
7. foundation-model interoperability regressions;
8. SourceWeigher regressions across supported Python versions;
9. ORION contracts and controlled tokenizer benchmark;
10. dedicated mech-int software/evidence gates, CPU tutorial execution, and ecosystem import compatibility.

These remain below hardware qualification, real-dataset evidence, closed-loop qualification, and clinical evidence in the project evidence hierarchy.

## 18. Completed convergence and remaining architecture work

The following migrations are complete on `main`:

- canonical `SignalFrame` / `DecoderOutput` contracts;
- config compilation to `RuntimeGraph`;
- native graph execution for standard single/multimodal paths;
- config-first CLI operation;
- persistent lossless recording/replay with integrity verification;
- NWB/Zarr exports;
- plugin entry-point discovery;
- product/research/archive repository separation;
- deterministic scientific/runtime quality gates;
- ORION tokenizer contracts and benchmark;
- foundation-model interoperability layer;
- SourceWeigher reliability layer;
- mechanistic-evidence v1 contracts;
- faithful inspectable task-model layer.

The highest-value remaining architectural work is now narrower and more consequential:

1. clean-install/package compatibility and release gates;
2. explicit hardware qualification manifests and one real-device reference pipeline;
3. durable, non-pickle promoted model artifacts bound to input and analysis-manifest fingerprints;
4. real-dataset ORION/model/transfer/mechanism benchmarks with subject/session/device-disjoint protocols;
5. uncertainty-aware multimodal fusion;
6. auditable adaptation with rollback;
7. a first-class closed-loop safety/constraint plane;
8. externally maintained plugins and reference deployments built without kernel forks.

The desired endpoint is not maximum module count. It is a small, trustworthy execution kernel connected to a rigorous neural-intelligence and evidence ecosystem on which serious BCI systems can safely compound.
