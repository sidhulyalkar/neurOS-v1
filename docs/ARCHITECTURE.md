# neurOS Kernel Architecture

This document is the architectural source of truth for the neurOS runtime.

## 1. Design objective

neurOS should provide the smallest stable set of abstractions needed to construct reliable BCI systems across changing hardware, signal modalities, models, and research ideas.

The kernel is intentionally narrower than the repository. Research packages are allowed to move quickly. Kernel contracts should move slowly.

## 2. System boundary

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         Applications                                 │
│ communication | prosthetics | research | adaptive UX | experiments  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                         user-facing SDK
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
             ▼                                     ▼
┌─────────────────────────┐           ┌──────────────────────────────┐
│         neurOS          │           │            ORION             │
│                         │           │                              │
│ acquisition             │           │ tokenization                 │
│ data contracts          │◀─────────▶│ learned representations      │
│ synchronization         │           │ adaptive decoding            │
│ processing/runtime      │           │ personalization              │
│ recording/replay        │           │ foundation-model research    │
│ observability           │           │                              │
└────────────┬────────────┘           └──────────────┬───────────────┘
             │                                       │
             └─────────────────┬─────────────────────┘
                               ▼
                    Stable neural contracts
```

neurOS owns reliable execution. ORION owns neural intelligence.

## 3. Package responsibilities

### `neuros-core`

The kernel. It contains:

- canonical data contracts,
- runtime queue and lifecycle semantics,
- processing primitives,
- clock synchronization,
- recording/replay primitives,
- versioned configuration schemas,
- plugin discovery,
- orchestration compatibility APIs.

**Dependency rule:** `neuros-core` must not import concrete driver or decoder packages.

### `neuros-drivers`

Concrete hardware and dataset sources. Drivers may depend on `neuros-core`; the reverse dependency is forbidden.

Heavy or hardware-specific dependencies belong in extras such as EEG, video, audio, or NWB support.

### `neuros-models`

Conventional task-specific decoders and model adapters. Models implement the kernel decoder contracts while retaining legacy `train`/`predict` compatibility.

### `neuros-foundation`

Adapters and integrations for external neural representation/foundation-model ecosystems. It is not the home of proprietary ORION intelligence.

### `neuros`

The user-facing meta-package and CLI. This composition layer is allowed to depend on core, drivers, and models.

### `orion` / distribution `neuros-orion`

The stable ORION contract surface. The package currently defines token, representation, encoder, adaptive-decoder, and adaptation-proposal interfaces. Research implementations should migrate behind these contracts once validated.

### Research packages

`neuros-neurofm`, `neuros-mechint`, and experimental notebooks are research surfaces. They may depend on stable runtime contracts but must not become implicit dependencies of the kernel.

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

A plain floating-point timestamp is insufficient as a long-term BCI interchange contract. The frame explicitly represents different clocks so latency, synchronization, and provenance are not conflated.

`SignalFrame.timestamp_ns` resolves in this order:

1. synchronized time,
2. device time,
3. host receive time.

Legacy drivers can continue yielding `(timestamp_seconds, ndarray)` while they migrate. `BaseDriver.frames()` adapts that representation into `SignalFrame`.

## 5. Clock synchronization

`ClockSynchronizer` estimates an affine map:

```text
host_time = scale * device_time + offset
```

from a bounded window of timestamp pairs. It reports:

- offset,
- scale,
- drift in parts per million,
- residual uncertainty,
- number of observations.

The synchronizer can annotate a `SignalFrame` with synchronized time and `CLOCK_UNCERTAIN` quality state when residual timing error exceeds the configured threshold.

This software estimator is a fallback/alignment primitive. Hardware synchronization remains preferable when available.

## 6. Decoder semantics

The canonical model response is `DecoderOutput`:

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

**Critical rule:** absence of calibrated probability is represented as `confidence=None`. It must never be converted to artificial confidence such as `1.0`.

`BaseModel` remains compatible with traditional `train(X, y)` / `predict(X)` APIs but now exposes `infer()` that returns `DecoderOutput`.

## 7. Runtime graph

The target runtime is a typed directed graph:

```text
SOURCE -> TRANSFORM -> ... -> FUSION -> DECODER -> SINK
                  \                    /
                   ------ MONITOR -----
```

`RuntimeNode` records:

- node ID,
- node kind,
- operator,
- execution policy,
- optional latency budget,
- metadata.

`RuntimeEdge` records:

- source and destination,
- bounded capacity,
- overflow policy.

The current `Pipeline`/`Orchestrator` APIs remain as compatibility surfaces while execution progressively converges on this graph representation.

## 8. Backpressure

Every bounded runtime edge must have explicit overload behavior:

- `BLOCK`
- `DROP_OLDEST`
- `DROP_NEWEST`
- `FAIL`

`DROP_OLDEST` is the default for real-time compatibility because stale neural data is often less useful than fresh data, but applications should choose policy deliberately.

Queue telemetry includes:

- accepted items,
- dropped items,
- high-water mark.

Dropped samples are therefore measurable system behavior, not a hidden implementation detail.

## 9. Lifecycle

Runtime lifecycle states are explicit:

```text
CREATED -> STARTING -> RUNNING -> DRAINING -> STOPPED
                      |
                      +-> DEGRADED / FAILED
```

The single-stream orchestrator exposes public:

```python
await runtime.start()
async for result in runtime.stream_results():
    ...
await runtime.stop()
```

`run()` is a convenience wrapper over the same lifecycle rather than a separate execution implementation.

## 10. Recording and replay

Canonical frames can be recorded and replayed without changing their timestamps.

`FrameRecorder` provides a minimal sink for deterministic tests and experiments. `ReplaySource` implements the source contract and can replay either immediately or according to recorded timing at a configurable speed.

Long-term storage implementations such as NWB/Zarr should preserve these same contracts and provenance semantics.

Replay is a first-class requirement because it enables:

- deterministic regression tests,
- hardware-independent debugging,
- latency experiments,
- ORION training/evaluation,
- fault injection,
- reproducibility.

## 11. Configuration

The versioned `PipelineConfig` schema describes:

- named streams,
- source plugins,
- per-stream transforms,
- decoder plugin,
- sinks,
- monitors,
- queue capacity,
- overflow policy,
- metadata.

Config schema version 1 is deliberately small. Future migrations must be explicit rather than interpreting old experiment files differently without notice.

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

`PluginRegistry` supports both programmatic registration and installed entry-point discovery.

This allows hardware/model integrations to evolve independently from the kernel.

## 13. ORION boundary

ORION starts where raw/processed neural data becomes a machine-native neural representation:

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

## 14. Neurotokenization research

The existing neurotokenization research plan should target `orion.NeuroTokenizer` rather than create a parallel runtime interface. Event, binned-count, ISI, burst, synchrony, VQ motif, and assembly tokenizers can then be compared behind one contract.

Scientific promotion criteria remain based on fair comparisons across downstream decoding, transfer, robustness, motif recovery, interpretability, compression, and sample efficiency rather than reconstruction alone.

## 15. Dependency direction

Allowed:

```text
neuros-drivers ----> neuros-core
neuros-models -----> neuros-core
neuros-ui ---------> neuros-core
neuros-cloud ------> neuros-core
orion ------------> neuros-core
neuros ------------> core + drivers + models
research ----------> stable packages
```

Forbidden:

```text
neuros-core -> concrete drivers
neuros-core -> concrete models
neuros-core -> UI/cloud
neuros-core -> NeuroFM/mechint/ORION implementations
```

A PR that adds one of the forbidden dependencies must justify an architectural change, not merely an import convenience.

## 16. Quality gates

The repository CI is organized into three initial layers:

1. kernel contract tests across supported Python versions,
2. BCI end-to-end mock pipeline smoke tests,
3. ORION contract tests.

The intended expansion is:

```text
unit
contract
integration
replay
scientific validity
performance regression
hardware qualification
```

Hardware qualification should remain separate from generic CI and record exact device/firmware/runtime metadata.

## 17. Migration strategy

This architecture is being adopted incrementally to protect working research code.

### Compatibility kept now

- existing `BaseDriver` subclasses,
- tuple-based driver iteration,
- `BaseModel.train/predict`,
- `Pipeline` and `MultiModalPipeline`,
- existing CLI composition.

### Preferred new APIs

- `SignalFrame` / `StreamDescriptor`,
- `DecoderOutput`,
- versioned config,
- plugin registry,
- runtime overflow/lifecycle primitives,
- ORION representation contracts.

### Next migrations

1. compile `PipelineConfig` directly to `RuntimeGraph`,
2. make the native graph executor consume `SignalFrame` end to end,
3. add production NWB/Zarr recorder/replay adapters,
4. migrate built-in integrations to plugin entry points,
5. separate supported examples from research experiments,
6. archive historical migration/session documents,
7. add scientific and latency regression suites,
8. migrate validated NeuroFM/tokenization work behind ORION contracts.

The desired endpoint is not maximum module count. It is a small, trustworthy kernel on which sophisticated BCI research can safely compound.
