# neurOS API Surface

This document is a maintained map of the stable architectural surfaces. It is intentionally smaller than the historical hand-written API catalog, which is preserved under `docs/archive/development/API_REFERENCE_PRE_KERNEL.md`.

For exact signatures, use the package source/docstrings for the installed version. The purpose of this page is to make ownership and dependency direction unambiguous.

## `neuros.contracts`

### `SignalFrame`

Canonical neural-data interchange object.

Important fields include:

- `stream_id`
- `sequence_id`
- `data`
- `sample_rate_hz`
- `host_receive_time_ns`
- `device_time_ns`
- `synchronized_time_ns`
- `clock_domain`
- `quality`
- `metadata`

### `StreamDescriptor`

Describes stream modality, sample rate, channel names/types/units, device/manufacturer, clock domain, and stable stream metadata.

### `DecoderOutput`

Structured inference result. Confidence and uncertainty are optional. neurOS does not fabricate certainty when a decoder cannot supply it.

### Operator protocols

The kernel defines structural contracts for sources, transforms, decoders, sinks, monitors, and synchronization-related behavior. Implementations live in plugin packages rather than the kernel.

## `neuros.runtime`

### `RuntimeGraph`

Typed directed acyclic graph of `RuntimeNode` and `RuntimeEdge` objects.

Node kinds:

- source
- transform
- fusion
- decoder
- sink
- monitor

Edges carry bounded queue capacity and explicit overflow policy.

### `RuntimeExecutor`

Native execution engine shared by live, replay, single-stream, and multimodal paths.

Key methods:

```python
await executor.start()
async for output in executor.outputs():
    ...
await executor.stop()

# finite graph
snapshot = await executor.run()

# timed graph
snapshot = await executor.run_for(2.0)
```

Snapshots include runtime state, failures, per-node latency/activity, and per-edge accepted/dropped/high-water metrics.

### `OverflowPolicy`

Supported policies:

- `block`
- `drop_oldest`
- `drop_newest`
- `fail`

Overflow behavior must be explicit. Silent, unmeasured data loss is not a supported runtime semantic.

## `neuros.config`

### `PipelineConfig`

Versioned configuration schema for streams, plugins, decoder, runtime policy, sinks, monitors, and metadata.

### `load_config(path)`

Load and validate YAML configuration.

### `resolve_config(config, source_overrides=None)`

Instantiate installed plugins and compile the configuration to a validated `RuntimeGraph`.

`source_overrides` lets deterministic replay replace live hardware sources without instantiating the original device SDK.

## `neuros.plugins`

Entry-point groups:

```text
neuros.sources
neuros.transforms
neuros.tokenizers
neuros.encoders
neuros.decoders
neuros.sinks
neuros.monitors
```

Use the CLI to inspect the actual installed set:

```bash
neuros plugins --json
neuros devices --json
```

## `neuros.pipeline`

### `Pipeline`

Compatibility and convenience facade for a standard single-stream BCI path. Standard execution compiles to `RuntimeGraph` and uses `RuntimeExecutor`.

### `MultiModalPipeline`

Compatibility/convenience facade for multiple sources. Standard execution uses the same graph executor and typed fusion path.

Custom historical processing-agent classes remain an explicit migration escape hatch and should be converted to transforms when maintained.

## `neuros.recording`

### `SessionArchiveWriter`

Dependency-free, lossless persistent session writer for `SignalFrame` streams.

Records:

- exact frame timing and sequence fields,
- stream descriptors,
- quality flags and metadata,
- per-frame payload SHA-256,
- config hash,
- Git SHA and package versions,
- runtime metrics and model-artifact references.

### `SessionArchiveReader`

Read and optionally verify persistent archives.

### `ArchiveReplaySource`

Expose one archived stream through the same source contract used by live hardware.

### `RecordingSource`

Decorate a source to record exact incoming frames before forwarding them.

### `export_nwb` / `export_zarr`

Optional interoperability exports. The canonical neurOS archive remains the authoritative lossless replay representation.

## `neuros.quality`

### `QualityThresholds`

Versionable runtime acceptance criteria for activity, queue loss, failures, and p99 latency.

### `evaluate_runtime_snapshot`

Evaluate one executor snapshot against explicit thresholds.

### `FaultProfile` / `PerturbedSource`

Deterministic fault injection for packet loss, channel dropout, timestamp jitter, additive noise, and clock drift.

### `BenchmarkManifest`

Capture Git/config/data/artifact/package/host provenance for benchmark evidence.

### scientific probes

Small deterministic known-ground-truth probes validate that processing behavior retains expected scientific semantics.

## `neuros.drivers`

Drivers implement the source boundary. Hardware-specific dependencies belong in driver extras or external plugin packages.

All maintained drivers should expose or be adaptable to:

```python
source.descriptor
await source.start()
async for frame in source.frames():
    ...
await source.stop()
```

Legacy tuple iteration may remain for compatibility but is not the long-term neural ABI.

## `neuros.models`

Conventional decoders live here. Maintained decoders should expose `infer(X) -> DecoderOutput` and capabilities describing probabilities, uncertainty, online fit, embeddings, or state when supported.

A training-free `ThresholdDecoder` exists for installation/config/runtime smoke tests. It is not a scientific BCI baseline.

## `orion`

ORION is intentionally separate from runtime execution.

Core contracts:

- `NeuroTokenizer`
- `NeuroTokenBatch`
- `NeuralEncoder`
- `RepresentationBatch`
- `AdaptiveDecoder`
- `AdaptationProposal`

Initial tokenizers:

- `EventSpikeTokenizer`
- `BinnedCountTokenizer`
- `ISIRelativeTimeTokenizer`
- `BurstTokenizer`
- `SynchronyPacketTokenizer`
- `VQMotifTokenizer`
- `AssemblyTokenizer`

Fit-requiring implementations must be fit explicitly on training data before encode/evaluation.

## CLI

Primary commands:

```text
neuros doctor
neuros plugins
neuros devices
neuros validate CONFIG
neuros run CONFIG
neuros benchmark CONFIG
neuros record CONFIG --output SESSION
neuros inspect SESSION
neuros replay SESSION --config CONFIG
```

Legacy model-registry/demo commands remain for compatibility but should not become the architecture for new functionality.

## Stability rule

The stable direction is:

```text
contracts
   <- runtime / config / quality
   <- drivers / models / SDK
   <- ORION and other research packages
```

Kernel packages must not import research implementations. New capabilities should enter through contracts and plugin interfaces rather than new cross-package coupling.
