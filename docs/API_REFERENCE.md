# neurOS API Surface

This document is a maintained map of the stable and promoted architectural surfaces. It is intentionally smaller than a generated symbol catalog. For exact signatures, use the package source/docstrings for the installed version.

The purpose of this page is to make ownership, dependency direction, and evidence boundaries unambiguous.

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

A decoder may also expose probabilities, logits, pooled embeddings, model identity/version, inference timing, and structured metadata.

### Operator protocols

The kernel defines structural contracts for sources, transforms, decoders, sinks, monitors, and synchronization-related behavior. Implementations live in plugin packages rather than the kernel.

## `neuros.runtime`

### `RuntimeGraph`

Maintained typed directed acyclic graph of `RuntimeNode` and `RuntimeEdge` objects.

Node kinds include:

- source
- transform
- fusion
- decoder
- sink
- monitor

Edges carry bounded queue capacity and explicit overflow policy.

### `RuntimeExecutor`

Native execution engine shared by live, replay, single-stream, and standard multimodal paths.

Key usage:

```python
await executor.start()
async for output in executor.outputs():
    ...
await executor.stop()

snapshot = await executor.run()          # finite graph
snapshot = await executor.run_for(2.0)   # timed graph
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

Inspect the actual installed set with:

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

Dependency-light, lossless persistent session writer for `SignalFrame` streams.

Records can include:

- exact frame timing and sequence fields;
- stream descriptors;
- quality flags and metadata;
- per-frame payload SHA-256;
- config hash;
- Git SHA and package versions;
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

### Scientific probes

Small deterministic known-ground-truth probes validate that processing behavior retains expected scientific semantics.

## `neuros.drivers`

Drivers implement the source boundary. Hardware-specific dependencies belong in driver extras or external plugin packages.

Maintained drivers should expose or be adaptable to:

```python
source.descriptor
await source.start()
async for frame in source.frames():
    ...
await source.stop()
```

Legacy tuple iteration may remain for compatibility but is not the long-term neural ABI.

Importability is not hardware qualification. Named device/firmware/transport combinations require separate qualification evidence.

## `neuros.models`

`neuros-models` owns task-specific decoder implementations and the model-side contract for representation and mechanistic analysis.

### `BaseModel`

Maintains the familiar training/prediction surface and structured `infer(...) -> DecoderOutput` behavior.

### Neural decoders

Promoted deep families include:

- `EEGNetModel`
- `EEGConformerModel`
- `TemporalTransformerModel` / `TransformerModel`
- `CNNModel`
- `LSTMModel`
- `AttentionFusionModel`

Classical baselines include SVM, random forest, k-NN, gradient-boosting, and simple-classifier surfaces.

A model name is expected to match the algorithm actually executed. Optional backend absence should fail clearly rather than silently substitute another model family.

### `InterpretabilityManifest`

An explicit model-side declaration of stable analysis surfaces.

Related types include:

- `AnalysisCapability`
- `AnalysisSurface`
- `InterpretabilityManifest`
- `MechanisticallyInspectable`
- `validate_manifest_paths(...)`

A manifest can describe architecture family, backend, input axes, component paths, semantic roles, tensor axes, supported operations, recommended analyses, and limitations.

A manifest identifies valid intervention locations. It does **not** certify the meaning or biological interpretation of those components.

### Representations

Maintained neural decoders may expose `encode(X)` to return pooled representations. Structured inference can include the same embedding alongside logits/probabilities and a model/analysis-manifest fingerprint.

This representation seam can be consumed by foundation probes, SourceWeigher, or mech-int without making those packages dependencies of the model layer.

### Model discovery

```bash
neuros-models list
neuros-models list --mechint-ready
neuros-models show eeg-conformer
neuros-models doctor
```

`DecoderCard`, `list_decoder_cards()`, and `get_decoder_card(...)` provide a programmatic catalog surface.

### Mechanistic bridge

Inspectable models can create a `neuros-mechint` adapter when the optional research package is installed. The bridge validates the model's declared manifest paths against the actual backend module graph before an experiment begins.

## `neuros.foundation_models`

`neuros-foundation` is a registry-first interoperability and evaluation layer for external neural foundation-model ecosystems.

### Capability/catalog types

Promoted schema types include:

- `FoundationModelCard`
- `NeuralModality`
- `ModelTask`
- `ModelStatus`
- `AccessLevel`
- `AdapterAvailability`
- `IntegrationLevel`

`DEFAULT_MODEL_CARDS` and `catalog_by_id(...)` expose curated model metadata.

Catalog presence is not equivalent to local execution or reproduced upstream performance.

### Adapter layer

Key adapter interfaces/types include:

- `FoundationAdapter`
- `CallableAdapter`
- `ZunaAdapter`
- `NeuroFMXAdapter`
- `ModelRegistry`
- `DEFAULT_REGISTRY`
- `build_default_registry()`

Availability errors such as `AdapterUnavailableError` and `UnsupportedCapabilityError` provide fail-closed behavior when an integration or capability is unavailable.

Historical wrappers such as POYO/NDT/CEBRA/Neuroformer classes remain for compatibility, but the modern registry/adapters should be preferred for scientific comparisons.

### Representation probes

Maintained probes include:

- `effective_rank(...)`
- `mean_pairwise_cosine(...)`
- `linear_cka(...)`
- `pairwise_cka(...)`
- `invariance_score(...)`
- `linear_probe(...)`
- `domain_leakage_probe(...)`
- `representation_report(...)`

These characterize representation geometry or predictive utility. They are not by themselves causal mechanism tests.

### Benchmark protocol

`EvaluationProtocol`, `BenchmarkReport`, `benchmark_embeddings(...)`, and `sample_efficiency_curve(...)` provide protocol-stamped comparison surfaces so subject/session/site/device split semantics are explicit rather than buried in notebook code.

### `FoundationEmbeddingDecoder`

Wrap a frozen or callable representation encoder with a neurOS task readout so representation models can participate in standard decoder/runtime comparisons without being misrepresented as native task architectures.

## `neuros_sourceweigher`

`neuros-sourceweigher` owns reliability-aware source/domain selection and fusion.

### Core weighting

- `SourceWeigher`
- `WeightingResult`
- `WeightingDiagnostics`
- `project_to_simplex(...)`

### Strategies

- `DistanceWeigher`
- `GibbsRiskWeigher`
- `OnlineSourceWeigher`
- `MMDSourceWeigher`
- `RiemannianCovarianceWeigher`

Supporting distribution metrics include `rbf_mmd2(...)` and `spd_affine_invariant_distance(...)`.

### Representation/runtime integration

- `RepresentationSourceWeigher`
- `ReliabilityWeightedFusion`
- `RunningFeatureSummary`
- `summarize_features(...)`

The package can therefore weight subjects/sessions/sites/models in representation space or provide reliability-aware runtime fusion without moving the numerical core behind an HTTP dependency.

### Diagnostics

Diagnostics include effective sample size, leave-one-source-out stability, target perturbation sensitivity, and weight-shift measures. A plausible weighting result should be accompanied by stability evidence rather than only a normalized vector.

The optional service boundary is installed separately with `neuros-sourceweigher[service]`.

## `neuros_mechint`

`neuros-mechint` is the causal mechanism and evidence layer.

It owns research contracts for:

- activation capture/replacement and causal intervention;
- circuit/path faithfulness;
- held-out evidence packs;
- factorial comparisons;
- feature correspondence and causal substitution;
- replication and hierarchical uncertainty;
- dose-response/intervention sweeps;
- immutable model/data/config/artifact provenance.

### `NeurOSModelAdapter`

Native duck-typed bridge from an inspectable neurOS decoder to the generic PyTorch intervention machinery. It validates the decoder's complete analysis manifest before use and deliberately does not make `neuros-mechint` depend directly on `neuros-models`.

External ecosystems such as TransformerLens, NNsight, and SAELens are integrated through similarly narrow adapters.

**Claim boundary:** tool integration, attribution, sparse features, or a causal effect in one trained model do not establish a stable biological mechanism. Empirical claims require appropriate held-out and cross-deployment-unit evidence.

## `orion`

ORION is intentionally separate from runtime execution.

Core contracts include:

- `NeuroTokenizer`
- `NeuroTokenBatch`
- `NeuralEncoder`
- `RepresentationBatch`
- `AdaptiveDecoder`
- `AdaptationProposal`

Initial tokenizers include:

- `EventSpikeTokenizer`
- `BinnedCountTokenizer`
- `ISIRelativeTimeTokenizer`
- `BurstTokenizer`
- `SynchronyPacketTokenizer`
- `VQMotifTokenizer`
- `AssemblyTokenizer`

Fit-requiring implementations must be fit explicitly on training data before encode/evaluation.

## CLI

Primary neurOS commands:

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

Additional package CLIs include:

```text
neuros-models ...
neuros-foundation ...
neuros-mechint ...
```

Legacy model-registry/demo commands remain for compatibility but should not become the architecture for new functionality.

## Stability and dependency rule

The intended direction is:

```text
contracts
   <- runtime / config / recording / quality
   <- drivers / task models / SDK
   <- ORION and model/evidence integrations
   <- experiments / research studies
```

Kernel packages must not import research implementations. New capabilities should enter through contracts and plugin/adapter interfaces rather than new cross-package coupling.

For the current maturity of each package, see [`PROJECT_STATUS.md`](PROJECT_STATUS.md). For the next qualification sequence, see [`../ROADMAP.md`](../ROADMAP.md).
