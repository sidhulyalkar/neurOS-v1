# Ecosystem Compatibility

neurOS treats compatibility as an evidence-bearing contract, not a logo wall. Every integration shown as supported must point to executable repository evidence, and the strongest public claim stops at the highest tier actually exercised.

Inspect the same source of truth programmatically:

```bash
neuros compatibility
neuros compatibility mne --json
neuros compatibility --status supported --json
```

The registry lives in `neuros.compatibility` and is covered by contract tests. Documentation should summarize that registry rather than invent a second support matrix with different semantics.

## Current matrix

| Integration | Status | Current capabilities | Strongest evidence | Qualification boundary |
| --- | --- | --- | --- | --- |
| BrainFlow | supported | source, continuous stream, device metadata | software contract | no named physical board/firmware/transport is hardware-qualified yet |
| Lab Streaming Layer | supported | source, continuous stream, explicit clock correction | software contract | no network topology or device clock has completed deployment qualification |
| MNE-Python | supported | Raw adapter, `SignalFrame` bridge, stream descriptors | integration | object interoperability only; arbitrary MNE preprocessing is not validated by neurOS |
| NWB / PyNWB | supported | export, session provenance | integration | neurOS archive remains authoritative for exact runtime replay semantics |
| Zarr | supported | export, session provenance | integration | export interoperability, not a replacement runtime format |
| MOABB | experimental | dataset adapter, longitudinal split authority, model ladder | real dataset | benchmark surface is still evolving; result claims remain protocol-specific |
| Braindecode | experimental | neural-window model adapter, upstream training bridge, decoder bridge | integration | qualified 1.7 whitelist only; no real-dataset, stable mech-int, hardware, or closed-loop claim yet |
| OpenBCI | indirect | reachable through BrainFlow | none | no named OpenBCI configuration is hardware-qualified |
| Meta NeuralBench | planned | isolated benchmark worker, evidence extension | none | upstream runtime requirements must stay outside the kernel |
| DANDI | planned | dataset discovery, provenance | none | no support claim yet |
| SpikeInterface | planned | invasive recording/analyzer bridge | none | no support claim yet |
| py_neuromodulation | planned | feature-transform adapter | none | no closed-loop qualification claim |
| Open Ephys | planned | source/plugin bridge | none | no support claim yet |

The CLI output is authoritative when this table and code ever disagree.

## Evidence tiers

Compatibility uses the same evidence ladder as the rest of neurOS:

1. **software-contract**: deterministic fixtures/fakes prove the adapter's local semantics;
2. **integration**: real upstream objects/files/SDK interfaces cross the neurOS boundary correctly;
3. **real-dataset**: named public datasets run under frozen evaluation semantics;
4. **hardware**: a named hardware + firmware + transport + software configuration passes measured qualification;
5. **closed-loop**: the complete sensing-to-action loop passes timing, failure, and constraint qualification;
6. **clinical**: a separate clinical/regulatory evidence process supports the claim.

These tiers are monotonic in responsibility, not automatic badges. A software-contract result does not imply hardware behavior; a real-dataset benchmark does not imply closed-loop safety.

## MNE interoperability

MNE is the first direct scientific object bridge added under the convergence architecture.

Install:

```bash
pip install "neuros[interop-mne]"
```

Convert MNE `Raw` into canonical frames:

```python
from neuros.interop import frames_from_raw, stream_descriptor_from_raw

raw = ...  # existing MNE Raw object
descriptor = stream_descriptor_from_raw(raw, stream_id="subject-01/eeg")
frames = tuple(
    frames_from_raw(
        raw,
        stream_id=descriptor.stream_id,
        chunk_samples=256,
    )
)
```

MNE stores arrays as `channel x sample`. neurOS chunk frames use `sample x channel` and explicitly stamp:

```text
axis_order = ("sample", "channel")
```

The reverse adapter refuses ambiguous two-dimensional frames that do not carry this metadata. It never guesses an axis, resamples data, pads missing samples, changes channel order, or repairs non-finite values.

```python
from neuros.interop import raw_from_signal_frames

reconstructed = raw_from_signal_frames(frames, descriptor=descriptor)
```

When MNE exposes an absolute `meas_date`, the adapter derives synchronized frame timestamps from the measurement origin plus sample position. Without an absolute measurement time, recording-relative position remains provenance metadata and neurOS does not fabricate a synchronized clock.

## Live acquisition

### BrainFlow

BrainFlow is a hardware-family adapter behind the neurOS source contract. Deterministic tests cover board-aware channel selection and fail-closed behavior without pretending a simulated board proves physical-hardware qualification.

### Lab Streaming Layer

The first-class LSL driver uses deterministic stream discovery and keeps hidden liblsl post-processing disabled. The frame records the raw LSL timestamp and the exact clock-correction estimate used to produce synchronized time.

This keeps timing transformations inspectable and replayable.

## Benchmark interoperability

### MOABB

MOABB is already used by the longitudinal EEG evidence program. The important neurOS contribution is not another dataset wrapper. It is preservation of subject/session/run identity, frozen calibration/evaluation authority, and the ability to compare task decoders, frozen representations, and transfer strategies without changing the evaluation target.

See:

- [Real-World Evidence](REAL_WORLD_EVIDENCE.md)
- [Longitudinal EEG Showcase](LONGITUDINAL_EEG_SHOWCASE.md)
- [Longitudinal Model Ladder](LONGITUDINAL_MODEL_LADDER.md)

### Braindecode

Braindecode is now an **experimental integration**, backed by a dedicated optional-dependency qualification lane rather than by copied architectures. Install with:

```bash
pip install "neuros[braindecode]"
```

The initial qualified surface is deliberately narrow:

- `EEGNet`
- `EEGConformer`
- `ShallowFBCSPNet`
- `Deep4Net`

All four are exercised against the canonical neurOS `(batch, channel, time)` window contract on the pinned Braindecode 1.7 line. EEGNet is additionally trained through the real upstream `EEGClassifier` path and executed through `SignalFrame -> NeuralWindow -> RuntimeGraph -> DecoderOutput` with window provenance retained.

The adapter never silently resamples, pads, crops, filters, changes channel order, or constructs sensor geometry. It records upstream version identity and a deterministic adapter/training configuration fingerprint. Python 3.10 remains free of the optional Braindecode dependency.

This is an **integration** claim, not a task-performance claim. Braindecode remains below real-dataset evidence until one of its adapters runs under the frozen neurOS longitudinal authority on a named public dataset. Stable mechanistic hook paths are also deliberately unclaimed until architecture/version-specific intervention surfaces are qualified.

See [Braindecode interoperability](BRAINDECODE.md) for the full boundary.

### NeuralBench

NeuralBench should run as an isolated benchmark worker, especially while its Python/runtime requirements differ from the neurOS kernel support matrix. The worker should consume or export explicit dataset/model/protocol identities and return neurOS evidence artifacts containing additional dimensions such as:

- subject/session/device transfer;
- representation geometry;
- montage robustness;
- artifact sensitivity;
- calibration cost;
- mechanistic intervention results;
- runtime latency and memory.

neurOS should extend NeuralBench evidence rather than recreate its task registry.

## Invasive and public-data lane

The next interoperability cluster is NWB + DANDI + SpikeInterface + py_neuromodulation. The intended direction is:

```text
DANDI / NWB / SpikeInterface
             |
             v
        neurOS adapter
             |
             v
 SignalFrame / event contracts
             |
     runtime + replay
             |
             +------> ORION
             |
             +------> Evidence
```

The invasive lane should not force spike-sorting, dataset-client, or neuromodulation dependencies into `neuros-core`.

## Adding an integration

A new integration should land in this order:

1. define the exact external boundary and what neurOS will not reimplement;
2. implement an optional adapter/plugin outside `neuros-core`;
3. add deterministic contract tests;
4. add a registry entry at the weakest accurate status/evidence tier;
5. add a clean-install CI lane with the optional dependency;
6. add real upstream integration evidence where practical;
7. promote status or evidence tier only in the same change that adds the missing evidence.

A README example alone is never sufficient to mark an integration supported.
