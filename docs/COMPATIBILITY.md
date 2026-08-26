# Ecosystem Compatibility

neurOS treats compatibility as an evidence-bearing contract, not a logo wall. Every integration shown as supported or evidence-bearing must point to executable repository evidence, and the strongest public claim stops at the highest tier actually exercised.

Inspect the same source of truth programmatically:

```bash
neuros compatibility
neuros compatibility mne --json
neuros compatibility ngclearn --json
neuros compatibility snap --json
neuros compatibility --status supported --json
```

The registry lives in `neuros.compatibility` and is covered by contract tests. Documentation summarizes that registry rather than inventing a second support matrix with different semantics.

## Current matrix

| Integration | Status | Current capabilities | Strongest evidence | Qualification boundary |
| --- | --- | --- | --- | --- |
| BrainFlow | supported | source, continuous stream, device metadata | software contract | no named physical board/firmware/transport is hardware-qualified yet |
| Lab Streaming Layer | supported | source, continuous stream, explicit clock correction | software contract | no network topology or device clock has completed deployment qualification |
| MNE-Python | supported | Raw adapter, `SignalFrame` bridge, stream descriptors | integration | object interoperability only; arbitrary MNE preprocessing is not validated by neurOS |
| NWB / PyNWB | supported | export, session provenance | integration | neurOS archive remains authoritative for exact runtime replay semantics |
| Zarr | supported | export, session provenance | integration | export interoperability, not a replacement runtime format |
| MOABB | experimental | dataset adapter, longitudinal split authority, model ladder | real dataset | benchmark surface is still evolving; result claims remain protocol-specific |
| Braindecode | experimental | neural-window model adapter, upstream training bridge, decoder bridge | integration | qualified 1.7 whitelist only; stable mech-int, hardware, and closed-loop claims remain separate |
| SNAP spectral alignment | experimental | positive-rank spectrum, task power, residual target power | software contract + upstream conformance | no published-experiment reproduction or biological-alignment claim |
| ngc-learn | experimental | RateCell, fixed predictive reconstruction, governed Hebbian predictive adaptation, exact learning-state rollback | integration | deterministic upstream integration only; real-data efficacy, calibration reduction, STDP/spiking, online adaptation, hardware, and closed loop remain unqualified |
| OpenBCI | indirect | reachable through BrainFlow | none | no named OpenBCI configuration is hardware-qualified |
| Meta NeuralBench | planned | isolated benchmark worker, evidence extension | none | upstream runtime requirements must stay outside the kernel |
| IBM NeuroAIKit | planned | isolated SNU reference worker | none | legacy TensorFlow-era environment is intentionally isolated |
| NeuroAI Lab mouse-vision | planned | neural-predictivity benchmark | none | authoritative neural-data/artifact identities must be established first |
| NeuroAI Lab TDANN | planned | topographic representation evidence | none | external benchmark only; licensing/artifact identity must be resolved before code reuse |
| DANDI | planned | dataset discovery, provenance | none | no support claim yet |
| SpikeInterface | planned | invasive recording/analyzer bridge | none | no support claim yet |
| py_neuromodulation | planned | feature-transform adapter | none | no closed-loop qualification claim |
| Open Ephys | planned | source/plugin bridge | none | no support claim yet |

The CLI output is authoritative when this table and code ever disagree.

## Evidence tiers

Compatibility uses the same evidence ladder as the rest of neurOS:

1. **software-contract**: deterministic fixtures/fakes prove the adapter or numerical operator's local semantics;
2. **integration**: real upstream objects/files/SDK/package interfaces cross the neurOS boundary correctly;
3. **real-dataset**: named public datasets run under frozen evaluation semantics;
4. **hardware**: a named hardware + firmware + transport + software configuration passes measured qualification;
5. **closed-loop**: the complete sensing-to-action loop passes timing, failure, and constraint qualification;
6. **clinical**: a separate clinical/regulatory evidence process supports the claim.

These tiers are monotonic in responsibility, not automatic badges. A software-contract result does not imply hardware behavior; a real-dataset benchmark does not imply closed-loop safety.

## MNE interoperability

MNE is a direct scientific object bridge under the convergence architecture.

Install:

```bash
pip install "neuros[interop-mne]"
```

Convert MNE `Raw` into canonical frames:

```python
from neuros.interop import frames_from_raw, stream_descriptor_from_raw

raw = ...
descriptor = stream_descriptor_from_raw(raw, stream_id="subject-01/eeg")
frames = tuple(frames_from_raw(raw, stream_id=descriptor.stream_id, chunk_samples=256))
```

MNE stores arrays as `channel x sample`. neurOS chunk frames use `sample x channel` and explicitly stamp:

```text
axis_order = ("sample", "channel")
```

The reverse adapter refuses ambiguous two-dimensional frames that do not carry this metadata. It never guesses an axis, resamples data, pads missing samples, changes channel order, or repairs non-finite values.

## Live acquisition

### BrainFlow

BrainFlow is a hardware-family adapter behind the neurOS source contract. Deterministic tests cover board-aware channel selection and fail-closed behavior without pretending a simulated board proves physical-hardware qualification.

### Lab Streaming Layer

The first-class LSL driver uses deterministic stream discovery and keeps hidden liblsl post-processing disabled. The frame records the raw LSL timestamp and the exact clock-correction estimate used to produce synchronized time.

This keeps timing transformations inspectable and replayable.

## Benchmark interoperability

### MOABB

MOABB is used by the longitudinal EEG evidence program. The neurOS contribution is not another dataset wrapper. It is preservation of subject/session/run identity, frozen calibration/evaluation authority, and the ability to compare task decoders, frozen representations, and transfer strategies without changing the evaluation target.

See:

- [Real-World Evidence](REAL_WORLD_EVIDENCE.md)
- [Longitudinal EEG Showcase](LONGITUDINAL_EEG_SHOWCASE.md)
- [Longitudinal Model Ladder](LONGITUDINAL_MODEL_LADDER.md)

### Braindecode

Braindecode is an **experimental integration**, backed by a dedicated optional-dependency qualification lane rather than copied architectures.

The initial qualified surface is deliberately narrow:

- `EEGNet`
- `EEGConformer`
- `ShallowFBCSPNet`
- `Deep4Net`

The adapter never silently resamples, pads, crops, filters, changes channel order, or constructs sensor geometry. It records upstream version identity and a deterministic adapter/training configuration fingerprint.

### SNAP-derived spectral evidence

neurOS exposes SNAP-derived representation evidence without adding Torch/CUDA to the foundation package's required dependencies.

```python
from neuros.foundation_models import spectral_alignment_evidence

evidence = spectral_alignment_evidence(embeddings, neural_targets)
```

The implementation reports positive-rank spectral modes individually and aggregates target power outside that span into one invariant residual term because individual zero-eigenvalue eigenvectors are not uniquely defined. A dedicated CI lane also executes a pinned copy of the authors' real SNAP metric code and compares invariant quantities.

This is not a claim that the paper's experiments were reproduced or that a representation is biologically aligned.

See [NeuroAI Ecosystem Evidence](NEUROAI_ECOSYSTEM.md).

### ngc-learn

Install the isolated research integration with:

```bash
pip install "neuros-foundation[ngclearn]"
```

Three real upstream ngc-learn 3.2.x surfaces are now qualified at the **integration** tier.

#### RateCell transform

The basic transform records exact upstream/JAX identity, sample-rate-derived integration step, time-by-channel geometry, configuration, and hashes. It does not silently resample, filter, normalize, pad, reorder channels, or fit.

#### Fixed-weight predictive reconstruction

```python
from neuros.foundation_models import NgcLearnPredictiveCodingTransform

pc = NgcLearnPredictiveCodingTransform(
    latent_dim=4,
    settling_steps=30,
    seed=7,
)
result = pc.transform(samples, sample_rate_hz=250.0)
```

The circuit uses a real upstream `RateCell` latent, `GaussianErrorCell` reconstruction residual, and `StaticSynapse` generative/feedback connections. It resets per observation, clamps the input as the prediction-error target, ties feedback to the transpose of the fixed generative weights, and iteratively settles the latent.

Evidence includes latent/reconstruction shapes, exact component/runtime identity, settling parameters, weight/input/latent/reconstruction/trajectory hashes, and reconstruction-error reduction. The known-ground-truth upstream test uses an identity generative dictionary and requires the real circuit to reduce reconstruction error by more than 90% on Python 3.10 and 3.11.

#### Governed Hebbian predictive adaptation

```python
from neuros.foundation_models import NgcLearnHebbianPredictiveCoding

learner = NgcLearnHebbianPredictiveCoding(
    latent_dim=8,
    settling_steps=20,
    learning_rate=1e-3,
    optimizer="adam",
    seed=7,
)

adaptation = learner.adapt(calibration_samples, sample_rate_hz=250.0, epochs=2)
```

The adaptive circuit performs iterative predictive inference before every real upstream `HebbianSynapse.evolve()` M-step. The final latent activity is the pre statistic and the `GaussianErrorCell.dmu` residual is the post statistic. Feedback is retied to the current generative transpose before inference. No hidden post-update row normalization is applied.

A snapshot binds the exact upstream weight array and optimizer pytree into one combined state identity. This matters for Adam: restoring weights while leaving moments or the optimizer timestep changed is not a valid rollback. The integration tests require exact full-state restoration and the same future learning trajectory after rollback.

The cross-package evidence worker in `scripts/evidence/run_ngclearn_hebbian_authority.py` composes this learner with ORION's `AdaptationAuthority` without creating a foundation-to-ORION package dependency. It proves:

- adaptation receives exactly the complete authority-selected canonical calibration matrix;
- the learner's own adaptation-input SHA-256 matches the authority worker's hash;
- proposal/approval evidence is calibration-only;
- qualification inference is read-only;
- retain or exact full-state rollback follows a predeclared metric threshold;
- repeated complete evidence runs are byte-identical.

The dedicated real-upstream lane executes this on Python 3.10 and 3.11.

**Important statistical boundary:** the held-out qualification partition may determine whether the adapted state is retained or rolled back. It is therefore a state-selection set, not an untouched final scientific assessment set. Real efficacy and calibration-reduction studies require a third independent final-assessment partition after state selection.

The current integration does **not** qualify STDP, spiking-network adaptation, real-neural-data efficacy, calibration reduction, cross-subject/device transfer superiority, online adaptation, hardware behavior, closed-loop behavior, or clinical use.

See [NeuroAI Ecosystem Evidence](NEUROAI_ECOSYSTEM.md) for the scientific rationale and ORION comparison path.

### NeuralBench

NeuralBench should run as an isolated benchmark worker, especially while its Python/runtime requirements differ from the neurOS kernel support matrix. neurOS should extend its evidence rather than recreate its task registry.

## NeuroAI research references

The ecosystem registry names concrete research projects rather than pretending an entire lab or GitHub organization is one integration.

- IBM NeuroAIKit remains a planned isolated SNU reference worker.
- NeuroAI Lab `mouse-vision` is a planned neural-predictivity benchmark.
- NeuroAI Lab TDANN is a planned topographic-representation benchmark.
- Cognizant Neuro SAN Studio is intentionally **not** a neuroscience compatibility entry; its value to neurOS is product/community design inspiration rather than neural-runtime interoperability.

## Invasive and public-data lane

The next interoperability cluster is NWB + DANDI + SpikeInterface + py_neuromodulation.

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
2. implement an optional adapter/plugin or evidence operator outside `neuros-core`;
3. add deterministic contract tests;
4. add a registry entry at the weakest accurate status/evidence tier;
5. add a clean-install CI lane with the optional dependency where appropriate;
6. add real upstream integration evidence;
7. add named real-data, hardware, or closed-loop evidence only when actually exercised;
8. promote status or evidence tier only in the same change that adds the missing evidence.

A README example alone is never sufficient to mark an integration supported.
