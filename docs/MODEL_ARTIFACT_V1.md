# Model Artifact v1

Model Artifact v1 is the promoted decoder persistence boundary for neurOS.

It exists because a trained Python object is not a reproducible deployment artifact. A useful BCI/NeuroAI model needs an immutable identity that binds the exact weights to the model construction contract, neural input assumptions, software environment, scientific provenance, and mechanistic-analysis surface.

The core invariant is:

> A promoted decoder must be reconstructable from explicit trusted code plus inert tensor data, and its identity must change whenever any identity-bearing model, input, provenance, or weight content changes.

## What it replaces

`ModelRegistry` predates this contract and serializes complete Python objects through pickle. It remains available only for backward compatibility with trusted local files.

Pickle is not a safe exchange or deployment format. Loading an untrusted pickle can execute arbitrary code before neurOS has an opportunity to validate the resulting object.

Model Artifact v1 therefore never calls `pickle.load`, never accepts a pickle weight file, and never imports a Python path supplied by artifact contents.

## Bundle layout

A v1 artifact contains exactly two regular files:

```text
my-decoder/
├── manifest.json
└── weights.safetensors
```

Extra files, subdirectories, and symbolic links are rejected by the verifier. This prevents an apparently verified artifact directory from carrying unbound side payloads.

`weights.safetensors` contains tensor-only PyTorch state. `manifest.json` is strict UTF-8 JSON with duplicate keys, non-finite values, unknown fields, and stale derived hashes rejected.

Install the optional safe artifact profile with:

```bash
pip install "neuros-models[artifact]"
```

The base `neuros-models` installation does not require PyTorch or safetensors merely to inspect model catalogs or use classical decoders.

## Identity hierarchy

Model Artifact v1 deliberately keeps full identities separate from display fingerprints.

### Weight identity

`weights_sha256` hashes the exact `weights.safetensors` bytes.

A bit change in the weight file invalidates verification before model construction.

### Manifest identity

`manifest_sha256` hashes canonical identity-bearing manifest content, including:

- artifact and safe-factory IDs;
- model type and backend/version;
- constructor configuration;
- neural input contract;
- weight SHA-256;
- the embedded interpretability manifest and its full SHA-256;
- Git commit SHA;
- exact package versions;
- training/evaluation authority SHA-256 values;
- preprocessing and calibration state SHA-256 values;
- optional Scientific Authority study SHA-256;
- deterministic user metadata.

### Artifact identity

`artifact_sha256` is domain-separated as a Model Artifact v1 identity and binds the canonical manifest identity to the exact weight identity.

`display_fingerprint` is only the first 16 characters of `artifact_sha256`. It is for logs and UI labels, not durable evidence joins.

## Interpretability identity

`InterpretabilityManifest.sha256()` is now the durable identity for the mechanistic surface exposed by a model.

The historical `InterpretabilityManifest.fingerprint()` API remains available, but it is display-only and returns a 16-character prefix of the full SHA.

A promoted artifact embeds both the complete interpretability manifest and `interpretability_manifest_sha256`. Loading reconstructs the registered model and refuses the artifact if the installed model's current interpretability contract differs.

This prevents a checkpoint from silently retaining an old mechanistic-evidence label after named layers, roles, or analysis semantics change.

## Input contract

`ModelInputContract` makes neural input assumptions part of artifact identity rather than relying on positional convention.

The v1 contract records:

```text
axes
shape
dtype
channel_names
sample_rate_hz
signal_unit
metadata
```

Shape dimensions may use `None` where a dimension is intentionally variable. Fixed dimensions are enforced before inference. Dtype is exact rather than silently coerced by the artifact wrapper.

For channel-based decoders, channel names may be declared only when the contract contains a fixed `channel` axis of matching length.

A typical EEG contract is:

```python
from neuros.models import ModelInputContract

input_contract = ModelInputContract(
    axes=("batch", "channel", "time"),
    shape=(None, 22, 512),
    dtype="float32",
    channel_names=("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P3", "P4"),
    sample_rate_hz=512.0,
    signal_unit="uV",
)
```

This contract does not perform resampling, re-referencing, unit conversion, or montage repair. Those operations belong upstream and their fitted/fixed state should be bound through preprocessing authority.

## Safe model reconstruction

Artifacts do not carry an import string such as `some_package.module:Class` that is executed during load.

Model Artifact v1 uses a fixed built-in factory registry for the initial promoted PyTorch decoders:

- EEGNet;
- temporal CNN;
- LSTM;
- temporal Transformer;
- EEG-Conformer;
- attention-fusion decoder.

A manifest selects a stable factory ID. Unknown factory IDs fail closed.

External or foundation-model artifacts can be added later only through an explicit serializer/factory contract. V1 does not fall back to dynamic imports or pickle when a model is unsupported.

## Export

A model must already be trained before it can be promoted.

```python
from neuros.models import EEGNetModel, ModelInputContract, export_model_artifact

model = EEGNetModel(n_channels=22, n_classes=2, device="cpu")
model.train(X_train, y_train)

manifest = export_model_artifact(
    model,
    "artifacts/eegnet-session-transfer",
    artifact_id="kumar2024-eegnet",
    input_contract=ModelInputContract(
        axes=("batch", "channel", "time"),
        shape=(None, 22, 512),
        dtype="float32",
        sample_rate_hz=512.0,
        signal_unit="uV",
    ),
    git_sha="<40-character-git-sha>",
    training_authority_sha256s=("<training-observation-authority-sha256>",),
    evaluation_authority_sha256s=("<qualified-result-or-evaluation-authority-sha256>",),
    preprocessing_state_sha256s=("<preprocessing-state-sha256>",),
    scientific_study_sha256="<scientific-study-sha256>",
)

print(manifest.artifact_sha256)
```

Export is atomic. neurOS writes into a sibling temporary directory, hashes and parses its own output through the public verifier, and only then renames the verified directory into the requested final location.

A promoted artifact is immutable. Export refuses an existing destination rather than offering `overwrite=True`.

## Verify and inspect

Verification does not construct the decoder:

```python
from neuros.models import verify_model_artifact

manifest = verify_model_artifact("artifacts/eegnet-session-transfer")
print(manifest.artifact_sha256)
```

The CLI exposes the same read-only operation:

```bash
neuros-models artifact verify artifacts/eegnet-session-transfer --json
```

Verification checks the manifest identity and exact weight-file SHA. It does not claim that the model is scientifically valid or that the publisher is trusted.

## Load

```python
from neuros.models import load_model_artifact

model = load_model_artifact("artifacts/eegnet-session-transfer", device="cpu")
output = model.infer(X_float32)
```

Loading performs the following sequence:

1. strict artifact-envelope verification;
2. exact package-version verification;
3. safe built-in factory lookup;
4. model reconstruction from canonical constructor configuration;
5. current-vs-artifact interpretability-contract SHA check;
6. safetensors parsing;
7. exact tensor-name comparison;
8. exact tensor shape and dtype comparison;
9. strict state-dict load;
10. return of an `ArtifactBackedDecoder` that enforces the neural input contract.

The artifact-backed wrapper refuses `train`, `partial_fit`, and in-place `adapt` operations. Its inference output includes the full artifact, manifest, and interpretability SHA identities.

The underlying framework model can still be exposed for mechanistic experiments. Such interventions create mutable in-memory research state. Reload the immutable artifact before authoritative deployment inference after parameter-changing experiments.

## Content-addressed store and rollback

`ModelArtifactStore` separates immutable model objects from mutable deployment selection.

```text
store/
├── artifacts/
│   ├── <artifact-sha-A>/
│   └── <artifact-sha-B>/
└── refs/
    └── active.json
```

Publish verified artifacts:

```python
from neuros.models import ModelArtifactStore

store = ModelArtifactStore("model-store")
a = store.publish("exports/model-a")
b = store.publish("exports/model-b")
```

Move the active deployment reference:

```python
store.activate("active", b.artifact_sha256)
```

Rollback:

```python
store.rollback("active", a.artifact_sha256)
```

Rollback changes only `refs/active.json`. The historical artifact directories are not modified or overwritten.

A model can also be loaded directly by full content identity:

```python
model = store.load(a.artifact_sha256)
```

Resolve refs from the CLI:

```bash
neuros-models artifact resolve model-store active --json
```

## Scientific Authority integration

Model Artifact v1 is designed to bind runtime model state to Scientific Authority v2 without making `neuros-models` depend on ORION.

The manifest therefore stores full SHA references rather than importing Scientific Authority classes:

- `training_authority_sha256s`;
- `evaluation_authority_sha256s`;
- `preprocessing_state_sha256s`;
- `calibration_state_sha256s`;
- `scientific_study_sha256`.

The evidence layer remains responsible for verifying what those authorities mean. A random 64-character value is syntactically valid artifact provenance but does not become scientific evidence merely because it is stored in the manifest.

This separation keeps the dependency graph clean:

```text
Scientific Authority / evidence
          |
          | full SHA references
          v
Model Artifact v1
          |
          v
trusted model factory + safe tensor weights
          |
          v
runtime inference
```

## Environment identity

A promoted artifact records exact package versions for the artifact runtime, including neurOS models/core, PyTorch, and safetensors.

The v1 loader requires those versions to match exactly. This is intentionally conservative. A future explicit migration/compatibility contract may safely widen version ranges, but v1 does not silently assume that a checkpoint has identical semantics under a different runtime.

Git SHA is recorded as provenance but is not independently recoverable from every installed wheel. Reproducible release/qualification bundles should therefore preserve the Git-to-package mapping externally as well.

## Threat model

Model Artifact v1 protects against several concrete failure modes:

- arbitrary pickle execution in the promoted load path;
- artifact-provided arbitrary Python import paths;
- bit-level weight corruption;
- manifest-field tampering;
- duplicate-key JSON ambiguity;
- unbound extra files or symlinks;
- stale/forged derived hashes;
- tensor name, shape, or dtype mismatch;
- hidden input dtype/shape coercion at the artifact boundary;
- accidental overwrite of a promoted artifact;
- rollback implemented by mutating historical model bytes.

It does **not** provide publisher authentication or code signing. SHA-256 establishes content identity and integrity relative to an expected hash; it does not establish who produced the artifact or whether the producer should be trusted.

Supply-chain signing, transparency, remote registry authorization, key management, and organization-level promotion policy are separate future concerns.

## Unsupported promotion paths

V1 intentionally does not call joblib/pickle for sklearn models. Classical estimators remain usable as research/runtime models but are not promoted through this safe artifact contract until a serializer is defined that does not reintroduce arbitrary object execution.

Similarly, foundation models, external plugins, quantized runtimes, ONNX, CoreML, TensorRT, and device-specific compiled engines need explicit artifact adapters. They should not be smuggled into v1 as opaque blobs.

## Evidence boundary

Model Artifact v1 establishes software evidence for:

- exact model-weight identity;
- exact manifest/provenance identity;
- safe built-in reconstruction semantics;
- input-contract enforcement;
- deterministic same-environment reload equivalence where qualified;
- content-addressed rollback semantics.

It does not itself establish:

- decoder superiority;
- physiological validity;
- generalization to unseen participants/sessions/devices;
- calibration quality;
- physical device timing or reliability;
- online closed-loop benefit;
- participant safety or benefit;
- clinical validity or safety.

Those claims require their own Scientific Authority and evidence planes.
