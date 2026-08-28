# Model Artifact v1

Model Artifact v1 is the promoted decoder persistence boundary for neurOS.

A trained Python object is mutable research state, not a reproducible deployment artifact. A promoted BCI/NeuroAI decoder needs a durable identity that binds exact weights to model construction, neural input assumptions, output semantics, software environment, scientific provenance, and mechanistic-analysis surfaces.

The core invariant is:

> A promoted decoder must be reconstructable from explicit trusted code plus inert tensor data, and its identity must change whenever any identity-bearing model, input, output, provenance, or weight content changes.

A second deployment invariant is equally important:

> Public inspection of a promoted artifact must not be able to mutate the model state used for authoritative inference while the decoder continues to report the original artifact identity.

## What it replaces

`ModelRegistry` predates this contract and serializes complete Python objects through pickle. It remains available only for backward compatibility with trusted local files.

Pickle is not a safe exchange or deployment format. Loading an untrusted pickle can execute arbitrary code before neurOS can validate the resulting object.

Model Artifact v1 therefore never calls `pickle.load`, never accepts a pickle weight file, and never imports a Python path supplied by artifact contents.

## Three distinct layers

neurOS deliberately separates mutable research state, immutable-by-contract artifacts, and mutable deployment selection:

```text
train/adapt model
      |
      v
mutable research model
      |
      | promote
      v
content-addressed Model Artifact
      |
      | publish
      v
artifact store
      |
      | active/candidate/rollback refs
      v
deployment selection
```

The model may change during research. A promoted artifact does not. A deployment reference may move between already-published artifacts without rewriting their bytes.

"Immutable" here means immutable through the neurOS artifact API and identity contract. It is not a claim that a hostile operating-system administrator cannot modify files on disk. Any later byte modification is detected when the artifact is verified again.

## Bundle layout

A v1 artifact contains exactly two regular files:

```text
my-decoder/
├── manifest.json
└── weights.safetensors
```

Extra files, subdirectories, root/internal symbolic links, duplicate JSON keys, non-finite values, unknown fields, and stale derived hashes are rejected.

`weights.safetensors` contains tensor-only PyTorch state. The artifact never contains a serialized Python object graph.

Install the optional artifact profile with:

```bash
pip install "neuros-models[artifact]"
```

The base package does not require PyTorch or safetensors merely to inspect catalogs or use classical decoders.

## Identity hierarchy

Model Artifact v1 keeps durable identities separate from display fingerprints.

### Weight identity

`weights_sha256` hashes the exact `weights.safetensors` bytes.

### Manifest identity

`manifest_sha256` hashes canonical identity-bearing manifest content, including:

- artifact and safe-factory IDs;
- model type and backend/version;
- constructor configuration;
- neural input contract;
- decoder output contract;
- weight SHA-256;
- embedded interpretability manifest and its full SHA-256;
- Git commit SHA;
- exact package versions;
- training/evaluation authority SHA-256 values;
- preprocessing and calibration state SHA-256 values;
- optional Scientific Authority study SHA-256;
- deterministic user metadata.

### Artifact identity

`artifact_sha256` is domain-separated as a Model Artifact v1 identity and binds canonical manifest identity to exact weight identity.

`display_fingerprint` is only the first 16 characters. It is suitable for UI/log labels, not durable evidence joins.

## Interpretability identity

`InterpretabilityManifest.sha256()` is the durable identity for a model's mechanistic-analysis contract.

The historical `fingerprint()` API remains display-only and returns a 16-character prefix.

A promoted artifact embeds both the complete manifest and its full SHA. Loading reconstructs the registered model and refuses the artifact if the installed model's current interpretability contract differs.

### Deployment inspection is detached

A previous draft of Model Artifact v1 exposed the live inference `torch.nn.Module` through `analysis_model()`. That was unsafe: a mechanistic intervention could mutate deployment weights while subsequent inference still reported the old artifact SHA.

The hardened contract does not expose the live deployment module. `ArtifactBackedDecoder.analysis_model()` returns a detached evaluation-mode snapshot with gradients disabled. Researchers may intervene on that snapshot without changing subsequent authoritative artifact-backed inference.

If a parameter-changing intervention should become a deployable model, it must be promoted as a new artifact with a new weight and artifact identity.

## Input contract

`ModelInputContract` makes neural input assumptions part of artifact identity instead of relying on positional convention.

The v1 contract records:

```text
axes
shape
dtype
channel_names
sample_rate_hz
signal_unit
stream_descriptor_sha256   # optional exact source identity
metadata
```

Shape dimensions may use `None` only where a dimension is intentionally variable. Dtype is exact rather than silently coerced by the artifact wrapper.

For channel-based decoders, channel names may be declared only when the contract contains a fixed `channel` axis of matching length.

A typical EEG contract is:

```python
from neuros.models import ModelInputContract

input_contract = ModelInputContract(
    axes=("batch", "channel", "time"),
    shape=(None, 22, 512),
    dtype="float32",
    channel_names=(
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1",
        "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1",
        "CP2", "CP6", "P3", "P4",
    ),
    sample_rate_hz=512.0,
    signal_unit="uV",
)
```

### StreamDescriptor binding

The array contract can also be checked against neurOS's canonical `StreamDescriptor` before inference:

```python
artifact_decoder.validate_stream_descriptor(descriptor)
```

This checks declared channel identity and nominal sampling-rate semantics, and can bind the complete canonical descriptor through `stream_descriptor_sha256`.

Exact descriptor-SHA binding is optional intentionally. A decoder qualified for one exact source configuration may require it. A transfer study designed to span devices or sessions may instead bind only the semantic input properties that are scientifically intended to remain invariant. Cross-device portability must not be accidentally destroyed by making stream IDs or manufacturer strings part of every model identity.

This contract does not resample, re-reference, reorder channels, convert units, or repair a montage. Those transformations belong upstream and their fitted/fixed state should be bound through preprocessing authority.

## Output contract

Input compatibility is only half of deployment correctness. Downstream BCI logic must also know what a decoder output actually means.

`ModelOutputContract` binds:

```text
class_labels
task
score_semantics
probability_semantics
uncertainty_semantics
probability_calibration_method
probability_calibration_sha256
metadata
```

Supported probability semantics are explicit:

- `uncalibrated_softmax`
- `calibrated_probability`
- `unavailable`

The built-in PyTorch v1 factories produce raw class logits and softmax scores. They are therefore promoted as `uncalibrated_softmax` and **cannot** be relabeled as calibrated probabilities by metadata alone.

A future calibrated artifact must use a qualified calibration implementation and bind the exact calibration state/method. This prevents a common safety error where a numerically normalized softmax score is treated as a calibrated confidence estimate.

Artifact-backed `DecoderOutput.metadata` carries the class labels and score/probability/uncertainty semantics so Evidence, Studio, ORION, and closed-loop policy code can distinguish them mechanically.

## Safe model reconstruction

Artifacts do not carry executable import strings such as `some_package.module:Class`.

Model Artifact v1 uses a fixed safe-factory registry for the initially promoted PyTorch decoders:

- EEGNet;
- temporal CNN;
- LSTM;
- temporal Transformer;
- EEG-Conformer;
- attention-fusion decoder.

Unknown factory IDs fail closed.

External/foundation-model artifacts can be added only through an explicit serializer/factory contract. V1 never falls back to dynamic imports, pickle, or joblib when a model is unsupported.

## Export

A model must already be trained before promotion.

```python
from neuros.models import (
    EEGNetModel,
    ModelInputContract,
    ModelOutputContract,
    export_model_artifact,
)

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
    output_contract=ModelOutputContract(
        class_labels=("left", "right"),
        probability_semantics="uncalibrated_softmax",
    ),
    git_sha="<40-character-git-sha>",
    training_authority_sha256s=("<training-authority-sha256>",),
    evaluation_authority_sha256s=("<evaluation-authority-sha256>",),
    preprocessing_state_sha256s=("<preprocessing-state-sha256>",),
    scientific_study_sha256="<scientific-study-sha256>",
)
```

Export is atomic: neurOS writes into a sibling temporary directory, verifies its own output through the public verifier, and only then renames the verified directory into the requested location.

An existing destination is rejected rather than overwritten.

## Verify versus load

These are intentionally different operations.

### Envelope verification

```python
from neuros.models import verify_model_artifact

manifest = verify_model_artifact("artifacts/eegnet-session-transfer")
```

CLI:

```bash
neuros-models artifact verify artifacts/eegnet-session-transfer --json
```

Verification checks the strict manifest and exact weight-file identity without constructing the decoder. It establishes bundle integrity, not executable compatibility, scientific validity, or publisher trust.

### Reconstruction/load verification

```python
from neuros.models import load_model_artifact

model = load_model_artifact("artifacts/eegnet-session-transfer", device="cpu")
output = model.infer(X_float32)
```

Loading additionally performs:

1. strict artifact-envelope verification;
2. exact package-version verification;
3. safe built-in factory lookup;
4. model reconstruction from canonical configuration;
5. current-vs-artifact interpretability SHA check;
6. output-contract/factory compatibility check;
7. safetensors parsing;
8. exact tensor-name comparison;
9. exact tensor shape/dtype comparison;
10. strict state-dict load;
11. evaluation-mode, gradient-disabled deployment state;
12. return of a read-only public artifact-backed decoder.

## Content-addressed store and rollback

`ModelArtifactStore` separates historical objects from deployment selection:

```text
store/
├── artifacts/
│   ├── <artifact-sha-A>/
│   └── <artifact-sha-B>/
└── refs/
    └── active.json
```

```python
from neuros.models import ModelArtifactStore

store = ModelArtifactStore("model-store")
a = store.publish("exports/model-a")
b = store.publish("exports/model-b")
store.activate("active", b.artifact_sha256)
store.rollback("active", a.artifact_sha256)
```

Rollback changes only the reference. Historical model bytes and provenance are not rewritten through the store API.

The store rejects path-traversal ref names, SHA-shaped ambiguous refs, symlinked refs, symlinked content addresses, and misaddressed artifact directories.

Direct content-identity loading remains available:

```python
model = store.load(a.artifact_sha256)
```

CLI ref resolution:

```bash
neuros-models artifact resolve model-store active --json
```

## Scientific Authority integration

Model Artifact v1 binds runtime model state to Scientific Authority v2 without making `neuros-models` depend on ORION implementation classes.

The manifest stores full SHA references:

- `training_authority_sha256s`;
- `evaluation_authority_sha256s`;
- `preprocessing_state_sha256s`;
- `calibration_state_sha256s`;
- `scientific_study_sha256`.

The evidence layer remains responsible for verifying what those authorities mean. A syntactically valid 64-character value does not become scientific evidence merely because it appears in an artifact.

```text
Scientific Authority / Evidence
          |
          | full SHA references
          v
Model Artifact v1
          |
          v
trusted factory + inert tensor state
          |
          v
runtime inference
```

## Environment identity

A promoted artifact records exact versions for the reconstruction-sensitive runtime, including `neuros-models`, `neuros-core`, NumPy, PyTorch, and safetensors.

V1 requires exact matches. This is intentionally conservative for scientific reproduction. Future operational compatibility policies may safely widen runtime ranges, but they must not erase the exact environment in which the artifact was promoted/qualified.

Git SHA remains provenance rather than a substitute for package identity. Qualification bundles should preserve Git-to-wheel mappings.

Python version, operating system, accelerator, CUDA/cuDNN, and hardware qualification belong in the broader runtime/evidence environment when they materially affect a claim. They are not silently inferred from package versions.

## Threat model

Model Artifact v1 protects against:

- arbitrary pickle execution in the promoted path;
- artifact-provided arbitrary Python imports;
- bit-level weight corruption;
- manifest-field tampering;
- duplicate-key/non-finite JSON ambiguity;
- unbound extra files and artifact/ref symlinks;
- stale/forged derived hashes;
- tensor name/shape/dtype mismatch;
- hidden array dtype/shape coercion;
- accidental overwrite through the artifact API;
- rollback by mutating historical model bytes;
- public mechanistic inspection mutating authoritative deployment state;
- raw softmax scores being relabeled as calibrated probability without a qualified calibration contract.

### Integrity is not authenticity

SHA-256 gives content identity and integrity relative to an expected hash. It does **not** tell us who produced the artifact, whether the publisher was authorized, or whether the source code was trustworthy.

Publisher authentication, signed attestations, transparency logs, remote registry authorization, and organization-level promotion policy are separate supply-chain controls. A future production registry should likely add Sigstore/DSSE-style attestation rather than overloading the artifact content hash with trust semantics.

### Integrity is not hostile-host protection

The API is content-addressed and non-overwriting, but a privileged process on the host can still edit files or private Python object state. Verification detects changed artifact bytes at the next authority boundary. Model Artifact v1 does not claim resistance to a fully compromised operating system or Python process.

### Verification has a time boundary

Envelope verification and subsequent file loading are separate filesystem operations. High-assurance deployment on an adversarial writable filesystem would require a stronger host/storage policy or descriptor-based immutable file handling. V1's supported threat model is a trusted runtime host processing potentially untrusted artifact contents.

## Unsupported promotion paths

V1 intentionally does not use joblib/pickle for sklearn models. Classical estimators remain usable research/runtime models but are not promoted until a safe serializer is defined.

Foundation models, external plugins, quantized runtimes, ONNX, CoreML, TensorRT, and device-specific compiled engines need explicit adapters and qualification. They must not be smuggled into v1 as opaque blobs.

## Evidence boundary

Model Artifact v1 can establish software evidence for:

- exact model-weight identity;
- exact manifest/provenance identity;
- safe built-in reconstruction semantics;
- explicit input and output semantics;
- detached mechanistic inspection;
- deterministic same-environment reload equivalence where qualified;
- content-addressed rollback behavior.

It does not itself establish:

- decoder superiority;
- physiological validity;
- generalization to unseen participants/sessions/devices;
- probability calibration quality;
- publisher authenticity;
- resistance to a compromised host;
- physical device timing/reliability;
- online closed-loop benefit;
- participant safety or benefit;
- clinical validity or safety.

Those claims require their own Scientific Authority, runtime qualification, supply-chain, hardware, human, and clinical evidence planes.
