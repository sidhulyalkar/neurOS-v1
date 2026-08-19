# neurOS Foundation Models

**A model-agnostic interoperability and evaluation layer for the neuroscience foundation-model ecosystem.**

`neuros-foundation` is not another collection of copied Transformer implementations. Its job is to make a fragmented and fast-moving landscape **discoverable, comparable, probeable, and usable inside neurOS** without erasing the important differences between EEG, intracortical spikes, calcium imaging, iEEG, fMRI, and multimodal neural models.

The package is organized around one workflow:

```text
Discover  ->  Inspect capabilities  ->  Load a verified adapter
    ->  Extract representations  ->  Probe geometry/robustness
    ->  Benchmark under one protocol  ->  Integrate with neurOS
```

The key rule is simple: **metadata can be universal; preprocessing and model execution cannot.** A 256 Hz arbitrary-montage EEG reconstruction model and an autoregressive intracortical spike decoder should share a catalog and evaluation vocabulary, not a fake common forward pass.

## Why this package exists

Neural foundation models now differ along several independent axes:

- recording modality: EEG, MEG, ECoG/SEEG, spikes, LFP, calcium, fMRI, multimodal data;
- spatial identity: fixed channel names, unit IDs, learned session embeddings, 3D electrode coordinates, topology-agnostic latent queries;
- temporal tokenization: waveform patches, spectral patches, event/spike tokens, latent states, discrete neural tokens;
- objectives: masked reconstruction, denoising, autoregression, contrastive learning, multimodal alignment, diffusion/flow reconstruction, supervised multi-task decoding;
- adaptation: zero-shot inference, linear probing, few-shot calibration, parameter-efficient tuning, full fine-tuning, continual adaptation;
- openness: open weights, gated weights, code-only research, cloud APIs, and closed commercial systems.

Those differences make a flat `ModelA().predict(X)` abstraction scientifically misleading. `neuros-foundation` instead provides a **capability graph** and a **protocol-first benchmark layer**.

## What changed in v2.1

### 1. Curated cross-modality model catalog

The registry includes representative academic, open-source, neurOS-native, and industry systems, including:

| Area | Representative entries |
|---|---|
| Modern EEG | ZUNA1.1, REVE, NeurIPT, LUNA, CSBrain, EEG-X, EEGPT, LaBraM, CBraMod |
| EEG + language | NeuroLM |
| Commercial / industry EEG | Emotiv Axon, NeuroDX MANAS-1, Zyphra ZUNA1.1 |
| Intracortical population models | NDT3, POYO+, NeuroFM-X |
| Intracranial EEG | Brant, MVPFormer |
| fMRI | BrainLM |
| Cross-session representation learning | CEBRA |
| Multimodal neural generation | Neuroformer |

Each `FoundationModelCard` captures modality, tasks, architecture family, pretraining objective, spatial assumptions, transfer regimes, access level, integration maturity, source links, and important caveats.

The catalog is deliberately conservative. Unknown values stay unknown rather than being guessed from secondary summaries.

### 2. Fail-closed adapters

The previous package contained placeholder wrappers whose missing-backend paths could return random arrays or zeros. Those legacy classes remain importable for compatibility, but **the new registry never treats placeholder output as a model result**.

```python
from neuros.foundation_models import DEFAULT_REGISTRY

status = DEFAULT_REGISTRY.availability("zuna-1.1")
print(status)

# Raises a clear AdapterUnavailableError if the upstream package is absent.
adapter = DEFAULT_REGISTRY.adapter("zuna-1.1")
```

A model can be useful in the catalog even if neurOS cannot execute it. This distinction is essential for closed industry models and newly published research.

### 3. Universal representation probes

The package now exposes dependency-light probes that operate on embedding matrices:

- effective representation rank;
- mean pairwise cosine / anisotropy;
- linear CKA between model spaces;
- aligned cross-session / perturbation invariance;
- deterministic ridge linear probes;
- subject/site/device leakage probes;
- representation health reports;
- pairwise model similarity tables;
- downstream sample-efficiency curves.

These probes provide a common vocabulary without pretending the raw data formats are interchangeable.

### 4. Protocol-first benchmarking

A leaderboard number is uninterpretable without knowing *how* it was obtained. Every benchmark can carry an `EvaluationProtocol` describing:

- split unit: subject, session, site, device, recording, or sample;
- transfer regime: zero-shot, linear probe, few-shot, PEFT, full fine-tune, reconstruction;
- pooling policy;
- preprocessing policy;
- leakage controls;
- random seed;
- a stable protocol fingerprint.

```python
from neuros.foundation_models import EvaluationProtocol, benchmark_embeddings

protocol = EvaluationProtocol(
    name="cross-subject-motor-imagery-v1",
    split_unit="subject",
    transfer_regime="linear_probe",
    pooling="mean of model-recommended token embeddings",
    preprocessing="per-model upstream preprocessing; downstream split held fixed",
)

report = benchmark_embeddings(
    train_embeddings={"model-a": z_a_train, "model-b": z_b_train},
    test_embeddings={"model-a": z_a_test, "model-b": z_b_test},
    train_targets=y_train,
    test_targets=y_test,
    train_domains=subject_train,
    test_domains=subject_test,
    protocol=protocol,
)

print(report.to_json())
```

### 5. neurOS-native bridge

Any trustworthy encoder that returns an embedding matrix can be attached to a transparent linear readout and used through the standard `neuros.models.BaseModel` contract:

```python
from neuros.foundation_models import FoundationEmbeddingDecoder

model = FoundationEmbeddingDecoder(
    encoder=my_foundation_encoder,
    task="classification",
    model_id="reve-base",
)
model.train(X_train, y_train)
output = model.infer(X_test[:1])
```

This is intentionally a small bridge. Foundation-model preprocessing should live before the decoder node in a `RuntimeGraph`, or be performed offline, rather than silently passing raw EEG through the default band-power `Pipeline` transform.

## Installation

### Lightweight discovery + probes

```bash
pip install neuros-foundation
```

PyTorch is no longer required just to inspect the landscape or compare precomputed embeddings.

### Optional upstream integrations

```bash
pip install "neuros-foundation[zuna]"
pip install "neuros-foundation[cebra]"
pip install "neuros-foundation[eeg]"
pip install "neuros-foundation[all]"
```

For `neuros-neurofm` development inside this monorepo, install the package from `packages/neuros-neurofm` in the same environment. The registry detects it without making the public `neuros-foundation` package depend on an experimental model package.

## Discover the landscape

### Python

```python
from neuros.foundation_models import DEFAULT_REGISTRY

for card in DEFAULT_REGISTRY.filter(modality="eeg", min_year=2025):
    print(card.id, card.name, card.access.value, card.integration.value)

for card in DEFAULT_REGISTRY.filter(task="reconstruction"):
    print(card.id, card.pretraining_objective)

rows = DEFAULT_REGISTRY.compare(["zuna-1.1", "reve-base", "neuript", "luna-eeg"])
```

### CLI

```bash
neuros-foundation list
neuros-foundation list --modality eeg --min-year 2025
neuros-foundation list --task reconstruction
neuros-foundation show zuna-1.1
neuros-foundation compare zuna-1.1 reve-base neuript luna-eeg
neuros-foundation doctor
```

`doctor` checks only registered execution adapters. Catalog-only models are not reported as broken installations.

## ZUNA1.1: verified upstream adapter

ZUNA1.1 is integrated by delegating directly to Zyphra's `zuna` package. neurOS does not copy the architecture or checkpoint loader.

```python
from neuros.foundation_models import DEFAULT_REGISTRY

zuna = DEFAULT_REGISTRY.adapter("zuna-1.1")
zuna.reconstruct_fif(
    input_dir="recordings/input",
    output_dir="recordings/output",
    figures_dir="recordings/figures",
    montage="standard_1020",
    gpu_device="",  # CPU; use a GPU id when configured upstream
)
```

**Scientific guardrail:** reconstructed EEG is model-imputed data. It must remain distinguishable from measured samples and must not be treated as ground truth or a clinical measurement.

## NeuroFM-X: integration with `neuros-neurofm`

`neuros-neurofm` remains the neurOS-native research package for developing NeuroFM-X. `neuros-foundation` provides the discovery and interoperability boundary:

```python
from neuros.foundation_models import DEFAULT_REGISTRY

adapter = DEFAULT_REGISTRY.adapter("neuros-neurofmx")
model = adapter.load(...your_neurofm_configuration...)
raw_model = adapter.raw_model
```

That separation gives the monorepo a clear architecture:

```text
neuros-drivers / datasets
          |
          v
      neuros-core  <-------------------------------+
          |                                         |
          v                                         |
     neuros-models                                  |
          |                                         |
          +----------> neuros-foundation            |
          |             catalog / adapters          |
          |             probes / benchmarks         |
          |                    |                    |
          |                    +--> external FMs    |
          |                    |    ZUNA, REVE ...  |
          |                    |                    |
          |                    +--> neuros-neurofm -+
          |                         NeuroFM-X R&D
          |
          +----------> neuros-mechint
                         representation / circuit analysis
```

A useful next integration is for `neuros-mechint` to consume the same embedding/probe reports, so mechanistic analyses and downstream benchmark evidence refer to the same model IDs and protocol fingerprints.

## What should be compared across models?

A useful universal comparison is not “which Transformer is largest?” It is a structured set of questions.

### Representation

- Does the representation retain useful dimensionality?
- Is it collapsed or strongly anisotropic?
- Which task-relevant variables are linearly accessible?
- At what layer/token pooling strategy do they emerge?
- Do different models converge on similar geometry (CKA/RSA)?

### Generalization

- new subject;
- new session;
- new site;
- new device or montage;
- missing/noisy channels;
- sampling-rate shift;
- brain-region shift;
- species shift for population models;
- task and label shift.

### Adaptation cost

- zero-shot performance;
- linear-probe performance;
- few-shot sample efficiency;
- parameter-efficient adaptation;
- full fine-tuning;
- calibration time for an online BCI.

### Deployment

- checkpoint/license availability;
- required preprocessing;
- memory and latency;
- channel-count assumptions;
- streaming support;
- deterministic behavior;
- uncertainty/calibration;
- failure behavior under missing inputs.

### Scientific validity

- subject/session leakage;
- site/device identity leakage;
- test-time preprocessing leakage;
- whether pooling discards token-level information;
- whether generative reconstruction is confused with measurement;
- whether a commercial claim is independently reproducible.

## Recommended benchmark ladder

For serious cross-model comparisons, use a ladder rather than a single aggregate score:

1. **Input contract audit**: channel geometry, sampling rate, sequence length, masking assumptions.
2. **Frozen representation health**: finite values, rank, anisotropy, layer/token pooling.
3. **Frozen downstream probe**: one fixed linear probe and identical split.
4. **Domain leakage probe**: subject, session, site, and device decodability.
5. **Robustness sweep**: channel drop, regional drop, noise, temporal masking, montage perturbation.
6. **Sample-efficiency curve**: 1%, 5%, 10%, 25%, 50%, 100% of downstream labels.
7. **Full adaptation**: fine-tuning under a fixed compute budget.
8. **Deployment audit**: latency, memory, determinism, missing-channel behavior.
9. **neurOS runtime test**: real driver -> model-ready transform -> decoder -> monitoring.

This is much harder to game than a single accuracy number and much more informative for BCI engineering.

## Extending the registry

Third-party integrations can register a model card and an adapter without modifying the universal probes:

```python
from neuros.foundation_models import (
    CallableAdapter,
    FoundationModelCard,
    ModelRegistry,
    NeuralModality,
    ModelTask,
)

registry = ModelRegistry()
registry.register_card(
    FoundationModelCard(
        id="my-eeg-model",
        name="My EEG Model",
        organization="My Lab",
        year=2026,
        modalities=(NeuralModality.EEG,),
        tasks=(ModelTask.REPRESENTATION,),
        architecture="...",
        pretraining_objective="...",
        input_geometry="...",
    )
)
registry.register_adapter(
    CallableAdapter(
        "my-eeg-model",
        package="my_model_package",
        capabilities=("encode",),
        operations={"encode": my_encode_function},
    )
)
```

The long-term goal is for model authors to ship their own thin adapter while neurOS owns the stable schema, evaluation protocol, and probes.

## Legacy wrappers

The following imports are preserved for compatibility:

```python
from neuros.foundation_models import (
    POYOModel,
    POYOPlusModel,
    NDT2Model,
    NDT3Model,
    CEBRAModel,
    NeuroformerModel,
)
```

They predate this registry architecture. Several contain placeholder/model-simulation paths and therefore **must not be interpreted as faithful upstream implementations or used to reproduce published model performance**. New scientific work should use verified upstream adapters or catalog entries until a faithful adapter exists.

This explicit distinction is preferable to silently returning random embeddings that look numerically plausible.

## Design principles

1. **Upstream is execution truth.** Prefer thin adapters to architecture forks.
2. **No fake outputs.** Missing model dependencies fail closed.
3. **Modality-specific inputs, universal evidence.** Standardize reports, not raw biology.
4. **Protocols are first-class artifacts.** A score always carries its evaluation context.
5. **Commercial and academic evidence stay distinguishable.** Company claims can be cataloged without being treated as reproduced results.
6. **Reconstruction is not measurement.** Generative outputs retain provenance.
7. **Lightweight by default.** Browsing the landscape should not require CUDA or a 10 GB install.
8. **neurOS-native models are peers, not special cases.** NeuroFM-X should be benchmarked under the same evidence contracts as external models.

## Roadmap

The architecture is designed for the next additions without destabilizing the public API:

- faithful REVE / EEG-X / LUNA / NDT3 adapters as their upstream inference APIs stabilize;
- layer-wise and token-wise probe extraction;
- standardized perturbation suites for EEG montage/noise and population-neural drift;
- model-card provenance checks and automated release-date refreshes;
- DANDI / NWB / MNE benchmark dataset adapters;
- `neuros-mechint` hooks for circuit-level representation analysis;
- runtime latency/memory benchmarking through `neuros-core`;
- benchmark result manifests suitable for reproducible public leaderboards.

## License

`neuros-foundation` is MIT licensed. Individual upstream models, datasets, checkpoints, and commercial APIs retain their own licenses and terms. Always check the model card and upstream source before redistributing weights or using a model in clinical/commercial settings.
