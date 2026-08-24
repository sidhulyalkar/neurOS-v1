# neurOS Models

**Task-specific neural decoders with first-class mechanistic analysis contracts.**

`neuros-models` is the model layer between neurOS signal/runtime contracts and the broader research ecosystem. It provides honest classical baselines, inspectable PyTorch neural decoders, structured inference, stable representation extraction, and a narrow bridge into `neuros-mechint` for causal experiments.

The core design rule is:

> A model name must describe the algorithm it actually runs, and an explanation must identify a testable mechanism rather than merely visualize an activation.

## Why this package exists

neurOS separates four concerns that are often mixed together:

```text
neuros-core          acquisition/runtime/contracts
       |
       v
neuros-models        task-specific decoders + stable analysis surfaces
       | \
       |  +------------------------+
       v                           v
neuros-mechint                neuros-sourceweigher
causal mechanism studies      source/domain reliability
       |
       +------------+
                    v
             neuros-foundation
       model/representation ecosystem
```

`neuros-models` does **not** own mechanistic-interpretability algorithms. It owns the model-side contract those algorithms need: a faithful backend, stable component paths, pooled embeddings, logits, and an explicit declaration of what can and cannot be intervened on safely.

## What changed in v2.1

### 1. Removed pseudo-model behavior

Previous releases exposed several misleading fallbacks:

- `TransformerModel` was an sklearn `MLPClassifier` rather than a Transformer.
- `EEGNetModel` was an MLP or generic Conv1D depending on whether TensorFlow happened to be installed.
- `CNNModel` similarly changed identity with the environment.
- `DinoV3Model` substituted torchvision ViT-B/16 ImageNet weights when DINOv3 was unavailable.

Those patterns are removed. Deep models now use one explicit PyTorch backend and fail clearly when the optional dependency is not installed. `DinoV3Model` is a fail-closed deprecated compatibility shim rather than a fabricated foundation-model integration.

### 2. One PyTorch decoder contract

`TorchDecoderModel` standardizes:

- deterministic seeding;
- device selection;
- AdamW training;
- class probabilities;
- logits;
- pooled embeddings;
- input validation;
- training history;
- access to the underlying `torch.nn.Module` for research tooling.

This means the neurOS runtime receives the same structured `DecoderOutput` semantics regardless of which deep architecture is selected.

### 3. Mechanistic analysis manifests

Every inspectable model declares an `InterpretabilityManifest` containing:

- architecture family and backend;
- input axes and output semantics;
- stable `named_modules()` paths;
- the semantic role of each analysis surface;
- tensor-axis descriptions;
- supported operations;
- recommended analyses;
- limitations and claim boundaries.

```python
from neuros.models import EEGConformerModel

model = EEGConformerModel(n_channels=22, n_classes=4)
manifest = model.analysis_manifest()

print(manifest.architecture)
for surface in manifest.surfaces:
    print(surface.path, surface.role, surface.recommended_methods)
```

This turns mechanistic analysis from model-specific archaeology into an explicit API.

### 4. Native `neuros-mechint` bridge

With the optional research dependency installed:

```bash
pip install "neuros-models[mechint]"
```

any inspectable decoder can create a causal tracing adapter:

```python
adapter = model.mechint_adapter()
print(adapter.recommended_paths)
```

The adapter validates that every manifest path actually exists in the model's `named_modules()` graph. A stale model manifest therefore fails immediately rather than producing an analysis of the wrong layer.

### 5. Honest embeddings in runtime output

Deep decoders now expose `encode(X)`. `BaseModel.infer(...)` can place the pooled representation alongside probabilities/logits in the structured `DecoderOutput`.

That creates a shared representation seam for:

- `neuros-foundation` representation probes;
- `neuros-sourceweigher` source/subject similarity;
- `neuros-mechint` linear probes and representational analyses;
- ORION deployment/monitoring decisions.

## Decoder families

Run:

```bash
neuros-models list
neuros-models list --mechint-ready
neuros-models show eeg-conformer
neuros-models doctor
```

### EEGNetModel

A faithful compact EEGNet-style temporal/spatial depthwise-separable convolutional decoder.

Useful mechanistic surfaces:

```text
temporal
    learned temporal filter bank
        |
spatial
    full-montage depthwise electrode projection
        |
separable_pointwise
    learned feature mixing
        |
embedding_pool
    compact latent representation
        |
classifier
```

Good experiments include temporal-frequency response audits, electrode/channel ablations, activation patching, representation probes, and readout attribution.

EEGNet remains a canonical compact baseline rather than being advertised as universally state of the art.

### EEGConformerModel

A modern convolution-first EEG decoder inspired by EEG-Conformer: temporal filtering and full-electrode spatial projection create a token sequence, then Transformer blocks integrate long-range context.

```text
raw EEG
  |
  v
patch_embedding.temporal      learned temporal filters
  |
patch_embedding.spatial       electrode/montage projection
  |
patch_embedding.projection    EEG token sequence
  |
  +--> encoder.layers.0.self_attn
  +--> encoder.layers.0.linear1
  +--> ...
  +--> encoder.layers.N.*
  |
embedding_norm
  |
classifier
```

This is the showcase architecture for the mechanistic workflow because hypotheses can be localized at several scales:

1. temporal filters;
2. electrode projections;
3. token-level representations;
4. Transformer attention/MLP computations;
5. pooled representations;
6. decision readout.

### TransformerModel / TemporalTransformerModel

A real temporal Transformer encoder. The old MLP placeholder is gone.

Inputs are `(batch, channels, time)`. Channel vectors are tokenized at each time point and contextualized with self-attention.

The MLP expansion modules (`encoder.layers.*.linear1`) are intentionally exposed as stable tensor-output hook points suitable for activation patching, sparse autoencoders, transcoders, and circuit discovery.

### CNNModel

A residual dilated temporal CNN with explicit block-level hook points. It is a useful efficient baseline for comparing whether attention actually provides value beyond a strong convolutional receptive field.

### LSTMModel

A recurrent baseline with an inspectable final hidden representation. The raw PyTorch `nn.LSTM` output is structured, so generic tensor-replacement experiments should target `embedding_norm` or use selector-aware tooling.

### AttentionFusionModel

A true trainable sample-dependent modality gating model. Previous releases initialized projections randomly and trained the final classifier using uniform fusion, so the advertised attention was not actually learned end to end.

The new model learns:

```text
EEG ----- projection --+
EMG ----- projection --+--> shared gate --> weighted latent --> classifier
IMU ----- projection --+
```

`get_attention_weights(X)` exposes the learned routing weights, but the documentation deliberately does not call these weights an explanation. To establish modality dependence, suppress or patch modalities and measure the downstream effect.

## Mechanistic interpretability workflow

The recommended neurOS evidence ladder is:

```text
1. performance baseline
2. activation/representation characterization
3. candidate mechanism nomination
4. causal intervention
5. necessity + sufficiency
6. held-out validation
7. robustness / counterfactual transfer
8. evidence pack + model fingerprint
```

### Capture activations

```python
import torch

adapter = model.mechint_adapter()
backend = model.analysis_model()
backend.eval()

x = torch.as_tensor(X[:8], dtype=torch.float32, device=next(backend.parameters()).device)
activations = adapter.capture_outputs(
    x,
    ["patch_embedding.temporal", "encoder.layers.0.linear1"],
)
```

### Causal activation replacement

```python
clean = adapter.forward(x)
cache = adapter.capture_outputs(x, ["embedding_norm"])

replacement = torch.zeros_like(cache["embedding_norm"])
ablated = adapter.forward_with_replacements(
    x,
    {"embedding_norm": replacement},
)

causal_effect = (clean - ablated).abs().mean().item()
```

A large causal effect establishes that the component matters for this input/output. It does not automatically establish what concept the component represents.

### Sparse feature dictionaries and transcoders

Modern mechanistic interpretability often decomposes hidden states into learned sparse features. `neuros-mechint` already supports SAE ecosystem adapters and circuit-tracing workflows. The model package exposes stable locations where such dictionaries can be trained and compared.

For Transformer/Conformer models, `encoder.layers.*.linear1` is a natural first target because it sits inside the nonlinear feed-forward computation and returns a simple tensor.

Sparse features are **candidate variables**. neurOS follows them with intervention-based faithfulness tests rather than treating high activation examples as proof of mechanism.

### Circuit tracing

Attribution graphs can nominate computational paths, but a graph is not automatically a causal circuit. The neurOS/mech-int boundary therefore keeps three quantities separate:

```text
attribution strength
causal intervention effect
held-out faithfulness
```

This mirrors the direction of current mechanistic-interpretability research, where attribution/circuit localization, sparse feature decomposition, and causal validation are evaluated as separate problems.

## Scientific claim boundaries

### Attention is not explanation

Attention weights can be useful diagnostics and routing summaries. They do not prove causal importance. Use head/token interventions, activation patching, or other controlled perturbations.

### Saliency is not mechanism

Gradients and integrated-gradient style maps answer local sensitivity questions. They can nominate hypotheses. Strong mechanism claims require interventions and held-out tests.

### Sparse features are not automatically better than neurons

SAE/transcoder features can be useful decompositions, but decomposition quality, interpretability, and causal faithfulness are separate axes. Always compare against simpler neuron/component baselines when possible.

### In-distribution ablations are not enough

For BCI models, mechanisms should be checked across subject, session, device/montage, task, and artifact conditions. A circuit that appears only because a single subject has a stable artifact is not a general neural mechanism.

### Mechanistic model features are not biological homology

A model feature correlated with an EEG rhythm, cortical region, neuron population, or behavior does not imply that the artificial component is biologically homologous to that system. Biological claims need independent neuroscientific validation.

## Installation

### Classical models + contracts

```bash
pip install neuros-models
```

### Inspectable neural decoders

```bash
pip install "neuros-models[pytorch]"
```

### Full mechanistic research path

```bash
pip install "neuros-models[mechint]"
```

### Development

```bash
pip install "neuros-models[dev]"
```

## Quick start

```python
import numpy as np
from neuros.models import EEGConformerModel

rng = np.random.default_rng(0)
X = rng.normal(size=(64, 8, 256)).astype("float32")
y = (X[:, 0, 80:120].mean(axis=1) > 0).astype(int)

model = EEGConformerModel(
    n_channels=8,
    n_classes=2,
    embedding_dim=32,
    n_heads=4,
    n_layers=2,
    n_epochs=3,
)
model.train(X, y)

print(model.predict(X[:4]))
print(model.predict_proba(X[:4]))
print(model.encode(X[:4]).shape)
print(model.infer(X[:1]))
```

## Input contracts

The deep neural-window models use:

```text
(batch, channels, time)
```

They are **not** silently fed the default band-power vector produced by every neurOS runtime pipeline. For raw-window deep decoding, configure the upstream transform/windowing node explicitly.

Feature-vector models use:

```text
(batch, features)
```

`AttentionFusionModel` expects concatenated modality features according to its declared `modality_dims`.

This explicit distinction prevents a frequent BCI failure mode where an architecture is nominally correct but receives semantically incompatible data.

## Package architecture

```text
neuros/models/
├── base_model.py              runtime decoder contract
├── analysis.py                interpretability manifest schema
├── torch_base.py              common PyTorch training/inference behavior
├── catalog.py                 honest decoder/model cards
├── cli.py                     discovery + dependency doctor
├── eegnet_model.py            compact EEG CNN
├── eeg_conformer_model.py     convolutional Transformer EEG decoder
├── transformer_model.py       temporal Transformer
├── cnn_model.py               residual temporal CNN
├── lstm_model.py              recurrent decoder
├── attention_fusion_model.py  learned multimodal gating
├── classical models           sklearn baselines
└── model_registry.py          local artifact registry (legacy pickle format)
```

## Relationship to other neurOS packages

### `neuros-foundation`

Foundation-model adapters own their upstream preprocessing/checkpoint semantics. When they produce embeddings, task-specific readouts can live in `neuros-models`, while the same embeddings can be compared/probed in `neuros-foundation`.

### `neuros-mechint`

`neuros-models` declares surfaces; `neuros-mechint` performs causal experiments. This avoids putting attribution/circuit algorithms into every model class.

### `neuros-sourceweigher`

SourceWeigher can consume `model.encode(X)` representations to determine which source subject/session/device is most relevant to a target. `AttentionFusionModel` solves a different problem: within one sample, which modality representations should be combined for prediction.

### ORION

ORION should consume model capabilities plus evidence generated by `neuros-mechint`: deployment eligibility can eventually depend not only on accuracy/latency but also on whether critical mechanisms are stable across subject/session/device perturbations.

## Model selection philosophy

neurOS should not become a zoo of every published neural architecture. A model belongs here when it contributes at least one of:

- a strong, transparent baseline;
- an important inductive bias for BCI/neural data;
- a materially different deployment trade-off;
- a useful mechanistic test bed;
- a verified external integration that cannot live more cleanly in `neuros-foundation`.

For broader rapidly changing pretrained model coverage, use `neuros-foundation` rather than copying architectures here.

## Security note on the local registry

The historical `ModelRegistry` persists Python objects using pickle. Only load artifacts you trust. Future registry work should migrate inspectable PyTorch models toward config + `state_dict` artifacts and retain the analysis-manifest fingerprint beside the weights.

## References and methodological context

- Lawhern et al., *EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces*, Journal of Neural Engineering (2018).
- Song et al., *EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization*, IEEE TNSRE (2023).
- Anthropic, *Circuit Tracing: Revealing Computational Graphs in Language Models* (2025).
- Mueller et al., *MIB: A Mechanistic Interpretability Benchmark* (2025).
- Paulo et al., *Transcoders Beat Sparse Autoencoders for Interpretability* (2025).

These references motivate methods and evaluation patterns. neurOS does not claim that methods validated on language models transfer automatically to neural decoders; that transfer itself should be tested.

## License

MIT
