# Ecosystem adapters

`neuros-mechint` integrates model ecosystems through narrow adapter boundaries so scientific intervention logic does not depend on framework-specific execution details.

The integrated surfaces include:

- neurOS task-specific decoders (`NeurOSModelAdapter`);
- ordinary PyTorch modules;
- TransformerLens;
- NNsight;
- SAELens;
- circuit-tracer attribution normalization.

The governing evidence flow is:

```text
model / external tool
        |
candidate activation or feature surface
        |
ModelAdapter
        |
causal intervention
        |
quantitative faithfulness
        |
held-out evidence pack
        |
matched comparison / replication
```

An adapter being integrated means the package maintains the execution contract. It does **not** mean every mechanism discovered through it is correct.

## neurOS models

`NeurOSModelAdapter` is the native bridge from `neuros-models` v2.1+.

A neurOS decoder declares an `InterpretabilityManifest` and exposes its underlying research backend through `analysis_model()`. The adapter validates that every declared path exists in the backend model before running experiments.

```python
from neuros.models import EEGConformerModel

model = EEGConformerModel(n_channels=22, n_classes=4)
adapter = model.mechint_adapter()
print(adapter.recommended_paths)
```

The dependency direction is intentionally duck-typed: `neuros-mechint` does not import or depend on `neuros-models`. This keeps mech-int usable with external models while letting neurOS decoders opt into the common experiment layer.

`recommended_paths` contains only tensor-output hook points safe for the generic PyTorch capture/replacement adapter. Structured modules such as raw `nn.LSTM` or `nn.MultiheadAttention` remain declared in the manifest but require selector-aware tooling for direct replacement.

The most important boundary is semantic:

> a model manifest identifies where an experiment can intervene; it does not certify what a component means.

Meaning and mechanism must be established through analysis plus held-out causal tests.

## TransformerLens

`TransformerLensAdapter` uses TransformerLens cache/hook semantics for named activation capture and replacement.

Typical use:

- discover or nominate hook-point candidates;
- run necessity/sufficiency through the generic faithfulness API;
- package the frozen result into held-out evidence;
- compare matched cells only when the experimental design is estimable.

Optional install:

```bash
pip install -e "packages/neuros-mechint[transformer-lens]"
```

## NNsight

`NNsightAdapter` uses trace-time output capture and assignment.

Structured/tuple outputs require explicit selectors rather than silently replacing an entire module output with one tensor.

Optional install:

```bash
pip install -e "packages/neuros-mechint[nnsight]"
```

## SAELens

`SAELensFeatureAdapter` wraps public SAE encode/decode behavior.

SAE feature interventions first report reconstruction error. Faithfulness is interpreted relative to the reconstruction baseline so the dictionary approximation itself cannot masquerade as feature causality.

Optional install:

```bash
pip install -e "packages/neuros-mechint[sae-lens]"
```

Future feature-correspondence studies should freeze SAE/dictionary revisions and discover feature mappings separately from held-out causal-transfer validation.

## circuit-tracer

`CircuitTracerAdapter` normalizes attribution-graph feature identities and direct feature-to-logit attribution summaries.

The key claim boundary remains:

> an attribution graph can nominate a candidate; it is not automatically a causal circuit.

A circuit-tracer candidate must still pass intervention-based faithfulness and held-out evidence before entering a strong comparative claim.

`circuit-tracer` remains upstream-installed rather than hidden behind an unversioned Git dependency.

## Real-package CI

The dedicated mech-int workflow includes real-package dependency/import checks for published optional extras including TransformerLens, NNsight, and SAELens. neurOS model compatibility is tested in the main monorepo CI against the local `neuros-models` package.

These jobs validate dependency compatibility and import surfaces. They deliberately do not download pretrained checkpoints during normal PR CI.

## Real-model recipes

List maintained starting points with:

```bash
neuros-mechint evidence-recipes
```

The recipes identify candidate model surfaces and recommended execution environments. They are **not measured evidence**.

Before publishing a study, resolve mutable model/tokenizer/SAE/transcoder names to immutable revisions.

## Claim boundary

External or native integration alone establishes none of the following:

- correct causal localization;
- a faithful circuit;
- held-out mechanism generalization;
- an architecture/tokenizer effect;
- causal feature correspondence;
- biological homology.

Those claims require progressively stronger layers of intervention, held-out validation, matched comparison, and replication.
