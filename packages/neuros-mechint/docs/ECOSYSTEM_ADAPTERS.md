# Ecosystem adapters

`neuros-mechint` integrates external interpretability ecosystems through narrow adapter boundaries so scientific intervention logic does not depend on framework-specific execution details.

The current integrated surfaces include:

- TransformerLens;
- NNsight;
- SAELens;
- circuit-tracer attribution normalization.

These integrations remain important in v0.7, but their role is deliberately bounded:

```text
external tool
    |
 candidate / activation surface
    |
ModelAdapter or feature adapter
    |
causal intervention
    |
quantitative faithfulness
    |
v0.6 held-out evidence pack
    |
v0.7 matched factorial comparison
```

An adapter being Integrated means the package maintains the relevant execution contract. It does **not** mean that every mechanism discovered through the external tool is correct.

## TransformerLens

`TransformerLensAdapter` uses TransformerLens cache/hook semantics for named activation capture and replacement.

Typical use:

- discover or nominate hook-point candidates;
- run necessity/sufficiency through the generic faithfulness API;
- package the frozen result into v0.6 held-out evidence;
- compare matched cells in v0.7 only when the experimental design is estimable.

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

A future v0.8 feature-correspondence study should also freeze SAE/dictionary revisions and discover feature mappings separately from held-out causal-transfer validation.

## circuit-tracer

`CircuitTracerAdapter` normalizes attribution-graph feature identities and direct feature-to-logit attribution summaries.

The key claim boundary remains:

> an attribution graph can nominate a candidate; it is not automatically a causal circuit.

A circuit-tracer candidate must still pass intervention-based faithfulness and held-out evidence before entering a strong comparative claim.

`circuit-tracer` remains upstream-installed rather than hidden behind an unversioned Git dependency.

## Real-package CI

The dedicated mech-int workflow includes real-package dependency/import checks for the published optional extras:

- TransformerLens 3.x;
- NNsight 0.7.x;
- SAELens 6.x.

These jobs validate dependency compatibility and import surfaces. They deliberately do not download pretrained checkpoints during normal PR CI.

## Real-model recipes

List maintained starting points with:

```bash
neuros-mechint evidence-recipes
```

The recipes identify candidate model surfaces and recommended execution environments. They are **not measured evidence**.

Before publishing a study, resolve mutable model/tokenizer/SAE/transcoder names to immutable revisions.

## Using external models in v0.7

For factorial architecture/tokenizer science, produce one v0.6 evidence pack for every observed cell.

The pack metadata should record the factorial design information expected by `run_factorial_evidence_study(...)`:

- architecture;
- checkpoint;
- checkpoint maturity;
- session and subject;
- training seed;
- semantic discovery partition ID;
- semantic validation partition ID;
- token budget;
- temporal resolution;
- downstream capacity;
- training compute;
- additional preregistered matched covariates.

The v0.7 bridge validates those fields rather than trusting filenames or human notes.

## Claim boundary

External integration alone establishes none of the following:

- correct causal localization;
- a faithful circuit;
- held-out mechanism generalization;
- an architecture/tokenizer effect;
- causal feature correspondence;
- biological homology.

Those claims require progressively stronger layers of intervention, held-out validation, matched comparison, and replication.
