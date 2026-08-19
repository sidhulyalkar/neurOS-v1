# Causal feature-correspondence studies

This directory is the repository home for v0.8 feature-correspondence evidence artifacts and study notes.

## What belongs here

A correspondence study should preserve enough information to reconstruct the scientific comparison without committing raw neural recordings, model inputs, or large activation tensors.

Recommended layout:

```text
experiments/mechint/correspondence_studies/
  <study-id>/
    README.md
    correspondence.json
    environment.txt          # optional
    notes.md                 # optional
```

`correspondence.json` should be created by `write_correspondence_artifact(...)`.

## Required study description

The accompanying README should state:

- scientific question;
- source model / tokenizer / representation and immutable revisions;
- target model / tokenizer / representation and immutable revisions;
- dataset and immutable revision;
- subject/session/checkpoint identities;
- discovery and validation semantic partition IDs;
- source and target feature sets;
- correspondence kind: one-to-one, one-to-many, or subspace;
- mapping discovery method;
- feature projector and feature-axis semantics;
- scalar metrics and metric direction;
- intervention family;
- promotion policy;
- random-source control count and seed;
- whether the study was nominated by a v0.7 factorial contrast;
- execution environment / accelerator when relevant.

## Retain every outcome

Do not create this directory only for successful correspondences.

Retain:

- promoted correspondence;
- null substitution;
- high similarity with failed causal transfer;
- source-causal / target-noncausal mismatches;
- target-causal / source-noncausal mismatches;
- shuffled-pair failures;
- random-control failures;
- discovery-to-validation collapse;
- invalid intervention normalizations;
- operationally failed studies when the failure changes what can be claimed.

A negative artifact is useful evidence when the study design and revisions are reproducible.

## Similarity is not the headline

Always report separately:

```text
activation correlation
geometric similarity
semantic-label overlap
held-out predictive R²
source intervention effect
target intervention effect
causal recovery
shuffled-pair causal score
random-source causal score / percentile
```

Do not summarize these as a single correspondence score.

In particular, a feature with high correlation and predictive transfer but zero source-ablation effect is a **non-causal similarity result**, not a successful correspondence.

## Source/target feature geometry

Raw feature indices are local identifiers only.

For SAE studies, record dictionary/model revisions. For checkpoint studies, record the checkpoint. For ORION/NeuroFM studies, document how event/time coordinates were preserved by the feature projector.

The default `TensorFeatureProjector` averages non-feature axes. If the scientific claim involves temporal or token-specific structure, use and document a projector that preserves those coordinates.

## Factorial provenance

If v0.7 nominated the study, the artifact should contain a `FactorialCorrespondenceOrigin` with:

- upstream factorial study fingerprint;
- estimable contrast ID;
- participating factorial cell IDs.

Do not create an origin from a non-estimable factorial contrast.

## Verification

After copying, downloading, or publishing an artifact:

```bash
neuros-mechint verify-correspondence-artifact correspondence.json
```

The verifier checks the artifact schema and full-content integrity hash.
