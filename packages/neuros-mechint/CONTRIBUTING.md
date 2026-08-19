# Contributing to neuros-mechint

`neuros-mechint` treats mechanistic interpretability as experimental science. Contributions are evaluated not only for code quality, but also for whether the implementation makes a clear, falsifiable scientific object.

## Before adding a method

State:

1. **What object does this method estimate?**
2. **What intervention or measurement supports that estimate?**
3. **What result would falsify the claim?**
4. **What matched control is required?**
5. **What is the independent scientific unit of the claim?**
6. **What evidence tier does the current implementation actually reach?**
7. **What method maturity should it receive?**

Do not attach a canonical method name to an approximation unless the approximation is explicitly qualified.

## Maturity is not evidence strength

Method maturity describes the maintained implementation/claim surface:

- Stable
- Integrated
- Research
- Experimental
- Deprecated

Evidence strength describes a particular run. A Stable API used on one toy example remains low evidence; real data does not automatically make an Experimental algorithm Stable.

Use:

```bash
neuros-mechint evidence
neuros-mechint methods
```

## Required scientific habits

### Candidate discovery and evidence must be separated

If an algorithm searches for a mechanism or correspondence using examples, those examples belong to discovery.

A stronger claim freezes the candidate, mapping, projector choice, and learned perturbation donors before testing on held-out examples unavailable to discovery.

Do not tune intervention donors, thresholds, mapping regularization, candidate size, feature sets, dose grids, or projector choices on the validation set.

### Negative results are first-class

Keep:

- held-out failures;
- invalid perturbation normalizations;
- failed random-control thresholds;
- non-estimable comparative contrasts;
- missing factorial cells;
- high-similarity but causally rejected correspondence candidates;
- failed shuffled-pair correspondence controls;
- non-estimable replication analyses;
- estimable but sign-inconsistent replication results;
- confidence intervals crossing the preregistered null;
- non-monotonic dose responses.

A scientific API should be able to say:

```text
no evidence
not estimable
similar but not causal
causal within one unit but not replicated
```

### Match the control to the claim

Examples:

- localization → known nuisance components;
- circuit faithfulness → same-cardinality random circuits;
- causal discovery → simple same-size baseline;
- token intervention → temporal permutation or another matched token control;
- SAE feature intervention → reconstruction baseline;
- architecture/tokenizer comparison → matched information, capacity, compute, task performance, split semantics, and evidence protocol;
- feature correspondence → source ablation + target ablation + shuffled semantic-pair donors + same-cardinality random source mappings;
- higher-level replication → independent units at the scientific claim axis;
- endpoint substitution → preregistered dose response when a graded intervention is meaningful;
- in-manifold replacement claim → explicit donor/generator construction and provenance.

### Do not count correlated observations as independent replication

The resampling unit follows the scientific claim.

Examples:

```text
300 trials from one model seed = 1 model seed
20 sessions from one subject   = 1 subject
4 independently trained seeds  = 4 model-seed units
```

Examples from one prompt, session, subject, model seed, SAE dictionary, or projector cannot silently become independent higher-level replications merely because many perturbations were run.

## v0.7 comparative-evidence rule

A new architecture, tokenizer, checkpoint, or other factorial comparison defines its **estimability conditions before inspecting the requested effect**.

For a primary architecture/tokenizer comparison, explicitly declare:

- every intended factorial cell;
- missing cells and reasons;
- model/tokenizer/dataset revisions;
- semantic discovery and validation partition IDs;
- training seed;
- checkpoint and maturity rule;
- metric;
- discovery method;
- target universe;
- token budget;
- temporal resolution;
- downstream capacity;
- training compute;
- additional matched covariates;
- task-performance tolerance;
- evidence protocol;
- preregistered primary contrasts.

If one of these dimensions violates the declared comparison policy, mark the primary effect non-estimable rather than reporting a number with a warning attached.

## v0.8 causal-correspondence rule

Do not use “correspondence” as a synonym for correlation, CKA, semantic-label agreement, or predictive transfer.

A primary causal-correspondence study declares:

- immutable source and target feature-space identities;
- source and target feature sets;
- one-to-one, one-to-many, or subspace shape;
- every source/target context difference;
- semantic discovery/validation trial partitions;
- mapping discovery method and regularization;
- feature projector and feature-axis semantics;
- source/target scalar metrics and direction;
- source ablation;
- target ablation;
- mapped substitution;
- shuffled semantic-pair control;
- same-cardinality random-source controls;
- random-control budget and seed;
- causal-relevance and recovery thresholds;
- optional upstream factorial provenance.

A source feature that predicts the target accurately but has no source-ablation effect receives no causal credit.

Random source controls should get their own discovery-only map to the same target feature set. Do not compare a fit candidate against unfit random coordinates.

For SAE-scale spaces, sample controls without materializing the entire combinatorial feature universe.

## v0.9 replication rule

A replication contribution must declare **which scientific unit is independent for the claim** before aggregating the source results.

For a v0.9 study, explicitly declare:

- replication family ID;
- primary metric;
- claim axis;
- active hierarchy;
- null value;
- expected direction, if directional;
- minimum independent-unit count;
- confidence level and hierarchical-bootstrap budget;
- minimum independent-unit sign agreement;
- minimum estimable-source fraction;
- minimum absolute effect;
- whether the confidence interval must exclude the null;
- source study fingerprints;
- model-training seeds;
- subject/session identities where relevant;
- dictionary/projector identities where relevant;
- dataset identity when a dataset-level claim is made.

### Claim axis discipline

Choose the axis from the claim, not from whichever axis has the largest sample count.

Examples:

- “stable across trials” → trial;
- “stable across sessions” → session;
- “subject-general” → subject;
- “architecture-level” → independent model-training seed;
- “dictionary robust” → independent dictionary conditions;
- “cross-dataset” → dataset.

Do not relabel a within-seed confidence interval as architecture uncertainty.

### Preserve source estimability

`observation_from_factorial_contrast(...)` preserves non-estimable v0.7 contrasts. Hierarchical aggregation must never repair a confounded design by averaging it with valid designs.

A failed v0.8 correspondence can be a valid negative replica. Preserve it rather than filtering to promoted correspondences only.

### Dose-response discipline

When an intervention is meaningfully graded, preregister:

- dose grid;
- independent units;
- expected direction;
- endpoint criterion;
- monotonicity criterion;
- common-grid policy;
- intervention-manifold assumption.

Do not choose a dose grid after seeing the response curve.

### Manifold discipline

If an intervention uses empirical, nearest-neighbor, quantile, conditional, generative, causal-scrubbing-style, or custom replacement values, record the donor/generator semantics.

Donor-based methods must identify the donor pool. Conditional and generative donors must identify the partition used to fit the donor model.

Do not call an intervention “in manifold” merely because its replacement vector has plausible magnitude.

## Tests

Maintained package tests are named:

```text
test_mechint_*.py
```

A maintained test should:

- execute against the current public API;
- have a clear scientific expected result;
- include relevant controls;
- use deterministic randomness;
- avoid unnecessary network/model downloads;
- state skip behavior for optional dependencies.

Where possible, include both:

1. a positive known-ground-truth case; and
2. a negative/confounded/pseudoreplicated case the method must reject.

Current scientific CLI gates:

```bash
neuros-mechint ground-truth --json
neuros-mechint shared-computation-ground-truth --json
neuros-mechint mechanism-emergence-ground-truth --json
neuros-mechint circuit-faithfulness-ground-truth --json
neuros-mechint evidence-pack-generalization-ground-truth --json
neuros-mechint factorial-ground-truth --json
neuros-mechint correspondence-ground-truth --json
neuros-mechint replication-ground-truth --json
```

The correspondence gate rejects a nearly perfectly predictive, semantically matched but causally unused decoy.

The replication gate rejects hundreds of strong trials from one seed as architecture/model-seed replication and rejects a four-seed 50/50 sign disagreement.

## Optional integrations

Stable package import should not eagerly require large external interpretability stacks.

Keep optional dependencies behind adapter/integration modules and extras where practical.

Protocol-faithful CPU fixtures can test adapter semantics; separate CI jobs can test real package solver/import compatibility.

Avoid making routine PR checks dependent on downloading pretrained checkpoints.

## Tutorials

Maintained teaching notebooks live under:

```text
tutorials/mechint/
```

A maintained notebook should teach:

```text
question
→ hypothesis
→ measurement
→ intervention
→ matched control
→ falsification
→ held-out validation
→ conclusion + uncertainty
```

Comparative notebooks additionally teach:

```text
cell-level evidence
→ preregistered contrast
→ estimability audit
→ effect or rejection
```

Correspondence notebooks additionally teach:

```text
statistical alignment
→ freeze mapping
→ source necessity
→ target necessity
→ held-out substitution
→ shuffled/random controls
→ causal promotion or rejection
```

Replication notebooks additionally teach:

```text
declare claim axis
→ preserve source estimability
→ balance lower hierarchy levels
→ hierarchical uncertainty
→ independent-unit sign agreement
→ replicated OR rejected
→ dose/manifold robustness
```

Exploratory notebooks should normally live under `experiments/` until maintained against current APIs.

## Artifacts and provenance

Prefer self-checking, versioned artifacts for real studies.

```text
experiments/mechint/evidence_packs/
experiments/mechint/factorial_studies/
experiments/mechint/correspondence_studies/
experiments/mechint/replication_studies/
```

Pin immutable model, tokenizer, dataset, checkpoint, SAE/dictionary, projector, and transcoder revisions before publication. Do not rely on mutable aliases as the sole provenance record.

## Historical modules

The package retains a broad exploratory pre-v0.2 surface. Historical code is useful provenance, but code existence is not evidence of current Stable maturity.

Promote historical methods one at a time by defining their scientific object, controls, known-ground-truth benchmark, dependencies, maintained tests, and independent-unit semantics where the claim involves replication.
