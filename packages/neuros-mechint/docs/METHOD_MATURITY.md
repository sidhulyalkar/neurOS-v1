# Method maturity in neuros-mechint

`neuros-mechint` separates **implementation maturity** from **scientific evidence strength**.

That distinction is non-negotiable:

- a Stable API can support a weak experiment;
- a Research method can produce real-data evidence without becoming Stable;
- an Integrated adapter can correctly execute an external model without proving the mechanism it is used to study;
- a statistically excellent representation alignment is not automatically causal correspondence;
- a precise within-seed estimate is not automatically higher-level replication.

Inspect executable method cards with:

```bash
neuros-mechint methods
```

## Maturity levels

### Stable

A maintained package contract with defined semantics, deterministic coverage where practical, and explicit scientific limitations.

Current Stable surfaces include:

- module activation patching;
- module-output ablation;
- generic input causal audits;
- causal-map comparison statistics;
- circuit necessity/sufficiency and same-size random controls;
- held-out evidence-pack bookkeeping;
- activation-magnitude discovery baseline;
- factorial mechanism design;
- matched factorial contrasts;
- feature-correspondence design and split/mapping bookkeeping.

Stable does not mean every result produced through the contract is strong evidence.

### Integrated

A maintained bridge into another neurOS layer, evidence layer, or external ecosystem.

Current Integrated surfaces include:

- ORION token and representation causal audits;
- ORION tokenizer studies;
- NeuroFM representation probes;
- TransformerLens, NNsight, and SAELens adapters;
- circuit-tracer attribution normalization;
- external-model evidence recipes;
- evidence-pack → factorial bridge;
- ModelAdapter feature-correspondence execution;
- estimable factorial-contrast → correspondence provenance linking;
- factorial-contrast → v0.9 hierarchical observation bridge.

### Research

A scientifically useful analysis whose intervention assumptions, statistical treatment, or generality require stronger validation.

Current Research surfaces include:

- module-level path patching;
- shared-computation hypothesis generation;
- checkpoint mechanism emergence;
- single-target ablation-effect candidate discovery;
- legacy descriptive cross-session factorial replication summaries;
- held-out cross-model causal feature substitution;
- claim-aware hierarchical replication;
- correspondence replication across independent scientific units;
- intervention dose-response and manifold robustness.

### Experimental

Exploratory algorithms whose implementation exists but whose scientific semantics are not promoted as package contracts.

The ACDC-inspired module-pruning baseline remains Experimental and is not represented as canonical edge-level ACDC.

### Deprecated

An older API retained temporarily for migration but not recommended for new evidence claims.

## Evidence strength is a separate axis

The repository evidence ladder is exposed by:

```bash
neuros-mechint evidence
```

A useful mental model is:

```text
                         evidence strength
                         low ---------------- high
implementation  Stable     API contract       replicated held-out study
maturity        Research   validated method   real research evidence
                Experimental prototype        exploratory real-data result
```

Neither axis substitutes for the other.

## v0.6: discovery is not held-out evidence

A mechanism selected on discovery examples must be frozen before validation.

```text
Research discovery
       ↓
freeze candidate and donors
       ↓
Stable held-out evidence protocol
       ↓
promote OR reject
```

A Stable evidence protocol can validate or reject a Research discovery heuristic without changing that heuristic's maturity.

## v0.7: valid cells are not automatically valid comparisons

A scientifically well-formed cell can still be unusable for an architecture or tokenizer effect if another cell changes a required nuisance dimension.

Stable v0.7 factorial machinery establishes only that:

- intended and missing cells were declared;
- semantic trial partitions and evidence protocols were checked;
- non-varied axes were checked;
- matched covariates were checked;
- task-performance/checkpoint tolerances were enforced;
- target-universe compatibility was checked;
- the preregistered contrast was applied only when estimable;
- rejection reasons were retained otherwise.

It does not make an estimable effect universal beyond the tested design.

## v0.8: similarity is not automatically correspondence

v0.8 introduced the boundary:

> **A valid statistical alignment is not automatically a valid mechanistic correspondence.**

The Stable `feature_correspondence_design` layer establishes:

- explicit source/target feature-space identities;
- declared context differences;
- distinct semantic discovery/validation partitions;
- discovery-only mapping fit;
- frozen coefficients before validation;
- separate activation, geometry, semantic, predictive, and causal quantities;
- fairly fitted same-cardinality random source mappings;
- shuffled semantic-trial donor controls;
- one artifact contract for positive and negative results.

This design layer does not establish causal correspondence by itself.

### `held_out_causal_feature_substitution` — Research

Conditionally establishes:

- source feature relevance under ablation;
- target feature relevance under ablation;
- held-out restoration of target contribution by a frozen mapped source activation;
- superiority to declared shuffled-pair and random-source controls.

It remains Research because replacement may be off manifold, projector choice can change intervention semantics, successful substitution does not establish uniqueness/homology, and a single model pair does not establish higher-level replication.

## v0.9: precision is not automatically replication

v0.9 introduces another evidence boundary:

> **Lower-level repetition can improve precision without adding a single independent unit at the level of the scientific claim.**

A model-seed claim requires independently trained model seeds. A subject claim requires multiple subjects. A dataset claim requires multiple datasets. A dictionary-robust claim requires independent dictionary conditions when dictionary variability is part of the claim.

Hundreds of trials from one seed remain one seed.

## v0.9 method cards

### `claim_aware_hierarchical_replication` — Research

Establishes, conditionally:

- an explicit independent claim axis;
- unit-balanced hierarchical aggregation;
- hierarchical-bootstrap uncertainty under the declared nesting structure;
- between-independent-unit variability;
- independent-unit sign agreement;
- separate estimability and replication decisions.

Required controls:

- preregistered claim axis and null;
- explicit hierarchy coordinates;
- minimum independent-unit count;
- preserved negative and non-estimable source observations;
- no lower-level repetition counted as higher-level replication.

It remains Research because hierarchical bootstrap intervals can be unstable with few higher-level units, independence can be violated by undeclared shared causes, and replication under one hierarchy does not imply transfer to an untested dataset/task/species.

### `hierarchical_factorial_uncertainty` — Integrated

Converts v0.7 contrasts into v0.9 observations while preserving:

- contrast ID;
- scalar effects;
- estimability;
- rejection reasons;
- source factorial-study fingerprint.

The bridge cannot repair a non-estimable design. It only makes the source evidence compatible with the replication layer.

### `correspondence_replication` — Research

Aggregates v0.8 correspondence metrics across declared independent units such as model seeds, subjects, sessions, dictionaries, or datasets.

It remains Research because a replicated causal substitution is still conditional on the feature surface, mapping family, projector, intervention manifold, and task/metric.

### `intervention_dose_response` — Research

Establishes whether an intervention metric changes coherently over a preregistered dose grid while recording the intervention-manifold assumption.

It reports:

- endpoint effect;
- monotonic fraction;
- aggregate response curve;
- normalized area under the response curve;
- pass/rejection reasons.

It remains Research because monotonicity is supportive rather than uniquely mechanistic, and in-manifold validity depends on how the donor or generator was constructed.

## Intervention-manifold maturity boundary

`InterventionManifoldAssumption` makes an intervention claim explicit. Supported labels include zero, mean, empirical donor, nearest neighbor, quantile match, conditional resampling, generative donor, causal-scrubbing-style donor, and custom semantics.

This metadata does **not** certify that an intervention is truly in distribution.

A donor-based claim should additionally establish that:

- the donor pool is scientifically appropriate;
- learned/conditional donors were fit without validation leakage;
- donor identity does not introduce a new confound;
- the projector preserves the coordinates needed by the claim.

## v0.7 replication summary versus v0.9 replication evidence

`FactorialReplicationSummary` is retained as a descriptive Research object. It can report how many contrasts/sessions were observed and whether signs agree.

It is not a substitute for `HierarchicalReplicationResult`.

Only the v0.9 object declares the independent claim axis, balances lower levels, computes hierarchical uncertainty, and distinguishes estimability from replication.

## What a passing v0.9 result does not establish

It does not, by itself, prove:

- independence of units that share an undeclared upstream cause;
- transfer to an untested species, task, dataset, architecture, tokenizer, dictionary, or projector;
- biological homology;
- exact frequentist coverage with only a few higher-level units;
- that a monotonic dose response uniquely identifies the mechanism;
- that a replicated linear feature map is the unique correct correspondence.

Those are stronger claims with stronger experimental requirements.

## Synthetic scientific gates

The maintained gates include positive recovery and designed failure behavior:

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

Important negative cases:

- v0.6 succeeds only when a known discovery-overfit mechanism is rejected on held-out data;
- v0.7 succeeds only when a planted interaction is recovered **and** confounded/missing-cell comparisons are refused;
- v0.8 succeeds only when a real causal correspondence is recovered **and** a highly predictive but causally unused decoy is rejected;
- v0.9 succeeds only when a true four-seed effect is recovered **and** hundreds of trials from one seed plus a sign-inconsistent multi-seed effect are rejected as replication.

A scientific framework needs executable fixtures where the correct answer is “no evidence,” “not estimable,” “similar but not causal,” or “precise but not replicated.”

## Promotion principles

A method should move toward Stable only when:

1. its scientific object is defined;
2. implementation matches the advertised method name;
3. known-mechanism fixtures recover expected answers;
4. negative controls fail in the expected direction;
5. confounds are explicit;
6. provenance and randomness are reproducible;
7. optional dependency boundaries are isolated;
8. the teaching surface states what the method does and does not establish.

Comparative, correspondence, and replication methods additionally need:

9. explicit estimability/compatibility criteria;
10. preserved missing, invalid, null, and negative results;
11. an explicit independent replication unit;
12. discovery/validation separation whenever a candidate or map was fit;
13. controls with comparable fitting opportunity;
14. separate reporting of statistical similarity and causal evidence;
15. higher-level uncertainty that does not count lower-level repetition as independent replication;
16. explicit manifold/projector assumptions when replacement interventions support the claim.

## Historical Phase-2 modules

The repository retains a broad pre-v0.2 exploratory surface. Code existence is not a maturity promotion. Historical methods must independently satisfy current contract and evidence policy before being represented as Stable.

## Next maturity target: v1 evidence closure

The next target is not another similarity or attribution score. It is evidence closure:

- freeze schemas and migrations;
- execute maintained tutorials in evidence CI;
- run a real matched architecture × tokenizer neural-data experiment;
- replicate at least one correspondence across independent model seeds;
- quantify session/subject uncertainty where supported;
- run a real cross-session or cross-dataset causal study;
- add stronger manifold-aware intervention controls;
- independently reproduce at least one artifact family.
