# neuros-mechint architecture

`neuros-mechint` is the causal experiment and comparative-evidence layer of neurOS.

It is not a parallel runtime, a second provenance system, or a catch-all home for neural analysis. Its narrower job is:

> define, test, compare, replicate, and preserve falsifiable claims about how computation is implemented in artificial and neural-data models.

## Dependency direction

The dependency graph remains one-way:

```text
neuros-core
  stable contracts, provenance, quality
       ^
       |
       +--------------------------+
       |                          |
     ORION                   neuros-mechint
 neural token /             causal experiments,
 representation              held-out evidence,
 contracts                   comparative science
       ^                          ^
       |                          |
       +------ optional ----------+
                integrations

NeuroFM / external model ecosystems
       |
       v
  adapters and probes
       |
       v
  neuros-mechint evidence
```

`neuros-core` and ORION must not import mechanistic-interpretability implementations.

## Scientific layers

The maintained architecture is deliberately layered. A later layer may consume an earlier scientific object, but it must not erase the earlier object's validity conditions.

### Layer 1: causal experiment kernel

`neuros_mechint.core`

Provides:

- typed components;
- counterfactual pairs;
- scalar metrics;
- interventions;
- independent intervention experiments;
- repository-aligned evidence tiers;
- method maturity cards;
- experiment manifests and content hashing.

The key invariant is that interventions are evaluated from a fixed reference state rather than accumulating accidentally across conditions.

### Layer 2: model adapters

`neuros_mechint.adapters`

`ModelAdapter` separates scientific intervention logic from framework-specific execution.

Maintained adapter surfaces include:

- PyTorch;
- TransformerLens;
- NNsight;
- SAELens feature workflows;
- circuit-tracer attribution normalization.

Adapter integration establishes an execution contract. It does not certify a mechanism.

### Layer 3: single-context causal maps

The v0.2-v0.4 benchmark layer provides:

- known-route localization;
- ORION token/representation causal audits;
- cross-context effect-map stability;
- shared-computation comparison;
- tokenizer-specific causal-map comparison;
- checkpoint mechanism-emergence analysis.

These tools answer:

> Which interventions matter in this context, and how stable is that intervention profile across matched contexts?

They do not yet establish sufficiency or held-out generalization.

### Layer 4: quantitative circuit faithfulness

v0.5 adds:

- `CircuitCandidate`;
- necessity;
- sufficiency;
- joint faithfulness;
- same-cardinality random controls;
- SAE reconstruction-aware feature faithfulness.

A candidate mechanism becomes an intervention-tested object rather than an attribution-only object.

### Layer 5: held-out evidence packs

v0.6 creates the discovery-versus-evidence boundary:

```text
discovery examples
       |
       +--> candidate selection
       +--> perturbation donor fitting
       |
       v
     freeze
       |
       v
validation examples
       |
       +--> necessity / sufficiency
       +--> alternative baselines
       +--> same-size controls
       +--> uncertainty
       |
       v
promote OR reject
       |
       v
self-checking EvidencePackResult
```

Important invariants:

- validation examples never enter candidate discovery;
- learned intervention donors are fit on discovery data only;
- discovery/validation input content can be checked for duplication;
- model state can be fingerprinted before and after evaluation;
- invalid perturbations remain visible;
- correlated perturbations from one example are not counted as independent examples;
- positive and negative results share the same artifact contract.

The evidence pack is the cell-level scientific unit consumed by v0.7.

### Layer 6: factorial comparison

v0.7 compares completed evidence cells under a declared matched design:

```text
v0.6 cell A1,T1       v0.6 cell A2,T1
       \                    /
        +--- declared grid +
                 |
          estimability audit
                 |
     +-----------+-----------+
     |           |           |
architecture  tokenizer     A×T
   effect       effect   interaction
```

The central rule is:

> comparative statistics do not bypass cell-level evidence or design validity.

`FactorialCellSpec` declares model/tokenizer/data revisions, session/subject, training seed, checkpoint, semantic discovery/validation partitions, target universe, task/evidence protocol, matched covariates, and explicit missing cells.

`FactorialContrastSpec` supports:

- architecture main effects;
- tokenizer main effects;
- checkpoint contrasts;
- architecture × tokenizer interactions.

For a 2 × 2 slice:

```text
interaction = (A2,T2 - A1,T2) - (A2,T1 - A1,T1)
```

Before an effect is computed, v0.7 checks cell availability, fixed axes, split semantics, evidence protocol, model performance, checkpoint maturity, matched covariates, and target-universe compatibility.

The output is either:

```text
estimable=True + effect
```

or:

```text
estimable=False + explicit reasons
```

There is no fallback that averages a broken design into an effect estimate.

### Layer 7: held-out causal feature correspondence

v0.8 asks whether apparently aligned representations transfer causal contribution.

```text
paired scientific trials
        |
        +--------- discovery ---------+
        |                             |
 source feature space          target feature space
        |                             |
        +---- fit candidate map ------+
                     |
                   FREEZE
                     |
        +---------- validation ----------+
        |                                |
 source feature ablation          target feature ablation
        |                                |
        +---- mapped target substitution-+
                     |
          +----------+-----------+
          |                      |
 shuffled trial donors   random source feature sets
          |                      |
          +----------+-----------+
                     |
             promote OR reject
```

#### Feature-space identity

`FeatureSpaceIdentity` pins model/revision, representation/component, architecture, tokenizer/revision, dataset/revision, session, subject, checkpoint, and ordered feature names.

Raw feature indices remain local identifiers. Every source/target context difference must be declared before mapping fit.

#### Discovery-only mapping

The maintained fitter supports one-to-one, one-to-many, and subspace mappings. It fits a rank-tolerant ridge-linear transformation on discovery trials only and freezes it before validation.

#### Similarity and causality remain separate

v0.8 preserves independently:

```text
activation correlation
linear CKA geometry
semantic-label overlap
held-out predictive R²
source intervention effect
target intervention effect
causal recovery
```

High similarity or predictive transfer does not substitute for causal evidence.

#### Causal substitution and controls

The evaluator measures source clean/ablated and target clean/ablated/substituted metrics. A candidate receives causal credit only when both source and target feature sets are themselves intervention-relevant.

Controls include:

1. shuffled semantic-trial donors;
2. same-cardinality random source feature sets, each fit fairly on discovery data.

The v0.8 ground-truth gate specifically rejects a nearly perfectly predictive, semantically matched decoy whose source ablation effect is zero.

### Layer 8: claim-aware hierarchical replication

v0.9 turns replication structure into an executable scientific contract.

```text
mechanistic family
       |
       v
   dataset
       |
   model seed
       |
   checkpoint
       |
 dictionary / projector
       |
    subject
       |
    session
       |
     trial
```

The hierarchy used by a specific study may be smaller. It must be declared explicitly.

#### `ReplicationCoordinates`

Every observation can identify:

- dataset;
- model-training seed;
- checkpoint;
- feature dictionary;
- intervention projector;
- subject;
- session;
- trial.

These are scientific identities, not arbitrary dataframe row indices.

#### `HierarchicalReplicationSpec`

A replication study preregisters:

- replication family;
- primary metric;
- **claim axis**, which determines the independent unit;
- active hierarchy;
- null value;
- expected direction when directional;
- minimum independent-unit count;
- bootstrap budget and confidence level;
- minimum independent-unit sign agreement;
- minimum estimable-observation fraction;
- minimum absolute effect;
- whether the confidence interval must exclude the null.

A missing required hierarchy coordinate makes the analysis non-estimable.

#### Unit-balanced aggregation

Lower-level repetition is recursively summarized before the claim-level units are combined.

For a model-seed claim:

```text
trial → session → subject → ... → model seed
```

A seed with 1,000 trials therefore does not receive more claim-level weight than a seed with 20 trials. Extra lower-level observations can improve the internal precision of that seed's estimate, not manufacture extra independent seeds.

#### Hierarchical bootstrap

For each bootstrap draw, the estimator recursively resamples the declared levels with replacement and recomputes the unit-balanced metric.

The resulting `MetricReplicationEstimate` records:

- point estimate;
- confidence interval;
- independent-unit count;
- between-unit standard deviation;
- sign agreement;
- bootstrap budget.

The resampling is deterministic for a fixed study seed.

#### Estimability and replication remain separate

A study can be:

```text
non-estimable
```

because it lacks independent units, required coordinates, metric coverage, or a sufficient fraction of estimable source observations.

Or it can be:

```text
estimable but not replicated
```

because independent units disagree in sign, the interval crosses the null, the effect is too small, or the expected direction fails.

This makes higher-level negative results first-class evidence.

#### v0.7/v0.8 bridges

`observation_from_factorial_contrast(...)` preserves a source contrast's estimability and reasons. Hierarchical analysis cannot repair a confounded factorial design by aggregation.

`observation_from_correspondence(...)` carries v0.8 causal recovery, causal score, predictive transfer, control margins, and source/target effects into a declared replication family.

A correspondence that failed v0.8 promotion may remain an estimable negative replica.

#### Dose response and manifold assumptions

v0.9 also adds `DoseResponseSpec` and `analyze_dose_response(...)` for controlled dose grids such as:

```text
0.00  0.25  0.50  0.75  1.00
```

The result preserves independent units and reports endpoint effect, monotonic fraction, aggregate curve, and normalized response area.

`InterventionManifoldAssumption` records whether an intervention uses:

- zero;
- mean;
- empirical donor;
- nearest neighbor;
- quantile match;
- conditional resampling;
- generative donor;
- causal-scrubbing-style donor;
- custom semantics.

Donor-based methods identify their donor pool. Conditional/generative methods also identify the partition on which the donor model was fitted.

The framework therefore does not silently equate an off-manifold zero vector with a held-out empirical or learned donor.

## Separate scientific objects

The package intentionally keeps these objects distinct:

```text
task metric
    !=
causal effect map
    !=
circuit faithfulness
    !=
held-out evidence pack
    !=
factorial contrast
    !=
representation similarity
    !=
causal feature correspondence
    !=
hierarchical replication
    !=
dose-response robustness
```

Examples:

- high task performance does not imply a faithful circuit;
- similar activations do not imply the same mechanism;
- a faithful circuit in each cell does not imply a valid tokenizer contrast;
- an estimable factorial interaction does not identify corresponding features;
- a predictive feature map does not imply causal substitutability;
- one successful correspondence does not imply model-seed or subject replication;
- hundreds of trials from one seed do not imply architecture-level replication;
- a monotonic dose response supports but does not uniquely identify a mechanism.

This separation is a scientific design feature, not bookkeeping overhead.

## v0.7 replication summaries versus v0.9 replication evidence

v0.7 `FactorialReplicationSummary` remains a useful descriptive object. It reports declared contrast replicas, estimable count, sessions represented, sign agreement, and median effects.

It is **not** the v0.9 uncertainty model.

v0.9 `HierarchicalReplicationResult` is the claim-aware object that:

- declares the independent unit;
- balances lower levels;
- quantifies higher-level uncertainty;
- preserves non-estimable/negative replicas;
- decides estimability separately from replication.

This distinction prevents descriptive cross-session summaries from being interpreted as subject- or architecture-level random-effects evidence.

## ORION / NeuroFM integration

ORION and NeuroFM remain upstream representation/model surfaces:

```text
NeuroTokenBatch / RepresentationBatch
            |
      causal interventions
            |
    CausalEffectRecord
            |
      candidate mechanism
            |
    v0.6 EvidencePackResult
            |
   v0.7 FactorialMechanismReport
            |
      estimable contrast
            |
   v0.8 causal correspondence
            |
   v0.9 hierarchical replication
```

For ORION/NeuroFM temporal claims, projectors should preserve event-relative coordinates when the claim depends on time or event identity.

Subject-level conclusions require multiple subjects. Session-level conclusions require multiple sessions. Model-architecture conclusions require independently trained model seeds.

## Artifact architecture

Four self-checking artifact families are maintained.

### Evidence pack

Contains one frozen candidate study with source revisions, split identities, candidate/controls, intervention donors, per-example cases, held-out aggregates, uncertainty, decision, and integrity hash.

### Factorial study

Contains a declared grid, explicit missing cells, preregistered contrasts, estimability decisions, scalar/map effects, source study identities, deterministic fingerprint, and integrity hash.

### Feature correspondence

Contains immutable source/target feature spaces, split semantics, frozen mapping coefficients, similarity/predictive metrics, candidate/control causal-transfer cases, unmatched features, decision, deterministic fingerprint, and integrity hash.

### Hierarchical replication

Contains:

- frozen replication spec;
- every source observation, including negative/non-estimable ones;
- hierarchy coordinates;
- independent-unit summaries;
- metric estimates and intervals;
- replication decision;
- source study fingerprints;
- deterministic study fingerprint;
- integrity hash.

Raw model inputs and raw feature activations are not required in these evidence artifacts.

## Scientific gates

The maintained synthetic gate hierarchy is now:

```text
known localization
      ↓
shared computation
      ↓
mechanism emergence
      ↓
circuit faithfulness
      ↓
held-out generalization / rejection
      ↓
factorial interaction / confound rejection
      ↓
causal correspondence / predictive-decoy rejection
      ↓
hierarchical replication / pseudoreplication rejection
```

The v0.9 gate must:

- recover a positive mechanism across four independent model seeds;
- count four seeds despite unequal lower-level sample counts;
- reject hundreds of strong trials from one seed as model-seed replication;
- reject a four-seed result with 50/50 sign disagreement;
- recover a planted monotonic five-dose response.

## Historical modules

The pre-v0.2 research package contains many exploratory analyses. They remain importable where dependencies permit, but they are outside the maintained Stable architecture unless promoted through current method/evidence policy.

The top-level package uses lazy compatibility exports so historical optional dependencies do not make the maintained import surface fragile.

## Path to v1

v0.9 completes the main statistical evidence ladder. The remaining path should prioritize evidence closure:

- schema freeze and migrations;
- executable maintained tutorials in evidence CI;
- a real matched architecture × tokenizer neural-data study;
- at least one v0.8 correspondence replicated across independent model seeds;
- session/subject-level uncertainty where the data support those claims;
- a real cross-session or cross-dataset causal study;
- dose response and stronger manifold-aware substitution controls;
- independent reproduction of at least one artifact family;
- published negative results.

See `HIERARCHICAL_REPLICATION.md`, `CAUSAL_FEATURE_CORRESPONDENCE.md`, `FACTORIAL_MECHANISM_STUDIES.md`, and `ROADMAP_V0_9_TO_V1.md` for the active protocol.
