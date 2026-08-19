# Hierarchical replication and uncertainty in v0.9

v0.9 asks a different question from the earlier evidence layers:

> **At what scientific level is this mechanistic result actually replicated?**

A large number of perturbations can make a within-model estimate precise without creating a single additional independent model, subject, session, or dataset.

The maintained hierarchy is deliberately explicit:

```text
architecture / tokenizer / mechanism family
  -> dataset
    -> model-training seed
      -> checkpoint
        -> feature dictionary
          -> intervention projector
            -> subject
              -> session
                -> trial
```

A study does not need to use every axis. It must declare the axes that belong to its scientific claim.

## Core rule: the claim chooses the independent unit

Examples:

| Claim | Minimum independent unit |
| --- | --- |
| held-out trial effect | trial |
| cross-session stability | session |
| subject-general mechanism | subject |
| architecture-level mechanism | independent model-training seed |
| SAE-feature mechanism | independent dictionary/dictionary seed when dictionary variability is part of the claim |
| projector-robust mechanism | independently declared projector condition |
| cross-dataset meaning | dataset |

The v0.9 engine refuses to convert lower-level repetition into higher-level replication.

Three hundred trials from one model seed still contain **one model seed**.

## `ReplicationCoordinates`

Every observation can carry:

- dataset ID;
- model-training seed;
- checkpoint;
- feature dictionary ID;
- intervention projector ID;
- subject ID;
- session ID;
- trial ID.

These are scientific identities, not row numbers.

For example:

```python
ReplicationCoordinates(
    dataset_id="ibl-choice-v1",
    model_seed=3,
    checkpoint="step:10000",
    dictionary_id="sae-seed:2",
    projector_id="movement-onset-window-v1",
    subject_id="mouse:17",
    session_id="mouse:17/session:04",
    trial_id="mouse:17/session:04/trial:381",
)
```

## `HierarchicalReplicationSpec`

A replication study preregisters:

- `family_id`: the mechanistic hypothesis being replicated;
- `claim_axis`: the independent unit for the claim;
- `primary_metric`;
- the hierarchy actually used by the study;
- the null value;
- expected direction, when directional;
- minimum independent-unit count;
- bootstrap sample count and confidence level;
- minimum independent-unit sign agreement;
- minimum estimable-observation fraction;
- minimum absolute effect;
- whether the confidence interval must exclude the null.

The hierarchy is executable. If a required coordinate is missing, the analysis is non-estimable rather than silently flattening the data.

## Unit-balanced aggregation

Suppose four model seeds have the following held-out trial counts:

```text
seed 0: 1000 trials
seed 1:   20 trials
seed 2:   20 trials
seed 3:   20 trials
```

A model-seed claim does **not** average all 1,060 trial effects directly.

v0.9 recursively averages within lower levels first:

```text
trial -> session -> subject -> ... -> model seed
```

and then gives each model seed one contribution to the architecture-level estimate.

Trial count improves the precision of a seed's internal estimate. It does not give that seed additional votes in a seed-level claim.

## Hierarchical bootstrap

For uncertainty, v0.9 resamples according to the declared nesting structure.

For a model-seed claim:

1. resample model seeds with replacement;
2. within every sampled seed, resample the next declared level;
3. continue recursively through the hierarchy;
4. compute the unit-balanced metric;
5. repeat for the preregistered bootstrap budget.

The output records:

- point estimate;
- confidence interval;
- number of independent claim-level units;
- between-unit standard deviation;
- sign agreement across independent units.

The bootstrap is deterministic for a fixed study seed.

## Estimable is not replicated

These states remain separate.

### Non-estimable

Examples:

- one model seed for a model-seed claim;
- missing subject/session identity required by the declared hierarchy;
- mixed replication families;
- too many source studies marked non-estimable;
- missing primary metric.

### Estimable but not replicated

Examples:

- four independent seeds exist, but two effects are positive and two are negative;
- the confidence interval crosses the preregistered null;
- the effect is smaller than the preregistered minimum;
- sign agreement is below threshold.

### Replicated under the declared hierarchy

All estimability and replication criteria pass.

This remains a conditional claim. Replication across model seeds is not replication across subjects or datasets unless those are also independent units in the tested claim.

## v0.7 factorial bridge

`observation_from_factorial_contrast(...)` turns a v0.7 contrast into a v0.9 replication observation.

Critically, a source contrast's `estimable` flag and rejection reasons are preserved.

A hierarchical analysis cannot launder a confounded or incomplete factorial contrast into a valid random-effects estimate.

This permits uncertainty-aware replication of:

- architecture effects;
- tokenizer effects;
- architecture x tokenizer interactions;
- checkpoint effects.

## v0.8 correspondence bridge

`observation_from_correspondence(...)` carries forward held-out correspondence metrics including:

- median causal recovery;
- median causal score;
- validation predictive R2;
- random-control margin;
- shuffled-donor margin;
- median source effect;
- median target effect.

The original v0.8 promotion status is retained as metadata.

A failed v0.8 correspondence can therefore remain an **estimable negative replica** rather than disappearing from the study.

## Dose-response

Binary ablation and full substitution can occasionally produce a lucky endpoint.

v0.9 adds controlled dose-response analysis over a preregistered grid such as:

```text
0.00  0.25  0.50  0.75  1.00
```

A dose-response study records independent units separately and reports:

- aggregate curve;
- oriented endpoint effect;
- mean within-unit monotonic fraction;
- normalized area under the response curve;
- explicit pass/rejection reasons.

A monotonic response supports a mechanistic interpretation but does not prove one by itself.

## Intervention-manifold assumptions

Every dose-response/substitution protocol can describe how its intervention values relate to the learned activation manifold.

Supported labels include:

- `zero`;
- `mean`;
- `empirical_donor`;
- `nearest_neighbor`;
- `quantile_matched`;
- `conditional_resample`;
- `generative`;
- `causal_scrubbing`;
- `custom`.

Donor-based methods must identify their donor pool. Conditional/generative methods must additionally record the partition on which they were fitted.

This prevents an off-manifold zero intervention and a held-out empirical donor from being reported as if they made the same intervention assumption.

## Self-checking artifacts

`write_replication_artifact(...)` stores:

- frozen replication spec;
- every source observation, including non-estimable/negative observations;
- independent-unit summaries;
- metric estimates and confidence intervals;
- final estimability/replication decision;
- source study fingerprints;
- deterministic study fingerprint;
- artifact integrity hash.

Verify with:

```bash
neuros-mechint verify-replication-artifact path/to/study.json --json
```

## Ground-truth gate

Run:

```bash
neuros-mechint replication-ground-truth --json
```

The benchmark must simultaneously:

1. recover a real positive mechanism across four independent model seeds;
2. report exactly four independent seed units despite unequal lower-level sample counts;
3. reject 300 strong trials from a single seed as model-seed replication;
4. reject an estimable four-seed study with 50/50 sign disagreement;
5. recover a known monotonic five-dose substitution curve.

The gate therefore tests both uncertainty estimation and the framework's ability to say **no** to pseudoreplication.

## Recommended real-study sequence

For the architecture x tokenizer program:

```text
v0.6 evidence pack per condition
    -> v0.7 estimable architecture x tokenizer interaction
      -> v0.8 held-out causal feature correspondence
        -> v0.9 independent-seed / subject / session replication
          -> dose response + alternative valid projector
```

A strong result would not merely say that Relative-ISI features can substitute between two trained models. It would show that the causal substitution effect recurs across independently trained seeds and neural recording units with quantified higher-level uncertainty.

## Claim boundary

A passing v0.9 replication supports:

> Under the declared family, metric, hierarchy, independent-unit definition, null, direction, intervention protocol, and replication policy, the mechanistic effect was consistent enough across the declared independent units to satisfy the preregistered uncertainty and sign-agreement criteria.

It does **not** establish:

- independence of units that share undeclared upstream causes;
- transfer to a new dataset, species, task, architecture, tokenizer, dictionary, or projector not represented by the claim;
- biological homology merely because a correspondence replicated;
- that bootstrap uncertainty is exact with very few independent units;
- that monotonic dose response uniquely identifies a causal mechanism.
