# Causal feature correspondence

## Purpose

`neuros-mechint` v0.8 turns feature correspondence into a held-out causal experiment.

The motivating problem is simple to state and easy to get wrong:

> Two hidden features can correlate, align geometrically, receive the same semantic label, and predict one another accurately without implementing the same computation.

v0.8 therefore separates six questions:

1. do the activations correlate?
2. do the feature spaces have similar geometry?
3. do human or automated labels agree?
4. does a mapping learned on discovery trials predict target activations on held-out trials?
5. do source and target interventions have similar effects?
6. can the frozen source mapping causally substitute for the target feature contribution on held-out trials?

Only the final layers support a mechanistic-correspondence claim.

## Scientific pipeline

```text
paired scientific trials
        |
        +---------------- discovery partition ----------------+
        |                                                       |
source representation                                   target representation
        |                                                       |
        +---- activation / geometry / semantic evidence --------+
                                |
                         fit candidate mapping
                                |
                              FREEZE
                                |
        +--------------- held-out validation ------------------+
        |                                                       |
 source feature ablation                                target feature ablation
        |                                                       |
        +----------- mapped source -> target substitution ------+
                                |
                    +-----------+-----------+
                    |                       |
             shuffled trial pairs   random source feature sets
                    |                       |
                    +-----------+-----------+
                                |
                     promote OR reject correspondence
```

The mapping fitter never receives validation examples.

## Typed feature-space identity

`FeatureSpaceIdentity` records the representation being compared rather than assuming that an integer feature index has portable meaning.

Each feature space records:

- model ID and immutable revision;
- representation/component ID;
- architecture;
- tokenizer ID and revision;
- dataset ID and revision;
- session and optional subject;
- checkpoint;
- ordered feature names;
- optional feature semantic labels;
- optional metadata.

A study must explicitly declare every context axis on which source and target differ. Undeclared differences fail construction.

Raw feature index equality is never treated as correspondence.

## Correspondence shapes

v0.8 supports three typed candidate shapes.

### One-to-one

```text
source feature s1 -> target feature t7
```

### One-to-many

```text
source feature s1 -> [t7, t19, t31]
```

This is useful when one representation factorizes a computation that another keeps concentrated.

### Subspace

```text
[source s2, s8, s14] -> [target t3, t4]
```

Subspace correspondence is often a better hypothesis than forcing arbitrary latent bases into one-to-one matches.

The maintained discovery method is a small ridge-linear map with an intercept. The discovery method is deliberately modest: v0.8 is primarily a causal-validation framework, not a claim that linear alignment is universally optimal.

## Semantic trial identity

Feature vectors from two models can be different even when they describe the same experimental trial. v0.8 therefore uses both:

- `example_id`: identity of the paired representation observation;
- `semantic_trial_id`: identity of the underlying scientific trial.

A trial cannot appear once in discovery and again under a renamed validation example ID.

Each example also carries an explicit discovery or validation `partition_id`.

## Similarity is reported, not promoted automatically

The frozen candidate reports:

- activation correlation;
- linear CKA geometry;
- semantic-label overlap when labels exist;
- discovery predictive R².

Held-out validation independently reports:

- activation correlation;
- linear CKA;
- predictive R²;
- discovery-to-validation R² drop.

These quantities remain visible even when the causal test fails.

That is intentional. A result such as:

```text
semantic overlap        1.00
validation R²           0.9999
source ablation effect  0.0000
causal correspondence   REJECTED
```

is scientifically informative.

## Paired causal substitution

A `CausalSubstitutionEvaluator` supplies five scalar metrics for a held-out trial:

```text
source_clean_metric
source_ablated_metric

target_clean_metric
target_ablated_metric
target_substituted_metric
```

After orienting the metric so larger means better, define:

```text
source_effect = source_clean - source_ablated
target_effect = target_clean - target_ablated
```

For a causally meaningful target effect:

```text
recovery =
    (target_substituted - target_ablated)
    / (target_clean - target_ablated)
```

The reported causal score clips recovery to `[0, 1]`, but only if both the source and target effects exceed their preregistered relevance thresholds.

Thus a source feature that predicts the target beautifully but is unused by the source model receives no causal credit.

## Why both source and target necessity matter

Without target ablation, a mapped value can appear successful simply because the target feature was irrelevant.

Without source ablation, a correlated source feature can appear to transfer the computation even if the source model never uses it.

v0.8 therefore requires both directions of relevance before substitution is considered mechanistic evidence.

## Shuffled semantic-pair control

The candidate feature set is retained, but its source activation is taken from a different held-out trial.

```text
target trial i <- mapped source feature from trial j
```

This asks whether the observed restoration depends on the scientifically paired state rather than merely injecting a plausible activation from the correct feature family.

## Same-cardinality random-source controls

Random controls replace the preregistered source feature set with other source feature sets of identical cardinality.

Each random feature set receives its own discovery-only mapping to the same target feature set, then faces the same held-out causal test.

This is stricter than comparing the candidate against unfit random coordinates.

The candidate is compared against the distribution of per-control held-out median causal scores.

## Promotion policy

`FeatureCorrespondencePolicy` can require:

- minimum discovery and validation sample counts;
- minimum valid-transfer fraction;
- minimum held-out predictive R²;
- maximum discovery-to-validation R² degradation;
- minimum source intervention effect;
- minimum target intervention effect;
- minimum median causal recovery;
- minimum random-control percentile;
- minimum margin over shuffled trial pairs;
- minimum margin over random source feature sets;
- optional source/target intervention-effect correlation;
- rejection of exact paired activation content duplicated across splits.

A failed criterion remains in `CorrespondencePromotionDecision.reasons`.

## ModelAdapter execution

`AdapterFeatureSpaceView` connects the correspondence framework to an ordinary `ModelAdapter`.

The integration:

1. captures the named source/target component;
2. projects each activation onto a declared feature vector;
3. runs source-feature ablation;
4. runs target-feature ablation;
5. injects mapped source values into target features;
6. evaluates the declared scalar metrics;
7. verifies that model fingerprints did not change during the study.

### Default tensor projector

`TensorFeatureProjector` requires an explicit feature axis.

Its default vectorization averages all non-feature axes. Its replacement operation broadcasts selected feature values over those axes.

That makes the default suitable for claims about aggregate feature channels. It is **not** automatically suitable for claims about token positions, neural events, or temporal windows.

Temporal/event-preserving studies should provide a projector with the same `vector` / `replace` semantics that preserves the experimental coordinates required by the claim.

## v0.7 factorial provenance

A v0.7 architecture × tokenizer result can nominate a v0.8 study.

`factorial_origin_from_report(...)` creates a `FactorialCorrespondenceOrigin` only when the referenced contrast is estimable.

This records:

- factorial study fingerprint;
- contrast ID;
- source cell IDs.

The link is provenance, not evidence that particular features correspond.

## Ground-truth gate

Run:

```bash
neuros-mechint correspondence-ground-truth --json
```

The benchmark contains two source features:

- a genuinely causal feature;
- a correlated decoy with nearly identical activation statistics and the same semantic label.

Both predict the target feature extremely well.

The benchmark passes only when:

1. the true mapping retains held-out predictive transfer;
2. the true source feature is causally relevant;
3. substitution restores the target contribution;
4. shuffled trial-pair controls lose;
5. same-cardinality random-source controls lose;
6. the highly predictive correlated decoy is nevertheless rejected because its source ablation effect is zero.

This benchmark is intentionally designed so correlation-based correspondence fails.

## Artifact contract

Use:

```python
write_correspondence_artifact(result, path)
read_correspondence_artifact(path)
```

or:

```bash
neuros-mechint verify-correspondence-artifact path.json
```

The JSON artifact contains:

- frozen study specification;
- source/target immutable identities;
- discovery and validation partition identities;
- example IDs, semantic trial IDs, and activation-pair hashes;
- frozen mapping coefficients;
- discovery and validation similarity metrics;
- every candidate and control causal-transfer case;
- promotion/rejection reasons;
- unmatched source/target features;
- deterministic study fingerprint;
- full-content artifact hash.

Raw activation arrays and raw model inputs are not serialized in the evidence artifact.

## Scientific claim boundary

A passing v0.8 study supports only the conditional statement:

> Under the frozen source/target model, tokenizer, data, session, checkpoint, feature-space, projector, metric, discovery mapping, semantic trial pairing, intervention family, and control policy, the selected source feature set was causally relevant and its frozen mapping restored the held-out target feature contribution better than the configured shuffled-pair and same-cardinality random-source controls.

It does **not** establish:

- feature uniqueness;
- biological homology;
- universal semantic identity;
- equality of raw latent indices;
- causal equivalence under every intervention;
- in-manifold replacement;
- cross-dataset generalization;
- cross-subject generalization;
- stability across model or dictionary seeds;
- that a linear mapping is the uniquely correct alignment.

Negative and similarity-without-causality results are first-class evidence.
