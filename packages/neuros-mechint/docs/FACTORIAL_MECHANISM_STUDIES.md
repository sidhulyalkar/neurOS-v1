# Factorial mechanism studies in v0.7

v0.7 asks a stricter question than a tokenizer leaderboard:

> When architecture or tokenization changes, does the **held-out causal mechanism evidence** change under an otherwise matched experiment?

The unit of a factorial study is not a raw model checkpoint. It is a completed v0.6 held-out evidence pack.

```text
architecture x tokenizer cell
          |
          v
v0.6 held-out evidence pack
 candidate frozen on discovery data
 zero/mean interventions
 necessity + sufficiency
 same-size controls
 held-out validation
          |
          v
v0.7 FactorialCellOutcome
          |
          v
preregistered matched contrast
          |
      estimable?
       /      \
     no        yes
     |          |
 record why   direct effect / interaction
                |
          cross-session replication
```

## Why build on evidence packs?

A factorial analysis should not rediscover each circuit after seeing the entire grid. That would mix hypothesis generation with comparison.

v0.7 therefore consumes already-completed evidence packs. The source pack contributes:

- held-out task metric;
- candidate size;
- held-out sufficiency;
- held-out necessity;
- held-out joint faithfulness;
- same-size random-control percentile;
- discovery-to-validation degradation;
- sensitivity to alternative intervention baselines;
- source study fingerprint and run hash.

An optional v0.3/v0.4 causal effect map can be attached to the same cell. Effect-map similarity and circuit faithfulness remain different outcomes.

## Factorial cells

`FactorialCellSpec` declares the intended experiment before comparison. A present cell records:

- architecture;
- tokenizer;
- immutable model revision;
- immutable tokenizer revision;
- immutable dataset revision;
- session and subject;
- training seed;
- checkpoint and checkpoint maturity;
- metric;
- discovery method;
- semantic discovery partition ID;
- semantic validation partition ID;
- intervention target universe;
- matched experimental covariates.

A missing cell is also declared explicitly:

```python
FactorialCellSpec(
    ...,
    available=False,
    missing_reason="checkpoint unavailable",
)
```

Missing cells are not silently dropped from an interaction.

## What each source evidence pack must record

`run_factorial_evidence_study(...)` validates the source `EvidencePackSpec` against the declared factorial cell. Put the factorial context in `EvidencePackSpec.metadata` when the pack is created:

```python
metadata = {
    "architecture": "transformer",
    "checkpoint": "step:10000",
    "checkpoint_maturity": 1.0,
    "discovery_partition_id": "animal-17/session-04/discovery-v1",
    "validation_partition_id": "animal-17/session-04/validation-v1",
    "session_id": "session-04",
    "subject_id": "animal-17",
    "training_seed": 0,
    "token_budget": 128,
    "temporal_resolution_ms": 10.0,
    "downstream_capacity": 256,
    "training_compute": 1.0,
}
```

The bridge additionally validates the pack's model/tokenizer/dataset revisions, metric, discovery method, and target universe.

This makes the comparative design machine-checkable rather than a convention stored only in filenames or lab notes.

## Semantic partitions instead of raw-token equality

Two tokenizers can represent the same neural trials with completely different tensors. Their raw input hashes therefore should not match.

For factorial comparison, each cell instead declares semantic partition identities such as:

```text
discovery_partition_id = animal-17/session-04/discovery-v1
validation_partition_id = animal-17/session-04/heldout-trials-v1
```

The underlying v0.6 pack still content-hashes its own inputs to prevent within-pack leakage. The v0.7 partition IDs state that different tokenizations came from the same scientific trial partition.

These identities should come from the dataset/split construction pipeline, not be improvised after results are observed.

## Evidence-protocol matching

The bridge also fingerprints the v0.6 evidence protocol:

- discovery method;
- metric;
- target universe;
- intervention baselines;
- faithfulness policy;
- evidence-pack promotion policy;
- number of random controls.

Two cells produced under different protocols are not directly comparable as a v0.7 matched contrast.

This prevents an apparent architecture effect from being created by, for example, stricter faithfulness thresholds in one cell than another.

## Matched covariates

`MatchedCovariate` makes nuisance dimensions executable design constraints.

For tokenizer studies, useful preregistered covariates include:

- token budget;
- temporal resolution;
- downstream model capacity;
- training compute;
- number of optimization steps;
- context/window duration;
- augmentation policy;
- supervision budget.

A covariate can be exact or numeric with an absolute/relative tolerance.

A comparison that violates a required covariate becomes **non-estimable**. The result retains the reason.

## Task-performance matching

Architecture and tokenizer comparisons can be misleading when one cell simply learned the task better.

`FactorialAnalysisPolicy.max_task_metric_delta` therefore bounds task-performance differences within a primary contrast.

This does not make the models scientifically identical. It removes one obvious explanation for a mechanism difference.

Checkpoint contrasts use the same idea. A checkpoint-emergence comparison can vary checkpoint while still requiring matched task performance, allowing the study to ask whether mechanism changed beyond ordinary performance maturation.

## Preregistered contrasts

v0.7 supports four contrast kinds.

### Architecture main effect

At fixed tokenizer:

```text
A2,T1 - A1,T1
```

All non-architecture design axes must match.

### Tokenizer main effect

At fixed architecture:

```text
A1,T2 - A1,T1
```

All non-tokenizer design axes must match.

### Architecture x tokenizer interaction

For a matched 2 x 2 slice:

```text
interaction = (A2,T2 - A1,T2) - (A2,T1 - A1,T1)
```

This is computed separately for each scalar mechanism outcome.

When aligned effect maps are present, the same difference-in-differences is computed per shared intervention target.

The interaction asks whether changing tokenizer has a different mechanistic consequence in one architecture than another. It does not establish why that interaction exists.

### Checkpoint contrast

At fixed architecture and tokenizer:

```text
late checkpoint - early checkpoint
```

Checkpoint is allowed to vary, while the configured task-performance tolerance remains enforced.

## No omnibus mechanism score

v0.7 does not collapse everything into one “tokenizer quality” number.

A contrast can have separate effects on:

- predictive/task performance;
- circuit size;
- sufficiency;
- necessity;
- joint faithfulness;
- random-control percentile;
- discovery-to-validation degradation;
- intervention-baseline sensitivity;
- causal-map stability.

That separation is intentional. A tokenizer might preserve task score while changing which components are necessary, or yield a smaller faithful circuit without improving predictive performance.

## Estimability comes before effect size

A large numeric difference is ignored as a primary factorial estimate when the design is invalid.

A contrast becomes non-estimable when, for example:

- a required cell is missing;
- an observed cell has no outcome;
- a non-varied axis changed;
- semantic discovery/validation partitions differ;
- evidence protocols differ;
- task performance differs beyond tolerance;
- checkpoint maturity differs beyond tolerance for a non-checkpoint contrast;
- a required matched covariate differs;
- target universes are incompatible.

`FactorialContrastResult.reasons` preserves every detected failure.

## Cross-session replication

`preregister_2x2_contrasts(...)` can assign corresponding contrasts from different sessions to the same replication group.

A replication group reports:

- all declared contrast IDs;
- number that were estimable;
- distinct sessions represented;
- sign agreement for shared scalar outcomes;
- median effect for each outcome;
- whether the group satisfies the minimum cross-session replication requirement.

This remains Research maturity. Two sessions do not substitute for multiple animals, training seeds, or datasets.

## Artifact contract

`write_factorial_artifact(...)` creates a self-checking JSON envelope containing:

- the frozen factorial design;
- every declared cell, including missing cells;
- every preregistered contrast;
- every estimability decision and reason;
- scalar mechanism effects;
- effect-map stability/interaction data when supplied;
- replication summaries;
- source evidence-pack study/run identities;
- a deterministic factorial-study fingerprint;
- an integrity hash over the full result.

Verify it with:

```bash
neuros-mechint verify-factorial-artifact path/to/study.json --json
```

## Ground-truth gate

Run:

```bash
neuros-mechint factorial-ground-truth --json
```

The synthetic design contains a known `-0.5` architecture x tokenizer interaction in held-out joint faithfulness across two sessions.

The gate passes only when v0.7:

1. recovers the interaction in both sessions;
2. marks the interaction as cross-session replicated;
3. rejects a tokenizer contrast whose token budget changed;
4. rejects a 2 x 2 interaction with an explicitly missing cell.

The benchmark therefore tests both **positive recovery** and the ability to refuse invalid comparisons.

## Recommended first neural-tokenizer study

Do not begin with every architecture and tokenizer family.

A useful first grid is:

```text
2 architectures
x 2 tokenizers
x 2 neural sessions
x 2 training seeds, if compute allows
```

For example:

```text
architectures: Transformer, SSM
tokenizers: event, relative-ISI
```

Then repeat with binned counts, burst tokens, synchrony packets, VQ motifs, or assembly tokens only after the first design is operationally clean.

The scientific target is not:

> Which tokenizer wins?

It is:

> Under matched information, compute, capacity, task performance, and held-out evidence protocol, where does tokenization change the causal computation learned by a neural model, and where is that computation invariant?

## Claim boundary

An estimable architecture x tokenizer interaction supports a conditional statement about the declared experimental grid. It does **not** establish:

- a universal advantage for a tokenizer;
- that the interaction generalizes to other architectures;
- that the intervention is in-distribution;
- that aligned effect-map targets are biologically homologous;
- that one tokenizer preserves more biological “meaning” without additional cross-dataset/session evidence;
- that two training seeds characterize training uncertainty.

v0.8 and v0.9 are intended to attack feature correspondence and hierarchical replication directly.
