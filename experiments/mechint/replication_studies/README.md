# Hierarchical replication study artifacts

This directory is the recommended repository location for v0.9 `HierarchicalReplicationResult` artifacts and small study manifests.

The scientific unit is a **replication family with an explicitly declared independent claim axis**, not a folder containing many repeated perturbations.

## Recommended layout

```text
experiments/mechint/replication_studies/
  <study-id>/
    README.md
    replication.json
    source_artifacts.md
```

Keep large model checkpoints, activations, raw neural data, and private subject-level data outside Git. Reference immutable revisions or content-addressed storage instead.

## What belongs in the study README

Record at minimum:

- scientific question;
- replication family ID;
- primary metric;
- claim axis;
- full declared hierarchy;
- null and expected direction;
- minimum independent-unit count;
- confidence level and bootstrap budget;
- minimum sign agreement;
- model/tokenizer/dataset revisions;
- model-training seeds;
- feature dictionary/projector IDs where relevant;
- subject and session inclusion rules;
- source v0.7/v0.8 artifact fingerprints;
- intervention family;
- intervention-manifold assumption;
- dose grid, if used;
- negative/non-estimable replica policy;
- preregistered exclusions.

## Independence checklist

Before reporting a result as replicated, ask:

```text
What is the claim?
        ↓
What must vary independently for that claim?
        ↓
Did those units actually vary independently?
        ↓
Are lower-level repetitions being counted only as precision?
```

Examples:

- 300 trials from one model seed → **1 seed**
- 20 sessions from one subject → **1 subject**
- 4 independently trained model seeds → **4 model-seed units**
- 3 SAE dictionaries initialized from 3 independent dictionary seeds → **3 dictionary units**, if dictionary robustness is the claim

Do not inflate the independent-unit count because one unit contains more trials, more neurons, more tokens, or more perturbations.

## Negative evidence

Keep studies that are:

- non-estimable because too few independent units exist;
- estimable but fail sign agreement;
- estimable but have a confidence interval crossing the null;
- strong within one seed but unstable across seeds;
- stable across seeds but unstable across subjects;
- correspondence-positive under one projector but negative under another defensible projector;
- non-monotonic under intervention dose response.

Those are useful scientific results, not failed bookkeeping.

## Artifact integrity

Write with:

```python
write_replication_artifact(result, "replication.json")
```

Verify with:

```bash
neuros-mechint verify-replication-artifact replication.json --json
```

The artifact intentionally stores summaries, coordinates, decisions, and source fingerprints rather than raw activation tensors or raw neural inputs.
