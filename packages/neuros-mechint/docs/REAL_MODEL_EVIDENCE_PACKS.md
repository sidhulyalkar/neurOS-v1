# Real-model evidence packs in v0.6+

A real-model mechanistic claim should not be promoted because a discovery method found an attractive circuit on the examples used to search for it.

The v0.6 evidence-pack contract separates discovery from validation and remains the **cell-level evidence unit** for v0.7 factorial studies.

```text
discovery examples only
         |
 candidate generation
 + donor estimation
         |
       freeze
         |
 held-out validation
         |
 necessity / sufficiency
 matched controls
 paired uncertainty
         |
  EvidencePackResult
         |
   +-----+------+
   |            |
positive     negative
   |            |
   +-----+------+
         |
self-checking artifact
         |
optional v0.7 factorial cell
```

## What is frozen before validation

At minimum:

- model identity and immutable revision;
- tokenizer identity/revision when applicable;
- dataset identity/revision;
- task metric;
- intervention target universe;
- discovery method;
- candidate circuit;
- intervention baselines;
- discovery-fitted mean donors;
- faithfulness thresholds;
- evidence-pack promotion policy;
- random-control budget.

Candidate discovery callbacks receive discovery examples only.

## Input leakage protection

Every evidence input must support deterministic full-content hashing.

Duplicate input content anywhere within one evidence pack is rejected, including a discovery example copied into validation under a different ID.

For v0.7 cross-tokenizer comparisons, raw token tensors across cells are expected to differ. Factorial cells therefore additionally carry semantic discovery/validation partition IDs identifying the underlying scientific trials from which each tokenizer representation was produced.

## Mean-ablation donors

Mean replacement statistics are fit from discovery activations only, frozen, and reused during held-out evaluation.

This avoids allowing the validation example to adapt its own perturbation baseline.

## Model mutation guard

When an adapter exposes a deterministic model fingerprint payload, model state is hashed:

1. before discovery;
2. after candidate selection/donor fitting;
3. after intervention evaluation.

Discovery code that silently trains or mutates the validated model therefore invalidates the study.

## Faithfulness outcomes

For every valid example/baseline case, the pack retains:

- intact/all-target metric;
- null metric;
- candidate-retained metric;
- candidate-removed metric;
- sufficiency;
- necessity;
- joint faithfulness;
- same-cardinality random controls;
- pass/fail under the frozen faithfulness policy.

Alternative intervention baselines remain paired within the same example.

## Invalid perturbations

A perturbation can make the faithfulness normalization scientifically meaningless, for example when the all-target and null metrics are indistinguishable or when the declared null intervention outperforms the intact computation under the chosen direction.

Such cases remain explicit invalid records. They are not silently discarded or sign-flipped.

## Uncertainty

When one example is evaluated under several perturbation baselines, those observations are correlated.

The evidence-pack bootstrap resamples examples rather than individual intervention cases, preserving that pairing and avoiding pseudo-replication.

## Same-size simple baseline

A discovered candidate can optionally be compared with a same-cardinality activation-magnitude candidate fitted only on discovery examples.

This asks whether the causal discovery procedure contributes more than a simple “largest activation” heuristic.

## Promotion is separate from publication readiness

`promotion.passed` asks whether the frozen candidate survived the scientific criteria.

`publication_ready` asks whether the artifact has immutable revision provenance.

Therefore:

- a pinned negative result can be publication-ready;
- a positive result using mutable revisions can fail publication readiness.

This distinction is deliberate.

## Artifact contents

`write_evidence_pack_artifact(...)` stores:

- frozen study specification;
- input IDs, hashes, split roles, and metadata;
- candidate and optional simple baseline;
- discovery-fitted donors;
- every valid/invalid case;
- discovery and validation aggregates;
- paired bootstrap interval;
- promotion/rejection reasons;
- repository-standard model/data provenance;
- package versions;
- wall time;
- peak Python memory;
- peak CUDA memory where available;
- deterministic study fingerprint;
- run-specific hash;
- full-content integrity hash.

Raw model inputs are intentionally not serialized into the artifact.

Verify a copied artifact with:

```bash
neuros-mechint verify-evidence-artifact result.json --json
```

## Known-shift gate

Run:

```bash
neuros-mechint evidence-pack-generalization-ground-truth --json
```

The synthetic system intentionally uses one real causal route on discovery examples and another on validation examples.

The correct outcome is:

```text
discovery succeeds
held-out validation fails
promotion rejected
benchmark passes
```

A framework that cannot reject its own discovery is not performing held-out validation.

## External-model starting recipes

List maintained starting configurations:

```bash
neuros-mechint evidence-recipes
```

Current recipe families cover:

- TransformerLens;
- NNsight;
- SAELens;
- circuit-tracer candidate generation.

These recipes are execution starting points, not measured evidence. Before publishing a real artifact, resolve mutable upstream names to immutable model/tokenizer/SAE/transcoder revisions.

## Repository location

Cell-level evidence artifacts belong under:

```text
experiments/mechint/evidence_packs/
```

Keep negative runs as well as positive runs.

## Using a pack in v0.7

A v0.7 factorial cell should declare enough metadata that the source evidence pack can be validated against the intended comparison:

```text
architecture
session / subject
training seed
checkpoint + maturity
semantic discovery partition ID
semantic validation partition ID
token budget
temporal resolution
downstream capacity
training compute
other preregistered covariates
```

Store these values in `EvidencePackSpec.metadata` when generating the cell evidence.

`run_factorial_evidence_study(...)` then validates:

- model/tokenizer/dataset revisions;
- metric;
- discovery method;
- target universe;
- all declared factorial metadata/covariates.

The converted `FactorialCellOutcome` also carries an evidence-protocol fingerprint so two cells cannot be treated as a matched primary contrast if their v0.6 scientific protocols differ.

## Recommended progression

For a real neural-tokenizer factorial study:

1. create semantic discovery/validation trial partitions once;
2. generate each tokenizer representation from those same partitions;
3. train each architecture/tokenizer cell under matched compute/capacity budgets;
4. choose checkpoint-matched cells using a preregistered maturity rule;
5. produce one v0.6 evidence pack per cell;
6. preserve negative cell-level evidence;
7. run the v0.7 estimability audit;
8. compute only valid preregistered factorial contrasts;
9. replicate the primary contrast across sessions, subjects, and model seeds as claim strength increases.

See `FACTORIAL_MECHANISM_STUDIES.md` for the comparative protocol.

## Claim boundary

A passing evidence pack supports a conditional held-out mechanism statement for one frozen condition. It does not establish:

- circuit uniqueness;
- in-distribution interventions;
- cross-dataset transfer;
- cross-session stability;
- cross-seed stability;
- feature correspondence;
- biological homology;
- an architecture or tokenizer effect.

The last item is exactly why v0.7 exists: a valid cell-level result still requires a matched factorial design before becoming a comparative claim.
