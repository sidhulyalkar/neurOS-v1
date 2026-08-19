# Circuit faithfulness in neuros-mechint

A circuit candidate is not accepted because it has a compelling attribution or activation score. The package asks whether the candidate preserves the chosen computation when retained and damages it when removed.

This v0.5 faithfulness contract remains the quantitative core inside each v0.6 evidence pack and therefore inside every observed v0.7 factorial cell.

## Core quantities

Let:

- `M_all` be the metric with every audited target retained;
- `M_null` be the metric with every audited target ablated;
- `M_circuit` be the metric with only the candidate retained;
- `M_complement` be the metric with the candidate removed and its complement retained.

For a higher-is-better metric:

```text
span = M_all - M_null

sufficiency = (M_circuit - M_null) / span
necessity   = (M_all - M_complement) / span
joint       = min(sufficiency, necessity)
```

For lower-is-better metrics, values are sign-oriented before the same calculation.

Scores are not clamped. Values above 1 or below 0 can reveal compensation, destructive interference, or a poorly behaved intervention.

A comparison is invalid when the all-target/null span does not define the intended normalization. v0.6 evidence packs preserve such cases as explicit invalid evidence instead of silently dropping them.

## Why both necessity and sufficiency?

A circuit can be necessary but not sufficient. Removing one member of a serial route can collapse output even if retaining that member alone cannot perform the computation.

A circuit can also be sufficient but not necessary when redundant pathways exist.

The package therefore keeps both quantities and uses joint faithfulness only as one conservative summary.

## Equal-cardinality random controls

A candidate is compared with target sets of the same cardinality.

Small target universes can enumerate alternatives; larger universes use deterministic seeded sampling.

Each random candidate receives:

- sufficiency;
- necessity;
- joint faithfulness.

The primary random-control statistic compares:

```text
candidate_joint = min(candidate_sufficiency, candidate_necessity)
```

with the random circuits' joint scores.

Separate necessity and sufficiency percentiles remain diagnostic.

## Default single-case policy

`FaithfulnessPolicy` defaults to thresholds for:

```text
sufficiency
necessity
joint random percentile
```

These are package defaults, not universal scientific constants. Real studies should preregister or justify thresholds and include sensitivity analyses.

## Adapter-level evaluation

Any `ModelAdapter` can use the same benchmark:

```python
from neuros_mechint.benchmarks import (
    CircuitCandidate,
    evaluate_adapter_circuit_faithfulness,
)

report = evaluate_adapter_circuit_faithfulness(
    adapter=adapter,
    inputs=inputs,
    metric=metric,
    all_targets=all_hook_points,
    candidate=CircuitCandidate(
        name="candidate",
        targets=("component_a", "component_b"),
    ),
)
```

This lets PyTorch, TransformerLens, and NNsight candidates face the same quantitative object.

## SAE feature evaluation

SAE feature faithfulness first establishes the reconstruction baseline:

```text
original activation
       |
       v
SAE encode -> decode
       |
       +---- reconstruction gap reported
       |
       v
feature-subset intervention
       |
       v
necessity / sufficiency / controls
```

This prevents SAE reconstruction error from masquerading as feature causality.

## Held-out evidence packs

Single-example circuit faithfulness is not held-out generalization.

v0.6 therefore wraps the same mathematics in a frozen study:

```text
discovery examples
       |
 candidate selection + donor fitting
       |
     freeze
       |
validation examples
       |
faithfulness under planned interventions
       |
paired uncertainty + promotion/rejection
```

The same candidate is tested across held-out examples and intervention baselines. Invalid perturbations stay visible.

See `REAL_MODEL_EVIDENCE_PACKS.md`.

## v0.7 factorial use

A valid evidence pack answers:

> Is this candidate faithful enough in this one frozen condition?

It does **not** answer:

> Did architecture or tokenizer cause the difference between this condition and another?

v0.7 adds the matched comparison layer.

Each factorial cell imports its held-out faithfulness outcomes separately:

- mean held-out sufficiency;
- mean held-out necessity;
- median held-out joint faithfulness;
- mean joint random percentile;
- discovery-to-validation degradation;
- intervention-baseline sensitivity;
- candidate size.

A primary architecture/tokenizer/checkpoint contrast is computed only after the estimability layer verifies the source cells were produced under compatible semantic partitions, evidence protocols, task performance, target universes, and declared covariates.

This produces a useful hierarchy:

```text
attribution / discovery
       ↓
circuit faithfulness
       ↓
held-out evidence pack
       ↓
matched factorial contrast
       ↓
replication
```

Skipping a level weakens the corresponding claim.

## Known-ground-truth gates

Circuit faithfulness itself:

```bash
neuros-mechint circuit-faithfulness-ground-truth --json
```

Held-out discovery/generalization separation:

```bash
neuros-mechint evidence-pack-generalization-ground-truth --json
```

Factorial comparison:

```bash
neuros-mechint factorial-ground-truth --json
```

The factorial gate has a known interaction and also contains invalid comparisons that must be rejected.

## What a passing faithfulness report establishes

A passing single-condition report supports only:

> Under this input distribution, metric, target universe, intervention family, and policy, the nominated target set is sufficiently performance-preserving when retained, sufficiently damaging when removed, and exceeds the configured same-size control threshold.

It does not establish:

- circuit uniqueness;
- in-distribution interventions;
- held-out generalization;
- cross-session generalization;
- cross-seed stability;
- architecture or tokenizer effects;
- causal feature correspondence;
- biological homology;
- a unique semantic interpretation for a feature.

Those stronger statements require the higher evidence layers described above.
