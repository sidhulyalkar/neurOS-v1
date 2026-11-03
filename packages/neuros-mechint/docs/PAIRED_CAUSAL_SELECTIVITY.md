# Paired causal selectivity (research v1.1 candidate)

**Status: experimental, offline synthetic contract only.** This component does
not modify the frozen v1 evidence-pack artifact catalog or the earlier Pythia
pilot's identity. It addresses the *measurement* portion of [#207](https://github.com/sidhulyalkar/neurOS-v1/issues/207),
not the pending real-model qualification in [#201](https://github.com/sidhulyalkar/neurOS-v1/issues/201).

## Why another metric?

The raw next-token logit margin on `A B C D A B C` may be positive from
token-frequency priors or generic context effects. A high ablation delta may
also simply mean that an intervention breaks language modeling generally.

For the **same semantic trial** and the **same circuit intervention**, record:

```text
         intact                    intervened
repeat  r0                        r1
control c0                        c1
```

The selective effect is `(r0 - c0) - (r1 - c1)`.
A generic equal degradation of the repeated and control conditions has effect
zero. The primary control is **reordered**: `A B C D B A C`; the **novel**
`A B C D F G H` suffix is only a sensitivity diagnostic, as it changes
lexical inventory and final context.

This *ordered-repeat* contrast does **not** detect every legitimate
single-token C-to-D induction mechanism: both repeated and reordered prompts
end in C and previously contain C followed by D. Negative results do not
exclude all induction; positive results do not identify an attention head.

## API and scientific contract

```python
from neuros_mechint.benchmarks.paired_selectivity import (
    PairedSelectivityDesign, PairedSelectivityObservation,
    evaluate_paired_selectivity,
)

# Freeze PairedSelectivityDesign before inspecting validation model outputs.
# Store input hashes computed from actual matched token IDs, not filenames.
# Run identical validation pairs through candidate, magnitude and random
# same-cardinality interventions; record the four scalar margins per row.
result = evaluate_paired_selectivity(design, tuple(observations))
print(result["status"], result.get("random_control_plus_one_rank_fraction"))
```

The design binds a candidate, a distinct equal-size magnitude baseline, every
random control's target set, exact held-out pair hashes, model revision,
discovery commitment, descriptive bootstrap interval policy, and a separate
max_random_rank_fraction diagnostic threshold.

The evaluator rejects duplicated/unplanned rows, content mismatches,
nonfinite scores, discovery rows, and baseline drift. Missing runs are recorded
as nonestimable, **not** silently treated as zero. Every received row is
preserved in a deterministic `received_observations` inventory containing
its pair ID, circuit ID and observation fingerprint. The inventory's
`received_observation_root` binds the exact partial or complete inputs, but
the actual source observations must be retained separately to permit a
fully independent re-audit. Hashes alone cannot reconstruct missing scores.
Resampling is by paired
validation trial; candidate-vs-magnitude comparisons preserve pairing.
Random controls are aggregated across exactly the same pairs, with a
descriptive plus-one upper-tail **rank fraction**:
`r=(1+#random>=candidate)/(1+#random)`. This is **not a calibrated
p-value** unless a separately justified exchangeable control-sampling
scheme, frozen candidate-selection procedure and multiple-testing plan
are established. The evaluator intentionally reports
`random_control_exchangeability_verified=false` and
`random_control_rank_is_calibrated_p_value=false`.

A supportive *diagnostic* requires: enough held-out pairs; a positive baseline
repetition advantage and repeated margin; a positive candidate difference-in-
differences with bootstrap interval above zero; superiority to a magnitude
baseline; and a plus-one random-control rank fraction below the separately
declared `max_random_rank_fraction` threshold. The `alpha` field controls
descriptive bootstrap intervals, not rank-fraction interpretation.
The default 24 random controls permit a minimum descriptive rank
fraction of 0.04; this is **not 0.04 frequentist significance**. Random
control selection is not independently certified by this evaluator.
Multiple candidate selection without multiplicity correction is unsupported.
No claim is made from a score if an expected observation is missing.

### Limits and subsequent work

- A SHA-256 discovery commitment is a binding identifier, **not proof** that a
  protocol was frozen before held-out outputs were read. Independent
  timestamped preregistration or external execution audit remains necessary.
- Identical target cardinality does not guarantee identical physical
  perturbation magnitude, head-level intervention identity, or exchangeable
  null circuits. `magnitude_matched_empirically_verified=false` and
  `target_universe_execution_verified=false` remain explicit.
- All four raw scores must be finite, and intermediate contrasts, paired
  difference-in-differences, candidate-comparator differences, and bootstrap
  means must also be finite. Derived overflow fails closed.
- The analyzer cannot prove tokenization integrity, model RNG replay, lack of
  data leakage, or intervention-manifold validity. The **runner** must enforce
  those, using the v1 strict execution and held-out evidence facilities.
- Bootstrap intervals from eight synthetic or lexical pairs do not give
  population-level inference. Multiple circuits/prompts from one model are
  not independent model replications.
- No downloaded Pythia checkpoint, real-model causal claim, external
  reproduction, or neural/biological interpretation is included here.
- This is a *diagnostic companion* to existing necessity/sufficiency evidence
  packs, not a replacement for them.

## Next implementation increments

1. Add a pinned checkpoint runner that binds exact preflight token-trial IDs
   to the paired observations. Do not run it in PR CI; preserve denied,
   incomplete and nonestimable outputs as negative artifacts.
2. Integrate frozen discovery candidate and donor fitting, with conditional
   empirical donors in addition to mean/zero interventions.
3. Add an independent audit of candidate selection and study sealing, a
   same-cardinality control budget with multiplicity correction, real-model
   interventions, and a second-environment reproduction.
4. Extend beyond this token-order proxy to independent causal probes and
   public neural-decoder failure studies using subject/session-aware units.

Run offline tests:

```bash
pytest -q packages/neuros-mechint/tests/test_mechint_paired_selectivity.py
```
