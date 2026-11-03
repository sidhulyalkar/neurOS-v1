# MechInt research strategy: mechanism claims must predict interventions

Status: design document for post-v1 experiments. It is **not** a claim that
all described capabilities exist or that empirical qualification is complete.

## Research north star

For an unseen context `x` and a predeclared intervention `a`, a discovered
mechanism `H` should predict a distribution of output differences:
`p(Delta y | H, x, a)`. Good explanations should win on calibrated
counterfactual prediction, not just attractive circuit graphs.

Maintain two separate axes:

- **Software maturity:** experimental → research → integrated → stable.
- **Evidence authority:** synthetic known-answer → held-out real checkpoint
  → independent reproduction → cross-model/dataset transfer.

Never infer one axis from the other.

## Map of open problems to existing code and next falsification

| Open problem | Existing MechInt surface | Next falsifiable test |
| --- | --- | --- |
| Are features identifiable? | `representations/`, `concept_sae.py`, `sae_faithfulness.py` | Rotate, split and correlate planted latents; measure causal recovery across seeds rather than matching feature labels |
| Do circuits encode native computation? | `circuits/path_patching.py`, `circuits/acdc.py`, `benchmarks/faithfulness.py` | Plant decoys, backup pathways and dormant routes; test counterfactual prediction under interventions |
| Does an explanation generalize? | `benchmarks/evidence_pack.py`, `benchmarks/stability.py` | Freeze discovery, test unseen contexts/perturbations, preserve all invalid evidence |
| Are circuit claims selective to the task? | `benchmarks/repetition_preflight.py`, `benchmarks/paired_selectivity.py` | Paired behavior and difference-in-differences vs reordered/novel controls; independent real-model execution |
| Do representations transfer across models? | `benchmarks/correspondence.py`, `benchmarks/replication.py` | Predict swapped-feature intervention effects across model seeds and architectures |
| Are mechanisms useful to neuroscience? | ORION / NeuroFM adapters and factorial studies | Predict decoder failures or targeted recalibration on held-out subject/session units |
| Can agents accelerate discovery safely? | Evidence recipes and optional NVIDIA NIM research infrastructure | Agents suggest hypotheses and counterexamples; deterministic evaluator alone owns scientific decisions |

## Program A: targeted measurement, now (PR #218, issue #207)

- Keep old Pythia v1 and the matched behavioral v2 scientific identities frozen.
- Add a validation-only, model-agnostic paired selectivity evaluator as an
  experimental companion, **not** another way of promoting old evidence.
- Demand equal target cardinality, paired test IDs/content hashes, consistent
  intact baselines, positive competence, positive matched effect, random-control
  comparison, and a simple magnitude comparator.
- Tests: planted selective route succeeds diagnostically; token-prior and
  generic-degradation traps fail; missing/nonfinite/mismatched evidence cannot
  become positive scores.
- Follow-on: real pinned-checkpoint runner and immutable externally auditable
  discovery seal, with conditional donors and full provenance.

This work tests the *measurement instrument*, not whether Pythia actually uses
a discovered repetition circuit.

## Program B: intervention prediction (next major increment)

Freeze a candidate hypothesis from discovery only:

- target set and causal sign;
- input contexts to which it is expected to apply;
- the exact intervention and donor policy;
- predicted **direction and quantitative interval** for unseen delta scores;
- documented alternative hypotheses and possible interventions that would
  distinguish them.

Run on blind validation cases. Score both predictive error and calibration,
with matched baselines: univariate probe, activation magnitude, random circuit,
and a simple regression on observed baseline logits.

**Primary metric:** held-out counterfactual prediction error relative to the
best baseline, with paired uncertainty. This must be computed on real target
interventions, not just a function fitted to synthetic known-answer labels.

**Fail condition:** interpretability predictions are no better than the
predeclared baseline, or gains disappear under matched intervention controls.
Preserve the failure; do not redescribe retrospective fit as prediction.

## Program C: representation identifiability and causal stability

Build synthetic model organisms with:

- linear and rotated ground-truth factors;
- strongly correlated factors and multiple equivalent dictionaries;
- redundant causal routes and dormant backup pathways;
- nonlinear features and compositional interactions;
- nuisance/task shortcuts that disappear under distribution shift.

Report both matching scores and intervention transfer. A feature direction that
is easy to decode but fails causal transfer is not a recovered mechanism.
Compare seeds, checkpoint stages, and controlled distribution shifts. Report
non-identifiability instead of imposing arbitrary one-to-one feature matching.

## Program D: first real evidence plus neural-decoder application

1. Execute the pinned Pythia open-weight pilot on real model weights. Preserve
   weak behavioral outcomes and independent-run provenance (issues #201/#207).
2. Independently reproduce numerical outcomes and the accept/reject decision,
   including disconfirming findings.
3. Test mechanism-specific counterfactual prediction, including unseen prompt
   families, with preregistered model/activation access.
4. Transfer to a neural decoder with *subject/session* hierarchy and explicit
   nuisance controls; no biological interpretation based on model-only probes.
5. Only after qualification, return to matched Transformer × SSM × Event ×
   Relative-ISI contrasts and cross-model causal correspondence.

## Package boundaries

Keep `adapters/` focused on model instrumentation and structural call safety;
`core/` on immutable experiment design, manifests, split/identity checks and
scalar validation; `benchmarks/` on statistical evidence, known-answer
falsification and explicit estimability; `integrations/` on NeuroFM/ORION
bridges; `reporting/` on read-only claim-scoped interpretation.

No LLM or multi-agent consensus may self-certify a mechanism. Agents can write
proposed hypotheses, choose *discovery-only* experiments and critique results;
only independently checkable protocols and held-out outcomes affect scientific
status. Preserve negative evidence and uncertainty in machine-readable form.

## Promotion checklist for each new research family

1. Explicit causal estimand, counterfactual and admissible claim scope.
2. Sealed discovery/validation split and content-addressed input/model IDs.
3. Decoy/null and adversarial planted known-answer fixtures.
4. Paired baselines, random controls and predeclared multiplicity policy.
5. Failure-preserving artifacts and integrity verification.
6. One real open-weight execution using a pinned revision.
7. Independent reproduction from a distinct execution path.
8. Separate subject/session/seed inference where generalizing beyond examples.
9. Demonstrated prediction or engineering utility beyond simple baselines.
10. Only then promote method/claim authority, with explicit limitations.

Do not mark real empirical evidence complete solely from a passing CI suite.
