# v1 real-evidence program

This directory is the landing zone for the empirical evidence required to move `neuros-mechint` from **software-contract ready** to **empirically closed for a specific neural-model claim**.

No placeholder result should be promoted simply to make the status green.

## First study: 2 × 2 × 3 matched grid

Use one real neural spiking dataset with stable semantic event identities and enough sessions/subjects for the level of inference you intend to make.

```text
architectures: Transformer, SSM
tokenizers: Event, Relative-ISI
model seeds: 0, 1, 2
minimum trained conditions: 12
```

Freeze before training:

- dataset ID, revision, and content hash;
- subject/session/trial IDs;
- discovery and validation partition IDs;
- tokenizer revisions and temporal resolution;
- information/token budget;
- architecture capacity and training-compute budget;
- downstream task and primary metric;
- checkpoint-selection rule;
- causal target surface;
- intervention baselines and donor pools;
- factorial contrasts;
- correspondence feature surface;
- higher-level claim axis and minimum independent units.

## Required artifact flow

For each factorial cell and seed:

```text
v0.6 evidence pack
  ↓
v0.7 factorial cell outcome
  ↓
v0.8 causal correspondence candidate/result when nominated
  ↓
v0.9/v1 replication observation
```

Then produce:

1. matched architecture and tokenizer contrasts;
2. architecture × tokenizer difference-in-differences;
3. at least one held-out correspondence study;
4. seed-level hierarchical replication;
5. session/subject-level replication only when those units are independently available;
6. a five-point dose response for the strongest correspondence/intervention;
7. at least one empirical/nearest-neighbor/conditional or otherwise stronger manifold-aware control;
8. one fresh independent execution assessed with `ReproductionSpec`.

## Minimal acceptance logic

A positive headline is not required. The study is successful as research infrastructure if it produces a valid negative artifact.

Examples of publishable outcomes include:

- Relative-ISI improves task performance but has no estimable causal-mechanism advantage;
- a Transformer/SSM interaction is non-estimable because token budget could not be matched;
- a highly predictive cross-model feature map fails source ablation and is rejected as noncausal;
- a correspondence works in one model seed but fails hierarchical replication;
- a mechanism is session-stable within a subject but subject-level inference is impossible;
- a full substitution works but the graded dose response is non-monotonic;
- the result reproduces qualitatively on a fresh execution but one numerical metric exceeds tolerance.

Those are scientifically useful outcomes. Keep them.

## Completion

After real artifacts exist, update the evidence status through code/data review with their immutable fingerprints. Until then:

```bash
neuros-mechint release-status --json
```

must continue to report `empirical_evidence_complete: false`.
