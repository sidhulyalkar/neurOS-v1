# Longitudinal EEG Model Ladder

This document defines the authoritative multi-method longitudinal EEG comparison layer built on the prospective split and calibration authority in neurOS.

The governing rule is simple:

> One frozen data authority, many methods.

Every method must consume the same serialized source-history, target-calibration and final-evaluation identity. Method-specific convenience is never allowed to silently redefine the scientific question.

## Why this exists

A longitudinal BCI benchmark becomes misleading quickly if each model is allowed to choose a different target test set, use future sessions under a prospective claim, estimate adaptation state from final evaluation samples, or recreate nominally identical frozen encoders independently.

The ladder therefore separates three identities:

1. **data identity**: processed-data SHA-256 plus exact source/calibration/evaluation indices;
2. **learned-state identity**: SHA-256 of the complete trained model state, including registered buffers;
3. **representation identity**: SHA-256 over the exact frozen source/calibration/evaluation embedding tensors used by a downstream method.

Configuration and seed are provenance. They are not substitutes for learned-state identity.

## Comparison lanes

The initial executable ladder supports:

- `csp-lda`
- `eegnet`
- `eeg-conformer`
- `frozen-eegnet`
- `frozen-eeg-conformer`
- `sourceweigher-eegnet`
- `sourceweigher-eeg-conformer`

The family intentionally spans a transparent classical baseline, task-specific deep decoders, fixed-representation transfer, and reliability-weighted transfer.

### CSP + LDA

The classical floor. This remains important because a sophisticated neural system should not be promoted merely for beating a weak or incorrectly configured baseline.

### EEGNet and EEG-Conformer

Task decoders are refit end to end at each declared target calibration budget. Their parameter count, fit time, training history, learned-state hash, analysis-manifest identity, calibration behavior and inference latency remain visible beside predictive performance.

The benchmark does not claim the two architectures are capacity matched.

### Frozen encoder + matched readout

A frozen encoder is trained once from declared prior history. Its source, target-calibration and target-evaluation embeddings are then materialized and fingerprinted.

Only the matched logistic readout is refit across target calibration budgets. This isolates representation transfer from encoder adaptation.

### SourceWeigher + frozen encoder

SourceWeigher consumes the **same frozen encoder and exact same embedding tensors** as the corresponding unweighted frozen lane.

This is a critical comparison invariant. The weighted and unweighted lanes must share:

- `model_state_sha256`;
- `representation_sha256`;
- encoder-state fingerprint;
- encoder configuration;
- analysis-manifest identity.

Only the transfer strategy changes.

Target-dependent source weights may be estimated from declared target calibration embeddings. Final evaluation embeddings are forbidden from target-moment estimation.

## LongitudinalCaseAuthority

Each subject / target-session case is serialized as a `LongitudinalCaseAuthority` before model fitting.

The authority binds:

- subject identity;
- target session;
- observed chronology;
- source-history indices;
- ordered target calibration indices;
- immutable final evaluation indices;
- processed-data SHA-256;
- partition fingerprint;
- calibration fingerprint;
- case fingerprint;
- dataset-specific metadata such as original protocol/cohort.

A method restores and validates that authority before fitting. Changed processed EEG bytes, sample order, labels, grouping metadata or case identity fail before a model is trained.

## Prospective history semantics

The default evidence question is prospective next-session transfer.

For held-out target session `S_k`, only sessions observed before `S_k` may enter the source history. Later sessions are future-excluded.

A conventional all-other-session analysis may still be useful as a secondary symmetric benchmark, but it must not be narrated as prospective evidence.

## Fixed target evaluation identity

For each target session, neurOS creates one deterministic balanced calibration pool and one immutable final evaluation set.

Increasing the calibration budget changes only how much of the calibration pool a method may use. The final evaluation indices never move.

This avoids an especially common source of false calibration gains: testing larger-budget models on a smaller or easier remainder of the same held-out session.

## Nested calibration budgets

Calibration examples are deterministic and nested per class.

If the budgets are:

```text
0, 1, 2, 5, 10
```

the budget-2 set must contain the budget-1 examples, the budget-5 set must contain budget 2, and so on.

The resulting frontier measures the marginal value of additional labeled target calibration under a fixed final evaluation identity.

## Zero-calibration semantics

Target-independent methods can report budget 0.

A target-dependent SourceWeigher lane is explicitly unavailable at budget 0 because no legal target calibration observations exist. neurOS does not substitute final evaluation examples as unlabeled target observations.

This produces two frontier summaries:

- **full calibration-frontier AUC**, only for methods genuinely defined from zero calibration onward;
- **positive-budget adaptation AUC**, for methods whose adaptation begins only after target calibration becomes available.

An unavailable zero-calibration point is not a failure.

## PreparedFrozenEncoderCase

Frozen representation work is prepared once per case, encoder architecture and seed.

The prepared state contains:

- the trained encoder;
- complete model-state SHA-256;
- exact source embeddings;
- exact ordered target-calibration embeddings;
- exact final-evaluation embeddings;
- representation SHA-256;
- encoder configuration and parameter count;
- training history;
- mechanistic-analysis manifest identity;
- representation-geometry summaries.

Downstream frozen and SourceWeigher methods reuse that object rather than independently retraining nominally identical encoders.

This is what makes weighted-vs-unweighted transfer a clean intervention on the transfer policy rather than an uncontrolled comparison between two separately trained networks.

## Failure preservation

Method failures are part of the evidence record.

For every requested method, case and calibration budget, the result surface must contain a row. Rows may be:

- successful;
- expected unavailable under declared method semantics;
- failed with an explicit error.

A failed method cannot disappear from an aggregate plot.

Structural authority failures are more severe: if data identity, chronology, partition identity or processed-data fingerprint no longer match the frozen authority, the study aborts rather than emitting a degraded comparison.

## Paired-case promotion gate

A descriptive bundle is promotion-ready only if:

1. no requested method rows failed;
2. no unexpected unavailable rows exist;
3. every method retains the same subject/session case membership across every budget it claims to support.

SourceWeigher budget 0 is the only expected unavailable point in the initial ladder.

This prevents an apparently stronger high-budget curve from being produced by silently losing difficult cases.

## Metrics

Per-method evidence should include, where applicable:

- balanced accuracy;
- accuracy;
- ROC-AUC;
- expected calibration error;
- Brier score;
- negative log-likelihood;
- fit wall-clock time;
- trial inference latency;
- parameter count;
- resolved model configuration;
- training history;
- model seed;
- learned-state SHA-256;
- representation SHA-256;
- analysis-manifest fingerprint;
- representation geometry;
- exact authority/partition/calibration fingerprints.

No single metric is the product claim.

For BCI productization, one of the most economically legible summaries is **calibration saved**: how many labeled target trials are required to reach a declared performance/reliability operating point.

## Kumar2024 preregistration

For Kumar2024, the ladder preserves the dataset's original `GR` / `PAR` protocol grouping in every case authority and result row.

Reports should include:

- pooled descriptive summaries;
- cohort-specific summaries;
- target-session summaries;
- full-frontier AUC where defined;
- positive-budget adaptation AUC.

Repeated sessions from one participant are not independent biological replicates. Promoted inferential claims must account for that hierarchy rather than treating every session row as a separate subject.

## Artifact bundle

A complete ladder execution emits:

```text
study_manifest.json
split_authority.json
method_runs.json
results.csv
summary.json
report.md
artifact_hashes.json
```

`artifact_hashes.json` binds the machine-readable result surfaces. The human report is rendered from those same rows rather than maintained as an independent narrative source of truth.

## Public API boundary

Scientific ladder logic is package-owned.

The CLI should remain primarily responsible for:

- collecting a supported dataset through its upstream adapter;
- creating/restoring frozen authority;
- selecting declared method configurations;
- writing evidence artifacts.

Core public contracts include:

- `LongitudinalCaseAuthority`;
- `PreparedFrozenEncoderCase`;
- method specifications;
- `LadderRuntimeConfig`;
- `run_ladder_method(...)`;
- paired case-set helpers;
- calibration-frontier summaries;
- report/artifact generation.

This keeps the scientific comparison reusable outside one script.

## Execution

The public study runner is:

```bash
python scripts/evidence/run_moabb_model_ladder.py \
  --dataset kumar2024 \
  --subjects 1,2,3 \
  --budgets 0,1,2,5,10 \
  --methods csp-lda,eegnet,eeg-conformer,frozen-eegnet,frozen-eeg-conformer,sourceweigher-eegnet,sourceweigher-eeg-conformer \
  --output evidence/kumar2024-model-ladder
```

The corresponding GitHub Actions study workflow is manual only. Large public datasets are never downloaded as a normal push/PR side effect.

## External research extensions

External research packages, including the documented QuantumBCI extension, must reuse the same frozen `LongitudinalCaseAuthority` and evidence identities if they join this comparison. An extension may add a method lane, mechanism analysis, or resource accounting, but it may not create a second hidden split/calibration authority and still claim a matched comparison.

This keeps external experiments interoperable with neurOS Evidence without making those projects dependencies of the core runtime.

## Evidence presentation

A public result should keep the following identities visually adjacent to the performance curve:

```text
dataset/version
Git revision
method ID
model seed
processed-data SHA-256
authority fingerprint
partition fingerprint
calibration fingerprint
artifact verification
```

## Claim boundary

This ladder can establish controlled **offline real-dataset evidence** for longitudinal transfer and labeled recalibration burden.

It does not establish:

- live hardware reliability;
- online closed-loop improvement;
- clinical benefit;
- causal biological interpretation;
- foundation-model superiority without a verified executable foundation encoder;
- ORION superiority until an actual ORION EEG representation is evaluated under this same authority.

The long-term neurOS/ORION claim should be earned by moving the calibration frontier under increasingly difficult real-world shifts while preserving this evidence discipline.
