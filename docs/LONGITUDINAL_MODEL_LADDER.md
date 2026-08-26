# Longitudinal EEG Model Ladder

This document defines the authoritative multi-method longitudinal EEG comparison layer built on the prospective split and calibration authority in neurOS.

The governing rule is simple:

> One frozen data authority, many methods.

Every method must consume the same serialized source-history and target-partition identity. Method-specific convenience is never allowed to silently redefine the scientific question.

For predetermined, non-state-selecting model comparisons, the existing v1 `LongitudinalCaseAuthority` remains the authoritative contract. Adaptive efficacy studies in which held-out data influence retain/rollback or other state selection must use the additive v2 `ThreeWayLongitudinalCaseAuthority` described below.

## Why this exists

A longitudinal BCI benchmark becomes misleading quickly if each model is allowed to choose a different target test set, use future sessions under a prospective claim, estimate adaptation state from final evaluation samples, or recreate nominally identical frozen encoders independently.

The ladder therefore separates three identities:

1. **data identity**: processed-data SHA-256 plus exact source and target partition indices;
2. **learned-state identity**: SHA-256 of the complete trained model state, including registered buffers;
3. **representation identity**: SHA-256 over the exact frozen source/target embedding tensors used by a downstream method.

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

## V1 longitudinal authority

Each subject / target-session case in the current ladder is serialized as a `LongitudinalCaseAuthority` before model fitting.

The v1 authority binds:

- subject identity;
- target session;
- observed chronology;
- source-history indices;
- ordered target calibration indices;
- immutable evaluation indices;
- processed-data SHA-256;
- partition fingerprint;
- calibration fingerprint;
- case fingerprint;
- dataset-specific metadata such as original protocol/cohort.

A method restores and validates that authority before fitting. Changed processed EEG bytes, sample order, labels, grouping metadata or case identity fail before a model is trained.

Existing v1 evidence bundles remain valid and replayable. V1 is suitable when the evaluation rows never influence model-state selection, threshold selection, hyperparameters, or the definition of the reported operating point.

## Prospective history semantics

The default evidence question is prospective next-session transfer.

For held-out target session `S_k`, only sessions observed before `S_k` may enter the source history. Later sessions are future-excluded.

A conventional all-other-session analysis may still be useful as a secondary symmetric benchmark, but it must not be narrated as prospective evidence.

## Fixed target evaluation identity in v1

For each target session, the v1 ladder creates one deterministic balanced calibration pool and one immutable evaluation set.

Increasing the calibration budget changes only how much of the calibration pool a method may use. The evaluation indices never move.

This avoids an especially common source of false calibration gains: testing larger-budget models on a smaller or easier remainder of the same held-out session.

The v1 evaluation set is a valid final evaluation only when it does not influence model or policy selection. If those rows are used to decide retain versus rollback, choose thresholds, stop training, or select a model, they become qualification/state-selection data and cannot later be described as untouched final-assessment evidence.

## Nested calibration budgets

Calibration examples are deterministic and nested per class.

If the budgets are:

```text
0, 1, 2, 5, 10
```

the budget-2 set must contain the budget-1 examples, the budget-5 set must contain budget 2, and so on.

The resulting frontier measures the marginal value of additional labeled target calibration under a fixed held-out identity.

## Zero-calibration semantics

Target-independent methods can report budget 0.

A target-dependent SourceWeigher lane is explicitly unavailable at budget 0 because no legal target calibration observations exist. neurOS does not substitute held-out evaluation examples as unlabeled target observations.

This produces two frontier summaries:

- **full calibration-frontier AUC**, only for methods genuinely defined from zero calibration onward;
- **positive-budget adaptation AUC**, for methods whose adaptation begins only after target calibration becomes available.

An unavailable zero-calibration point is not a failure.

For future state-changing v2 studies, budget 0 is a frozen no-update baseline. It must not be represented as a fake adaptation event with an empty adaptation set.

## Adaptive efficacy extension: v2 three-way authority

State-changing methods require a stronger contract when qualification evidence can determine which state is ultimately reported.

The additive v2 `ThreeWayLongitudinalCaseAuthority` freezes:

```text
prior source history
        |
        +--> calibration pool
        |      nested balanced budgets
        |      state may change
        |
        +--> qualification set
        |      state read-only
        |      may choose retain/rollback
        |
        +--> final-assessment set
               state/policy immutable
               scientific score only
```

The corresponding `ThreeWayCalibrationSplit` preserves the original chronological source partition while dividing the held-out target deployment unit into calibration, qualification, and final-assessment roles exactly once.

V2 enforces:

- calibration, qualification, and final-assessment rows are pairwise disjoint;
- the three roles cover the target partition exactly;
- qualification and final assessment preserve complete target class support;
- calibration pools are class-consistent;
- calibration budgets remain balanced and nested;
- selected calibration rows use canonical source-row execution order;
- qualification and final-assessment sets are fixed across every calibration budget;
- each calibration budget, qualification set, and final-assessment set has a SHA-256 identity bound to the processed-data SHA-256;
- malformed serialized numeric fields are rejected rather than coerced;
- semantic metadata must be deterministic and finite;
- declared prospective history is checked before use;
- `restore(data)` revalidates processed neural bytes, group/label support, chronology, partition identity, and three-way split identity.

A serialized authority by itself is an identity declaration. It becomes dataset-backed evidence only after successful restore against the processed dataset.

### Matched adaptive comparison protocol

A future governed adaptive ladder should give the same restored v2 case to every method:

```text
same processed EEG
same source chronology
same calibration membership
same qualification rows
same final-assessment rows
        |
        +--> conventional adaptation baseline
        +--> governed Hebbian predictive coding
        +--> ORION personalization
        +--> any frozen/no-update control
```

For each positive calibration budget:

1. adapt only on the exact authorized calibration indices;
2. freeze learning before qualification;
3. apply one predeclared retain/rollback policy on the full qualification set;
4. freeze the selected state, policy, thresholds, and hyperparameters;
5. score the complete final-assessment set once.

The final-assessment partition must not influence architecture choice, learning rate, epoch count, calibration budget, early stopping, retain/rollback, threshold selection, metric selection, or operating-point definition.

Only this final stage can support an adaptive efficacy or calibration-saved claim after state selection has occurred.

## PreparedFrozenEncoderCase

Frozen representation work is prepared once per case, encoder architecture and seed.

The prepared state contains:

- the trained encoder;
- complete model-state SHA-256;
- exact source embeddings;
- exact ordered target-calibration embeddings;
- exact held-out evaluation embeddings;
- representation SHA-256;
- encoder configuration and parameter count;
- training history;
- mechanistic-analysis manifest identity;
- representation-geometry summaries.

Downstream frozen and SourceWeigher methods reuse that object rather than independently retraining nominally identical encoders.

This is what makes weighted-vs-unweighted transfer a clean intervention on the transfer policy rather than an uncontrolled comparison between two separately trained networks.

The current `PreparedFrozenEncoderCase` belongs to the v1 ladder. A future adaptive-v2 extension must materialize qualification and final-assessment representations separately rather than relabeling one v1 evaluation tensor after state selection.

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

For a future v2 adaptive bundle, promotion must additionally require identical qualification and final-assessment identities across all compared methods and calibration budgets.

This prevents an apparently stronger high-budget curve from being produced by silently losing difficult cases or moving the test set.

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

For v2 adaptive studies, evidence should additionally carry qualification-set SHA-256, final-assessment-set SHA-256, calibration-budget SHA-256, pre/post mutable-state identities, and the retain/rollback outcome identity.

No single metric is the product claim.

For BCI productization, one of the most economically legible summaries is **calibration saved**: how many labeled target trials are required to reach a declared performance/reliability operating point on untouched final assessment.

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

A complete current ladder execution emits:

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

A future v2 adaptive bundle should preserve the v2 authority itself plus explicit state-selection and final-assessment records rather than overwriting the existing v1 artifact schema.

## Public API boundary

Scientific ladder logic is package-owned.

The CLI should remain primarily responsible for:

- collecting a supported dataset through its upstream adapter;
- creating/restoring frozen authority;
- selecting declared method configurations;
- writing evidence artifacts.

Core public contracts include:

- `LongitudinalCaseAuthority` for the current v1 ladder;
- `ThreeWayCalibrationSplit` and `ThreeWayLongitudinalCaseAuthority` for adaptive v2 studies;
- `PreparedFrozenEncoderCase`;
- method specifications;
- `LadderRuntimeConfig`;
- `run_ladder_method(...)`;
- paired case-set helpers;
- calibration-frontier summaries;
- report/artifact generation.

This keeps the scientific comparison reusable outside one script.

## Execution

The current public v1 study runner is:

```bash
python scripts/evidence/run_moabb_model_ladder.py \
  --dataset kumar2024 \
  --subjects 1,2,3 \
  --budgets 0,1,2,5,10 \
  --methods csp-lda,eegnet,eeg-conformer,frozen-eegnet,frozen-eeg-conformer,sourceweigher-eegnet,sourceweigher-eeg-conformer \
  --output evidence/kumar2024-model-ladder
```

The corresponding GitHub Actions study workflow is manual only. Large public datasets are never downloaded as a normal push/PR side effect.

The v2 authority is currently qualified as a software/evaluation contract. It is not yet wired into this public real-dataset runner, and its existence must not be narrated as a real adaptive result.

## External research extensions

External research packages, including the documented QuantumBCI extension, must reuse the same frozen authority and evidence identities if they join a comparison. An extension may add a method lane, mechanism analysis, or resource accounting, but it may not create a second hidden split/calibration authority and still claim a matched comparison.

For adaptive comparisons that perform state selection, this means reusing the same restored v2 authority, not merely the same v1 target session.

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
calibration fingerprint / budget SHA-256
qualification-set SHA-256 when applicable
final-assessment-set SHA-256 when applicable
artifact verification
```

## Claim boundary

The current v1 ladder can establish controlled **offline real-dataset evidence** for longitudinal transfer and labeled recalibration burden when its frozen evaluation set is not used for state selection.

The v2 authority can establish the **evaluation/process integrity** needed for future adaptive efficacy studies. By itself it does not establish that adaptation works.

Neither establishes:

- live hardware reliability;
- online closed-loop improvement;
- clinical benefit;
- causal biological interpretation;
- foundation-model superiority without a verified executable foundation encoder;
- ORION superiority until an actual ORION EEG representation/adaptation method is evaluated under a restored matched authority and untouched final assessment.

The long-term neurOS/ORION claim should be earned by moving the calibration frontier under increasingly difficult real-world shifts while preserving this evidence discipline.
