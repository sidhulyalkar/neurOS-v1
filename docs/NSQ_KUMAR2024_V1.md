# NSQ Kumar2024 v1

NSQ Kumar2024 v1 is the first peer-facing real-data study for the neurOS Neural System Qualification program.

Its question is deliberately narrower than “which EEG model is best?”

> **Under identical prospective longitudinal authority, how much held-out task performance can each method achieve as a function of per-user labeled calibration cost?**

The first comparison is intentionally external. ORION does not enter a superiority comparison until the external baseline protocol and artifacts are frozen.

## What this study is

This is a new neurOS longitudinal re-evaluation of the **MOABB Kumar2024 bar-feedback subset**.

Kumar2024 contains 18 BCI-naive participants recorded over six separate-day sessions for left-hand versus right-hand motor imagery. MOABB exposes the bar-feedback runs and excludes the car-racing runs from the original study. The original subjects 1–9 used Generic Recentering (GR), while subjects 10–18 used Personally Assisted Recentering (PAR).

The study preserves GR/PAR as participant metadata because the recorded trajectories arose under different original training histories.

## What this study is not

It is **not** a reproduction of the original online GR/PAR intervention.

It does not establish:

- physiological mechanism;
- physical hardware reliability;
- online BCI efficacy;
- clinical benefit;
- ORION superiority;
- that a synthetic, offline, or software-qualified result automatically transfers to a live participant.

## Authority chain

The implementation deliberately composes existing production authorities instead of creating another benchmark framework:

```text
ORION DatasetLineage
        |
        v
MOABB processed MNE Epochs contract
        |
        v
GroupedEvaluationData
        |
        v
LongitudinalCaseAuthority
        |
        v
QualificationProtocolSpec
        |
        v
NSQ Runner v1
 external factory -> fresh model per calibration budget
        |
        v
ClassificationScorecardV1
        |
        v
failure-preserving content-addressed study bundle
```

`neuros-foundation` does not depend on ORION for this composition. The user-facing `neuros.evidence.kumar2024` module is the top-level orchestration surface.

## Frozen longitudinal design

For each participant, the observed session chronology must be exactly:

```text
0, 1, 2, 3, 4, 5
```

Session `0` is available only as historical source data. The five online-session indices are evaluated prospectively:

```text
target 1 <- source 0
target 2 <- source 0,1
target 3 <- source 0,1,2
target 4 <- source 0,1,2,3
target 5 <- source 0,1,2,3,4
```

Within each target session, calibration and final assessment are split once. Calibration examples are nested so every method receives the exact same labeled frontier:

```text
0, 1, 2, 5, 10 examples / class
```

The final assessment indices remain identical across budgets and methods. If an actual participant/session case cannot support the complete frontier, the promoted study fails closed rather than silently producing a different right edge for that participant.

Unlabeled target adaptation is not authorized in v1. It requires the separate observation-role work tracked by issue #81.

## Preprocessing authority

The requested preprocessing is frozen as:

- MOABB `LeftRightImagery`;
- 8–30 Hz by default;
- no resampling by default;
- no additional neurOS normalization;
- `return_epochs=True`.

The last point is important. The study retains the actual processed MNE `Epochs` object long enough to content-address its observed:

- channel order;
- channel types;
- sampling rate;
- number of samples per epoch;
- epoch start/end times;
- event-ID mapping.

Only then are the values converted into the shared `GroupedEvaluationData` array contract. MNE `Epochs.get_data(units=None)` returns channel-type-specific default SI units; for EEG this study therefore binds the processed values as volts. Any later microvolt conversion or standardization used by a deep baseline must be declared as a separate fixed or fitted preprocessing authority.

The MNE `event_id` mapping is retained verbatim. Task labels (`left_hand` / `right_hand`) are separately bound to the processed row order by `GroupedEvaluationData` and its processed-data SHA rather than being guessed from event-ID display names.

This fixes a provenance weakness in the older array-only MOABB runners, which could identify array geometry but could not prove the processed channel order supplied to a decoder.

The preprocessing authority is not a raw-data checksum. `DatasetLineage.content_sha256` remains unset unless the downloaded raw corpus is hashed under a defined canonical rule.

## Dataset lineage

The ORION `DatasetLineage` records at least:

- dataset ID `moabb-kumar2024`;
- installed MOABB version;
- dataset class `moabb.datasets.Kumar2024`;
- paper DOI `10.1093/pnasnexus/pgae076`;
- data DOI `10.5281/zenodo.10694880`;
- actual requested participant identity set;
- session identities `0..5`;
- upstream 22-channel / 512-Hz EEG acquisition assumptions;
- CPz reference and AFz ground as declared upstream;
- exact processed epoch contract;
- preprocessing history;
- the explicit MOABB bar-feedback-only scope.

Lineage completeness is `partial` until exact raw content and all ancestry are independently established.

## External methods v1

The promoted Kumar2024 study surface supports:

1. direct MNE CSP + scikit-learn LDA;
2. direct pyRiemann covariance + tangent-space + scikit-learn logistic regression;
3. direct upstream Braindecode EEGNet under the frozen neural-training authority below.

The generic Braindecode adapter can still probe other installed upstream architectures, including EEGConformer when available, but **EEGConformer is not a promoted Kumar2024 efficacy method yet**. It needs its own model-appropriate training authority before entering this study.

The Riemannian comparator is frozen as sample covariance (`scm`) -> `TangentSpace(metric="riemann", tsupdate=False)` -> L2 `LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)`. `tsupdate=False` is part of the evidence contract so evaluation-batch composition cannot alter the tangent reference.

The EEGNet v1 training authority is frozen before any promoted EEGNet final-assessment result is inspected:

- the same NSQ/MOABB 8–30 Hz epoch arrays used by the classical methods, with no hidden model-specific normalization;
- direct upstream `braindecode.models.EEGNet` + `EEGClassifier`;
- `torch.optim.Adam`, learning rate `0.000625`, weight decay `0`;
- batch size `64`;
- maximum `1000` epochs as a ceiling;
- `skorch.dataset.ValidSplit(0.2, stratified=False, random_state=17011)` inside the NSQ-authorized fit set;
- early stopping on validation loss with patience `300`, zero improvement threshold, and restoration of the best observed validation-loss module state;
- model seed `31415`, separate from the study split seed and analysis seed;
- deterministic cuDNN flags when CUDA is available;
- exact validation membership recorded as relative indices plus SHA-256 in learned-state evidence;
- the restored inference tensor/buffer state content-addressed without pickle.

The canonical CLI pilot still defaults to **MNE CSP + LDA only**. The exact-main archival classical workflow explicitly runs CSP + RG together. EEGNet must be requested explicitly and should not be executed against promoted final-assessment data until this training authority itself has passed exact-head qualification.

These methods participate through the same NSQ factory protocol. neurOS does not substitute its own model implementation behind an external method name. Every calibration budget creates a **fresh external model instance**; warm-starting one frontier point from another is forbidden.

## Metrics

The production `ClassificationScorecardV1` is part of the frozen protocol identity.

Primary:

- balanced accuracy.

Secondary:

- accuracy;
- binary ROC AUC when valid probability semantics permit it;
- Brier score when probability output exists;
- expected calibration error when probability output exists.

Probability-dependent metrics remain explicitly unavailable for label-only methods. They are never synthesized from labels.

## Statistical unit

The participant is the independent inferential unit.

Sessions and trials are not treated as additional independent people. The study therefore:

1. aggregates repeated session results within participant;
2. estimates uncertainty by participant bootstrap;
3. reports paired method differences only on matched participant/session/budget cases, then aggregates those differences within participant before inference;
4. reports GR/PAR summaries descriptively unless a separate inferential cohort study is preregistered;
5. reports normalized area under the performance-versus-label-budget frontier as a calibration-efficiency summary.

Failed, unavailable, OOM, skipped, and nonconvergent cases remain in the raw result frontier and in failure counts.

## Profiles

### Pilot

The default command is a provenance/execution pilot:

```text
subjects = 1,10
split seed = 2026
budgets = 0,1,2,5,10 / class
default method = MNE CSP + LDA only
```

The stored pilot configuration now carries the frozen EEGNet training authority above, but no deep model is part of the default pilot method set. This separates *method authority* from *whether a final-assessment run has been authorized*.

One participant comes from each original GR/PAR cohort. This pilot exists to validate data identity, runtime behavior, artifact semantics, and dependency feasibility. It is **not** the headline model-comparison claim.

### Promoted all-subject comparison: intentionally blocked

The code contains an internal all-18-subject reference configuration for feasibility work, but it is **not** exposed as a CLI profile and is **not** the promoted preregistration.

Before any headline all-subject comparison, issue #27 must be encoded explicitly:

```text
shared split seeds = 2026, 3407, 9109
primary study endpoint = paired normalized balanced-accuracy calibration-frontier AUC
neural model seeds = predeclared and reported separately from split variation
```

The training budget must also be frozen before final-assessment results are inspected. A convenient all-subject run must never masquerade as the promoted study.

## Installation

Core real-data evidence profile:

```bash
pip install "neuros[evidence]"
```

Full external deep-baseline profile on supported Python versions:

```bash
pip install "neuros[evidence-braindecode]"
```

From a repository checkout, equivalent workspace installs are used by qualification CI.

## Execute

Pilot:

```bash
neuros-nsq-kumar2024 \
  --profile pilot \
  --output /tmp/nsq-kumar2024-pilot
```

A method-restricted plumbing run is allowed, but its distinct configuration SHA prevents it from masquerading as the full study:

```bash
neuros-nsq-kumar2024 \
  --profile pilot \
  --methods mne-csp-lda \
  --output /tmp/nsq-kumar2024-csp
```

## Bundle

A completed study writes:

```text
study_manifest.json
case_authorities.json
case_results.json
results.csv
analysis.json
report.md
artifact_hashes.json
```

The bundle binds:

- configuration SHA-256;
- dataset-lineage SHA-256;
- preprocessing-authority SHA-256;
- protocol and metric-scorecard identity;
- every case-authority SHA-256;
- every method-spec SHA-256;
- every NSQ case-result SHA-256;
- package/runtime versions;
- processed signal descriptors;
- all result and failure rows;
- participant/session/cohort metadata;
- participant-level analysis generated from those rows.

## Canonical real-data execution

The pull-request data job is a pre-merge feasibility check because GitHub checks out a synthetic PR merge ref. The archival CSP pilot is produced by `.github/workflows/nsq-kumar2024-study.yml` from an exact durable `main` commit. The workflow runs once when first added to `main` and remains manually dispatchable afterward. It asserts that the bundle records the exact workflow SHA, all ten case authorities use literal split seed `2026`, all five budgets are present, and the expected 50 CSP result rows are preserved before uploading the artifact.

## Verify without retraining

```bash
neuros-nsq-kumar2024 \
  --verify-only \
  --output /tmp/nsq-kumar2024-v1
```

Verification recomputes every managed file digest and the bundle root SHA-256. It does not retrain a model or silently regenerate results.

## Promotion rule

Do not admit ORION, SourceWeigher, foundation representations, or mechanistic claims to the public superiority frontier until this external baseline bundle is frozen and qualified.

After that point each candidate must cross the exact same participant/session/calibration/final-assessment authority. An internally convenient method does not get a friendlier split.
