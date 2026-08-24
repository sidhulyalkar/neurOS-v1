# Longitudinal EEG Evidence Showcase

This study is the first executable real-dataset showcase for the neurOS evidence plane.

The goal is not to manufacture a flattering EEG accuracy number. The goal is to answer a deployment question that a BCI engineer, researcher, or buyer can understand immediately:

> **If this decoder worked yesterday, how much data does it need tomorrow?**

## First study: Kumar2024

The recommended first run is the MOABB `Kumar2024` motor-imagery dataset because it contains repeated separate-day sessions and an offline-to-online training progression.

Start with a small evidence run to validate the complete artifact chain:

```bash
python -m pip install -e "packages/neuros-foundation[evidence]"

python scripts/evidence/run_moabb_longitudinal.py \
  --dataset kumar2024 \
  --subjects 1,2,3 \
  --budgets 0,1,2,5,10 \
  --history-policy prior \
  --output evidence/kumar2024-csp-lda
```

Then scale to the predeclared subject set only after the artifact identities and dataset chronology have been inspected.

## Two different questions, two explicit policies

### `prior` — prospective-style longitudinal evidence

This is the default.

For a target session `S3`:

```text
S1          S2          S3                         S4          S5
|-----------|-----------|--------------------------|-----------|
   HISTORY      HISTORY     CALIBRATION + EVAL        EXCLUDED

fit source data: S1 + S2
calibration:     declared subset of S3
final test:      frozen disjoint subset of S3
future data:     S4 + S5 never enters fitting
```

The first observed session cannot be evaluated under this policy because no prior history exists.

This is the appropriate default for claims such as:

- next-session robustness;
- next-day calibration burden;
- prospective adaptation;
- degradation as time since initial calibration grows.

First-observed upstream metadata order is treated as chronology by the software. Before publishing or promoting a result, verify that ordering assumption against the dataset documentation and preserve the exact upstream data/version identity.

### `all-other` — symmetric cross-session evaluation

For the same target session `S3`:

```text
S1          S2          S3                         S4          S5
|-----------|-----------|--------------------------|-----------|
   TRAIN        TRAIN      CALIBRATION + EVAL          TRAIN       TRAIN
```

This mode is useful for conventional cross-session comparison, but it may learn from recordings collected after the target session. It must never be labeled as prospective or next-session deployment evidence.

Run it explicitly:

```bash
python scripts/evidence/run_moabb_longitudinal.py \
  --dataset kumar2024 \
  --subjects 1,2,3 \
  --history-policy all-other \
  --output evidence/kumar2024-csp-lda-all-other
```

## Fixed evaluation identity

Within the held-out session, neurOS freezes two disjoint sets once:

```text
held-out session
      |
      +--> ordered calibration pool
      |
      +--> frozen evaluation set
```

The evaluation set never changes when calibration budget changes.

For a two-class task:

```text
0 / class   -> source history only
1 / class   -> source history + 2 calibration trials
2 / class   -> source history + 4 calibration trials
5 / class   -> source history + 10 calibration trials
10 / class  -> source history + 20 calibration trials
```

All five models are evaluated on exactly the same frozen held-out examples.

This prevents a calibration curve from quietly becoming easier as more calibration examples are consumed.

## Initial baseline

The first benchmark intentionally uses:

**CSP + Linear Discriminant Analysis**

This gives neurOS a transparent floor before introducing higher-capacity models.

The progression should be:

```text
CSP + LDA
    |
    +--> EEGNet
    +--> EEG-Conformer
    +--> frozen foundation representation + matched readout
    +--> SourceWeigher transfer
    +--> ORION EEG representation, only after an explicit EEG contract exists
```

Every method must consume the same frozen split and calibration identities.

## Artifact contract

A completed study emits:

```text
evidence/kumar2024-csp-lda/
├── study_manifest.json
├── results.csv
├── summary.json
├── report.md
└── artifact_hashes.json
```

### `study_manifest.json`

Contains the evidence authority:

- dataset/source card;
- Git/package/platform identity;
- paradigm and event semantics;
- chronological or symmetric history policy;
- upstream-observed session order;
- source sessions for every target session;
- partition fingerprints;
- fixed calibration/evaluation split fingerprints;
- requested calibration budgets;
- preprocessing and model identity;
- explicit limitations.

### `results.csv`

One row per:

```text
subject × held-out session × calibration budget
```

with at least:

- accuracy;
- balanced accuracy;
- ROC-AUC where valid;
- fit time;
- inference time per trial;
- partition fingerprint;
- calibration-split fingerprint.

### `summary.json`

Contains the aggregate calibration frontier used by the report/UI.

### `report.md`

Human-readable rendering of the same result rows. It is a view, not a second source of truth.

### `artifact_hashes.json`

SHA-256 identities for the evidence files.

## Evidence Console design

The public visualization should consume the artifact bundle directly.

### Panel 1 — Session chronology

Show a horizontal session timeline:

```text
PAST                              TARGET                         FUTURE
S1 -------- S2 -------- S3 -------- S4 -------- S5 -------- S6
history     history     history     held out      excluded    excluded
                                      |
                              calibration | evaluation
```

Selecting a target session should reveal:

- which sessions were allowed into source training;
- how many calibration examples were consumed;
- the immutable evaluation fingerprint;
- any excluded future sessions.

The future-excluded region should be visibly different from train data. This makes leakage semantics inspectable rather than buried in a methods paragraph.

### Panel 2 — Calibration frontier

Primary chart:

```text
held-out balanced accuracy
^
|                         ORION / best validated method
|                  *------*
|            *-----
|      *-----         foundation representation
|  *---
| *        CSP+LDA
+---------------------------------> labeled calibration / class
 0        1       2        5       10
```

Do not display only the final point.

Useful derived quantities:

- zero-calibration score;
- largest-budget score;
- area under the calibration frontier;
- labels needed to reach a predeclared performance threshold;
- adaptation wall-clock required to reach that threshold.

The economically legible BCI metric is often **calibration saved**, not raw accuracy gained.

### Panel 3 — Session drift

Display zero-calibration performance versus session index/time.

This reveals whether a method is actually stable or merely easy to recalibrate.

### Panel 4 — Evidence identity

Keep the following visible beside every chart:

```text
dataset revision
Git revision
method/model revision
history policy
partition fingerprint
calibration split fingerprint
artifact hash status
```

A screenshot without these identities should not be treated as promoted neurOS evidence.

## Comparison promotion rules

A new model may be added to the longitudinal figure only when:

1. it consumes the exact same partition/calibration manifests;
2. preprocessing fit boundaries are declared;
3. hyperparameter/model selection does not inspect final evaluation examples;
4. random seeds and uncertainty are reported where relevant;
5. failures are retained in the artifact rather than filtered from the plot;
6. model identity and weight/artifact hashes are pinned;
7. the result is clearly labeled offline real-dataset evidence, not hardware or clinical qualification.

## Next implementation sequence

After the CSP+LDA runner is qualified:

1. freeze one small Kumar2024 evidence bundle;
2. add EEGNet and EEG-Conformer under identical manifests;
3. add representation-only foundation-model comparisons using matched linear readouts;
4. add SourceWeigher using prior sessions as candidate sources and the target-session calibration pool only for allowed adaptation;
5. measure subject/session leakage and representation geometry;
6. scale the predeclared study across participants;
7. render the final evidence bundle in the neurOS Evidence Console;
8. repeat the same calibration-frontier idea on FALCON with its native chronological day split.

The showcase succeeds when a viewer can understand, in seconds, **what data the model was allowed to know, how badly it drifted, how much calibration recovered it, and whether every number can be reproduced.**
