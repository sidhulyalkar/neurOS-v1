# Longitudinal EEG Evidence Showcase

The longitudinal EEG showcase is the first public-facing demonstration of the neurOS evidence plane.

The central question remains deliberately concrete:

> **If this neural decoder worked on prior sessions, how much new target-session data does it need now?**

The showcase now has two executable layers:

1. a transparent CSP + LDA evidence floor;
2. an authoritative multi-method model ladder under the exact same deployment semantics.

For the implementation contract and seven comparison lanes, see [Longitudinal Model Ladder](LONGITUDINAL_MODEL_LADDER.md).

## Flagship study: Kumar2024

Kumar2024 remains the recommended first longitudinal EEG study because it provides six separate-day sessions from 18 BCI-naive participants, with MOABB session `0` representing the offline session and sessions `1` through `5` the chronological online sessions.

The original experiment also contains two different training/adaptation cohorts:

- MOABB subjects 1–9: `GR`;
- MOABB subjects 10–18: `PAR`.

neurOS preserves that cohort identity in every promoted case rather than pooling the trajectories blindly.

The preregistered promoted frontier in Issue #27 uses:

```text
0, 1, 2, 5, 10 labeled target examples / class
```

with prospective-style prior-session history.

## Prospective chronology

`history_policy = prior` is the primary policy.

For target session `S3`:

```text
S0             S1             S2             S3             S4
|--------------|--------------|--------------|--------------|
  SOURCE          SOURCE         TARGET          FUTURE
                                  |   |           EXCLUDED
                            calibration eval
```

Only sessions preceding the target may enter source fitting.

The target session is partitioned once into:

```text
target session
   |
   +--> ordered calibration pool
   |
   +--> immutable final evaluation set
```

Future sessions never enter fitting under a prospective claim.

`history_policy = all-other` remains available as a separately labeled symmetric cross-session sensitivity analysis. It may use sessions recorded after the target and therefore is not next-session deployment evidence.

## One authority, every method

The multi-method showcase no longer relies on each model recreating a split from configuration.

For every subject/target-session case neurOS serializes a `LongitudinalCaseAuthority` containing the actual source, calibration and evaluation indices plus a hash of the processed EEG.

```text
processed EEG + labels + groups
             |
             v
 LongitudinalCaseAuthority
             |
    +--------+---------+----------+
    |        |         |          |
 CSP+LDA   EEGNet   Conformer   frozen transfer
                                  |
                             SourceWeigher
```

A method must restore this authority before fitting.

If processed samples or identity change, the benchmark fails before training rather than producing an incomparable result.

## Current executable ladder

The Evidence Console can now render:

- CSP + LDA;
- EEGNet;
- EEG-Conformer;
- frozen EEGNet + matched logistic readout;
- frozen EEG-Conformer + matched logistic readout;
- SourceWeigher on the exact frozen EEGNet representation;
- SourceWeigher on the exact frozen EEG-Conformer representation.

The frozen and SourceWeigher versions of one encoder share the exact same trained encoder state and representation tensors. The evidence bundle records both the learned-state SHA-256 and representation SHA-256.

That lets the weighted-vs-unweighted comparison isolate the transfer strategy instead of quietly retraining two different encoders.

## SourceWeigher is not secretly transductive

SourceWeigher may use:

```text
prior-session frozen embeddings
              +
declared target calibration embeddings
```

It may not inspect final evaluation embeddings when estimating target similarity.

At labeled calibration budget `0`, target-dependent SourceWeigher is therefore explicitly unavailable.

The UI should display that absence rather than drawing a fabricated zero-shot point.

## Evidence identity stack

Every promoted neural operating point should expose three different identities:

### Data identity

- dataset/version;
- processed-data SHA-256;
- source/calibration/evaluation indices;
- authority, partition and calibration fingerprints.

### Learned-model identity

- method configuration;
- model seed;
- parameter count;
- complete trained `state_dict` SHA-256;
- analysis-manifest fingerprint.

### Representation identity

For frozen-transfer lanes:

- exact source embedding SHA contribution;
- exact target calibration-pool embedding SHA contribution;
- exact final evaluation embedding SHA contribution;
- combined `representation_sha256`;
- `encoder_state_fingerprint`.

A configuration hash is not presented as a substitute for a learned-model hash.

## The Evidence Console

The strongest public interface is not a generic leaderboard. It is a linked explanation of **chronology, recalibration burden, source trust and provenance**.

### 1. Chronology ribbon

```text
PAST                               TARGET                        FUTURE
S0 -------- S1 -------- S2 -------- S3 -------- S4 -------- S5
source      source      source      held out     excluded    excluded
                                     |
                              calibration | eval
```

Selecting a target should reveal:

- source sessions;
- excluded future sessions;
- calibration examples consumed;
- immutable evaluation fingerprint;
- GR/PAR cohort for Kumar2024.

### 2. Calibration frontier

```text
balanced accuracy
^
|                         method A
|                  *------*
|             *----        method B
|        *----
|   *----                    CSP+LDA
+------------------------------------> labels / class
   0    1    2       5          10
```

Do not reduce the result to the largest-budget endpoint.

The primary Kumar comparison for methods defined at zero target calibration is normalized area under the full `0 -> 10` frontier.

Target-dependent adaptation methods additionally receive a separate **positive-budget adaptation AUC**. These two quantities must remain visibly distinct.

### 3. Source trust panel

For SourceWeigher show:

```text
S0  ███████░  0.41
S1  ███░░░░░  0.18
S2  ████████  0.41
```

alongside:

- effective sample size;
- entropy;
- largest source weight;
- moment-matching residual;
- convergence diagnostics.

This turns adaptation from a black box into an inspectable statement about which historical neural states the system considered useful.

### 4. Drift and cohort views

The same artifact bundle should provide:

- pooled frontier;
- GR frontier;
- PAR frontier;
- zero-calibration performance by target-session index;
- failure rate by method/session;
- fit/adaptation time and inference latency.

These are views of one authority, not separately maintained analyses.

### 5. Model and representation diagnostics

Beside task score expose:

- model parameter count;
- learned-state SHA;
- training wall-clock;
- trial inference latency;
- representation effective rank;
- feature variance;
- mean norm;
- anisotropy proxy.

This is particularly important because the current EEGNet and EEG-Conformer baselines are not capacity matched. A performance difference must not automatically be narrated as an attention-vs-convolution mechanism.

### 6. Reproducibility drawer

Keep visible or one click away:

```text
dataset revision
Git revision
package versions
history policy
split seed
model seed
processed-data SHA
model-state SHA
representation SHA when applicable
authority fingerprint
artifact verification status
```

A screenshot without evidence identity is a visualization, not a promoted neurOS result.

## Artifact bundle

The multi-method study emits:

```text
study_manifest.json
split_authority.json
method_runs.json
results.csv
summary.json
report.md
artifact_hashes.json
```

The `report.md` and eventual web Evidence Console must be rendered from these machine-readable artifacts.

The scientific record and polished demonstration must not become two hand-maintained surfaces.

## Failure preservation

Failures remain part of the evidence.

If a model fails for a case/seed, the result table contains failed operating points instead of silently deleting that subject/session.

A descriptive bundle passes its promotion gate only when:

- method failures are absent;
- unexpected unavailable points are absent;
- every method preserves identical case membership across the calibration budgets it claims to support.

## How to run the two layers

### Transparent floor

```bash
python scripts/evidence/run_moabb_longitudinal.py \
  --dataset kumar2024 \
  --subjects 1,10 \
  --budgets 0,1,2,5,10 \
  --history-policy prior \
  --output evidence/kumar-csp-pilot
```

### Full model ladder

Install the sibling SourceWeigher workspace package explicitly, then the foundation evidence/ladder profile:

```bash
python -m pip install -e packages/neuros-core
python -m pip install -e "packages/neuros-models[pytorch]"
python -m pip install -e packages/neuros-sourceweigher
python -m pip install -e "packages/neuros-foundation[evidence,ladder]"

python scripts/evidence/run_moabb_model_ladder.py \
  --dataset kumar2024 \
  --subjects 1,10 \
  --methods csp-lda,eegnet,eeg-conformer,frozen-eegnet,frozen-eeg-conformer,sourceweigher-eegnet,sourceweigher-eeg-conformer \
  --model-seeds 101 \
  --budgets 0,1,2,5,10 \
  --history-policy prior \
  --split-seed 2026 \
  --output evidence/kumar-model-ladder-pilot
```

The GitHub Actions workflow **neurOS longitudinal EEG model ladder** provides the same real-dataset study as an explicit manual run and uploads the complete evidence bundle.

## What comes next

Once the Kumar provenance pilot is inspected and the exact study dependencies are frozen:

1. execute the complete CSP floor across all preregistered cases;
2. scale the task-decoder and frozen-transfer lanes under the same authorities;
3. add subject/session leakage and cross-session CKA under those same frozen examples;
4. add montage/channel perturbation robustness;
5. connect the existing mechanistic evidence-pack format to the same case authority;
6. move the calibration-frontier concept to FALCON's native chronological cross-day splits;
7. admit an ORION EEG representation only after ORION has an explicit EEG input/token contract.

The public claim should grow only as the evidence grows.

The showcase succeeds when a viewer can understand in seconds **what the system was allowed to know, how the neural distribution moved, how much calibration was needed, which prior sources were trusted, and exactly what data/model/code produced the result.**