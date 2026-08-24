# Longitudinal EEG Model Ladder

The longitudinal model ladder is the first neurOS benchmark where multiple neural decoding strategies are forced to answer the **same deployment question on the same samples**.

The benchmark is not a model zoo. It is a controlled comparison of how much useful decoding performance survives a session shift and how much target-session calibration is required to recover it.

## The authority comes before the model

For each subject and target session, neurOS first creates a `LongitudinalCaseAuthority`.

That object freezes:

```text
prior source indices
        |
        +----------------------+
                               |
target session                 |
  +--> ordered calibration pool|
  +--> immutable final eval    |
                               v
                    LongitudinalCaseAuthority
                               |
                 processed-data SHA-256
                 partition fingerprint
                 calibration fingerprint
                 exact integer indices
                               |
          +--------------------+--------------------+
          |                    |                    |
        CSP+LDA              EEGNet          EEG-Conformer
          |                    |                    |
          +--------- frozen representation lanes --+
                               |
                         SourceWeigher
```

Every method restores and validates that authority before fitting.

A method cannot silently regenerate a convenient split. If processed EEG bytes, labels, group order, or sample identity change, authority restoration fails before model training.

## Supported comparison lanes

The executable runner currently exposes seven lanes:

| Method ID | Representation behavior | Target-session behavior |
| --- | --- | --- |
| `csp-lda` | CSP fit at each budget | source + declared labeled calibration |
| `eegnet` | end-to-end EEGNet fit at each budget | source + declared labeled calibration |
| `eeg-conformer` | end-to-end EEG-Conformer fit at each budget | source + declared labeled calibration |
| `frozen-eegnet` | EEGNet trained once on prior history, then frozen | logistic readout refit with declared calibration |
| `frozen-eeg-conformer` | EEG-Conformer trained once on prior history, then frozen | logistic readout refit with declared calibration |
| `sourceweigher-eegnet` | one frozen source-trained EEGNet representation | source-session weights estimated from target calibration embeddings only |
| `sourceweigher-eeg-conformer` | one frozen source-trained EEG-Conformer representation | source-session weights estimated from target calibration embeddings only |

The frozen lanes answer a different scientific question from end-to-end retraining:

> **Did the representation itself remain useful after the neural distribution moved?**

## SourceWeigher target authority

SourceWeigher operates in one common frozen embedding space.

For target session `S4`:

```text
S1 embeddings ----+
S2 embeddings ----+----> RepresentationSourceWeigher ----> source weights
S3 embeddings ----+                         ^
                                            |
                           declared S4 calibration embeddings
```

The final S4 evaluation examples are never passed to SourceWeigher.

At labeled calibration budget `0`, target-dependent SourceWeigher is explicitly unavailable:

```text
status = unavailable_no_target_observations
```

neurOS does **not** reuse final-evaluation EEG as unlabeled target data and call the result zero-shot.

If a future experiment allows unlabeled target observations, that is a separate protocol with its own observation budget.

## Method identity is not transfer strategy

The evidence schema keeps these concepts separate.

For example:

```text
method_id = sourceweigher-eegnet
strategy  = sourceweigher-mean
encoder   = eegnet
```

and:

```text
method_id = sourceweigher-eeg-conformer
strategy  = sourceweigher-mean
encoder   = eeg-conformer
```

This prevents aggregation from collapsing two distinct encoders merely because they use the same transfer algorithm.

## Calibration metrics

### Primary full frontier

For methods genuinely defined at zero target calibration, the preregistered Kumar comparison uses:

```text
0, 1, 2, 5, 10 labeled examples / class
```

and computes normalized area under the balanced-accuracy calibration frontier.

```text
balanced accuracy
^
|                      *
|                 *----
|            *----
|       *----
|  *----
+------------------------------> labels / class
   0   1   2      5        10
```

This rewards useful zero/few-shot behavior rather than only the largest-budget endpoint.

### Positive-budget adaptation frontier

Target-dependent SourceWeigher is not defined at zero calibration. neurOS therefore reports a **separate secondary AUC over strictly positive budgets**.

This allows adaptation strategies to be compared without manufacturing a zero-calibration operating point.

The two metrics are never silently mixed.

## Model capacity is part of the evidence

EEGNet and EEG-Conformer are not capacity-matched architectures.

The evidence rows therefore record:

- resolved model configuration;
- model seed;
- parameter count;
- fit wall-clock;
- inference latency per trial;
- analysis-manifest fingerprint;
- training history;
- representation geometry.

A larger contextual model beating a compact reference model is not automatically evidence that attention is the causal reason for the gain.

## Representation measurements

Task decoders expose stable `encode(...)` representations. The ladder records representation-health measurements such as:

- effective rank;
- feature variance;
- mean representation norm;
- mean pairwise cosine / anisotropy proxy.

These measurements are supplementary to task performance. They help identify whether cross-session failure comes with representation collapse, excessive domain-specific structure, or a change in representation geometry.

Later phases can add frozen-authority:

- session leakage probes;
- cross-session CKA;
- channel/montage perturbation tests;
- mechanistic intervention replication.

Those analyses should reuse the same serialized case authority rather than inventing new test examples.

## Failure preservation

A method crash is not deleted from the study.

For a failed method/case/seed, neurOS emits explicit failed rows for every requested calibration budget:

```text
status = failed
failure_reason = <exception type and message>
```

Structural authority failures abort the study because they invalidate the comparison itself.

Method-level failures remain in the artifact so aggregate plots cannot improve by silently discarding difficult subject/session cases.

## Paired-case promotion gate

For every method, neurOS audits whether each supported calibration budget contains the same frozen subject/session cases.

SourceWeigher's supported set excludes budget `0` by design. Other methods must retain the same cases across the full requested frontier.

A descriptive bundle is promotion-ready only when:

1. there are no method failure rows;
2. there are no unexpected unavailable rows;
3. each method has identical case membership across all budgets it claims to support.

This is a software/evidence gate, not statistical significance.

## Kumar2024 cohort awareness

Kumar2024 contains two original training/adaptation protocols:

- MOABB subjects 1–9: `GR`;
- MOABB subjects 10–18: `PAR`.

The model-ladder result bundle therefore carries `original_protocol` in every case and emits three descriptive views:

```text
pooled frontier
GR frontier
PAR frontier
```

It also emits target-session-index summaries.

This prevents decoder adaptation from being confused with the different original participant training histories.

Promoted statistical inference follows Issue #27 and must account for repeated sessions within participant rather than treating every subject-session case as independent.

## Artifact bundle

A model-ladder study emits:

```text
study_manifest.json
split_authority.json
method_runs.json
results.csv
summary.json
report.md
artifact_hashes.json
```

### `split_authority.json`

This is the central scientific authority.

It contains each case's:

- dataset identity;
- history policy;
- observed chronology;
- source group values;
- held-out group;
- source indices;
- ordered calibration indices by class;
- immutable evaluation indices;
- processed-data SHA-256;
- partition fingerprint;
- calibration-split fingerprint;
- authority fingerprint;
- case metadata such as Kumar GR/PAR identity.

### `method_runs.json`

Preserves resolved method requests and method-level result manifests, including encoder configuration, parameter counts, training time, and failures.

### `results.csv`

This is the flat Evidence Console table. Every row is traceable back to one frozen authority.

### `summary.json`

Contains:

- seed-averaged budget summaries;
- cohort summaries;
- target-session summaries;
- full-frontier AUC;
- positive-budget adaptation AUC;
- paired case-set audits;
- failure/unavailable counts;
- descriptive promotion-gate state.

### `report.md`

A human-readable view rendered from the machine-readable result bundle.

### `artifact_hashes.json`

SHA-256 identities for every preceding artifact.

## Run locally

Install the complete ladder profile:

```bash
python -m pip install -e packages/neuros-core
python -m pip install -e "packages/neuros-models[pytorch]"
python -m pip install -e packages/neuros-sourceweigher
python -m pip install -e "packages/neuros-foundation[evidence,ladder]"
```

Then run a small provenance pilot:

```bash
python scripts/evidence/run_moabb_model_ladder.py \
  --dataset kumar2024 \
  --subjects 1,10 \
  --methods csp-lda,eegnet,eeg-conformer,frozen-eegnet,frozen-eeg-conformer,sourceweigher-eegnet,sourceweigher-eeg-conformer \
  --model-seeds 101 \
  --budgets 0,1,2,5,10 \
  --history-policy prior \
  --split-seed 2026 \
  --output evidence/kumar2024-pilot
```

The first real run should remain a provenance pilot. Do not increase model seeds or scale all participants until the complete artifact has been manually inspected.

## Run from GitHub Actions

Use the manual workflow:

```text
neurOS longitudinal EEG model ladder
```

It is `workflow_dispatch` only. Normal pushes and pull requests never download large public datasets.

The workflow records Python, pip, Git and requested-method identity, executes the real MOABB study, verifies the emitted SHA-256 hashes and authority list, and uploads the evidence bundle as a GitHub artifact.

## Evidence Console layout

The recommended public console is five linked views.

### 1. Chronology

```text
history -------- history -------- TARGET -------- future excluded
                                  |     |
                            calibration eval
```

### 2. Calibration frontier

Show all methods and every operating point, not only the best endpoint.

Methods without a legal zero-calibration point should begin where their declared target-observation requirement begins.

### 3. Historical source trust

For SourceWeigher, render source-session weights, ESS, entropy, residual and sensitivity diagnostics beside the target session.

### 4. Cohort and session drift

Provide pooled, GR, PAR and target-session-index views without changing the underlying result authority.

### 5. Evidence identity

Always keep visible:

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