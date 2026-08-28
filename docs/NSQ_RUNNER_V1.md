# NSQ Runner v1

The **Neural System Qualification (NSQ) Runner v1** is the executable referee for the peer-facing NSQ participation contract.

Its job is intentionally narrower than model training:

> Given a frozen neural-data case, a frozen scientific protocol, and an external model factory, control exactly which observations may cross the model boundary and preserve what happened under a cryptographically identified evidence contract.

The runner does **not** rewrite the submitted model, optimizer, architecture, augmentation policy, or training loop.

## Authority chain

A successful budget row binds:

```text
observed upstream dataset lineage
              |
              v
frozen QualificationProtocolSpec
              |
              v
frozen LongitudinalCaseAuthority
              |
              +--> source-history observation SHA
              +--> labeled-target observation SHA
              +--> exact supervised-fit observation SHA
              +--> untouched final-assessment SHA
              |
              v
ExternalDecoderMethodSpec
              |
              v
QualificationRunContract
              |
              v
fresh factory.create()
              |
              +--> fit(authorized X, y)
              |
              v
ExternalLearnedState
              |
              v
QualificationModelState
              |
              +--> predict(untouched X_final)
              +--> optional predict_proba(X_final)
              |
              v
frozen metric scorecard
              |
              v
failure-preserving QualificationBudgetResult
```

The full SHA-256 values are scientific joins. Sixteen-character prefixes are display conveniences only.

## Two different model-state identities

The runner deliberately exposes two different hashes when possible.

### `external_learned_state_sha256`

This is the upstream model's exact tensor/checkpoint identity when the external implementation can honestly provide one.

The direct Braindecode reference hashes the fitted upstream PyTorch module's complete `state_dict()`, including registered buffers. Optimizer state is not part of the inference-state hash and is declared separately in metadata.

### `qualification_model_state_sha256`

This is the NSQ binding of the learned-state record to the exact method and run authority that produced it.

These hashes answer different questions and must not be conflated:

```text
external_learned_state_sha256
    = "what fitted inference state was this?"

qualification_model_state_sha256
    = "under which exact scientific authority did this state participate?"
```

For the MNE/scikit-learn CSP + LDA reference, `external_learned_state_sha256` remains `None`. NSQ does not pickle/joblib an arbitrary estimator merely to manufacture a strong scientific state identity. The method may still participate, and the run binding remains identified, but the learned estimator is explicitly `opaque_unverified`.

## Exact observation identity

Counts are insufficient scientific authority.

A row that says `4 labeled target examples` does not identify *which* four examples were used. Therefore each run contract also binds full SHA-256 identities for:

- source-history indices;
- labeled-target calibration indices;
- complete supervised-fit indices;
- untouched final-assessment indices.

Each observation-set SHA is domain-separated and includes the processed-data SHA. The same integer indices on a different processed array therefore do not silently become the same authority.

## Upstream versus processed data identity

NSQ keeps two provenance layers separate:

- `observed_dataset_lineage_sha256` identifies the upstream dataset/revision actually loaded;
- `processed_data_sha256` identifies the exact processed neural array governed by the longitudinal case authority.

Both must match before external model construction begins.

This prevents a benchmark from validating only the final array while quietly changing its upstream corpus provenance, or validating only the corpus name while changing the actual samples used by the decoder.

## Labeled target calibration

For `LongitudinalCaseAuthority`, Runner v1 executes the frozen calibration ladder exactly.

At every declared budget:

1. restore the same frozen case authority;
2. select the exact balanced labeled-target indices for that budget;
3. construct a **fresh external decoder**;
4. fit on source history plus only those labeled-target rows;
5. bind learned state;
6. predict on the untouched final-assessment rows;
7. validate output semantics;
8. score under the exact scorecard SHA;
9. retain success or failure as a result row.

No warm-start across budgets is permitted by the runner.

## Why unlabeled adaptation is refused in this executor

The generic NSQ participation schema supports a distinct `adapt_unlabeled(X)` capability and separately records unlabeled target information.

However, the first `LongitudinalCaseAuthority` has only:

- source history;
- a labeled calibration pool;
- untouched evaluation rows.

It does **not** freeze a scientifically distinct unlabeled-adaptation pool.

An earlier runner draft attempted to use calibration rows not yet selected by the current labeled budget as unlabeled target observations. That was rejected because the unlabeled set changed as labeled budget increased and disappeared entirely at the maximum budget. The same observation could therefore change scientific role across the calibration frontier.

Runner v1 now fails **before external model creation** if unlabeled target observations are requested under this authority.

A future adaptive executor must first introduce or reuse an authority that freezes a distinct unlabeled target role with no overlap with labeled calibration, state-selection/qualification, or final assessment. We will not infer that role from "leftover" rows.

## Output semantics

Every external decoder must provide task-label `predict()`.

Probability output is optional. A probability-capable method additionally must provide:

- `predict_proba()`;
- fitted probability-column class order.

The class order must exactly equal the canonical source-derived class vocabulary before ROC AUC, Brier score, or ECE are computed.

The runner never renormalizes malformed probabilities or guesses their class order.

### Label-only methods

A label-only method remains a valid participant:

```text
balanced accuracy       available
accuracy                available
ROC AUC                 unavailable_probability_output
Brier score             unavailable_probability_output
ECE                     unavailable_probability_output
```

Unavailable metrics remain explicit evidence rather than becoming dropped cells or synthetic probabilities.

## Classification scorecard v1

The dependency-light `ClassificationScorecardV1` binds these semantics into a full SHA-256:

- **balanced accuracy:** macro mean recall over the source-derived class vocabulary;
- **accuracy:** exact task-label fraction;
- **ROC AUC:** binary rank AUC only in v1, with the second canonical source label as positive class;
- **Brier score:** mean per-sample sum of squared multiclass probability error;
- **ECE:** top-label expected calibration error with 10 equal-width bins by default.

The protocol cannot execute unless its `metric_scorecard_sha256` equals the scorecard supplied to the runner.

## Failure preservation

External model/runtime failures do not disappear from the frontier.

Runner v1 preserves statuses including:

- `success`;
- `unavailable`;
- `oom`;
- `failed`;
- `skipped`;
- `nonconverged`.

If fitting succeeded and a later output-semantic check failed, the row may retain the already-produced learned-state evidence while remaining a failed result. This prevents a result bundle from pretending the model was never fitted simply because its output contract was invalid.

Scientific-authority mismatches such as wrong dataset lineage, tampered processed data, wrong metric authority, an unfrozen protocol, or a forbidden unlabeled-target request fail before external code sees data.

## First external proving methods

### MNE CSP + scikit-learn LDA

`MNECSPLDAFactory` uses maintained upstream components:

- `mne.decoding.CSP`;
- `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`;
- sklearn pipeline composition.

The runner does not reimplement CSP or LDA.

### Direct upstream Braindecode

`UpstreamBraindecodeFactory` constructs:

- `braindecode.models.<model>`;
- upstream `braindecode.EEGClassifier`;
- PyTorch cross-entropy;
- PyTorch AdamW.

It deliberately does not route training through the existing neurOS `BraindecodeDecoder` wrapper. This makes the interoperability claim stronger: upstream Braindecode is the contestant; neurOS supplies scientific authority and scoring.

## What this proves

Passing Runner v1 contracts can establish that:

- the declared upstream/processed data authority matched;
- exact source/calibration/final observation sets were controlled;
- a fresh external model was used at every budget;
- final-assessment rows did not enter fitting;
- output semantics were checked rather than repaired;
- metric semantics were frozen;
- failures remained visible;
- real MNE/sklearn and Braindecode implementations can participate without adopting neurOS training code.

It does **not** establish that any model is better, that EEG features are physiologically meaningful, that hardware is reliable, that an online BCI works, or that a system is clinically useful.

## Next scientific milestone

After this runner is production-qualified, the next milestone is not another abstraction. It is a real frozen longitudinal study:

1. bind the Kumar2024 dataset lineage and processed-data authorities;
2. freeze the v1 metric scorecard;
3. execute the same participant/session cases for MNE CSP + LDA and upstream Braindecode;
4. add additional strong upstream decoders;
5. audit pretrained foundation-model lineage before admitting foundation representations;
6. only then place ORION variants into the same qualification frontier.

The flagship comparison should be **task performance versus per-user calibration cost**, with participant as the independent inferential unit and failures retained.
