# Neural System Qualification v1

Neural System Qualification (NSQ) v1 is the peer-facing evidence contract for neurOS.

It is intentionally **not** another training framework, model zoo, dataset catalog, or preprocessing stack. Researchers should keep using mature upstream tools such as MNE, MOABB, Braindecode, scikit-learn, BIDS/NWB, BrainFlow, and their own trusted code.

The role of neurOS is narrower:

> Bind an external method, a frozen scientific protocol, an explicit target-information budget, the learned state produced by that exact run, and a failure-preserving result into a qualification envelope that is difficult to overstate accidentally.

## Why this exists

A benchmark row such as `method -> score` hides scientific authority that matters in neural systems:

- whether the participant/session/device was actually unseen;
- whether a pretrained representation already observed the evaluation corpus;
- which data fitted preprocessing or normalization state;
- how many labeled and unlabeled target observations adaptation consumed;
- whether the checkpoint assessed is the checkpoint later reported/deployed;
- whether a probability is calibrated, uncalibrated, or unavailable;
- whether failed/OOM/nonconvergent cases disappeared from the aggregate;
- whether repeated participant-session observations were treated as independent people.

NSQ exists to make those distinctions machine-readable and, where possible, mechanically enforceable.

## First proving ground

The first intended proving ground is the existing Kumar2024 longitudinal motor-imagery study already frozen inside neurOS evidence tooling. Reusing that authority is deliberate: the qualification layer should strengthen an existing preregistered experiment before inventing another benchmark.

The candidate protocol is:

```text
dataset: MOABB Kumar2024
task: left vs right motor imagery
independent unit: participant
hierarchy: participant -> session -> trial
calibration budgets/class: 0, 1, 2, 5, 10
primary metric: balanced accuracy
secondary: accuracy, ROC AUC where defined, Brier score, ECE
final assessment: untouched
```

The protocol is **draft** until its exact dataset-lineage authority and immutable metric scorecard are bound, the real-data execution is complete, and independent reproduction has occurred.

## Four identities

NSQ deliberately separates four objects that are easy to blur in conventional benchmark code.

### 1. `QualificationProtocolSpec`

The protocol identifies the scientific question independently of any model.

Its SHA-256 binds protocol lifecycle, dataset/task identity, full dataset-lineage SHA-256, repeated-measures hierarchy, calibration budget ladder, human-readable metric names, full immutable metric-scorecard SHA-256, robustness axes, untouched final-assessment role, and deterministic metadata.

Metric names are display metadata, not sufficient scientific identity. The metric-scorecard SHA is expected to bind the exact metric definitions, averaging/class semantics, implementation/version, aggregation unit, failure policy, and inference rules defined by Scientific Authority or an equivalent immutable scorecard.

A draft protocol may exist before that scorecard is finalized. A protocol cannot become `frozen` without `metric_scorecard_sha256`. Adding a new model does not change the protocol SHA.

### 2. `ExternalDecoderMethodSpec`

The method spec identifies an algorithm/configuration supplied outside neurOS. It binds method ID, implementation/package identity and version, input-axis convention, probability semantics, unlabeled-target adaptation capability, uncertainty semantics, optional full model-lineage SHA-256, citation/source reference, and deterministic method metadata.

It intentionally **does not contain learned-state hashes**. A method remains the same method when freshly trained at a different calibration budget.

`model_lineage_sha256=None` means lineage is **unknown**, not disjoint. This permits an external pretrained/foundation method to enter a comparison while preventing it from earning a clean pretraining-disjoint claim until its lineage is supplied and independently audited against the evaluation dataset lineage.

The method spec contains no dynamically executed import path. The researcher constructs the implementation in trusted code.

### 3. `QualificationRunContract`

A run contract binds one method to one frozen case and one exact target-information budget. Its SHA-256 includes protocol SHA, method-spec SHA, case-authority SHA, labeled target examples, unlabeled target examples/seconds, preprocessing authorities, calibration authorities, and deterministic run metadata.

A run is `zero_shot` only if **all** labeled and unlabeled target-information budgets are zero.

A nonzero unlabeled-target budget is not merely metadata. It requires the method to declare `target_adaptation_mode="unlabeled"` and the fitted decoder to expose the separate `adapt_unlabeled(X)` authority surface.

### 4. `QualificationModelState`

The learned state is bound *after fitting and any authorized adaptation* to the method and run that produced it.

An external adapter reports `ExternalLearnedState` as `tensor_sha256`, `checkpoint_sha256`, or `opaque_unverified`.

`opaque_unverified` is allowed for scientific comparison. neurOS must not manufacture a durable state identity by serializing an arbitrary Python object with pickle merely to make the evidence look stronger.

A tensor/checkpoint SHA makes the fitted state **content-addressable**, not automatically deployment-qualified. Model Artifact v1, runtime compatibility, real-data evidence, hardware evidence, and higher evidence tiers remain separate gates.

If the method claims `calibrated_probability`, the fitted state must additionally expose the calibration-state SHA-256 for that exact run.

## External participation does not require neurOS training code

Every submitted classifier exposes the minimum task-utility surface:

```python
class ExternalQualificationDecoder(Protocol):
    def fit(self, X, y) -> None: ...
    def predict(self, X): ...
    def learned_state(self) -> ExternalLearnedState: ...
```

Probabilities are an **optional capability**, not a universal assumption:

```python
class ExternalProbabilityDecoder(Protocol):
    def predict_proba(self, X): ...
```

This distinction is necessary for real ecosystem interoperability. A label-only SVM or lab-specific decoder can participate in balanced-accuracy/accuracy comparisons while probability-dependent metrics are explicitly unavailable. neurOS must not synthesize or normalize a probability vector merely to fill a benchmark column.

Unlabeled target adaptation is another separate capability:

```python
class ExternalUnlabeledTargetAdapter(Protocol):
    def adapt_unlabeled(self, X) -> None: ...
```

The trusted-code factory remains deliberately small:

```python
class ExternalQualificationFactory(Protocol):
    @property
    def method_spec(self) -> ExternalDecoderMethodSpec: ...
    def create(self) -> ExternalQualificationDecoder: ...
```

The **factory** is important. The benchmark runner must request a fresh decoder for every calibration budget rather than reusing a model that has already seen a larger target budget.

The intended control boundary is:

```text
frozen neurOS authority
        |
        +-> fresh factory.create()
        |
        +-> authorized X_train, y_train -> external trusted fit()
        |
        +-> optional authorized X_target -> adapt_unlabeled()
        |
        +-> learned_state() ------------> run-bound state identity
        |
        +-> untouched X_final ----------> external predict()
        |                                  optional predict_proba()
        |
        +-> semantic validation --------> failure-preserving result
```

The untouched final-assessment observations are never routed through `fit()` or `adapt_unlabeled()`.

neurOS does not rewrite the submitted optimizer, architecture, augmentation policy, representation learner, or training loop. Its authority comes from controlling which observations cross the external-model boundary and recording what those observations were permitted to mean.

## Prediction and probability semantics

Task labels are mandatory. `validate_prediction_output()` requires one prediction per assessment sample, rejects object-dtype/invalid numeric output, and can reject labels outside the declared task classes without coercing the external method.

Probability semantics are separately declared as:

- `uncalibrated_probability`: a model-native probability estimate with no calibration claim, suitable for methods such as logistic regression;
- `uncalibrated_softmax`: specifically a softmax-derived probability vector with no calibration claim;
- `calibrated_probability`: a probability backed by a run-specific calibration-state SHA-256;
- `unavailable`: the method does not expose qualified probability output.

If probability semantics are not `unavailable`, the decoder must implement `predict_proba(X)`. `validate_probability_output()` requires exact `[sample, class]` shape, floating dtype, finite values in `[0, 1]`, and rows that already sum to one. neurOS does not silently renormalize malformed output because that would change the submitted method.

Probability-dependent metrics should be explicitly unavailable for a method that declares `probability_semantics="unavailable"`, while valid task-utility metrics remain eligible. Missing capability is preserved as evidence rather than silently excluding the method.

## What the executable runner must enforce

The next implementation slice must:

1. restore a full `LongitudinalCaseAuthority` before exposing any arrays;
2. verify the dataset-lineage, metric-scorecard, protocol/case/method, and model-lineage SHA chain;
3. independently audit pretraining overlap rather than treating known lineage as disjoint lineage;
4. call `factory.create()` separately for every calibration budget;
5. expose only source + authorized labeled target observations to `fit()`;
6. expose unlabeled target observations only through `adapt_unlabeled()` and only when both method declaration and run budget authorize them;
7. keep final-assessment rows unavailable to preprocessing, calibration, adaptation, and model selection;
8. bind the learned state after all authorized fit/adaptation to that exact run contract;
9. validate labels and optional probabilities without repairing submitted output;
10. mark probability metrics unavailable, rather than the whole case failed, when probability output is legitimately unavailable;
11. preserve failed, skipped, OOM, unavailable, and nonconvergent cases;
12. emit full SHA-256 identities rather than short fingerprints as scientific joins.

## Baseline philosophy

The first public benchmark should include strong non-neurOS methods before ORION is allowed to make a superiority claim:

- a classical sklearn/MNE baseline;
- at least two maintained Braindecode decoders;
- a competitive foundation representation when licensing and checkpoint lineage permit;
- ORION variants under matched downstream capacity.

A sophisticated neurOS method should never be benchmarked only against weak internal baselines.

## Relationship to existing neurOS authority

NSQ is a participation layer over existing contracts, not a replacement for them:

- `LongitudinalCaseAuthority` freezes source/calibration/final rows;
- ORION Scientific Authority v2 governs lineage, pretraining overlap, information roles, preprocessing fit, immutable metric definitions, repeated measures, and failure preservation;
- Model Artifact v1 governs safe promoted state for supported factories;
- runtime descriptor lineage (#74) will eventually bind stream/transform semantics to artifact input authority;
- Arena remains a pre-human systems falsification layer, not evidence of human performance.

The long-term runner should compose those authorities rather than reproduce weaker local versions of them.

## Evidence boundary

Passing NSQ contracts proves only that a submitted comparison obeys the declared software/scientific authority.

It does not establish model superiority before real-data execution, physiological validity, hardware reliability, online efficacy, participant benefit, clinical validity, or publisher authenticity.

## Peer-acceptance criterion

The contract succeeds when an external researcher can truthfully say:

> I did not need neurOS to train my model. I used neurOS because it let my method enter a frozen neural-system comparison with explicit calibration, leakage, state, and claim authority.

That is the intended differentiator.
