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
- whether a probability is calibrated or merely a softmax score;
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

The protocol is **draft** until its exact dataset-lineage authority is bound, the real-data execution is complete, and independent reproduction has occurred.

## Four identities

NSQ deliberately separates four objects that are easy to blur in conventional benchmark code.

### 1. `QualificationProtocolSpec`

The protocol identifies the scientific question independently of any model.

Its SHA-256 binds:

- protocol ID and lifecycle (`draft`, `frozen`, `retired`);
- human-readable dataset/task identity;
- full dataset-lineage SHA-256;
- independent repeated-measures unit and hierarchy;
- calibration budget ladder;
- primary/secondary metrics;
- declared robustness axes;
- untouched final-assessment role;
- deterministic metadata.

Adding a new model does not change the protocol SHA.

### 2. `ExternalDecoderMethodSpec`

The method spec identifies an algorithm/configuration supplied outside neurOS.

It binds:

- method ID;
- implementation/package identity and version;
- input-axis convention;
- probability semantics;
- whether the method may consume an explicit unlabeled-target adaptation channel;
- uncertainty semantics;
- citation/source reference;
- deterministic method metadata.

It intentionally **does not contain learned-state hashes**. A method is still the same method when it is freshly trained at a different calibration budget.

It also contains no dynamically executed import path. The researcher constructs the implementation in trusted code.

### 3. `QualificationRunContract`

A run contract binds one method to one frozen case and one exact target-information budget.

Its SHA-256 includes:

- protocol SHA-256;
- method-spec SHA-256;
- case-authority SHA-256;
- labeled target examples consumed;
- unlabeled target examples consumed;
- unlabeled target seconds consumed;
- preprocessing authority SHA-256s;
- calibration authority SHA-256s.

A run is `zero_shot` only if **all** labeled and unlabeled target-information budgets are zero.

A nonzero unlabeled-target budget is not merely metadata. It requires the method to declare `target_adaptation_mode="unlabeled"` and the fitted decoder to expose the separate `adapt_unlabeled(X)` authority surface. This prevents an experiment from claiming controlled unsupervised adaptation when the benchmark cannot identify where target information entered the method.

### 4. `QualificationModelState`

The learned state is bound *after fitting and any authorized adaptation* to the method and run that produced it.

An external adapter reports `ExternalLearnedState` using one of:

- `tensor_sha256`;
- `checkpoint_sha256`;
- `opaque_unverified`.

`opaque_unverified` is allowed for scientific comparison. neurOS must not manufacture a durable state identity by serializing an arbitrary Python object with pickle merely to make the evidence look stronger.

A tensor/checkpoint SHA makes the fitted state **content-addressable**, not automatically deployment-qualified. Model Artifact v1, runtime compatibility, real-data evidence, hardware evidence, and higher evidence tiers remain separate gates.

If the method claims `calibrated_probability`, the fitted state must additionally expose the calibration-state SHA-256 for that exact run.

## External participation does not require neurOS training code

The trusted-code participation surface is structural:

```python
class ExternalQualificationDecoder(Protocol):
    def fit(self, X, y) -> None: ...
    def predict_proba(self, X): ...
    def learned_state(self) -> ExternalLearnedState: ...

class ExternalUnlabeledTargetAdapter(Protocol):
    def adapt_unlabeled(self, X) -> None: ...

class ExternalQualificationFactory(Protocol):
    @property
    def method_spec(self) -> ExternalDecoderMethodSpec: ...
    def create(self) -> ExternalQualificationDecoder: ...
```

The **factory** is important. The benchmark runner must request a fresh decoder for every calibration budget rather than reusing a model that has already seen a larger target budget.

The unlabeled adaptation surface is equally important. `fit(X, y)` and `adapt_unlabeled(X)` are separate channels because they consume scientifically different information. A method may implement only the first. If it declares unlabeled adaptation, the runner validates that the second surface actually exists before exposing any unlabeled target observations.

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
        +-> untouched X_final ----------> external predict_proba()
        |
        +-> semantic validation --------> failure-preserving result
```

The untouched final-assessment observations are never routed through `fit()` or `adapt_unlabeled()`.

neurOS does not rewrite the submitted optimizer, architecture, augmentation policy, representation learner, or training loop. Its authority comes from controlling which observations cross the external-model boundary and recording what those observations were permitted to mean.

## Probability semantics

NSQ does not collapse all classifier output into a generic `confidence` field.

Supported v1 probability semantics are:

- `uncalibrated_softmax`;
- `calibrated_probability`;
- `unavailable`.

`validate_probability_output()` requires an exact `[sample, class]` floating array, finite values, values in `[0, 1]`, and rows that already sum to one. neurOS does not silently renormalize malformed outputs because that would change the submitted method.

A method declaring `unavailable` cannot have arbitrary scores silently treated as probabilities. A method declaring `calibrated_probability` must bind the calibration state for every fitted run.

## What the executable runner must enforce

The next implementation slice must:

1. restore a full `LongitudinalCaseAuthority` before exposing any arrays;
2. verify the run protocol/case/method SHA chain;
3. call `factory.create()` separately for every calibration budget;
4. expose only source + authorized labeled target observations to `fit()`;
5. expose unlabeled target observations only through `adapt_unlabeled()` and only when both the method declaration and run budget authorize them;
6. keep final-assessment rows unavailable to preprocessing, calibration, adaptation, and model selection;
7. bind the learned state after all authorized fit/adaptation to that exact run contract;
8. validate output semantics without repairing the submitted prediction;
9. preserve failed, skipped, OOM, unavailable, and nonconvergent cases;
10. emit full SHA-256 identities rather than short fingerprints as scientific joins.

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
- ORION Scientific Authority v2 governs lineage, information roles, preprocessing fit, metrics, repeated measures, and failure preservation;
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
