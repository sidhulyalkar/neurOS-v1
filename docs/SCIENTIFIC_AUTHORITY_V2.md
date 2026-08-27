# Scientific Authority v2

Scientific Authority v2 governs the information boundary around neural model comparison. It complements ORION adaptation and final-assessment authority rather than replacing them.

The core invariant is:

> A promoted comparison must make it mechanically difficult to consume observations outside the declared authority and mechanically obvious when a pretrained model has already seen the evaluation domain.

## What it governs

### Dataset and model lineage

`DatasetLineage` and `ModelLineage` retain canonical upstream identity, version/revision, content or checkpoint SHA-256 where available, declared parent datasets, participant/session/site/device/run identity availability, preprocessing/input assumptions, licensing/citation provenance, and an explicit lineage-completeness state.

Scientific identities are full 64-character SHA-256 digests. Sixteen-character fingerprints are display conveniences only.

Unknown lineage is not disjoint lineage.

### Pretraining overlap

`audit_pretraining_overlap()` emits exactly one of:

- `disjoint_verified`
- `overlap_detected`
- `possible_overlap`
- `unknown_lineage`

The first regression fixture treats BENDR pretraining on TUEG as overlapping TUAB/TUEV when those evaluation datasets declare TUEG ancestry. An overlap does not forbid evaluation. It changes the permitted claim qualification.

A model/evaluation claim marked `clean` requires an explicit `disjoint_verified` audit. Known overlap requires `contaminated_pretraining_overlap`. Possible or unknown lineage requires `unknown_pretraining_lineage`.

## Observation roles

Scientific Authority v2 does not reduce the experiment to train/test. `ObservationRole` distinguishes:

- pretraining;
- supervised training;
- source history;
- labeled target calibration;
- unlabeled target observation;
- qualification/model selection;
- mechanistic discovery;
- untouched final assessment.

`ObservationConsumption` binds a state-changing operation to the exact observation-authority SHA-256 values it consumed.

The default role policy is intentionally strict:

| operation | permitted observations |
| --- | --- |
| pretraining | pretraining |
| preprocessing fit | pretraining, supervised training, source history, labeled calibration, unlabeled target observation |
| model training | pretraining, supervised training, source history, labeled calibration, unlabeled target observation |
| adaptation | source history, labeled calibration, unlabeled target observation |
| model selection | qualification only |
| mechanistic discovery | mechanistic-discovery authority only |
| final assessment | final-assessment authority only |

In particular, final-assessment observations cannot be silently consumed by normalization, source weighting, adaptation, early stopping, model selection, or mechanistic discovery.

## Target-observation budget

`TargetObservationBudget` reports labeled and unlabeled information independently:

```text
labeled_examples
unlabeled_examples
unlabeled_seconds
```

Zero labeled calibration is therefore not automatically zero-shot. A method that observes an unlabeled target distribution has a nonzero target-observation budget even when `labeled_examples == 0`.

## Fitted preprocessing

`PreprocessingFitAuthority` distinguishes:

- `predeclared_fixed`: no data fit is claimed;
- `data_fitted`: exact preprocessing-fit observation consumption is mandatory.

The fitted state receives its own SHA-256 identity. Examples include normalization state, covariance estimates, learned filters, imputation, feature selection, source weights, representation adapters, and artifact models.

## Metrics

Promoted metrics are immutable `MetricSpec` objects rather than bare strings. A metric definition includes:

- ID and version;
- optimization direction;
- averaging and class semantics;
- positive class when relevant;
- probability/calibration requirement;
- estimator implementation and version;
- aggregation unit;
- failure policy;
- uncertainty/inference method;
- primary/secondary status.

The failure policy deliberately has no silent `drop` option.

## Repeated measures

`RepeatedMeasuresAuthority` records the deployment hierarchy and inference unit explicitly. The Kumar2024 study should use participant as the independent unit while preserving participant -> session -> run -> trial hierarchy, GR/PAR strata, and target-session structure.

Ninety participant-session cases are not ninety independent participants.

## Failure preservation

`FailurePreservingResultSet` requires a complete Cartesian product of declared methods and declared cases. Each method/case pair must be represented by an explicit `CaseOutcome`, including:

- `ok`
- `failed`
- `skipped`
- `oom`
- `nonconverged`
- `unavailable`

A method therefore cannot improve its aggregate by disappearing from difficult cases.

## Longitudinal #26/#27 integration

`bind_longitudinal_case_authority()` consumes the existing serialized `LongitudinalCaseAuthority` produced by the longitudinal benchmark. It does not regenerate or alter the split.

For a declared calibration budget it binds:

- `source_train_indices` -> `source_history`;
- the exact prefix of each class calibration order -> `labeled_target_calibration`;
- `evaluation_indices` -> `final_assessment`;
- optional separately declared unlabeled target indices -> `unlabeled_target_observation`.

The adapter fails if source/calibration/final sets overlap. It also refuses to use final-assessment rows as unlabeled target observations. This preserves the frozen Kumar2024 protocol while adding information-budget governance.

## Evidence domains

`ScientificStudyAuthority.report()` separates claims into:

- task utility;
- representation geometry;
- mechanism;
- runtime;
- hardware;
- clinical.

Representation similarity is not task utility. Task utility is not mechanism evidence. Simulator/runtime evidence is not hardware evidence. Hardware evidence is not clinical evidence.

This separation is intentional and should remain visible in Studio/Evidence rather than collapsed into one generic confidence score.

## Example

```bash
python examples/orion/scientific_authority_v2.py
```

The example emits deterministic `orion.scientific_authority.v2` JSON with a full study SHA-256 and display fingerprint.

## Evidence boundary

Scientific Authority v2 improves auditability, leakage resistance, and reproducibility of software scientific comparisons. It does not itself demonstrate:

- decoder or foundation-model superiority;
- physiological validity;
- physical device or clock accuracy;
- online closed-loop efficacy;
- participant benefit;
- clinical validity or safety.
