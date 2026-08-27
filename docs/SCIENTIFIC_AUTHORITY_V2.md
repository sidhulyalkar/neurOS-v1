# Scientific Authority v2

Scientific Authority v2 governs the information boundary around neural model comparison. It complements ORION adaptation and final-assessment authority rather than replacing them.

The core invariant is:

> A promoted comparison must make it mechanically difficult to consume observations outside the declared authority and mechanically obvious when a pretrained model has already seen the evaluation domain.

The implementation is split into focused modules under `orion.scientific` for lineage, observation/preprocessing authority, evaluation/inference authority, study claims, and longitudinal integration. `orion.scientific_authority` remains a compatibility facade.

## Scientific identity

Promoted scientific identities are full 64-character SHA-256 digests. Sixteen-character fingerprints are display conveniences only.

A short fingerprint may appear in a table or UI, but it is not sufficient durable identity for a dataset lineage, model lineage, observation authority, fitted preprocessing state, result set, or study.

## Dataset and model lineage

`DatasetLineage` retains:

- canonical dataset ID and upstream source;
- version/revision;
- content SHA-256 where available;
- parent dataset/domain identities;
- participant/session/site/device/run identity availability;
- preprocessing history;
- sampling/channel assumptions;
- license/citation provenance;
- explicit lineage completeness.

`ModelLineage` retains the equivalent model/checkpoint identity plus:

- checkpoint SHA-256 where available;
- declared pretraining datasets/domains;
- declared pretraining participant/session/site/device/run identity sets when available;
- pretraining preprocessing history;
- model input assumptions;
- explicit pretraining-lineage completeness.

Unknown lineage is not disjoint lineage.

## Pretraining overlap

`audit_pretraining_overlap()` emits exactly one of:

- `disjoint_verified`
- `overlap_detected`
- `possible_overlap`
- `unknown_lineage`

The audit walks transitive dataset ancestry. A checkpoint pretrained on TUEG therefore overlaps an evaluation dataset whose parent or grandparent domain is TUEG.

If a declared parent dataset cannot be resolved, a complete-looking leaf dataset does not receive `disjoint_verified`; the result is `possible_overlap`.

The audit also checks declared entity identities. If pretraining and evaluation expose an overlapping participant/session/site/device/run identity at the same level, the overlap is machine-visible even when the top-level dataset names differ.

The regression suite treats BENDR pretraining on TUEG as overlapping TUAB/TUEV when those evaluation datasets declare TUEG ancestry. An overlap does not forbid evaluation. It changes the permitted claim qualification.

A model/evaluation claim marked `clean` requires an explicit `disjoint_verified` audit whose full model/dataset lineage SHA-256 values still match the study. A stale or forged audit is rejected. Known overlap requires `contaminated_pretraining_overlap`. Possible or unknown lineage requires `unknown_pretraining_lineage`.

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

Final-assessment observations therefore cannot be silently consumed by normalization, source weighting, adaptation, early stopping, model selection, or mechanistic discovery.

The top-level `ScientificStudyAuthority` also verifies that every observation references a dataset lineage actually declared by the study, and that every data-fitted preprocessing transform consumes only observation authorities in the declared study universe.

## Target-observation budget and zero-shot semantics

`TargetObservationBudget` reports labeled and unlabeled information independently:

```text
labeled_examples
labeled_examples_per_class
unlabeled_examples
unlabeled_seconds
```

Zero labeled calibration is not automatically zero-shot. A method that observes target-session statistics, target covariance, embeddings, moments, or other unlabeled target information has a nonzero target-observation budget even when `labeled_examples == 0`.

A claim explicitly marked `zero_shot_claim=True` must point to a declared target budget with zero labeled examples, zero unlabeled examples, and zero unlabeled seconds. Otherwise study construction fails.

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
- positive class where relevant;
- probability/calibration requirement;
- estimator implementation and version;
- aggregation unit;
- failure policy;
- uncertainty/inference method;
- primary/secondary status.

The failure policy deliberately has no silent `drop` option. Probability-dependent metrics must make their probability/calibration semantics explicit.

A promoted study must declare exactly one primary metric.

## Repeated measures

`RepeatedMeasuresAuthority` records the deployment hierarchy and inference unit explicitly. The Kumar2024 study should use participant as the independent unit while preserving participant -> session -> run -> trial hierarchy, GR/PAR strata, and target-session structure.

The declared clustering authority must include the independent experimental unit. Ninety participant-session cases are not ninety independent participants.

## Failure preservation and result evidence

`FailurePreservingResultSet` requires a complete Cartesian product of declared methods and declared cases. Each method/case pair must be represented by an explicit `CaseOutcome`, including:

- `ok`
- `failed`
- `skipped`
- `oom`
- `nonconverged`
- `unavailable`

Successful rows must report the declared metric scorecard. Unknown metric names are rejected. A method therefore cannot improve its aggregate by disappearing from difficult cases or changing its scorecard case by case.

Task-utility claims must cite at least one embedded failure-preserving **result-set SHA-256**. A metric-definition SHA says how the study intended to score; it is not evidence that the model actually achieved a result.

## Longitudinal #26/#27 integration

`bind_longitudinal_case_authority()` consumes the existing serialized `LongitudinalCaseAuthority` produced by the longitudinal benchmark. It does not regenerate or alter the split.

The bridge derives a full SHA-256 over the exact serialized frozen case authority after removing only derived identity fields. If a future longitudinal authority supplies its own full `authority_sha256`, the bridge verifies it rather than trusting it. The legacy 16-character `authority_fingerprint` is preserved only as display metadata.

For a declared calibration budget it binds:

- `source_train_indices` -> `source_history`;
- the exact prefix of each class calibration order -> `labeled_target_calibration`;
- `evaluation_indices` -> `final_assessment`;
- optional separately declared unlabeled target indices -> `unlabeled_target_observation`.

Indices must be actual non-negative integers. Floats and booleans are not silently coerced. When `n_samples` is present, every index is range-checked.

Source, labeled calibration, unlabeled target observation, and final assessment must remain disjoint. In particular, the adapter refuses to relabel source-history or final-assessment rows as unlabeled target observations.

This preserves the frozen Kumar2024 protocol while adding full identity and information-budget governance.

## Evidence domains and claim scope

`ScientificStudyAuthority.report()` separates claims into:

- task utility;
- representation geometry;
- mechanism;
- runtime;
- hardware;
- clinical.

Representation similarity is not task utility. Task utility is not mechanism evidence. Simulator/runtime evidence is not hardware evidence. Hardware evidence is not clinical evidence.

This separation is intentional and should remain visible in Studio/Evidence rather than collapsed into one generic confidence score.

## Structural immutability

Scientific authorities detach caller-owned sequences and recursively freeze provenance mappings where they cross the authority boundary. The top-level study stores target budgets as a read-only mapping. A study identity must not change because a caller still holds a mutable alias.

## Example

```bash
python examples/orion/scientific_authority_v2.py
```

The example emits deterministic `orion.scientific_authority.v2` JSON with:

- full study SHA-256;
- display fingerprint;
- explicit zero-target information budget;
- failure-preserving result-set SHA-256;
- overlap status;
- separated evidence domains and claim scope.

The example's numeric score is a deterministic report-shape fixture, not real-data evidence.

## Evidence boundary

Scientific Authority v2 improves auditability, leakage resistance, and reproducibility of software scientific comparisons. It does not itself demonstrate:

- decoder or foundation-model superiority;
- physiological validity;
- physical device or clock accuracy;
- online closed-loop efficacy;
- participant benefit;
- clinical validity or safety.
