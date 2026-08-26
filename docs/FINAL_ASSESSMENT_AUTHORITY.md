# Final Assessment Authority

Final assessment is the last evidence boundary in an adaptive neural study.

A model may use calibration data to change state. A qualification set may then be used to retain or roll back that change. Once those decisions have been made, the selected state is frozen and evaluated on a separate final-assessment set whose rows and metric scorecard were fixed in advance.

ORION exposes this boundary as executable contracts rather than relying on notebook discipline.

## Why a third partition matters

Two held-out roles are not enough when one of them influences model selection.

If a qualification set decides whether an update is retained, which threshold is used, which hyperparameter wins, or which state becomes active, that set is part of the selection process. Reporting it afterward as an untouched test set would produce optimistic evidence.

The intended lifecycle is:

```text
historical / source data
        |
        v
CALIBRATION
state may change
        |
        v
QUALIFICATION
read-only scoring
may select retain / rollback
        |
        v
SELECTED STATE
state and policy frozen
        |
        v
FINAL ASSESSMENT
exact frozen rows
exact predeclared scorecard
        |
        v
scientific result
```

## ORION contracts

### `FinalAssessmentAuthority`

Freezes:

- dataset identity;
- deployment split unit;
- exact final-assessment sample indices in exact order;
- processed-data SHA-256;
- total sample count;
- source study-authority fingerprint;
- exact metric names to be reported;
- optional protocol fingerprint;
- deterministic seed and immutable metadata.

The authority rejects subsets, supersets, reordering, duplicate indices, out-of-range indices, duplicate metric names, non-finite metadata, opaque metadata objects, and metadata-key collisions after normalization.

The metric names are part of the authority. A method cannot look at final results and then decide to omit a predeclared calibration or reliability metric.

### `SelectedState`

Represents the exact artifact that is frozen before final scoring.

Two selection modes are supported.

#### Frozen

A predeclared or zero-calibration baseline becomes a frozen selected state directly:

```python
selected = SelectedState.frozen(
    selection_id="subject-07/session-03/baseline",
    source_authority_fingerprint=source_fingerprint,
    artifact=frozen_decoder_identity,
)
```

This is important because budget zero is **not** an adaptation with zero observations. No mutation happened, so the evidence record must not invent one.

#### Adapted

A state-changing method becomes selected only through an `AdaptationOutcome`:

```python
selected = SelectedState.from_adaptation_outcome(
    outcome,
    selection_id="subject-07/session-03/adapted",
    source_authority_fingerprint=source_fingerprint,
)
```

The selected artifact is the exact active artifact from the retain/rollback outcome.

### `AdaptiveStudyAuthority`

Binds one positive-budget `AdaptationAuthority` to one `FinalAssessmentAuthority`.

The binding requires:

- one source study-authority fingerprint;
- one dataset identity;
- one processed-data SHA-256;
- one sample count;
- one deployment split unit;
- compatible protocol fingerprints and seeds;
- disjoint calibration, qualification, and final-assessment rows.

The final-assessment authority can remain identical across several calibration budgets while the calibration authority changes. That is the core requirement for a valid calibration-efficiency curve.

### `FinalAssessmentRecord`

Records final metrics only for an already-selected state.

It requires:

- the same source authority as the selected state;
- the complete final-assessment index set in exact order;
- exactly the predeclared metric names;
- finite numeric metric values;
- the exact selected artifact identity.

The record is deterministic and fingerprinted.

## Longitudinal neurOS binding

`neuros-foundation` owns the dataset-specific `ThreeWayLongitudinalCaseAuthority` introduced for longitudinal neural studies.

ORION does not import `neuros-foundation`. The evidence layer derives ORION authorities from that already-frozen source contract:

```text
ThreeWayLongitudinalCaseAuthority
        |
        +--> calibration budget --> AdaptationAuthority
        |
        +--> qualification rows --> AdaptationAuthority.evaluation_indices
        |
        +--> final rows ---------> FinalAssessmentAuthority
        |
        +--> processed data SHA, protocol, seed, source fingerprint
```

The executable cross-package contract is:

```bash
python scripts/evidence/verify_three_way_adaptive_study.py
```

Its CI lane runs on Python 3.10, 3.11, and 3.12 and requires byte-identical evidence replay.

## One final authority, many methods

The final-assessment endpoint is intentionally method-neutral.

A matched study can therefore produce:

```text
CSP / frozen classical baseline -------------------+
frozen EEGNet -------------------------------------+
frozen EEG-Conformer ------------------------------+--> same FinalAssessmentAuthority
conventional target-session fine-tuning -----------+
governed Hebbian predictive adaptation ------------+
future ORION personalization ----------------------+ 
```

Each method may have different internal state-selection logic, but a promoted comparison must share the same source case, final rows, scorecard, and provenance identity.

## Calibration-burden studies

The economically and scientifically useful question is not merely which method has the largest accuracy at one arbitrary calibration budget.

A stronger study asks how much target-specific calibration is needed to reach a predeclared operating point while preserving reliability.

For each method and calibration budget, record at least:

- calibration examples per class;
- calibration minutes when trial duration is known;
- final balanced accuracy / task utility;
- expected calibration error;
- Brier score or negative log likelihood where applicable;
- inference latency and resource cost;
- selected-artifact identity;
- source study-authority fingerprint;
- final-assessment authority fingerprint.

Additional neurOS evidence can include montage dropout, timing jitter, artifact sensitivity, representation geometry, subject/domain leakage, and intervention stability.

## Statistical boundary

A final-assessment set must not influence:

- whether adaptation occurs;
- learning rate, epoch count, or calibration budget;
- early stopping;
- retain/rollback;
- model or architecture selection;
- threshold selection;
- metric selection;
- uncertainty-calibration policy;
- any other state or policy decision later presented as fixed.

If final rows influence one of those decisions, they are no longer final-assessment data and the study must be re-partitioned.

## What this contract proves

A correctly constructed authority and assessment record can prove process properties such as:

- exact sample-role separation;
- exact selected-state identity;
- scorecard preregistration at the software-contract level;
- deterministic provenance;
- matched final evidence across methods and budgets.

It does **not** prove:

- that an adaptive method improves neural decoding;
- that ORION reduces calibration burden;
- that one model transfers better across subjects or sessions;
- online or closed-loop safety;
- hardware reliability;
- clinical benefit.

Those claims require real neural data and their own evidence tier.

## Next evidence rung

The next promoted study should use a real MOABB longitudinal dataset and the same three-way authority to compare, at minimum:

1. a frozen/no-update baseline;
2. a conventional supervised target-session adaptation baseline;
3. governed ngc-learn Hebbian predictive adaptation;
4. an actual ORION representation/personalization method once implemented.

The selected state for every method should be frozen before the common final-assessment set is scored once.
