# ORION

ORION is the neural-intelligence layer above the neurOS runtime.

neurOS owns acquisition, timing, processing, execution, recording/replay, observability, and device/model integration. ORION owns neural tokenization, learned representations, adaptive decoding, personalization, and the evidence contracts required to change neural model state safely and evaluate it without leakage.

The package is intentionally contract-first. It does **not** claim a finished neural foundation model or proven calibration advantage. Those claims must be earned by executable real-data studies.

```text
hardware -> neurOS SignalFrame -> ORION NeuroTokenizer
         -> NeuroTokenBatch -> NeuralEncoder
         -> RepresentationBatch -> AdaptiveDecoder
```

## Stable responsibility split

```text
neurOS runtime
  acquisition / clocks / backpressure / replay / provenance
        |
        v
ORION representation plane
  tokenization / encoding / adaptive decoding
        |
        v
ORION evidence plane
  adaptation authority / state selection / final assessment
```

The evidence plane is algorithm-agnostic. A conventional optimizer, local Hebbian rule, learned ORION personalization method, or external adaptive decoder can use the same governance contracts.

## State-changing adaptation

`AdaptationAuthority` freezes the exact observations allowed to influence an update and the exact held-out qualification rows allowed to influence retain/rollback.

The governed lifecycle is:

```text
proposal -> approval -> application -> qualification -> retain / rollback
```

Complete mutable model state should be represented by an `ArtifactIdentity`. For optimizers with moments, schedules, or other state, the artifact identity should cover that complete state rather than only visible weights.

See [`docs/ADAPTATION_AUTHORITY.md`](../../docs/ADAPTATION_AUTHORITY.md).

## Final assessment

Qualification data used for state selection are not an untouched final test set.

ORION therefore exposes a separate final-assessment plane:

- `FinalAssessmentAuthority`
- `SelectedState`
- `AdaptiveStudyAuthority`
- `FinalAssessmentRecord`

A zero-calibration baseline becomes `SelectedState.frozen(...)` directly. It is not represented as an empty adaptation. An adapted state becomes selected only through an `AdaptationOutcome`.

The final authority freezes both the exact final rows and the metric names to be reported. This prevents a method from changing either the sample set or scorecard after seeing final results.

See [`docs/FINAL_ASSESSMENT_AUTHORITY.md`](../../docs/FINAL_ASSESSMENT_AUTHORITY.md).

## Longitudinal study binding

The dataset-specific three-way split lives in `neuros-foundation`. ORION does not import that package. The evidence layer derives method-level ORION authorities from the already-frozen source authority:

```text
ThreeWayLongitudinalCaseAuthority
   calibration rows ----> AdaptationAuthority
   qualification rows ---> AdaptationAuthority.evaluation_indices
   final rows -----------> FinalAssessmentAuthority
```

The executable contract is:

```bash
python scripts/evidence/verify_three_way_adaptive_study.py
```

A dedicated Python 3.10/3.11/3.12 CI lane verifies the cross-package binding and requires deterministic byte-identical evidence replay.

## Tokenization research

Current tokenizers include event, count, relative-time, burst, synchrony, vector-quantized motif, and assembly-oriented representations. Fit-requiring tokenizers must be fit explicitly on training data before evaluation.

The longer-term ORION thesis is falsifiable:

> achieve equivalent or superior held-out neural utility with materially less user-specific calibration while preserving robustness, latency, provenance, uncertainty calibration, and stable representation structure.

The package should not claim that result until real longitudinal evidence demonstrates it.

## Scientific claim boundary

Software contracts can establish process integrity. They do not establish neural efficacy.

Current ORION authority surfaces do not by themselves prove:

- reduced calibration burden;
- cross-subject/session/device transfer superiority;
- biological correctness of a learned or local update rule;
- online closed-loop safety;
- hardware qualification;
- clinical benefit.

Those are separate evidence tiers and should remain separate in public documentation and releases.
