# Unicorn simulator rigor ledger

This document is the adversarial review checklist for the neurOS Unicorn Hybrid Black simulation stack. Its purpose is to make the simulator **harder to overclaim**, not easier to market.

A simulator is useful when it tells developers exactly which surprises it can remove before hardware arrives and exactly which surprises remain unknown.

## Evidence classes

| class | meaning | allowed wording |
|---|---|---|
| `exact_contract` | frozen public interface or product fact reproduced directly | “matches the documented interface contract” |
| `reference_implementation` | transparent neurOS implementation where public behavior is underspecified | “neurOS reference implementation” |
| `measured_observation` | captured from identified physical test conditions | “observed under the recorded conditions” |
| `synthetic_assumption` | deliberately invented stress/test policy | “synthetic stress profile” |
| `unknown` | evidence is insufficient | “not yet established” |

Never silently promote one class into another.

## Current layer-by-layer assessment

### Green: interface structure

The strongest part of the simulator is the game-facing interface contract:

- eight EEG channels at the documented nominal 250 Hz acquisition rate;
- 17-value raw UDP payload shape and field order;
- 68-byte float payload size;
- API/Recorder versus standalone raw-UDP tail ordering;
- Recorder field schema;
- Bandpower payload layout and documented update cadence;
- direct-API lifecycle semantics represented by the Python twin.

These surfaces still do **not** establish proprietary binary compatibility with every Unicorn Suite release.

### Green/yellow: deterministic transport and consumer safety

The transport engine can generate deterministic loss, delay, duplicate and adjacent-reorder schedules. These are test policies, not measured Bluetooth statistics.

The receiver guard fails neural gameplay authority closed on:

- malformed packets;
- stale streams;
- `VALID=0`;
- counter gaps;
- duplicates;
- out-of-order delivery;
- counter precision beyond guaranteed float32 unit-step integer representation.

Packet decodability, sequence continuity, validation and gameplay authority are separate diagnostics. A packet can therefore be both `VALID=0` and part of a counter gap without one condition hiding the other.

### Yellow: timing

Raw source cadence and Bandpower feature cadence are explicit. Bandpower now models the initial 250-sample analysis-window warm-up separately from its later 25 Hz feature updates.

Still unknown until measurement:

- physical Bluetooth latency distribution;
- operating-system scheduling jitter for the actual vendor path;
- device/host clock relationship;
- dropped or batched packet behavior under RF interference;
- exact timing semantics across vendor software versions.

The approximately 40 ms compensation used by the reference path remains a `reference_implementation`, not a physical latency specification.

### Yellow: trace calibration

Privacy-safe trace receipts retain packet/timing/counter/validation/battery diagnostics and persist no raw EEG samples or packet bytes.

Loss accounting is reorder-aware:

1. a forward gap opens an unresolved missing interval;
2. a late packet inside that interval reconciles the missing count;
3. an unexplained backward counter marks the counter epoch ambiguous;
4. exact missing-packet claims are suppressed when float32 counter precision is ambiguous.

Synthetic and user-declared physical trace summaries can be compared descriptively. neurOS intentionally ships no default “close enough to hardware” threshold.

### Red/yellow: physiological digital-twin realism

`SyntheticEEGGenerator` is a controlled nuisance generator, **not a validated human or Unicorn physiological twin**.

Known issues that must be resolved before promoting its replay/realism claims:

1. **Chunk-dependent colored-noise normalization.** Colored noise is currently normalized using statistics from each render call. The same seed can therefore produce different scaled data when the same duration is requested with different chunk boundaries. This violates partition-invariant replay.
2. **Dropout duration leak.** A dropout active at the start of a render call can zero the selected channel for the entire returned block even when the requested artifact expires partway through that block.
3. **Single-artifact state.** Artifact injection currently permits one active artifact at a time, while real gameplay can combine eye, jaw, controller and head-motion contamination.
4. **Stationary alpha.** Alpha frequency/amplitude are fixed rather than slowly varying across a session.
5. **No explicit line-noise process in the base generator.** Arena can add environmental line noise, but the low-level generator itself does not yet exercise it.
6. **Simplified spatial covariance.** Channel weighting is hand-authored and useful for stress tests, but not fitted to measured Unicorn covariance.
7. **Simplified SSVEP morphology.** Fundamental/harmonic weights are controlled synthetic parameters, not a population model learned from Unicorn participants.
8. **“Saturation” is a stress label, not yet a calibrated amplifier-clipping model.** It must not be described as measured ADC saturation behavior.

These are not reasons to discard the generator. They define the next implementation gates.

## Counter precision boundary

The public raw-UDP interface transports `CNT` as float32. IEEE-754 float32 represents every integer exactly only through `2^24`.

At 250 samples/s:

```text
2^24 / 250 ≈ 67,109 seconds ≈ 18.64 hours
```

This arithmetic does **not** prove the physical device counts continuously for 18.64 hours. Vendor reset/wrap semantics are not documented by the frozen raw-UDP source. neurOS therefore treats loss inference beyond unit-step float32 exactness as ambiguous and revokes game authority rather than inventing wrap behavior.

## Rigor gates

### R0: deterministic software

Required:

- same seed and same scenario produce the same data;
- replay is invariant to reasonable render chunk partitioning;
- artifact durations are sample-exact across chunk boundaries;
- transport fault schedule is deterministic;
- Python and C# receiver authority decisions agree on shared fixtures.

### R1: public interface conformance

Required:

- exact documented shapes, order and cadence contracts;
- compatibility receipt names frozen upstream sources;
- reference implementations are visibly distinct from exact contracts.

### R2: adversarial game integration

Required:

- controller remains authoritative during BCI failure;
- stale/invalid/gap/duplicate/reorder all fail closed;
- recovery requires a fresh healthy streak;
- game telemetry records synthetic versus user-declared physical source out of band;
- no raw EEG crosses a gameplay boundary unless the application intentionally owns raw decoding.

### R3: physical interface observation

Required before claiming hardware-observed compatibility:

- trace at least one identified Unicorn Hybrid Black configuration;
- record OS, vendor software version, transport path and capture conditions;
- compare cadence/counter/validation behavior against the simulator;
- turn differences into versioned simulator corrections rather than game-specific workarounds.

### R4: human closed-loop qualification

Required before claiming BCI performance:

- participant-level calibration and held-out trials;
- accepted precision and abstention rate;
- false-switch rate;
- decision latency decomposition;
- movement/artifact stress;
- usability and comfort;
- full gameplay testing.

## Development rule

When realism is uncertain, prefer a named parameter or explicit `unknown` over a hidden constant. When a physical discrepancy appears, correct the lowest layer that is actually wrong.

The target is not a simulator that looks convincing. The target is a simulator whose **failures are interpretable**.
