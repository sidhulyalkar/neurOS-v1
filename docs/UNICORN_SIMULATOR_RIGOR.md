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

### Green/yellow: deterministic synthetic-world semantics

`SyntheticEEGGenerator` is now explicitly versioned and replay-oriented. The software contract establishes that a fixed generator configuration, control sequence and sample-indexed nuisance schedule produce the same sample sequence regardless of ordinary `render()` partitioning.

Implemented invariants include:

- independent seeded streams for phase, colored background and white noise;
- stationary initialization for the colored AR components;
- fixed theoretical colored-noise scaling rather than per-block normalization;
- time-major stochastic draws where stream state is consumed sequentially;
- sample-exact artifact expiration across render boundaries;
- a sample-indexed artifact scheduler that can represent overlapping blink, jaw, controller, motion and source-level masking/offset stressors;
- independent deterministic seeds for stochastic artifact events;
- canonical event rendering order so schedule insertion order cannot change floating-point summation order;
- explicit event IDs, start/end samples, severity, channel support, seed and `synthetic_assumption` evidence class in returned block provenance;
- legacy `inject_artifact()` replacement semantics preserved for existing consumers;
- completed event/noise state pruned rather than accumulated indefinitely.

The static stream descriptor identifies the generator and artifact-scheduler contracts, but does **not** pretend to contain the dynamic experiment schedule. Recorded sample arrays remain replay authority. A scenario that must be regenerated rather than replayed must persist its dynamic control/artifact schedule separately.

### Yellow: timing

Raw source cadence and Bandpower feature cadence are explicit. Bandpower models the initial 250-sample analysis-window warm-up separately from its later 25 Hz feature updates.

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

### Red/yellow: empirical physiology realism

`SyntheticEEGGenerator` remains a controlled nuisance generator, **not a validated human or Unicorn physiological twin**.

Important remaining limitations are deliberate and visible:

1. **Stationary alpha.** Alpha frequency/amplitude are fixed rather than slowly varying across a session.
2. **No explicit line-noise process in the base generator.** Arena can add environmental line noise, but the low-level generator itself does not yet exercise it.
3. **Simplified spatial covariance.** Channel weighting and independent colored backgrounds are hand-authored stress assumptions, not fits to measured Unicorn covariance.
4. **Simplified SSVEP morphology.** Fundamental/harmonic weights and posterior topography are controlled parameters, not a participant population model.
5. **Instantaneous attention control.** `set_attention()` changes the synthetic response immediately at a render boundary. It does not model attentional acquisition latency, fatigue or hysteresis.
6. **Persistent channel gain is not yet sample-scheduled.** Contact degradation can be represented, but its temporal transitions are currently caller-controlled rather than part of the same event timeline.
7. **Source-level `saturation` is only a compatibility stress label.** It adds a large source offset and is not physical amplifier clipping. Actual Unicorn sensitivity clipping/quantization is modeled in the device layer.
8. **Source-level `dropout` is a masking stressor, not a claim about electrode physics or transport packet loss.** Those failure mechanisms belong to separate causal layers.
9. **Artifact magnitudes and spatial patterns are not measured population distributions.** They are useful adversarial assumptions until physical recordings provide an empirical basis.
10. **Dynamic scenario provenance is not yet a first-class archive artifact.** Recorded data can be replayed exactly, but exact regeneration of a complex synthetic world still requires the caller to persist the schedule separately.

These are not reasons to discard the generator. They define the next implementation and measurement gates.

## Causal-layer rule

Keep failure mechanisms in the lowest layer that actually owns them:

```text
participant/source world
  background EEG, alpha, SSVEP, blink, jaw, movement/EMG contamination
        ↓
sensor/contact world
  channel gain/contact quality/electrode-specific nuisance assumptions
        ↓
device world
  sensitivity envelope, quantization, VALID, battery, counter, IMU
        ↓
transport world
  delay, jitter, packet loss, duplicate, reorder, silence/recovery
        ↓
consumer world
  decoding, quality gates, abstention, gameplay authority
```

Do not fix a transport discrepancy by retuning synthetic physiology. Do not call a source offset “hardware saturation” when the device layer owns clipping. Do not tune game logic around a simulator mismatch that should instead correct the simulator.

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

- same generator contract, configuration, controls and scenario produce the same data;
- replay is invariant to reasonable render chunk partitioning;
- overlapping event output is invariant to schedule insertion order;
- stochastic events own independent seeds so unrelated events cannot steal random draws;
- artifact durations and channel support are sample-exact across chunk boundaries;
- completed artifact state is bounded/pruned;
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
- no raw EEG crosses a gameplay boundary unless the application intentionally owns raw decoding;
- combined neural/source, sensor/contact, device and transport faults can be exercised without collapsing them into one ambiguous artifact label.

### R3: physical interface observation

Required before claiming hardware-observed compatibility:

- trace at least one identified Unicorn Hybrid Black configuration;
- record OS, vendor software version, transport path and capture conditions;
- compare cadence/counter/validation behavior against the simulator;
- measure the actual clipping/contact/failure behavior before calibrating those synthetic policies;
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
