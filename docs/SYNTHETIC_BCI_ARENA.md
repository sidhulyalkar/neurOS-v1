# Synthetic BCI Arena

## Purpose

The neurOS Synthetic BCI Arena is a hardware-independent systems test environment for interactive brain-computer interfaces.

It is built around a narrow claim:

> A synthetic arena can provide exact causal ground truth and reproducible operational stress. It cannot prove human physiological performance.

This makes it useful before hardware access, between hardware sessions, in continuous integration, and for creative developers who may never own the target headset.

## System layers

A world manifest composes independently testable causal layers:

```text
scenario / experiment timeline
        ↓
participant response model
        ↓
source nuisances and artifacts
        ↓
stimulus / display presentation
        ↓
sensor / contact state
        ↓
acquisition device
        ↓
transport / clocks / packetization
        ↓
decoder / quality authority
        ↓
application / game / artwork
        ↓
closed-loop behavior
```

The rule is simple: a fault should belong to the lowest causal layer that actually owns it. A source dropout is not packet loss, a source offset is not physical ADC clipping, and packet chunk size must not reach backward and change the simulated neural source.

The current Arena provides dependency-light SSVEP participant simulation plus display, source-artifact, device and transport models. The model boundary remains paradigm-neutral so P300, c-VEP, motor imagery and other generators can be added without changing application-facing scenario semantics.

## Sample-indexed scenario authority

Arena manifest v2 makes artifact identity and timing explicit. A stage artifact may carry:

- an onset relative to its stage;
- duration and severity;
- an optional stable `event_id`;
- optional channel support;
- an optional event-local seed.

For built-in world models, the runner compiles stage-relative events into one absolute source-sample timeline before neural rendering begins. The compiled schedule is recorded in the Arena report.

The important invariants are:

1. reordering a manifest's artifact list does not change the synthetic data;
2. stochastic artifacts own independent deterministic seeds;
3. dropout/channel masks are sample-exact and may cross stage boundaries;
4. changing device packet/chunk size does not change the neural source;
5. built-in world models share the canonical `neuros.synthetic_eeg.artifact_schedule.v1` renderer instead of maintaining separate artifact mathematics;
6. older external world-model plugins that only implement `inject_artifact()` remain usable, but their report is explicitly labelled `legacy_injection` rather than being credited with sample-indexed semantics.

Arena currently renders neural-world blocks at a fixed internal five-sample cadence. That execution cadence is reported as `render_chunk_samples` and is deliberately independent of `DeviceProfile.chunk_samples`. The latter belongs to acquisition/packetization only.

This does not make the participant model physiologically exact. It makes the software timeline unambiguous.

## Verification ladder

### A0 — Determinism

Same manifest + seed must produce the same synthetic signal, timestamps, stimulus trace, fault schedule and report. Equivalent event ordering must not alter source data.

### A1 — Controlled neural nuisance

Sweep response strength, alpha overlap, switching delay, response attenuation, gaze duty cycle, blink/jaw/controller/motion contamination and channel loss. Expected causal labels remain known.

### A2 — Display fidelity

Simulate refresh quantization, response lag, jitter and dropped frames. Report the observed transition frequency and timing error separately from the requested code.

### A3 — Device fidelity

Apply montage selection, sensor noise, mains interference, ADC clipping/quantization, clock drift and acquisition chunk size. Device chunk size must not alter upstream neural samples.

### A4 — Transport adversity

Inject packet loss, delivery jitter, reordering and explicit source-silence windows. Measure delivery fraction, p95 delay and worst arrival gap.

### A5 — Decoder conformance

Feed an external decoder's timestamped accepted/abstained decisions back to Arena. Score accepted precision, false activation during known rest, and switch latency against ground truth.

### A6 — Closed-loop application conformance

Application adapters should expose state transitions such as `BCI_LOST`, recovery, participant stop, calibration ready/failed, or game actions. Arena can then assert safety properties such as “transport loss never becomes an unintended attack.”

### A7 — Real-data anchoring

Use recordings from public datasets or local sessions to compute observable feature signatures, then compare synthetic and real distributions. The result is a feature-distance report, not a universal realism score.

Recommended interoperability:

- **MOABB** for standardized public BCI datasets and evaluation splits;
- **MNE-Python** for forward/source simulation, EOG/ECG/noise injection and signal analysis;
- **MNE-LSL / LSL** for replaying recordings through the same online transport boundary;
- **XDF/LabRecorder** for synchronized multimodal session capture;
- **BrainFlow** for device abstraction and comparison with its hardware-free Synthetic Board.

### A8 — Hardware substitution

A real device replaces the synthetic source while every downstream boundary remains unchanged. This is the first tier that can support device-specific physical claims.

## Portable creative worlds

New Arena manifests are written as:

`neuros.synthetic_bci_arena.manifest.v2`

The v2 schema extends artifact provenance while retaining read compatibility with frozen `neuros.synthetic_bci_arena.manifest.v1` files. Existing v1 examples intentionally remain in the repository as compatibility fixtures.

A manifest describes participant, scenario, world model, device, display and transport state. It should be small enough to paste into a bug report and complete enough to reproduce the declared synthetic world.

Example:

```bash
neuros-arena \
  --manifest examples/arena/dual_target_custom.json \
  --output report.json \
  --npz run.npz
```

When `--write-manifest` is used, the resolved world is written as v2.

## Evidence and replay boundary

The manifest is the requested synthetic-world specification. The Arena report additionally records the resolved artifact schedule and operational metrics. Derived arrays may be exported to NPZ for inspection.

These are synthetic-world regeneration artifacts, not substitutes for observed hardware or human recordings. When real recordings exist, the recorded samples remain evidence authority for what physically happened.

## What “verified synthetic” should mean

A project may say:

> “The application passed Synthetic Arena scenario X across seeds 1–1000, including the specified display, source-artifact, device and transport faults.”

It should not say:

> “The application is 95% accurate on humans.”

unless that claim is independently supported by human recordings under an appropriate protocol.

## Known realism limits

Current deterministic rigor must not be mistaken for physiological realism. Important open modeling gaps include:

- slowly varying alpha frequency/amplitude rather than a stationary oscillator;
- richer participant response variation and fatigue models;
- measured rather than hand-authored spatial covariance for fully synthetic worlds;
- measured artifact amplitude/population distributions;
- sensor/contact dynamics distinct from source dropout;
- population-informed SSVEP morphology;
- physical display timing and headset observations;
- participant-level closed-loop validation.

Those are future evidence/modeling layers. They should not be hidden inside arbitrary constants merely to make a simulator look more lifelike.

## Roadmap

1. Sample-indexed SSVEP closed-loop conformance and portable manifests.
2. Application event adapters and generic decoder protocol.
3. MNE-LSL/XDF replay adapter.
4. MOABB reference-suite adapter for SSVEP/P300/MI datasets.
5. Forward-model participant plugins using MNE head models.
6. P300, c-VEP and motor-imagery synthetic participant plugins.
7. Eye/gaze, IMU and controller multimodal nuisance streams.
8. Monte Carlo population sweeps with coverage maps rather than one “average user.”
9. Mutation/fuzz testing that automatically searches for scenarios causing unsafe application behavior.
10. A small visual Arena dashboard for creators to build and share worlds without writing code.
