# Synthetic BCI Arena

## Purpose

The neurOS Synthetic BCI Arena is a hardware-independent systems test environment for interactive brain-computer interfaces.

It is built around a narrow claim:

> A synthetic arena can provide exact causal ground truth and reproducible operational stress. It cannot prove human physiological performance.

This makes it useful before hardware access, between hardware sessions, in continuous integration, and for creative developers who may never own the target headset.

## System layers

A world manifest composes independently swappable layers:

```text
participant response model
        ↓
stimulus / display presentation
        ↓
electrode + acquisition device
        ↓
transport / clocks / packetization
        ↓
decoder / quality authority
        ↓
application / game / artwork
        ↓
closed-loop behavior
```

The first Arena release provides dependency-light SSVEP participant simulation plus display, device and transport models. The contract is intentionally paradigm-neutral so P300, c-VEP, motor imagery and other generators can be added without changing application-facing scenario semantics.

## Verification ladder

### A0 — Determinism

Same manifest + seed must produce the same synthetic signal, timestamps, stimulus trace, fault schedule and report.

### A1 — Controlled neural nuisance

Sweep response strength, alpha overlap, switching delay, response attenuation, gaze duty cycle, blink/jaw/controller/motion contamination and electrode loss. The expected causal labels remain known.

### A2 — Display fidelity

Simulate refresh quantization, response lag, jitter and dropped frames. Report the observed transition frequency and timing error separately from the requested code.

### A3 — Device fidelity

Apply montage selection, sensor noise, mains interference, ADC clipping/quantization, clock drift and device chunk size.

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

Arena manifests use schema:

`neuros.synthetic_bci_arena.manifest.v1`

They describe participant, scenario, device, display and transport state. A manifest should be small enough to paste into a bug report and complete enough to reproduce the run.

Example:

```bash
neuros-arena \
  --manifest examples/arena/dual_target_custom.json \
  --output report.json \
  --npz run.npz
```

This enables shared challenge worlds for game jams, classroom assignments, accessibility research and benchmark suites.

## What “verified synthetic” should mean

A project may say:

> “The application passed Synthetic Arena scenario X across seeds 1–1000, including specified display, device and transport faults.”

It should not say:

> “The application is 95% accurate on humans.”

unless that claim is independently supported by human recordings under an appropriate protocol.

## Roadmap

1. SSVEP closed-loop conformance and portable manifests.
2. Application event adapters and generic decoder protocol.
3. MNE-LSL/XDF replay adapter.
4. MOABB reference-suite adapter for SSVEP/P300/MI datasets.
5. Forward-model participant plugins using MNE head models.
6. P300, c-VEP and motor-imagery synthetic participant plugins.
7. Eye/gaze, IMU and controller multimodal nuisance streams.
8. Monte Carlo population sweeps with coverage maps rather than one “average user.”
9. Mutation/fuzz testing that automatically searches for scenarios causing unsafe application behavior.
10. A small visual Arena dashboard for creators to build and share worlds without writing code.
