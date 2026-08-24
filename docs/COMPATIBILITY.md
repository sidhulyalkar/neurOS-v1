# neurOS Compatibility Matrix

The compatibility matrix is a **claim registry**, not a logo wall.

Every integration is assigned a public status and, when evidence exists, the strongest evidence tier neurOS can currently defend. A `supported` label means that the repository contains an executable contract for the named capability. It does **not** automatically mean that a particular physical device, network, study protocol, or clinical workflow is qualified.

The same registry is available from the installed SDK:

```bash
neuros compatibility --json
neuros compatibility lsl --json
```

## Evidence tiers

Evidence is intentionally monotonic:

1. `software-contract` — deterministic unit/contract behavior is enforced;
2. `integration` — the external format/system is exercised end to end at the software boundary;
3. `real-dataset` — a named external dataset/version has been executed under frozen scientific authority;
4. `hardware` — a named device/firmware/transport/software configuration has passed a recorded qualification protocol;
5. `closed-loop` — online action/feedback behavior has been measured under a declared safety and timing protocol;
6. `clinical` — a clinical claim has evidence appropriate to that claim and study context.

Higher tiers are never inferred from lower ones.

## Current matrix

| Integration | Public status | Current qualified surface | Strongest tier |
| --- | --- | --- | --- |
| BrainFlow | supported | source, continuous stream, board/device metadata | software-contract |
| Lab Streaming Layer | supported | source, continuous stream, explicit clock correction | software-contract |
| NWB / PyNWB | supported | recording export interoperability | integration |
| Zarr | supported | recording export interoperability | integration |
| MOABB | experimental | dataset adapter, frozen longitudinal authority, model ladder | software-contract |
| MNE-Python | planned | first-class signal/preprocessing bridge | — |
| Braindecode | planned | faithful model/training/decoder adapters | — |
| Meta NeuralBench | planned | isolated benchmark worker + neurOS evidence extension | — |
| DANDI | planned | dataset discovery + provenance | — |
| SpikeInterface | planned | invasive recording/analyzer bridge | — |
| py_neuromodulation | planned | real-time feature-transform adapter | — |
| OpenBCI | indirect | reachable through BrainFlow device support | — |
| Open Ephys | planned | first-class source/plugin bridge | — |

The table above is descriptive. `neuros.compatibility` is the machine-readable authority and its supported claims are enforced by tests.

## What a green row means

### BrainFlow

The current software-contract evidence verifies fail-closed optional dependency behavior, board-specific EEG rows, destructive ringbuffer drain semantics, actual prepared-session sample rate, malformed data handling, and cleanup. It does not qualify a physical board.

### LSL

The current software-contract evidence verifies deterministic discovery, fail-on-ambiguity semantics, regular-rate/channel geometry, explicit `raw LSL timestamp + time_correction` mapping, synchronized `SignalFrame` output, disabled hidden timestamp post-processing, recovery gating, and failure cleanup. It does not qualify a particular LAN, Wi-Fi topology, clock uncertainty, or producer application.

### NWB and Zarr

The recording CI executes export interoperability from the canonical neurOS session archive. The archive remains the replay authority; community formats are interoperable exports rather than silent replacements for provenance.

### MOABB

The longitudinal benchmark layer has frozen data/calibration/evaluation authority and model-ladder contracts. Expensive public-dataset studies are manual by design. A specific paper/result should only move to `real-dataset` evidence when its dataset version, run artifacts, hashes, and report are retained as evidence.

## Promotion rule

An integration moves upward only when a stronger artifact exists.

For example, the intended live-EEG progression is:

```text
BrainFlow / LSL software contract
        ↓
real SDK / network integration
        ↓
HardwareQualificationManifest
        ↓
recorded sustained qualification run
        ↓
replay + hashes + latency/loss/clock report
        ↓
hardware-qualified named configuration
```

This separation is central to neurOS. It lets the platform become broad without quietly converting software support into biomedical claims.

## Near-term promotion targets

The most valuable next promotions are:

1. build a first-class `SignalFrame ↔ MNE` adapter and round-trip tests;
2. bind Braindecode models through the existing neurOS model/interpretability contracts;
3. run one pinned MOABB longitudinal study and retain a verified evidence bundle;
4. define `HardwareQualificationManifest` and qualify one accessible EEG configuration through BrainFlow + LSL;
5. add DANDI/SpikeInterface as the first invasive/offline interoperability lane;
6. add NeuralBench as an optional worker, extending its task/model substrate with neurOS transfer, deployment, robustness, mechanistic, latency, and provenance evidence.

That sequence grows the matrix by **earned capability**, not by dependency count.
