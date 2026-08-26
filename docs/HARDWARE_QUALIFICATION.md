# Hardware Qualification Evidence

Software tests can prove a driver contract. They cannot prove a physical headset, amplifier, wireless link, firmware build, or clock behaved correctly on the bench.

neurOS therefore treats **hardware qualification as a measured evidence tier**, not as a property inferred from package installation or a passing mock-device test.

## Promotion boundary

A hardware result is eligible for `hardware_qualified = true` only when all three conditions hold:

1. the manifest represents a non-synthetic physical measurement run;
2. the manifest references a neurOS runtime qualification bundle whose externally pinned `bundle_sha256` has actually been verified;
3. every configured reliability, timing, sample-rate, reconnect, and source-to-decision latency threshold passes.

Numerically perfect synthetic CI measurements deliberately fail condition 1. A JSON file containing a plausible root hash but no verified qualification bundle deliberately fails condition 2.

This keeps the software test harness from awarding itself a physical-world badge.

## Evidence object

`HardwareQualificationManifest` records five classes of evidence.

### Device identity

- manufacturer;
- exact device name;
- board/device ID;
- firmware version;
- acquisition library and version;
- operating system;
- physical transport such as USB, serial, BLE, Wi-Fi, or Ethernet.

The combination should identify the acquisition configuration precisely enough that a later run can determine whether it is genuinely comparable.

### Signal geometry

- channel names;
- channel types;
- units;
- nominal device sample rate;
- measured sample rate.

Channel arrays must have identical lengths and channel names must be unique. The measured sample rate is compared against the declared nominal rate rather than silently treated as equivalent.

### Timing evidence

- timestamp source;
- clock domain;
- clock-offset p50/p95;
- clock drift in ppm;
- p95 clock uncertainty.

These fields should come from the actual synchronization/measurement procedure used for the named run. neurOS does not synthesize an uncertainty estimate when one was never measured.

### Reliability evidence

- sustained run duration;
- expected and observed samples;
- neurOS queue accepted/dropped counts;
- reconnect attempts/successes;
- whether reconnect behavior was deliberately exercised.

Sample loss and queue-drop fractions are derived from these counts rather than entered independently, removing one opportunity for inconsistent evidence.

### Decision latency evidence

- source-to-decision p50;
- source-to-decision p95;
- source-to-decision p99;
- latency sample count.

Percentiles must be ordered and non-negative. These are physical/end-to-end timing observations and remain distinct from semantic decoder reproducibility.

## Default threshold profile

The v1 software contract ships conservative **reference defaults**, not universal medical/product requirements:

| Gate | Default |
| --- | ---: |
| Minimum sustained duration | 300 s |
| Maximum sample loss | 0.1% |
| Maximum neurOS queue drop | 0% |
| Maximum sample-rate error | 1% |
| Maximum absolute clock drift | 100 ppm |
| Maximum p95 clock uncertainty | 5 ms |
| Maximum source-to-decision p95 | 100 ms |
| Maximum source-to-decision p99 | 200 ms |
| Reconnect test | optional |

A device/program may require a much tighter, task-specific threshold profile. Changing thresholds must be explicit evidence configuration, not a hidden code tweak performed after observing the result.

## Relationship to `neuros qualify`

First produce the runtime evidence bundle:

```bash
neuros qualify my_hardware_pipeline.yaml \
  --output qualification/run-001
```

Preserve the emitted `bundle_sha256` independently. The hardware manifest must reference that exact root. Hardware evaluation then verifies the bundle using the referenced root before hardware promotion can become possible.

Conceptually:

```text
physical acquisition run
        │
        ├── exact SignalFrames ──> neurOS qualification bundle
        │                              │
        │                              └── externally pinned bundle_sha256
        │
        └── measured device/timing/reliability/latency evidence
                                       │
                                       ▼
                           HardwareQualificationManifest
                                       │
                              fail-closed gate evaluation
                                       │
                   ┌───────────────────┴──────────────────┐
                   │                                      │
             hardware_qualified=false              hardware_qualified=true
             + failed/not-tested gates             + all evidence prerequisites
```

## Synthetic contract tests

CI fixtures use:

```json
{
  "measurement_origin": "synthetic_contract_test",
  "physical_run": false,
  "synthetic_contract_test": true
}
```

Those fixtures are allowed to pass every numeric threshold so the threshold engine is exercised. They are nevertheless structurally unable to produce `hardware_qualified=true`.

That distinction is intentional. Green CI means **the evidence evaluator behaves correctly**, not **a physical BCI device has been qualified**.

## First real reference configuration

The recommended first physical qualification target remains a named OpenBCI Cyton-class setup through the hardened BrainFlow source, optionally paired with an LSL marker stream. A real evidence run should pin at minimum:

- exact OpenBCI board/hardware revision;
- BrainFlow board ID and BrainFlow version;
- firmware;
- USB/serial/radio transport details;
- host OS;
- exact channel map and units;
- observed device sampling rate;
- BrainFlow/neurOS timestamp semantics;
- measured clock drift and uncertainty;
- sample and queue loss;
- reconnect behavior;
- sustained recording duration;
- source-to-decision latency distribution;
- the externally pinned neurOS qualification bundle root.

Until those measurements are collected from the physical system, neurOS should say **hardware contract implemented, physical qualification pending**, not “OpenBCI qualified.”

## Future extensions

The schema is intentionally compatible with later evidence layers:

- repeated qualification across multiple hosts/OS versions;
- firmware compatibility matrices;
- hardware regression histories;
- signed qualification roots;
- fleet/device certificates;
- closed-loop actuation safety evidence;
- device-specific ORION calibration/transfer evidence.

The important invariant is that stronger claims always require stronger measured evidence. No later layer should weaken that rule.
