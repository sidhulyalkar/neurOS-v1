# neurOS Unicorn Hybrid Black Simulation Suite

## Purpose

The Unicorn Hybrid Black Simulation Suite is a hardware-substitution layer for EEG game and BCI development when a physical Unicorn Hybrid Black is not continuously available.

It is designed to answer:

> **If my application were connected through one of the documented Unicorn interfaces, does my acquisition, timing, decoding, failure handling, and game logic behave correctly?**

It is **not** a claim that synthetic EEG is an exact human brain, that Bluetooth is physically reproduced, or that passing the simulator predicts a participant's BCI performance.

The suite deliberately separates three systems:

```text
neural world model
what voltages exist at the electrodes?
        ↓
Unicorn device twin
how does Hybrid Black acquire/expose those voltages?
        ↓
application interface
API / UDP / Recorder / LSL / Bandpower / game
```

That separation is essential. Better EEG world models can be introduced without rewriting the device interface, and device-interface corrections do not silently alter physiology.

---

## Published Hybrid Black acquisition envelope

The canonical device constants used by neurOS are drawn from current public g.tec documentation:

| Property | neurOS contract |
|---|---:|
| EEG channels | 8 |
| EEG sample rate | 250 Hz/channel |
| ADC resolution | 24 bit |
| Sensitivity | ±750 mV |
| Published input impedance | >1 GΩ |
| Accelerometer | 3 axis |
| Gyroscope | 3 axis |
| Published battery duration | >3 h |
| Hybrid electrodes | wet/dry |
| g.Pype acquisition-delay compensation | approximately 40 ms |

Canonical EEG montage used by the standard cap/game profile:

```text
Fz C3 Cz C4 Pz PO7 Oz PO8
```

The >1 GΩ specification is stored as provenance. neurOS does **not** synthesize an imaginary impedance sensor. Electrode/contact degradation belongs to the neural/world layer and influences observable signal quality instead.

Current upstream references:

- g.tec Unicorn Hybrid Black product page: https://www.gtec.at/product/unicorn-hybrid-black-bci-platform/
- g.tec Unicorn Education Kit specification: https://www.gtec.at/product/unicorn-education-kit-teaching-platform/
- g.Pype Hybrid Black source contract: https://gpype.gtec.at/content/7_sdk_reference/backend_sources/hybrid_black.html
- Unicorn Suite: https://www.gtec.at/product/unicorn-suite/

Specifications can evolve. A future neurOS release must not silently update them. Device-profile changes require explicit tests and release notes.

---

# Compatibility surfaces

## 1. `eeg8_anatomical`

Eight EEG channels only:

```text
Fz C3 Cz C4 Pz PO7 Oz PO8
```

Use this for:

- FBCCA/SSVEP game pipelines;
- model/decoder development;
- low-level signal-quality testing;
- Arena synthetic populations;
- software that intentionally ignores motion/auxiliary channels.

This is a normalized neurOS view, not the raw channel naming of every Unicorn API.

---

## 2. Direct API 17-channel acquisition

The documented direct acquisition/API ordering is modeled as:

```text
EEG 1
EEG 2
EEG 3
EEG 4
EEG 5
EEG 6
EEG 7
EEG 8
Accelerometer X
Accelerometer Y
Accelerometer Z
Gyroscope X
Gyroscope Y
Gyroscope Z
Counter
Battery Level
Validation Indicator
```

The simulator provides:

- configuration state;
- enabled/disabled channels;
- enabled-channel index lookup;
- number of acquired channels;
- acquisition start/stop;
- measurement/test-signal mode;
- 8-bit digital output state;
- deterministic injected buffer underflow;
- deterministic injected buffer overflow;
- deterministic connection failures;
- scan-major float32 reads;
- optional binary scan buffers.

`UnicornPythonApiSimulator` is a **conceptual API twin**, not a drop-in binary replacement for g.tec's licensed DLL/Python package.

Synthetic serial identifiers must begin with `SIM-` so test evidence cannot masquerade as a physical device session.

---

## 3. Standalone raw UDP 17-float wire contract

The standalone Unicorn UDP interface documents:

```text
250 packets/s
17 float values / packet
68 bytes / packet
```

A crucial compatibility distinction is encoded explicitly.

Direct API / Recorder auxiliary tail:

```text
CNT → BAT → VALID
```

Standalone raw UDP auxiliary tail:

```text
BAT → CNT → VALID
```

Therefore neurOS maintains a **separate UDP wire order** rather than reusing the API array blindly.

The encoder transforms:

```text
API17
EEG ×8 | ACC ×3 | GYR ×3 | CNT | BAT | VALID
                     ↓
raw UDP17
EEG ×8 | ACC ×3 | GYR ×3 | BAT | CNT | VALID
```

This difference is protected by regression tests.

The public documentation specifies payload size and field order but does not clearly specify endianness. neurOS defaults to little-endian for the Windows/x86 Unicorn Suite environment and labels that choice as a compatibility assumption. Receivers can be fuzzed with alternative byte order intentionally.

---

## 4. Recorder/network 19-field view

The Recorder-style view is modeled as:

```text
EEG 1..8
ACC X/Y/Z
GYR X/Y/Z
CNT
BAT
VALID
DT
STATUS
```

At 250 Hz the default synthetic `DT` is 4 ms.

`STATUS` can be controlled to exercise trigger/state consumers.

This view targets the documented channel/field contract. neurOS does not claim to clone Recorder's GUI or proprietary file-writing internals.

---

## 5. LSL substitution endpoint

Run:

```bash
python examples/unicorn_hybrid_black_simulator.py \
  --schema device17_api \
  --name UnicornMock \
  --no-stdin
```

The endpoint publishes synthetic data through LSL with explicit provenance:

```text
synthetic=true
producer=neurOS
emulated_manufacturer=g.tec medical engineering
emulated_device=Unicorn Hybrid Black
contract=neuros.unicorn_hybrid_black_sim.lsl.v1
```

It intentionally does **not** invent a physical serial number.

The stream can expose:

- EEG8 anatomical view;
- direct API17 view;
- Recorder19 view.

Local deterministic controls include target attention, EEG artifacts, motion telemetry, validation state, status/trigger state, response gain, and source silence.

The public Unicorn LSL application does not expose enough stable documentation for neurOS to claim XML-metadata byte-for-byte parity. The LSL contract is therefore a clearly identified neurOS substitution interface.

---

## 6. Unicorn Bandpower-compatible feature stream

Public Unicorn Bandpower documentation defines seven bands:

```text
delta      1–4 Hz
theta      4–8 Hz
alpha      8–12 Hz
beta low  12–16 Hz
beta mid  16–20 Hz
beta high 20–30 Hz
gamma     30–50 Hz
```

The documented default settings are:

```text
Unicorn sample rate = 250 Hz
buffer size         = 250 samples
buffer overlap      = 240 samples
hop                 = 10 samples
feature update      = 25 Hz
```

The UDP/CSV payload always contains 70 values:

```text
56  = 7 bands × 8 channels
 7  = band averages across enabled channels
 7  = averages across all enabled bipolar derivations
---
70
```

Disabled-channel values are represented as `NaN` in the per-channel portion.

neurOS implements a transparent FFT/PSD reference estimator behind the exact public payload shape, band definitions, disabled-channel behavior, and cadence.

**Important:** the public documentation does not specify enough numerical estimation detail to claim bit-for-bit equality with Unicorn Bandpower. Therefore this surface is classified as `reference_implementation`, not `exact_contract` for feature values.

Reference:

- https://github.com/unicorn-bi/Unicorn-Bandpower-Hybrid-Black

---

# Three evidence classes

Every compatibility surface is classified as one of:

## `exact_contract`

Public documentation provides enough structural information for a deterministic compatibility assertion.

Examples:

- 250 Hz EEG acquisition;
- 8 EEG channels;
- API17 channel layout;
- raw UDP 17-float/68-byte layout;
- raw UDP BAT/CNT ordering;
- Recorder19 layout;
- Bandpower 70-field layout and 25 Hz default cadence.

## `reference_implementation`

The public interface is defined, but proprietary numerical internals are not sufficiently documented.

Example:

- exact Bandpower estimator values.

## `synthetic_assumption`

A useful stress model not attributed to the manufacturer.

Examples:

- accelerometer sensor-noise standard deviation;
- gyroscope sensor-noise standard deviation;
- battery discharge curve;
- distribution of device-delay jitter;
- Bluetooth packet-loss/jitter model;
- synthetic participant/contact-quality distribution.

This classification is included in `neuros.unicorn_hybrid_black_sim.compatibility.v1` reports.

---

# Generate a compatibility receipt

Run:

```bash
python examples/unicorn_sim_conformance.py \
  --seed 41 \
  --output unicorn-simulation-compatibility.json
```

The receipt exercises:

- EEG8 device envelope;
- API17 scans;
- standalone UDP17 wire order;
- Recorder19 fields;
- API lifecycle and underflow recovery;
- Bandpower70 layout/cadence;
- acquisition-delay timing;
- explicit synthetic policy boundaries.

A failing required surface exits non-zero.

The receipt's evidence boundary is intentionally strong:

> Synthetic device/interface compatibility cannot qualify Bluetooth radio behavior, physical electrode contact, real Unicorn firmware, proprietary Bandpower numerical parity, or human EEG performance.

---

# What should be simulated where?

## Device twin

Owns:

- sample rate;
- ADC sensitivity/quantization;
- channel schemas;
- API lifecycle;
- auxiliary telemetry;
- sample counter;
- validation/status;
- acquisition availability delay;
- protocol/wire layouts.

## Arena neural world model

Owns:

- endogenous EEG;
- evoked responses;
- participant variation;
- gaze/attention effects;
- electrode/contact degradation effects on EEG;
- blink, jaw/EMG and movement contamination;
- source-space or recorded-background realism.

## Arena transport world

Owns:

- clock offset/drift/jitter;
- synchronization correction residuals;
- packet loss;
- reordering;
- radio/network silence;
- burst delivery.

## Game/application

Owns:

- calibration policy;
- decoder;
- confidence/quality authority;
- abstention;
- stale-event handling;
- gameplay actions;
- fair source-loss behavior;
- telemetry.

This prevents causality from becoming soup.

---

# Recommended game-development matrix

A BCI game targeting Unicorn Hybrid Black should eventually test all of these:

| Layer | Test |
|---|---|
| EEG | strong/weak/no responder worlds |
| Endogenous activity | alpha collision near code frequencies |
| Contact | posterior channel attenuation/dropout |
| Artifact | blink, jaw, controller/hand, head movement |
| Display | actual emitted stimulus after refresh quantization/drops |
| EEG device | 24-bit / ±750 mV / 250 Hz |
| Motion | still, turn, shake telemetry |
| API | disabled channels and index changes |
| Acquisition | 40 ms availability boundary and underruns |
| Counter | continuity and deliberate gaps |
| Validation | 1 → 0 → recovery |
| Raw UDP | 68-byte scan and BAT/CNT order |
| Recorder | 19-field schema and STATUS/DT |
| Bandpower | 70 values and 25 Hz default cadence |
| LSL | source discovery, stale data and restart |
| Transport | loss/jitter/reordering/silence |
| Decoder | false positives, abstention, switch time |
| Game | no authority during invalid/stale states |
| Replay | deterministic exact-world reproduction |

Passing this matrix still means **hardware-free conformance**, not a human validation result.

---

# The physical substitution experiment

The most important eventual validation of this suite is not another synthetic test. It is a paired run with a physical Unicorn.

For the same consumer application:

```text
neurOS Unicorn twin ─┐
                    ├─→ identical acquisition adapter → decoder → game
physical Unicorn ───┘
```

Then compare:

- discovered channel schemas;
- units;
- scan cadence;
- counter semantics;
- validation behavior;
- motion channels;
- acquisition age/delay;
- LSL/UDP behavior;
- source loss and reconnection;
- receiver behavior.

Any discrepancy becomes a versioned simulator correction rather than an anecdote.

Until such paired hardware observations exist, the suite remains an unusually strong **device/interface emulator**, not a certified replacement for physical Unicorn validation.
