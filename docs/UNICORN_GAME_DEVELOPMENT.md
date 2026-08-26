# Developing EEG games with the neurOS Unicorn suite

This guide is the shortest path from a game prototype with no headset attached to a game that can be stressed against the neurOS Unicorn Hybrid Black substitution stack and later compared with a physical device.

The simulator reduces **software and integration risk**. It does not prove that a participant will produce a decodable EEG response.

## Keep the layers separate

```text
participant / neural world
        ↓
Unicorn device twin
250 Hz, quantization, schemas, CNT, BAT, VALID, IMU
        ↓
interface
Python API | raw UDP | Recorder19 | LSL | Bandpower70
        ↓
transport faults
loss | duplicate | delay | reorder | stale
        ↓
receiver diagnostics + authority guard
        ↓
decoder
        ↓
gameplay
```

A packet-loss test should not silently change physiology. A weak responder should not silently change UDP ordering. A bad neural stream should revoke BCI authority without freezing the rest of the game.

---

## 1. Pick the interface your application will actually ship

### Raw UDP

Use raw UDP when the application owns real-time signal processing.

Documented compatibility target:

```text
250 source packets/s
68 bytes/packet
17 float32 values
EEG1..8 | ACC X/Y/Z | GYR X/Y/Z | BAT | CNT | VALID
```

The standalone raw-UDP tail is `BAT, CNT, VALID`. The direct API / Recorder ordering uses `CNT, BAT, VALID`.

### Bandpower UDP

Use this only when the application intentionally consumes the public Bandpower-style feature interface.

```text
70 ASCII values/update
250-sample analysis window
10-sample hop
25 Hz steady-state update cadence
```

A stream beginning with no history cannot produce its first complete feature window at `t=0`. The neurOS reference stream therefore emits its first nominal Bandpower update at **1.0 s**, then every **40 ms**.

The payload layout and cadence are compatibility targets. Numerical spectral values are a transparent neurOS reference estimator, not a claim of bit-for-bit proprietary estimator equivalence.

### LSL

Use LSL when acquisition and decoding live outside the game process or when a research stack already uses Lab Streaming Layer.

The synthetic source identifies itself as synthetic in metadata. neurOS does not claim metadata parity with every Unicorn Suite release.

### Direct Python API twin

Use the Python twin to exercise enumeration, configuration, channel enable/disable, acquisition lifecycle and failures.

It is an application-facing conceptual twin, not a replacement for vendor binaries.

---

## 2. Start a synthetic raw-UDP source

```bash
python examples/unicorn_udp_simulator.py \
  --mode raw \
  --host 127.0.0.1 \
  --port 19745 \
  --fault-profile pristine
```

For deterministic CI without sockets:

```bash
python examples/unicorn_udp_simulator.py \
  --mode raw \
  --fault-profile mixed-torture \
  --dry-run-packets 500
```

The source clock targets the documented nominal 250 Hz interface. Python and operating-system scheduling are not presented as calibrated Bluetooth timing measurements.

---

## 3. Do not equate traffic with neural authority

`UnicornRawUdpGuard` keeps four questions separate:

1. **packet status**: can the 68-byte datagram be decoded safely?
2. **sequence status**: is `CNT` first, sequential, gapped, duplicated, reordered or precision-ambiguous?
3. **validation**: is `VALID` asserted?
4. **authority**: may the neural stream currently change gameplay?

`health` remains a compact compatibility summary for simple clients, but advanced telemetry should log the orthogonal fields.

A single packet can therefore be both:

```text
VALID = 0
sequence_status = gap
```

without one condition erasing the other.

Default fail-closed policy:

```text
stale threshold     100 ms
recovery streak     3 healthy packets
VALID required      yes
```

These are neurOS application defaults, not manufacturer specifications.

Authority is revoked on malformed packets, `VALID=0`, counter gaps, duplicates, out-of-order delivery, stale traffic and counter-step ambiguity. Recovery requires a fresh healthy streak.

### CNT precision boundary

The public raw-UDP interface transports `CNT` as float32. IEEE-754 float32 guarantees unit-step integer representation only through `2^24`.

At 250 samples/s, that arithmetic corresponds to roughly 18.64 hours of uninterrupted counting. This **does not** establish physical device reset or wrap behavior. Because the public source does not document those semantics, the reference guard fails closed once exact step inference is no longer justified.

---

## 4. Unity / Godot C# integration

`examples/game_engines/csharp/UnicornRawUdpClient.cs` is dependency-free C# targeting `netstandard2.0`.

Conceptual usage:

```csharp
private UnicornRawUdpClient unicorn;

void Awake()
{
    unicorn = new UnicornRawUdpClient(19745);
    unicorn.Start();
}

void Update()
{
    if (!unicorn.AuthorityAllowed)
    {
        // Controller/game authority remains live.
        return;
    }

    if (unicorn.TryGetLatest(out var sample))
    {
        // Decode or bridge sample.Values[0..7].
    }
}

void OnDestroy()
{
    unicorn.Dispose();
}
```

Raw UDP intentionally carries no synthetic/physical provenance bit. Log source provenance separately:

```text
bci_source_kind  = synthetic
bci_source_label = neuros-unicorn-seed-7
```

or:

```text
bci_source_kind  = user_declared_physical
bci_source_label = lab-headset-a
```

Do not infer provenance from packet bytes.

---

## 5. Torture the transport deliberately

Named deterministic profiles:

| profile | purpose |
|---|---|
| `pristine` | no synthetic transport fault |
| `periodic-loss` | missing-packet handling |
| `duplicate-probe` | duplicate handling |
| `reorder-probe` | adjacent inversion |
| `delay-probe` | delayed delivery |
| `mixed-torture` | loss + duplicate + delay + reorder |

Example:

```bash
python examples/unicorn_udp_simulator.py \
  --mode raw \
  --port 19745 \
  --fault-profile mixed-torture
```

These cadences are synthetic test choices, not measured Unicorn Bluetooth statistics.

A robust game stays playable while BCI authority disappears and recovers. Controller authority, simulation state and save state should remain coherent throughout.

---

## 6. Use neural worlds for physiology, not the packet layer

The low-level device twin should not invent participant physiology.

Use `neuros-arena` or an explicit EEG source for worlds such as:

- strong and weak SSVEP responders;
- no-response / abstention cases;
- endogenous alpha near a stimulus frequency;
- blink, jaw and controller contamination;
- head motion;
- posterior contact degradation;
- display-frame perturbations;
- fatigue / response attenuation;
- recorded-background or source-space models.

Then wrap that EEG in the same Unicorn device/interface contract.

Built-in neural worlds are phenomenological software-test models unless separately reality-anchored. They are not human digital twins.

---

## 7. Capture physical interface diagnostics without persisting EEG

When a physical Unicorn is available:

```bash
python examples/unicorn_raw_udp_trace_capture.py \
  --host 127.0.0.1 \
  --port 19745 \
  --seconds 30 \
  --source-kind user_declared_physical \
  --source-label lab-headset-a \
  --output headset-a-trace.json
```

The v2 capture receipt stores reduced diagnostics such as:

- packet and decode counts;
- arrival rate and inter-arrival summaries;
- counter-gap events;
- **unresolved** missing counters after late-packet reconciliation;
- recovered reordered packets;
- duplicates and out-of-order packets;
- `VALID=0` count;
- counter precision / epoch ambiguity;
- observed battery range.

It records:

```text
contains_raw_eeg      = false
raw_packets_persisted = false
```

Raw datagrams exist transiently in process memory while the summary is computed. This tool does not persist their EEG samples or packet bytes.

`user_declared_physical` is an operator statement, not cryptographic device provenance.

---

## 8. Compare synthetic and physical traces descriptively

```bash
python examples/unicorn_trace_compare.py \
  synthetic-trace.json \
  headset-a-trace.json \
  --output comparison.json
```

The comparison reports deltas for cadence, inter-arrival timing, malformed fraction, validation-zero fraction, reorder fraction and unresolved-missing fraction.

It intentionally returns:

```text
passed = null
```

neurOS does not invent a default “close enough to hardware” threshold. Tolerances should come from measured conditions and a declared validation objective.

If physical behavior disagrees with the simulator, correct the lowest layer that is actually wrong:

| discrepancy | correct layer |
|---|---|
| field ordering | interface contract |
| CNT semantics | device / receiver contract |
| LSL metadata | LSL substitution |
| measured arrival jitter | transport model |
| EEG covariance/amplitude | neural-world model |
| participant cannot select a target | paradigm / decoder |

Do not tune game logic around a simulator defect.

---

## 9. Recommended pre-hardware qualification

At minimum, exercise:

1. pristine strong-responder world;
2. weak responder;
3. no-response / abstention;
4. alpha-frequency collision;
5. blink, jaw and controller artifacts;
6. posterior-channel degradation;
7. `VALID=0` and recovery;
8. packet loss;
9. duplicates;
10. reordering;
11. delay;
12. stale source;
13. mixed transport torture;
14. controller-only continuity while BCI authority is false;
15. deterministic replay across different acquisition chunk sizes.

Passing this matrix means the software is harder to surprise. It does not validate human BCI performance.

---

## Evidence vocabulary

Safe before hardware:

- “tested against the neurOS Unicorn Hybrid Black simulator”
- “hardware-free interface conformance”
- “synthetic transport torture tested”
- “synthetic neural-world robustness”

Requires physical observation:

- “observed on Unicorn Hybrid Black hardware”
- “measured physical arrival cadence”
- “observed physical validation behavior”

Requires human closed-loop data:

- participant SSVEP accuracy;
- accepted precision / abstention;
- false-switch rate;
- decision-time decomposition;
- calibration time;
- comfort and usability;
- real gameplay performance with EEG.

For the adversarial evidence ledger and current known limitations, see `docs/UNICORN_SIMULATOR_RIGOR.md`.
