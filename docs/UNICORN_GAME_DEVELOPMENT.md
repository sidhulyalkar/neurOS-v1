# Developing EEG games against the neurOS Unicorn Hybrid Black suite

This guide is the shortest path from a game prototype with no headset attached to a game that can be exercised against the neurOS Unicorn Hybrid Black device/interface twin and later compared with a physical device.

The simulator is designed to reduce **software and integration risk**. It cannot establish that a real participant will produce a decodable neural response.

## The contract stack

```text
Arena neural world
participant + display + artifacts + evoked response
        ↓
Unicorn device twin
250 Hz + 24-bit + channel schemas + counter + IMU + validation
        ↓
interface
Python API / raw UDP / Recorder19 / LSL / Bandpower70
        ↓
receiver health + authority guard
        ↓
decoder
        ↓
gameplay
```

Do not collapse these layers. A packet-loss test should not silently change the simulated participant. A weak SSVEP responder should not silently change UDP ordering. A receiver fault should revoke BCI authority without pausing the whole game.

---

## 1. Choose the interface your game will actually use

### Raw UDP

Best when the game owns the real-time signal-processing pipeline.

Contract:

```text
250 source packets/s
68 bytes/packet
17 float32 values
EEG1..8 | ACC X/Y/Z | GYR X/Y/Z | BAT | CNT | VALID
```

Important: standalone raw UDP uses `BAT, CNT, VALID` at the tail. The direct API/Recorder ordering uses `CNT, BAT, VALID`.

### LSL

Best when acquisition/decoding lives in Python and the game consumes a derived event stream, or when a research stack already uses Lab Streaming Layer.

The neurOS LSL endpoint marks itself as synthetic in stream metadata. It does not claim byte-for-byte metadata parity with every Unicorn Suite release.

### Bandpower UDP

Best for prototypes that intentionally use the public Unicorn Bandpower-style feature interface.

Contract:

```text
70 ASCII values/update
25 Hz default update cadence
56 channel-band values
7 channel averages
7 bipolar averages
```

The payload layout/cadence are compatibility targets. Numerical spectral values come from a transparent neurOS reference estimator because the proprietary estimator is not sufficiently documented for bit-for-bit claims.

### Direct Python API twin

Best for application code written around device enumeration, configuration, channel enabling/disabling, acquisition start/stop and failure handling.

This is a conceptual API twin, not a replacement for the licensed vendor binary package.

---

## 2. Start a pristine raw UDP source

```bash
python examples/unicorn_udp_simulator.py \
  --mode raw \
  --host 127.0.0.1 \
  --port 19745 \
  --fault-profile pristine
```

The source targets the documented 250 Hz raw-UDP cadence. Python/OS scheduling is not presented as a calibrated Bluetooth timing measurement.

Before launching the game, you can verify the exact encoder and fault engine without opening a socket:

```bash
python examples/unicorn_udp_simulator.py \
  --mode raw \
  --fault-profile pristine \
  --dry-run-packets 500
```

---

## 3. Put a receiver guard in front of game authority

A receiver should not translate “I received some bytes” into “the brain may control gameplay.”

`UnicornRawUdpGuard` implements a reference fail-closed policy:

- malformed 68-byte payload → revoke authority;
- non-finite values → revoke authority;
- `VALID=0` → stream can still be alive, but revoke authority;
- repeated counter → duplicate → revoke authority;
- counter moves backwards → out of order → revoke authority;
- counter skips forward → gap → revoke authority;
- no decodable packet for the configured stale interval → revoke authority;
- require N healthy sequential validated packets before authority returns.

Default consumer policy:

```text
stale threshold   100 ms
recovery streak   3 packets
validation        required
```

Those values are neurOS application defaults, **not manufacturer specifications**.

A game can keep rendering, moving the player and accepting controller input while BCI authority is false. The neural subsystem simply stops changing gameplay state.

---

## 4. Unity / Godot C# integration

`examples/game_engines/csharp/UnicornRawUdpClient.cs` is dependency-free C# targeting `netstandard2.0`.

Copy it into the engine project, then wrap it with an engine lifecycle component.

Conceptual Unity usage:

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
        // Keep the game alive, but do not apply new BCI-derived authority.
        return;
    }

    if (unicorn.TryGetLatest(out var sample))
    {
        // Feed sample.Values[0..7] into the decoder or signal bridge.
    }
}

void OnDestroy()
{
    unicorn.Dispose();
}
```

Raw UDP deliberately carries no “simulated vs physical” provenance field. The application must log the selected source separately, for example:

```text
bci_source_kind = synthetic
bci_source_label = neuros-unicorn-pristine-seed-7
```

or:

```text
bci_source_kind = user_declared_physical
bci_source_label = hackathon-headset-a
```

Do not infer provenance from raw packet bytes.

---

## 5. Turn on deterministic transport failures

Named profiles:

| profile | purpose |
|---|---|
| `pristine` | no synthetic transport faults |
| `periodic-loss` | deterministic packet-loss probe |
| `duplicate-probe` | repeated-packet handling |
| `reorder-probe` | adjacent packet inversion |
| `delay-probe` | periodic delivery delay |
| `mixed-torture` | loss + duplicate + delay + reorder |

Example:

```bash
python examples/unicorn_udp_simulator.py \
  --mode raw \
  --port 19745 \
  --fault-profile mixed-torture
```

These cadences are deliberately synthetic test patterns. They are **not measured Unicorn Bluetooth statistics**.

A correct game should remain playable through every fault profile. BCI authority may disappear temporarily; controller/game authority should not become corrupted.

---

## 6. Exercise Bandpower consumers

```bash
python examples/unicorn_udp_simulator.py \
  --mode bandpower \
  --port 19746 \
  --fault-profile pristine
```

Or run a deterministic feature-stream check:

```bash
python examples/unicorn_udp_simulator.py \
  --mode bandpower \
  --fault-profile delay-probe \
  --dry-run-packets 100
```

The first update internally requires the configured 250-sample analysis buffer. Later updates use the documented 10-sample hop.

---

## 7. Use Arena for neural/game worlds

The device twin should not be asked to invent participant physiology.

Use `neuros-arena` for worlds such as:

- strong SSVEP responder;
- weak responder;
- endogenous alpha collision near a stimulus frequency;
- blink contamination;
- jaw/EMG contamination;
- head/controller movement;
- posterior contact degradation;
- display frame drops;
- participant fatigue/response attenuation;
- source-space or recorded-background worlds.

Then wrap the resulting sensor-space EEG in the Unicorn device/interface contract.

For Unicorn EEG-only Arena scenarios:

```python
from neuros.arena import unicorn_hybrid_black_eeg_profile

profile = unicorn_hybrid_black_eeg_profile(
    sensor_noise_uv=0.5,
    line_frequency_hz=60.0,
    line_noise_uv=1.0,
    chunk_samples=5,
)
```

Environmental parameters remain explicit and should not be described as published headset behavior.

---

## 8. Capture a real-device interface trace without saving EEG

When a physical Unicorn becomes available, run the game-facing raw UDP path and capture only interface diagnostics:

```bash
python examples/unicorn_raw_udp_trace_capture.py \
  --host 127.0.0.1 \
  --port 19745 \
  --seconds 30 \
  --source-kind user_declared_physical \
  --source-label hackathon-headset-a \
  --output headset-a-trace.json
```

The output contains:

- packet count;
- decoded/malformed count;
- observed arrival cadence;
- mean/p95 inter-arrival time;
- first/last counter;
- counter-gap events;
- inferred missing packets at gap time;
- duplicates;
- out-of-order packets;
- `VALID=0` count;
- observed battery range;
- nominal-contract diagnostics.

It explicitly records:

```text
contains_raw_eeg = false
raw_packets_persisted = false
```

`user_declared_physical` means exactly that. It is an operator statement, not cryptographic proof of device provenance.

---

## 9. Convert discrepancies into simulator corrections

The desired hardware-validation loop is:

```text
same receiver/game
      ↑           ↑
neurOS twin    physical Unicorn
      ↓           ↓
trace summary  trace summary
      └──── compare ────┘
```

If a physical capture disagrees with the simulator, do not tune the game around the simulator. Correct the relevant simulator layer and version the change.

Examples:

- raw field ordering mismatch → interface contract correction;
- counter semantics mismatch → device-twin correction;
- LSL metadata mismatch → LSL substitution correction;
- physical arrival jitter differs → measured transport profile, clearly labeled as observed data;
- EEG amplitude/covariance mismatch → Arena/reality-anchoring work, not UDP code;
- participant cannot select the target → neural paradigm/decoder problem, not packet serialization.

---

# Recommended pre-hardware game qualification

For an EEG action game, a useful minimum matrix is:

1. pristine strong-responder world;
2. weak-responder world;
3. no-response/abstention world;
4. alpha-frequency collision;
5. blink/jaw/controller artifact sequence;
6. posterior-channel degradation;
7. `VALID=0` period and recovery;
8. periodic packet loss;
9. duplicate packets;
10. reordering;
11. delayed packets;
12. source stale interval;
13. mixed transport torture;
14. game remains controllable without BCI authority;
15. replay produces the same synthetic world and decisions.

Passing all fifteen means the **software is much harder to surprise** when real hardware arrives. It does not mean the human BCI has been validated.

---

# Evidence vocabulary

Use these phrases deliberately:

### Safe before physical hardware

- “tested against the neurOS Unicorn Hybrid Black simulator”
- “hardware-free device/interface conformance”
- “synthetic transport torture tested”
- “Arena synthetic-population robustness”

### Requires actual device observation

- “observed on Unicorn Hybrid Black hardware”
- “measured physical arrival cadence”
- “observed physical validation behavior”

### Requires human closed-loop data

- “participant SSVEP accuracy”
- “human false-switch rate”
- “calibration time”
- “comfort/usability”
- “real gameplay performance with EEG”

The simulator exists to make the transition between those evidence levels clean rather than blurry.
