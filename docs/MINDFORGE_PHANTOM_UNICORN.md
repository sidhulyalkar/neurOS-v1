# Mindforge Phantom Unicorn

neurOS provides a deterministic Unicorn-like synthetic EEG source for Mindforge qualification. It is an engineering adversary, not a physiological digital twin and not evidence of human BCI performance.

## Pipeline boundary

```text
neurOS synthetic EEG -> LSL UnicornMock -> Mindforge Python FBCCA -> NeuralEvent -> UDP -> Unity
```

Raw synthetic/physical EEG stays outside Unity.

## Start the source

```bash
python examples/mindforge_phantom_unicorn.py --no-stdin
```

Defaults:

- LSL name: `UnicornMock`
- LSL type: `EEG`
- source ID: `mindforge-phantom-unicorn`
- 8 channels / 250 Hz / microvolts
- localhost control: UDP `127.0.0.1:19744`
- generator contract: `neuros.synthetic_eeg.v3`
- artifact scheduler: `neuros.synthetic_eeg.artifact_schedule.v1`

The two synthetic contract identifiers are also emitted in the LSL stream metadata. They identify the software world only; a physical Unicorn stream is not required or expected to advertise them.

Interactive stdin remains available if `--no-stdin` is omitted.

## Manual convenience commands

Send one UTF-8 datagram to port 19744:

```text
0            no attended target / resting baseline
1            attend 10 Hz / Sight
2            attend 12 Hz / Guard
b            immediate blink transient
j            immediate jaw contamination
c            immediate controller/hand contamination
m            immediate movement contamination
s            immediate source-offset stressor

d            persistently mask Oz at the synthetic-source layer
r            restore Oz source gain

x            2 s source silence
silence:2.5  explicit source-silence duration
gain:0.65    set SSVEP response gain
+ / -        step response gain
q            stop source
```

The single-letter artifact commands deliberately preserve the historical **single-slot replacement** behavior for fast manual rehearsals. If `b` is followed immediately by `c`, the controller artifact replaces the blink. Use the explicit scheduler below when overlap is part of the experiment.

## Exact composable artifact schedule

For reproducible multi-artifact scenarios, schedule events on the synthetic sample clock:

```text
artifact:ID:KIND:START_SAMPLE:DURATION_SECONDS:SEVERITY[:CHANNELS][:SEED]
```

Examples:

```text
artifact:blink-a:blink:500:0.30:0.7:*:101
artifact:controller-a:controller:480:0.40:0.9:*:202
artifact:posterior-loss:dropout:525:0.10:1.0:PO7,Oz:303
```

Those three events overlap in sample time. The generator renders them compositionally rather than replacing one with another.

Fields:

- `ID` is the stable event identity. The interactive/UDP reader normalizes commands to lowercase, so use lowercase IDs when external scripts need to cancel them later.
- `KIND` is one of `blink`, `jaw`, `controller`, `motion`, `saturation`, or `dropout`.
- `START_SAMPLE` is an absolute integer sample index on this generator run. A command that arrives after that sample has already been rendered is rejected rather than moved to the present.
- `DURATION_SECONDS` must be positive and finite.
- `SEVERITY` must be non-negative and finite.
- `CHANNELS` is optional. Use `*` for the artifact's default spatial support or a comma-separated scalp list such as `PO7,Oz`. Channel labels from the lowercased control path are resolved back to the canonical montage names.
- `SEED` is optional. When supplied it must be a non-negative integer. When omitted, the generator derives an event-local deterministic seed from generator seed + event identity + event support.

Cancel a scheduled event with:

```text
cancel:posterior-loss
```

A completed event cannot be retroactively cancelled. A rejected malformed/late command is printed and does not kill the source.

## Why sample-indexed scheduling matters

Wall-clock arrival is convenient for a person pressing a key, but it is a weak replay identity. A sample-indexed schedule lets a scripted rehearsal state the intended synthetic world directly:

```text
same generator contract
+ same generator configuration
+ same attention/control history
+ same sample-indexed artifact schedule
= same source sample sequence
```

That equality is a **software determinism claim**. It is not a claim that the generated artifacts have the same distribution as human EEG.

The current Phantom still changes attention through immediate commands (`0/1/2/gain:*`). Exact regeneration of a complete experiment therefore also requires preserving that dynamic attention/control history. A first-class scenario manifest is a logical follow-up; recorded EEG remains the canonical replay authority today.

## Keep causal layers distinct

Several stress mechanisms can look superficially similar but belong to different layers:

| control/model | layer | meaning |
|---|---|---|
| `dropout` artifact / `d` | synthetic source/contact stress | masks selected synthetic EEG channels |
| `saturation` / `s` | synthetic source stress | large source offset used to test downstream artifact handling |
| Unicorn `_quantize_eeg()` clipping | device model | sensitivity-envelope clipping + 24-bit quantization |
| `--drop-probability` | LSL delivery stress | synthetic chunk delivery loss |
| `--jitter-ms` | LSL delivery stress | synthetic extra delivery delay |
| `silence:*` | source/transport availability rehearsal | emits no LSL chunks during the interval |

In particular, the `saturation` artifact is **not** a measured Unicorn amplifier-saturation model. Physical clipping/contact behavior must be measured and qualified separately.

## Transport stress

Use `--drop-probability` and `--jitter-ms` for continuous delivery stress. Use `silence:SECONDS` for a coherent source-death/recovery rehearsal.

This is intentionally separate from EEG artifact scheduling. A controller EMG artifact should not manufacture packet loss, and an LSL dropout should not mutate the underlying synthetic participant waveform.

## Qualification boundary

Synthetic success must never be reported as physical Unicorn or human SSVEP performance.

The Phantom can establish that software reacts correctly to known synthetic worlds and transport failures. It cannot establish:

- physical Bluetooth latency or loss distributions;
- physical channel order/units without observation;
- human SSVEP response magnitude or morphology;
- false-switch/abstention performance in participants;
- comfort or photosensitivity;
- full closed-loop gameplay efficacy.
