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

Interactive stdin remains available if `--no-stdin` is omitted.

## Deterministic control commands

Send one UTF-8 datagram to port 19744:

```text
0            no attended target / resting baseline
1            attend 10 Hz / Sight
2            attend 12 Hz / Guard
b            blink transient
j            jaw contamination
c            controller/hand contamination
m            movement artifact
s            saturation
d            drop Oz
r            restore Oz
x            2 s source silence
silence:2.5  explicit source-silence duration
gain:0.65    set SSVEP response gain
+ / -        step response gain
q            stop source
```

The control channel changes only the simulator state/fault schedule. LSL samples and metadata retain the same external contract used by Mindforge acquisition. This lets the Unity/Python calibration ritual drive Phantom automatically while preserving the eventual physical-source substitution boundary.

## Transport stress

Use `--drop-probability` and `--jitter-ms` for continuous delivery stress. Use `silence:SECONDS` for a coherent source-death/recovery rehearsal.

Synthetic success must never be reported as physical Unicorn or human SSVEP performance.
