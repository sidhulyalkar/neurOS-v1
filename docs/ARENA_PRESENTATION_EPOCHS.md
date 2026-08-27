# Arena physical presentation epochs

Synthetic BCI Arena separates **scenario structure** from **physical display events**.

A stage label, task annotation or scoring boundary is not automatically a new visual stimulus. Treating every stage boundary as a display restart makes authoring choices causal: splitting one continuous target into two stages can restart display response lag, flicker phase, frame-jitter randomness and dropped-frame history even though nothing physical changed.

The presentation contract removes that ambiguity.

## Contracts

Presentation compiler:

`neuros.arena.presentation_epochs.v1`

Synthetic display trace:

`neuros.arena.display_trace.v2`

Each `StageSpec` may declare:

- `stimulus_id`: optional stable identity of the physical coded object;
- `stimulus_retrigger`: explicit request to start a new physical presentation epoch.

Adjacent stages share one presentation epoch when all of the following hold:

1. target frequency is unchanged;
2. `stimulus_id` is unchanged;
3. the later stage does not set `stimulus_retrigger=True`.

A frequency change or `stimulus_id` change starts a new epoch automatically.

## Why frequency alone is insufficient

Frequency is a code, not an object identity.

Two spatial targets can both use 10 Hz. If the participant moves attention from a left 10 Hz orb to a right 10 Hz orb, Arena must represent a real target switch even though the decoder code is numerically unchanged.

For the current frequency-target participant model, attentional identity is therefore:

```text
(target_frequency_hz, stimulus_id)
```

This affects participant response delay and switching state.

Presentation retrigger is intentionally separate. Restarting the same physical object's flicker does not automatically mean that the participant stopped attending that object.

## Command time and modeled emission time

The display trace keeps two clocks separate:

- `command_frame_times_s`: when the application/display scheduler requests each frame state;
- `frame_times_s`: when that frame is modeled as physically emitted after the declared display response lag.

The v2 display contract evaluates coded luminance on the command clock and then delays emission:

```text
commanded luminance at t
        ↓
response_lag_ms
        ↓
modeled emission at t + lag
```

Equivalently:

```text
emitted(t + lag) = commanded(t)
```

This is important for phase-sensitive stimulation. A constant 6 ms response lag must shift the complete emitted waveform by 6 ms. It must not delay only the first visible frame and then evaluate subsequent luminance against an undelayed global oscillator.

`frame_jitter_ms` currently represents variability in the scheduler/frame cadence before the constant response-lag transform. Pixel-specific variable response latency is deliberately not invented here. That belongs to measured display calibration.

This remains a synthetic timing model. A physical monitor can falsify it.

## Causal timeline

```text
ArenaScenario stages
        ↓
physical stimulus identity + explicit retrigger
        ↓
PresentationPlan
        ↓
PresentationEpoch 0, 1, ...
        ↓
command-frame clock / coded phase / frame jitter / drops
        ↓
constant modeled command→emission response lag
        ↓
emitted luminance sampled on the EEG source clock
        ↓
neural world model
```

Frame timing and frame RNG belong to the presentation epoch, not to the authoring stage.

## Stage segmentation invariant

Given the same resolved source sample timeline:

> Splitting one stage into adjacent stages with the same frequency, same `stimulus_id`, and no explicit retrigger must not change emitted luminance or downstream EEG.

The Arena regression suite binds this with non-zero response lag, frame jitter and dropped-frame probability, so the invariant is not merely a zero-noise special case.

A separate regression binds the response-lag contract itself: changing only `response_lag_ms` must leave command-frame times and commanded luminance unchanged while shifting every modeled emission timestamp by the declared lag.

## Explicit retrigger

Use:

```python
StageSpec(
    "sight-second-block",
    1.0,
    target_frequency_hz=10.0,
    stimulus_id="sight-orb",
    stimulus_retrigger=True,
)
```

when the application physically restarts that presentation.

A retrigger creates a new display epoch and therefore a new response-lag/frame/drop/phase realization. It does **not** itself create an attentional target switch if `(frequency, stimulus_id)` is unchanged.

## Same-frequency target switch

Use distinct stimulus identities for distinct physical objects even if their code is identical:

```python
StageSpec("left", 1.0, 10.0, stimulus_id="left-orb")
StageSpec("right", 1.0, 10.0, stimulus_id="right-orb")
```

This creates:

- two presentation epochs;
- one participant target transition at the boundary;
- a new participant response-delay interval;
- unchanged frequency-valued decoder ground truth.

That distinction is important for systems studying spatial selection or code reuse.

## Report evidence

Arena reports expose:

`metrics.presentation.model`

`metrics.presentation.epoch_count`

`metrics.presentation.stage_epoch_index`

and one record per physical epoch containing:

- presentation model and display-trace model;
- source-sample start/end;
- resolved timing;
- command start and first modeled emission;
- modeled response lag;
- target frequency;
- `stimulus_id`;
- member stage indices/labels;
- observed transition frequency;
- frame-drop fraction;
- interval jitter.

The historical per-stage `metrics.display` list remains available and names the presentation epoch used by each stage.

`ArenaRun.stimulus_traces` remains stage-addressable for compatibility. Adjacent stages in one physical presentation intentionally reference the same complete trace. The report's presentation section is the canonical source for physical-epoch interpretation.

## What is guaranteed

Within the declared synthetic model, Arena can guarantee:

- stage-label changes do not restart a continuing presentation;
- frame RNG belongs to physical epochs rather than stage count;
- explicit retrigger is represented separately from task segmentation;
- same-frequency/different-object switches remain identifiable;
- presentation timing is resolved on the same source sample clock used by participant and neural-world layers;
- constant declared response lag is represented as a persistent command→emission delay rather than an onset-only blank.

## What is not claimed

This contract does not establish:

- physical monitor frame timing;
- actual pixel response latency or its variability;
- photodiode-measured luminance;
- human SSVEP phase locking;
- human response latency after a flicker restart;
- safe or comfortable stimulation parameters.

Those require measured display and participant evidence.

## Physical qualification

The correct next evidence ladder remains:

1. render the exact competition stimulus;
2. measure emitted luminance with a photodiode;
3. compare requested vs measured frame/transition timing;
4. repeat under idle and full application load;
5. repeat across ordinary combat presentation events;
6. test physical headset acquisition and synchronized stimulus markers;
7. only then evaluate participant-level neural response and decoder performance.

The presentation epoch model is a software authority contract. It is designed so those measurements can falsify or calibrate the synthetic layer rather than being replaced by it.
