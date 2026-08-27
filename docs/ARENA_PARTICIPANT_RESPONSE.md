# Arena participant-response contract

## Purpose

Synthetic BCI Arena needs a layer between a scenario's requested task state and a neural world model. Without an explicit participant layer, response delay, target switching, fatigue and gaze assumptions tend to leak into whichever world model happens to be running.

Arena therefore treats participant response as a separate causal layer:

```text
scenario / requested target
        ↓
participant response policy
        ↓
per-source-sample participant state
        ↓
neural world model
        ↓
sensor-space EEG
        ↓
device / transport / decoder / application
```

The first contract is:

`neuros.arena.frequency_target_response.v1`

It is deliberately scoped to **frequency-target visual attention**, primarily SSVEP-style synthetic worlds. It is not a general model of attention, cognition, P300, motor imagery, auditory BCI, workload, intent or human behavior.

## Why this layer exists

Before this contract, Arena computed effective participant gain once per neural render block. That created two undesirable properties:

1. internal render chunk size could change the synthetic participant;
2. every scenario stage boundary restarted response delay, even when two adjacent stages requested the same target.

Those are software artifacts, not participant dynamics.

The v1 compiler resolves the entire participant response on the **source sample clock before neural rendering**. World-model render partitioning can then change implementation batching without changing the upstream participant state.

## Inputs

The v1 compiler consumes the existing `ParticipantProfile` plus explicit frequency-target identity from `ArenaScenario`:

- `StageSpec.target_frequency_hz`;
- `StageSpec.stimulus_id`;
- `StageSpec.attention_gain`;
- `ParticipantProfile.response_delay_s`;
- `ParticipantProfile.switch_time_constant_s`;
- `ParticipantProfile.gaze_duty_cycle`;
- `ParticipantProfile.response_attenuation_per_minute`.

For this model, attentional target identity is:

```text
(target_frequency_hz, stimulus_id)
```

Frequency is a decoder code, not necessarily a unique physical target. Two different objects may reuse the same frequency and must therefore remain distinguishable.

These parameter values define a synthetic world. They are not population estimates unless a separate empirical study justifies a particular distribution.

## Outputs

`compile_participant_state_trace(...)` returns `ParticipantStateTrace` with one value per source sample:

- `attention_gain`: effective synthetic participant drive;
- `requested_attention_gain`: requested drive after declared gaze/fatigue scaling;
- `target_frequency_hz`: active frequency target, with NaN for rest/no-frequency target;
- `target_switch`: attentional target-transition marker;
- `sampling_rate_hz`;
- versioned model identifier.

The report exposes a compact participant-state summary including transition samples, the declared target-identity rule, and effective/requested gain statistics.

## Timing authority

Participant timing is sample-indexed.

For a declared response delay `d` and source sampling rate `fs`, the delay sample count is:

```text
ceil(d × fs)
```

This is intentionally conservative. A delay is a lower bound. For example:

```text
41 ms at 250 Hz = 10.25 samples
```

The synthetic response cannot begin on sample 10 at 40 ms. It first becomes eligible on sample 11 at 44 ms.

Stage durations use the same resolved source sample clock as the Arena runner, so participant transitions align with stage/sample evidence rather than a second floating-duration timeline.

## Target-transition semantics

Stage labels are not participant causes.

Two adjacent stages such as:

```text
sight-a: 10 Hz, stimulus_id="sight-orb"
sight-b: 10 Hz, stimulus_id="sight-orb"
```

preserve the participant response state. Renaming or splitting a stage does not restart response delay.

A true frequency change such as:

```text
10 Hz → 12 Hz
```

resets the v1 response state and begins the declared response delay.

A physical target-identity change also resets response even if the frequency is reused:

```text
10 Hz, stimulus_id="left-orb"
→
10 Hz, stimulus_id="right-orb"
```

This is a real attentional target switch despite identical frequency-valued decoder truth.

By contrast, `stimulus_retrigger=True` restarts the **display presentation** of the same physical object. It does not by itself create an attentional target switch when `(target_frequency_hz, stimulus_id)` is unchanged. Presentation state and participant attention are separate causal layers.

Transitions into or out of rest are also recorded. Initial rest is not reported as a transition because no target change has yet occurred.

## Response dynamics

After the delay, v1 uses a first-order discrete response state that approaches the current requested gain with `switch_time_constant_s`.

`gaze_duty_cycle` is currently implemented as a deterministic attenuation multiplier. The name does **not** mean Arena is simulating literal eye-open/eye-closed occupancy sample by sample.

`response_attenuation_per_minute` is a declared synthetic time-dependent attenuation. It should not be interpreted as measured human fatigue without external evidence.

## WorldInputBlock

The paradigm-neutral world-model boundary distinguishes:

```text
participant_state
```

for scalar compatibility/summary values, and:

```text
participant_streams
```

for sample-aligned participant state.

Built-in W1/W2/W3 frequency-target worlds consume `participant_streams["attention_gain"]` sample by sample.

Older external plugins may continue to consume the scalar compatibility surface. They do not automatically acquire sample-indexed participant semantics.

## World-model ladder

### W0 `legacy_synthetic`

W0 intentionally retains its historical scalar attention adapter. It exists as a regression fixture and does not claim sample-indexed participant-response fidelity.

### W1 `driven_state_space`

W1 consumes the effective attention stream per source sample. Background noise, entrainment and resonator dynamics advance in source-sample order.

### W2 `semi_synthetic_replay`

W2 consumes the same participant stream while preserving a recorded EEG background. Its injected response and harmonic term use per-sample entrainment state rather than a block-final value.

### W3 `leadfield_driven`

W3 consumes the same participant stream before projecting the visual response through the frozen lead-field topography. Random channel/background draws also advance per source sample so render batching cannot remap stochastic draws across channel/time coordinates.

## Render-partition invariance

For W1/W2/W3, the following is a software contract:

> Given the same scenario, participant profile, display, world parameters, seed and source sample clock, changing Arena's internal neural render chunk size must not change the generated source/device EEG.

The test suite compares 1-sample and 37-sample render chunks exactly.

This is not a claim that the synthetic EEG is physiologically realistic. It proves that implementation batching is not part of the synthetic causal world.

## Presentation boundary

Participant state and physical display state are related but distinct.

The presentation layer owns:

- physical stimulus identity;
- explicit retrigger;
- display response lag;
- frame timing/jitter/drop realization;
- coded waveform phase.

The participant layer owns the synthetic response to the attentional target identity.

See `docs/ARENA_PRESENTATION_EPOCHS.md` for the presentation-epoch contract.

## Paradigm boundary

The v1 participant compiler keys off `target_frequency_hz`. A P300, motor-imagery or other non-frequency world therefore receives zero values from this particular participant stream unless it implements its own participant semantics.

That behavior is intentional. Arena should not invent a universal participant model merely to fill a field.

Future participant-response models should be explicitly versioned and paradigm-aware, for example:

- oddball/ERP response-state models;
- motor-imagery engagement/state models;
- gaze/eye-tracking coupled models;
- auditory attention models;
- learned or empirically fitted participant models with held-out validation.

They should plug into the same `WorldInputBlock.participant_streams` boundary without changing device, transport or application layers.

## What this contract proves

It can support statements such as:

- participant state is deterministic for a fixed manifest/profile/seed;
- stage-label-only splits do not reset frequency-target response;
- same-frequency/different-stimulus switches remain real target transitions;
- pure display retriggers do not invent attention switches;
- declared delays never begin before the source sample at-or-after the delay;
- W1/W2/W3 are invariant to internal neural render partitioning;
- participant-stream geometry/non-finite values fail closed.

## What it does not prove

It does **not** prove:

- human SSVEP latency distributions;
- human gaze duty cycles;
- a physiological fatigue law;
- cortical mechanism validity;
- target-hardware performance;
- decoder accuracy on humans;
- closed-loop efficacy;
- suitability for clinical claims.

Those require recordings, physical hardware and/or human studies under appropriate protocols.

## Next scientific steps

The useful next work is empirical rather than adding arbitrary simulator knobs:

1. record physical display timing and Unicorn source timestamps;
2. collect short stationary frequency-target sessions;
3. estimate response onset, amplitude and switching distributions with uncertainty;
4. repeat under controller movement and light combat;
5. compare those measurements against the declared synthetic participant envelope;
6. fit or reject participant parameter distributions using held-out sessions;
7. preserve raw/derived evidence so synthetic calibration never becomes circular validation.

The goal is not to make the simulator harder to distinguish from reality by eye. The goal is to make every assumption explicit enough that real observations can falsify it.
