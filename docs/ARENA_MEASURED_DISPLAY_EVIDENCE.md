# Arena measured display evidence

Synthetic BCI Arena can model a display, but software frame timing is not physical luminance evidence.

This bridge is the boundary where an Arena presentation can be compared with an **observed luminance-like trace**, such as a photodiode voltage recorded from the actual competition display.

The guiding rule is:

> The simulator predicts. The observation decides what physically happened.

## Schemas and model provenance

Observation:

`neuros.arena.display_observation.v1`

Qualification report:

`neuros.arena.display_qualification.v1`

Transition detector:

`neuros.arena.schmitt_transition_detector.v1`

The current synthetic display trace is:

`neuros.arena.display_trace.v2`

A qualification artifact records that model explicitly. This matters because aligned residuals compare measured transitions with a **modeled physical-emission clock**, not with an abstract target frequency alone.

The qualification facade also records:

- the synthetic display-trace model identifier;
- the command-clock reference;
- modeled first-emission time;
- modeled response lag;
- epoch-zero semantics;
- planned clock domain;
- residual semantics.

The measured observation remains evidence authority regardless of how detailed the synthetic model becomes.

## Minimal capture format

The dependency-light CSV format is:

```csv
timestamp_s,luminance
0.000000,0.13
0.001000,0.14
0.002000,0.91
...
```

`timestamp_s` must be finite and strictly increasing.

`luminance` is a deliberately generic field. It may contain:

- photodiode voltage;
- calibrated luminance;
- ADC counts;
- normalized sensor output;
- a synthetic fixture used only to test the analysis software.

Always declare the amplitude units.

## Evidence class is explicit

A CSV file is never automatically treated as physical evidence.

Supported evidence classes are:

- `unverified_observation`
- `synthetic_fixture`
- `measured_photodiode`
- `measured_other`

CI fixtures must use `synthetic_fixture`.

A real photodiode capture should use `measured_photodiode` only when its provenance actually supports that description.

The report preserves:

- input source label/path;
- SHA-256 of the original CSV bytes;
- SHA-256 of normalized timestamp/luminance arrays;
- units;
- evidence class;
- sample count and timing geometry;
- caller-supplied string metadata.

## Transition detection

Photodiode signals are often noisy analog traces. A raw threshold can chatter near the decision boundary and create fake display transitions.

Arena therefore uses an explicit quantile-based Schmitt detector:

1. estimate low/high levels from configurable quantiles;
2. reject captures without sufficient contrast;
3. create separate low and high hysteresis thresholds around the midpoint;
4. detect alternating low-to-high and high-to-low crossings;
5. optionally suppress transitions closer than a declared minimum separation;
6. interpolate crossing time at the midpoint between adjacent samples.

Default policy:

```text
low quantile                 0.10
high quantile                0.90
hysteresis fraction          0.20
minimum transition spacing   0 s
```

Every parameter is written into the qualification report.

The detector estimates:

- transition count and timestamps;
- rising/falling labels;
- robust low/high levels and contrast;
- observed frequency from median half-period;
- transition-interval jitter.

These values describe the supplied observation only.

## Target-frequency comparison

For a selected Arena `PresentationEpoch`, the report compares observed cadence with the declared target frequency:

- target frequency;
- observed frequency;
- absolute error in Hz;
- relative error;
- error in ppm;
- transition count;
- observed transition-interval jitter;
- low/high contrast.

This comparison does **not** require the photodiode clock to be synchronized with Arena. Frequency is clock-local as long as the observation timestamps themselves are valid.

## Command clock versus modeled emission clock

Arena display trace v2 separates:

```text
commanded frame state at t
        ↓
modeled response lag
        ↓
modeled physical emission at t + lag
```

The qualification artifact therefore distinguishes two different references:

- `presentation_command_epoch_zero`: Arena command/scheduler epoch `t=0`;
- modeled emission timestamps: the synthetic display's predicted physical transition times after declared response lag.

A 17 ms modeled response lag is not hidden inside a timing residual. It is written into the epoch summary as a model parameter. A measured transition is then compared with the modeled delayed emission.

This distinction prevents a physically important latency from being counted twice or silently subtracted away.

## Clock alignment is never inferred

Onset, phase and transition timing require a shared time reference.

Arena does not estimate an offset from the same transitions it later scores. That would make a poor timing capture look artificially well aligned.

If you know the observation timestamp corresponding to **Arena presentation-command epoch `t=0`**, supply:

```text
--epoch-zero-s <timestamp>
```

Do **not** substitute the timestamp of the first observed photodiode transition. The first emitted transition occurs after whatever physical latency the display actually produced.

Only when a trustworthy command-aligned marker is available does the report include `aligned_comparison`.

Without it:

```json
"aligned_comparison": null
```

and the evidence boundary explicitly states that onset/phase residuals were omitted.

The qualification artifact records the epoch-zero interpretation as:

`observation_timestamp_corresponding_to_presentation_command_epoch_zero`

## Aligned transition comparison

When `epoch_zero_s` is supplied, measured transition timestamps are shifted into presentation-command local time and matched monotonically against the synthetic display trace's modeled emission transitions.

The planned clock domain is recorded as:

`modeled_display_emission_seconds_relative_to_command_epoch_zero`

The residual definition is recorded as:

`measured_transition_minus_modeled_emission_transition`

The comparison reports:

- planned transition count;
- measured transition count in the epoch window;
- matched, missed and extra transitions;
- match fraction;
- transition-time mean residual;
- RMSE;
- p95 absolute residual;
- maximum absolute residual;
- first matched-transition residual;
- matched transition pairs.

Transition polarity is currently **not** used for matching because photodiode circuits may invert voltage polarity. The report says so explicitly.

Arena never silently realigns the observation to improve these metrics. A wrong epoch-zero marker remains visible as a timing residual.

### What the residual means

An aligned residual is a **model-comparison residual**:

```text
measured physical transition
-
synthetic display_trace.v2 modeled emission transition
```

It is not automatically a direct measurement of monitor response latency because the synthetic model already contains a declared response-lag term.

To measure requested-to-photon latency independently, preserve an external command marker and analyze the measured transition directly against that marker. The Arena model comparison can then be used as a separate validation layer.

## CLI

The installed wheel exposes:

```bash
neuros-arena-display \
  --manifest world.json \
  --observation photodiode.csv \
  --epoch 1 \
  --units volts \
  --evidence-class measured_photodiode \
  --output display-qualification.json
```

For synchronized timing evidence:

```bash
neuros-arena-display \
  --manifest world.json \
  --observation photodiode.csv \
  --epoch 1 \
  --units volts \
  --evidence-class measured_photodiode \
  --epoch-zero-s 314.159265 \
  --output display-qualification.json
```

`--epoch-zero-s` must refer to the observation-clock timestamp aligned with Arena's **presentation command start**, not a detected photodiode edge.

Detector controls are available through CLI flags rather than hidden constants.

## Recommended physical capture

A useful competition qualification capture should record:

1. the exact monitor and refresh configuration;
2. application build/commit identifier;
3. presentation epoch / stimulus identity;
4. photodiode sensor and acquisition-device identity;
5. observation units and sampling rate;
6. a synchronized **command/start marker** when onset/phase evidence is needed;
7. raw timestamp/luminance samples without smoothing them away;
8. environmental/load condition, such as idle gameplay or full combat.

Prefer preserving the raw capture separately from the derived Arena report.

For independent requested-to-photon latency, the command marker and photodiode capture should share a trustworthy clock or independently qualified synchronization path.

## Qualification ladder

Recommended sequence:

### D0 software fixture

Run the detector against deterministic synthetic fixtures. This validates analysis software only.

### D1 stationary physical presentation

Measure Sight and Guard targets independently on the actual monitor.

Evaluate frequency, contrast and transition jitter.

### D2 synchronized command-to-photon timing

Add a trustworthy presentation-command marker and evaluate measured command-to-transition latency plus Arena model-comparison residuals as distinct quantities.

### D3 application load

Repeat during representative GPU/CPU load, including ordinary combat rendering.

### D4 presentation events

Repeat across hit-stop, Signal Break, Concord/Twin Eclipse and other game presentation states while keeping coded VEP targets classifier-blind.

### D5 session reproducibility

Repeat across launches and, if relevant, multiple display configurations.

Only after these display gates should physical EEG response testing be interpreted as evidence about the exact emitted stimulus.

## Calibration versus qualification

Do not tune detector thresholds, display parameters or game timing against one capture and report that same capture as held-out confirmation.

Use separate data for:

- calibration / threshold selection;
- qualification / acceptance evidence.

The same principle applies later when fitting synthetic participant distributions from human recordings.

## Evidence boundary

A report with `evidence_class="measured_photodiode"` may support **physical display-emission claims for that captured setup**.

It does not establish:

- human SSVEP response;
- decoder accuracy;
- attention-selection performance;
- closed-loop game efficacy;
- comfort or photosafety;
- clinical validity.

A `synthetic_fixture` report supports only the correctness of the analysis path.
