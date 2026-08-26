# EEG World Models in neurOS Arena

## Goal

`neuros-arena` is not trying to produce one universal fake human brain.

It provides a versioned environment in which multiple neural world models can be tested against the **same** display, device, clock, transport, decoder and application conditions.

The central causal chain is:

```text
application / experiment state
        ↓
requested stimulus or neural task
        ↓
actual emitted display / sensory history
        ↓
neural world model + latent state
        ↓
sensor-space neural signal
        ↓
device / montage / ADC / device clock
        ↓
transport / packetization / loss / jitter
        ↓
decoder + quality authority
        ↓
application action
```

A synthetic system can provide exact causal ground truth and reproduce faults. It cannot, by itself, establish human performance.

## World-model plugin contract

World models are neurOS plugins under:

`neuros.world_models`

A model receives:

- EEG sample timestamps;
- the **actually emitted** sensory/stimulus waveform after display timing effects;
- the intended target/task state;
- an attention/engagement drive for the current v1 SSVEP scenario;
- model-specific, versioned parameters.

It emits:

- channels × samples sensor-space data;
- channel names;
- inspectable latent summaries such as entrainment strength or model state;
- explicit provenance through the manifest/plugin identity.

The device and transport layers remain outside the model. This prevents a learned generator from silently baking one headset or network into its physiology.

## Current model ladder

### W0: `legacy_synthetic`

The original deterministic neurOS synthetic EEG generator.

Purpose:

- regression fixtures;
- driver smoke tests;
- simple frequency-decoder qualification.

Limitation:

- responds to the nominal target frequency rather than the emitted display history.

It is retained deliberately so new models can be compared against a stable legacy baseline.

### W1: `driven_state_space`

Default dependency-light Arena model.

Contains:

- stochastic correlated background state;
- endogenous posterior alpha;
- response/entrainment state;
- damped stimulus-driven dynamics;
- posterior sensor topography;
- explicit artifact overlays.

Crucially, its response is forced by Arena's sample-and-held display luminance. Frame drops, frame timing jitter and held frames therefore alter downstream synthetic EEG.

This is a phenomenological dynamical model, not a neural-mass or cellular simulation.

### W2: `semi_synthetic_replay`

Replays a recorded EEG background while injecting a known, display-driven BCI response.

This retains real background covariance, spontaneous rhythms and nuisance structure while preserving known causal labels for the injected signal.

Portable NPZ baseline contract:

- `data_uv`: channels × samples;
- `sampling_rate_hz`;
- `channel_names`.

This tier is particularly useful with public MOABB/MNE recordings before local headset access.

It does **not** imply that the injected response is the true physiological response of the recorded participant.

### W3: `leadfield_driven`

Projects the Arena-driven neural response through a frozen sensor topography derived from a forward/lead-field model.

The expensive research step can use MNE-Python:

```python
from neuros.arena import export_mne_forward_bundle

export_mne_forward_bundle(
    "subject-fwd.fif",
    "visual-forward.npz",
    visual_source_indices=[...],
    nuisance_source_indices=[...],
)
```

Arena then consumes the resulting compact bundle without requiring MNE at runtime.

Source indices are explicit. Arena does not silently guess which cortical vertices correspond to visual cortex or another task generator.

### W4: source/forward simulation adapters

Planned optional research backends:

- MNE `SourceSimulator` / `simulate_raw`;
- ERP/EOG/ECG/noise injection;
- head-position / geometry perturbation where supported;
- subject-specific or template forward solutions.

These models should be used offline or for smaller population studies when full source-space simulation is worth the cost.

### W5: neural-mass / network world models

Planned adapters for packages such as The Virtual Brain or neurolib.

Potential uses:

- sensory drive into occipital network nodes;
- changing background network state;
- cross-frequency and network-level propagation;
- slow state drift;
- multimodal source generation.

The neural-mass backend must still project to a declared sensor montage and must expose its assumptions. More biological detail does not automatically mean more accurate BCI prediction.

### W6: learned generative/dynamics models

Future `neuros-models` plugins may implement the same world-model contract using learned EEG dynamics, diffusion/state-space generators, latent sequence models or future neural foundation models.

A learned model is not promoted merely because its generated traces look realistic.

Promotion should require:

1. deterministic/reproducible sampling under a fixed seed;
2. conditioning on declared causal inputs;
3. held-out real-data comparisons;
4. preservation of known interventions in generated outputs;
5. uncertainty/provenance reporting;
6. conformance under Arena metamorphic tests;
7. mechanistic/representation diagnostics where useful.

`neuros-mechint` should be used to probe learned generators for stimulus dependence, temporal shortcuts and spurious spectral memorization.

## Synthetic populations, not an average fake person

`neuros-arena` supports distributions over world parameters.

Examples:

- SSVEP response amplitude;
- alpha frequency and amplitude;
- response delay;
- gaze duty cycle;
- display frame drops;
- ADC characteristics;
- packet loss and jitter;
- world-model parameters.

A population run reports quantiles and retains per-world parameter values.

The result describes coverage over the **declared synthetic envelope**, not prevalence in a human population.

## Metamorphic verification

Many useful properties can be tested without a perfect physiological model.

Examples:

- a higher deterministic packet-drop probability cannot deliver more packets;
- a higher deterministic frame-drop probability cannot reduce the dropped-frame set;
- a caller-declared degraded world must not create more gameplay authority;
- a stale packet must never reactivate an expired application action;
- source silence must produce a measurable transport gap and safe recovery;
- display timing changes must causally alter a display-coupled world model.

Arena can search a declared parameter envelope and return the worst resolved manifests as portable counterexamples.

## Real-data anchoring with neurOS SourceWeigher

Synthetic world weights are not manually tuned until one waveform "looks real."

Instead, a bank of worlds can be compared with an observed real/public domain using a declared feature geometry.

Current optional integrations:

- `RiemannianCovarianceWeigher` for sensor covariance geometry;
- `RepresentationSourceWeigher` for shared `neuros-foundation` embedding spaces.

The resulting weights answer:

> Which simulated domains are most similar to this observed domain under this declared comparison?

They do **not** answer:

> What is the probability that this simulated world is the participant's true brain state?

## neurOS package responsibilities

```text
neuros-core
    plugin contracts, provenance, runtime semantics

neuros-drivers
    physical/synthetic acquisition boundaries

neuros-arena
    causal worlds, population sweeps, faults, conformance

neuros-foundation / neuros-neurofm
    real/synthetic representation extraction and probing

neuros-sourceweigher
    domain similarity, transfer risk, population reweighting

neuros-models
    learned decoders and future learned world-model plugins

neuros-mechint
    learned generator/decoder mechanism and shortcut analysis

neuros-cloud
    distributed population and adversarial world execution

neuros-ui
    future visual Arena Studio / world builder
```

No package should duplicate another package's authority.

## Public-data bridge

Recommended progression before local hardware:

1. use MOABB to obtain public SSVEP/P300/MI recordings with standardized paradigm metadata;
2. convert selected resting/task background windows to the Arena semi-synthetic NPZ contract;
3. replay real recordings through MNE-LSL/LSL to exercise online pipelines;
4. extract sensor covariance and foundation-model embeddings;
5. use SourceWeigher to reweight the synthetic world bank toward observed domains;
6. hold out subjects/sessions when evaluating whether this improves predictive usefulness.

Current useful SSVEP references in MOABB include datasets with 10 and 12 Hz conditions, making them directly relevant to Mindforge-style dual-target experiments.

## Paradigm-general roadmap

Manifest v1 remains backward-compatible and SSVEP-oriented.

A future manifest v2 should introduce a generic neural drive/event interface rather than overloading `target_frequency_hz`:

```text
WorldInput
  paradigm
  task / intent label
  sensory event timeline
  emitted stimulus channels
  gaze / eye state
  movement / IMU state
  controller actions
  feedback state
  context covariates
```

That enables world models for:

- SSVEP / c-VEP;
- P300 and other ERPs;
- motor imagery / sensorimotor rhythms;
- neurofeedback;
- auditory BCIs;
- hybrid EEG + IMU/controller systems;
- creative custom paradigms.

The v2 design should adapt v1 scenarios rather than invalidate existing shared worlds.

## Evidence ladder

```text
A0 deterministic implementation
A1 nuisance and intervention ground truth
A2 display/sensory causality
A3 acquisition device model
A4 transport/clock model
A5 decoder conformance
A6 application conformance
A7 public/recorded human-data anchoring
A8 physical hardware substitution
A9 local human closed-loop validation
```

Only the final tiers support corresponding physical/human claims.
