# neurOS Synthetic BCI Arena

**A reproducible BCI wind tunnel for games, interactive neurotechnology and research teams that do not have continuous access to physical hardware.**

Arena is not a universal fake brain. It is a systems environment that keeps the causal layers of an interactive BCI separate and testable:

```text
application / task state
        ↓
requested stimulus
        ↓
ACTUALLY EMITTED sensory history
        ↓
neural world model
        ↓
sensor-space EEG
        ↓
device / montage / ADC / source clock
        ↓
synchronization + transport
        ↓
decoder / quality authority
        ↓
game or application behavior
```

The same application can be exercised against deterministic fixtures, stochastic causal worlds, recorded EEG backgrounds, source-projected worlds, public datasets and eventually physical hardware without changing the application-facing evidence model.

Arena deliberately does **not** claim that synthetic success predicts human physiological performance. Every report carries a scientific evidence boundary.

## Install

```bash
pip install neuros-arena
```

Research/public-data extras:

```bash
pip install "neuros-arena[real]"
```

`[real]` is intended for MNE/MOABB/replay and domain-anchoring workflows. The normal game/CI path remains dependency-light.

---

## 1. Game developer: run a complete synthetic world

```bash
neuros-arena \
  --preset dual-target-smoke \
  --output arena-report.json \
  --write-manifest resolved-world.json
```

Or reproduce a world shared in a bug report:

```bash
neuros-arena \
  --manifest examples/arena/dual_target_custom.json \
  --output report.json \
  --npz derived-signal.npz
```

The NPZ contains both source/device timestamps and causal ground-truth timestamps so synchronization behavior can be inspected rather than hidden behind network latency.

---

## 2. Game developer: prove what the game actually did

Unity, Godot, Web and custom engines can export the engine-neutral application trace schema:

`neuros.synthetic_bci_arena.application_trace.v1`

Example events include:

```text
neural_action
neural_accept
neural_abstain
calibration_ready
calibration_failed
bci_lost
bci_recovered
participant_stop
application_state
```

Then score the trace against the exact world:

```bash
neuros-arena \
  --manifest resolved-world.json \
  --application-trace game-trace.json \
  --output closed-loop-evidence.json
```

Arena can report observations such as:

- neural actions during known no-target/rest intervals;
- actions during declared transport silence;
- stale/non-monotonic source sequences;
- actions after participant stop;
- link-loss and recovery timing;
- authority distributions.

Arena reports the observations. Your benchmark pack decides which thresholds are acceptable for your particular game.

---

## 3. Team/CI: run a versioned conformance pack

Benchmark packs are portable JSON artifacts:

`neuros.synthetic_bci_arena.benchmark_pack.v1`

Run the initial systems pack:

```bash
neuros-arena \
  --benchmark-pack examples/arena/benchmark_packs/eeg_game_system_v1.json \
  --output benchmark-result.json
```

A failing pack exits non-zero and preserves the failed metric assertions plus the complete world report.

The initial `eeg-game-systems` v1 pack checks two deliberately narrow claims:

1. a clean display-driven EEG world remains causally coupled and lossless;
2. a large source-clock offset/drift remains distinguishable from decoder-facing synchronization error.

Passing the pack is a **systems conformance claim**, not a decoder leaderboard or human-performance claim.

---

## 4. Neural world-model ladder

The generator is selected independently from device/display/network state through `WorldModelProfile` and the `neuros.world_models` plugin group.

Current built-ins:

### `legacy_synthetic` — W0

Stable deterministic frequency fixture for regression/driver/decoder tests.

It is intentionally **not** promoted as display-causal physiology.

### `driven_state_space` — W1

Default causal phenomenological model:

- endogenous posterior alpha;
- correlated stochastic background state;
- response/entrainment state;
- switching dynamics;
- artifact overlays;
- driven by the display trace that was actually emitted after frame holds/jitter/drops.

### `semi_synthetic_replay` — W2

Recorded EEG background plus a known injected causal response.

Useful when you want authentic covariance/rhythms/nuisance texture while retaining exact ground truth for the injected intervention.

It does **not** claim that the injected response occurred in the recorded participant.

### `leadfield_driven` — W3

Display-driven response projected through an explicit frozen lead-field/topography bundle.

`export_mne_forward_bundle(...)` can derive the portable bundle from explicit source columns in an MNE forward solution. Source assumptions remain visible instead of being silently guessed.

### External world models

Third-party packages can register:

`neuros.world_models`

New models may implement the original SSVEP-friendly `render(...)` contract or the richer paradigm-neutral `render_world(WorldInputBlock)` contract.

External plugins without a scientific evidence card remain runnable but are reported as `W?-external-unqualified` rather than inheriting built-in claims.

---

## 5. WorldInput: not trapped inside SSVEP

Manifest v1 remains backwards compatible with frequency-coded visual stages, but the model boundary now carries a richer `WorldInputBlock`:

```text
sample_times
paradigm
stage label
emitted sensory streams
semantic target state
task/application state
participant state
```

Today a visual stage supplies `visual_luminance` after display simulation.

The same contract is intended to support future plugins for:

- c-VEP;
- P300/ERP;
- motor imagery;
- auditory BCI;
- neurofeedback;
- EEG + eye tracking/IMU/controller hybrids;
- VR interactions;
- creative paradigms that do not fit an existing BCI taxonomy.

Old models and manifests continue to work through the compatibility adapter.

---

## 6. Timing: three clocks, not one latency number

Interactive EEG timing is modeled explicitly as:

```text
causal truth clock
       ↓
device/source clock
(offset + drift + sample timestamp jitter)
       ↓
corrected decoder-facing clock
(residual synchronization error)
```

Network delivery is measured separately.

Arena reports metrics including:

- source clock offset;
- source clock drift in ppm;
- source timestamp jitter;
- packet loss;
- p95 delivery delay;
- sequence inversions;
- corrected timestamp RMSE;
- corrected p95/max absolute timestamp error.

This lets you test a synchronization strategy rather than assuming “low network latency” means event-aligned EEG.

---

## 7. Populations instead of one fake user

```python
from neuros.arena import (
    ParameterDistribution,
    PopulationSpec,
    run_population,
)

population = PopulationSpec(
    size=1000,
    seed=7,
    parameters=(
        ParameterDistribution(
            "participant.ssvep_amplitude_uv",
            "uniform",
            low=1.0,
            high=10.0,
        ),
        ParameterDistribution(
            "participant.alpha_frequency_hz",
            "uniform",
            low=8.0,
            high=13.0,
        ),
        ParameterDistribution(
            "display.frame_drop_probability",
            "uniform",
            low=0.0,
            high=0.05,
        ),
    ),
)

result = run_population(manifest, population)
print(result.summary)
```

The result describes coverage over the **declared synthetic envelope**. It is not an estimate that some percentage of humans have those parameters.

---

## 8. Counterexample search and metamorphic tests

Some correctness properties do not require a perfect human model.

Arena can test invariants such as:

- increasing deterministic packet loss cannot deliver more packets;
- increasing deterministic display drops cannot reduce the dropped-frame set;
- degraded evidence must not increase a caller-defined gameplay authority metric;
- a display-causal world must change when emitted display timing changes.

`search_counterexamples(...)` explores a declared parameter envelope and retains complete failing manifests.

A GitHub issue can therefore contain the exact world that broke your decoder or game rather than “it sometimes gets weird when Wi-Fi is bad.”

---

## 9. Recorded EEG with traceable provenance

Arena does not replace BIDS/MNE/MOABB.

Preferred flow:

```text
BIDS / MOABB / MNE
      ↓
explicit participant/session/run/window selection
      ↓
Arena baseline NPZ
      +
RecordingMetadata JSON sidecar
```

`RecordingMetadata` carries BIDS-aligned concepts such as:

- dataset;
- subject/session/run;
- task/acquisition;
- source locator/license;
- reference and line frequency;
- channel units/types;
- electrode coordinates and coordinate system;
- preprocessing provenance.

The compact NPZ remains a derived Arena artifact, not a homebrew claim of BIDS compliance.

### MNE

```python
from neuros.arena import RecordingMetadata, export_mne_raw_baseline

export_mne_raw_baseline(
    raw,
    "background.npz",
    channel_names=["PO7", "Oz", "PO8"],
    duration_s=30,
    recording_metadata=RecordingMetadata(
        dataset="my-bids-dataset",
        subject="01",
        session="01",
        task="ssvep",
    ),
)
```

### MOABB

`iter_moabb_raw_runs(...)` follows MOABB's documented `subject → session → run → MNE Raw` contract.

`export_moabb_run_window(...)` exports only a window explicitly chosen by the study author. Arena does not decide that an arbitrary task segment is “resting baseline.”

---

## 10. Reality anchoring must be held out

With `neuros-sourceweigher`, a synthetic world bank can be weighted toward observed EEG using declared geometries such as:

- affine-invariant covariance distance;
- shared `neuros-foundation` representations.

The weights describe **similarity under that geometry**, not probabilities that a world is the participant's true brain.

### Within-recording guard split

`split_contiguous_recording(...)` and `validate_covariance_anchor_held_out(...)` prevent fitting and judging on the same samples.

### Leave-one-domain-out cohort study

`run_leave_one_domain_out_covariance_study(...)` is stronger:

1. hold out a whole participant/session domain;
2. estimate world weights only from the remaining domains;
3. average/freeze training-domain weights;
4. score the frozen weights against the unseen EEG;
5. compare against uniform synthetic weighting.

This is the starting point for testing whether synthetic calibration transfers rather than merely overfits recordings.

See `docs/PUBLIC_EEG_ANCHORING_STUDY.md`.

---

## 11. Observable EEG audit, without a magic realism score

```python
from neuros.arena.audit import eeg_observable_audit

audit = eeg_observable_audit(data_uv, sampling_rate_hz=250)
```

The versioned audit independently reports dimensions such as:

- RMS/MAD/amplitude tails;
- delta/theta/alpha/beta/gamma fractions;
- alpha peak;
- spectral entropy;
- approximate log-log spectral slope;
- temporal autocorrelation;
- zero-crossing behavior;
- channel correlation;
- covariance effective rank;
- simple flat/extreme-sample indicators.

It intentionally does **not** collapse those values into one universal realism score.

A learned generator should be capable of failing in several different ways.

---

## 12. Source/forward simulation

`export_mne_forward_bundle(...)` converts explicit fixed-orientation source columns from an MNE forward solution into a portable sensor-topography bundle.

That enables a two-stage workflow:

```text
research workstation
MNE source/forward model
       ↓
portable lead-field bundle
       ↓
CI / student laptop / game studio
leadfield_driven Arena worlds
```

Arena therefore gains source-to-sensor spatial structure without forcing every creator to install a full source-modeling environment.

---

## 13. Evidence cards and validation axes

Every world model has a machine-readable evidence card.

World-model levels:

```text
W0 regression fixture
W1 causal phenomenological
W2 recorded-background semi-synthetic
W3 source/lead-field projected
W4 future biophysical/neural-mass
W5 future learned conditional generator
W6 future externally calibrated ensemble
```

System-validation levels are separate:

```text
A0 deterministic implementation
A1 nuisance/intervention ground truth
A2 display/sensory causality
A3 acquisition device
A4 synchronization/transport
A5 decoder conformance
A6 application/game conformance
A7 held-out public/recorded EEG anchoring
A8 target physical hardware
A9 human closed-loop validation
```

A result can therefore say something concrete such as `W2/A7` rather than “our simulator is realistic.”

See `docs/SCIENTIFIC_VALIDATION_POLICY.md`.

---

## 14. Writing a third-party world model

Install/package entry point:

```toml
[project.entry-points."neuros.world_models"]
my_model = "my_package:MyWorldModel"
```

Minimum runtime surface:

```python
class MyWorldModel:
    channel_names = ("Fz", "Cz", "Pz", "Oz")

    def inject_artifact(self, kind, duration_seconds, severity):
        ...

    def render_world(self, block):
        # block.emitted_streams contains physical/simulated sensory history
        # block.target/task_state carry declared causal/task metadata
        return WorldModelEmission(data_uv=..., latent={...})

    def evidence_card(self):
        return {...}
```

A plugin may omit an evidence card during experimentation, but Arena will mark it scientifically unqualified rather than assigning it a built-in evidence tier.

---

## 15. What Arena is trying to become

There are already excellent tools for individual layers of the field:

- MNE for EEG/MEG analysis and source simulation;
- MOABB for standardized public BCI datasets/evaluation;
- BIDS/MNE-BIDS for canonical neuroscience data organization;
- LSL/XDF/MNE-LSL for synchronized streaming and replay;
- BrainFlow for board/device abstraction;
- dedicated neural-data simulators for other electrophysiology regimes.

Arena's intended role is narrower and complementary:

> **reproducible closed-loop EEG systems conformance for interactive applications.**

The long-term target is a shared ecosystem where a game developer can publish a world or benchmark pack, another developer can reproduce it anywhere, a neuroscience lab can anchor it to held-out public/human EEG, and physical hardware can later substitute into the same application boundary.

That should make scarce headset/human sessions more valuable, not make them disappear.
