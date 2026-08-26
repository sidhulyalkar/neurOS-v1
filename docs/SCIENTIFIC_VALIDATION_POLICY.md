# Synthetic BCI Arena Scientific Validation Policy

## Purpose

`neuros-arena` exists to accelerate interactive BCI development when physical hardware and human testing are limited. Its strongest contribution is not a claim to perfectly simulate a human brain. It is a reproducible environment in which causal assumptions, timing, device behavior, transport failures, decoder authority, application behavior, and synthetic-to-recorded similarity can be tested explicitly.

The governing rule is:

> **A more complex neural generator does not receive a larger scientific claim scope unless independent evidence justifies it.**

Synthetic conformance, recorded-data similarity, device validation, and human closed-loop validation are distinct evidence classes. Reports must preserve those distinctions.

## 1. Two independent evidence axes

Arena separates **world-model evidence** from **system-validation evidence**.

### 1.1 World-model evidence levels

These levels describe what a neural generator represents. They are not a linear score of biological truth.

| Level | Meaning | Typical use | Explicit limitation |
|---|---|---|---|
| W0 | Regression fixture / analytic signal source | driver, decoder, CI smoke tests | may not be stimulus-causal or physiologically meaningful |
| W1 | Causal phenomenological dynamics | closed-loop stress, population testing, timing/display studies | not a biophysical cortical model or human distribution |
| W2 | Recorded-background semi-synthetic model | real nuisance/covariance with known injected interventions | injected response is not evidence that the recorded participant produced it |
| W3 | Explicit source-to-sensor / lead-field projection | montage/topography/source sensitivity | source response amplitudes/dynamics remain assumptions unless independently calibrated |
| W4 | Biophysical / neural-mass model | mechanistic hypothesis testing | equations and parameters require external validation; complexity is not automatically realism |
| W5 | Learned conditional neural generator | high-dimensional conditional worlds | vulnerable to memorization, shortcut learning, mode collapse and distribution leakage |
| W6 | Externally calibrated multi-model ensemble | uncertainty-aware world populations anchored to held-out recordings | remains a model of a declared domain, not a substitute for human closed-loop testing |

Built-in reports include a machine-readable `WorldModelEvidenceCard`. Third-party world models without an evidence card remain runnable but are reported as scientifically unqualified.

### 1.2 System-validation levels

| Level | Evidence |
|---|---|
| A0 | deterministic seed / manifest reproducibility |
| A1 | known nuisance and intervention ground truth |
| A2 | emitted-stimulus / display causality |
| A3 | device/montage/ADC/clock behavior |
| A4 | transport, synchronization, interruption and recovery behavior |
| A5 | decoder / quality-authority conformance against known truth |
| A6 | closed-loop application/game conformance |
| A7 | held-out recorded/public EEG anchoring |
| A8 | target physical device substitution and timing verification |
| A9 | human closed-loop validation under a declared protocol |

A project should report both axes. For example, `W2/A7` conveys a different evidence state from `W1/A6`.

## 2. Claims Arena may and may not support

### Synthetic evidence can support

- deterministic software behavior under a declared world;
- causal response to a known simulated intervention;
- timing error relative to simulator ground truth;
- fail-closed behavior during corrupted or absent input;
- decoder/application performance over a declared synthetic parameter envelope;
- relative differences between declared synthetic scenarios;
- whether synthetic-domain weighting transfers to held-out recorded data under a declared similarity metric.

### Synthetic evidence alone cannot support

- expected accuracy on humans;
- prevalence of responder/non-responder phenotypes;
- human comfort, fatigue, cybersickness, visual safety or usability;
- subject-specific physiology;
- a claim that an injected semi-synthetic response occurred in the recorded participant;
- anatomical localization without independently justified geometry/source assumptions;
- clinical validity or diagnostic utility.

## 3. Leakage and split policy

Synthetic-to-real evaluation must be designed so that the result could fail.

### 3.1 Never tune and judge on the same recording

If world parameters or SourceWeigher weights are calibrated on real EEG, final similarity or downstream performance must be evaluated on independent data.

Preferred split hierarchy, strongest first:

1. held-out participants;
2. held-out participant-sessions;
3. held-out runs;
4. temporally separated windows with a guard interval.

Randomly shuffling overlapping EEG samples is not an acceptable independent validation split.

### 3.2 Preprocessing belongs inside the split

Any data-dependent operation must be fitted only on the training/calibration partition, including:

- normalization/statistical scalers;
- covariance shrinkage choices when tuned;
- learned filters;
- foundation-model adapters or task heads;
- source/domain weighting temperatures when optimized;
- generative-model training or fine-tuning;
- learned artifact models.

Fixed, predeclared signal transforms may be reused across partitions but must be recorded.

### 3.3 Cohort calibration

For public-data calibration, Arena supports leave-one-domain-out studies. A domain should normally be a participant or participant-session. Synthetic-world weights are fitted only on the non-held-out domains, frozen, then scored against the held-out EEG.

The principal comparison is against a predeclared baseline such as uniform synthetic-world weighting. A positive transfer result is evidence for the anchoring procedure, not proof that the synthetic worlds are physiological digital twins.

## 4. Timing and synchronization policy

Interactive EEG systems must distinguish three clocks:

1. **causal ground-truth time**: the simulator/application event timeline;
2. **source/device time**: acquisition timestamps including offset, drift and timestamp jitter;
3. **decoder-facing corrected time**: timestamps after synchronization/correction, including residual error.

Network delivery latency is a fourth quantity and must not be used as a proxy for timestamp accuracy.

Arena timing reports should include, where applicable:

- source clock offset;
- source clock drift in ppm;
- source timestamp jitter;
- packet loss;
- arrival delay distribution;
- sequence inversions/reordering;
- interruption duration;
- corrected timestamp RMSE;
- corrected timestamp p95 absolute error;
- maximum corrected timestamp error;
- stimulus/event-marker to EEG alignment error.

Physical A8 validation should compare software timestamps to independent measurements when feasible, such as photodiode traces for visual stimuli.

## 5. Display and sensory causality

For stimulus-driven paradigms, neural models promoted beyond W0 must consume the **emitted** stimulus after presentation effects, not merely the requested label/frequency.

For visual BCI this includes, when modeled:

- refresh-rate quantization;
- sample-and-hold behavior;
- frame drops/holds;
- render jitter;
- display response lag;
- luminance modulation.

Future paradigms should use `WorldInputBlock.emitted_streams` for actual sensory input, including audio, haptic or multimodal streams.

A model that receives only the target class or requested frequency may remain useful as a regression fixture, but must not claim presentation-to-neural causality.

## 6. Recorded EEG and BIDS alignment

Arena does not invent a new canonical EEG storage standard.

Preferred upstream flow:

```text
public/private EEG
      ↓
BIDS / MNE / MOABB
      ↓
explicit subject/session/run/task split
      ↓
Arena derived baseline + recording metadata sidecar
```

Arena's compact baseline NPZ is a derived simulation artifact. The accompanying `RecordingMetadata` sidecar records BIDS-aligned provenance such as dataset, subject/session/run identifiers, task, acquisition, channel units/types, reference, line frequency, electrode coordinates/coordinate system, preprocessing and source locator.

Do not put direct participant identifiers or secrets in Arena metadata.

## 7. Evaluation of learned EEG world models

GAN/VAE/diffusion/foundation-style generators are experimental until they satisfy more than visual or spectral plausibility.

A W5 candidate should be evaluated on at least the following dimensions.

### 7.1 Held-out generalization

- held-out participants;
- preferably held-out sessions/sites/devices;
- no test-subject fine-tuning unless separately reported as calibration.

### 7.2 Observable signal structure

- spectral density/band structure;
- channel covariance / SPD geometry;
- temporal autocorrelation;
- cross-channel coherence where appropriate;
- amplitude distributions and tails;
- artifact statistics;
- event-related or stimulus-locked structure appropriate to the paradigm.

No single scalar should be called a universal realism score.

### 7.3 Representation-level comparison

Use predeclared `neuros-foundation`/external embeddings and SourceWeigher or other declared metrics to test whether synthetic and held-out real EEG occupy comparable representation domains.

The embedding model used for evaluation should not be trained solely to make the generator look good.

### 7.4 Intervention faithfulness

Changing a known causal input should change the generated signal in the expected direction while preserving unrelated nuisance structure as appropriate.

Examples:

- remove visual modulation → stimulus-locked response should disappear/reduce;
- weaken attention/gaze coupling → target evidence should not increase;
- change source location → projected topography should change coherently;
- increase device dropout → decoder authority must not increase solely because of missing data.

### 7.5 Metamorphic testing

A learned model must pass declared invariants and adversarial world search. Counterexamples are retained rather than averaged away.

### 7.6 Mechanistic/shortcut audit

Where model architecture permits, use `neuros-mechint` and controlled counterfactuals to determine whether output conditioning depends on intended temporal/spatial structure or shortcuts such as class-specific amplitude constants, trial position, subject identity leakage or preprocessing fingerprints.

### 7.7 Uncertainty

A generative model should expose stochastic variability or an ensemble/distribution over plausible worlds. One deterministic waveform should not be presented as the expected human response.

## 8. Model promotion gates

A new world model should enter Arena as experimental and be promoted only when its evidence card is updated with concrete validation artifacts.

Minimum requirements:

### W0 → W1

- seeded determinism;
- explicit causal inputs;
- emitted-stimulus coupling tests;
- intervention/metamorphic tests;
- failure behavior and provenance.

### W1 → W2/W3

- W2: recorded-human background provenance and held-out usage policy;
- W3: explicit source/forward projection provenance and geometry checks.

### W3/W4/W5 → W6 ensemble

- multiple independently useful world families;
- held-out recorded-domain calibration;
- transfer improvement over a declared baseline;
- uncertainty/population reporting;
- stable benchmark-pack results;
- no hidden use of held-out test domains for tuning.

No synthetic model can be promoted to A8/A9. Those are system evidence levels requiring physical hardware/human validation.

## 9. Benchmark-pack policy

Arena benchmark packs are versioned artifacts:

`neuros.synthetic_bci_arena.benchmark_pack.v1`

A pack should contain:

- exact world manifests;
- exact metric paths and thresholds;
- a semantic version;
- a written claim scope;
- enough provenance to reproduce the run.

Changing a threshold, world distribution, model family or evidence expectation requires a version change.

A benchmark result must preserve failed assertions and the underlying world report. Passing a pack means only what the pack's `claim_scope` states.

The initial `eeg-game-systems` pack tests clean causal display coupling and clock-domain recovery. It is deliberately a systems benchmark, not a decoder leaderboard or human-performance benchmark.

## 10. Game/application integration policy

A BCI game should keep fast physical controls and bounded neural authority semantically distinct unless the paradigm explicitly requires otherwise.

Recommended application-level conformance properties include:

- no neural action during declared rest unless the decoder explicitly accepts one;
- no stale packet may resurrect expired authority;
- transport loss cannot create gameplay authority;
- degraded neural evidence should increase abstention/fallback rather than confident random action;
- participant stop is terminal until explicitly reset;
- calibration failure cannot silently enter live-authority mode;
- recovery after link loss requires a stable re-entry period;
- rendering/game hit-stop must not alter independently clocked stimulus timing assumptions without being detected.

Developers may define additional metrics through benchmark evaluators rather than modifying Arena core.

## 11. Public human anchoring campaign

The first public SSVEP campaign should use multiple datasets and held-out domains rather than optimizing only for Mindforge frequencies.

Initial candidates include MOABB-exposed SSVEP datasets such as:

- MAMEM2, useful for 250 Hz recordings and a dense montage;
- Lee2019_SSVEP, useful for larger subject count and two-session evaluation;
- Nakanishi2015 / Wang2016 for broader frequency/codebook coverage;
- additional datasets as licensing and acquisition metadata permit.

Recommended experiment:

1. convert/download through MOABB/MNE and preserve upstream provenance;
2. define participant/session domains before analysis;
3. select a predeclared common channel/montage strategy;
4. create Arena baselines with recording sidecars;
5. generate a declared synthetic world bank;
6. fit world weights only on training domains;
7. freeze the world bank and weights;
8. evaluate covariance/representation transfer on held-out domains;
9. evaluate whether synthetic-selected decoder/system configurations improve held-out real-data behavior relative to baselines;
10. publish failures, negative results and counterexamples alongside positive results.

Mindforge should be one downstream showcase, not the scientific definition of the benchmark.

## 12. External ecosystem interoperability

Arena should compose with existing tools rather than duplicate them:

- **MNE-Python**: source/forward simulation, EEG signal analysis and artifact/noise utilities;
- **MOABB**: standardized public BCI datasets/paradigms and BIDS conversion;
- **BIDS / MNE-BIDS**: canonical dataset organization and metadata;
- **LSL / XDF / MNE-LSL**: synchronized streaming and replay;
- **BrainFlow**: board/device abstraction and synthetic-board comparison;
- **TVB / neurolib**: optional neural-mass / whole-brain dynamics;
- **external neural-data simulators**: complementary invasive/electrophysiology or task simulators;
- **Unity/Godot/Web engines**: application adapters and event traces.

Arena's unique focus is **reproducible closed-loop EEG systems conformance for interactive applications**, not ownership of every layer of neuroscience simulation.

## 13. Reproducibility checklist

Every published Arena result should record:

- repository/package version or commit;
- benchmark-pack version if used;
- world manifest(s);
- seeds;
- world-model evidence cards;
- dataset/source provenance for recorded EEG;
- subject/session/run split;
- preprocessing fitted on each split;
- device/display/transport profiles;
- timing metrics;
- evaluation metrics and uncertainty/dispersion;
- failed cases/counterexamples;
- explicit unsupported claims.

If an external party cannot reconstruct the causal world and evaluation boundary, the result is not yet an Arena-grade scientific artifact.
