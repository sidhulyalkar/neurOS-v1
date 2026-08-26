# Public EEG Anchoring Study v1

## Research question

Can a population of causal synthetic EEG worlds, calibrated only on training participants/sessions, provide useful information about **held-out real EEG domains** and help select closed-loop BCI configurations that transfer better than unweighted synthetic testing?

This study is designed to fail cleanly. A negative result is useful evidence that a world family, similarity geometry, or calibration procedure does not transfer.

## Two hypotheses, kept separate

### H1: domain transfer

Synthetic-world weights learned from training EEG should reduce the average declared domain distance to held-out EEG compared with a uniform synthetic-world population.

Primary endpoint:

`leave-one-domain-out relative improvement in held-out covariance/representation distance`

This is a signal-domain hypothesis only.

### H2: system-selection transfer

BCI configurations selected using the training-anchored synthetic population should outperform or be more robust than configurations selected using an unweighted synthetic population when evaluated on held-out real EEG.

Possible endpoints:

- accepted precision;
- abstention rate;
- false activation rate;
- switch/decision latency;
- calibration burden;
- robustness to channel removal or timing perturbation;
- application-level authority errors when replayed through a closed-loop adapter.

H2 is stronger than H1 and must not be inferred from H1 alone.

## Initial datasets

The campaign should begin with multiple MOABB-exposed SSVEP datasets rather than one Mindforge-specific dataset.

### MAMEM2

Useful properties:

- 10 participants;
- 256 EEG channels;
- 250 Hz acquisition;
- five SSVEP classes;
- 20–30 trials per class;
- 3 s trials;
- raw EEG available through MOABB.

This is especially useful for dense spatial/covariance studies and aligns well with 250 Hz device simulations.

### Lee2019_SSVEP

Useful properties:

- 54 participants;
- 62 EEG channels;
- two sessions;
- four SSVEP classes;
- 50 trials per class;
- 4 s trials;
- raw EEG available;
- MOABB exposes clean BIDS conversion.

This dataset is valuable for participant-held-out and session-held-out transfer because it has a substantially larger cohort and repeated sessions.

### Broader-frequency follow-up

After the first pipeline is stable, add datasets such as Nakanishi2015 and Wang2016 to test whether Arena conclusions survive broader stimulus-frequency/codebook regimes rather than only the frequencies used by one game.

## Canonical upstream data flow

```text
MOABB dataset
    ↓
dataset.get_data(...)
subject → session → run → MNE Raw
    ↓
optional clean BIDS export / source provenance
    ↓
predeclared channel and time-window policy
    ↓
Arena baseline NPZ + RecordingMetadata sidecar
```

Arena must not silently infer participant identity, task semantics, reference or coordinate assumptions that belong to the upstream dataset.

## Split policy

### Participant-held-out study

Default unit of independence: participant.

For each fold:

1. choose one participant as held out;
2. fit any world-population/domain-weight parameters only on the remaining participants;
3. freeze parameters;
4. compute held-out EEG similarity and downstream decoder/system metrics;
5. store the full fold artifact.

### Session-held-out study

For Lee2019, add a stronger longitudinal test:

- fit on session 1 only;
- evaluate the same participant in session 2;
- and the reverse direction where appropriate.

This directly tests session drift rather than allowing a simulator to overfit one recording day.

### Run/window split

Run-level or temporal splits are secondary analyses, not substitutes for participant-held-out evaluation.

If windows are used, employ contiguous splits with a guard interval large enough to prevent overlap/preprocessing leakage.

## Channel strategy

Run at least two channel regimes.

### Native montage

Compare worlds in each dataset's native available EEG montage when a compatible world/source model exists.

Purpose:

- evaluate spatial covariance/topography faithfully within a dataset;
- avoid throwing away useful dense-array information.

### Portable game montage

Define a predeclared low-channel posterior montage that approximates common game/consumer BCI access, for example available occipital/parietal channels nearest to a target set.

Purpose:

- test whether findings survive the constrained sensor regime relevant to interactive BCI applications;
- compare device/channel sensitivity.

Channel mapping rules must be declared before held-out evaluation and stored in provenance.

## Preprocessing policy

Keep a minimally processed branch for domain-realism analysis and a decoder-specific branch for task evaluation.

### Domain comparison branch

Suggested starting operations:

- channel selection;
- explicit unit normalization;
- resampling only when needed for a common device model;
- optional fixed notch/band limits chosen before held-out evaluation;
- no subject-specific data-dependent normalization fitted using held-out data.

### Decoder branch

Decoder-specific preprocessing may be used, but every data-dependent transform must be fitted only on training/calibration partitions.

The raw/derived path must be recorded in `RecordingMetadata.preprocessing`.

## Synthetic world bank

The v1 world bank should deliberately contain multiple model families rather than thousands of parameter variants from one family.

Suggested composition:

- W1 `driven_state_space` population;
- W2 `semi_synthetic_replay` worlds using only training-domain backgrounds;
- W3 `leadfield_driven` worlds with explicit source/topography variants;
- selected nuisance/device/display/clock/network variants.

Do not allow a held-out participant's background EEG into a W2 world used to predict that same participant.

Population axes should include predeclared ranges for:

- target response strength;
- alpha peak/amplitude;
- switching/entrainment dynamics;
- channel/background covariance family;
- artifact susceptibility;
- sensor noise;
- channel loss;
- source clock offset/drift/jitter;
- residual synchronization error;
- display timing perturbation where the task simulation includes presentation.

Synthetic ranges are engineering envelopes until empirical calibration justifies interpreting them as population priors.

## Reality anchoring

Run several declared geometries rather than selecting whichever looks best after seeing test results.

### Sensor covariance

Use `RiemannianCovarianceWeigher` through Arena's covariance anchor.

Advantages:

- interpretable for EEG domain structure;
- independent of a learned embedding model;
- useful baseline.

### Foundation representation

Use one or more fixed, predeclared `neuros-foundation` or external EEG representations.

Rules:

- embedding model must be frozen before held-out evaluation;
- any adapter/task head must be trained only on training domains;
- report source weighting diagnostics and effective world count;
- do not call the resulting weight a probability of physiological truth.

### Observable features

Retain simple spectral/amplitude/correlation signatures as an audit layer. They are useful for explaining why a representation-level metric may behave unexpectedly.

## Leave-one-domain-out test

Arena implements:

`run_leave_one_domain_out_covariance_study(...)`

For every held-out domain:

1. independently compute world weights for every training domain;
2. average and normalize those training-domain weights;
3. freeze them;
4. inspect the held-out domain only to compute evaluation distances;
5. compare weighted distance with uniform synthetic weighting.

Primary summary:

- mean relative improvement;
- median relative improvement;
- fraction of held-out domains improved;
- mean weighted distance;
- mean uniform distance;
- per-domain counterexamples.

A model family should not be promoted merely because the mean improves if a meaningful subgroup degrades severely.

## Decoder/system-selection experiment

Choose a small predeclared set of candidate decoder/system configurations. For SSVEP this could include variants of:

- analysis-window length;
- hop size;
- harmonic count;
- filter-bank configuration;
- quality threshold;
- winner/runner-up margin threshold;
- dwell/confirmation count;
- channel subset;
- frequency/codebook configuration where appropriate.

For every training fold:

1. evaluate candidates over the training-anchored synthetic population;
2. select the candidate under a declared utility function;
3. freeze it;
4. evaluate on the held-out real participant/session;
5. compare against baselines such as default configuration, real-training-only selection, and unweighted synthetic selection.

A useful initial utility should reward accepted precision and fail-closed behavior rather than raw selection rate alone.

## Application replay experiment

Once game/application traces are available:

```text
held-out EEG or semi-synthetic replay
        ↓
real decoder
        ↓
portable neural events
        ↓
application/game adapter
        ↓
ApplicationTrace
        ↓
Arena causal scoring
```

Use the engine-neutral `ApplicationTrace` contract to measure:

- unintended neural actions during known no-target periods;
- actions during source silence where disallowed by the application policy;
- stale sequence regressions;
- post-participant-stop actions;
- link-loss and recovery timing;
- application-specific authority metrics.

Mindforge can be the first complete showcase, but the trace contract and benchmark packs must remain game-engine and application neutral.

## Statistical reporting

The first study should report distributions/folds rather than only a pooled mean.

At minimum:

- every held-out fold;
- mean and median;
- bootstrap confidence intervals where appropriate;
- fraction of folds improved;
- worst fold;
- effect relative to uniform synthetic baseline;
- all study/benchmark seeds and manifests;
- failures and excluded recordings with reasons.

If multiple metrics/geometries are explored, distinguish exploratory analyses from preregistered/primary endpoints.

## Promotion criterion for reality anchoring

Do not claim that Arena is empirically calibrated to public human EEG until a frozen study shows reproducible held-out transfer.

A reasonable first promotion gate is:

1. positive median held-out improvement over uniform world weighting across at least two independent datasets or dataset/session regimes;
2. no catastrophic hidden subgroup failure under the declared metric;
3. reproducible world/recording provenance;
4. independent rerun from a clean environment;
5. H2 system-selection evidence reported separately from H1 signal-domain evidence.

If this gate is not met, keep reality anchoring labeled experimental and publish the negative result.

## Why this matters for games

Game developers rarely have hundreds of headset sessions available while designing mechanics. A validated synthetic ecology could help answer questions earlier:

- Is this BCI mechanic viable across weak/strong responders?
- Does a longer evidence window make combat safer but unusably sluggish?
- How much channel loss can the game tolerate?
- What happens when rendering corrupts a visual code?
- Does link recovery preserve agency?
- Which settings remain robust when the real EEG domain shifts?

The goal is not to remove human playtesting. The goal is to make the expensive human sessions much more informative because obvious software, timing and robustness failures have already been explored systematically.
