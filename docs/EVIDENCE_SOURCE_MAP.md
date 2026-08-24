# Evidence Source Map

This page answers a narrower question than the general roadmap: **which public datasets should neurOS use, and what can each one actually prove?**

A large dataset is not automatically strong evidence. The best source is the one whose variation matches the failure mode we want to test.

## Recommended portfolio

| Priority | Source | Modality / scale | What it can test | Best neurOS / ORION use |
| --- | --- | --- | --- | --- |
| **A1** | [Kumar2024 / MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Kumar2024.html) | EEG, 18 participants, 6 separate-day sessions | Longitudinal drift, offline-to-online transition, adaptation/calibration cost | First executable longitudinal benchmark; SourceWeigher + task decoders + foundation representations |
| **A1** | [FALCON H1/H2](https://snel-repo.github.io/falcon/datasets.html) | Human intracortical Utah arrays, multiple days | Real iBCI cross-day stability with deliberately small new-day calibration | Primary ORION few-shot/zero-shot adaptation benchmark |
| **A1** | [IBL Repeated Site](https://docs.internationalbrainlab.org/notebooks_external/2024_data_release_repro_ephys.html) | 91 Neuropixels sessions across 12 labs | Cross-lab/site reproducibility with a standardized task and repeated target | SourceWeigher site reliability; representation invariance; mechanism replication |
| **A2** | [Wang2026 / MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Wang2026.html) | EEG, 39 participants, 5 sessions, online 1D/2D cursor | Repeated-session generalization in a real online control paradigm | Flagship non-invasive BCI evidence once exact upstream adapter/data version is pinned |
| **A2** | [Ma2020 / MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Ma2020.html) | EEG, 25 participants, 15 MI sessions | Dense session drift | Stress-test representation geometry, calibration burden, and mechanism stability |
| **A2** | [Lee2019 OpenBMI family / MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html) | EEG, same 54-person cohort across MI, ERP/P300, SSVEP | Cross-paradigm reuse while holding participant cohort relatively controlled | Test whether a foundation representation is useful beyond one BCI paradigm |
| **B1** | [Neural Latents Benchmark](https://neurallatents.github.io/datasets.html) | Macaque neural population spikes across motor, sensory, timing tasks | Canonical latent/population modeling | Validate ORION population representations against established neural-modeling baselines |
| **B1** | [IBL Brain Wide Map](https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html) | 459 sessions, 699 insertions, 139 mice, 12 labs | Large cross-lab / cross-region population-neural generalization | Pretraining, source reliability, cross-region representation and mechanism replication |
| **B2** | [EEGDash](https://eegdash.org/) | BIDS-first corpus spanning hundreds of electrophysiology datasets | Scale, dataset diversity, external-corpus transfer | Dataset discovery and pretraining reservoir; broad stress tests after focused benchmarks are stable |

## Why Kumar2024 should be first

Kumar2024 is a particularly clean first longitudinal study because it has a natural product narrative:

```text
first day / offline calibration
        -> later days / online feedback
        -> neural distribution changes
        -> how much new-day calibration does each method need?
```

The headline output should therefore be a **calibration-efficiency curve**, not only a final accuracy:

```text
held-out-session score
^
|                       best adaptive method
|                    ___/
|              _____/
|        _____/        task decoder
|   ____/
|__/ fixed decoder
+----------------------------------> labeled calibration examples / seconds
  0
```

The most interesting number is the amount of new-session calibration needed to reach a predefined performance target, with uncertainty across subjects and sessions.

## Why FALCON is the strongest ORION product benchmark

FALCON explicitly separates early held-in sessions from later held-out sessions and intentionally provides only a small calibration subset on held-out days. That makes it unusually difficult to hide the recalibration problem behind random train/test splits.

The comparison ladder should be:

```text
fixed historical decoder
  -> recent-session-only calibration
  -> pooled historical sessions
  -> SourceWeigher
  -> representation adaptation
  -> ORION
```

For each point report:

- official FALCON task metric;
- zero-calibration intercept;
- score at matched calibration budgets;
- area under the calibration-efficiency curve;
- adaptation time and memory;
- run-to-run uncertainty;
- failure behavior when a new day differs substantially from prior days.

Do not modify FALCON's chronological split to make neurOS easier to evaluate.

## Why IBL Repeated Site matters to neurOS even though it is not a human BCI

Human BCI evidence and neural-systems evidence answer different questions.

The IBL repeated-site release is valuable because multiple laboratories used a standardized behavioral task and repeatedly targeted the same brain location. That lets us ask:

- Does a learned representation carry lab identity more strongly than task information?
- Does SourceWeigher learn to discount an outlying lab/session for defensible reasons?
- Does a candidate circuit/mechanism replicate when the same experimental idea is repeated elsewhere?
- Are mechanism claims stable to spike-sorting / unit-set / recording differences?
- Which results are truly cross-site rather than artifacts of one acquisition pipeline?

This is a much stronger mechanistic-interpretability stress test than repeatedly splitting a single recording.

## Why NLB should be a secondary ORION benchmark

NLB remains a useful canonical neural-population benchmark and its former hidden test data are now available for local evaluation. Its MC_Maze, MC_RTT, Area2_Bump, and DMFC_RSG tasks span different cortical systems and dynamical regimes.

Use NLB to test whether ORION is a good **neural population model**, but not as the sole evidence of deployment stability. Several NLB datasets are single-session, so they cannot answer the same cross-day question as FALCON.

Because the test data are now public, neurOS reports should emphasize repeated random seeds and avoid repeated manual tuning on the released test split.

## Why EEGDash should not be the first benchmark

The live EEGDash catalogue is extraordinarily valuable because it standardizes BIDS entities such as subject, session, task, run, montage/acquisition information, and participant metadata across a very large corpus.

But a giant heterogeneous corpus creates many degrees of freedom:

- which datasets were selected;
- which tasks were mapped together;
- which labels were harmonized;
- which montages/sample rates were accepted;
- which recordings were excluded;
- which subjects/sites entered pretraining versus evaluation.

Therefore EEGDash should initially be used for:

1. metadata-first dataset discovery;
2. pretraining corpus construction with an immutable corpus manifest;
3. external-corpus transfer after a model is frozen;
4. leave-dataset-out stress tests;
5. identifying datasets with the exact longitudinal/site/device structure required by a new evidence question.

Every corpus build should emit the full selected recording IDs and BIDS entities before training.

## Evidence matrix

The system should eventually maintain an evidence matrix rather than a single leaderboard:

| Claim dimension | Kumar / Wang / Ma | Lee2019 family | FALCON H1/H2 | NLB | IBL repeated site | EEGDash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-session | **strong** | strong | **strong** | limited | moderate | source-dependent |
| Cross-subject | strong | **strong** | not H1/H2 | limited | strong | **very strong** |
| Cross-paradigm | limited | **strong** | movement vs communication across datasets | strong task diversity | fixed task | **very strong** |
| Cross-site/lab | limited | limited | limited | limited | **excellent** | strong if selected explicitly |
| Online BCI relevance | **strong** | strong | **excellent** | indirect | indirect | source-dependent |
| Few-shot adaptation | **strong** | possible | **benchmark-defining** | not primary | possible | source-dependent |
| Population tokenization | not current ORION contract | not current ORION contract | **strong candidate** | **excellent** | **excellent** | iEEG/EEG need separate contracts |
| Mechanism replication | session-level | paradigm-level | day-level | task/session-level | **excellent** | possible at scale |
| Pretraining scale | modest | modest | modest | modest | large | **excellent** |

## Showcase order

The public showcase should progress in this order:

1. **Reproducibility:** mock/live run -> record -> verify -> replay.
2. **Longitudinal EEG:** Kumar2024 held-out-session calibration curve.
3. **BCI adaptation:** FALCON H1/H2 later-day calibration frontier.
4. **Scientific trust:** IBL repeated-site cross-lab representation / SourceWeigher / mechanism replication.
5. **Scale:** freeze the selected representation, then test it on a manifest-defined EEGDash or Brain Wide Map corpus.

Each stage answers a different objection an external user may have:

```text
"Can I reproduce it?"
        -> "Does it survive tomorrow?"
        -> "Does it adapt with little new data?"
        -> "Does the mechanism replicate elsewhere?"
        -> "Does it generalize at scale?"
```

That sequence is the neurOS story.
