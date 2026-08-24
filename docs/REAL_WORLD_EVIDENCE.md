# Real-World Evidence Program

neurOS should be showcased by the **quality of the evidence chain**, not by one favorable accuracy number.

The public story should answer three progressively harder questions:

1. **Does the software reproduce exactly?** Can a run be recorded, verified, replayed, and tied to the same config/model/data identity?
2. **Does the decoder survive deployment shift?** What happens on a new subject, session, day, or recording, and how much calibration is needed to recover performance?
3. **Does the physical system behave as claimed?** On named hardware/firmware/transports, what are packet loss, clock uncertainty, queue loss, latency, reconnect behavior, and sustained recording reliability?

A result is valuable only if the evidence tier and unsupported claims are explicit.

## Source selection principles

Prefer public sources that have at least one of these properties:

- repeated sessions/days from the same participant;
- actual online BCI feedback rather than only offline classification;
- a benchmark-defined chronological held-in/held-out split;
- enough subjects to test cross-subject transfer;
- standardized metadata such as BIDS or NWB;
- stable DOI/version identity and a redistributable or clearly documented license;
- an external benchmark ecosystem so neurOS does not grade its own homework.

Large corpora without a clear deployment question belong in pretraining/discovery, not in the headline benchmark.

## Recommended evidence sources

The machine-readable subset of this table is exposed by:

```bash
neuros-foundation evidence
neuros-foundation evidence --role longitudinal_bci
neuros-foundation evidence --modality intracortical --json
```

### Tier A: longitudinal non-invasive BCI

| Source | Why it matters | Primary neurOS question |
| --- | --- | --- |
| [Wang2026 / MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Wang2026.html) | 39 participants, 5 sessions, 62-channel EEG, four MI classes, online 1D/2D cursor control and multiple training/control cohorts | Can a decoder/representation remain useful across sessions in a real online BCI training protocol? |
| [Kumar2024 / MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Kumar2024.html) | 18 BCI-naive participants over 6 separate-day sessions; first session offline, later sessions use continuous online feedback | How much labeled or unlabeled calibration is required on a new day, and does transfer reduce it? |
| [Ma2020 / MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Ma2020.html) | 25 participants and 15 MI sessions per participant across three days | Does representation geometry, decoding utility, and candidate mechanism stability survive dense session drift? |

**Use:** this should be the primary non-invasive benchmark family. MOABB supplies an external, reproducibility-focused evaluation ecosystem rather than a neurOS-specific dataset loader.

### Tier A: invasive BCI adaptation

[FALCON](https://snel-repo.github.io/falcon/) is especially aligned with ORION because its central question is stable neural decoding on later days with very little new-day calibration.

| Source | Public identity | neurOS role |
| --- | --- | --- |
| [FALCON H1](https://snel-repo.github.io/falcon/datasets.html) | DANDI `000954`; human Utah-array reach/grasp iBCI | Cross-day continuous decoding and few-shot adaptation |
| [FALCON H2](https://snel-repo.github.io/falcon/datasets.html) | DANDI `000950`; human handwriting / brain-to-text iBCI | Cross-day communication decoding and calibration efficiency |

FALCON already defines chronological held-in and later held-out sets. Preserve those splits exactly. Do not create an easier random neurOS split and compare it to FALCON results.

Before applying current ORION event/tokenization methods, inspect the released neural representation for each task and record any conversion from source features to neurOS `SignalFrame`/token contracts. A tokenization benchmark is meaningful only when the source representation supports that interpretation.

### Tier B: scale and representation pretraining

[EEGDash](https://eegdash.org/) exposes 700+ BIDS-first electrophysiology datasets through standardized subject/session/task/run metadata and interoperates with MNE/Braindecode.

Use EEGDash for:

- corpus discovery;
- pretraining dataset mixtures;
- external-dataset stress tests;
- checking whether a representation transfers beyond the benchmark it was tuned on.

Do **not** reduce EEGDash to one aggregate accuracy number. Dataset composition, task distribution, licenses, preprocessing, and subject/site leakage must remain visible.

### Tier C: canonical fast regression

Keep one small, established MOABB motor-imagery dataset as a quick developer/CI benchmark. Its job is not to prove a product claim; its job is to catch benchmark-pipeline regressions before expensive longitudinal runs.

## Evidence tracks

### Track 1 — Longitudinal EEG: “Does it still work next session?”

Start with Kumar2024 because the offline-to-online progression maps naturally onto calibration/adaptation experiments. Run Ma2020 as the high-session-count stress test. Use Wang2026 as the flagship online-control study once the exact adapter/data version used by the run is pinned.

For every dataset, compare the same split and downstream capacity across:

1. transparent classical baseline (for example CSP + linear classifier where appropriate);
2. task-specific neurOS decoder such as EEGNet / EEG-Conformer;
3. frozen foundation-model representation + matched linear readout;
4. optional SourceWeigher transfer strategy;
5. any future ORION EEG representation only after it has an explicit EEG input/token contract.

Report more than task score:

- balanced accuracy / ROC-AUC or task-appropriate primary metric;
- per-session degradation from the source/calibration domain;
- calibration samples or seconds required to reach target performance;
- performance-vs-calibration curve and area under that curve;
- expected calibration error / Brier score where probabilistic output exists;
- subject/session leakage probe on learned representations;
- representation effective rank / anisotropy / CKA;
- channel-dropout and montage perturbation robustness;
- repeated-seed uncertainty;
- training/adaptation wall-clock and inference latency measured separately.

### Track 2 — FALCON: “Can ORION reduce recalibration burden?”

Respect the benchmark's chronological held-in/held-out days. Evaluate a ladder rather than only the best method:

```text
fixed decoder
  -> simple recent-session recalibration
  -> pooled-source transfer
  -> SourceWeigher
  -> representation adaptation
  -> ORION strategy
```

The central x-axis is **new-day calibration budget**, not model size.

A strong figure is performance against calibration examples/minutes, with the fixed decoder as the zero-calibration intercept. For methods that update online, report both final score and the adaptation trajectory.

For H1/H2, keep the official FALCON task metric and latency definition alongside any neurOS diagnostics. neurOS-specific representation or mechanism metrics are supplementary evidence, not replacement leaderboard metrics.

### Track 3 — Hardware: “Can I trust what actually ran?”

Use one accessible reference configuration first, for example an OpenBCI Cyton-class EEG board through the hardened BrainFlow source, plus an LSL marker/event stream once LSL is promoted to a first-class neurOS source.

A hardware qualification profile should pin:

- physical device and board ID;
- firmware and acquisition library versions;
- operating system and transport (USB/Bluetooth/Wi-Fi/serial as applicable);
- channel names/types/units and actual device sampling rate;
- timestamp source and clock domain;
- LSL clock-offset measurements if LSL is used;
- packet/sample loss and neurOS queue loss;
- clock drift/uncertainty;
- reconnect/recovery behavior;
- sustained run duration;
- source-to-decision p50/p95/p99 latency;
- recording integrity and successful deterministic replay;
- exact config, package, Git, model, and artifact identity.

Run at least a short interactive demo and a longer soak test. The long test is what turns “it worked on my laptop” into hardware evidence.

## Leakage-resistant partition contract

`neuros-foundation` now provides a pre-model split contract:

```python
from neuros.foundation_models import (
    GroupedEvaluationData,
    hold_out_groups,
)

# Conventional MOABB output: X, labels, metadata
bundle = GroupedEvaluationData.from_moabb_result(
    (X, labels, metadata),
    dataset_id="moabb-kumar2024",
)

partition = hold_out_groups(
    bundle,
    split_unit="session",
    held_out_values=["5"],
)
protocol = partition.protocol(
    name="kumar2024-held-out-session",
    transfer_regime="few_shot",
)
manifest = partition.manifest(protocol=protocol)
```

The manifest is emitted **before model fitting** and records the deployment-unit split, protocol fingerprint, sample counts, source card, and partition fingerprint. It cannot prove that downstream code avoided every leakage route, but it makes accidental subject/session mixing substantially harder to hide.

For an installed MOABB object/paradigm:

```python
from neuros.foundation_models import collect_moabb

bundle = collect_moabb(dataset, paradigm, subjects=[1, 2, 3])
```

Install the optional evidence ecosystem with:

```bash
python -m pip install -e "packages/neuros-foundation[evidence]"
```

Pin the exact MOABB/data version in promoted benchmark artifacts. The curated source registry may describe a dataset available in current upstream documentation before that adapter appears in a particular installed release.

## The showcase: neurOS Evidence Console

The eventual UI should not look like a generic ML dashboard. It should look like an **evidence console** with a single run identity and four linked panels.

### 1. Reproducibility

Show:

- run / artifact fingerprint;
- Git and package identity;
- dataset DOI/version;
- config hash;
- replay integrity status;
- exact train/calibration/test split.

Headline: **“Can this result be reproduced?”**

### 2. Generalization

Show:

- held-out subject/session/day score;
- source-domain score beside it;
- degradation across time;
- calibration-efficiency curve;
- comparison to simple baselines.

Headline: **“What happens when the neural distribution moves?”**

### 3. Trust / mechanism

Show separately:

- domain leakage;
- SourceWeigher weights and stability;
- channel/unit perturbation sensitivity;
- candidate mechanism intervention effect;
- mechanism replication/stability across deployment units.

Headline: **“Why should we trust the transfer?”**

Do not collapse attribution, causal faithfulness, and biological interpretation into one “explainability” score.

### 4. Runtime / hardware

Show:

- live stream/device descriptors;
- queue loss;
- clock uncertainty/drift;
- p50/p95/p99 latency;
- reconnect events;
- recording/replay integrity.

Headline: **“Did the physical system satisfy its contract?”**

## Recommended first public demo

A compelling public demo can be run as a three-act sequence:

### Act I — Exactness

```text
mock/live source -> neurOS graph -> record -> verify -> replay
```

Show that replay preserves the same signal/runtime contract and evidence identity.

### Act II — Shift

Load one longitudinal MOABB dataset. Fit only on allowed sessions, then reveal a held-out session. Compare a transparent baseline, a neurOS decoder, a foundation representation, and SourceWeigher. Move a calibration-budget slider and redraw performance.

### Act III — Adaptation

Switch to a FALCON cross-day task. Show the official early-day/later-day boundary and the performance-vs-calibration curve. If ORION improves the frontier, the visual claim becomes extremely concrete: **the same neural interface needs less recalibration on a later day**.

That is a much stronger product demonstration than “our transformer got 2% more accuracy.”

## Promotion gates

A result may be featured publicly only when the artifact includes:

- immutable source identity/version;
- explicit split unit and held-out identities;
- preprocessing fit boundary;
- model/representation identity and hash;
- calibration/adaptation budget;
- repeated-seed uncertainty or justified deterministic protocol;
- primary task metric plus relevant robustness/calibration metrics;
- exact code/package environment;
- limitations and unsupported claims;
- replay/reproduction instructions where applicable.

Hardware claims additionally require a named qualification manifest. Closed-loop or clinical claims require separate evidence tiers and must never be inferred from offline benchmark success.
