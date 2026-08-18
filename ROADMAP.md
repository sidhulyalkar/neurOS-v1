# neurOS + ORION Roadmap

This roadmap describes the current architectural sequence and the evidence required to promote new capabilities. Historical plans and session reports live under `docs/archive/` and are not active roadmaps.

## Current refactor stack

The current kernel-to-research refactor is intentionally split into reviewable stacked pull requests. Merge bottom-up only after reviewing and validating each layer:

1. **PR #1: neurOS kernel and ORION contracts**
   - canonical `SignalFrame`, stream metadata, decoder outputs, queue policies, timing, config/plugin foundations, replay primitives, workspace packaging.
2. **PR #8: Runtime v3**
   - native `RuntimeGraph` executor, failure supervision, typed execution classes, p50/p95/p99 telemetry, single/multimodal execution convergence.
3. **PR #9: CLI v3**
   - config-first `doctor`, `plugins`, `devices`, `validate`, `run`, and benchmark execution with modular command ownership.
4. **PR #10: Recording v3**
   - lossless session archives, exact replay, source overrides, provenance, integrity verification, optional NWB/Zarr exports.
5. **PR #11: Quality v3**
   - version-controlled runtime quality gates, deterministic fault injection, benchmark manifests, known-signal scientific validity probes.
6. **PR #12: ORION tokenization v1**
   - seven tokenizer families, synthetic motif ground truth, fit/test separation, robustness benchmarks, tokenizer evidence reports.
7. **PR #13: Repository v3**
   - product/research/archive separation, historical cleanup, current documentation, and repository-hygiene enforcement.

The stack is complete only when the exact head of every PR has green CI and each PR remains reviewable relative to the branch immediately below it.

---

## North-star architecture

```text
physical neural system
        |
        v
hardware / sensors / datasets / replay
        |
        v
neuros Source plugins
        |
        v
SignalFrame + StreamDescriptor
        |
        v
RuntimeGraph
  acquisition
  synchronization
  processing
  fusion
  inference
  sinks
        |
        +---------------> recording / replay / quality evidence
        |
        v
ORION representation boundary
  tokenization
  neural encoders
  adaptation
  personalization
  interpretable state
        |
        v
application / closed-loop controller
```

neurOS should make neural computation operationally predictable. ORION should make neural representation and adaptation increasingly powerful without weakening runtime reliability.

---

# Post-stack Phase A: hardware qualification

## Goal

Move from simulated and software-level integration evidence to explicit device qualification without conflating hardware validation with generic CI.

## Work

- define a `HardwareQualificationManifest` containing manufacturer, device/model, firmware, driver/plugin version, OS, transport, channel layout, reference/montage, nominal sample rate, clock source, experiment config hash, Git SHA, and environment versions;
- implement deterministic qualification protocols for representative EEG and biosignal hardware;
- measure packet loss, reconnect behavior, clock drift, synchronization uncertainty, source-to-host latency, queue pressure, and sustained recording reliability;
- validate stop/restart/disconnect behavior and corrupted/partial-session recovery;
- publish machine-readable qualification reports rather than prose performance claims;
- maintain hardware-specific thresholds outside generic CI.

## Exit gates

A device is called **qualified** only when its exact hardware/firmware/software combination has a reproducible report and all mandatory runtime/recording checks pass.

---

# Post-stack Phase B: model artifacts and decoder safety

## Goal

Make model deployment reproducible, inspectable, and safe enough for long-lived BCI experiments.

## Work

- replace arbitrary trusted-environment pickle loading in production paths with backend-specific artifacts plus stable manifests;
- include architecture/input contract, training dataset hashes, subject/session scope, calibration state, metrics, Git SHA, package versions, and artifact SHA-256;
- add decoder compatibility checks against stream/representation schemas before runtime start;
- add probability/calibration capability tests and preserve `confidence=None` when uncertainty is unavailable;
- implement immutable model registry identifiers and explicit promotion stages such as experimental, validated, qualified;
- add rollback and replay-based regression tests for every promoted artifact.

## Exit gates

A promoted decoder must be reproducibly loadable, schema-compatible, replay-tested, and associated with a complete provenance manifest.

---

# Post-stack Phase C: real multimodal timing and fusion

## Goal

Turn explicit clock semantics into best-in-class multimodal BCI synchronization.

## Work

- extend the affine clock estimator to multiple hardware clock domains and intermittent synchronization observations;
- propagate synchronization uncertainty through fusion decisions;
- add fusion policies that can reject stale or temporally incompatible frames rather than blindly reuse the latest sample;
- benchmark synthetic known-offset/drift scenarios and recorded multimodal sessions;
- expose timing uncertainty to ORION so learned representations can distinguish neural uncertainty from biological variation.

## Exit gates

Fusion outputs must report which source frames contributed, their synchronized timestamps, and the uncertainty under which the fusion decision was made.

---

# Post-stack Phase D: ORION on real neural datasets

## Goal

Determine which neural tokenization and representation strategies survive contact with real data.

## Work

- add NWB spike-event adapters using the canonical replay/data contracts;
- evaluate tokenizers on multiple preparations, sessions, animals/subjects, and behavioral tasks;
- add held-out-unit prediction, next-window neural prediction, behavior/state decoding, cross-session transfer, few-shot adaptation, and compute-normalized metrics;
- evaluate robustness to jitter, unit dropout, sorting instability, rate shifts, and session drift;
- separate tokenizer fit data, representation-model training data, adaptation data, and held-out evaluation data;
- compare ORION tokenization against event, count, rate-summary, and randomized controls under matched model budgets.

## NeuroFM promotion rule

Existing `neuros-neurofm` implementations are research candidates, not automatically ORION components. A NeuroFM component moves behind an ORION interface only when it:

1. satisfies the interface and artifact contracts;
2. has leakage-controlled evidence;
3. improves a meaningful metric or offers a clear efficiency/interpretability advantage;
4. remains stable under perturbation and cross-session tests;
5. has reproducible compute and data manifests.

---

# Post-stack Phase E: adaptive and personalized BCI intelligence

## Goal

Make adaptation explicit, constrained, reversible, and scientifically measurable.

## Work

- implement adaptation proposal/review/apply lifecycle rather than hidden online mutation;
- distinguish calibration, unsupervised domain adaptation, supervised online learning, and user-specific personalization;
- maintain baseline and candidate model states with rollback;
- evaluate adaptation under simulated nonstationarity, electrode/channel degradation, and behavioral drift;
- log every adaptation trigger, training window, objective, resulting artifact, and before/after metrics;
- add mechanistic-interpretability hooks for explaining what state changed and which channels/tokens drove a decision.

## Exit gates

No adaptive component may silently alter a promoted decoder. Every update must be attributable, replayable, bounded, and reversible.

---

# Post-stack Phase F: closed-loop safety plane

## Goal

Make neurOS suitable as a research substrate for increasingly consequential closed-loop applications without pretending software tests constitute medical validation.

## Work

- introduce a first-class safety/constraint node type or equivalent policy layer;
- add action-rate limits, deadman states, confidence/quality gating, stale-data rejection, and emergency stop semantics;
- distinguish advisory decoder outputs from commands sent to an actuator;
- create hardware-in-the-loop simulation before physical closed-loop control;
- record proposed actions, accepted actions, rejected actions, reasons, neural quality, decoder state, and timing;
- develop task-specific hazard analyses and qualification profiles.

## Exit gates

A closed-loop demo must fail safe under source loss, runtime overload, decoder failure, stale synchronization, and explicit user/operator stop.

---

# Post-stack Phase G: developer ecosystem and productization

## Goal

Make the platform useful to researchers and BCI developers beyond this repository.

## Work

- stabilize versioned plugin APIs and compatibility policy;
- publish a small set of independently installable, well-tested device and decoder plugins;
- provide supported examples for motor imagery, replay analysis, multimodal fusion, and ORION tokenization;
- generate schema-aware configuration documentation and editor support;
- add package build/wheel/clean-install release gates;
- establish semantic versioning and deprecation periods for stable contracts;
- build UI surfaces from the same config/runtime/event APIs rather than separate orchestration logic.

## Exit gates

A new contributor should be able to install a standard profile, validate a config, run a mock experiment, record/replay it, inspect metrics, and add a plugin without modifying kernel code.

---

# Evidence hierarchy

Claims should always identify their evidence tier:

1. **Unit:** local function/class correctness.
2. **Contract:** implementation satisfies a stable neurOS/ORION interface.
3. **Integration:** multiple real packages execute together.
4. **Replay:** deterministic recorded session reproduces expected behavior.
5. **Scientific synthetic:** known ground truth is recovered under controlled perturbation.
6. **Dataset:** leakage-controlled real-data evaluation.
7. **Hardware qualification:** measured on named hardware/firmware/software.
8. **Closed-loop qualification:** hardware-in-the-loop or physical system safety/reliability testing.
9. **Clinical evidence:** separate regulated/clinical work, not implied by the software repository.

No lower tier should be described using language that implies a higher tier.

---

# Near-term priority after merging the stack

The highest-value next sequence is:

1. hardware qualification manifests and one real EEG device pipeline;
2. durable model-artifact loading and replay regression;
3. real NWB spike dataset adapter for ORION tokenization;
4. cross-session ORION tokenizer benchmark;
5. multimodal synchronization uncertainty and stale-frame-aware fusion;
6. adaptation lifecycle with rollback and provenance;
7. closed-loop safety policy layer.

The guiding question remains: **what are the smallest stable abstractions from which serious BCI systems can be built, measured, replayed, and improved?**
