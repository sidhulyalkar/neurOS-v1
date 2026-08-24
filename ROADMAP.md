# neurOS + ORION Roadmap

This roadmap describes the active qualification and productization sequence for neurOS and ORION. Historical plans and session reports live under `docs/archive/` and are not active roadmaps. For the package-by-package maturity snapshot, see [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Completed convergence foundation

The 2026 architecture/refactor sequence is now merged into `main`:

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
8. **PR #14: neuros-mechint v1**
   - reproducible causal-evidence contracts, artifact schemas, held-out evidence, replication and explicit software-vs-empirical release status.
9. **PR #16: neural foundation-model interoperability**
   - capability registry, fail-closed adapters, representation probes, protocol fingerprints, and fair cross-model benchmark surfaces.
10. **PR #17: SourceWeigher reliability engine**
    - constrained source mixtures, distribution/reliability methods, online drift adaptation, diagnostics, and runtime fusion.
11. **PR #18: mechanistically inspectable task models**
    - faithful PyTorch decoder implementations, analysis manifests, model embeddings, neurOS-to-mech-int adapter, and dedicated model/mech-int CI.

The repository therefore no longer needs another broad architectural rewrite. The next phase is **qualification, release discipline, evidence on real data/hardware, and a sharper public developer experience**.

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
 acquisition | synchronization | transforms | fusion | inference | sinks
        |
        +---------------------> recording / replay / quality evidence
        |
        +---------------------> task models / embeddings
        |                           |
        |                           +-> foundation-model interoperability
        |                           +-> SourceWeigher transfer reliability
        |                           +-> mech-int causal evidence
        |
        v
ORION representation boundary
 tokenization | neural encoders | adaptation | personalization
        |
        v
application / closed-loop controller
```

neurOS should make neural computation operationally predictable. ORION should make neural representation and adaptation increasingly powerful without weakening runtime reliability.

---

# Priority 0: public developer preview and release discipline

## Goal

Turn the monorepo from a sophisticated internal platform into something an external BCI researcher or engineer can install, understand, test, extend, and trust within one session.

## Work

- keep `docs/PROJECT_STATUS.md` as the canonical package maturity map;
- add clean-install build/wheel checks for every package promoted as maintained;
- verify local workspace installs separately from published-package installs;
- define semantic-versioning and deprecation policy for stable contracts;
- publish a package compatibility matrix for `neuros-core`, `neuros`, `neuros-models`, `neuros-foundation`, `neuros-sourceweigher`, `neuros-mechint`, and `neuros-orion`;
- add a generated or tested configuration/schema reference;
- provide one supported end-to-end developer journey covering install -> validate -> run -> record -> replay -> inspect;
- provide one plugin-author journey that creates an external source/transform/decoder without kernel edits;
- add dedicated qualification lanes before `neuros-ui` or `neuros-cloud` are presented as equally mature with the kernel.

## Exit gates

A clean external environment can install the documented profile, run the supported examples, build the maintained distributions, and reproduce the same contract checks without relying on an editable monorepo checkout.

---

# Priority 1: hardware qualification

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

# Priority 2: durable model artifacts and decoder deployment safety

## Goal

Make model deployment reproducible, inspectable, rollbackable, and safe enough for long-lived BCI experiments.

## Work

- replace trusted-environment pickle persistence in promoted paths with backend-specific artifacts plus stable manifests;
- store architecture/configuration, input contract, `state_dict` or backend-native weights, preprocessing, training dataset hashes, subject/session scope, calibration state, metrics, Git SHA, package versions, and artifact SHA-256;
- bind the `InterpretabilityManifest` fingerprint to the artifact so layer/path changes cannot silently invalidate mechanistic evidence;
- add decoder compatibility checks against stream/representation schemas before runtime start;
- add probability/calibration capability tests and preserve `confidence=None` when uncertainty is unavailable;
- implement immutable model registry identifiers and explicit stages such as experimental, validated, and qualified;
- add rollback and replay-based regression tests for every promoted artifact.

## Exit gates

A promoted decoder is reproducibly loadable, schema-compatible, replay-tested, mechanism-manifest-compatible where applicable, and associated with a complete provenance manifest.

---

# Priority 3: ORION and model benchmarks on real neural datasets

## Goal

Determine which tokenization, representation, transfer, and mechanistic ideas survive contact with real deployment-unit variation.

## Work

- add NWB/data adapters using the canonical replay/data contracts;
- evaluate across multiple preparations, sessions, animals/subjects, sites, devices/montages, and behavioral tasks where scientifically appropriate;
- benchmark task decoders and foundation representations under the same subject/session-disjoint protocols;
- include held-out neural prediction, behavior/state decoding, cross-session transfer, few-shot calibration, and compute-normalized metrics;
- evaluate robustness to jitter, unit/channel dropout, sorting instability, rate shifts, artifacts, montage changes, and session drift;
- keep tokenizer fit data, representation-model training data, adaptation data, mechanistic-discovery data, and final held-out evaluation data distinct;
- compare ORION tokenization against event, count, rate-summary, randomized, and architecture-matched controls;
- integrate SourceWeigher as an explicitly evaluated transfer strategy rather than an unconditional preprocessing step;
- test whether candidate mechanisms remain causal and stable across subjects/sessions/devices instead of only within one trained model distribution.

## NeuroFM promotion rule

Existing `neuros-neurofm` implementations are research candidates, not automatically ORION components. A NeuroFM component moves behind a promoted ORION interface only when it:

1. satisfies interface and artifact contracts;
2. has leakage-controlled evidence;
3. improves a meaningful metric or offers a clear efficiency/interpretability advantage;
4. remains stable under perturbation and cross-session tests;
5. has reproducible compute/data manifests;
6. is compared against simpler baselines under matched budgets.

---

# Priority 4: multimodal timing and reliability-aware fusion

## Goal

Turn explicit clock and source-reliability semantics into best-in-class multimodal BCI synchronization and fusion.

## Work

- extend the affine clock estimator to multiple hardware clock domains and intermittent synchronization observations;
- propagate synchronization uncertainty through fusion decisions;
- add fusion policies that reject stale or temporally incompatible frames rather than blindly reuse the latest sample;
- attach contributing-frame provenance and timing uncertainty to fusion outputs;
- combine timing/quality evidence with SourceWeigher reliability without conflating domain similarity, task utility, signal quality, and predictive uncertainty;
- benchmark synthetic known-offset/drift scenarios and recorded multimodal sessions;
- expose timing/reliability uncertainty to ORION so learned representations can distinguish system uncertainty from biological variation.

## Exit gates

Fusion outputs report contributing source frames, synchronized timestamps, quality/reliability evidence, and the uncertainty under which the fusion decision was made.

---

# Priority 5: adaptive and personalized BCI intelligence

## Goal

Make adaptation explicit, constrained, reversible, and scientifically measurable.

## Work

- implement adaptation proposal/review/apply lifecycle rather than hidden online mutation;
- distinguish calibration, unsupervised domain adaptation, supervised online learning, and user-specific personalization;
- maintain baseline and candidate model states with rollback;
- evaluate adaptation under simulated and real nonstationarity, electrode/channel degradation, and behavioral drift;
- log every adaptation trigger, training window, objective, resulting artifact, and before/after metrics;
- use mechanistic evidence as an additional stability signal only after its predictive value is validated;
- measure calibration cost and time-to-usable-control as first-class user-facing metrics.

## Exit gates

No adaptive component silently alters a promoted decoder. Every update is attributable, replayable, bounded, reversible, and evaluated against a no-adaptation baseline.

---

# Priority 6: closed-loop safety plane

## Goal

Make neurOS suitable as a research substrate for increasingly consequential closed-loop applications without pretending software tests constitute medical validation.

## Work

- introduce a first-class safety/constraint node type or equivalent policy layer;
- add action-rate limits, deadman states, confidence/quality gating, stale-data rejection, and emergency-stop semantics;
- distinguish advisory decoder outputs from commands sent to an actuator;
- create hardware-in-the-loop simulation before physical closed-loop control;
- record proposed actions, accepted actions, rejected actions, reasons, neural quality, decoder state, and timing;
- develop task-specific hazard analyses and qualification profiles.

## Exit gates

A closed-loop demo fails safe under source loss, runtime overload, decoder failure, stale synchronization, policy rejection, and explicit user/operator stop.

---

# Priority 7: ecosystem and productization

## Goal

Make the platform valuable to researchers, neurotechnology teams, and BCI developers beyond this repository while preserving an open, inspectable core.

## Work

- stabilize versioned plugin APIs and compatibility policy;
- publish independently installable, well-tested device and decoder plugins;
- provide supported examples for motor imagery, replay analysis, multimodal fusion, model/mechanism auditing, foundation representations, SourceWeigher transfer, and ORION tokenization;
- build UI surfaces from the same config/runtime/event APIs rather than separate orchestration logic;
- define a remote experiment/telemetry control plane without moving core real-time execution into the cloud;
- create machine-readable qualification and benchmark artifacts suitable for CI, papers, and enterprise audit trails;
- keep optional commercial integrations above open kernel contracts rather than forking the architecture.

## Exit gates

A new contributor can add a plugin without kernel changes, and a team can reproduce the same experiment/qualification artifact on another machine or site with explicit compatibility/provenance information.

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

# Immediate execution sequence

The highest-value next sequence is:

1. finish the public developer-preview/release gates;
2. define the model artifact v1 format and remove pickle from promoted deployment paths;
3. qualify one real EEG device end to end using a machine-readable hardware manifest;
4. add one real public neural dataset to the ORION/model benchmark harness;
5. run subject/session-disjoint decoder + foundation + SourceWeigher + mechanism-stability comparisons;
6. add uncertainty-aware multimodal fusion;
7. build adaptation lifecycle with rollback and provenance;
8. add a closed-loop safety policy layer before more consequential demos.

The guiding question remains: **what are the smallest stable abstractions from which serious BCI systems can be built, measured, replayed, falsified, and improved?**
