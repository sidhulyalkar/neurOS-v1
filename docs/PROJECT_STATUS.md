# neurOS + ORION Project Status

This page is the current maturity map for the repository. It deliberately separates software contracts, execution provenance, scientific evidence, hardware evidence, closed-loop evidence, and clinical evidence.

> Passing CI proves only the contract exercised by that CI lane. It does not establish biological correctness, model superiority, physical hardware validity, online BCI efficacy, safety certification, or clinical benefit.

## Current product thesis

neurOS is converging on an **open qualification and reproducible execution layer for neural AI and BCI systems**.

- **neurOS** owns acquisition/runtime contracts, timing, configuration, recording/replay, interoperability, deployment semantics, and reproducible execution.
- **ORION** owns neural tokenization, learned representations, transfer, personalization, and governed adaptation.
- **Evidence / NSQ** owns frozen protocol identity, observation-role authority, model participation contracts, calibration accounting, scoring semantics, failure preservation, artifact identity, and claim qualification.
- **Studio** is the future inspection surface over those same contracts. It must not become a second runtime.

Arena is an Evidence subsystem for deterministic systems falsification. It is not a separate product and it is not a biological-truth authority.

The core product distinction is increasingly simple:

> neurOS should make a neural-system experiment executable without allowing infrastructure, adapters, retries, or analysis convenience to silently redefine the experiment.

## What has genuinely landed

### Runtime and reproducibility

The maintained runtime has canonical immutable `SignalFrame` / `StreamDescriptor` contracts, config-first `RuntimeGraph` execution, bounded queue policies, timing/quality telemetry, deterministic recording/replay, archive integrity, causal streaming DSP, descriptor-bound session archives, and an installed-wheel developer-preview path.

### Packaging and external extension

The workspace builds independent wheels with unique namespace ownership. The developer-preview journey installs built wheels in a clean environment and exercises the CLI. Out-of-tree plugins are discovered through Python entry points rather than monorepo-only imports.

`neuros init` provides a deterministic first-user project, and `neuros init --template nsq-method` provides an executable starter for bringing an external sklearn-style method into the production Neural System Qualification referee.

### Scientific Authority

Scientific Authority binds dataset/model lineage, processed-data identity, observation roles, preprocessing authority, target-information budgets, metric semantics, repeated-measures structure, and failure preservation. Leakage and pretraining overlap are explicit verdicts rather than implied cleanliness.

### Model Artifact

Promoted decoder artifacts are content-addressed, reconstructable through bounded trusted loaders, and bind input/output semantics, preprocessing/calibration provenance, learned state, rollback identity, and scientific lineage. Pickle remains a legacy trusted-local compatibility path, not the promoted artifact boundary.

### Neural System Qualification

NSQ provides a peer-facing external-method contract plus an executable runner. External implementations retain their own training code while neurOS controls which observations may cross the boundary, what those observations mean, how target calibration is counted, what fitted state was evaluated, how outputs are scored, and whether failures remain visible.

Maintained proving paths include canonical MNE/scikit-learn CSP+LDA, pyRiemann RG+LR, and upstream Braindecode EEGNet participation.

The Kumar2024 promoted study has now progressed beyond benchmark design into a sealed execution authority:

- 18 participants;
- 5 target sessions;
- 3 preregistered split seeds;
- 270 case authorities;
- 5 method realizations;
- 1,350 execution shards;
- calibration budgets `(0, 1, 2, 5, 10)` on every shard;
- 6,750 planned fit attempts.

The frozen method realizations are:

1. MNE CSP + LDA, deterministic;
2. pyRiemann RG + logistic regression, deterministic;
3. Braindecode EEGNet seed `31415`;
4. Braindecode EEGNet seed `384165836`;
5. Braindecode EEGNet seed `3991196546`.

A full 18-subject **no-model** binding has been generated and independently verified on exact `main`. That binding contains no decoder execution, no final-assessment predictions, and no final-assessment metrics.

The only currently authorized execution is one preselected classical shard as a **non-headline systems qualification**. Its numerical result is explicitly non-interpretable. The complete 1,350-shard fleet is not authorized yet.

Draft PR #110 adds FleetAuthority semantics for immutable shard leases, bounded infrastructure retries, verified worker-artifact ingestion, append-only attempt history, duplicate/conflict rejection, and complete-fleet assembly. It remains draft until the one-shard systems qualification is independently verified.

### ORION authority

ORION has contract-first tokenization, representation, adaptation, state-selection, and untouched-final-assessment semantics. Its strongest current result is **process integrity**, not a claim that ORION already reduces calibration or outperforms established representations.

ORION remains outside the promoted Kumar2024 comparison until the external floor has been completely executed, independently assembled, and scientifically interpreted.

## Package maturity

| Package | Role | Status now |
| --- | --- | --- |
| `neuros-core` | contracts, runtime, timing, recording/replay, config, plugins | **maintained core** |
| `neuros` | SDK, CLI, interoperability composition | **maintained public entry point** |
| `neuros-drivers` | hardware/dataset/LSL/BrainFlow source integrations | **maintained integration layer**; device claims remain per-device |
| `neuros-orion` (`packages/orion`) | tokenization, representations, adaptation authority | **active strategic layer**; real-data efficacy still to be earned |
| `neuros-foundation` | upstream model/data adapters, longitudinal evidence, current NSQ implementation | **maintained evidence/interoperability layer** |
| `neuros-models` | task decoders and inspectable model surfaces | **maintained supporting layer** |
| `neuros-sourceweigher` | source/domain reliability and transfer weighting | **research-supported**; must prove incremental value under NSQ |
| `neuros-mechint` | intervention/faithfulness/replication contracts | **research-supported**; empirical mechanism claims remain study-specific |
| `neuros-arena` | causal synthetic worlds and counterexamples | **maintained falsification tool**, secondary to real-data NSQ |
| `neuros-neurofm` | native foundation-model R&D | **experimental alpha**; not promoted ORION by default |
| `neuros-ui` | Studio prototypes | **experimental integration surface**; package metadata currently overstates maturity |
| `neuros-cloud` | distributed/provider integrations | **experimental integration surface**; package metadata currently overstates maturity |

The root workspace is a build inventory, not evidence that every member has equal product maturity.

## What is not established yet

1. **No flagship real-data NSQ efficacy result yet.** The study authority is frozen, but the promoted model fleet has not run.
2. **No demonstrated ORION calibration advantage yet.** ORION remains outside the comparison until the external floor exists.
3. **No named physical EEG device is publicly qualified end to end.** Simulator and driver conformance are not device qualification.
4. **No independent external reproduction cohort yet.** Internal CI cannot substitute for outside users reproducing a frozen public artifact.
5. **No production closed-loop safety plane yet.** Action constraints, stale-data rejection, deadman behavior, emergency-stop semantics, and hardware-in-the-loop evidence remain future work.
6. **No clinical evidence.** The project should continue to say this plainly.
7. **No public release train yet.** There is no coordinated package publication/release provenance contract suitable for a v1.0 claim.

## Repository debt to remove

The largest remaining weaknesses are now governance and release engineering rather than a need for another broad architectural rewrite:

- `main` is currently unprotected and has no repository ruleset;
- roughly thirty workflow files create duplicated CI orchestration and a large maintenance surface;
- several general workflows still reference mutable GitHub Action tags even though the promoted NSQ lane uses immutable action SHAs;
- central CI enumerates selected test files explicitly, so new tests need deliberate lane ownership;
- historical agent/feature branches substantially outnumber active workstreams;
- GitHub repository metadata still understates the current qualification/evidence product thesis and exposes no search topics;
- `neuros-ui` / `neuros-cloud` advertise beta/2.0-style maturity without corresponding release qualification;
- some research packages intentionally contain incomplete surfaces and must remain visually distinct from promoted contracts;
- GitHub Actions artifacts are useful execution evidence but 90-day artifact retention is not a long-lived scientific archive;
- signed provenance / attestations and a durable content-addressed result store remain future hardening layers.

## Immediate execution gate

The current promoted Kumar2024 sequence is intentionally narrow:

```text
exact-main no-model binding
        |
        v
independent cryptographic + semantic verification
        |
        v
one preselected MNE CSP+LDA shard
(non-headline systems qualification;
 numerical result quarantined)
        |
        v
FleetAuthority + trusted attempt transport
        |
        v
complete 1,350-shard external-floor execution
        |
        v
independent complete-fleet assembly
        |
        v
scientific interpretation of external floor
        |
        v
only then may ORION enter a comparison
```

The flagship scientific question remains:

> Under identical prospective longitudinal authority, how much held-out task utility does each method achieve as a function of per-user labeled calibration cost?

The key difference from an ordinary benchmark is that the scheduler, retry system, environment, model adapter, and final assembler are not allowed to acquire hidden authority over that question.

## Promotion rule

New work should enter at the weakest accurate level and earn promotion:

```text
experiment
  -> research implementation
  -> stable contract
  -> integration/replay evidence
  -> scientific synthetic evidence
  -> real-dataset execution evidence
  -> independent reproduction
  -> hardware qualification
  -> closed-loop qualification
  -> clinical evidence (separate regulated work)
```

The repository no longer needs a broad rewrite. It needs **fewer simultaneous workstreams, stronger public governance, independent users, and a flagship result whose evidence boundary is difficult to reproduce without neurOS**.
