# neurOS + ORION Project Status

This page is the current maturity map for the repository. It intentionally distinguishes software contracts from scientific, hardware, closed-loop, and clinical evidence.

> Passing CI means a declared software contract passed its tests. It does not establish biological correctness, model superiority, hardware qualification, online BCI efficacy, safety certification, or clinical benefit.

## Current product thesis

neurOS is converging on an **open execution and qualification layer for neural systems**.

- **neurOS** owns acquisition/runtime contracts, timing, configuration, recording/replay, interoperability, deployment semantics, and reproducible execution.
- **ORION** owns neural tokenization, learned representations, transfer, personalization, and governed adaptation.
- **Evidence / NSQ** owns frozen protocol identity, observation-role authority, model participation contracts, scoring semantics, failure preservation, artifact identity, and claim qualification.
- **Studio** is the future inspection surface over those same contracts. It must not become a second runtime.

Arena is an Evidence subsystem for deterministic systems falsification. It is not a separate product and it is not a biological-truth authority.

## What has genuinely landed

### Runtime and reproducibility

The maintained runtime now has canonical immutable `SignalFrame`/`StreamDescriptor` contracts, config-first `RuntimeGraph` execution, bounded queue policies, timing/quality telemetry, deterministic recording/replay, archive integrity, causal streaming DSP, and descriptor-bound session archives.

### Packaging and external extension

The workspace builds independent wheels with unique namespace ownership. The developer-preview journey installs built wheels in a clean environment and exercises the CLI. An out-of-tree example plugin is built and executed as its own wheel through public entry points rather than relying on monorepo imports.

### Scientific Authority

Scientific Authority v2 binds dataset/model lineage, processed-data identity, observation roles, preprocessing authority, target-information budgets, metric semantics, repeated-measures structure, and failure preservation. Leakage and pretraining overlap are explicit verdicts rather than implied cleanliness.

### Model Artifact v1

Promoted decoder artifacts are content-addressed, reconstructable through bounded trusted loaders, and bind input/output semantics, preprocessing/calibration provenance, learned state, rollback identity, and scientific lineage. Pickle is not the promoted artifact boundary.

### Neural System Qualification

NSQ v1 now provides a peer-facing external-method contract plus an executable runner. External implementations can retain their own training code while neurOS controls which observations may cross the boundary, what those observations mean, how outputs are scored, and whether failures remain visible.

The production runner already has direct proving paths for canonical MNE/scikit-learn and upstream Braindecode participation. The next milestone is not another runner abstraction. It is the first frozen real-data qualification study.

### ORION authority

ORION has contract-first tokenization, representation, adaptation, state-selection, and untouched-final-assessment semantics. Its current strongest result is **process integrity**, not a claim that ORION already reduces calibration or outperforms established representations.

## Package maturity

| Package | Role | Status now |
| --- | --- | --- |
| `neuros-core` | contracts, runtime, timing, recording/replay, config, plugins | **maintained core** |
| `neuros` | SDK, CLI, interoperability composition | **maintained public entry point** |
| `neuros-drivers` | hardware/dataset/LSL/BrainFlow source integrations | **maintained integration layer**; device claims remain per-device |
| `neuros-orion` (`packages/orion`) | tokenization, representations, adaptation authority | **active strategic layer**; real-data efficacy still to be earned |
| `neuros-foundation` | upstream model/data adapters, longitudinal evidence, current NSQ implementation | **maintained evidence/interoperability layer**; namespace should be revisited only after NSQ proves broader use |
| `neuros-models` | task decoders and inspectable model surfaces | **maintained supporting layer** |
| `neuros-sourceweigher` | source/domain reliability and transfer weighting | **research-supported**; must prove incremental value under NSQ |
| `neuros-mechint` | intervention/faithfulness/replication contracts | **research-supported**; empirical mechanism claims remain study-specific |
| `neuros-arena` | causal synthetic worlds and counterexamples | **maintained falsification tool**, currently secondary to real-data NSQ |
| `neuros-neurofm` | native foundation-model R&D | **experimental alpha**; not promoted ORION by default |
| `neuros-ui` | Studio prototypes | **experimental integration surface**; package metadata currently overstates maturity |
| `neuros-cloud` | distributed/provider integrations | **experimental integration surface**; package metadata currently overstates maturity |

The root workspace is a build inventory, not evidence that every member has equal product maturity.

## What is not established yet

The repository still lacks the evidence needed for several strategically important claims:

1. **No flagship real-data NSQ result yet.** The Kumar2024 longitudinal motor-imagery study in issue #82 is the immediate scientific gate.
2. **No demonstrated ORION calibration advantage yet.** ORION should enter comparison only after strong external baselines are frozen under identical authority.
3. **No named physical EEG device is publicly qualified end to end.** Simulator and driver conformance are not device qualification.
4. **No independent external reproduction cohort yet.** Internal CI cannot substitute for outside users reproducing a frozen result from public artifacts.
5. **No production closed-loop safety plane yet.** Action constraints, stale-data rejection, deadman behavior, emergency-stop semantics, and hardware-in-the-loop evidence remain future work.
6. **No clinical evidence.** The project should continue to say this plainly.

## Repository debt to remove

The remaining cleanup is mostly organizational rather than architectural:

- roughly thirty GitHub workflow files create duplicated CI orchestration and a large maintenance surface;
- central CI still enumerates selected test files explicitly, so new tests need deliberate lane coverage rather than benefiting from universal discovery;
- historical agent/feature branches substantially outnumber active workstreams and should be deleted after unique research is rescued or explicitly archived;
- current docs have drifted behind recently merged Scientific Authority, Model Artifact, and NSQ work;
- `neuros-ui` / `neuros-cloud` advertise beta/2.0-style package maturity without corresponding release qualification;
- some experimental/research packages still contain TODO or intentionally incomplete surfaces and must remain visually distinct from promoted contracts;
- repository-level branch protection/rulesets should match the exact-head qualification discipline enforced in code.

## Immediate execution gate

The next coherent milestone is **NSQ Kumar2024 v1** (#82):

```text
frozen MOABB Kumar2024 lineage + preprocessing
        |
        v
prospective participant/session authority
        |
        +-> MNE CSP + LDA
        +-> upstream Braindecode EEGNet
        +-> upstream EEGConformer when supported
        |
        v
participant-level calibration frontier
        |
        v
immutable qualification bundle
```

The flagship question is:

> Under identical prospective longitudinal authority, how much held-out task utility does each method achieve as a function of per-user labeled calibration cost?

Only after that baseline is frozen should ORION, foundation representations, SourceWeigher, or governed adaptation compete for an improvement claim.

## Promotion rule

New work should enter at the weakest accurate level and earn promotion:

```text
experiment
  -> research implementation
  -> stable contract
  -> integration/replay evidence
  -> scientific synthetic evidence
  -> real-dataset evidence
  -> hardware qualification
  -> closed-loop qualification
  -> clinical evidence (separate regulated work)
```

The repository no longer needs a broad rewrite. It needs **fewer simultaneous workstreams, stronger external falsification, and a public result that makes the qualification layer indispensable**.
