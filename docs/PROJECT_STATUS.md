# neurOS + ORION Project Status

This page is the current capability and maturity map for the monorepo. It is intentionally stricter than a feature list: a package can contain useful code without being a qualified production surface.

> **Evidence boundary:** software maturity, package version, and passing CI do not imply hardware qualification, clinical validation, biological correctness, or safety certification.

## Platform shape

neurOS is organized around execution, intelligence, and evidence, with Studio as the inspection surface:

```text
neural hardware / datasets / replay
              |
              v
      neurOS runtime plane
 contracts | timing | graph execution | recording | quality
              |
              +-----------------------------+
              |                             |
              v                             v
       model / transfer plane              ORION
 task decoders | foundation adapters   tokenization | representations
 mechanism analysis | source trust     adaptation | personalization
              |                             |
              +--------------+--------------+
                             |
                             v
                       Evidence plane
        frozen protocols | Arena | qualification | claims
                             |
                             v
                           Studio
```

The runtime plane should remain small and conservative. Research and model layers may move faster, but they must cross explicit contracts before they become deployment dependencies. Arena belongs to Evidence: it is a deterministic adversarial systems laboratory, not a biological-truth authority.

## Package maturity map

| Surface | Role | Current evidence | Intended use now |
| --- | --- | --- | --- |
| `neuros-core` | data contracts, runtime graph, timing, recording/replay, config, quality, plugin semantics | kernel contract matrix on Python 3.10-3.12; replay and recording tests | maintained kernel and integration substrate |
| `neuros` | user-facing SDK and CLI composition | installed BCI/config/CLI/record/replay smoke execution | primary developer entry point |
| `neuros-drivers` | hardware, simulated, and dataset sources | BCI profile, plugin/config tests, dedicated driver contracts; device-specific qualification remains separate | maintained integration surface, not blanket hardware qualification |
| `neuros-arena` | causal synthetic BCI worlds, display/device/transport faults, populations, counterexamples, public-data anchoring utilities | dedicated deterministic Arena suite, scientific-validation policy, workspace-wheel/release-candidate qualification | maintained scientific-synthetic systems laboratory; not human physiological validation |
| `neuros-models` | task-specific classical and neural decoders | v2.1 model/mech-int contract job; faithful PyTorch model identity; analysis-manifest regression tests | maintained decoder layer; model artifacts still need stronger deployment serialization |
| `neuros-foundation` | neural foundation-model catalog, adapters, probes, benchmark protocols | package regressions plus dependency-light examples in monorepo CI | maintained interoperability/evaluation layer; upstream-model claims remain provenance-sensitive |
| `neuros-sourceweigher` | reliability-aware source/domain weighting and fusion | regression matrix on Python 3.10-3.12 plus dependency-light examples | maintained research/deployment-support component for transfer and declared domain similarity |
| `neuros-orion` (`packages/orion`) | tokenization, representation, adaptation, and final-assessment authority | ORION contracts, controlled tokenizer benchmarks, leakage/adaptation/final-assessment authority workflows | active neural-intelligence layer with explicit promotion gates |
| `neuros-mechint` | causal mechanism experiments, evidence artifacts, replication contracts | v1 software-contract gates, Python 3.10-3.12, executed CPU tutorials, ecosystem import checks | mature research software contract; empirical neuroscience evidence remains study-specific and incomplete by default |
| `neuros-neurofm` | native neural foundation-model R&D | alpha research package and ORION/mech-int integration tests | experimental model research, not a promoted ORION implementation by default |
| `neuros-ui` | dashboard/API/visualization surfaces | package metadata exists, but it is not yet a release-blocking qualification lane | integration/prototyping surface until dedicated product tests exist |
| `neuros-cloud` | distributed/cloud/export/monitoring integrations | package metadata exists, but it is not yet a release-blocking qualification lane | optional integration surface until provider-specific tests and release gates exist |

The repository directory is `packages/orion`, while the installable distribution is `neuros-orion`. The root `pyproject.toml` workspace is the canonical maintained-distribution inventory. CI, release-candidate wheel builds, and repository hygiene derive package membership from that authority rather than keeping parallel package lists.

## What is genuinely usable today

### 1. Build and execute a reproducible software BCI pipeline

```bash
python scripts/bootstrap.py --profile bci --test-tools
neuros doctor --json
neuros validate configs/examples/mock_bci.yaml --json
neuros run configs/examples/mock_bci.yaml --duration 2 --json
```

The config path resolves plugins into the same native `RuntimeGraph` used by the Python pipeline facades.

### 2. Record, verify, and replay the same neural contract

```bash
neuros record configs/examples/mock_bci.yaml --output /tmp/session --session-id demo --duration 2
neuros inspect /tmp/session --verify --json
neuros replay /tmp/session --config configs/examples/mock_bci.yaml --json
```

The canonical session archive preserves neurOS timing, quality, sequence, provenance, and integrity semantics. NWB and Zarr are interoperability exports rather than the authoritative replay format.

### 3. Stress a BCI system in the deterministic Arena

Arena separates requested stimulus, actually emitted display history, neural-world dynamics, sensor/device effects, transport faults, decoder behavior, and application authority. Built-in presets provide reproducible smoke and torture worlds, while portable manifests preserve complete world identity.

```bash
neuros-arena --preset dual-target-smoke --output arena-report.json
```

World models are extensible through the `neuros.world_models` entry-point group. The default driven state-space model is phenomenological. Semi-synthetic and lead-field-backed modes strengthen specific simulation assumptions without turning synthetic output into evidence about human prevalence or clinical performance.

### 4. Compare task-specific neural decoders without changing their algorithm identity

```bash
neuros-models list
neuros-models list --mechint-ready
neuros-models show eeg-conformer
```

The maintained deep decoders expose logits, embeddings, stable model identity, and explicit mechanistic-analysis surfaces. Missing optional backends fail closed instead of silently swapping in a different algorithm.

### 5. Inspect neural representations and foundation-model integrations

`neuros-foundation` separates catalog metadata from locally runnable adapters. Its probes and protocol fingerprints are intended to make cross-model comparisons explicit about subject/session/site/device split semantics and representation preprocessing.

### 6. Estimate which sources should be trusted for a target domain

`neuros-sourceweigher` provides constrained source mixtures, distribution-distance baselines, online drift adaptation, representation-space weighting, and neurOS runtime fusion without forcing the HTTP service into the core install.

### 7. Run causal mechanism studies without confusing tooling with evidence

`neuros-mechint` provides causal intervention, faithfulness, held-out evidence, replication, and artifact contracts. Its release status deliberately separates **software contract readiness** from **empirical evidence completion**.

### 8. Benchmark tokenization and auditable adaptation behind ORION contracts

ORION compares tokenizer families under controlled synthetic motifs and exposes separate authority for calibration, qualification/model selection, and untouched final assessment. Synthetic known-ground-truth tests remain falsification tools. Real-data promotion requires deployment-unit-disjoint evidence and frozen assessment identity.

## Important gaps before neurOS should be marketed as a broadly deployable BCI platform

### External plugin authoring journey

The kernel has a real entry-point registry, but the developer preview still needs a maintained out-of-tree plugin template, clean-wheel install test, source/transform examples, and explicit compatibility/version negotiation. This is the next major usability gate because extensibility must work without editing the monorepo.

### Hardware qualification

There is not yet a public qualification matrix showing reproducible packet loss, drift, synchronization uncertainty, reconnect behavior, sustained recording reliability, and latency for named hardware/firmware/software combinations.

### Durable model artifacts

The model layer still needs a production artifact format based on explicit architecture/configuration, `state_dict` or backend-native weights, input schema, calibration state, training/evaluation provenance, and immutable hashes. Trusted Python pickle should not be the long-term deployment boundary.

### Real-data ORION and Arena anchoring evidence

Synthetic known-ground-truth tests are valuable falsification tools, but ORION needs leakage-controlled multi-session real neural datasets before tokenizer or representation superiority claims should be made. Arena likewise needs held-out public-subject anchoring studies to characterize where its declared synthetic envelope resembles or misses real data. Similarity weighting is not a truth probability.

### Cross-subject/session/device mechanism stability

Mechanistic interpretability becomes strategically valuable for BCI only when candidate mechanisms remain predictive and causal across the deployment units that matter. Subject/session/montage/device robustness should be measured as a first-class benchmark dimension.

### Closed-loop safety plane

The runtime needs explicit action constraints, deadman behavior, stale-data rejection, quality/confidence gates, rate limits, emergency stop semantics, and hardware-in-the-loop qualification before consequential closed-loop control is treated as a product capability.

### UI/cloud qualification

`neuros-ui` and `neuros-cloud` should either gain dedicated contract/release tests and supported reference deployments or be labeled more explicitly as optional integration packages. Public package version numbers alone should not imply equal maturity with the kernel.

## Contribution and promotion rule

New work should enter at the weakest accurate maturity level:

```text
experiment
   -> research package
   -> stable contract adapter
   -> integration/replay evidence
   -> scientific-synthetic evidence
   -> real-dataset evidence
   -> hardware qualification
   -> closed-loop qualification
```

Promotion should require evidence, not enthusiasm. A sophisticated method that cannot be replayed, versioned, falsified, or compared fairly is still research debt.

## Near-term release objective

The next coherent public milestone should be a **developer preview** in which an external contributor can:

1. install from a clean environment and verify exact built wheels;
2. run and inspect a mock pipeline;
3. record and replay it;
4. run a deterministic Arena world and understand its evidence boundary;
5. select an inspectable decoder;
6. compare or adapt representations through the foundation/SourceWeigher layers;
7. run an ORION tokenizer/adaptation study under frozen authority;
8. add an external plugin without modifying kernel code;
9. understand exactly which claims are supported by software, synthetic, dataset, hardware, or closed-loop evidence.

That milestone is more valuable than simply adding more algorithms. It turns neurOS from a large repository into a legible, extensible platform.
