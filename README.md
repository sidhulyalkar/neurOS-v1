# neurOS

**neurOS is an open execution and qualification layer for neural systems.**

Most neural-interface projects can train a model. Far fewer can answer, precisely and reproducibly:

- which neural observations were allowed to influence training, calibration, adaptation, or model selection;
- whether the deployed subject/session/device was genuinely unseen under the declared data and pretraining lineage;
- whether preprocessing or adaptation touched the final assessment set;
- whether a model artifact is exactly the state that was qualified;
- whether an improvement survives calibration cost, session shift, montage changes, artifacts, latency, and failure cases;
- what part of a claim is supported by software, real data, physical hardware, closed-loop evidence, or clinical work.

neurOS is being built to make those questions executable rather than rhetorical.

**ORION** is the complementary neural-intelligence layer. It explores tokenization, learned representations, transfer, personalization, and governed adaptation under the same evidence authority.

> **Current status:** active research and engineering platform. Passing CI establishes specific software contracts only. It does not imply biological correctness, model superiority, hardware qualification, online BCI efficacy, safety certification, or clinical benefit.

## The platform

The public architecture is intentionally smaller than the monorepo:

| Surface | Responsibility |
| --- | --- |
| **neurOS** | acquisition/runtime contracts, clocks, configuration, recording/replay, interoperability, deployment semantics |
| **ORION** | neural tokenization, representations, transfer, personalization, governed adaptation |
| **Evidence / NSQ** | frozen protocols, observation-role authority, scoring, model participation, artifacts, falsification, claim qualification |
| **Studio** | future inspection of runtime, evidence, representations, adaptation, quality, and provenance without creating a second runtime |

`neuros-arena` belongs to Evidence. It is a deterministic systems wind tunnel for finding failures across display, neural-world, device, transport, decoder, and application boundaries. Synthetic conformance is not human physiological validation.

```text
hardware / LSL / public data / replay
                 |
                 v
               neurOS
 SignalFrame -> RuntimeGraph -> recording / replay / quality
                 |
          +------+------+
          |             |
          v             v
        ORION       external methods
 representation    MNE / Braindecode /
 + adaptation      external plugins
          \             /
           +-----+-----+
                 v
                NSQ
 protocol + observation authority + scoring + artifact identity
                 |
                 v
        reproducible evidence
                 |
                 v
               Studio
```

## What is solid today

The repository has moved beyond a prototype architecture in several important ways:

- immutable canonical signal/frame and descriptor contracts;
- config-first native runtime graphs with bounded queue policy and latency/quality telemetry;
- deterministic recording, integrity verification, and replay;
- causal chunk-invariant streaming DSP;
- clean workspace wheel ownership and installed-wheel developer-preview testing;
- out-of-tree plugin authoring through Python entry points;
- Scientific Authority v2 for lineage, leakage, preprocessing, information roles, metrics, repeated measures, and failure preservation;
- Model Artifact v1 for content-addressed model promotion, reconstruction, rollback, and provenance;
- Neural System Qualification v1 plus an executable external-method runner;
- direct proving paths for canonical MNE/scikit-learn and upstream Braindecode participation;
- ORION adaptation/state-selection/final-assessment authority;
- deterministic Arena falsification and counterexample tooling.

The most important missing result is also clear: **the first frozen real-data NSQ study has not yet been completed.** ORION therefore does not currently claim a calibration or representation advantage.

See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for the strict maturity map.

## Quick start

This is a multi-distribution Python workspace. For the standard BCI development profile:

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python scripts/bootstrap.py --profile bci --test-tools
```

Exercise the installed platform:

```bash
neuros doctor --json
neuros plugins --json
neuros devices --json
neuros compatibility --json
neuros validate configs/examples/mock_bci.yaml --json
neuros run configs/examples/mock_bci.yaml --duration 2 --json
```

The checked-in smoke configuration is deliberately training-free. Installation checks should not hide model fitting behind the CLI.

## Record, replay, qualify

A live or deterministic source can be recorded through the canonical neurOS archive and replayed through the same runtime graph:

```bash
neuros record configs/examples/mock_bci.yaml \
  --output /tmp/session \
  --session-id demo \
  --duration 10

neuros inspect /tmp/session --verify --json

neuros replay /tmp/session \
  --config configs/examples/mock_bci.yaml \
  --json
```

The archive preserves sequence/timing/quality metadata, stream descriptors, configuration identity, Git/package provenance, runtime metrics, and per-frame integrity. NWB and Zarr are interoperability exports rather than substitutes for exact neurOS replay semantics.

Software qualification bundles can also be sealed and independently verified:

```bash
neuros qualify configs/examples/mock_bci.yaml \
  --output /tmp/qualification \
  --duration 1.0

neuros reproduce /tmp/qualification
```

A synthetic qualification bundle cannot self-award a physical hardware claim.

## Neural System Qualification

NSQ is becoming the peer-facing wedge of the project.

An external method may own its architecture, optimizer, and training implementation. neurOS owns the scientific boundary around it:

```text
frozen dataset lineage
      |
frozen observation roles
      |
external method factory
      |
exact learned-state binding
      |
trusted scorecard semantics
      |
preserved success / failure rows
      |
immutable qualification result
```

The next flagship study is tracked in [issue #82](https://github.com/sidhulyalkar/neurOS-v1/issues/82): a longitudinal motor-imagery comparison on MOABB Kumar2024 using MNE CSP+LDA and direct upstream Braindecode baselines under identical prospective calibration authority.

The key question is not merely “which model has the highest accuracy?” It is:

> **How much held-out neural utility does each method achieve as a function of per-user calibration cost?**

ORION should compete only after that external baseline floor is frozen.

See [`docs/NEURAL_SYSTEM_QUALIFICATION_V1.md`](docs/NEURAL_SYSTEM_QUALIFICATION_V1.md) and [`docs/NSQ_RUNNER_V1.md`](docs/NSQ_RUNNER_V1.md).

## ORION

ORION starts where provenance-rich neural data becomes a machine-native representation:

```text
SignalFrame(s)
    |
    v
NeuroTokenizer
    |
    v
NeuroTokenBatch
    |
    v
NeuralEncoder
    |
    v
RepresentationBatch
    |
    v
AdaptiveDecoder
    |
    v
DecoderOutput
```

Current tokenization research includes exact events, binned counts, relative-ISI WAIT/SPIKE representations, burst/pause/rebound tokens, synchrony packets, vector-quantized motifs, and population assemblies.

```bash
python scripts/orion/run_tokenizer_benchmark.py \
  configs/orion/tokenization_smoke.yaml \
  --output /tmp/orion-tokenization
```

Synthetic motif recovery is a falsification tool, not evidence that a tokenizer is better on real human neural data.

The long-term ORION hypothesis is explicit and testable:

> Preserve or improve held-out neural utility while materially reducing user-specific calibration, without sacrificing robustness, latency, provenance, uncertainty calibration, or representation stability.

Real longitudinal evidence decides whether that hypothesis survives.

## Interoperate instead of reimplement

neurOS should not become a replacement for mature neuroscience ecosystems. It should make their boundaries more reproducible and their claims more auditable.

Current evidence-backed integration lanes include BrainFlow, Lab Streaming Layer, MNE-Python, NWB/Zarr, MOABB, Braindecode, SNAP, and a narrow ngc-learn boundary.

Examples:

- BrainFlow / LSL for acquisition and transport rather than a giant proprietary driver catalog;
- MNE for neurophysiology analysis/preprocessing rather than a competing preprocessing universe;
- MOABB for public EEG access and benchmark data rather than a duplicate dataset registry;
- Braindecode for maintained neural architectures rather than copied model implementations;
- NWB/Zarr for interoperable export while neurOS preserves exact runtime replay semantics.

See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) and [`docs/NEUROAI_ECOSYSTEM.md`](docs/NEUROAI_ECOSYSTEM.md).

## Internal workspace map

The repository still contains multiple implementation distributions. They are not twelve equal product identities.

```text
packages/
  neuros-core/          runtime/data/replay/plugin contracts
  neuros/               public SDK + CLI composition
  neuros-drivers/       hardware, simulated, dataset and LSL sources
  orion/                ORION representation/adaptation contracts
  neuros-foundation/    external models/data + current NSQ/evidence implementation
  neuros-models/        supporting task decoders
  neuros-sourceweigher/ source/domain reliability research
  neuros-mechint/       intervention/faithfulness evidence research
  neuros-arena/         deterministic systems falsification
  neuros-neurofm/       experimental native foundation-model R&D
  neuros-ui/            experimental Studio substrate
  neuros-cloud/         experimental distributed/provider integrations
```

The root workspace is a packaging inventory, not proof that every member has equal maturity. Experimental packages should remain visibly experimental until they earn promotion.

## Scientific evidence hierarchy

A strong result names the evidence level it actually supports:

```text
software contract
      -> integration
      -> deterministic replay
      -> scientific synthetic
      -> real dataset
      -> physical hardware
      -> closed loop
      -> clinical evidence
```

A result in one tier does not silently promote another. Representation similarity is not task utility. Attribution is not mechanism. Synthetic conformance is not human performance. Hardware qualification is not closed-loop qualification. Closed-loop evidence is not clinical certification.

See [`docs/SCIENTIFIC_CLAIMS.md`](docs/SCIENTIFIC_CLAIMS.md).

## What happens next

The current execution order is intentionally narrow:

1. finish the first frozen real-data NSQ study (#82);
2. run ORION and competitive foundation/transfer methods against that exact authority;
3. add adaptive NSQ observation-role authority for unlabeled target adaptation (#81);
4. qualify one named physical EEG system end to end;
5. recruit independent users to reproduce the public qualification workflow;
6. resume the parked Arena v2 stack only against concrete real-data/hardware falsification targets (#83);
7. build Studio and stronger closed-loop safety only after the evidence plane earns the need.

See [`ROADMAP.md`](ROADMAP.md).

## Contributing

New subsystems should pass a build-vs-integrate test before becoming maintained code: identify the established ecosystem owner, explain why a thin adapter or upstream contribution is insufficient, state the unique neurOS authority being added, define an external falsification target, quantify maintenance cost, and name the condition under which the work should be removed or upstreamed.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`GOVERNANCE.md`](GOVERNANCE.md).

## License

MIT. See [`LICENSE`](LICENSE).
