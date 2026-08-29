# neurOS

**The open qualification and reproducible execution layer for neural AI and BCI systems.**

Your model can train. Can you prove exactly **what data it saw, what calibration it consumed, what state was evaluated, what failed, and what claim the result actually supports?**

neurOS turns those questions into executable contracts and content-addressed evidence.

Bring MNE, MOABB, Braindecode, BrainFlow/LSL, a lab model, or your own Python implementation. neurOS is not trying to replace those ecosystems. It adds the missing layer around them: **runtime identity, exact replay, observation-role authority, leakage-controlled qualification, model-state provenance, calibration accounting, failure preservation, and evidence grading.**

**ORION** is the optional neural-intelligence plane on top: tokenization, representations, transfer, personalization, and governed adaptation tested under the same authority rather than a privileged benchmark path.

> **Status:** active research and engineering developer preview. Current CI proves named software contracts, not biological correctness, model superiority, physical hardware validity, online BCI efficacy, safety certification, or clinical benefit. The first frozen full real-data NSQ comparison is still being qualified.

## Get useful evidence in minutes

Until coordinated public package publishing is qualified, install from the repository:

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python scripts/bootstrap.py --profile bci --test-tools
```

Create a clean starter project:

```bash
neuros init my-neuros-project
cd my-neuros-project

neuros doctor
neuros validate neuros.yaml
neuros run neuros.yaml --duration 2
neuros qualify neuros.yaml --output evidence/qualification --duration 1
neuros reproduce evidence/qualification
```

The starter uses a deterministic mock stream so installation, runtime, replay, and provenance failures cannot hide behind model fitting. Its qualification bundle is **software/runtime evidence only**.

See [First 10 Minutes](docs/getting-started/first-10-minutes.md).

## Bring your own model

You do not subclass a neurOS neural network or adopt a neurOS training loop.

An external model participates through a deliberately small contract:

```text
ExternalDecoderMethodSpec
        |
factory.create() -> fresh decoder
        |
        +-- fit(authorized X, y)
        +-- predict(untouched X_final)
        +-- optional predict_proba(X_final)
        +-- learned_state()
        |
        v
       NSQ
 observation roles + calibration budget + score semantics
 + provenance + failures + immutable result identity
```

That means an sklearn pipeline, upstream Braindecode model, private lab decoder, or pretrained representation can be evaluated by the same referee without moving its training implementation into this repository.

Start with [Bring Your Own Model](docs/getting-started/bring-your-own-model.md) and [NSQ Runner v1](docs/NSQ_RUNNER_V1.md).

## The missing ecosystem layer

| Existing ecosystem | What it already does well | What neurOS adds |
| --- | --- | --- |
| **MNE-Python** | neurophysiology preprocessing, analysis, visualization | exact execution/replay and claim-bound provenance around the pipeline |
| **MOABB / EEG data ecosystems** | public EEG datasets and benchmark plumbing | frozen observation roles, calibration authority, failure-preserving system qualification |
| **Braindecode / model libraries** | maintained neural architectures and training tools | external participation under identical data, calibration, state-selection, and scoring authority |
| **BrainFlow / LSL / device stacks** | acquisition and transport | descriptor/timing provenance, replay, measured hardware qualification, evidence boundaries |
| **NWB / BIDS / Zarr** | interoperable data organization/storage | runtime-to-artifact identity and exact execution provenance rather than a competing file format |
| **Your lab code** | the model or experiment you actually care about | a reproducible qualification envelope without forcing a framework rewrite |

The intended value proposition is simple:

> **Use neurOS when the hard problem is no longer “can I run a model?” but “can another researcher audit exactly what this neural-system claim means and reproduce the evidence boundary?”**

## Neural System Qualification

NSQ is the peer-facing wedge of neurOS.

A frozen qualification binds:

```text
upstream dataset / revision
        |
processed-data authority
        |
source / calibration / adaptation / final observation roles
        |
external method identity
        |
fresh model execution at each authorized budget
        |
learned-state identity
        |
frozen score semantics
        |
success + failure rows
        |
content-addressed evidence artifact
```

The flagship question is not just “which decoder has the highest pooled accuracy?”

> **How much held-out neural utility does a method achieve as a function of user-specific calibration cost, under exactly the same prospective authority?**

The current promoted Kumar2024 work is deliberately freezing external classical and Braindecode baselines before ORION is allowed into a superiority comparison. See [issue #82](https://github.com/sidhulyalkar/neurOS-v1/issues/82).

## ORION

ORION is the intelligence plane, not a shortcut around qualification:

```text
neural observations
      |
NeuroTokenizer
      |
NeuroTokenBatch
      |
NeuralEncoder
      |
RepresentationBatch
      |
AdaptiveDecoder
      |
DecoderOutput
```

Its long-term hypothesis is falsifiable:

> Preserve or improve held-out neural utility while materially reducing user-specific calibration, without sacrificing robustness, latency, provenance, uncertainty calibration, or representation stability.

Current tokenization and adaptation contracts are research surfaces. Real deployment-disjoint evidence decides whether the hypothesis survives.

## What is solid today

The strongest maintained capabilities include:

- canonical signal/frame and stream-descriptor contracts;
- config-first runtime graphs with bounded queues and latency/quality telemetry;
- deterministic recording, integrity verification, and replay;
- installed-wheel developer-preview qualification;
- out-of-tree plugin discovery through Python entry points;
- Scientific Authority for lineage, information roles, leakage, metrics, repeated measures, and failures;
- content-addressed Model Artifact promotion/reconstruction boundaries;
- Neural System Qualification with external model participation;
- upstream MNE/scikit-learn, pyRiemann, and Braindecode proving paths;
- governed ORION adaptation/state-selection/final-assessment contracts;
- deterministic systems falsification through Arena.

See [Project Status](docs/PROJECT_STATUS.md) for the strict maturity map.

## Record, replay, qualify

```bash
neuros record neuros.yaml \
  --output sessions/example \
  --session-id demo \
  --duration 10

neuros inspect sessions/example --verify
neuros replay sessions/example --config neuros.yaml
```

The canonical archive preserves sequence/timing/quality metadata, descriptors, configuration identity, package/Git provenance, runtime evidence, and integrity hashes. NWB and Zarr remain interoperability exports rather than substitutes for exact neurOS replay semantics.

## Evidence levels are not interchangeable

```text
software contract
      -> integration
      -> deterministic replay / scientific synthetic
      -> real dataset
      -> physical hardware
      -> closed loop
      -> clinical evidence
```

A result cannot self-promote to a stronger tier because it looks persuasive. Representation similarity is not task utility. Attribution is not mechanism. Hardware interoperability is not hardware qualification. Closed-loop evidence is not clinical certification.

See [Scientific Claims](docs/SCIENTIFIC_CLAIMS.md).

## Public architecture

The monorepo contains multiple distributions, but the public mental model should stay small:

| Surface | Responsibility |
| --- | --- |
| **neurOS** | runtime, clocks, configuration, recording/replay, interoperability, qualification composition |
| **ORION** | neural tokenization, representation, transfer, personalization, governed adaptation |
| **Evidence / NSQ** | frozen scientific authority, external participation, scoring, artifacts, falsification, claim boundaries |
| **Studio** | future evidence/runtime inspection surface that must not create a second authority |

Internal packages are implementation boundaries, not twelve separate products. Experimental packages should remain visibly experimental until they earn promotion.

## Contributing

The project needs external models, datasets, integrations, falsification cases, and independent reproduction more than it needs another internal subsystem.

Use the GitHub issue forms for:

- reproducible defects;
- ecosystem integration proposals;
- external model/method participation.

New maintained subsystems must answer a build-vs-integrate test: who already owns the adjacent capability, why a thin adapter/upstream contribution is insufficient, what unique neurOS authority is added, how it can be externally falsified, what it costs to maintain, and when it should be removed.

See [Contributing](CONTRIBUTING.md), [Governance](GOVERNANCE.md), and the [Roadmap](ROADMAP.md).

## Near-term release gates

Before calling neurOS a broadly usable public release, the project should earn all of the following:

1. a clean package install path that does not require understanding the monorepo;
2. a tagged developer-preview release with built wheels and reproducible artifacts;
3. a frozen public NSQ benchmark slice reproducible in one command;
4. at least one external model submitted without editing neurOS internals;
5. independent reproduction by researchers outside the implementation loop;
6. a stable protected `main` check surface;
7. public documentation/discussion channels that are maintained like product infrastructure.

Stars are not the primary target. **Independent researchers relying on neurOS because it makes their claims harder to accidentally overstate is the target.** Popularity should follow utility and trust.

## License

MIT. See [LICENSE](LICENSE).
