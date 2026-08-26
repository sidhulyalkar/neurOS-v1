# neurOS

**neurOS is an open execution and evidence platform for neural interfaces.**

The goal is not to replace the neuroscience ecosystem. neurOS provides stable runtime, replay, interoperability, synthetic systems falsification, and evidence semantics needed to make neural systems reproducible across hardware, models, datasets, sessions, and people.

**ORION** is the complementary neural-intelligence plane for tokenization, learned representations, transfer, personalization, auditable adaptation, and frozen final assessment.

> **Status:** active research and engineering platform. Software maturity, package versions, synthetic studies, and passing CI do not imply hardware qualification, clinical validation, biological correctness, or safety certification.

## Four public surfaces

The repository contains multiple internal distributions, but the public product architecture is deliberately small:

| Surface | Responsibility |
| --- | --- |
| **neurOS** | acquisition contracts, timing, runtime graphs, recording/replay, plugins, configuration, deployment semantics |
| **ORION** | neural tokenization, representations, foundation encoders, transfer, personalization, adaptation |
| **Evidence** | protocol identity, real-dataset benchmarks, Arena worlds, robustness, transfer, mechanistic interventions, qualification and claim authority |
| **Studio** | inspection of live/replayed runtime state, signals, Arena worlds, representations, adaptation, latency, and evidence |

`neuros-arena` belongs to **Evidence**. It is a deterministic BCI systems wind tunnel for finding failures across display, neural-world, device, transport, decoder, and application boundaries. It is not a fifth product surface and a simulator cannot self-promote a biological claim.

Read [`docs/PLATFORM.md`](docs/PLATFORM.md) for the governing architecture, [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) for the evidence-backed ecosystem matrix, [`docs/SYNTHETIC_BCI_ARENA.md`](docs/SYNTHETIC_BCI_ARENA.md) for the Arena contract, and [`docs/getting-started/first-10-minutes.md`](docs/getting-started/first-10-minutes.md) for the shortest newcomer path.

## Architecture

```text
hardware / LSL / datasets / replay
               |
               v
           SignalFrame
               |
               v
          RuntimeGraph
 source -> transform -> fusion -> decoder -> sink
               |                       |
               |                       +-> DecoderOutput / embedding
               |
 timing / queues / quality / recording / replay / provenance
               |
               +--------------+----------------+
                              |                |
                              v                v
                            ORION           Evidence
                     representation +      benchmark +
                        adaptation         qualification
                                                |
                                                +-> Arena / counterexamples
                              \                /
                               +------+-------+
                                      v
                                    Studio
```

The boundaries are intentional:

- **neurOS** answers: _How should neural systems execute reliably and reproducibly?_
- **ORION** answers: _How should neural activity be represented, transferred, and adapted?_
- **Evidence** answers: _What exactly supports a scientific or deployment claim, and how can it be falsified?_
- **Studio** answers: _How can a developer inspect the running system without creating a second runtime?_

## Quick start

This is a multi-distribution Python workspace. For standard BCI development:

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python scripts/bootstrap.py --profile bci --test-tools
```

Then exercise the installed platform:

```bash
neuros doctor --json
neuros plugins --json
neuros devices --json
neuros compatibility --json
neuros validate configs/examples/mock_bci.yaml --json
neuros run configs/examples/mock_bci.yaml --duration 2 --json
```

The checked-in mock runtime is deliberately training-free so an installation smoke test never hides model fitting behind the CLI.

## Config-first runtime

`SignalFrame` is the stable neural interchange contract. It carries stream/sequence identity, array payload and sample rate, device/host/synchronized timing, explicit clock domain, quality flags, and immutable metadata/provenance.

A versioned YAML configuration resolves installed plugins into a `RuntimeGraph`:

```text
SOURCE -> TRANSFORM -> ... -> FUSION -> DECODER -> SINK
                  \                    /
                   ------ MONITOR -----
```

Every runtime edge has bounded capacity and an explicit overload policy: `block`, `drop_oldest`, `drop_newest`, or `fail`. Runtime snapshots expose accepted/dropped counts, queue high-water marks, node failures, and latency summaries including p50/p95/p99.

Convenience `Pipeline` and `MultiModalPipeline` APIs compile to the same native executor rather than maintaining a second preferred runtime.

## Record, replay, and qualify

A live or deterministic source can be persisted into the canonical neurOS archive and replayed through the same processing graph:

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

The archive preserves sequence/timing/quality metadata, stream descriptors, config identity, Git/package provenance, runtime metrics, and per-frame integrity hashes. NWB and Zarr are interoperability exports rather than replacements for exact neurOS replay semantics.

A reproducible software qualification bundle can be sealed and independently verified:

```bash
neuros qualify configs/examples/mock_bci.yaml \
  --output /tmp/qualification \
  --duration 1.0
neuros reproduce /tmp/qualification
```

Qualification bundles bind configuration, environment, compatibility, clocks/devices, model/runtime evidence, decoder outputs, the recorded session, file hashes, and a root artifact identity. Synthetic qualification cannot self-award a physical hardware claim.

## Synthetic BCI Arena

Arena is the systems-level adversary for hardware-free BCI development:

```bash
neuros-arena --preset dual-target-smoke --output arena-report.json
```

Its causal stack is explicit:

```text
application / task state
        -> requested stimulus
        -> actually emitted display history
        -> neural world model + latent state
        -> sensor-space EEG
        -> device / montage / ADC / device clock
        -> transport / loss / jitter / silence
        -> decoder / quality authority
        -> application behavior
```

Display-coupled world models consume the **sample-and-held emitted waveform after frame quantization, jitter, drops, and held frames**, not an ideal target-frequency oracle. Current model modes include a legacy deterministic fixture, a driven state-space model, semi-synthetic recorded-background replay, and a portable lead-field-driven model.

Arena also provides population sampling, metamorphic checks, portable counterexamples, application traces, and optional SourceWeigher-based reality anchoring. Population coverage is coverage over the declared synthetic envelope, not human prevalence. Similarity weights are not probabilities that a synthetic participant is physiologically true.

See [`docs/EEG_WORLD_MODELS.md`](docs/EEG_WORLD_MODELS.md), [`docs/SCIENTIFIC_VALIDATION_POLICY.md`](docs/SCIENTIFIC_VALIDATION_POLICY.md), and [`docs/PUBLIC_EEG_ANCHORING_STUDY.md`](docs/PUBLIC_EEG_ANCHORING_STUDY.md).

## Ecosystem compatibility

neurOS integrates established tools rather than building inferior copies of them. Compatibility is a machine-readable evidence contract:

```bash
neuros compatibility
neuros compatibility mne --json
neuros compatibility braindecode --json
neuros compatibility snap --json
neuros compatibility ngclearn --json
neuros compatibility --status planned --json
```

Current evidence-backed lanes include BrainFlow, Lab Streaming Layer, MNE-Python, NWB/Zarr, MOABB, Braindecode, SNAP, and a narrow ngc-learn execution boundary. Planned or exploratory targets are tracked separately and cannot claim a qualification tier.

OpenBCI is currently represented through the BrainFlow device family rather than falsely marked hardware-qualified. Physical qualification requires a named board, firmware, transport, host, protocol, and measured evidence bundle.

See [`docs/NEUROAI_ECOSYSTEM.md`](docs/NEUROAI_ECOSYSTEM.md) and [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

### MNE ↔ neurOS

Install the optional bridge:

```bash
pip install -e "packages/neuros[interop-mne]"
```

```python
from neuros.interop import frames_from_raw, raw_from_signal_frames, stream_descriptor_from_raw

raw = ...  # existing MNE Raw

descriptor = stream_descriptor_from_raw(raw, stream_id="subject-01/eeg")
frames = tuple(
    frames_from_raw(
        raw,
        stream_id=descriptor.stream_id,
        chunk_samples=256,
    )
)
reconstructed = raw_from_signal_frames(frames, descriptor=descriptor)
```

The bridge does **not** resample, pad, reorder, or silently repair data. MNE `channel x sample` arrays become explicit neurOS `sample x channel` chunks with axis metadata. Ambiguous 2-D frames fail closed instead of guessing.

## Evidence and scientific claims

The project treats evidence as progressively stronger responsibility:

```text
software contract
      -> integration
      -> replay / scientific synthetic
      -> real dataset
      -> hardware
      -> closed loop
      -> clinical
```

The real-world program includes subject/session/run-aware longitudinal EEG evaluation through MOABB, frozen calibration/evaluation authority, model ladders, representation transfer lanes, and SourceWeigher comparisons. ORION now additionally separates calibration, qualification/model selection, selected state, and untouched final assessment.

A useful neural-system result should describe the dimensions relevant to its claim, such as task utility, calibration cost, subject/session/site/device transfer, montage robustness, artifact sensitivity, representation geometry, causal/mechanistic evidence, uncertainty, runtime latency/resource use, and immutable protocol/model/data identities.

A strong result in one dimension does not silently promote another. Representation similarity is not task utility. Attribution is not mechanism. Synthetic conformance is not human performance. Hardware qualification is not closed-loop qualification. Closed-loop evidence is not clinical certification.

See [`docs/REAL_WORLD_EVIDENCE.md`](docs/REAL_WORLD_EVIDENCE.md), [`docs/SCIENTIFIC_CLAIMS.md`](docs/SCIENTIFIC_CLAIMS.md), and [`docs/FINAL_ASSESSMENT_AUTHORITY.md`](docs/FINAL_ASSESSMENT_AUTHORITY.md).

## ORION

ORION begins where provenance-rich neural data becomes a machine-native representation:

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

Current tokenization research includes exact events, binned counts, relative-ISI WAIT/SPIKE tokens, burst/pause/rebound tokens, synchrony packets, vector-quantized motifs, and population assemblies.

```bash
python scripts/orion/run_tokenizer_benchmark.py \
  configs/orion/tokenization_smoke.yaml \
  --output /tmp/orion-tokenization
```

Synthetic motif benchmarks are falsification tools, not proof that a tokenizer is superior on real human neural data. ORION promotion requires leakage-controlled, deployment-unit-disjoint evidence. Adaptation cannot consume final-assessment rows, and the selected state evaluated at final assessment must be the exact state qualified before those rows were opened.

A central ORION target is to measure whether learned representations can preserve or improve held-out neural utility while reducing per-user calibration, without trading away robustness, latency, provenance, uncertainty calibration, or representation stability.

## Models, transfer, and mechanisms

Internal research distributions remain modular behind the public surfaces:

- `neuros-models` provides faithful task decoders and inspectable analysis manifests;
- `neuros-foundation` provides external foundation-model discovery, adapters, representation probes, conformance operators, and benchmark authority;
- `neuros-sourceweigher` estimates source/session/subject/device reliability and transfer utility;
- `neuros-mechint` provides intervention, faithfulness, replication, dose-response, and evidence-artifact contracts;
- `neuros-neurofm` remains experimental native foundation-model R&D.

These packages answer different questions while sharing stable neural/evidence contracts instead of importing one another opportunistically.

## Plugins

External packages can extend neurOS through Python entry points:

```text
neuros.sources
neuros.transforms
neuros.tokenizers
neuros.encoders
neuros.decoders
neuros.sinks
neuros.monitors
neuros.world_models
```

The kernel does not import concrete hardware, model ecosystems, UI/cloud integrations, ORION implementations, or Arena world-model implementations. A world-model plugin owns neural dynamics, not display/device/transport/application semantics.

A useful integration should solve a concrete boundary problem: canonical conversion, synchronization, conformance, reproducibility, evidence, transfer/adaptation, or duplicated glue assumptions. Package-name accumulation is not a goal.

A maintained out-of-tree plugin authoring kit and clean-wheel compatibility lane are the next major developer-preview usability gate.

## Scientific trust and releases

The public repository treats project citizenship as an executable surface rather than paperwork:

- [`CITATION.cff`](CITATION.cff) defines machine-readable citation metadata without inventing an archival DOI;
- [`GOVERNANCE.md`](GOVERNANCE.md) defines maintainer, contract-change, RFC, evidence-promotion, and merge rules;
- [`SECURITY.md`](SECURITY.md) covers software, private neural-data, artifact-integrity, plugin, and eventual actuator-risk reporting boundaries;
- [`SUPPORT.md`](SUPPORT.md) defines reproducible support requests and integration criteria;
- [`docs/SCIENTIFIC_CLAIMS.md`](docs/SCIENTIFIC_CLAIMS.md) governs evidence-tier language;
- [`docs/RELEASE_POLICY.md`](docs/RELEASE_POLICY.md) governs versioning, deprecation, exact-head release qualification, checksums, and publishing authority;
- [`CHANGELOG.md`](CHANGELOG.md) records maintained user-facing changes going forward.

The root `pyproject.toml` workspace is the single maintained-package inventory. CI, repository hygiene, and release-candidate tooling derive package membership from it through `scripts/list_workspace_packages.py`. This prevents a new distribution from being added to the workspace while silently disappearing from wheel builds, checksums, or maintained-package documentation checks.

Release-candidate CI builds every workspace wheel, validates package metadata, creates SHA-256/component manifests, and smoke-installs the public SDK plus Arena from the exact wheel set produced by the source revision. Package publication remains intentionally separate from pull-request authority. PyPI publishing should only be enabled after trusted publishing/OIDC is configured and reviewed.

## Internal workspace map

```text
packages/
  neuros-core/          stable runtime/data/replay/plugin contracts
  neuros-drivers/       hardware, simulated, dataset, and LSL sources
  neuros/               user-facing SDK, CLI, interoperability composition
  orion/                stable ORION token/representation/adaptation/assessment contracts
  neuros-arena/         causal synthetic worlds, faults, populations, counterexamples
  neuros-models/        task decoders and inspectable model surfaces
  neuros-foundation/    foundation-model and real-dataset evidence adapters
  neuros-sourceweigher/ transfer/source reliability and similarity weighting
  neuros-mechint/       causal/mechanistic evidence framework
  neuros-neurofm/       experimental native neural foundation-model R&D
  neuros-ui/            Studio implementation substrate
  neuros-cloud/         optional distributed/provider integrations
```

These are implementation boundaries, not twelve competing product identities.

## Quality

CI separates kernel contracts across Python 3.10-3.12, installed BCI execution, workspace wheel builds, recording/replay and NWB/Zarr interoperability, scientific/latency gates, model/mechanistic contracts, foundation interoperability, SourceWeigher, ORION authority, longitudinal real-dataset evidence, hardware-boundary drivers, ecosystem compatibility, NeuroAI upstream conformance, Synthetic BCI Arena, public trust, and release-candidate artifacts.

Hardware-specific claims require recorded qualification manifests for the exact device, firmware, transport, host, configuration, model, and artifact identities involved.

## Roadmap

The highest-value sequence is now:

1. ship a maintained out-of-tree plugin authoring kit and clean-wheel compatibility lane so external labs can extend neurOS without kernel forks;
2. execute the predeclared longitudinal EEG model ladder and ORION calibration-reduction studies under identical frozen split/final-assessment authority;
3. anchor Arena world banks against held-out public EEG and expand world models only behind the common causal contract;
4. establish DANDI/SpikeInterface invasive-data interoperability and selected visual-neuroscience benchmark evidence;
5. qualify one named real-device reference pipeline end to end;
6. converge `neuros-ui` into a coherent Studio experience that displays runtime, Arena, ORION, and claim/evidence status without creating a second execution engine;
7. add an explicit closed-loop safety/constraint plane before making stronger closed-loop product claims;
8. make externally maintained plugins, reference deployments, and reproducible release artifacts easy to build without kernel forks.

See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md), [`ROADMAP.md`](ROADMAP.md), and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).
