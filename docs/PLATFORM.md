# neurOS Platform

neurOS is being converged around four public product surfaces. The repository may contain multiple distributions, experiments, and adapters, but users should not need to learn the monorepo topology before they can understand what the platform does.

## The four public surfaces

| Surface | Promise | Owns | Does not own |
| --- | --- | --- | --- |
| **neurOS** | Execute neural systems reliably | acquisition contracts, synchronization, runtime graphs, queues, recording/replay, plugins, configuration, deployment constraints | third-party model zoos, dataset catalogs, speculative research algorithms |
| **ORION** | Build neural representations that can adapt | tokenization, learned representations, neural encoders, transfer, personalization, auditable adaptation | hardware drivers, transport protocols, application UI orchestration |
| **Evidence** | Say exactly what a neural-system claim is supported by | benchmark authority, split identity, provenance, calibration curves, robustness, transfer, mechanistic interventions, synthetic conformance worlds, qualification tiers | marketing claims, implicit validation, hidden benchmark state |
| **Studio** | Make the running neural system inspectable | runtime graph visualization, signal/quality views, latency, replay comparison, representation diagnostics, adaptation events, Arena worlds, evidence inspection | alternate runtime semantics or hidden execution paths |

These are conceptual product surfaces, not a demand for four Python distributions. Internal packages should stay split where dependency, release, or scientific boundaries justify it.

`neuros-arena` belongs primarily to the **Evidence** surface. It is the deterministic systems wind tunnel for falsifying assumptions across display, neural-world, device, transport, decoder, and application boundaries. Synthetic success in Arena does not promote a physiological, hardware, human, or clinical claim.

## Architectural shape

```text
                         applications
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          +-------------+           +-------------+
          |   neurOS    |<--------->|    ORION    |
          | execution   |           | intelligence|
          +------+------+           +------+------+
                 |                         |
                 +------------+------------+
                              |
                              v
                       +-------------+
                       |  Evidence   |
                       | claims +    |
                       | provenance  |
                       +------+------+ 
                              |
                +-------------+-------------+
                |                           |
                v                           v
         +-------------+             +-------------+
         |    Arena    |             |   Studio    |
         | falsify +   |             | inspection  |
         | stress-test |             |             |
         +-------------+             +-------------+
```

Arena produces evidence inputs and counterexamples. Studio consumes stable runtime/evidence contracts. Neither is allowed to become a second orchestration system.

## Platform invariants

### 1. Stable contracts point inward; integrations point outward

`neuros-core` may define `SignalFrame`, `StreamDescriptor`, `DecoderOutput`, runtime queues, graph semantics, recording, and plugin contracts. It must not import MNE, BrainFlow, LSL, Braindecode, NeuralBench, MOABB, SpikeInterface, DANDI, ORION implementations, or UI/cloud packages.

Third-party ecosystems are integrated at the edge through adapters and plugins. Neural world models follow the same rule through the `neuros.world_models` entry-point contract.

### 2. Interoperate instead of reimplement

neurOS should not build a competing replacement for MNE preprocessing, the Braindecode model zoo, MOABB protocol catalog, NeuralBench task registry, SpikeInterface sorters, DANDI storage, LSL transport, or mature neural-mass simulators.

The neurOS contribution is to translate these ecosystems into shared execution and evidence semantics:

```text
external object / stream / model
              |
              v
      explicit neurOS adapter
              |
              v
stable contract + provenance
              |
      +-------+-------+
      |               |
      v               v
 runtime/replay      Evidence
```

Arena follows the same principle. MNE forward models, public EEG, neural-mass simulators, and learned generators may provide world-model inputs or plugins while Arena owns the causal systems envelope around them.

### 3. A support claim is a data structure

External compatibility is exposed through `neuros.compatibility` and the CLI:

```bash
neuros compatibility
neuros compatibility mne --json
neuros compatibility --status planned
```

Each integration records its public support state, capabilities, strongest evidence tier, evidence paths, and install hint. A planned integration cannot claim an evidence tier. A supported integration must point at executable evidence.

See [Compatibility](COMPATIBILITY.md).

### 4. Scientific evidence and product qualification are different ladders

The common evidence hierarchy is:

```text
software contract
      |
      v
integration
      |
      v
scientific synthetic / replay
      |
      v
real-dataset evidence
      |
      v
hardware qualification
      |
      v
closed-loop qualification
      |
      v
clinical evidence
```

A higher software version never promotes a claim up this ladder automatically. Arena lives in the synthetic/replay portion unless an explicitly separate real-data, hardware, or human protocol supplies stronger evidence.

### 5. Replay is the bridge between research and deployment

A live session should be recordable into a canonical neurOS archive and replayable through the same runtime graph. That allows hardware-independent regression, model comparison, ORION experiments, fault injection, latency analysis, Arena semi-synthetic studies, and evidence generation without changing the underlying neural contract.

### 6. Ambiguity fails closed

Adapters must not silently guess channel axes, units, stream identity, model family, timestamps, benchmark partitions, source-space identity, or assessment rows. When information is insufficient, the correct behavior is a clear failure with enough context to repair the input.

The MNE bridge follows this rule: two-dimensional `SignalFrame` data can only be exported when `axis_order=('sample', 'channel')` is explicit. Arena lead-field bundles require explicit source indices rather than silently guessing a visual source.

### 7. Synthetic systems are adversaries, not authorities

A simulator is useful when it can expose a software or systems failure before hardware or participant time is spent. It is dangerous when a plausible waveform is allowed to masquerade as biological evidence.

Arena therefore keeps requested stimulus, actually emitted display history, neural world state, sensor/device effects, transport faults, decoder outputs, and application behavior as separate inspectable layers. Population summaries describe coverage over the declared synthetic envelope, not prevalence in people. Reality-anchor weights describe similarity under a declared geometry, not a probability that a synthetic participant is physiologically true.

## Internal package mapping

The current workspace maps onto the public surfaces as follows. `packages/orion` is the repository directory; its Python distribution name is `neuros-orion`.

| Internal distribution | Public surface | Role |
| --- | --- | --- |
| `neuros-core` | neurOS | kernel contracts/runtime/replay/plugin semantics |
| `neuros-drivers` | neurOS | physical, dataset, replay, and synthetic acquisition boundaries |
| `neuros` | neurOS | SDK, CLI, interoperability composition |
| `neuros-orion` (`packages/orion`) | ORION | stable token/representation/adaptation/assessment interfaces |
| `neuros-arena` | Evidence | causal synthetic worlds, system faults, populations, counterexamples, reality anchoring |
| `neuros-models` | neurOS + Evidence | deployable decoders and inspectable model contracts |
| `neuros-foundation` | ORION + Evidence | external foundation-model adapters and representation benchmarks |
| `neuros-sourceweigher` | ORION + Evidence | transfer/source reliability and declared similarity weighting |
| `neuros-mechint` | Evidence | intervention and mechanistic evidence contracts |
| `neuros-neurofm` | ORION research | native foundation-model experiments |
| `neuros-ui` | Studio | dashboard and visualization surfaces |
| `neuros-cloud` | optional infrastructure | distributed/provider integrations |

This mapping is intentionally asymmetric. Public architecture should be simple even when implementation architecture is modular.

## What gets built next

The preferred sequence is evidence-driven:

1. **Compatibility spine:** keep BrainFlow, LSL, MNE, NWB/Zarr, MOABB, Braindecode, and selected NeuroAI integrations explicit and evidence-backed.
2. **Arena public anchoring:** run held-out public EEG through reproducible semi-synthetic and reality-anchor protocols, without using synthetic results as human claims.
3. **Offline neural interoperability:** mature MNE/Braindecode/MOABB and invasive-data bridges without copying upstream algorithms.
4. **External benchmark workers:** isolate version-sensitive benchmark/model ecosystems and emit neurOS evidence artifacts rather than kernel dependencies.
5. **Real-device qualification:** qualify named hardware/firmware/transport combinations with measured loss, drift, reconnect behavior, latency, and reproducible manifests.
6. **ORION real-data promotion:** compare representations and adaptation on deployment-unit-disjoint datasets with calibration, transfer, robustness, uncertainty, and final-assessment authority.
7. **World-model ladder:** add richer source-space, neural-mass, and learned world models only behind the common Arena contract and only with held-out real-data plus metamorphic qualification.
8. **Studio convergence:** provide one inspection surface for live runtime state, replay, Arena worlds, ORION representations, and evidence.
9. **Closed-loop safety plane:** add explicit stale-data rejection, confidence/quality gates, action constraints, deadman behavior, rate limits, and emergency-stop semantics before stronger closed-loop product claims.

## Design test for new work

Before adding a package or major abstraction, ask:

1. Does an established ecosystem already own this problem well?
2. Can neurOS integrate it through a stable contract instead?
3. Which of the four public surfaces does the feature strengthen?
4. What evidence tier does the implementation genuinely reach?
5. Can the behavior be replayed, measured, falsified, and reproduced?
6. If the component is synthetic or learned, what prevents it from silently promoting its own scientific claim?

If those questions do not have crisp answers, the feature probably belongs in `experiments/` rather than the stable platform.
