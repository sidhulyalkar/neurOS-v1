# neurOS Platform

neurOS is being converged around four public product surfaces. The repository may contain many internal distributions, experiments, and adapters, but users should not need to learn the monorepo topology before they can understand what the platform does.

## The four public surfaces

| Surface | Promise | Owns | Does not own |
| --- | --- | --- | --- |
| **neurOS** | Execute neural systems reliably | acquisition contracts, synchronization, runtime graphs, queues, recording/replay, plugins, configuration, deployment constraints | third-party model zoos, dataset catalogs, speculative research algorithms |
| **ORION** | Build neural representations that can adapt | tokenization, learned representations, neural encoders, transfer, personalization, auditable adaptation | hardware drivers, transport protocols, application UI orchestration |
| **Evidence** | Say exactly what a neural-system claim is supported by | benchmark authority, split identity, provenance, calibration curves, robustness, transfer, mechanistic interventions, qualification tiers | marketing claims, implicit validation, hidden benchmark state |
| **Studio** | Make the running neural system inspectable | runtime graph visualization, signal/quality views, latency, replay comparison, representation diagnostics, adaptation events, evidence inspection | alternate runtime semantics or hidden execution paths |

These are conceptual product surfaces, not a demand for four Python distributions. Internal packages should stay split where dependency, release, or scientific boundaries justify it.

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
                              v
                       +-------------+
                       |   Studio    |
                       | inspection  |
                       +-------------+
```

Studio is a consumer of stable runtime/evidence contracts. It must never become a second orchestration system.

## Platform invariants

### 1. Stable contracts point inward; integrations point outward

`neuros-core` may define `SignalFrame`, `StreamDescriptor`, `DecoderOutput`, runtime queues, graph semantics, recording, and plugin contracts. It must not import MNE, BrainFlow, LSL, Braindecode, NeuralBench, MOABB, SpikeInterface, DANDI, ORION implementations, or UI/cloud packages.

Third-party ecosystems are integrated at the edge through adapters and plugins.

### 2. Interoperate instead of reimplement

neurOS should not build a competing replacement for MNE preprocessing, the Braindecode model zoo, MOABB protocol catalog, NeuralBench task registry, SpikeInterface sorters, DANDI storage, or LSL transport.

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

A higher software version never promotes a claim up this ladder automatically.

### 5. Replay is the bridge between research and deployment

A live session should be recordable into a canonical neurOS archive and replayable through the same runtime graph. That allows hardware-independent regression, model comparison, ORION experiments, fault injection, latency analysis, and evidence generation without changing the underlying neural contract.

### 6. Ambiguity fails closed

Adapters must not silently guess channel axes, units, stream identity, model family, timestamps, or benchmark partitions. When information is insufficient, the correct behavior is a clear failure with enough context to repair the input.

The MNE bridge follows this rule: two-dimensional `SignalFrame` data can only be exported when `axis_order=('sample', 'channel')` is explicit.

## Internal package mapping

The current workspace maps onto the public surfaces as follows:

| Internal distribution | Public surface | Role |
| --- | --- | --- |
| `neuros-core` | neurOS | kernel contracts/runtime/replay |
| `neuros-drivers` | neurOS | acquisition plugins |
| `neuros` | neurOS | SDK, CLI, interoperability composition |
| `neuros-models` | neurOS + Evidence | deployable decoders and inspectable model contracts |
| `neuros-foundation` | ORION + Evidence | external foundation-model adapters and representation benchmarks |
| `neuros-sourceweigher` | ORION + Evidence | transfer/source reliability |
| `neuros-mechint` | Evidence | intervention and mechanistic evidence contracts |
| `neuros-orion` | ORION | stable token/representation/adaptation interfaces |
| `neuros-neurofm` | ORION research | native foundation-model experiments |
| `neuros-ui` | Studio | dashboard and visualization surfaces |
| `neuros-cloud` | optional infrastructure | distributed/provider integrations |

This mapping is intentionally asymmetric. Public architecture should be simple even when implementation architecture is modular.

## What gets built next

The preferred sequence is evidence-driven:

1. **Compatibility spine:** BrainFlow, LSL, MNE, NWB/Zarr and MOABB have explicit evidence-backed inventory entries.
2. **Offline EEG interoperability:** mature MNE/Braindecode/MOABB adapters, without copying their algorithms.
3. **External benchmark workers:** NeuralBench runs in an isolated environment and emits neurOS evidence artifacts rather than becoming a kernel dependency.
4. **Invasive data lane:** NWB/DANDI/SpikeInterface adapters and replayable electrophysiology examples.
5. **Real-device qualification:** named hardware/firmware/transport manifests with measured loss, drift, reconnect behavior and latency.
6. **ORION real-data promotion:** representations compared on deployment-unit-disjoint datasets with calibration and transfer curves.
7. **Studio convergence:** one inspection surface for live runtime state, replay, ORION representations and evidence.
8. **Closed-loop safety plane:** explicit stale-data rejection, confidence/quality gates, action constraints, deadman behavior, rate limits and emergency stop semantics.

## Design test for new work

Before adding a package or major abstraction, ask:

1. Does an established ecosystem already own this problem well?
2. Can neurOS integrate it through a stable contract instead?
3. Which of the four public surfaces does the feature strengthen?
4. What evidence tier does the implementation genuinely reach?
5. Can the behavior be replayed, measured and falsified?

If those questions do not have crisp answers, the feature probably belongs in `experiments/` rather than the stable platform.
