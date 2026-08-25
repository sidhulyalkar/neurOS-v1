# neurOS Documentation

**neurOS is an open execution and evidence platform for neural interfaces.** It is designed so neural systems can be replayed, measured, compared, adapted, and eventually qualified across changing hardware, models, datasets, and people.

The public architecture has four surfaces:

| Surface | Question it answers |
| --- | --- |
| **neurOS** | How should neural systems execute reliably and reproducibly? |
| **ORION** | How should neural activity be represented, transferred, and adapted? |
| **Evidence** | What does a neural-system claim actually have evidence for? |
| **Studio** | How can a developer inspect the live/replayed system and its evidence? |

> neurOS is an active research and engineering platform. Software tests are not hardware qualification, medical validation, biological proof, or clinical certification.

## Start here

- [Platform architecture and design rules](PLATFORM.md)
- [Evidence-backed ecosystem compatibility](COMPATIBILITY.md)
- [Project status and maturity](PROJECT_STATUS.md)
- [Real-world evidence program](REAL_WORLD_EVIDENCE.md)
- [Installation](getting-started/installation.md)
- [Detailed runtime architecture](ARCHITECTURE.md)
- [API surface](API_REFERENCE.md)

## One platform, many ecosystems

neurOS should integrate established neuroscience software rather than replace it.

```text
BrainFlow / LSL / datasets
           |
           v
       SignalFrame
           |
           +-----------> MNE / scientific interop
           |
           v
       RuntimeGraph
 source -> transform -> fusion -> decoder -> sink
           |                       |
           |                       +-> DecoderOutput / representation
           |
 timing / queues / recording / replay / quality
           |
           +-------------+-------------------+
                         |                   |
                         v                   v
                       ORION              Evidence
                 representation +      benchmarks +
                    adaptation          qualification
                         \                   /
                          +--------+--------+
                                   v
                                 Studio
```

Compatibility is machine-readable:

```bash
neuros compatibility
neuros compatibility mne --json
neuros compatibility --status planned --json
```

A supported integration must point to executable evidence. Planned integrations cannot claim a qualification tier.

## Quick runtime

```bash
neuros doctor --json
neuros plugins --json
neuros devices --json
neuros validate configs/examples/mock_bci.yaml --json
neuros run configs/examples/mock_bci.yaml --duration 2 --json
```

Configuration is versioned and resolved through entry-point plugins. Standard `Pipeline` facades compile to the same native `RuntimeGraph` executor used by the config-first path.

## Record once, replay everywhere

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

The canonical archive preserves sequence, timing, quality, provenance, configuration identity, runtime metrics, and frame integrity. NWB and Zarr remain interoperability exports rather than alternate runtime authorities.

## MNE bridge

MNE is the first direct scientific object bridge under the convergence architecture:

```bash
pip install "neuros[interop-mne]"
```

```python
from neuros.interop import frames_from_raw, raw_from_signal_frames, stream_descriptor_from_raw

raw = ...
descriptor = stream_descriptor_from_raw(raw, stream_id="subject-01/eeg")
frames = tuple(frames_from_raw(raw, stream_id=descriptor.stream_id, chunk_samples=256))
reconstructed = raw_from_signal_frames(frames, descriptor=descriptor)
```

The adapter preserves channel identity and sampling rate, records explicit array-axis semantics, propagates absolute measurement timing when MNE provides it, and refuses ambiguous geometry rather than guessing.

## Evidence before breadth

The repository separates evidence into progressively stronger responsibilities:

```text
software contract
      -> integration
      -> real-dataset evidence
      -> hardware qualification
      -> closed-loop qualification
      -> clinical evidence
```

The longitudinal EEG program already establishes frozen subject/session/run-aware evaluation and real MOABB dataset lanes. The next goal is not to add dozens of algorithms. It is to make the same evidence authority work across external model ecosystems, ORION representations, transfer methods, hardware sessions, and mechanistic interventions.

## ORION

ORION is the neural-intelligence plane. Current stable work begins with explicit neural tokenization and representation contracts, while research packages explore learned encoders, transfer, source reliability, adaptation, and native foundation models.

Promotion should require deployment-unit-disjoint evidence. A representation that wins on random windows but fails across subjects or sessions is not a useful ORION representation.

## Studio

`neuros-ui` is being treated as the implementation substrate for Studio, not as an alternate orchestration framework. Studio should ultimately show:

- runtime graph state and queue pressure;
- synchronized live/replayed signals and quality flags;
- latency and failure telemetry;
- decoder probabilities, uncertainty, and embeddings;
- ORION representations and adaptation events;
- source/transfer weights;
- benchmark and qualification evidence;
- replay-to-replay regression comparisons.

The runtime remains authoritative. Studio observes it.

## Repository organization

The workspace remains internally modular because dependency boundaries matter. Users should think in the four public surfaces rather than memorizing every internal distribution.

Research notebooks and exploratory algorithms remain under `experiments/` or research packages until they satisfy a stable contract and evidence gate. Historical migration notes live under `docs/archive/` and are retained for provenance, not as current architecture guidance.
