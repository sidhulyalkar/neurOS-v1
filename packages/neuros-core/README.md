# neurOS Core

`neuros-core` is the dependency-light kernel for the neurOS BCI runtime. It owns the contracts and execution semantics that hardware, models, ORION, recording, and higher-level applications share.

## What belongs here

- canonical `SignalFrame`, `StreamDescriptor`, and `DecoderOutput` contracts;
- `RuntimeGraph`, bounded edges, overflow policies, lifecycle, and supervised execution;
- configuration schemas and plugin discovery;
- clock synchronization primitives;
- processing transforms;
- canonical recording/replay semantics and integrity/provenance support;
- generic scientific/runtime quality gates.

Concrete device SDKs, task-specific neural networks, cloud vendors, UI frameworks, and research implementations should remain outside the kernel.

## Installation

```bash
pip install neuros-core
```

Optional extras:

```bash
pip install "neuros-core[evaluation]"
pip install "neuros-core[recording]"  # NWB + Zarr interoperability
pip install "neuros-core[test]"
```

## Runtime graph

A maintained neurOS execution path is a typed graph:

```text
SOURCE -> TRANSFORM -> FUSION -> DECODER -> SINK
                  \              /
                   --- MONITOR ---
```

Every runtime edge has explicit capacity and overflow policy. Runtime snapshots expose accepted/dropped items, queue high-water marks, node failures, and bounded latency summaries.

```python
from neuros.runtime import OverflowPolicy, RuntimeGraph

# Concrete operators are supplied by installed packages/plugins.
graph = RuntimeGraph()
print(OverflowPolicy.DROP_OLDEST)
```

For a complete runnable workflow, install the user-facing `neuros` package and use the checked-in mock configuration.

## Recording and replay

The canonical archive preserves the neurOS neural-data contract rather than flattening it to a generic array. Exact sequence identity, timing fields, quality flags, stream descriptors, metadata/provenance, and payload integrity can therefore be replayed through the same runtime used by live sources.

NWB and Zarr are optional interoperability exports. They do not replace neurOS' authoritative replay representation.

## Architectural guarantee

The intended dependency direction is:

```text
neuros-core
    <- drivers / task models / user SDK
    <- ORION / foundation / transfer / mechanism integrations
    <- experiments and applications
```

A kernel change should be general enough to support multiple concrete devices/models and should arrive with contract, replay, failure, and overload tests appropriate to its impact.

## Documentation

Current documentation lives in the repository:

- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `ROADMAP.md`
- `CONTRIBUTING.md`

Repository: https://github.com/sidhulyalkar/neurOS-v1

## License

MIT License.
