# Runtime Process Transport Authority

This document defines the promoted contract boundary for process-executed neurOS runtime nodes.

The short version is deliberately conservative:

- process execution is **fault isolation**, not a security sandbox;
- `pickle` remains the compatibility default for payloads it can represent;
- `shared_memory` is explicit opt-in for canonical/numeric payloads that need it;
- shared-memory mailbox capacities are explicit authority, not hints;
- unsupported payloads and capacity overflow fail closed;
- there is no silent fallback from shared memory to pickle;
- callback inputs are materialized into child-local memory before arbitrary operator code runs, so this is **not an end-to-end zero-copy contract**.

## Programmatic runtime surface

Process execution is currently configured through `RuntimeNode` / `RuntimeGraph`:

```python
RuntimeNode(
    node_id="transform:eeg:0",
    kind=NodeKind.TRANSFORM,
    operator=transform,
    executor="process",
    execution_timeout_s=2.0,
    process_transport="shared_memory",
    process_request_capacity_bytes=2 * 1024 * 1024,
    process_response_capacity_bytes=2 * 1024 * 1024,
)
```

`executor="process"` requires an explicit finite positive `execution_timeout_s`.
Booleans, strings, zero/negative values, NaN, and infinities are rejected as duration authority.

`process_transport` is one of:

- `pickle`
- `shared_memory`

`pickle` requires no mailbox-capacity declaration.

`shared_memory` requires explicit positive integer request and response capacities. Booleans are not integers for this authority boundary.

Process-only declarations on non-process nodes are rejected. Source nodes remain inline because source lifecycle/stream isolation is a separate capability and has not been promoted by this tranche.

## Configuration compiler boundary

The versioned `PipelineConfig -> resolve_config -> RuntimeGraph` compiler does **not** yet expose these process-execution and payload-transport fields.

Do not imply that YAML configuration can select shared-memory process transport today. That configuration-authority work is tracked separately so schema/versioning compatibility can be answered explicitly rather than being smuggled into the runtime primitive.

Programmatic `RuntimeNode` construction is the authoritative configuration surface for this tranche.

## What each transport can represent

Transport support is a representation question, not merely a performance question.

At the current contract revision:

| Payload | `pickle` process transport | `shared_memory` process transport |
|---|---|---|
| NumPy numeric/bool ndarray | supported | supported |
| deterministic scalar/list/tuple/string-keyed mapping payloads | supported when pickleable | supported by canonical codec |
| `SignalFrame` | canonical instance is not generically pickleable | supported |
| `NeuralWindow` | canonical instance is not generically pickleable | supported |
| current `DecoderOutput` | canonical instance is not generically pickleable | supported |
| `TransformEmission` containing supported canonical items | depends on contained pickleability | supported |
| unsupported Python object / object-dtype ndarray | pickle semantics apply | rejected fail-closed |

The canonical contracts use immutable provenance structures such as `MappingProxyType`. Generic pickle therefore cannot currently represent several canonical neurOS objects without changing their representation.

That is not converted into an automatic fallback. For example, a process transform, sink, or monitor that must receive a canonical `SignalFrame` should explicitly select `shared_memory` and capacities large enough for its declared payload envelope.

A process decoder is a distinct case: the standard executor converts `SignalFrame` / `NeuralWindow` decoder input to the numeric model batch before crossing the process boundary. Therefore a decoder can still use pickle when the actual process payload is the resulting NumPy array.

## Shared-memory payload language

The shared-memory transport keeps process control metadata on the multiprocessing pipe and moves numeric array bytes through fixed-capacity parent-owned shared-memory mailboxes.

The v1 payload language supports:

- `None`, strings, booleans, integers, floats and complex scalars;
- NumPy numeric and boolean arrays;
- tuples and lists;
- deterministic mappings with string keys;
- `SignalFrame`;
- `NeuralWindow`;
- `DecoderOutput`;
- `TransformEmission`.

Object-dtype arrays and unsupported Python object graphs are rejected.

The reader accepts the concrete built-in manifest/container shapes emitted by the writer. Protocol identity, lease identity, array geometry, alignment, non-overlap and declared byte boundaries are validated exactly. Reader permissiveness is intentionally not broader than the writer's protocol language.

Once the v1 protocol is promoted, incompatible reader/writer changes should be treated as an explicit protocol-version decision rather than silently tightening or widening the accepted wire language.

## Mailbox authority and ownership

Shared-memory mailboxes are fixed-capacity by design.

- request and response capacities are independent;
- an encode that would exceed capacity raises a transport-capacity error;
- the transport does not resize behind the caller's back;
- the transport does not fall back to pickle;
- the creator owns unlink authority;
- an attached peer never owns unlink authority;
- parent unlink happens only after direct-child death is proven.

The explicit-capacity contract keeps resource ownership, deployment memory budgeting, and failure provenance inspectable. A future ergonomic sizing layer may derive recommendations, but it must not weaken the underlying authority contract.

## Materialization and the zero-copy boundary

Shared memory avoids serializing/copying bulk numeric bytes through the multiprocessing pipe, but arbitrary operator callbacks do not receive views into the live mailbox.

The child decoder materializes arrays into independent local NumPy memory before invoking operator code. Parent-side results are likewise detached from subsequent mailbox reuse.

This prevents a callback from retaining a view that is silently mutated on the next mailbox lease and keeps canonical objects independent of shared-memory lifetime.

Accordingly, the promoted claim is:

> shared-memory payload transport for numeric/canonical process messages

It is **not**:

> end-to-end zero-copy neural execution

A future zero-copy callback contract would need separate lifetime/lease ownership semantics and its own adversarial qualification.

## Process lifecycle and failure precedence

Pickle and shared memory use one common `PersistentProcessWorker` lifecycle authority. They do not own separate process engines.

The common state machine owns:

- spawn-safe direct-child creation;
- generation and request identity;
- ready/heartbeat control;
- persistent operator state;
- hard execution timeout;
- asyncio cancellation containment;
- crash detection;
- receipts;
- graceful shutdown;
- terminate/join/kill escalation.

Primary execution evidence is not overwritten by secondary cleanup noise.

For example, if a call exceeds its hard timeout and shared-memory cleanup then fails after direct-child death has already been proven, the timeout remains the primary operation failure. Cleanup degradation remains secondary evidence and cleanup may be retried by executor-owned close authority.

Failure to prove direct-child death is different: loss of containment authority is allowed to supersede the original operation failure because the runtime can no longer truthfully claim that the owned child was terminated.

## Snapshot provenance

Runtime snapshots record process execution declarations and per-call receipts.

For each process node, `process_execution` records:

- transport;
- hard execution timeout;
- request capacity when applicable;
- response capacity when applicable.

`process_receipts` records generation/request identity and outcome history.

A separate follow-up may expose historical cleanup degradation that was later successfully retried. Such telemetry must remain secondary provenance and must not rewrite the primary operation failure.

## Performance evidence

Performance evidence is descriptive and non-gating.

### Raw-array crossover experiment

The original transport crossover experiment measures identical contiguous `float32` NumPy round trips. It excludes process startup through warmups, alternates transport order, verifies every returned array and records latency distributions across Ubuntu, macOS and Windows.

Those results show that tiny payloads are near parity while shared memory becomes materially faster for large arrays. The crossover and tail behavior differ by operating system, so the evidence does **not** justify one universal automatic transport threshold.

These numbers are **array transport evidence**. They must not be presented as a blanket speedup for arbitrary canonical neural objects.

### Canonical workload experiment

A second semantic-source-pinned experiment measures:

- raw ndarray;
- `SignalFrame`;
- `NeuralWindow`;
- multi-array `DecoderOutput`.

Across Ubuntu 24.04, macOS 14 and Windows 2025, shared memory successfully round-tripped all four workloads. Generic pickle supported the ndarray workload but canonical `SignalFrame`, `NeuralWindow` and current `DecoderOutput` failed pickle preflight because of immutable provenance representation.

At roughly 1 MiB of numeric array payload, observed shared-memory p50 latency was approximately:

| OS | `SignalFrame` | `NeuralWindow` | `DecoderOutput` |
|---|---:|---:|---:|
| Ubuntu 24.04 | 1.43 ms | 1.30 ms | 1.00 ms |
| Windows 2025 | 2.22 ms | 2.26 ms | 1.46 ms |
| macOS 14 | 0.90 ms | 0.79 ms | 0.69 ms |

The corresponding control manifests remained below roughly 1 KiB in this experiment.

Canonical reconstruction still has real cost. At 8 MiB on Ubuntu, for example, `SignalFrame` was about 8.7 ms p50 while a naked ndarray was about 2.9 ms p50 over shared memory. On Windows those values were about 10.2 ms and 4.9 ms respectively.

Therefore the evidence supports two separate conclusions:

1. shared memory materially improves large numeric process transport and enables canonical payload classes that generic pickle cannot currently represent;
2. canonical object reconstruction is not free, and shared-memory canonical payloads must not be described as equivalent to raw ndarray transport.

Benchmark results are environment-specific measurements, not real-time guarantees or scientific-performance claims.

## Security and trust boundary

Neither process transport is a hostile-code sandbox.

Operators are trusted Python objects. The multiprocessing control plane uses Python serialization internally, and child code executes with the privileges of the neurOS process environment.

The promoted guarantees are about:

- fault isolation;
- deterministic control/request identity;
- timeout/cancellation/crash containment;
- transport representation integrity;
- direct-child termination authority;
- resource ownership and provenance.

They are not guarantees against malicious operator code.

## Promotion checklist for a process node

Before selecting shared-memory process transport for a maintained path, require:

1. the payload type is in the canonical supported language;
2. request/response worst-case sizing is known and capacities are declared with margin;
3. hard execution timeout is finite, positive and operationally justified;
4. capacity overflow is expected to fail closed rather than resize/fallback;
5. downstream code does not assume zero-copy callback lifetime;
6. runtime snapshots/receipts are archived when the run is evidence-bearing;
7. cross-platform qualification covers the exact runtime semantics being promoted;
8. performance claims are scoped to the measured payload/workload and host class.
