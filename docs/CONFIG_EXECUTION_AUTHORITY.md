# Configuration Execution Authority

This document defines the promoted configuration contract for declaring runtime execution policy in neurOS.

The contract is intentionally narrow. It describes how a serialized pipeline configuration becomes a validated `RuntimeGraph`, which execution declarations the runtime can currently honor, and how the accepted policy is represented in runtime evidence. It does not turn configuration into a security sandbox, a hardware qualification, or a scientific-validity claim.

## Authority boundary

The authoritative path is:

```text
YAML / mapping
    -> PipelineConfig
    -> PluginConfig + ExecutionConfig
    -> resolve_config(...)
    -> RuntimeNode
    -> RuntimeExecutor
    -> snapshot()["execution"]
```

Each stage must preserve or reject execution policy explicitly. neurOS does not silently reinterpret an unsupported execution declaration as a weaker execution mode.

`RuntimeNode` remains the final programmatic authority for node execution semantics. Configuration deliberately reuses that authority instead of maintaining a second normalization system.

## Schema versions

### Schema version 1

Schema v1 is the legacy compatibility contract.

Valid v1 configurations continue to resolve with the historical inline execution defaults. A v1 plugin configuration may not contain an `execution:` declaration, even if that declaration only requests `inline`.

This prohibition is deliberate. Older neurOS readers may ignore unknown plugin keys. If execution authority were added to schema v1, an older reader could silently discard a newer isolation declaration and execute the operator inline. A schema-version boundary prevents that downgrade from looking valid.

### Schema version 2

Schema v2 is the first serialized configuration version that can declare per-plugin execution policy.

For v2, unknown keys fail closed at the following authority-bearing layers:

- configuration root;
- stream declaration;
- plugin declaration;
- runtime declaration;
- execution declaration.

Schema v2 also rejects ambiguous coercions for authority-bearing values. For example, text is not converted to an integer queue capacity and a numeric stream ID is not converted to a string identity.

## Execution declaration

A v2 plugin may contain:

```yaml
execution:
  executor: process
  execution_timeout_s: 2.0
  process_transport: shared_memory
  process_request_capacity_bytes: 4194304
  process_response_capacity_bytes: 4194304
```

The complete promoted `ExecutionConfig` surface is:

- `executor`;
- `execution_timeout_s`;
- `process_transport`;
- `process_request_capacity_bytes`;
- `process_response_capacity_bytes`.

These field names intentionally match `RuntimeNode` rather than introducing configuration-only aliases.

## Executor classes

### `inline`

The callback executes in the runtime event-loop domain.

This is the default execution class.

### `thread`

The callback executes in a worker thread through the runtime's thread-dispatch authority.

Synchronous and asynchronous callbacks are both supported by the qualified runtime path. Async callbacks execute inside the worker thread rather than being bounced back to the main runtime event loop.

### `process`

The callback executes through a persistent direct child process owned by the `RuntimeExecutor` for that node.

Process execution requires an explicit finite positive `execution_timeout_s`.

The timeout is hard execution-containment authority. It is distinct from latency telemetry or a latency service-level objective. Expiry causes the process execution path to fail closed and invoke the worker termination authority rather than merely recording that the callback was slow.

neurOS does not automatically retry scientific operator calls after a process failure.

### `gpu`

`gpu` currently preserves scheduling intent but does **not** create a distinct GPU isolation domain.

The callback is invoked in the event-loop execution domain and the operator or framework owns its device/context behavior.

Runtime evidence therefore reports both facts:

```text
requested_executor: gpu
execution_domain: event_loop
```

A `gpu` declaration must not be cited as proof of process isolation, thread isolation, device placement, CUDA-stream isolation, or GPU fault containment.

## Process transports

### `pickle`

`pickle` is the default process transport.

Process requests and results cross the direct parent/child boundary using the maintained process-worker protocol.

### `shared_memory`

`shared_memory` uses explicitly bounded shared-memory mailboxes for process request and response payloads.

A shared-memory process declaration requires both:

- `process_request_capacity_bytes`;
- `process_response_capacity_bytes`.

Both capacities must be genuine positive integral scalars. Boolean, floating-point, textual, zero, and negative values fail closed. Valid scientific-Python integral scalars are normalized to ordinary Python `int` values.

Mailbox capacity is protocol authority, not an adaptive allocation hint. Oversized payloads fail rather than silently expanding the declared mailbox.

## Plugin-role semantics

### Sources

Configured sources are currently inline-only.

Source lifecycle owns `start()`, async frame iteration, and `stop()`. neurOS does not yet provide a separate source lifecycle isolation domain, so schema v2 rejects non-default execution policy for source plugins.

This is an explicit limitation rather than an implicit fallback.

### Transforms

Transforms can use the runtime execution classes supported by `RuntimeNode`, including thread and process execution.

### Decoders

Decoders can use the same runtime execution authority. Process-backed decoder inference therefore uses the qualified persistent per-node worker path and explicit hard timeout semantics.

### Sinks

Sink `write(...)` callbacks are dispatched through the declared execution authority.

### Monitors

Monitors are executable observational callbacks, not independent graph data-plane tasks.

When the runtime emits an item, the executor routes the observation through its monitor notification path and invokes `monitor.update(payload)` using the monitor node's declared execution authority.

A monitor may therefore validly request inline, thread, process, or GPU-intent execution subject to the same runtime execution rules.

Runtime evidence reports monitor scheduling as:

```text
scheduling_mode: observation_callback
```

A monitor owns no data edges and does not acquire data-plane routing authority merely because its callback executes in another domain.

## Runtime-owned fusion

Configuration-created implicit fusion nodes are runtime-owned and use the runtime default policy.

For direct programmatic `RuntimeGraph` construction, a fusion node can currently carry an execution label even when it has no custom `fuse()` operator. The built-in concatenate-latest fusion path executes on the event loop in that case.

The resolved execution evidence reports the effective domain truthfully rather than converting the requested label into a false isolation claim. Broader executor protection against post-construction graph drift and unhonored direct fusion declarations is tracked separately in issue #140.

## Runtime configuration

`RuntimeConfig.queue_capacity` and `RuntimeConfig.overflow_policy` reuse `RuntimeEdge` canonicalization.

That means configuration and direct runtime graph construction share the same bounded-queue authority instead of implementing parallel definitions of valid capacity and overflow behavior.

## Strict identity and scalar handling

Authority-bearing configuration values are not normalized through convenience coercions.

Examples:

- `schema_version` must be an integral scalar and must be a supported version;
- booleans are not accepted as integers;
- stream IDs must already be explicit nonblank strings;
- plugin IDs must already be explicit nonblank strings;
- queue and mailbox capacities must already satisfy exact positive-integral semantics;
- process timeout must satisfy the runtime's finite-positive-real contract.

This is intended to make a configuration's meaning inspectable before execution rather than dependent on Python coercion behavior.

## Example v2 configuration

```yaml
schema_version: 2

streams:
  - id: eeg
    source:
      plugin: my-eeg-source
    transforms:
      - plugin: preprocess
        execution:
          executor: thread

decoder:
  plugin: my-decoder
  execution:
    executor: process
    execution_timeout_s: 2.0
    process_transport: shared_memory
    process_request_capacity_bytes: 4194304
    process_response_capacity_bytes: 4194304

sinks:
  - plugin: archive-sink
    execution:
      executor: inline

monitors:
  - plugin: quality-monitor
    execution:
      executor: thread

runtime:
  queue_capacity: 128
  overflow_policy: drop_oldest
```

Plugin names in this example are illustrative. Configuration validity does not imply that a named plugin is installed or qualified for a particular device, model, operating system, dataset, or scientific use.

## Resolved execution evidence

`RuntimeExecutor.snapshot()` exposes an all-node `execution` section captured from the validated graph accepted by the executor constructor.

Each node record contains:

- `kind`;
- `requested_executor`;
- `execution_domain`;
- `scheduling_mode`;
- `execution_timeout_s`;
- `process_transport`;
- `process_request_capacity_bytes`;
- `process_response_capacity_bytes`.

Execution domains currently include:

```text
event_loop
worker_thread
persistent_process
```

Scheduling modes currently include:

```text
source_task
unary_task
fusion_task
observation_callback
```

The manifest describes **accepted execution authority**, not observed activity. A node may have accepted policy even if a particular run processes zero items. Actual activity and failure information remain in node telemetry, process receipts, and runtime failure records.

## Evidence capture point

The all-node execution manifest is captured immediately after `RuntimeExecutor` validates the supplied graph.

This avoids recomputing the manifest later from externally mutable `RuntimeGraph.nodes` or `RuntimeGraph.edges` containers.

The snapshot returns a fresh dictionary representation so mutating one returned snapshot does not mutate the executor's internally captured authority record.

This guarantee is narrower than full executor graph immutability. `RuntimeExecutor` still retains the public mutable graph object. Issue #140 owns the broader requirement that external post-construction graph mutation cannot make actual execution drift from the graph accepted at construction.

## Process-specific compatibility surface

The existing `snapshot()["process_execution"]` section remains available for backward compatibility.

The new `execution` section is additive and all-node. It does not replace process semantic receipts, process cleanup telemetry, or the existing process-specific execution view.

## Recording and replay

The maintained recording path persists the complete runtime snapshot as runtime metrics. The replay path returns the normal executor snapshot.

Recording graph decoration replaces source operators while retaining the frozen runtime node fields, so accepted execution declarations survive that decoration path.

As a result, the resolved `execution` manifest participates in the existing recording/provenance surface without a second execution-policy serialization format.

This does not imply that a replay uses the same hardware, operating-system scheduling, process identifiers, or wall-clock timing as the original run. The execution manifest deliberately excludes host-, PID-, and wall-clock-specific operational metadata.

## Failure semantics

An invalid execution declaration fails before runtime execution authority is created.

A valid process declaration may still fail at execution time because of operator serialization, request/result serialization, protocol mismatch, child crash, timeout, cancellation, mailbox capacity, cleanup, or operator exceptions. Those failures remain governed by the qualified process-worker contracts.

The configuration layer does not convert such failures into retries or alternate execution modes.

## Claim boundaries

This contract establishes configuration and runtime execution semantics. It does **not** establish:

- a security sandbox for untrusted Python;
- containment of arbitrary descendants created by user code;
- hardware or driver qualification;
- GPU device placement or GPU isolation;
- real-time operating-system guarantees;
- clinical safety;
- model correctness;
- scientific validity;
- causal or biological interpretation;
- reproducibility of numerical results across arbitrary hardware/software environments.

Process containment authority applies to executor-owned direct child processes under the maintained worker contract.

## Qualification

The promoted configuration/runtime authority surface is bound to Runtime Fault Qualification across:

- Ubuntu 24.04;
- macOS 14;
- Windows 2025;
- Python 3.10;
- Python 3.11;
- Python 3.12.

The matrix exercises configuration authority together with graph, queue, process, transport, provenance, duration, and DecoderOutput fault contracts. Broader repository qualification additionally covers installed-wheel clean rooms, compatibility, recording, BCI smoke, release artifacts, and workspace build ownership.

A green workflow on one historical commit is not evidence for a later changed head. Qualification claims must name the exact source revision that produced them.

## Related contracts

- `docs/RUNTIME_GRAPH_CONTRACT.md` defines programmatic graph identity, topology, and structural trust.
- `docs/RUNTIME_PROCESS_TRANSPORT.md` defines process transport and containment semantics.
- `docs/DECODER_OUTPUT_CONTRACT.md` defines DecoderOutput value authority.
- issue #140 tracks post-construction executor graph drift and direct fusion execution-authority reconciliation.
