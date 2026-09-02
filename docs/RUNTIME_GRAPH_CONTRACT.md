# RuntimeGraph Authority Contract

This document defines the promoted programmatic contract for constructing and validating neurOS runtime graphs.

The short version:

- node and edge identities are explicit strings, not values silently coerced into strings;
- node kinds are canonical `NodeKind` values before topology or execution rules inspect them;
- queue and shared-memory mailbox capacities are explicit positive integral authority;
- booleans, floats, text and non-positive values cannot masquerade as capacities;
- graph constructor containers are detached from caller-owned dict/list aliases;
- public graph containers remain intentionally mutable, so authoritative graph operations re-establish structural integrity before trusting them;
- structural integrity and semantic DAG validity are related but distinct contracts;
- `RuntimeExecutor` validates the graph before creating runtime channels, tasks or process workers;
- configuration/YAML authority is a separate compiler boundary and is not promoted by this contract.

## RuntimeNode identity and kind authority

`RuntimeNode.node_id` must be an explicit nonblank string.

The validator checks blankness without stripping or otherwise rewriting a valid identifier. For example, a deliberately supplied nonblank identifier containing leading/trailing spaces retains those exact string bytes. neurOS rejects malformed identity rather than silently manufacturing a different identity.

`RuntimeNode.kind` accepts either:

- a `NodeKind` value; or
- a valid plain string such as `"transform"`.

A valid string is immediately canonicalized to `NodeKind`. Unknown strings and non-string/non-enum values fail during construction.

This matters because runtime topology and dispatch use enum identity. A plain string must never survive construction and then fail later because code expects `node.kind.value` or uses `node.kind is NodeKind.TRANSFORM`.

Canonicalization also closes policy bypasses. For example, `kind="source"` is canonicalized to `NodeKind.SOURCE` before source execution restrictions are evaluated, so a string declaration cannot bypass the current inline-only source boundary.

## Executor and process-transport declarations

`executor` and `process_transport` require explicit strings before their values are checked.

The runtime currently recognizes executors:

- `inline`
- `thread`
- `process`
- `gpu`

and process transports:

- `pickle`
- `shared_memory`

These names do not all represent equivalent isolation guarantees.

In the current executor, `inline` and `gpu` both invoke operator code directly on the runtime execution thread. `thread` uses the thread execution domain and `process` uses the persistent process-worker authority. Therefore `gpu` is **not** a promoted process/thread isolation guarantee merely because it is a valid `RuntimeNode.executor` value.

Source lifecycle isolation is not implemented. Source nodes therefore require `executor="inline"`.

Process execution requires an explicit finite positive `execution_timeout_s`. Process-only transport declarations on non-process nodes fail closed.

For the detailed process lifecycle and payload contract, see [`RUNTIME_PROCESS_TRANSPORT.md`](RUNTIME_PROCESS_TRANSPORT.md).

## Integral capacity authority

Runtime capacities use one canonical integer rule.

A capacity must be:

1. an integral numeric scalar;
2. not a boolean;
3. strictly greater than zero.

Valid scientific-Python integral scalars such as NumPy integer types are accepted and stored as ordinary Python `int` values.

The following are rejected rather than coerced:

- `True` / `False`;
- floating-point values such as `8.0`;
- text such as `"8"`;
- zero;
- negative values;
- `None` and unsupported objects.

This rule applies to `RuntimeEdge.capacity` and to shared-memory request/response mailbox capacities declared through `RuntimeNode`.

Shared-memory mailbox validation preserves its transport-specific public failure semantics while using the same underlying integral authority. Both request and response capacities are required for `process_transport="shared_memory"`.

## RuntimeEdge canonicalization

`RuntimeEdge.source` and `RuntimeEdge.target` must be explicit nonblank strings. They are not coerced from integers, booleans or arbitrary objects.

A valid overflow declaration may be supplied as an `OverflowPolicy` or its valid string value. It is stored canonically as the policy's string value.

Self edges are rejected during `RuntimeEdge` construction.

Registered-endpoint and duplicate-edge authority belongs to `RuntimeGraph`, because those checks depend on graph state rather than one standalone edge.

## RuntimeGraph constructor ownership

`RuntimeGraph` requires:

- `nodes` to be a `dict[str, RuntimeNode]` container;
- `edges` to be a `list[RuntimeEdge]` container.

When caller-owned containers are supplied to the constructor, neurOS copies the dict/list containers before retaining them.

This is **container alias detachment**, not deep immutability.

Mutating the original constructor dict/list after graph construction cannot mutate the graph. The graph's own `nodes` and `edges` containers, however, remain intentionally public and mutable in this contract revision.

The reason for retaining that mutability is compatibility and incremental graph construction. The corresponding authority rule is that neurOS cannot cache a permanent "validated" bit while external mutation remains possible.

## Structural integrity

Authoritative graph operations establish structural integrity before trusting public containers.

Structural integrity requires:

- `nodes` remains a dict;
- `edges` remains a list;
- every node key is a nonblank string;
- every node value is a `RuntimeNode`;
- every node dict key exactly equals the contained `RuntimeNode.node_id`;
- every edge value is a `RuntimeEdge`;
- no duplicate `(source, target)` edge exists;
- every edge endpoint refers to a registered node.

These checks protect against direct public-container mutation as well as ordinary method calls.

For example, injecting an arbitrary object into `graph.edges` and then calling `topological_order()`, `incoming()`, `outgoing()`, `add_node()`, `connect()` or `validate()` fails with a graph-contract error before traversal or mutation proceeds. It must not leak an incidental `AttributeError` from deeper implementation code.

Likewise, replacing `graph.nodes` or `graph.edges` themselves with incompatible container types is detected before the graph operation continues.

## Structural integrity vs semantic graph validity

Structural integrity answers:

> Is this object a coherent graph representation whose identities and endpoints can be trusted?

Full `RuntimeGraph.validate()` additionally answers:

> Does this coherent representation satisfy neurOS runtime topology semantics?

Semantic validation includes:

- the graph is acyclic;
- source nodes have no incoming edges;
- fusion nodes have at least two inputs;
- transforms, decoders and sinks have exactly one input;
- sinks have no outgoing edges;
- monitors are observational and own no data edges.

Public topology/query operations require structural integrity, but they do not all imply every semantic node-cardinality rule. `validate()` is the complete runtime-graph acceptance gate.

## Execution handoff

`RuntimeExecutor.__init__` calls `graph.validate()` before it builds runtime channels, queues, process-worker receipt state or execution tasks.

Therefore a malformed or externally corrupted graph is rejected before runtime execution authority is created.

This is the key fail-early boundary promoted by this contract: invalid programmatic graph declarations should not survive until an operator callback, queue operation or child process exposes the defect.

## Configuration compiler boundary

This contract governs the **programmatic runtime graph**.

The current `PipelineConfig -> resolve_config -> RuntimeGraph` path has its own versioned schema/parser responsibilities. Runtime graph hardening does not make YAML parsing automatically strict, and it does not authorize configuration fields that the compiler does not expose.

In particular, legacy configuration parsing still has coercion/versioning questions that must be resolved as a separate configuration-authority tranche.

The intended dependency direction is:

1. configuration parses and validates its declared schema;
2. the compiler deterministically translates that declaration to `RuntimeNode` / `RuntimeEdge` values;
3. the runtime graph constructors remain the final programmatic authority;
4. `RuntimeGraph.validate()` proves the resolved graph before execution.

The configuration layer must not invent a second, weaker definition of strings, numeric durations or integral capacities.

## Deliberate non-goals

This contract does not claim:

- graph metadata is recursively immutable;
- operator objects are immutable or trusted against malicious code;
- `gpu` is a distinct process/thread/device isolation domain;
- YAML/configuration parsing is no-coercion authoritative;
- runtime queue capacity is automatically sized;
- a structurally valid graph is scientifically valid;
- graph validation proves hardware, closed-loop, safety or clinical behavior.

Metadata provenance, configuration execution policy, process cleanup-history telemetry and scientific/model contracts have separate authority surfaces.

## Qualification boundary

The promoted graph authority is exercised in `tests/test_runtime_graph_authority.py` and bound into Runtime Fault Qualification across:

- Ubuntu 24.04;
- macOS 14;
- Windows 2025;
- Python 3.10, 3.11 and 3.12.

The suite includes malformed identities, enum declarations, capacities, constructor aliases, public-container corruption, topology queries, mutation methods and shared-memory capacity normalization.

Cross-platform qualification proves the declared software contract on the maintained runtime matrix. It does not convert that contract into a scientific-performance or real-time guarantee.
