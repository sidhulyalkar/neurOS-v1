# NSQ Kumar2024 GPU fleet Merkle roster authority

Status: **distribution-integrity software only. This does not authorize a provider, GPU execution, scientific interpretation, or the production fleet.**

This tranche adds a compact cryptographic inclusion layer above the promoted Kumar2024 GPU fleet lease authority. A provider worker can verify that one exact `LeaseSpec` belongs to the preregistered fleet without trusting mutable scheduler state or receiving the entire 810-lease manifest.

## Authority boundary

The implementation is deliberately additive. It does not modify `kumar2024_gpu_fleet.py`, the scientific execution graph, model code, seeds, calibration frontier, provider choice, retry semantics, settlement semantics, or score-handling boundary.

`FleetRosterAuthority` binds:

- fleet-authority SHA-256;
- exact expected lease count;
- ordered-leaf commitment SHA-256;
- Merkle algorithm and odd-node policy;
- Merkle root SHA-256;
- GPU binding SHA-256;
- execution-plan SHA-256;
- environment-authority SHA-256;
- fleet retry ceiling;
- fixed EEGNet method identifier;
- fixed `(0,1,2,5,10)` calibration frontier;
- explicit negative claim semantics for execution, efficacy, completeness, and ORION readiness.

The ordered-leaf commitment hashes the complete sequence of domain-separated leaf hashes. Each leaf is derived from the exact canonical `LeaseSpec` SHA-256, not from a reduced `(subject, session, seed)` tuple or provider metadata.

## Frozen Merkle v1

The algorithm is named:

`sha256-canonical-json-domain-separated-duplicate-last-v1`

Rules:

1. SHA-256 only.
2. Canonical JSON uses sorted keys, compact separators, ASCII escaping, and rejects NaN/Infinity.
3. Lease identity uses the already-promoted `neuros.nsq_kumar2024_gpu_fleet_lease.v1` domain.
4. Leaf identity uses `neuros.nsq_kumar2024_gpu_fleet_roster_leaf.v1`.
5. Internal nodes use `neuros.nsq_kumar2024_gpu_fleet_roster_node.v1` and commit ordered `left_sha256` / `right_sha256` children.
6. Leaves are ordered only by the existing canonical lease ordinal.
7. Odd levels duplicate the final unpaired node.
8. The empty tree is invalid.
9. A singleton root equals its one domain-separated leaf hash.
10. Proof verification never sorts nodes. It checks the explicit leaf index, expected sibling side at every level, proof depth, and duplicate-last behavior.

This removes implementation-dependent Merkle conventions from the evidence surface.

## Proof-carrying lease

`build_fleet_roster(...)` consumes complete canonical `LeaseSpec.to_dict()` records plus the independently supplied frozen fleet bindings. It returns one small roster authority and one `ProofCarryingLease` per canonical ordinal.

A dispatch package contains only:

- the canonical lease record;
- its lease SHA-256;
- roster-authority SHA-256;
- leaf index;
- ordered authentication path.

The roster authority is a small content-addressed header. A worker that resolves that header by SHA can call `verify_proof_carrying_lease(...)` before provider/model invocation. Verification independently performs:

`LeaseSpec payload -> lease SHA -> leaf SHA -> ordered authentication path -> roster root`

It also checks fleet authority, GPU binding, execution plan, environment authority, retry ceiling, method, and calibration frontier against the roster header.

## Adversarial qualification

The test tranche rejects:

- altered lease payload with unchanged proof;
- altered lease ordinal;
- swapped proof order;
- flipped sibling side;
- foreign sibling hash;
- proof from another roster;
- wrong environment authority;
- wrong execution plan;
- wrong GPU binding;
- wrong retry ceiling;
- duplicate lease identity;
- duplicate shard identity;
- missing/extra leaves;
- non-contiguous ordinals;
- scientific-result contamination;
- a proof that could only work after implicit reordering;
- invalid duplicate-last behavior.

It also requires differently ordered input iterables to produce byte-identical canonical authority and proof packages.

A synthetic production-shaped qualification constructs all **810** expected axis combinations using the frozen shape only:

- 18 subjects;
- sessions `1` through `5`;
- split seeds `2026`, `3407`, `9109`;
- EEGNet model seeds `31415`, `384165836`, `3991196546`;
- complete `(0,1,2,5,10)` frontier per lease.

The shard hashes used by that test are synthetic test identities. The test does not materialize neural data, execute EEGNet, or claim a production roster root.

Pre-publication local construction evidence: **23/23 tests passed**. CI must independently requalify the exact PR head on Python 3.10, 3.11, and 3.12; local evidence is not promotion evidence.

## Scientific firewall

The roster implementation is standard-library only and imports neither the fleet scheduler nor any neural/provider stack. CI AST-audits the import boundary.

Scientific outcome keys are rejected recursively at the roster boundary. Membership means only:

> this exact immutable assignment is one leaf committed by this exact roster root.

It does **not** mean the provider ran, an artifact exists, the fleet settled completely, the model worked well, an external floor is valid, or ORION comparison is permitted.

## Gate sequence

```text
promoted provider-free fleet + independent replay
        |
        v
Merkle roster authority + proof-carrying leases (this tranche)
        |
        v
real fixed T4 systems preflight
        |
        v
systems-only preflight admission
        |
        v
tiny live provider transport qualification
        |
        v
complete settlement + explicit production fleet authorization
```

Issue #165 remains the live-execution authority boundary. Issue #175 owns this distribution-integrity tranche.
