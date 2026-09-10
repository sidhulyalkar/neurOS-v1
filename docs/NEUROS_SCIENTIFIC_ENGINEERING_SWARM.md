# neurOS scientific-engineering swarm

This layer turns the existing qualified NVIDIA NIM research transport into a parallel,
dissent-preserving engineering council. It is intentionally outside neurOS scientific evidence
and execution authority.

## Why this exists

The existing NIM tournament is a strong serial pipeline for proposal generation, adversarial
critique, and synthesis. The council adds a different capability: multiple independent reviewers
receive the same sealed task before any cross-agent synthesis. Their job is to surface distinct
failure modes, not to vote a claim into existence.

## v0 contract

`SealedSwarmTask` binds:

- repository and exact source revision;
- development objective;
- explicit scientific claim boundary;
- optional file allowlist;
- relevant authority SHA-256 identities;
- forbidden actions;
- public-only context.

The task identity is canonical and content-addressed. Public context rejects secret/private-data
keys such as credentials, API keys, raw participant data, participant identifiers, hidden targets,
and private leaderboard feedback.

Each `CouncilMember` binds a stable member ID, role, live-qualified model ID, and role prompt hash.
The NVIDIA adapter constructs five roles:

1. architecture;
2. scientific adversary;
3. reproducibility/evidence;
4. implementation;
5. experimental design.

The available models are supplied only after the existing `QualifiedNvidiaNimClient` has performed
its bounded provider/model qualification. Roles are distributed round-robin across the qualified
model set so heterogeneous routes are used when available.

## Independent fan-out

`run_council()` dispatches all members concurrently through a provider-neutral transport protocol.
Every member receives the same exact sealed task SHA and body. A member cannot see another
member's response through this API.

The NVIDIA transport reuses `QualifiedNvidiaNimClient.chat_json()` and therefore inherits the
existing HTTPS endpoint restriction, route qualification, call journaling, request/response
fingerprinting, and credential handling. No NVIDIA credential is accepted by the council module.

## Finding schema

Every finding must contain:

- stable finding ID;
- severity;
- category;
- claim;
- evidence;
- falsification test;
- proposed repair;
- confidence in `[0, 1]`;
- human-judgment flag;
- optional exact reference.

The parser rejects extra or missing finding fields. Finding IDs must be unique within one reviewer.

## Dissent is data

`finding_support()` records which reviewers independently emitted an exact finding ID. Minority
findings are preserved even if four other agents are silent. Agreement is metadata only.

The run manifest explicitly records:

- `majority_vote_is_authority = false`;
- `merge_authority = false`;
- `provider_execution_authority = false`;
- `scientific_promotion_authority = false`.

No number of agreeing LLMs can change those values.

## Failure behavior

By default the council fails closed if any required member fails. Optional partial mode can retain
successful reviews while preserving only bounded member/error-class identity in the council
manifest. Provider-specific diagnostics stay with the already-qualified transport and are not
copied into the provider-neutral manifest.

## CI and credentials

The council test lane is fully offline and does not require `NVIDIA_API_KEY`. It uses mock transport
objects to prove fan-out, deterministic identities, schema enforcement, failure preservation,
minority-finding retention, model assignment, and compatibility with the existing NIM client
interface.

A future live workflow may use the existing repository `NVIDIA_API_KEY` secret, but the live model
outputs remain advisory proposal material.

## Next qualification tranche

The next step is not more orchestration. It is evaluation.

Build a benchmark corpus from known neurOS defects and controlled mutations, including stale
exact-head evidence, leakage-prone splits, duplicate identities, environment drift, malformed
proofs, outcome-adaptive scheduling, missing dependencies, and scientific overclaiming. Compare:

- one qualified model;
- the existing serial tournament;
- homogeneous five-member council;
- heterogeneous five-member council.

Measure validated defect recall, precision, unique valid findings, false-positive rate, latency,
token cost, and marginal value of each additional member. The council should remain optional unless
it demonstrates measurable value over a single strong reviewer.

## Scientific boundary

This layer can propose tests, repairs, controls, and experiments. Only executable tests,
reproducible computation, bound data, and the existing neurOS evidence authorities can promote a
result. It cannot authorize the Kumar2024 T4 preflight, the production 810-shard fleet, ORION
comparison, or any scientific claim.

Tracks #178.
