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
and private leaderboard feedback. The stored context is recursively immutable; transport
serialization returns a detached ordinary JSON copy.

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
minority-finding retention, model assignment, benchmark scoring, and compatibility with the
existing NIM client interface.

A future live workflow may use the existing repository `NVIDIA_API_KEY` secret, but live model
outputs remain advisory proposal material.

## Deterministic defect benchmark

The benchmark is deliberately split into two surfaces:

- `neuros.research.swarm_benchmark_cases` contains reviewer-visible taxonomy and stimuli, the
  committed scorer-corpus SHA-256, public manifest generation, and sealed task construction;
- `neuros.research.swarm_benchmark` owns scorer-side ground truth, metrics, and report comparison.

A live reviewer can therefore construct every sealed benchmark task without importing the module
that contains the answer key. The scorer independently derives the complete labeled-corpus digest,
and tests require it to equal the public commitment. The v1 corpus contains:

- 29 opaque cases;
- 20 fixed defect classes;
- 5 clean negative controls;
- 3 compound-defect cases;
- a second representation of an import-shadowing failure to reduce single-template memorization.

Cases are based on failure modes that matter to neurOS development: stale exact-head evidence,
participant leakage, missing participant aggregation, learned-state substitution, environment
drift, duplicate JSON keys, shadow files, CI dependency omissions, missing/duplicate fleet
identities, changed frozen seeds, outcome-adaptive scheduling, target leakage, held-out preprocessing
fit, incomplete provenance, nondeterminism, unit-of-analysis mismatch, retry-ceiling violations,
unbound artifacts, and scientific overclaiming.

### Anti-leakage contract

Reviewer-visible case IDs are opaque (`case-001`, `case-002`, ...). The task exposes the fixed
taxonomy and stimulus but never the scorer-side expected labels. The full scorer corpus, including
labels, is content-addressed by SHA-256; the public module carries only that digest and never imports
the scorer. The public task carries the digest without revealing the answers.

This public regression corpus is designed for immediate controlled evaluation, not permanent
hidden-holdout claims. Future long-lived model comparison should add a separately controlled
holdout or mutation corpus so benchmark familiarity cannot masquerade as reasoning quality.

A benchmark run is rejected if:

- a model emits a finding ID outside the 20-label taxonomy;
- a result is scored against the wrong case;
- the task binds a different source revision;
- a supplied `BenchmarkCase` payload differs from the frozen corpus;
- a reviewer is simultaneously represented as successful and failed;
- a failed member identity appears more than once under different error classes.

### Metrics

`score_benchmark()` computes deterministic council-level:

- true positives, false positives, and false negatives;
- precision, recall, and F1;
- exact-case accuracy;
- false-positive rate on clean controls;
- failed member calls;
- mean pairwise Jaccard disagreement.

It also reports each member's precision, recall, F1, reviewed/failed case count, and the number of
valid defects uniquely contributed by that member. This is important because a five-agent council
should not receive credit for diversity if one strong reviewer finds everything while the others
only add false alarms.

`compare_benchmark_reports()` compares two reports only when corpus, source identity, and evaluated
case slice match exactly. It reports metric deltas but explicitly grants no scientific authority.

## Next live evaluation tranche

The measurement apparatus must qualify before any model-performance claim is made. After this
benchmark is promoted, run the same exact corpus through:

1. one qualified strong NIM reviewer;
2. a homogeneous five-member council using one model family;
3. a heterogeneous five-member council using independently selected qualified models;
4. optionally the existing serial research tournament as a separate architecture.

For each configuration, preserve exact model IDs, prompt identities, call journal identities,
latency, token usage when available, and benchmark report SHA. Compare defect recall and precision
first, then operational cost and latency. Measure marginal gain from each additional reviewer rather
than assuming five agents are better than one.

The council should remain optional unless the heterogeneous configuration demonstrates repeatable
value over a single strong reviewer without an unacceptable false-positive or cost penalty.

## Scientific boundary

This layer can propose tests, repairs, controls, and experiments. Benchmark scores measure reviewer
performance only. They are not neuroscience results. Only executable tests, reproducible
computation, bound data, and the existing neurOS evidence authorities can promote a scientific
result. The swarm cannot authorize the Kumar2024 T4 preflight, the production 810-shard fleet,
ORION comparison, or any scientific claim.

Tracks #178.
