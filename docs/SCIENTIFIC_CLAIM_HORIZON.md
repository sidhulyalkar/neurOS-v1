# Scientific Claim Horizon

## North star

neurOS should make a resulting scientific claim auditable.

The long-term object is not a paper, model, notebook, or workflow. It is a
machine-readable scientific proposition bound to the protocol, authority,
execution, evidence, failures, and attestations that determine what the
proposition is allowed to mean.

```text
human / AI agent / autonomous lab
              |
              v
      ScientificClaimSpec
              |
       protocol authority
              |
              v
 existing Scientific Authority / NSQ
              |
              v
     immutable evidence
              |
              v
       claim qualification
              |
              +--> paper / report
              +--> reviewer
              +--> external reproducer
              +--> future machine auditor
```

This document is a horizon architecture, not permission to expand the active
implementation surface. The current scientific priority remains the first
frozen real-data NSQ result in #82.

## Why the abstraction changes

Traditional scientific software organizes around datasets, models, runs, and
metrics. Those remain implementation objects. The durable public object should
instead answer:

- what proposition is being evaluated?
- what is its intended population/scope and inferential unit?
- what authority must exist before the evidence can support it?
- what evidence objects support, contradict, replicate, or contextualize it?
- what evidence tier was actually earned?
- what remains unknown?
- which protocol and study identities produced the cited evidence?

As AI systems perform larger portions of hypothesis generation, code writing,
experiment search, analysis, and reporting, reproducibility of the final run is
not enough. Selection history, outcome access, adaptation budgets, and
prospective decisions increasingly determine whether a result is interpretable.

## v1 implemented surface

`neuros.evidence` defines a dependency-light proposition facade:

### `ScientificClaimSpec`

Content-addresses:

- `claim_id`;
- natural-language `statement`;
- namespaced/freeform `domain`;
- declared `scope`;
- independent `inference_unit`;
- the **target** evidence tier the future protocol intends to earn;
- explicit `EvidenceRequirement` objects;
- deterministic metadata.

`target_evidence_tier` is an intention, not a qualification. A real-data claim
spec does not become real-data evidence merely because it requests that tier.

### `EvidenceRequirement`

Declares one requirement in terms of:

- a stable requirement ID;
- an authority type such as `LongitudinalCaseAuthority`,
  `FailurePreservingResultSet`, `DatasetLineage`, or future authority types;
- a human-readable description;
- the evidence tier at which the requirement applies;
- whether the requirement is mandatory.

The v1 facade does not instantiate or validate those external authority objects.
Existing Scientific Authority / NSQ remains the enforcement system.

### `ClaimEvidenceRef`

Binds one content-addressed evidence object to the claim with a relation:

- `supports`;
- `contradicts`;
- `replicates`;
- `context`.

Each reference carries its evidence tier explicitly. The claim layer does not
infer a stronger tier from the presence of evidence.

### `ScientificClaimBundle`

Binds:

- the exact claim identity;
- one or more exact evidence identities;
- an optional protocol SHA-256;
- zero or more scientific-study SHA-256 identities.

It deliberately has no `truth`, `confidence`, or inferred
`achieved_evidence_tier` field.

## Relationship to existing ORION Scientific Authority

This layer must not duplicate the already-qualified Scientific Authority.

```text
ScientificClaimSpec
       |
       | proposition + requirements
       v
ScientificStudyAuthority
       |
       | executes and validates lineage / roles / metrics / failures
       v
EvidenceClaim
       |
       | earned qualification inside the study
       v
ClaimEvidenceRef
       |
       v
ScientificClaimBundle
```

ORION's existing `EvidenceClaim` remains the authority for study-local
qualification. `ScientificClaimSpec` is the public proposition above it.

A future bridge may compile or validate a `ScientificClaimSpec` against a
`ScientificStudyAuthority`, but it must reuse the existing authority objects and
recompute their identities rather than accepting manually asserted verdicts.

## Evidence tiers

The public claim layer uses the same maturity language already used by neurOS:

```text
software_contract
        |
integration
        |
replay_or_synthetic
        |
real_data
        |
physical_hardware
        |
closed_loop
        |
clinical
```

The Python enum intentionally does **not** implement ordering or automatic
promotion. Tier transitions must be earned by explicit qualification policy.

## Horizon plan

### 0-1 year: claims become first-class

Primary proof remains #82.

After the Kumar2024 baseline is frozen:

1. emit a `ScientificClaimSpec` for the flagship calibration-frontier question;
2. bind the exact NSQ study/result identities into a `ScientificClaimBundle`;
3. export the bundle beside the existing study artifact;
4. make a reviewer able to answer "what exactly is this result evidence for?"
   without reading implementation code;
5. add machine-readable mappings from requirements to the authority objects that
   satisfied or failed them.

Do not add another benchmark runner or model framework for this phase.

### 1-3 years: prospective and agentic science

Add only after a real study exposes the need:

#### `ProtocolSeal`

A content-addressed declaration that exists before protected outcomes are
available. Candidate fields include:

- hypotheses/claims;
- observation-role policy;
- inclusion/exclusion rules;
- metrics;
- calibration/adaptation budgets;
- model-selection/search policy;
- stopping rules;
- permitted outcome access;
- protocol creation/seal time and issuer identity.

The key property is temporal authority, not secrecy.

#### `OutcomeRevealReceipt`

Records the transition at which protected final outcomes became accessible to
the research process.

#### `ExplorationLedger`

Captures externally observable scientific decisions made during large
human/agent search:

- hypothesis/experiment proposals;
- tool executions;
- parameter/config identities;
- outcomes made available to the agent;
- selection/rejection decisions;
- promotion/stopping events;
- human approvals where required.

It must capture actions and decision provenance, not private model
chain-of-thought.

### 3-5 years: evidence becomes federated and graph-shaped

Add cross-site and cross-study objects only once there are independent
reproducers/labs.

Candidate primitives:

- `SiteEvidenceReceipt`;
- `ReproductionReceipt`;
- `EvidenceGraphEdge`;
- `ClaimRelation` such as supports/contradicts/replicates/refines;
- external signatures/attestations over immutable evidence identities.

Prefer interoperability mappings to PROV, RO-Crate, NWB/BIDS, DANDI, and
external registries over a neurOS-specific replacement ontology.

The goal is to answer questions such as:

- which independent studies support this claim?
- which apparently independent studies share dataset or pretraining lineage?
- which replication changed population, device, or preprocessing authority?
- which evidence is unavailable because a site failed rather than because the
  row disappeared from an aggregate?

### 5-10 years: autonomous experimental systems

When software can choose consequential physical actions, extend authority rather
than replacing it.

Candidate objects:

- `AgentIdentity`;
- `ActionAuthority`;
- `AgentActionReceipt`;
- `DeviceAuthority`;
- `SafetyEnvelope`;
- `EmergencyStopReceipt`;
- `ClosedLoopSessionAuthority`.

The scientific system should be able to prove what an autonomous agent was
allowed to modify and whether every action remained inside the predeclared
experimental/safety envelope.

This future work is downstream of physical-device and closed-loop qualification.
Software contracts alone must never self-promote into safety or clinical claims.

## Design invariants

1. **Claims outrank workflows.** New subsystems must identify which claim
   property they protect.
2. **Unknown is first-class.** Missing lineage or failed execution never becomes
   implied cleanliness.
3. **Failures are evidence.** Failure rows remain present in bundles and graphs.
4. **Authority precedes protected observation.** Outcome-sensitive decisions
   should be sealable before outcome reveal where scientifically required.
5. **Capture actions, not hidden reasoning.** Agent auditability is based on
   observable proposals, tool calls, data access, decisions, and approvals.
6. **Standards at the boundary.** Integrate existing research/provenance formats
   rather than creating another universal data format.
7. **No truth oracle.** neurOS qualifies support under declared conditions. It
   does not pronounce a proposition universally true.
8. **No automatic evidence promotion.** A higher-tier claim target and a
   lower-tier cited artifact remain visibly different.
9. **No new package without earned need.** Horizon primitives should begin in
   the public `neuros.evidence` facade and split only if external use proves a
   separate distribution is necessary.
10. **The runtime stays boring.** Claim evolution is not a reason to reopen the
    exact-clock/data-plane layer absent an external execution requirement.

## Immediate implementation sequence

1. Land `ScientificClaimSpec v1` as a dependency-light public evidence contract.
2. Keep #82 as the dominant scientific workstream.
3. Once #82 has a frozen result, make Kumar2024 the first real claim bundle.
4. Recruit an external reproducer and preserve their failures as evidence.
5. Design `ProtocolSeal` only after the first real claim bundle shows which
   prospective fields are actually needed.
6. Design `ExplorationLedger` against one concrete agent-driven study, not an
   abstract agent framework.

## Non-goals

Do not use this horizon to justify:

- a generic autonomous-scientist framework;
- a new notebook/runtime platform;
- a competing provenance ontology;
- a new data archive or file format;
- a model zoo;
- an agent chain-of-thought recorder;
- a blockchain;
- automatic scientific truth scoring;
- UI/cloud expansion before the claim/evidence contract has external users.

The strategic bet is narrower:

> As generating experiments becomes cheaper, trustworthy scientific evidence
> becomes more scarce. neurOS should make the boundary between result and claim
> explicit, executable, and independently auditable.
