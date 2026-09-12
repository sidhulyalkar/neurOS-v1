# Scientific claim horizon

Status: **architecture contract, not a scientific result.**

As generating experiments becomes cheaper, trustworthy evidence becomes more scarce. neurOS should make the boundary between a result and a claim explicit, executable, portable, and independently auditable.

The public claim layer therefore begins with a narrow proposition contract rather than a generic autonomous-scientist framework.

## The core separation

```text
claim intent
    |
    v
ScientificClaimSpec
    |
    | proposition + inference unit + declared evidence requirements
    v
protocol / ScientificStudyAuthority / NSQ
    |
    | qualified execution and study evidence
    v
ORION EvidenceClaim and immutable receipts
    |
    | earned authority remains study-local
    v
ClaimEvidenceRef
    |
    | portable citation to immutable evidence
    v
ScientificClaimBundle
    |
    v
human or separately authorized interpretation
```

The invariant is intentionally strict:

```text
claim intent != earned qualification != evidence relation != truth
```

A claim targeting real-data evidence has not earned real-data status. A citation declared to support a claim is not automatically valid support. A structurally qualified execution receipt is not an efficacy result. A collection of supportive artifacts is not a truth score.

## v1 public contracts

`neuros.evidence` exposes six dependency-light contracts:

- `EvidenceTier`: labels the evidence horizon without defining an automatic ordering or promotion rule.
- `EvidenceRelation`: records a declared relationship such as support, contradiction, replication, or context.
- `EvidenceRequirement`: predeclares one evidence requirement and the authority type expected to satisfy it.
- `ScientificClaimSpec`: identifies the proposition, scope, inference unit, target evidence horizon, and requirements.
- `ClaimEvidenceRef`: binds the claim layer to one immutable evidence digest and explicitly calls its tier `declared_evidence_tier`.
- `ScientificClaimBundle`: content-addresses one claim plus its cited evidence, protocol identity, and study identities.

These objects are manifests, not adjudicators.

## Identity semantics

Claim portability requires two labs, agents, or audit tools to derive the same identity from semantically identical content. v1 therefore treats several collections as set-like authority surfaces:

- requirements are canonicalized by `requirement_id`;
- evidence references are canonicalized by reference identity;
- study digests are canonicalized lexicographically;
- mapping keys are sorted and must be strings;
- nested metadata is frozen after validation;
- unordered Python sets are rejected;
- NaN and infinity are rejected;
- negative zero canonicalizes to zero;
- SHA-256 identities are domain-separated by manifest type.

The same underlying `evidence_sha256` may occur only once in a claim bundle. Changing relation labels, source aliases, or metadata cannot be used to count one artifact repeatedly.

Ordered sequences inside metadata remain ordered because their order may itself be scientific content.

## Evidence tiers are not a ladder implemented in code

v1 names useful strata:

```text
software_contract
integration
replay_or_synthetic
real_data
physical_hardware
closed_loop
clinical
```

The enum deliberately does not define `software_contract < real_data < clinical` as an automatic comparison operator. Different claims require different authority graphs, and a nominally "higher" environment does not repair a flawed inference unit, invalid split, missing control, or outcome-dependent retry.

Promotion belongs to explicit scientific policy, not enum arithmetic.

## Relation to ORION

ORION should remain the authority engine that records what an actual study earned. `ScientificClaimSpec` sits upstream and says what proposition is being pursued. `ClaimEvidenceRef` sits downstream and provides a portable citation boundary.

This avoids two failure modes:

1. forcing the public neurOS SDK to import ORION just to describe a proposition;
2. allowing a lightweight manifest to impersonate ORION qualification.

A future bridge may validate an ORION `EvidenceClaim`, `ScientificStudyAuthority`, protocol seal, or outcome receipt and then create a bound claim-evidence reference. That bridge should prove the authority relationship. The dependency-light claim object should not infer it.

## Relation to the Kumar2024 execution stack

The promoted Kumar2024 stack is a useful demonstration of why this separation matters. neurOS now has independently replayable execution authority, score-blind fleet control, provider-neutral external classical transport, and score-blind external admission.

Those systems can produce strong structural evidence without making the corresponding numerical result interpretable. A future Kumar2024 claim bundle could cite such receipts as software, integration, or transport evidence while still showing that an external scientific floor has not been earned.

The claim layer must never convert successful transport qualification into decoding superiority or ORION comparison permission.

## Near-term product direction

The most defensible neurOS product is not "a framework that runs neuroscience code." Capable labs can increasingly generate bespoke pipelines themselves.

A stronger wedge is an **evidence authority layer** that answers questions ordinary experiment tooling leaves implicit:

- What exact proposition was preregistered?
- What is the inference unit?
- What evidence was required before results existed?
- Which source, data, preprocessing, model, hardware, and protocol identities produced each artifact?
- Were retries score-blind?
- Which evidence objects support, contradict, replicate, or merely contextualize the claim?
- What authority was actually earned?
- What remains unqualified?

That becomes more valuable as agentic systems can generate hundreds of plausible experiments faster than humans can audit them.

## 1 to 3 year horizon

After claim manifests are exercised by real studies, the next abstractions should be earned by repeated operational need rather than added speculatively.

Likely candidates are:

### ProtocolSeal

An immutable binding of claim intent to protocol, dataset/materialization identities, inference unit, split policy, stopping rule, allowed retries, and analysis authority before outcome visibility.

### OutcomeRevealReceipt

A receipt that proves when outcome-bearing artifacts first became visible relative to protocol freeze, execution settlement, and analysis authorization.

### ExplorationLedger

An action-level ledger for agentic scientific exploration that records proposed experiments, authority-changing decisions, rejected branches, and outcome-visibility boundaries without storing private chain-of-thought.

The ledger should answer "what actions occurred under what information state?" rather than pretending internal reasoning can or should be audited.

## 3 to 5 year horizon

If independent groups adopt the contracts, evidence becomes graph-shaped rather than repository-shaped.

Useful extensions may include:

- cross-site reproduction receipts;
- site-specific environment and acquisition authorities;
- blinded replication references;
- federated evidence bundles where raw neural data cannot leave an institution;
- signed laboratory or instrument attestations;
- contradiction-preserving evidence graphs rather than winner-only summaries;
- revocation or supersession links when a receipt is later invalidated.

The hard problem is not storing a graph. It is preserving meaningful scientific authority while evidence crosses institutional boundaries.

## 5 to 10 year horizon

For autonomous physical neuroscience and BCI experimentation, claim authority will need to compose with action and safety authority.

A mature system may need to bind:

```text
scientific claim
      +
protocol authority
      +
participant / device / site authority
      +
safety envelope
      +
action authorization
      +
immutable observation receipts
```

This is where neurOS and ORION could become infrastructure for trustworthy closed-loop science rather than merely experiment orchestration.

The threshold for this layer is high. Clinical, stimulation, or participant-facing authority must never be inferred from software-contract success.

## What not to build from this horizon

This architecture does not justify:

- a generic autonomous-scientist agent framework;
- a blockchain for scientific provenance;
- a universal scalar truth or confidence score;
- automatic promotion between evidence tiers;
- a large cloud product before the contracts prove useful in real studies;
- storing model chain-of-thought as scientific provenance;
- treating cryptographic integrity as evidence that a scientific claim is correct.

Cryptography can prove that an artifact is the artifact that was authorized. It cannot prove that the hypothesis, experimental design, inference, or interpretation is scientifically valid.

## Immediate validation target

v1 should be judged by whether it improves real scientific work with minimal ceremony. The next useful test is to bind one existing neurOS study from predeclared proposition through qualified execution to a claim bundle while preserving negative and still-unearned evidence states.

A successful demonstration should make the evidence boundary clearer to an external reader without requiring them to understand the entire neurOS repository.
