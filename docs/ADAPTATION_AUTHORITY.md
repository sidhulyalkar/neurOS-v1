# Adaptation Authority

ORION treats adaptation as a state-changing operation that requires its own evidence authority.

A decoder, local learning rule, optimizer, or representation model may decide **how** to update state. It must not also decide **which data are allowed to cause the update** or **which held-out data are used to justify keeping it**.

`orion.adaptation` separates those responsibilities.

## State machine

```text
AdaptationProposal
       |
       v
GovernedAdaptationProposal
  frozen authority
  pre-update artifact
       |
       +--------> REJECTED
       |
       v
    APPROVED
       |
       v
    APPLIED
 exact adaptation indices
 before/after artifact SHA-256
       |
       v
   EVALUATED
 exact frozen evaluation indices
       |
       +--------> RETAINED
       |
       +--------> ROLLED BACK
                    |
                    v
          exact pre-update SHA-256
```

The evidence records are immutable and deterministically fingerprinted. They do not contain wall-clock time in their semantic identity, so replaying the same authority, proposal, decision, application, and evaluation yields the same fingerprints.

## Why this exists

Adaptive neural systems can accidentally produce optimistic results when:

- samples used to update a model are reused for evaluation;
- the method chooses a favorable evaluation subset after adaptation;
- a proposed update is recorded without preserving the exact pre-update state;
- a rejected update still mutates state;
- an update changes nothing but is counted as an adaptation event;
- rollback restores a similar model rather than the exact previous artifact;
- each algorithm invents different meanings for calibration, evaluation, and provenance.

The authority contract makes these failure modes explicit and testable.

## Core types

### `ArtifactIdentity`

Identifies a model, representation, optimizer, weight set, or other mutable state by:

- stable artifact ID;
- artifact type;
- SHA-256 identity;
- optional version;
- immutable metadata;
- deterministic fingerprint.

The SHA-256 is the rollback authority. Human-readable names are not sufficient.

### `AdaptationAuthority`

Freezes:

- dataset identity;
- deployment split unit;
- exact adaptation indices in exact order;
- exact held-out evaluation indices in exact order;
- processed-data SHA-256;
- total sample count;
- protocol fingerprint when available;
- source evidence-authority fingerprint when available;
- seed and structured metadata.

Adaptation and evaluation indices must be disjoint. Both must be in range. Final evaluation must consume the exact frozen evaluation set, not a subset, superset, or reordered selection.

### `GovernedAdaptationProposal`

Binds the existing lightweight `AdaptationProposal` to:

- one `AdaptationAuthority` fingerprint;
- the exact pre-update `ArtifactIdentity`;
- the exact authorized adaptation rows.

The original proposal API remains lightweight and backward compatible. Scientific/deployment authority is added only when a proposal crosses into governed state mutation.

### `AdaptationDecision`

Records explicit approval or rejection with an actor and reason.

A rejected proposal cannot produce an `AdaptationApplication`.

### `AdaptationApplication`

Records an approved mutation with:

- proposal, authority, and decision fingerprints;
- exact before/after artifact identities;
- exact adaptation indices;
- numerical update evidence such as update count or parameter delta.

An application whose pre- and post-update SHA-256 identities are identical fails closed. A no-op is not represented as a successful state change.

### `AdaptationEvaluation`

Records held-out metrics before and after adaptation using the **complete frozen evaluation index set**. Before/after metric names must match so an adaptation cannot quietly change the reported scorecard after seeing results.

### `AdaptationOutcome`

Finalizes the evidence as either:

- `retained`, where the active artifact must be the exact post-update SHA-256; or
- `rolled-back`, where the active artifact must be the exact pre-update SHA-256.

This record says what happened. It does not prescribe a universal policy for whether accuracy, calibration, latency, or another metric should dominate the retain/rollback decision.

## Example

```python
from orion import (
    AdaptationApplication,
    AdaptationAuthority,
    AdaptationDecision,
    AdaptationEvaluation,
    AdaptationOutcome,
    AdaptationProposal,
    ArtifactIdentity,
    GovernedAdaptationProposal,
)

authority = AdaptationAuthority(
    authority_id="subject-07/session-02/calibration-v1",
    dataset_id="my-eeg-study",
    split_unit="session",
    adaptation_indices=(100, 101, 104, 108),
    evaluation_indices=(102, 103, 105, 106, 107),
    processed_data_sha256="<64 hex characters>",
    n_samples=1000,
    protocol_fingerprint="subject-session-disjoint-v2",
)

before = ArtifactIdentity(
    artifact_id="decoder/subject-07/pre-calibration",
    artifact_type="decoder-state",
    sha256="<64 hex characters>",
)

proposal = GovernedAdaptationProposal.bind(
    AdaptationProposal(
        reason="target-session calibration",
        changes={"learning_rate": 1e-3, "steps": 4},
        requires_approval=True,
    ),
    authority=authority,
    before_artifact=before,
    adaptation_indices=authority.adaptation_indices,
)

decision = AdaptationDecision.approve(
    proposal,
    actor="operator/research-protocol",
    reason="calibration budget authorized",
)
```

After the algorithm updates only the authorized rows, hash the resulting state and create an `AdaptationApplication`. Evaluate both artifacts on `authority.evaluation_indices`, then record a retain or rollback outcome.

The repository includes a complete dependency-light example at `examples/orion/adaptation_authority.py`.

## Relationship to longitudinal evidence

The foundation evidence layer already owns stronger dataset-specific authorities such as `LongitudinalCaseAuthority`. ORION does **not** import `neuros-foundation` to reuse them, because that would reverse the stable dependency direction.

Instead, an experiment should derive the adaptation authority from the already-frozen case and carry its identity forward:

```text
LongitudinalCaseAuthority
  processed data SHA-256
  calibration ordering
  immutable evaluation indices
  authority fingerprint
            |
            v
AdaptationAuthority
  exact selected calibration budget
  same frozen evaluation indices
  source_authority_fingerprint
            |
            v
state-changing algorithm
```

This keeps ORION dependency-light while preserving an auditable chain back to the real-dataset evaluation authority.

## What this does not claim

A valid adaptation ledger proves **process integrity**, not adaptation benefit.

It does not prove that:

- the update improves real neural decoding;
- personalization reduces calibration cost;
- a Hebbian or STDP rule is biologically correct;
- an adapted representation transfers across subjects, sessions, sites, or devices;
- the update is safe for a physical actuator or clinical system.

Those require their own evidence under the project-wide scientific claim ladder.

## Next qualification rung

The intended next use is an ngc-learn Hebbian predictive-coding experiment in which:

1. a real `LongitudinalCaseAuthority` freezes target-session calibration/evaluation data;
2. an `AdaptationAuthority` selects one explicit calibration budget;
3. real upstream Hebbian synapses update only those rows;
4. before/after weights receive immutable SHA-256 identities;
5. synapses freeze before held-out evaluation;
6. the complete evaluation partition is scored before and after;
7. the result is retained or rolled back under explicit policy;
8. the same authority can be given to an ORION adaptation method for a matched comparison.

That sequence lets neurOS compare biologically local learning and ORION personalization without allowing either method to move the goalposts.
