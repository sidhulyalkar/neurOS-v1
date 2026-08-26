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
 exact frozen qualification indices
       |
       +--------> RETAINED
       |
       +--------> ROLLED BACK
                    |
                    v
          exact pre-update SHA-256
```

The evidence records are immutable and deterministically fingerprinted. They do not contain wall-clock time in their semantic identity, so replaying the same authority, proposal, decision, application, and evaluation yields the same fingerprints.

The ORION API field remains named `evaluation_indices` for compatibility, but when those rows are used to choose `RETAINED` versus `ROLLED_BACK`, they are scientifically a **qualification/state-selection partition**. They are not an untouched final test set.

## Why this exists

Adaptive neural systems can accidentally produce optimistic results when:

- samples used to update a model are reused for evaluation;
- the method chooses a favorable qualification subset after adaptation;
- held-out data influence the proposal or update before the state is frozen;
- a qualification set used for model selection is later reported as if it were an untouched final assessment;
- a proposed update is recorded without preserving the exact pre-update state;
- a rejected update still mutates state;
- an update changes nothing but is counted as an adaptation event;
- rollback restores a similar model rather than the exact previous artifact;
- each algorithm invents different meanings for calibration, qualification, final assessment, and provenance.

The authority contract makes these failure modes explicit and testable.

## Core ORION types

### `ArtifactIdentity`

Identifies a model, representation, optimizer, weight set, or other mutable state by:

- stable artifact ID;
- artifact type;
- SHA-256 identity;
- optional version;
- immutable metadata;
- deterministic fingerprint.

The SHA-256 is the rollback authority. Human-readable names are not sufficient. For adaptive algorithms with optimizer state, the artifact SHA should represent the **complete mutable state**, not only visible weights.

### `AdaptationAuthority`

Freezes:

- dataset identity;
- deployment split unit;
- exact adaptation indices in exact order;
- exact held-out qualification/evaluation indices in exact order;
- processed-data SHA-256;
- total sample count;
- protocol fingerprint when available;
- source evidence-authority fingerprint when available;
- seed and structured metadata.

Adaptation and evaluation indices must be disjoint. Both must be in range. Qualification must consume the exact frozen evaluation set, not a subset, superset, or reordered selection.

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

Records held-out metrics before and after adaptation using the **complete frozen evaluation index set**. Before/after metric names must match so an adaptation cannot quietly change the scorecard after seeing results.

If this record is used only to report a predetermined frozen model, it may be an evaluation. If it influences retain/rollback or hyperparameter selection, it is a qualification/state-selection record and a separate final-assessment partition is required for an unbiased efficacy claim.

### `AdaptationOutcome`

Finalizes the evidence as either:

- `retained`, where the active artifact must be the exact post-update SHA-256; or
- `rolled-back`, where the active artifact must be the exact pre-update SHA-256.

This record says what happened. It does not prescribe a universal policy for whether accuracy, calibration, latency, reconstruction error, or another metric should dominate the retain/rollback decision.

## ORION example

```python
from orion import (
    AdaptationAuthority,
    AdaptationDecision,
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

After the algorithm updates only the authorized rows, hash the resulting complete state and create an `AdaptationApplication`. Evaluate both artifacts on `authority.evaluation_indices`, then record a retain or rollback outcome.

The repository includes a complete dependency-light example at `examples/orion/adaptation_authority.py`.

## Real ngc-learn consumer

The first state-changing consumer is the governed ngc-learn predictive-reconstruction integration:

```text
AdaptationAuthority
      |
      +--> exact calibration rows
      |        |
      |        v
      |   NgcLearnHebbianPredictiveCoding.adapt()
      |        |
      |        +--> weight SHA-256
      |        +--> optimizer-state SHA-256
      |        +--> combined state SHA-256
      |
      +--> exact qualification rows
               |
               v
         read-only before/after
               |
        retain or exact rollback
```

`scripts/evidence/run_ngclearn_hebbian_authority.py` verifies that the canonical calibration matrix selected by the authority has the same SHA-256 that the real upstream learner records as its adaptation input. Proposal/approval evidence contains calibration evidence only. Qualification data appears only after mutation, in the state-selection step.

The worker runs against installed ngc-learn 3.2 on Python 3.10 and 3.11 and its complete evidence JSON is replayed twice for byte-identical output.

This qualifies **process integrity** for that integration. It does not establish real neural-data efficacy.

## Longitudinal authority versions

neurOS now has two complementary longitudinal evidence contracts.

### v1: `LongitudinalCaseAuthority`

The v1 authority freezes source history, nested target calibration, and one immutable evaluation set. Existing v1 artifacts remain valid and replayable.

V1 is appropriate when the held-out evaluation set is never used to select model state, hyperparameters, thresholds, metrics, or policy. For example, a predetermined model may be fit at a declared calibration budget and scored once on the same frozen evaluation set across methods.

V1 must **not** be reinterpreted as untouched final-assessment evidence if its evaluation rows were used to choose retain versus rollback or otherwise select the reported model state.

### v2: `ThreeWayLongitudinalCaseAuthority`

Adaptive efficacy studies should use the additive v2 contract:

```python
from neuros.foundation_models import (
    ThreeWayLongitudinalCaseAuthority,
    make_three_way_calibration_split,
)

split = make_three_way_calibration_split(
    prospective_partition,
    qualification_fraction=0.25,
    final_assessment_fraction=0.25,
    seed=23,
)

authority = ThreeWayLongitudinalCaseAuthority.from_split(
    split,
    case_id="subject-07/session-02/three-way-v2",
    history_policy="prior",
)
```

The v2 authority freezes four roles:

```text
historical/source rows
        |
        v
CALIBRATION
state may change
        |
        v
QUALIFICATION
read-only; may choose retain/rollback
        |
        v
frozen selected state + frozen policy
        |
        v
FINAL ASSESSMENT
read-only; cannot select state or policy
        |
        v
scientific result
```

The contract enforces:

- source history remains separate from the held-out deployment unit;
- calibration, qualification, and final-assessment rows are pairwise disjoint;
- those three target roles cover the held-out partition exactly;
- qualification and final assessment preserve complete held-out class support;
- calibration membership is balanced and nested per class;
- selected calibration rows are returned in canonical source-row order so order-sensitive adaptive methods receive the same execution order;
- qualification and final-assessment identities remain invariant across every calibration budget;
- exact SHA-256 identities bind qualification, final assessment, and each calibration budget to the processed-data identity;
- serialization rejects malformed numeric fields rather than truncating/coercing them;
- non-finite or opaque semantic metadata is rejected;
- declared chronology is checked when the authority is constructed;
- `restore(processed_data)` independently verifies processed neural bytes, labels/groups, chronology, partition identity, class support, and split identity.

The returned qualification/final arrays and calibration-order arrays are read-only. Exact-set guards reject subsets, supersets, or reordered qualification/final-assessment rows.

### Construction is not dataset qualification

A serialized v2 authority is an **identity declaration** until it has successfully restored against the processed dataset.

A SHA-256 over declared indices does not prove that those rows truly belong to the claimed subject/session/classes. `restore(data)` is the operation that reconnects the declaration to processed neural bytes and re-runs dataset-dependent invariants.

The deterministic software evidence worker therefore serializes the authority and restores it against the processed fixture before emitting evidence:

```bash
python scripts/evidence/verify_longitudinal_three_way_authority.py
```

CI executes that worker twice and requires byte-identical JSON.

## Relationship between v2 and ORION authority

The foundation evidence layer owns dataset-specific chronology and three-way partitioning. ORION does **not** import `neuros-foundation`, because that would reverse the stable dependency direction.

Instead, each positive calibration budget derives a method-specific `AdaptationAuthority` from the already-restored v2 longitudinal authority:

```text
ThreeWayLongitudinalCaseAuthority
  processed data SHA-256
  source chronology
  exact calibration budget SHA-256
  immutable qualification SHA-256
  immutable final-assessment SHA-256
  authority fingerprint
            |
            v
AdaptationAuthority
  exact selected calibration indices
  exact qualification indices
  source_authority_fingerprint
            |
            v
state-changing algorithm
            |
            v
retain / exact rollback
            |
            v
frozen state
            |
            v
ThreeWayLongitudinalCaseAuthority.final_assessment_indices
```

Budget 0 is a **frozen no-update baseline**. The current ORION `AdaptationAuthority` requires non-empty adaptation indices, so a zero-calibration point must not fabricate an empty adaptation event merely to fit the state-changing API.

## Three-way authority for efficacy studies

Two partitions are sufficient to qualify **adaptation process integrity**:

1. adaptation/calibration rows;
2. held-out qualification rows used to retain or roll back.

They are not sufficient for an unbiased final efficacy claim if the second partition influences model-state selection.

The final-assessment partition must not influence:

- whether adaptation occurs;
- learning rate, epoch count, calibration budget, or other hyperparameters;
- retain/rollback decisions;
- metric selection;
- threshold selection;
- early stopping;
- architecture/model selection;
- the definition of the reported operating point.

This is the authority neurOS should use for claims such as “X requires fewer calibration examples than Y” or “X transfers better across sessions.”

## What this does not claim

A valid adaptation ledger and v2 longitudinal authority prove **process/evaluation integrity**, not adaptation benefit.

They do not prove that:

- the update improves real neural decoding;
- personalization reduces calibration cost;
- a Hebbian or STDP rule is biologically correct;
- an adapted representation transfers across subjects, sessions, sites, or devices;
- the update is safe for a physical actuator or clinical system.

Those require their own evidence under the project-wide scientific claim ladder.

## Next qualification rung

After this software authority is qualified, the next study should use one restored v2 longitudinal case for all competing methods:

1. freeze source chronology and processed-data identity;
2. freeze calibration membership, qualification rows, and final-assessment rows once;
3. instantiate one explicit positive calibration budget per experiment;
4. adapt only on those calibration rows;
5. use qualification rows only under a predeclared state-selection policy;
6. retain or exactly roll back complete mutable state;
7. freeze the selected state and every policy/hyperparameter;
8. score the final-assessment set once;
9. give the same v2 authority to frozen baselines, governed Hebbian predictive coding, and ORION adaptation.

That sequence lets neurOS compare biologically local learning and ORION personalization without allowing either method to move the goalposts.
