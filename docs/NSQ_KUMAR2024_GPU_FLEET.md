# NSQ Kumar2024 GPU fleet authority v1

Status: **software contract under qualification. Full fleet execution is not yet authorized.**

This layer follows the promoted score-blind Kaggle T4 systems preflight. It does not
change the Kumar2024 scientific graph, preprocessing, split seeds, model seeds, budget
frontier, model implementation, or outcome interpretation.

## Authority chain

```text
promoted Kumar2024 scientific authority
        |
        v
promoted CUDA/T4 execution authority
main@07a6c5fa5f212d54aae408237f14bb56f2f6eee9
        |
        v
fixed T4 systems preflight
        |
        | systems evidence only
        | wall clock / accelerator / environment / artifact validity
        v
PreflightAdmission
        |
        v
FleetAuthority
        |
        +---- exact expected EEGNet shard roster
        |       |
        |       v
        |    LeaseSpec
        |       |
        |       v
        |    atomic ClaimEvent
        |       |
        |       v
        |    write-once attempt namespace
        |       |
        |       +---- valid artifact ---> ArtifactSettlementEvent
        |       |
        |       +---- trusted infrastructure failure
        |                         |
        |                         v
        |                    next attempt allowed
        |
        v
deterministic SettlementLedger
        |
        | complete only when every lease has exactly
        | one terminal accepted artifact
        v
participant-level assembly gate
```

## Why this exists

A scientific worker fleet needs stronger semantics than "launch N jobs."

The controller must be able to prove:

1. every expected shard was fixed before fleet execution;
2. no shard is silently selected or dropped because of its result;
3. every provider invocation had an immutable lease and claim first;
4. one attempt writes into one immutable artifact namespace;
5. retries occur only after trusted infrastructure evidence proves that no valid
   artifact exists;
6. a valid artifact terminates the lease regardless of its scientific score;
7. every accepted artifact binds its worker receipt, worker bundle, learned state,
   environment authority, provider receipt, and accelerator identity;
8. fleet completeness is reconstructed from immutable identities, not mutable worker
   status or leaderboard-like metrics.

## Frozen v1 scientific scope

Fleet v1 is fixed to:

- method: `braindecode-eegnet`;
- complete calibration frontier: `(0, 1, 2, 5, 10)`;
- promoted GPU execution authority:
  `07a6c5fa5f212d54aae408237f14bb56f2f6eee9`;
- archived shard identities from the already-sealed Kumar2024 execution plan.

The eventual full GPU roster is the preregistered **810 EEGNet shards / 4,050 fits**.
This module does not create that production roster until a real T4 systems preflight
has been admitted.

The overall Kumar2024 comparison remains larger than the GPU tranche. Classical
workers are governed separately.

## Preflight admission

`PreflightAdmission` may use only systems evidence:

- wall-clock seconds;
- provider cost;
- provider availability;
- accelerator identity;
- environment identity;
- cryptographic worker/artifact verification.

It requires the fixed T4-class preflight and preserves:

- `numerical_result_interpretable=false`;
- `global_analysis_performed=false`;
- `external_floor_claim_generated=false`;
- `orion_comparison_permitted=false`.

The controller may not use balanced accuracy, accuracy, AUROC, loss, predictions,
probabilities, method ranking, or a final-assessment metric to decide whether to
continue.

## Lease authority

`FleetAuthority` binds:

- the exact preflight admission identity;
- the promoted GPU execution-authority git revision;
- exact GPU binding SHA-256;
- exact execution-plan SHA-256;
- expected shard count;
- maximum attempts per lease;
- fixed EEGNet method identity;
- complete calibration frontier;
- the finite set of trusted infrastructure-failure classes;
- the systems-only provider-selection basis.

`build_leases(...)` accepts an exact minimal shard descriptor and rejects:

- extra fields;
- missing fields;
- scientific outcome fields;
- duplicate shard identities;
- non-EEGNet methods;
- incomplete budget frontiers;
- wrong fleet cardinality.

Lease ordering is canonical, so input iteration order cannot change the frozen fleet.

## Claim before invocation

A `ClaimEvent` binds:

- lease SHA-256;
- shard SHA-256;
- attempt number;
- worker identity;
- provider;
- provider-run identity.

The reference filesystem store persists a claim with `O_CREAT | O_EXCL`. The fixed
path is one claim per `(lease, attempt)` slot. A duplicate controller cannot claim
the same slot by overwriting it.

`prepare_attempt_namespace(...)` works only after the claim exists and creates a new
write-once directory whose name binds the attempt number and claim identity.

Provider adapters with remote stores must preserve the same create-if-absent/CAS
semantics. A weaker "last writer wins" object write is not equivalent authority.

## Retry semantics

Retry is permitted only after one of these trusted infrastructure failures:

- provider preemption;
- provider timeout;
- node loss;
- bootstrap failure;
- environment mismatch;
- artifact upload failure.

The failure event must carry a cryptographic evidence identity and
`valid_worker_artifact_exists=false`.

A model score, poor calibration result, unexpected prediction, or unfavorable
scientific outcome is never a retry reason.

Once an `ArtifactSettlementEvent` is accepted, the lease is terminal and later
attempts reject.

## Artifact-first settlement

An accepted artifact must bind:

- `gpu_worker_receipt_sha256`;
- `worker_bundle_sha256`;
- `learned_state_sha256`;
- `environment_authority_sha256`;
- `provider_receipt_sha256`;
- accelerator identity;
- exact lease/shard/attempt identities.

The learned-state identity is mandatory. This prevents the fleet ledger from
claiming a trained execution occurred while preserving only a scalar metric.

Scientific values remain inside the quarantined worker artifact. They are not copied
into lease, claim, failure, settlement, or ledger objects.

## Deterministic reconstruction

`reconstruct_settlement(...)` rejects:

- unknown leases;
- duplicate lease attempts;
- reused provider-run IDs;
- non-contiguous attempts;
- outcomes without claims;
- mismatched shard/lease/attempt identities;
- retries without a preceding infrastructure failure;
- attempts after a valid artifact;
- more than one accepted artifact for a lease.

The resulting ledger exposes only systems counts and identities. `complete=true`
requires every expected lease to have exactly one accepted terminal artifact and no
pending lease.

A complete fleet ledger is still not an efficacy claim. It only unlocks the next
preregistered analysis stage.

## Promotion and execution gates

The correct sequence is:

1. qualify this software contract on one exact PR head;
2. add `KAGGLE_USERNAME` and `KAGGLE_KEY` as GitHub Actions repository secrets;
3. rerun the already-promoted fixed T4 systems preflight;
4. independently verify the T4 receipt without consulting numerical efficacy;
5. construct a production `PreflightAdmission`;
6. use only systems timing/cost/availability to decide provider capacity;
7. if needed, run the already-planned deterministic timing grid;
8. freeze the production 810-shard fleet authority and lease roster;
9. run a tiny multi-shard live transport qualification;
10. verify claim, retry, artifact, learned-state, provider, and settlement receipts;
11. only then authorize the full EEGNet fleet.

Until steps 1 through 10 pass, this module is **not permission to launch all 810
GPU shards**.

## ORION boundary

Fleet completion does not automatically permit ORION comparison.

The execution layer proves that the preregistered worker graph ran completely under
the frozen authority. Participant-level analysis, external-floor construction,
mechanistic interpretation, and ORION comparison remain separate evidence stages
with their own promotion boundaries.
