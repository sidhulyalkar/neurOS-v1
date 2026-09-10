# NSQ Kumar2024 GPU fleet provider-free qualification

Status: **systems-emulation qualification only. This is not live provider evidence and does not authorize GPU fan-out.**

This tranche exercises the promoted Kumar2024 GPU fleet control plane without Kaggle credentials, neural data, model execution, GPU access, or scientific result inspection.

## Purpose

The real fleet will eventually schedule 810 preregistered EEGNet shards across a provider transport. Before any external provider is allowed to become part of that evidence chain, the controller should prove its state machine works end to end in a deterministic, provider-free environment.

The qualification deliberately tests the control plane rather than the neural model.

## Fixed synthetic scenario

The runner creates three synthetic EEGNet leases under a synthetic T4-shaped `PreflightAdmission`:

1. lease 0 succeeds on attempt 1;
2. lease 1 receives a trusted `provider_preemption` infrastructure failure, then succeeds on the only authorized retry;
3. lease 2 succeeds on attempt 1.

That produces:

- 3 expected leases;
- 4 total claims/attempts;
- 1 trusted infrastructure failure;
- 3 accepted terminal artifacts;
- 0 pending leases;
- one deterministic complete `SettlementLedger`.

Every accepted synthetic artifact carries the exact fleet environment authority, worker receipt identity, worker bundle identity, learned-state identity, provider receipt identity, accelerator identity, and lease/shard/attempt identities.

## What this proves

The qualification exercises:

- deterministic fleet and lease identity construction;
- persisted lease-before-claim ordering;
- claim-before-attempt namespace creation;
- create-once attempt directories;
- retry only after an admitted infrastructure failure;
- environment-bound artifact settlement;
- accepted-artifact finality;
- deterministic full-fleet reconstruction;
- score-blind completion semantics;
- a path-independent outer qualification receipt.

Running the same qualification in two different output directories must produce byte-identical receipts and the same qualification SHA-256.

## What this does not prove

It does **not** prove:

- Kaggle authentication;
- T4 availability;
- cloud scheduling or preemption behavior;
- CUDA determinism on real hardware;
- MOABB data access;
- EEGNet execution;
- model quality;
- external-floor validity;
- participant-level inference;
- ORION comparison readiness.

Those remain separately gated by issue #165 and the live preflight/transport sequence.

## Claim boundary

The synthetic receipt permanently records:

- `model_execution_performed=false`;
- `neural_data_accessed=false`;
- `scientific_outcomes_inspected=false`;
- `numerical_result_interpretable=false`;
- `external_floor_claim_generated=false`;
- `orion_comparison_permitted=false`.

Synthetic hashes are test identities only. They must never be substituted for a real GPU binding, provider receipt, worker artifact, learned state, or external scientific result.

## Qualification command

From repository root:

```bash
python scripts/evidence/qualify_kumar2024_gpu_fleet_synthetic.py \
  run \
  --repo-root . \
  --output /tmp/kumar2024-fleet-synthetic

python scripts/evidence/qualify_kumar2024_gpu_fleet_synthetic.py \
  verify \
  --output /tmp/kumar2024-fleet-synthetic
```

The output directory is write-once. Reusing an existing path fails closed.

## Relationship to live execution

The intended sequence is:

```text
promoted score-blind fleet software authority
        |
        v
provider-free synthetic qualification
        |
        v
fixed real T4 systems preflight
        |
        v
tiny live multi-shard transport qualification
        |
        v
explicit 810-shard fleet authorization
```

The synthetic layer removes controller-state-machine uncertainty before cloud credentials or GPU quota enter the picture. It does not replace the live T4 or live multi-shard gates.
