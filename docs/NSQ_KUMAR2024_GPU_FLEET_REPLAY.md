# NSQ Kumar2024 GPU fleet independent replay v1

Status: **independent synthetic-evidence audit only. No scientific execution or cloud admission.**

This tranche stacks above the provider-free synthetic fleet qualification in #170.
Its purpose is to prove that a top-level `complete=true` receipt is not trusted merely
because the same code that generated it says it is complete.

The verifier is deliberately implemented as a separate standard-library-only program
that imports neither neurOS nor the fleet/synthetic modules.

## Independent authority model

The verifier knows the frozen synthetic protocol, but it does not call the generator
to derive identities. It independently reconstructs:

- synthetic preflight payload and SHA-256;
- synthetic fleet-authority payload and SHA-256;
- all four canonical leases and lease SHA-256 identities;
- all five claims and claim SHA-256 identities;
- the one infrastructure-failure outcome;
- all four artifact-settlement outcomes;
- provider/run invocation identities;
- settlement-ledger identity;
- final synthetic qualification receipt identity.

All identities use the same explicit domain-separated schemas as the authority layer,
but the implementation is independent.

## Closed evidence packet

The synthetic evidence root is treated as a closed packet rather than an open working
directory.

Verification requires the exact expected file **and directory** topology:

```text
leases/
claims/
attempts/
outcomes/
synthetic-qualification.json
```

Each expected lease, claim, attempt invocation, and outcome path must exist exactly
once. Any missing file, unexpected shadow file, unexpected empty directory, or
symlink causes rejection.

This means an operator cannot delete an inconvenient terminal artifact, add a shadow
retry, or smuggle a second receipt into the packet while retaining the original
`complete=true` envelope.

## JSON and cryptographic hygiene

Every JSON object is parsed with:

- duplicate-key rejection;
- non-finite-number rejection;
- UTF-8 decoding;
- recursive scientific-field firewalling.

For each sealed lease, claim, infrastructure failure, and artifact settlement, the
verifier removes the declared identity field and recomputes the domain-separated
SHA-256 from the remaining canonical payload.

The declared identity must match both:

1. the recomputed object contents; and
2. the independently derived frozen synthetic protocol.

Changing a persisted body while leaving its old SHA field in place therefore fails.
Freshly resealing a different synthetic event also fails because it is not part of
the independently preregistered packet.

## Independent retry replay

The verifier does not infer success from file counts alone. It independently replays
the lease/claim/outcome graph and requires:

- attempts contiguous from 1;
- no attempt after an accepted artifact;
- the second attempt only after the exact first-attempt infrastructure failure;
- `valid_worker_artifact_exists=false` for retry authority;
- every accepted artifact bound to the expected claim;
- every lease to terminate with one accepted artifact;
- no pending lease.

Only then is the settlement-ledger SHA reconstructed.

## Final receipt verification

After the store independently earns complete settlement, the verifier reconstructs
the expected top-level receipt from the replayed evidence and requires exact equality
with `synthetic-qualification.json`.

It then recomputes `synthetic_qualification_sha256` independently.

A receipt that says four artifacts were accepted when the packet contains three, or
that names another fleet/ledger/environment identity, is rejected.

## Scientific firewall

The independent verifier has no imports from:

- neurOS;
- PyTorch;
- Kaggle;
- cloud/provider SDKs;
- networking clients;
- subprocess launchers;
- model/scientific result code.

It never reads:

- `case_result.json`;
- `shard_result.json`;
- predictions;
- probabilities;
- accuracy or balanced accuracy;
- loss;
- scientific rankings.

Its successful output preserves only systems identities/counts and:

- `scientific_execution_performed=false`;
- `scientific_outcomes_inspected=false`;
- `numerical_result_interpretable=false`;
- `orion_comparison_permitted=false`.

## Adversarial qualification

The replay tests require rejection of:

1. a mutated persisted claim;
2. a mutated learned-state identity in an accepted artifact;
3. a missing outcome;
4. an extra shadow file;
5. a modified top-level receipt;
6. duplicate JSON object keys;
7. symlink injection.

The positive path must independently reproduce the exact generator receipt and
settlement-ledger SHA.

## Usage

Generate the provider-free packet:

```bash
python scripts/evidence/qualify_kumar2024_gpu_fleet_synthetic.py \
  --output /tmp/kumar2024-synthetic-fleet
```

Then verify it with the independent program:

```bash
python scripts/evidence/verify_kumar2024_gpu_fleet_synthetic.py \
  --root /tmp/kumar2024-synthetic-fleet
```

## What this does not prove

Independent synthetic replay does not prove:

- Kaggle credentials or quota are available;
- a real T4 worker launches;
- provider artifact upload is reliable;
- a real learned state matches a scientific worker bundle;
- 810 EEGNet shards can complete economically;
- any model is accurate;
- any external baseline is established;
- ORION comparison is permitted.

It removes one narrower risk: the same orchestration code cannot generate a packet and
then self-certify an inconsistent packet without an independent deterministic replay
catching the inconsistency.

## Gate sequence

```text
#169 fleet authority software qualification
        |
        v
#170 provider-free synthetic state-machine qualification
        |
        v
independent synthetic packet replay (this tranche)
        |
        v
real fixed T4 preflight
        |
        v
small live multi-shard provider transport
        |
        v
production EEGNet fleet freeze/execution
```

This stage is systems/software evidence only and does not raise the scientific claim
ceiling.