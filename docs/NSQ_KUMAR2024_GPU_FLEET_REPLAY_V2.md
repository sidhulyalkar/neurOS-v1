# NSQ Kumar2024 GPU fleet independent replay v2

Status: **software/systems verification only. No live provider or scientific execution.**

This tranche independently audits the provider-free three-lease Kumar2024 fleet packet
promoted by `main@7ff39990361038e5c578d1100c61b3c7e7dac826`.

It exists because a synthetic generator must not be allowed to certify its own
`complete=true` output through the same implementation that produced it.

## Independence boundary

`scripts/evidence/verify_kumar2024_gpu_fleet_synthetic.py` is standard-library only.
It imports neither neurOS nor the synthetic generator. It independently reconstructs:

- the fixed synthetic preflight payload and SHA-256;
- the fleet authority and environment binding;
- all three canonical `LeaseSpec` identities;
- all four `ClaimEvent` identities;
- the one trusted `provider_preemption` failure;
- all three artifact settlements;
- retry legality and fleet completeness;
- the settlement-ledger SHA-256;
- the top-level synthetic qualification receipt SHA-256.

The verifier's independent derivation reproduces the already-observed promoted
identities:

- settlement ledger:
  `293184148318914f332b3c4fe9c6658ca7a0acabadc79d3e46d63239291ec3d5`
- synthetic qualification:
  `1e35e04e2bf49085b15e837d32c5abc1cee52b1f26ac872353f9089598a68df1`

## Closed packet topology

The promoted generator writes:

```text
synthetic_qualification_receipt.json
store/
  leases/
  claims/
  attempts/
  outcomes/
```

Every expected lease, claim, outcome, lease-specific directory, and immutable attempt
directory is derived independently from the frozen three-lease protocol. Attempt
directories are intentionally empty: provider/run identity is already sealed into
the claim, so this protocol does not invent a redundant invocation record.

Verification rejects missing or shadow files, unexpected directories, symlinks,
duplicate JSON keys, and non-finite JSON values.

## Cryptographic replay

Each lease, claim, infrastructure failure, and artifact settlement is checked twice:

1. its declared domain-separated identity is recomputed from its own contents;
2. the entire record must equal the independently preregistered protocol record.

Simply recomputing a SHA after changing a learned-state or environment identity does
not make that change acceptable.

The retry graph is independently replayed. The second attempt exists only for the
predeclared lease and only after the exact trusted `provider_preemption` outcome with
`valid_worker_artifact_exists=false`. Any attempt after an accepted artifact rejects.
Every accepted artifact must carry the exact frozen environment authority.

## Adversarial suite

The exact-head tests require rejection of mutated claims, freshly resealed
learned-state substitutions, freshly resealed environment substitutions, missing
terminal outcomes, shadow files, unexpected empty directories, freshly resealed false
top-level scientific claims, duplicate JSON keys, symlink injection, and a symlinked
evidence root.

The positive path generates a packet with the promoted generator, then verifies it
with the independent implementation and requires the two known SHA-256 identities
above.

## Claim boundary

Successful replay proves only that the provider-free systems packet is internally
consistent with the preregistered synthetic state machine.

It permanently preserves:

- `model_execution_performed=false`;
- `neural_data_accessed=false`;
- `scientific_outcomes_inspected=false`;
- `numerical_result_interpretable=false`;
- `orion_comparison_permitted=false`.

It does not prove Kaggle credentials, T4 execution, provider upload reliability,
scientific efficacy, external-floor validity, complete production execution, or
ORION readiness.

## Gate sequence

```text
promoted fleet authority
        |
        v
promoted provider-free synthetic qualification
        |
        v
independent three-lease replay (this tranche)
        |
        v
real fixed T4 systems preflight
        |
        v
systems-only preflight admission
        |
        v
tiny live multi-shard provider transport
        |
        v
explicit production fleet authorization
```

The production 810-shard EEGNet fleet remains blocked until the live gates are
satisfied.
