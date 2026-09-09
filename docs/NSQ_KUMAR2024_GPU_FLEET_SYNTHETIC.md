# NSQ Kumar2024 provider-free GPU fleet qualification

Status: **synthetic systems qualification only. No scientific execution is performed.**

This tranche exists so the Kumar2024 GPU fleet controller can be exercised end to
end before any Kaggle, Brev, NVIDIA, Lightning, or other external provider is
available.

It stacks on the score-blind fleet authority from #169 and does not modify the
scientific worker, preprocessing, split authority, model implementation, model
seeds, calibration frontier, or scientific result handling.

## What it exercises

The synthetic qualification runs a deterministic four-lease fleet with synthetic
SHA-256 identities only.

It proves the software path for:

1. preflight admission;
2. fleet authority construction;
3. canonical lease ordering;
4. persistence of immutable leases;
5. claim-before-invocation;
6. write-once attempt namespaces;
7. multiple provider identities;
8. one deliberately injected provider-preemption failure;
9. retry only after the trusted infrastructure-failure record;
10. accepted artifact settlement with learned-state, environment, worker, provider,
    and accelerator identities;
11. deterministic full-fleet settlement reconstruction;
12. a path-independent synthetic qualification receipt.

The test fleet contains four synthetic leases. One lease is deliberately attempted
twice, so the complete qualification contains:

- 4 leases;
- 5 claims;
- 5 attempt namespaces;
- 1 infrastructure failure;
- 4 accepted artifacts;
- 0 pending leases.

Two synthetic provider names are used to exercise provider/run identity semantics.
No external provider API is called.

## Determinism contract

The same synthetic fleet is executed under two different filesystem roots. The
resulting qualification receipt and settlement-ledger identity must be identical.

Filesystem paths, wall-clock timestamps, temporary-directory names, process IDs,
and runner identities are deliberately excluded from the scientific/control-plane
identity.

## Score blindness

The synthetic runner contains no neural data and performs no model execution.

It does not import:

- PyTorch;
- the Kumar2024 scientific worker;
- the GPU cloud worker;
- Kaggle;
- requests/network clients;
- subprocess/provider launch code.

It does not read or write:

- `case_result.json`;
- `shard_result.json`;
- balanced accuracy;
- accuracy/loss fields;
- predictions/probabilities;
- method rankings;
- final-assessment metrics.

The emitted receipt explicitly preserves:

- `scientific_execution_performed=false`;
- `scientific_outcomes_inspected=false`;
- `numerical_result_interpretable=false`;
- `external_floor_claim_generated=false`;
- `orion_comparison_permitted=false`.

A green synthetic qualification means only that the fleet state machine behaves as
intended under deterministic fake infrastructure events.

## Why this matters before live cloud execution

A GPU fleet can fail scientifically even when every model process is correct if the
control plane permits duplicate claims, score-driven retries, stale environment
identity, mutable attempt namespaces, provider-run reuse, or partial fleet
completion to masquerade as complete execution.

The synthetic lane attacks those orchestration semantics without spending GPU quota
or opening scientific outcomes.

That lets the eventual live transport test answer a narrower question:

> Does the real provider preserve the already-qualified lease/claim/artifact
> semantics under actual scheduling, preemption, upload, and accelerator behavior?

It does not need to simultaneously debug the underlying state machine.

## Qualification command

```bash
python scripts/evidence/qualify_kumar2024_gpu_fleet_synthetic.py \
  --output /tmp/kumar2024-synthetic-fleet
```

The output directory must not already exist. This preserves write-once execution
semantics for the qualification packet.

The final receipt is:

```text
/tmp/kumar2024-synthetic-fleet/synthetic-qualification.json
```

## CI contract

`NSQ Kumar2024 GPU fleet synthetic transport` runs the qualification on Python
3.10, 3.11, and 3.12 from the literal pull-request head SHA.

Each lane requires:

- a clean exact checkout;
- Python compilation;
- provider-free synthetic adversarial tests;
- one standalone qualification execution;
- an AST/import firewall proving the runner remains standard-library-only and does
  not gain cloud/scientific dependencies.

## Promotion boundary

This synthetic tranche can be software-qualified independently of Kaggle
credentials. It **must not** be interpreted as a substitute for the real fixed T4
preflight or the later tiny live multi-shard transport qualification.

The intended sequence remains:

```text
fleet authority software qualification
        |
        v
provider-free synthetic state-machine qualification
        |
        v
real fixed T4 systems preflight
        |
        v
small live provider transport qualification
        |
        v
freeze production 810-shard EEGNet fleet
        |
        v
full execution completeness
        |
        v
participant-level scientific analysis
```

No stage in this document grants efficacy, external-floor, or ORION comparison
authority.