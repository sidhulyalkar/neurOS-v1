# Kumar2024 Kaggle T4 execution authority

Status: **cloud execution tranche; systems preflight before scientific fan-out**.

This design accelerates the preregistered Kumar2024 promoted execution without
silently changing the already-qualified CPU authority. The existing promoted
CPU binding and classical worker paths remain unchanged.

## Why Kaggle T4

The cloud worker is intentionally pinned to Kaggle's `NvidiaTeslaT4` machine
shape. Kaggle currently exposes T4 selection through the official CLI and kernel
metadata. P100 is not used because the current Kaggle documentation warns that
the default PyTorch image does not include Pascal `sm_60` kernels.

The runner does not trust the Kaggle base Python environment. It creates an
isolated CPython 3.11.16 environment, installs the exact promoted constraints,
checks out the binding-owned neurOS Git SHA, and recomputes the promoted
environment authority before model execution.

## Authority layering

```text
qualified neurOS source revision
        |
        +--> isolated Kaggle quota gate
        |      +-- credentials present
        |      +-- >= 1.0 free GPU-hour
        |      +-- content-addressed systems receipt
        |
        v
canonical promoted comparison plan
        |
        v
CUDA-bound no-model promoted binding
        |
        +-- exact raw/processed/case authority
        +-- exact method specs with device="cuda"
        +-- exact environment authority
        +-- deterministic CUDA policy
        |
        v
private Kaggle launch pack
        |
        v
NvidiaTeslaT4 worker
        |
        v
canonical promoted atomic worker bundle
        |
        v
GPU systems receipt
```

The quota gate and scientific binding execute in separate GitHub Actions jobs.
This separation is deliberate: the Kaggle CLI needed to inspect account quota
must never become an installed distribution inside the environment captured by
the promoted scientific `EnvironmentAuthority`.

The CUDA adapter is deliberately separate from the canonical CPU path. Its
compatibility scope temporarily projects `promoted_materialization_config()` to
`device="cuda"` only within one sequential binder/worker process, then restores
the original functions. This keeps old CPU semantics unchanged while the GPU
path is qualified independently.

## Determinism policy

Before binding and before GPU execution, neurOS requires:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `torch.use_deterministic_algorithms(True)`;
- `torch.backends.cudnn.deterministic = True`;
- `torch.backends.cudnn.benchmark = False`;
- CUDA matmul TF32 disabled;
- cuDNN TF32 disabled.

The normal environment authority independently binds the requested device,
Torch/CUDA/cuDNN runtime and the deterministic-algorithm/cuDNN flags. The outer
GPU authority additionally seals the cuBLAS and TF32 policy.

Actual accelerator identity is observed at worker time. The first tranche fails
closed unless the device name is T4-class.

## Free-GPU quota gate

Before the 18-participant binding is materialized, a separate control-plane job
uses the pinned Kaggle CLI to query the account's current weekly accelerator
quota. The workflow requires exactly one GPU quota row and a predeclared launch
floor of at least `1.0` remaining GPU-hour.

The gate records only systems metadata:

- used GPU hours;
- remaining GPU hours;
- total weekly GPU hours;
- quota refresh time when supplied by Kaggle;
- the fixed minimum launch threshold;
- a domain-separated SHA-256 of the normalized quota snapshot.

It explicitly records `scientific_result_used=false`. No dataset values, worker
predictions, scores, rankings, or method results are available to this job.

The raw CSV and normalized quota receipt are retained as a separate GitHub
Actions artifact. Only the quota-receipt SHA, remaining-hours value and refresh
time cross into the later launch manifest. This makes the claim "the preflight
was started with free GPU quota available" auditable without allowing quota
inspection to alter scientific authority.

A quota failure stops the workflow before the expensive binding. This includes
missing Kaggle credentials, a missing/ambiguous GPU quota row, malformed quota
units, zero weekly allowance, or less than the predeclared one-hour launch
floor.

## Systems-preflight shard

The first cloud run is fixed before execution:

- subject: `1`;
- target session: `5`;
- split seed: `2026`;
- method: `braindecode-eegnet`;
- model seed: `31415`;
- budgets per class: `[0, 1, 2, 5, 10]`.

Selection is content-addressed through the archived `shard_spec_sha256`. The
scheduler cannot choose a different participant, session, model seed or budget
frontier at launch time.

### Preflight claim boundary

The preflight executes a real atomic worker, but it is **systems evidence only**.
The GitHub/Kaggle orchestration may use only:

- worker-bundle verification;
- elapsed wall-clock time;
- accelerator identity;
- environment identity;
- accelerator quota availability and refresh metadata.

It may not use balanced accuracy, scientific score, method ranking, final
assessment metrics, external-floor status or ORION comparison as a go/no-go
input. The bootstrap never reads `case_result.json` or `shard_result.json`; those
remain quarantined inside the sealed worker archive.

No scientific interpretation is permitted until the preregistered execution
completeness rules are satisfied.

## Orchestration

The manual workflow is:

`.github/workflows/nsq-kumar2024-kaggle-preflight.yml`

It performs the following in one provenance chain:

1. in an isolated control-plane job, requires Kaggle credentials and queries the
   current weekly GPU quota;
2. requires at least `1.0` remaining free GPU-hour and seals the quota receipt;
3. in a fresh job, requires an exact clean `main` checkout;
4. installs only the exact promoted scientific environment and creates the
   CUDA-bound no-model binding;
5. selects the fixed preflight shard;
6. binds the quota-receipt SHA and safe quota metadata into the systems launch
   manifest;
7. creates a unique private Kaggle Dataset containing only the sealed launch
   pack, not redistributed Kumar2024 raw data;
8. creates a unique private Kaggle script kernel with T4 + internet enabled;
9. polls the official Kaggle CLI to terminal state;
10. downloads the zipped systems output;
11. structurally verifies the returned worker bundle on GitHub without reading
    score fields;
12. uploads the binding, systems summary, quarantined worker ZIP and verification
    record as a 90-day GitHub Actions artifact.

The Kaggle kernel itself is
`scripts/evidence/kaggle_kumar2024_gpu_runner.py`.

## One-time account boundary

The workflow needs these GitHub Actions repository secrets:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

They are account credentials and must not be committed or pasted into issue/PR
text. Once present, the workflow uses Kaggle's official CLI; no browser-session
automation is needed.

## Expansion gate

Do not fan out all promoted EEGNet shards after code qualification alone.

The sequence is:

1. one fixed T4 systems preflight;
2. verify environment + bundle + wall-clock;
3. use systems-only timing to estimate full GPU-hours;
4. if feasible, define a deterministic 9-shard timing grid before execution;
5. only then freeze a chunking schedule for the full 810 EEGNet shards / 4,050
   fit attempts.

Chunking should group work to amortize raw-data downloads and must be fixed from
the execution plan rather than adapted to observed scientific scores.

## Non-goals

This tranche does not:

- change the promoted comparison plan;
- change preprocessing, case authority, budgets or model seeds;
- reinterpret existing CPU evidence as GPU evidence;
- compare CPU and GPU efficacy;
- run global analysis;
- generate an external-floor claim;
- permit ORION comparison;
- automatically inspect a preflight score;
- claim CUDA numerical identity with CPU execution;
- treat quota availability as scientific evidence.
