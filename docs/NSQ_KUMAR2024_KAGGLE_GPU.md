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
- environment identity.

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

1. requires an exact clean `main` checkout;
2. creates the CUDA-bound no-model binding in the pinned Python environment;
3. selects the fixed preflight shard;
4. creates a unique private Kaggle Dataset containing only the sealed launch
   pack, not redistributed Kumar2024 raw data;
5. creates a unique private Kaggle script kernel with T4 + internet enabled;
6. polls the official Kaggle CLI to terminal state;
7. downloads the zipped systems output;
8. structurally verifies the returned worker bundle on GitHub without reading
   score fields;
9. uploads the binding, systems summary, quarantined worker ZIP and verification
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
- claim CUDA numerical identity with CPU execution.
