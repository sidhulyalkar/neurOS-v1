# NSQ Kumar2024 external execution transport

Status: **current-main external transport candidate under fresh qualification**. This document does not modify the frozen Kumar2024 scientific comparison authority.

## Purpose

GitHub Actions may certify source code, but it must not be the only compute fabric capable of executing a promoted neurOS worker. The Kumar2024 worker already restores scientific dimensions from a sealed binding plus one `shard_spec_sha256`; external compute can therefore remain scientifically narrow when source, environment, input artifact, shard, and terminal evidence are all content-addressed.

The immediate target is the non-headline classical systems qualification bound to:

- scientific source: `56fd0c5132bec17575d68f62256cb80fd5661395`
- binding run: `33291842755`
- binding artifact: `9726471429`
- binding ZIP SHA-256: `107a9fc57fc913815131cdf165bc35d3a1130c8300828f1f97672b27441ef0f6`
- binding bundle SHA-256: `45679a95c614e2107f64d7cb9ce1f87f10179c617ff160bccfd899b7ff8688d3`
- environment authority: `c45e15561ab95b8a4be0734f2fecd993fca53bf24a6e38b3c8739e1424cd1cb9`
- raw materialization: `60b89be5ded4b1ca559260b781dfcce781cf7473ad17e92cec172671e6c70a5b`
- study materialization: `28bd5564ebe87ca396b2a6093094c53b879b3b461c9c413b1422fab92d9da43a`
- execution plan: `987bb3b5566d1d481141d9a549f3588994d34e25d05cc9536baccaaa4a4641ac`
- authorized classical shard: `b6943a6bd0692fb99c14d3b57b2eea04ea8bf16b79b92a18415912f2b8381ceb`

Its numerical result remains quarantined. This transport proves systems behavior only.

## Scientific invariants

The external runner accepts no participant, session, split seed, method, model seed, calibration budget, preprocessing, or score-selection override. The scientific selector is the already-authorized shard SHA.

The runner fails before worker execution unless:

1. scientific execution source is exactly the archived revision and clean;
2. the binding ZIP and sealed binding verify;
3. the realized `EnvironmentAuthority` exactly matches the archived authority;
4. the shard occurs exactly once and remains the authorized MNE CSP+LDA complete frontier;
5. raw materialization, processed shard, case authority, protocol, and method identity reproduce;
6. the worker can later verify its own terminal bundle.

## Portable archive mode

The archived Actions ZIP is itself a frozen transport input. Once its exact SHA-256 is preserved, external execution no longer depends on GitHub Actions compute quota or artifact API access.

`run_kumar2024_external_qualification.sh` supports:

```text
github_artifact
  read original run/artifact metadata
  download exact ZIP
  require archived ZIP SHA

verified_archive
  consume caller-provided ZIP
  require the same archived ZIP SHA
```

`verified_archive` is preferred for long-lived reproducibility.

## Write-once attempts

Every invocation requires a new `--work-root` that does not exist. A failed or partial attempt remains quarantined and cannot be reused to manufacture a clean rerun.

Operational installation/log files remain outside the sealed qualification subtree. The sealed evidence contains path-independent worker evidence plus an external qualification receipt. That receipt binds transport source revision and runner-script SHA as operational provenance, separately from scientific source authority.

## Frozen CPU lane

The original promoted v1 binds `device="cpu"`, including its historical EEGNet method specification. Therefore this classical external qualification remains CPU-only even when hosted by NVIDIA infrastructure.

A GPU host must never be used to silently reinterpret v1 as CUDA execution.

## Canonical GPU lane

GPU acceleration is a separately versioned authority on current `main`. The fixed T4 systems-preflight contract is intentionally distinct from this CPU transport and freezes a deterministic CUDA policy including:

- fixed T4 accelerator identity;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- deterministic PyTorch algorithms;
- cuDNN deterministic enabled / benchmark disabled;
- TF32 disabled for CUDA matmul and cuDNN;
- fixed score-blind EEGNet preflight shard;
- no global efficacy, provider-ranking, external-floor, or ORION claim.

Current `main` also contains the score-blind GPU fleet lease/claim/retry/settlement and independent replay contracts. Those contracts establish execution-control semantics only. Full scientific fleet execution remains separately gated.

The authority lanes therefore remain intentionally separated:

```text
external CPU lane = provider-neutral classical transport qualification
fixed T4 lane     = separately frozen GPU systems preflight
fleet lane        = score-blind lease/claim/retry/settlement authority
analysis lane     = separately gated complete-study interpretation
```

## Immediate execution paths

### NVIDIA Brev CPU

Use `scripts/evidence/launch_kumar2024_brev_cpu.sh`. It provisions a matching x86_64 CPU host, uploads only the frozen binding ZIP, pins the transport revision, runs the inner write-once qualification, verifies remotely and locally, and stops the instance without deleting failed-attempt state.

See `docs/NSQ_KUMAR2024_BREV_RUNBOOK.md`.

### Lightning CPU or local WSL2/Linux

On any Linux/x86_64 host:

```bash
bash scripts/evidence/run_kumar2024_external_qualification.sh \
  --control-repo "$PWD" \
  --work-root "$HOME/neuros-evidence/kumar2024-qual-attempt-001" \
  --provider lightning \
  --binding-zip /path/to/nsq-kumar2024-promoted-binding-56fd0c51.zip
```

The preserved ZIP must hash to:

```text
107a9fc57fc913815131cdf165bc35d3a1130c8300828f1f97672b27441ef0f6
```

Then independently audit:

```bash
python3 scripts/evidence/verify_kumar2024_external_qualification.py \
  "$HOME/neuros-evidence/kumar2024-qual-attempt-001/qualification"
```

The verifier imports no neurOS package and never prints accuracy or other scores.

## Receipt claim boundary

Every accepted external receipt must preserve:

```text
numerical_result_interpretable = false
global_analysis_performed = false
external_floor_claim_generated = false
orion_comparison_permitted = false
```

A structurally valid result qualifies transport only.

## Fleet status and direction

Do not scale directly from the one-shard shell runner. Current `main` already provides a separately frozen score-blind fleet control plane with:

- immutable lease-only inputs;
- atomic claim-before-invocation;
- one attempt to one immutable artifact namespace;
- artifact-first settlement;
- no numerical-performance-dependent retry;
- trusted infrastructure-failure receipts for retry authorization;
- explicit preemption/timeout/node/upload failure semantics;
- provider/hardware/runtime receipts;
- deterministic ledger reconstruction;
- exactly one accepted terminal artifact per expected shard;
- independent replay of the settlement graph.

Those software contracts do not themselves authorize full fleet execution. The safe progression is:

1. freshly qualify this current-main external classical transport and score-blind admission layer;
2. execute exactly one real external classical systems shard and admit it without inspecting efficacy;
3. execute and admit the separately frozen fixed-T4 systems preflight under its own authority;
4. qualify a tiny multi-shard GPU transport set through the promoted score-blind fleet substrate;
5. require explicit complete-study authority before any full preregistered fleet execution or participant-level analysis;
6. keep ORION outside the comparison until an external scientific floor is legitimately established.

## Compute-fabric principle

The long-term neurOS architecture should make GitHub Actions, Brev, NVIDIA Cloud Tasks, DGX Cloud Lepton, Lightning, Kubernetes, Slurm, and local lab hardware interchangeable **transport providers beneath the same cryptographic experiment authority**. Provider convenience may change scheduling, but it must never silently change the scientific method.
