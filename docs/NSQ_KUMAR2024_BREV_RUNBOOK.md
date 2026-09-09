# NSQ Kumar2024 NVIDIA Brev CPU runbook

Status: **operational runbook for the unpromoted external transport in PR #163**.

This runbook uses NVIDIA Brev only as a Linux/x86_64 compute transport for the already-frozen Kumar2024 v1 **classical** systems qualification. It does not change the scientific graph, does not enable CUDA for v1, and does not make numerical worker output interpretable.

The canonical GPU systems authority is already separately promoted by PR #164 at `main@07a6c5fa5f212d54aae408237f14bb56f2f6eee9`. That GPU lane is intentionally distinct from this CPU transport. Issue #165 owns the later GPU fleet/settlement boundary.

## Why CPU on NVIDIA infrastructure?

The frozen Kumar2024 v1 environment binds `device="cpu"`. The first authorized external systems shard is MNE CSP+LDA. A GPU therefore adds cost without adding scientific value for this gate.

The Brev launcher asks for the cheapest available **stoppable** CPU shape satisfying:

```text
architecture = x86_64
RAM >= 16 GiB
vCPU >= 4
disk >= 50 GB
stoppable = true
sort = price ascending
```

Requiring stoppable capacity is part of cost/failure safety: the launcher must be able to halt compute after either a successful or failed write-once attempt. GPU-backed NVIDIA infrastructure is reserved for the separately promoted #164 authority and its future fleet settlement in #165.

## Preconditions

On the local control machine:

1. install the NVIDIA Brev CLI;
2. run `brev login` and confirm access with `brev list`;
3. check out PR #163's transport branch;
4. require a clean Git worktree;
5. preserve the exact binding ZIP whose SHA-256 is:

```text
107a9fc57fc913815131cdf165bc35d3a1130c8300828f1f97672b27441ef0f6
```

The ZIP is the archived artifact from binding run `33291842755`, artifact `9726471429`, source `56fd0c5132bec17575d68f62256cb80fd5661395`.

## Dry-run the hardware selection

```bash
bash scripts/evidence/launch_kumar2024_brev_cpu.sh \
  --repo-root "$PWD" \
  --binding-zip /absolute/path/nsq-kumar2024-promoted-binding-56fd0c51.zip \
  --attempt-id kumar2024-qual-001 \
  --instance-name neuros-kumar-cpu-001 \
  --dry-run
```

No instance is created during `--dry-run`.

## Execute the one authorized systems shard

```bash
bash scripts/evidence/launch_kumar2024_brev_cpu.sh \
  --repo-root "$PWD" \
  --binding-zip /absolute/path/nsq-kumar2024-promoted-binding-56fd0c51.zip \
  --attempt-id kumar2024-qual-001 \
  --instance-name neuros-kumar-cpu-001
```

The launcher then:

1. hashes the local binding ZIP;
2. records the exact clean transport Git revision;
3. provisions a matching stoppable Brev CPU shape;
4. uploads only the frozen binding ZIP;
5. clones neurOS remotely and detaches at the exact transport revision;
6. invokes the provider-neutral inner runner;
7. the inner runner creates a detached scientific worktree at exact source `56fd0c51...`;
8. it recreates CPython 3.11.16 plus the frozen distribution frontier;
9. it fails before neural-data/model execution unless the archived `EnvironmentAuthority` reproduces exactly;
10. it verifies binding, materialization, case, protocol, method, and shard authority;
11. it executes exactly one MNE CSP+LDA frontier;
12. it verifies and seals the worker result without interpreting scores;
13. an independent stdlib verifier runs remotely;
14. only the sealed qualification subtree is retrieved;
15. the independent verifier runs again locally;
16. the Brev instance is stopped, not deleted.

If `brev stop` itself fails, the launcher reports a critical billing-safety error instead of silently claiming success. Stop the named instance manually before doing anything else.

## Failure semantics

A failed attempt is transport evidence and must not be silently erased.

- The remote attempt root is write-once.
- A repeated attempt requires a new `--attempt-id`.
- The instance is stopped on success or failure but never automatically deleted.
- Do not retry based on accuracy, loss, score, or any numerical model output.
- Retry is admissible only after an infrastructure/transport failure is established and no valid terminal worker artifact exists.

## Success semantics

A successful transport qualification does **not** make the CSP score interpretable. The outer receipt must retain:

```text
numerical_result_interpretable = false
global_analysis_performed = false
external_floor_claim_generated = false
orion_comparison_permitted = false
```

Success proves only that the frozen artifact-consuming worker can be reconstructed and executed outside GitHub Actions while preserving bound identities.

## Independent local audit

```bash
python3 scripts/evidence/verify_kumar2024_external_qualification.py \
  artifacts/external-kumar2024/kumar2024-qual-001/qualification
```

The verifier imports no neurOS package and never prints accuracy or other worker scores.

## Cost discipline

Brev instances consume credits while running. The launcher stops the instance after the attempt. After the qualification is independently accepted and relevant operational logs are preserved, the stopped instance may be manually deleted.

## Next boundary

Do not use this one classical qualification as permission to schedule the full study.

The sequence is:

1. qualify the provider-neutral classical worker path under #166;
2. independently admit its structural transport evidence;
3. transplant/qualify FleetAuthority and trusted artifact settlement against current main;
4. execute and admit the already-promoted #164 fixed T4 GPU systems preflight without inspecting efficacy;
5. use #165 to freeze fleet-scale GPU lease, retry, hardware-receipt, and artifact-settlement semantics;
6. qualify a tiny multi-shard GPU transport set;
7. only then authorize the complete preregistered fleet and participant-level analysis.
