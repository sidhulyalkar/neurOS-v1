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

## Preserve the expiring original binding ZIP

The exact original GitHub Actions archive is artifact **9726471429** from run
**33291842755**, not an extracted/repacked directory or a regenerated binding.
GitHub reported expiry at **2026-11-28 04:08:51 UTC** when last checked.

Save the original ZIP outside GitHub Actions, preferably in a durable object
store with independent retention and an immutable or versioned object policy.
Never commit the archive's participant metadata to the source repository.

After downloading the original **ZIP bytes**, first check their identity:

```bash
python3 scripts/evidence/preserve_kumar2024_binding.py \
  --source-zip /path/to/original-actions-artifact.zip
```

The stdlib-only checker requires the exact archived ZIP SHA, exactly six sealed
managed files plus the index, all six managed-file SHA-256 values, the canonical
domain-separated binding root, source/environment/execution-plan identities,
and explicit no-model/no-final-prediction/no-final-metric declarations. It does
not execute neurOS, load EEG, or inspect model results.

Then preserve a second byte-identical copy in a **pre-existing** archive
directory with atomic create-if-absent semantics:

```bash
mkdir -p "$HOME/neuros-evidence/frozen"
python3 scripts/evidence/preserve_kumar2024_binding.py \
  --source-zip /path/to/original-actions-artifact.zip \
  --destination "$HOME/neuros-evidence/frozen/kumar2024-binding-56fd0c51.zip" \
  > "$HOME/neuros-evidence/frozen/kumar2024-binding-verification.json"
```

The destination is never overwritten. A pre-existing output fails rather than
silently replacing evidence. Re-running verification without `--destination`
is safe. Copy the verified ZIP **and its separately stored checksum/receipt**
to long-term, access-controlled storage; the mere existence of a copy in a
working directory is not durable archival evidence.

To run exactly the authorized classical one-shard systems qualification on
Linux/x86_64, pass that preserved ZIP to the existing external runner
`--binding-zip` argument with a **new** `--work-root`. The runner
independently checks environment authority and shard identity again before
worker execution. No score-based decision-making is permitted.

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

## Read-only archive and host preflight

Before the first paid or write-once external run, inspect the original
**frozen** binding and local Linux host with the independently testable,
stdlib-only preflight. It does not clone, download, install, execute, create
an attempt directory, or authorize efficacy claims. It checks the exact
archived ZIP SHA-256, six declared managed-file digests, frozen participant /
case / execution counts, and **exactly one authorized CSP+LDA shard**:

~~~bash
python3 scripts/evidence/preflight_kumar2024_external_host.py \
  --archive-only \
  --binding-zip /absolute/path/nsq-kumar2024-promoted-binding-56fd0c51.zip
~~~

On the target Linux x86_64 host (including WSL2 with an x86_64 CPU), choose
an attempt root that **does not yet exist**:

~~~bash
python3 scripts/evidence/preflight_kumar2024_external_host.py \
  --binding-zip /absolute/path/nsq-kumar2024-promoted-binding-56fd0c51.zip \
  --control-repo "$PWD" \
  --work-root "$HOME/neuros-evidence/kumar2024-qual-attempt-001" \
  --provider local-wsl
~~~

Change \`--provider\` to \`local-linux\` or the actual permitted provider label
when appropriate. Machine architecture, tracked transport runner, Git/uv
availability and write-once namespace are **hard** checks. Capacity below
16 GiB RAM or 50 GiB free disk is a **warning**, not an invented change to
the frozen environment authority. The historical binding was produced with
CPU-requested inference but a CUDA-enabled Torch distribution, so installation
disk and native-runtime compatibility may be nontrivial.

Only an archive/host \`ready=true\` result warrants *considering* the existing
external execution command. The runner itself must independently reproduce the
exact archived \`EnvironmentAuthority\` before materialization or model work.
A successful preflight is not scientific qualification and cannot waive
that runner's gate. Keep the archive and any original failure attempts durable.

When the preflight passes and execution is separately authorized, use the
**existing** provider-neutral runner without changing its frozen arguments:

~~~bash
bash scripts/evidence/run_kumar2024_external_qualification.sh \
  --control-repo "$PWD" \
  --work-root "$HOME/neuros-evidence/kumar2024-qual-attempt-001" \
  --provider local-wsl \
  --binding-zip /absolute/path/nsq-kumar2024-promoted-binding-56fd0c51.zip
~~~

The runner creates the write-once attempt root; the preflight never does.
The original GitHub binding artifact (run \`33291842755\`, artifact
\`9726471429\`) reports expiry on **2026-11-28**; an independent copy of the
exact ZIP is therefore an operational prerequisite. Do not commit the ZIP,
processed EEG, predictions, or result bytes into source control.

## Original GitHub Actions one-shard lane (manual opt-in)

The **historical Actions worker** and the provider-neutral external Linux runner
produce **different outer receipt formats**. Do not pass a GitHub Actions
download to \`verify_kumar2024_external_qualification.py\`; that verifier is
specific to the external transport manifest.

The one authorized Actions worker is already present on historical ref
\`execution/kumar2024-authorized-56fd0c51\`, and the ref must resolve to
\`56fd0c5132bec17575d68f62256cb80fd5661395\`. Its own workflow refuses
a mismatched \`github.workflow_sha\`. Neither the workflow nor the binding may
be regenerated from today's \`main\` just to make a run pass.

Check that the historical ref is still pinned, the artifact remains available,
and the Actions workflow has **not** already been dispatched before a first run.
A successful code CI or T4 systems preflight is *not* this classical shard.

~~~bash
REPO=sidhulyalkar/neurOS-v1
SOURCE=56fd0c5132bec17575d68f62256cb80fd5661395
REF=execution/kumar2024-authorized-56fd0c51
WORKFLOW=nsq-kumar2024-classical-worker-qualification.yml
SHARD=b6943a6bd0692fb99c14d3b57b2eea04ea8bf16b79b92a18415912f2b8381ceb

test "$(git ls-remote origin "refs/heads/$REF" | cut -f1)" = "$SOURCE"
gh run list -R "$REPO" -w "$WORKFLOW" -b "$REF" \
  -e workflow_dispatch --limit 100 \
  --json databaseId,status,conclusion,headSha,createdAt
~~~

**Explicit owner action**: If that report shows no prior run and execution
is still authorized under #109, use GitHub Actions' **Run workflow** UI with
the frozen ref and the three inputs below, or run the following from an
authenticated GitHub CLI. **It initiates real neural-data download and CPU
model fitting; do not paste it into automated CI or run without intent.**

~~~bash
gh workflow run "$WORKFLOW" -R "$REPO" --ref "$REF" \
  -f binding_run_id=33291842755 \
  -f binding_artifact_id=9726471429 \
  -f shard_spec_sha256="$SHARD"
~~~

Follow the resulting workflow run, and record its actual run ID, checked-out
source revision and GitHub artifact ID separately. A failed/partial run is
retained as evidence; retries require their own infrastructure justification.

The historical output contains \`worker_run.json\`, \`qualification_manifest.json\`
and five worker files. After download, run the **Actions-specific**, stdlib
independent structural verifier (which imports neither the producing worker
nor neurOS). For example, replacing \`RUN_ID\` with the *actual* identifier:

~~~bash
mkdir -p "$HOME/neuros-evidence/actions-worker-001"
gh run download RUN_ID -R "$REPO" \
  -n "nsq-kumar2024-classical-worker-qualification-$SHARD" \
  -D "$HOME/neuros-evidence/actions-worker-001"

python3 scripts/evidence/verify_kumar2024_actions_qualification.py \
  "$HOME/neuros-evidence/actions-worker-001" \
  > "$HOME/neuros-evidence/actions-worker-001-audit.json"
~~~

This verifier requires exact source, binding and shard identities, complete
worker file inventory, domain-separated worker/outer manifest SHA-256
identities, five budget rows, and all four explicit **false** scientific
interpretation flags. It never prints numerical scores. Save the original
GitHub artifact digest, run URL and independent verification root in a
separate trusted record. A future re-audit can pass
\`--expected-files-root\` with that independently preserved digest to detect
subsequent byte drift. **An independently calculated hash is not a signed
publisher identity or a rerun of the experimental method.**

The original artifact may expire from GitHub Actions retention. Preserve
its original ZIP and audited evidence in access-controlled durable storage
according to dataset and artifact distribution terms. This one-shard receipt
**qualifies systems execution only**; it never licenses external-floor
rankings, ORION superiority or any BCI efficacy claim.

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
