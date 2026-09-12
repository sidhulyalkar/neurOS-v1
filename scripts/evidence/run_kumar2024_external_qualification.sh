#!/usr/bin/env bash
set -euo pipefail

# Provider-neutral external systems qualification for the already-authorized
# Kumar2024 classical worker shard. This is operational transport only.
# It does not alter the frozen scientific comparison graph or interpret scores.

SOURCE_REVISION="56fd0c5132bec17575d68f62256cb80fd5661395"
BINDING_RUN_ID="33291842755"
BINDING_ARTIFACT_ID="9726471429"
BINDING_ARTIFACT_NAME="nsq-kumar2024-promoted-binding-${SOURCE_REVISION}"
BINDING_ARTIFACT_SHA256="107a9fc57fc913815131cdf165bc35d3a1130c8300828f1f97672b27441ef0f6"
BINDING_BUNDLE_SHA256="45679a95c614e2107f64d7cb9ce1f87f10179c617ff160bccfd899b7ff8688d3"
ENVIRONMENT_AUTHORITY_SHA256="c45e15561ab95b8a4be0734f2fecd993fca53bf24a6e38b3c8739e1424cd1cb9"
RAW_MATERIALIZATION_SHA256="60b89be5ded4b1ca559260b781dfcce781cf7473ad17e92cec172671e6c70a5b"
STUDY_MATERIALIZATION_SHA256="28bd5564ebe87ca396b2a6093094c53b879b3b461c9c413b1422fab92d9da43a"
EXECUTION_PLAN_SHA256="987bb3b5566d1d481141d9a549f3588994d34e25d05cc9536baccaaa4a4641ac"
SHARD_SPEC_SHA256="b6943a6bd0692fb99c14d3b57b2eea04ea8bf16b79b92a18415912f2b8381ceb"
PROMOTED_CONSTRAINTS="requirements/nsq-kumar2024-promoted-py311.constraints.txt"
REPO="sidhulyalkar/neurOS-v1"

usage() {
  cat <<'EOF'
Usage:
  run_kumar2024_external_qualification.sh \
    --control-repo /path/to/neuros-v1 \
    --work-root /NEW/path/to/attempt \
    --provider lightning|nvidia-brev|nvidia-cloud-tasks|nvidia-lepton|local-wsl|local-linux \
    [--binding-zip /path/to/nsq-kumar2024-promoted-binding-56fd0c51.zip]

Requirements:
  - Linux x86_64 host
  - git
  - uv
  - gh authenticated to GitHub only when --binding-zip is omitted

The work root MUST NOT already exist. Every invocation gets a fresh write-once
attempt root. A failed or partial attempt is quarantined in place and must never
be erased and reused to manufacture a clean rerun.

When --binding-zip is provided, the runner performs no GitHub Actions API read.
The supplied ZIP must exactly match the archived SHA-256 authority. This keeps
execution possible after Actions quota or artifact-retention limits are reached.
EOF
}

fail() {
  local message="$1"
  local code="${2:-2}"
  printf '%s\n' "${message}" >&2
  exit "${code}"
}

CONTROL_REPO=""
WORK_ROOT=""
PROVIDER=""
BINDING_ZIP_INPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --control-repo)
      CONTROL_REPO="${2:?missing --control-repo value}"
      shift 2
      ;;
    --work-root)
      WORK_ROOT="${2:?missing --work-root value}"
      shift 2
      ;;
    --provider)
      PROVIDER="${2:?missing --provider value}"
      shift 2
      ;;
    --binding-zip)
      BINDING_ZIP_INPUT="${2:?missing --binding-zip value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ -n "${CONTROL_REPO}" && -n "${WORK_ROOT}" && -n "${PROVIDER}" ]] || {
  usage >&2
  exit 2
}

case "${PROVIDER}" in
  lightning|nvidia-brev|nvidia-cloud-tasks|nvidia-lepton|local-wsl|local-linux) ;;
  *) fail "unsupported provider label: ${PROVIDER}" ;;
esac

for command in git uv; do
  command -v "${command}" >/dev/null 2>&1 || fail "required command not found: ${command}"
done
if [[ -z "${BINDING_ZIP_INPUT}" ]]; then
  command -v gh >/dev/null 2>&1 || fail "gh is required when --binding-zip is omitted"
else
  [[ -f "${BINDING_ZIP_INPUT}" ]] || fail "--binding-zip must name an existing file"
  BINDING_ZIP_INPUT="$(readlink -f "${BINDING_ZIP_INPUT}")"
fi

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
  fail "promoted environment requires Linux x86_64"

[[ "$(git -C "${CONTROL_REPO}" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] || \
  fail "--control-repo must point inside a Git repository checkout"
CONTROL_REPO="$(git -C "${CONTROL_REPO}" rev-parse --show-toplevel)"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
case "${SCRIPT_PATH}" in
  "${CONTROL_REPO}"/*) ;;
  *) fail "external runner must execute from the declared control repository" ;;
esac
SCRIPT_RELATIVE="${SCRIPT_PATH#${CONTROL_REPO}/}"
git -C "${CONTROL_REPO}" ls-files --error-unmatch "${SCRIPT_RELATIVE}" >/dev/null 2>&1 || \
  fail "external runner must be tracked by the control repository"
git -C "${CONTROL_REPO}" diff --quiet HEAD -- "${SCRIPT_RELATIVE}" || \
  fail "external runner has unstaged source drift"
git -C "${CONTROL_REPO}" diff --cached --quiet HEAD -- "${SCRIPT_RELATIVE}" || \
  fail "external runner has staged source drift"
TRANSPORT_SOURCE_REVISION="$(git -C "${CONTROL_REPO}" rev-parse HEAD)"

[[ ! -e "${WORK_ROOT}" ]] || \
  fail "--work-root already exists; external qualification attempts are write-once" 3
mkdir -m 700 -p "${WORK_ROOT}"
WORK_ROOT="$(cd "${WORK_ROOT}" && pwd)"
printf '%s\n' "incomplete_external_qualification_attempt" > "${WORK_ROOT}/ATTEMPT_INCOMPLETE"

OPERATIONAL_ROOT="${WORK_ROOT}/operational"
SOURCE_ROOT="${WORK_ROOT}/source-${SOURCE_REVISION}"
VENV_ROOT="${WORK_ROOT}/venv-py31116"
BINDING_ZIP="${WORK_ROOT}/${BINDING_ARTIFACT_NAME}.zip"
BINDING_EXTRACT_ROOT="${WORK_ROOT}/binding-extracted"
BINDING_ROOT_MARKER="${WORK_ROOT}/BINDING_ROOT.txt"
QUALIFICATION_ROOT="${WORK_ROOT}/qualification"
WORKER_ROOT="${QUALIFICATION_ROOT}/worker"
mkdir -m 700 "${OPERATIONAL_ROOT}"

if [[ -n "${BINDING_ZIP_INPUT}" ]]; then
  BINDING_INPUT_MODE="verified_archive"
  cp -- "${BINDING_ZIP_INPUT}" "${BINDING_ZIP}"
else
  BINDING_INPUT_MODE="github_artifact"
  gh api "repos/${REPO}/actions/runs/${BINDING_RUN_ID}" \
    > "${OPERATIONAL_ROOT}/github_binding_run.json"
  gh api "repos/${REPO}/actions/artifacts/${BINDING_ARTIFACT_ID}" \
    > "${OPERATIONAL_ROOT}/github_binding_artifact.json"
  gh api "repos/${REPO}/actions/artifacts/${BINDING_ARTIFACT_ID}/zip" \
    > "${BINDING_ZIP}"
fi

# The worker itself must execute from the exact archived source revision.
if ! git -C "${CONTROL_REPO}" cat-file -e "${SOURCE_REVISION}^{commit}" 2>/dev/null; then
  git -C "${CONTROL_REPO}" fetch --no-tags origin "execution/kumar2024-authorized-56fd0c51"
fi
git -C "${CONTROL_REPO}" cat-file -e "${SOURCE_REVISION}^{commit}"
git -C "${CONTROL_REPO}" worktree add --detach "${SOURCE_ROOT}" "${SOURCE_REVISION}" \
  > "${OPERATIONAL_ROOT}/source_worktree.log" 2>&1
[[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" == "${SOURCE_REVISION}" ]] || \
  fail "exact source checkout failed" 3
[[ -z "$(git -C "${SOURCE_ROOT}" status --porcelain)" ]] || \
  fail "exact source worktree is not clean" 3

# uv is bootstrap transport only. The realized venv is what EnvironmentAuthority hashes.
uv python install 3.11.16 > "${OPERATIONAL_ROOT}/uv_python_install.log" 2>&1
uv venv --python 3.11.16 --seed "${VENV_ROOT}" \
  > "${OPERATIONAL_ROOT}/uv_venv.log" 2>&1
PYTHON="${VENV_ROOT}/bin/python"
[[ "$(${PYTHON} -c 'import platform; print(platform.python_version())')" == "3.11.16" ]] || \
  fail "failed to realize CPython 3.11.16" 4

# Verify the exact portable ZIP, optionally cross-checking live GitHub metadata,
# then extract it with traversal and symlink defenses.
BINDING_INPUT_MODE="${BINDING_INPUT_MODE}" \
RUN_JSON_PATH="${OPERATIONAL_ROOT}/github_binding_run.json" \
ARTIFACT_JSON_PATH="${OPERATIONAL_ROOT}/github_binding_artifact.json" \
EXPECTED_RUN_ID="${BINDING_RUN_ID}" \
EXPECTED_ARTIFACT_ID="${BINDING_ARTIFACT_ID}" \
EXPECTED_NAME="${BINDING_ARTIFACT_NAME}" \
EXPECTED_SOURCE_REVISION="${SOURCE_REVISION}" \
EXPECTED_ZIP_SHA256="${BINDING_ARTIFACT_SHA256}" \
BINDING_ZIP="${BINDING_ZIP}" \
BINDING_EXTRACT_ROOT="${BINDING_EXTRACT_ROOT}" \
BINDING_ROOT_MARKER="${BINDING_ROOT_MARKER}" \
"${PYTHON}" -P -s - <<'PY'
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

mode = os.environ["BINDING_INPUT_MODE"]
expected_run = int(os.environ["EXPECTED_RUN_ID"])
expected_artifact = int(os.environ["EXPECTED_ARTIFACT_ID"])
expected_revision = os.environ["EXPECTED_SOURCE_REVISION"]
expected_name = os.environ["EXPECTED_NAME"]

if mode == "github_artifact":
    run = json.loads(Path(os.environ["RUN_JSON_PATH"]).read_text(encoding="utf-8"))
    artifact = json.loads(Path(os.environ["ARTIFACT_JSON_PATH"]).read_text(encoding="utf-8"))
    expected_run_fields = {
        "id": expected_run,
        "event": "push",
        "head_branch": "main",
        "head_sha": expected_revision,
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/nsq-kumar2024-promoted-binding.yml",
    }
    for key, expected in expected_run_fields.items():
        if run.get(key) != expected:
            raise SystemExit(
                f"binding run authority mismatch for {key}: expected={expected!r}, observed={run.get(key)!r}"
            )
    if artifact.get("id") != expected_artifact:
        raise SystemExit("binding artifact id mismatch")
    if artifact.get("expired"):
        raise SystemExit("binding artifact is expired")
    if artifact.get("name") != expected_name:
        raise SystemExit("binding artifact name mismatch")
    if artifact.get("digest") != f"sha256:{os.environ['EXPECTED_ZIP_SHA256']}":
        raise SystemExit("binding artifact API digest mismatch")
    workflow_run = artifact.get("workflow_run") or {}
    if int(workflow_run.get("id", -1)) != expected_run:
        raise SystemExit("binding artifact workflow-run mismatch")
    if workflow_run.get("head_sha") not in (None, expected_revision):
        raise SystemExit("binding artifact source revision mismatch")
elif mode != "verified_archive":
    raise SystemExit(f"unsupported binding input mode: {mode!r}")

zip_path = Path(os.environ["BINDING_ZIP"])
digest = hashlib.sha256()
with zip_path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
actual_zip_sha = digest.hexdigest()
if actual_zip_sha != os.environ["EXPECTED_ZIP_SHA256"]:
    raise SystemExit(
        "binding ZIP digest differs from archived authority: "
        f"expected={os.environ['EXPECTED_ZIP_SHA256']}, observed={actual_zip_sha}"
    )

extract_root = Path(os.environ["BINDING_EXTRACT_ROOT"])
extract_root.mkdir(mode=0o700)
base = extract_root.resolve()
with zipfile.ZipFile(zip_path) as archive:
    for info in archive.infolist():
        logical = PurePosixPath(info.filename)
        if logical.is_absolute() or ".." in logical.parts or "." in logical.parts:
            raise SystemExit(f"binding ZIP contains unsafe path: {info.filename!r}")
        unix_mode = (info.external_attr >> 16) & 0o170000
        if unix_mode == stat.S_IFLNK:
            raise SystemExit(f"binding ZIP contains unsupported symlink: {info.filename!r}")
        target = (base / Path(*logical.parts)).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise SystemExit(f"binding ZIP path escapes extraction root: {info.filename!r}") from exc
    archive.extractall(base)

hash_manifests = list(base.rglob("artifact_hashes.json"))
if len(hash_manifests) != 1:
    raise SystemExit(
        f"binding archive must expose exactly one artifact_hashes.json, observed={len(hash_manifests)}"
    )
binding_root = hash_manifests[0].parent.resolve()
Path(os.environ["BINDING_ROOT_MARKER"]).write_text(str(binding_root) + "\n", encoding="utf-8")
PY

[[ -f "${BINDING_ROOT_MARKER}" ]] || fail "binding root marker was not produced" 4
BINDING_ROOT="$(tr -d '\r\n' < "${BINDING_ROOT_MARKER}")"

cd "${SOURCE_ROOT}"

# Reproduce the exact promoted package environment. Keep installation noise outside
# the sealed evidence subtree.
{
  "${PYTHON}" -m pip install --disable-pip-version-check \
    -c "${PROMOTED_CONSTRAINTS}" \
    "pip==26.2.1" "setuptools==79.0.1" "wheel==0.48.0"
  "${PYTHON}" -m pip install --disable-pip-version-check --no-build-isolation \
    -c "${PROMOTED_CONSTRAINTS}" -e packages/neuros-core
  "${PYTHON}" -m pip install --disable-pip-version-check --no-build-isolation \
    -c "${PROMOTED_CONSTRAINTS}" -e packages/neuros-drivers
  "${PYTHON}" -m pip install --disable-pip-version-check --no-build-isolation \
    -c "${PROMOTED_CONSTRAINTS}" -e packages/neuros-models
  "${PYTHON}" -m pip install --disable-pip-version-check --no-build-isolation \
    -c "${PROMOTED_CONSTRAINTS}" -e "packages/neuros-foundation[evidence,braindecode-evidence]"
  "${PYTHON}" -m pip install --disable-pip-version-check --no-build-isolation \
    -c "${PROMOTED_CONSTRAINTS}" -e packages/orion
  "${PYTHON}" -m pip install --disable-pip-version-check --no-build-isolation \
    -c "${PROMOTED_CONSTRAINTS}" -e packages/neuros
} > "${OPERATIONAL_ROOT}/environment_install.log" 2>&1

"${PYTHON}" scripts/evidence/validate_promoted_environment.py \
  --constraints "${PROMOTED_CONSTRAINTS}" \
  > "${OPERATIONAL_ROOT}/environment_constraint_verify.log" 2>&1

# Hard gate before neural data or model execution.
EXPECTED_ENVIRONMENT_AUTHORITY_SHA256="${ENVIRONMENT_AUTHORITY_SHA256}" \
"${PYTHON}" -P -s - <<'PY' > "${OPERATIONAL_ROOT}/environment_authority_verify.log" 2>&1
import os
from neuros.evidence.kumar2024_materialized_study import _runtime_authority
from neuros.evidence.kumar2024_promoted_binding import promoted_materialization_config

environment = _runtime_authority(promoted_materialization_config())
expected = os.environ["EXPECTED_ENVIRONMENT_AUTHORITY_SHA256"]
if environment.sha256 != expected:
    raise SystemExit(
        "external host does not reproduce promoted EnvironmentAuthority: "
        f"expected={expected}, observed={environment.sha256}"
    )
print(environment.sha256)
PY

"${PYTHON}" -P -s -m neuros.evidence.kumar2024_promoted_binding \
  --verify-only --output "${BINDING_ROOT}" \
  > "${OPERATIONAL_ROOT}/binding_verify.log" 2>&1

EXPECTED_BINDING_BUNDLE_SHA256="${BINDING_BUNDLE_SHA256}" \
EXPECTED_SOURCE_REVISION="${SOURCE_REVISION}" \
EXPECTED_ENVIRONMENT_AUTHORITY_SHA256="${ENVIRONMENT_AUTHORITY_SHA256}" \
EXPECTED_RAW_MATERIALIZATION_SHA256="${RAW_MATERIALIZATION_SHA256}" \
EXPECTED_STUDY_MATERIALIZATION_SHA256="${STUDY_MATERIALIZATION_SHA256}" \
EXPECTED_EXECUTION_PLAN_SHA256="${EXECUTION_PLAN_SHA256}" \
BINDING_ROOT="${BINDING_ROOT}" \
SHARD_SPEC_SHA256="${SHARD_SPEC_SHA256}" \
"${PYTHON}" -P -s - <<'PY' > "${OPERATIONAL_ROOT}/binding_authority_preflight.log" 2>&1
import json
import os
from pathlib import Path

root = Path(os.environ["BINDING_ROOT"])
hashes = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
if hashes.get("bundle_sha256") != os.environ["EXPECTED_BINDING_BUNDLE_SHA256"]:
    raise SystemExit("binding bundle SHA differs from authorized authority")
manifest = json.loads((root / "binding_manifest.json").read_text(encoding="utf-8"))
for key, expected in {
    "source_revision": os.environ["EXPECTED_SOURCE_REVISION"],
    "environment_authority_sha256": os.environ["EXPECTED_ENVIRONMENT_AUTHORITY_SHA256"],
    "raw_materialization_sha256": os.environ["EXPECTED_RAW_MATERIALIZATION_SHA256"],
    "study_materialization_sha256": os.environ["EXPECTED_STUDY_MATERIALIZATION_SHA256"],
}.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"binding manifest {key} differs from authorized authority")
execution = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
if execution.get("execution_plan_sha256") != os.environ["EXPECTED_EXECUTION_PLAN_SHA256"]:
    raise SystemExit("binding execution-plan SHA differs from authorized authority")
requested = os.environ["SHARD_SPEC_SHA256"]
matches = [
    item for item in execution["template"]["shards"]
    if item.get("shard_spec_sha256") == requested
]
if len(matches) != 1:
    raise SystemExit("authorized shard must occur exactly once in binding")
shard = matches[0]
if shard.get("method_id") != "mne-csp-lda":
    raise SystemExit("systems qualification is frozen to MNE CSP+LDA")
if tuple(shard.get("budgets_per_class", ())) != (0, 1, 2, 5, 10):
    raise SystemExit("authorized shard must retain complete budget frontier")
PY

# Only now create the sealed qualification subtree. A failure after this point
# leaves a quarantined partial attempt. Never reuse the same work root.
mkdir -m 700 "${QUALIFICATION_ROOT}"
mkdir -m 700 "${WORKER_ROOT}"

"${PYTHON}" -P -s -m neuros.evidence.kumar2024_promoted_worker \
  --binding "${BINDING_ROOT}" \
  --shard-spec-sha256 "${SHARD_SPEC_SHA256}" \
  --output "${WORKER_ROOT}" \
  > "${OPERATIONAL_ROOT}/worker_run.stdout.log" \
  2> "${OPERATIONAL_ROOT}/worker_run.stderr.log"

"${PYTHON}" -P -s -m neuros.evidence.kumar2024_promoted_worker \
  --verify-only \
  --binding "${BINDING_ROOT}" \
  --shard-spec-sha256 "${SHARD_SPEC_SHA256}" \
  --output "${WORKER_ROOT}" \
  > "${OPERATIONAL_ROOT}/worker_verify.stdout.log" \
  2> "${OPERATIONAL_ROOT}/worker_verify.stderr.log"

# Seal only canonical evidence. Host paths, provider logs, package-install logs,
# and other operational noise remain outside QUALIFICATION_ROOT.
PROVIDER="${PROVIDER}" \
BINDING_INPUT_MODE="${BINDING_INPUT_MODE}" \
TRANSPORT_SOURCE_REVISION="${TRANSPORT_SOURCE_REVISION}" \
TRANSPORT_SCRIPT_PATH="${SCRIPT_PATH}" \
SOURCE_REVISION="${SOURCE_REVISION}" \
BINDING_RUN_ID="${BINDING_RUN_ID}" \
BINDING_ARTIFACT_ID="${BINDING_ARTIFACT_ID}" \
BINDING_ARTIFACT_SHA256="${BINDING_ARTIFACT_SHA256}" \
BINDING_BUNDLE_SHA256="${BINDING_BUNDLE_SHA256}" \
ENVIRONMENT_AUTHORITY_SHA256="${ENVIRONMENT_AUTHORITY_SHA256}" \
RAW_MATERIALIZATION_SHA256="${RAW_MATERIALIZATION_SHA256}" \
STUDY_MATERIALIZATION_SHA256="${STUDY_MATERIALIZATION_SHA256}" \
EXECUTION_PLAN_SHA256="${EXECUTION_PLAN_SHA256}" \
SHARD_SPEC_SHA256="${SHARD_SPEC_SHA256}" \
QUALIFICATION_ROOT="${QUALIFICATION_ROOT}" \
WORKER_ROOT="${WORKER_ROOT}" \
"${PYTHON}" -P -s - <<'PY'
import hashlib
import json
import os
import platform
from pathlib import Path

qualification_root = Path(os.environ["QUALIFICATION_ROOT"])
worker_root = Path(os.environ["WORKER_ROOT"])
worker_hashes = json.loads((worker_root / "artifact_hashes.json").read_text(encoding="utf-8"))
worker_manifest = json.loads((worker_root / "worker_manifest.json").read_text(encoding="utf-8"))

for key, expected in {
    "source_revision": os.environ["SOURCE_REVISION"],
    "binding_bundle_sha256": os.environ["BINDING_BUNDLE_SHA256"],
    "environment_authority_sha256": os.environ["ENVIRONMENT_AUTHORITY_SHA256"],
    "raw_materialization_sha256": os.environ["RAW_MATERIALIZATION_SHA256"],
    "study_materialization_sha256": os.environ["STUDY_MATERIALIZATION_SHA256"],
    "execution_plan_sha256": os.environ["EXECUTION_PLAN_SHA256"],
    "shard_spec_sha256": os.environ["SHARD_SPEC_SHA256"],
}.items():
    if worker_manifest.get(key) != expected:
        raise SystemExit(f"verified worker manifest {key} differs from frozen authority")
if worker_manifest.get("global_analysis_performed") is not False:
    raise SystemExit("worker manifest must preserve global_analysis_performed=false")

script_path = Path(os.environ["TRANSPORT_SCRIPT_PATH"])
script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "artifact_kind": "promoted_classical_worker_external_systems_qualification",
    "transport_provider": os.environ["PROVIDER"],
    "binding_input_mode": os.environ["BINDING_INPUT_MODE"],
    "transport_source_revision": os.environ["TRANSPORT_SOURCE_REVISION"],
    "transport_script_sha256": script_sha,
    "transport_semantics": (
        "external provider executes frozen worker authority; provider status is not "
        "scientific retry authority"
    ),
    "source_revision": os.environ["SOURCE_REVISION"],
    "binding_run_id": int(os.environ["BINDING_RUN_ID"]),
    "binding_artifact_id": int(os.environ["BINDING_ARTIFACT_ID"]),
    "binding_artifact_sha256": os.environ["BINDING_ARTIFACT_SHA256"],
    "binding_bundle_sha256": os.environ["BINDING_BUNDLE_SHA256"],
    "environment_authority_sha256": os.environ["ENVIRONMENT_AUTHORITY_SHA256"],
    "raw_materialization_sha256": os.environ["RAW_MATERIALIZATION_SHA256"],
    "study_materialization_sha256": os.environ["STUDY_MATERIALIZATION_SHA256"],
    "execution_plan_sha256": os.environ["EXECUTION_PLAN_SHA256"],
    "shard_spec_sha256": os.environ["SHARD_SPEC_SHA256"],
    "worker_bundle_sha256": worker_hashes["bundle_sha256"],
    "worker_shard_result_sha256": worker_manifest["shard_result_sha256"],
    "platform": {"system": platform.system(), "machine": platform.machine()},
    "numerical_result_interpretable": False,
    "global_analysis_performed": False,
    "external_floor_claim_generated": False,
    "orion_comparison_permitted": False,
    "claim_boundary": (
        "systems qualification of one preauthorized classical artifact-consuming worker path only; "
        "all numerical scores are quarantined and cannot support efficacy, method-ranking, "
        "external-floor, or ORION claims"
    ),
}
encoded = json.dumps(
    {
        "schema": "neuros.promoted_classical_worker_external_systems_qualification.v1",
        "payload": payload,
    },
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
payload["qualification_sha256"] = hashlib.sha256(encoded).hexdigest()
(qualification_root / "qualification_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

managed = {}
for path in sorted(qualification_root.rglob("*")):
    if not path.is_file() or path.name == "external_artifact_hashes.json":
        continue
    relative = path.relative_to(qualification_root).as_posix()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    managed[relative] = digest.hexdigest()
root = hashlib.sha256(
    json.dumps(
        {
            "schema": "neuros.promoted_classical_worker_external_bundle.v1",
            "payload": {"files": managed},
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
(qualification_root / "external_artifact_hashes.json").write_text(
    json.dumps(
        {"schema_version": 1, "files": managed, "bundle_sha256": root},
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "qualification_complete": True,
            "qualification_sha256": payload["qualification_sha256"],
            "external_bundle_sha256": root,
            "worker_bundle_sha256": worker_hashes["bundle_sha256"],
            "transport_source_revision": payload["transport_source_revision"],
            "transport_script_sha256": payload["transport_script_sha256"],
            "numerical_result_interpretable": False,
        },
        sort_keys=True,
    )
)
PY

rm "${WORK_ROOT}/ATTEMPT_INCOMPLETE"
printf '%s\n' "complete_external_qualification_attempt" > "${WORK_ROOT}/ATTEMPT_COMPLETE"

printf '\nExternal qualification evidence written to:\n  %s\n' "${QUALIFICATION_ROOT}"
printf 'Operational logs remain separately under:\n  %s\n' "${OPERATIONAL_ROOT}"
printf 'Do not inspect or interpret worker numerical values before independent evidence audit.\n'
