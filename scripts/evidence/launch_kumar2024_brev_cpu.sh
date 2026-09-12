#!/usr/bin/env bash
set -euo pipefail

# Control-plane launcher for the provider-neutral Kumar2024 external qualification.
# This script runs on the user's local machine and uses NVIDIA Brev only as a
# Linux/x86_64 compute transport. It never changes the frozen scientific shard.

EXPECTED_BINDING_ZIP_SHA256="107a9fc57fc913815131cdf165bc35d3a1130c8300828f1f97672b27441ef0f6"
REPO_URL="https://github.com/sidhulyalkar/neurOS-v1.git"
REMOTE_BASE="/home/ubuntu/workspace"
MIN_DISK_GB="50"

usage() {
  cat <<'EOF'
Usage:
  launch_kumar2024_brev_cpu.sh \
    --repo-root /path/to/neuros-v1 \
    --binding-zip /path/to/nsq-kumar2024-promoted-binding-56fd0c51.zip \
    --attempt-id kumar2024-qual-001 \
    [--instance-name neuros-kumar-cpu-001] \
    [--output-root /path/to/local/evidence] \
    [--min-ram-gb 16] \
    [--min-vcpu 4] \
    [--dry-run]

The local repository must be a clean checkout containing PR #163. Its exact
HEAD SHA is used as the transport revision and checked out on the Brev host.

Only STOPPABLE Brev CPU shapes are eligible. On success or failure after
instance creation the launcher stops the instance but never deletes it. This
preserves the remote write-once attempt for forensic inspection while halting
compute billing.
EOF
}

REPO_ROOT=""
BINDING_ZIP=""
ATTEMPT_ID=""
INSTANCE_NAME="neuros-kumar-cpu-001"
OUTPUT_ROOT=""
MIN_RAM_GB="16"
MIN_VCPU="4"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="${2:?missing --repo-root value}"
      shift 2
      ;;
    --binding-zip)
      BINDING_ZIP="${2:?missing --binding-zip value}"
      shift 2
      ;;
    --attempt-id)
      ATTEMPT_ID="${2:?missing --attempt-id value}"
      shift 2
      ;;
    --instance-name)
      INSTANCE_NAME="${2:?missing --instance-name value}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing --output-root value}"
      shift 2
      ;;
    --min-ram-gb)
      MIN_RAM_GB="${2:?missing --min-ram-gb value}"
      shift 2
      ;;
    --min-vcpu)
      MIN_VCPU="${2:?missing --min-vcpu value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${REPO_ROOT}" || -z "${BINDING_ZIP}" || -z "${ATTEMPT_ID}" ]]; then
  usage >&2
  exit 2
fi

if [[ ! "${ATTEMPT_ID}" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]]; then
  echo "--attempt-id must match ^[a-z0-9][a-z0-9-]{0,62}$" >&2
  exit 2
fi
if [[ ! "${INSTANCE_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$ ]]; then
  echo "--instance-name contains unsupported characters" >&2
  exit 2
fi
if [[ ! "${MIN_RAM_GB}" =~ ^[0-9]+$ || ! "${MIN_VCPU}" =~ ^[0-9]+$ ]]; then
  echo "--min-ram-gb and --min-vcpu must be positive integers" >&2
  exit 2
fi
if (( MIN_RAM_GB <= 0 || MIN_VCPU <= 0 )); then
  echo "--min-ram-gb and --min-vcpu must be positive" >&2
  exit 2
fi

for command in git python3 brev; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "required command not found: ${command}" >&2
    exit 2
  }
done

REPO_ROOT="$(git -C "${REPO_ROOT}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "--repo-root must point inside the neurOS Git repository" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "local transport repository must be clean before launch" >&2
  exit 3
fi
TRANSPORT_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

RUNNER_REL="scripts/evidence/run_kumar2024_external_qualification.sh"
VERIFY_REL="scripts/evidence/verify_kumar2024_external_qualification.py"
if [[ ! -f "${REPO_ROOT}/${RUNNER_REL}" || ! -f "${REPO_ROOT}/${VERIFY_REL}" ]]; then
  echo "local checkout does not contain the external transport runner/verifier" >&2
  exit 3
fi

BINDING_ZIP="$(cd "$(dirname "${BINDING_ZIP}")" && pwd)/$(basename "${BINDING_ZIP}")"
if [[ ! -f "${BINDING_ZIP}" ]]; then
  echo "binding ZIP does not exist: ${BINDING_ZIP}" >&2
  exit 3
fi
ACTUAL_BINDING_SHA="$({ BINDING_ZIP="${BINDING_ZIP}" python3 - <<'PY'
import hashlib
import os
from pathlib import Path
path = Path(os.environ["BINDING_ZIP"])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
} )"
if [[ "${ACTUAL_BINDING_SHA}" != "${EXPECTED_BINDING_ZIP_SHA256}" ]]; then
  echo "binding ZIP SHA mismatch: expected=${EXPECTED_BINDING_ZIP_SHA256} observed=${ACTUAL_BINDING_SHA}" >&2
  exit 3
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/artifacts/external-kumar2024"
fi
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
LOCAL_ATTEMPT_ROOT="${OUTPUT_ROOT}/${ATTEMPT_ID}"
if [[ -e "${LOCAL_ATTEMPT_ROOT}" ]]; then
  echo "local attempt output already exists: ${LOCAL_ATTEMPT_ROOT}" >&2
  exit 3
fi

printf 'Transport revision: %s\n' "${TRANSPORT_REVISION}"
printf 'Binding ZIP SHA-256: %s\n' "${ACTUAL_BINDING_SHA}"
printf 'Searching NVIDIA Brev stoppable CPU shapes: x86_64, RAM >= %s GiB, vCPU >= %s, disk >= %s GB\n' \
  "${MIN_RAM_GB}" "${MIN_VCPU}" "${MIN_DISK_GB}"

INSTANCE_TYPE="$(brev search cpu \
  --arch x86_64 \
  --min-ram "${MIN_RAM_GB}" \
  --min-vcpu "${MIN_VCPU}" \
  --min-disk "${MIN_DISK_GB}" \
  --stoppable \
  --sort price \
  | head -n 1)"
if [[ -z "${INSTANCE_TYPE}" ]]; then
  echo "Brev returned no matching stoppable CPU instance type" >&2
  exit 4
fi
printf 'Selected Brev CPU instance type: %s\n' "${INSTANCE_TYPE}"

REMOTE_REPO="${REMOTE_BASE}/neuros-v1"
REMOTE_BINDING="${REMOTE_BASE}/$(basename "${BINDING_ZIP}")"
REMOTE_ATTEMPT="${REMOTE_BASE}/neuros-evidence/${ATTEMPT_ID}"

if [[ "${DRY_RUN}" == "true" ]]; then
  cat <<EOF
Dry run only. No instance was created.

Would create:
  instance: ${INSTANCE_NAME}
  type:     ${INSTANCE_TYPE}

Would bind transport revision:
  ${TRANSPORT_REVISION}

Would execute remote attempt root:
  ${REMOTE_ATTEMPT}

Would preserve local verified evidence under:
  ${LOCAL_ATTEMPT_ROOT}
EOF
  exit 0
fi

# Pipe exactly the selected CPU type into Brev's documented composable create path.
printf '%s\n' "${INSTANCE_TYPE}" | brev create "${INSTANCE_NAME}"
brev refresh

INSTANCE_CREATED="true"
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ "${INSTANCE_CREATED:-false}" == "true" ]]; then
    echo "Stopping Brev instance ${INSTANCE_NAME}; instance is preserved, not deleted."
    if ! brev stop "${INSTANCE_NAME}"; then
      echo "CRITICAL: failed to stop Brev instance ${INSTANCE_NAME}; stop it manually to prevent continued compute billing." >&2
      if [[ "${status}" -eq 0 ]]; then
        status=6
      fi
    fi
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

# Upload only the already-verified frozen binding artifact. Source code is cloned
# from the public repository and then pinned to the exact local transport SHA.
brev copy "${BINDING_ZIP}" "${INSTANCE_NAME}:${REMOTE_BINDING}"

REMOTE_COMMAND=$(cat <<EOF
set -euo pipefail
mkdir -p "${REMOTE_BASE}"
if [[ -e "${REMOTE_REPO}" ]]; then
  echo "remote repository path already exists; refusing ambiguous reuse" >&2
  exit 20
fi
git clone "${REPO_URL}" "${REMOTE_REPO}"
cd "${REMOTE_REPO}"
git checkout --detach "${TRANSPORT_REVISION}"
if [[ "\$(git rev-parse HEAD)" != "${TRANSPORT_REVISION}" ]]; then
  echo "remote transport revision mismatch" >&2
  exit 21
fi
if [[ -n "\$(git status --porcelain)" ]]; then
  echo "remote transport checkout is dirty" >&2
  exit 21
fi
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user --disable-pip-version-check uv
  export PATH="\${HOME}/.local/bin:\${PATH}"
fi
uv --version
bash "${RUNNER_REL}" \
  --control-repo "${REMOTE_REPO}" \
  --work-root "${REMOTE_ATTEMPT}" \
  --provider nvidia-brev \
  --binding-zip "${REMOTE_BINDING}"
python3 "${VERIFY_REL}" "${REMOTE_ATTEMPT}/qualification"
EOF
)

brev exec "${INSTANCE_NAME}" "${REMOTE_COMMAND}"

# Retrieve only the immutable attempt output after remote verification.
mkdir -p "${LOCAL_ATTEMPT_ROOT}"
brev copy -r \
  "${INSTANCE_NAME}:${REMOTE_ATTEMPT}/qualification" \
  "${LOCAL_ATTEMPT_ROOT}/"

LOCAL_QUALIFICATION="${LOCAL_ATTEMPT_ROOT}/qualification"
if [[ ! -d "${LOCAL_QUALIFICATION}" ]]; then
  echo "Brev evidence retrieval did not produce qualification directory" >&2
  exit 5
fi

# Independent local verification imports no neurOS package and never prints scores.
python3 "${REPO_ROOT}/${VERIFY_REL}" "${LOCAL_QUALIFICATION}"

cat <<EOF

NVIDIA Brev CPU systems qualification transport completed and independently verified.

Transport revision:
  ${TRANSPORT_REVISION}

Local sealed qualification:
  ${LOCAL_QUALIFICATION}

The numerical worker result remains quarantined and non-interpretable.
The Brev instance will now be stopped and preserved rather than deleted.
EOF
