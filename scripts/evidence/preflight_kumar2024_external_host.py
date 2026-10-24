#!/usr/bin/env python3
"""Read-only Kumar2024 archived-binding and Linux host preflight.

No network access, installation, model execution, attempt creation, or claim
authorization. The existing frozen runner remains the execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_ZIP_SHA = "107a9fc57fc913815131cdf165bc35d3a1130c8300828f1f97672b27441ef0f6"
EXPECTED_BUNDLE_SHA = "45679a95c614e2107f64d7cb9ce1f87f10179c617ff160bccfd899b7ff8688d3"
EXPECTED_SOURCE = "56fd0c5132bec17575d68f62256cb80fd5661395"
EXPECTED_SHARD = "b6943a6bd0692fb99c14d3b57b2eea04ea8bf16b79b92a18415912f2b8381ceb"
EXPECTED_SHARD_ID = "subject-01/session-1/split-2026/mne-csp-lda/deterministic"
EXPECTED_COUNTS = {
    "participants": 18,
    "case_authorities": 270,
    "execution_shards": 1350,
    "method_realizations": 5,
    "planned_fit_attempts": 6750,
}
ALLOWED_PROVIDERS = frozenset(
    {"local-wsl", "local-linux", "lightning", "nvidia-brev",
     "nvidia-cloud-tasks", "nvidia-lepton"}
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_archive(path: Path) -> dict[str, Any]:
    """Verify archived bytes and unique selected classical shard, not efficacy."""
    if not path.is_file():
        raise ValueError(f"archive not found: {path}")
    actual = _sha256(path.read_bytes())
    if actual != EXPECTED_ZIP_SHA:
        raise ValueError(f"archive sha256 mismatch: observed {actual}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive has duplicate members")
        for info in archive.infolist():
            logical = PurePosixPath(info.filename)
            if logical.is_absolute() or ".." in logical.parts or "." in logical.parts:
                raise ValueError("unsafe archive member")
            if (info.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
                raise ValueError("archive contains symlink")

        def read_json(name: str) -> dict[str, Any]:
            data = json.loads(archive.read(name))
            if not isinstance(data, dict):
                raise ValueError(f"{name} must be an object")
            return data

        hashes = read_json("artifact_hashes.json")
        members = hashes.get("files")
        if hashes.get("schema_version") != 1 or not isinstance(members, dict):
            raise ValueError("invalid managed-file hash manifest")
        if set(names) != set(members) | {"artifact_hashes.json"}:
            raise ValueError("archive contains missing or unexpected files")
        for name, expected in members.items():
            if _sha256(archive.read(name)) != expected:
                raise ValueError(f"managed-file sha256 mismatch: {name}")
        # Historical declaration is checked literally. We do not invent a
        # different bundle-root hashing rule to justify a false claim.
        if hashes.get("bundle_sha256") != EXPECTED_BUNDLE_SHA:
            raise ValueError("archived bundle root declaration mismatch")
        manifest = read_json("binding_manifest.json")
        plan = read_json("execution_plan.json")
        if manifest.get("source_revision") != EXPECTED_SOURCE:
            raise ValueError("binding source revision mismatch")
        if manifest.get("counts") != EXPECTED_COUNTS:
            raise ValueError("binding cardinality mismatch")
        if any(manifest.get(key) is not False for key in (
            "model_execution_performed",
            "final_assessment_predictions_generated",
            "final_assessment_metrics_generated",
        )):
            raise ValueError("archive is not a no-model binding")
        shards = plan.get("template", {}).get("shards", [])
        if not isinstance(shards, list) or len(shards) != 1350:
            raise ValueError("unrecognized shard roster")
        matches = [item for item in shards if item.get("shard_spec_sha256") == EXPECTED_SHARD]
        if len(matches) != 1:
            raise ValueError("preauthorized shard is missing or duplicated")
        chosen = matches[0]
        if (
            chosen.get("shard_id") != EXPECTED_SHARD_ID
            or chosen.get("method_id") != "mne-csp-lda"
            or chosen.get("model_seed") is not None
            or chosen.get("budgets_per_class") != [0, 1, 2, 5, 10]
        ):
            raise ValueError("authorized shard role or budgets drifted")
    return {
        "archive_integrity": "verified",
        "archive_sha256": actual,
        "declared_bundle_sha256": EXPECTED_BUNDLE_SHA,
        "source_revision": EXPECTED_SOURCE,
        "authorized_shard": EXPECTED_SHARD,
        "authorized_shard_id": EXPECTED_SHARD_ID,
        "counts": EXPECTED_COUNTS,
        "claim_boundary": "authority-only archive; no execution, external floor, or efficacy",
    }


def _git(repo: Path, *args: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    return result.returncode == 0, result.stdout.strip()


def inspect_host(repo: Path, work_root: Path, provider: str) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if provider not in ALLOWED_PROVIDERS:
        blockers.append("unsupported transport provider label")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        blockers.append("frozen external transport requires Linux/x86_64")
    if not work_root.is_absolute():
        blockers.append("attempt root must be an absolute path")
    if work_root.exists():
        blockers.append("write-once attempt root already exists; select a new path")
    for name in ("git", "uv"):
        if shutil.which(name) is None:
            blockers.append(f"required command unavailable: {name}")
    found, top = _git(repo, "rev-parse", "--show-toplevel")
    if not found:
        blockers.append("control repository is not an accessible git checkout")
    else:
        root = Path(top).resolve()
        relative = "scripts/evidence/run_kumar2024_external_qualification.sh"
        runner = root / relative
        if not runner.is_file():
            blockers.append("tracked external qualification runner is missing")
        else:
            tracked, _ = _git(root, "ls-files", "--error-unmatch", relative)
            if not tracked:
                blockers.append("external qualification runner is not git tracked")
            clean, changes = _git(root, "status", "--porcelain", "--", relative)
            if not clean or changes:
                blockers.append("external qualification runner has local modifications")
        has_source, _ = _git(root, "cat-file", "-e", f"{EXPECTED_SOURCE}^{{commit}}")
        if not has_source:
            warnings.append("historical source not cached; runner must fetch the authorized ref")
    try:
        parent = work_root.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        free_gib = round(shutil.disk_usage(parent).free / 1024**3, 2)
        if free_gib < 50:
            warnings.append("under 50 GiB free; pinned CUDA-enabled PyTorch may exhaust disk")
    except OSError:
        free_gib = None
        warnings.append("free disk capacity not determined")
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        mem_kb = int(next(line.split()[1] for line in lines if line.startswith("MemTotal:")))
        ram_gib = round(mem_kb / 1024**2, 2)
        if ram_gib < 16:
            warnings.append("under 16 GiB RAM; CPU qualification may exceed available memory")
    except (OSError, ValueError, StopIteration):
        ram_gib = None
    if (os.cpu_count() or 0) < 4:
        warnings.append("fewer than four logical CPUs may increase runtime")
    return {
        "transport_host_eligible": not blockers,
        "hard_blockers": blockers,
        "capacity_warnings": warnings,
        "provider": provider,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "attempt_root": str(work_root),
        "attempt_root_untouched": not work_root.exists(),
        "free_disk_gib": free_gib,
        "physical_memory_gib": ram_gib,
        "claim_boundary": "preflight only; does not reconstruct EnvironmentAuthority",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-zip", type=Path, required=True)
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--control-repo", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--provider", default="local-wsl")
    args = parser.parse_args()
    try:
        archive = inspect_archive(args.binding_zip)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(json.dumps({"ready": False, "archive_error": str(exc)}, indent=2))
        return 2
    if args.archive_only:
        print(json.dumps({"ready": True, "archive": archive,
                          "host_check": "not_requested"}, indent=2))
        return 0
    if args.control_repo is None or args.work_root is None:
        parser.error("host check requires --control-repo and --work-root")
    host = inspect_host(args.control_repo, args.work_root, args.provider)
    print(json.dumps({"ready": host["transport_host_eligible"],
                      "archive": archive, "host": host}, indent=2))
    return 0 if host["transport_host_eligible"] else 3


if __name__ == "__main__":
    sys.exit(main())
