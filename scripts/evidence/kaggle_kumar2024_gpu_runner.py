"""Kaggle bootstrap for one sealed Kumar2024 T4 systems-preflight worker.

This script is intentionally standalone: Kaggle runs it with the base image,
then it creates an isolated CPython environment, checks out the binding-owned
neurOS revision, reproduces the pinned promoted environment, and invokes the
GPU authority adapter. It never reads or prints scientific scores.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/neuros-kumar2024-gpu")
OUTPUT_ROOT = Path("/kaggle/working/neuros-kumar2024-gpu-output")
REPO_URL = "https://github.com/sidhulyalkar/neurOS-v1.git"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=cwd, env=env, check=True)


def _one_launch_pack() -> Path:
    matches = sorted(INPUT_ROOT.rglob("gpu_launch_pack.zip"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one gpu_launch_pack.zip, found {len(matches)}")
    return matches[0]


def _load_manifest(root: Path) -> dict:
    path = root / "launch_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "phase",
        "source_revision",
        "gpu_binding_sha256",
        "binding_bundle_sha256",
        "shard_spec_sha256",
        "constraints_sha256",
        "python_version",
        "uv_version",
        "machine_shape",
    }
    if set(payload) < required:
        raise ValueError(f"launch manifest missing keys: {sorted(required - set(payload))}")
    if payload["schema_version"] != 1 or payload["phase"] != "systems-preflight":
        raise ValueError("Kaggle runner accepts only schema-v1 systems-preflight launches")
    for key, length in (
        ("source_revision", 40),
        ("gpu_binding_sha256", 64),
        ("binding_bundle_sha256", 64),
        ("shard_spec_sha256", 64),
        ("constraints_sha256", 64),
    ):
        value = str(payload[key])
        if len(value) != length or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"launch manifest {key} is malformed")
    if payload["machine_shape"] != "NvidiaTeslaT4":
        raise ValueError("systems preflight is frozen to Kaggle NvidiaTeslaT4")
    if payload.get("numerical_result_interpretable") is not False:
        raise ValueError("systems preflight must explicitly forbid numerical interpretation")
    return payload


def _find_uv() -> str:
    candidates = [shutil.which("uv"), str(Path.home() / ".local/bin/uv")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("uv installation completed but executable was not found")


def main() -> int:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    WORK_ROOT.mkdir(parents=True)
    OUTPUT_ROOT.mkdir(parents=True)

    launch_pack = _one_launch_pack()
    pack_root = WORK_ROOT / "launch-pack"
    pack_root.mkdir()
    with zipfile.ZipFile(launch_pack) as archive:
        archive.extractall(pack_root)
    manifest = _load_manifest(pack_root)

    gpu_binding_root = pack_root / "gpu-binding"
    if not (gpu_binding_root / "gpu_artifact_hashes.json").is_file():
        raise FileNotFoundError("launch pack is missing the sealed GPU binding")

    uv_version = str(manifest["uv_version"])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            f"uv=={uv_version}",
        ]
    )
    uv = _find_uv()
    python_version = str(manifest["python_version"])
    _run([uv, "python", "install", python_version])

    venv = WORK_ROOT / "venv"
    _run([uv, "venv", "--python", python_version, str(venv)])
    py = venv / "bin/python"
    if not py.is_file():
        raise RuntimeError("isolated Python environment was not created")

    repo = WORK_ROOT / "repo"
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(repo)])
    _run(["git", "checkout", "--detach", str(manifest["source_revision"])], cwd=repo)
    observed_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if observed_revision != manifest["source_revision"]:
        raise RuntimeError("Kaggle checkout differs from binding-owned source revision")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError("Kaggle source checkout is not clean")

    constraints = repo / "requirements/nsq-kumar2024-promoted-py311.constraints.txt"
    if _sha256(constraints) != manifest["constraints_sha256"]:
        raise RuntimeError("promoted constraints file differs from launch manifest")

    pip_base = [
        uv,
        "pip",
        "install",
        "--python",
        str(py),
        "--constraint",
        str(constraints),
    ]
    _run([*pip_base, "pip==26.2.1", "setuptools==79.0.1", "wheel==0.48.0"])
    editable = [
        uv,
        "pip",
        "install",
        "--python",
        str(py),
        "--no-build-isolation",
        "--constraint",
        str(constraints),
    ]
    for package in (
        "packages/neuros-core",
        "packages/neuros-drivers",
        "packages/neuros-models",
        "packages/neuros-foundation[evidence,braindecode-evidence]",
        "packages/orion",
        "packages/neuros",
    ):
        _run([*editable, "-e", package], cwd=repo)

    _run(
        [
            str(py),
            "scripts/evidence/validate_promoted_environment.py",
            "--constraints",
            str(constraints),
        ],
        cwd=repo,
    )

    run_env = dict(os.environ)
    run_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _run(
        [
            str(py),
            "-P",
            "-s",
            "-m",
            "neuros.evidence.kumar2024_gpu_cloud",
            "verify-binding",
            "--output",
            str(gpu_binding_root),
        ],
        cwd=repo,
        env=run_env,
    )

    worker_root = OUTPUT_ROOT / "gpu-worker"
    _run(
        [
            str(py),
            "-P",
            "-s",
            "-m",
            "neuros.evidence.kumar2024_gpu_cloud",
            "run-worker",
            "--binding",
            str(gpu_binding_root),
            "--shard-spec-sha256",
            str(manifest["shard_spec_sha256"]),
            "--output",
            str(worker_root),
        ],
        cwd=repo,
        env=run_env,
    )
    _run(
        [
            str(py),
            "-P",
            "-s",
            "-m",
            "neuros.evidence.kumar2024_gpu_cloud",
            "verify-worker",
            "--binding",
            str(gpu_binding_root),
            "--output",
            str(worker_root),
        ],
        cwd=repo,
        env=run_env,
    )

    receipt = json.loads((worker_root / "gpu_receipt.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "phase": "systems-preflight",
        "source_revision": manifest["source_revision"],
        "gpu_binding_sha256": manifest["gpu_binding_sha256"],
        "binding_bundle_sha256": manifest["binding_bundle_sha256"],
        "shard_spec_sha256": manifest["shard_spec_sha256"],
        "gpu_worker_receipt_sha256": receipt["gpu_worker_receipt_sha256"],
        "worker_bundle_sha256": receipt["worker_bundle_sha256"],
        "device_name": receipt["accelerator_observation"]["device_name"],
        "compute_capability": receipt["accelerator_observation"]["compute_capability"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "statuses": receipt["statuses"],
        "numerical_result_interpretable": False,
        "score_fields_read_by_bootstrap": False,
        "go_no_go_inputs": receipt["go_no_go_inputs"],
        "forbidden_go_no_go_inputs": receipt["forbidden_go_no_go_inputs"],
    }
    (OUTPUT_ROOT / "kaggle_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.make_archive(
        "/kaggle/working/neuros_kumar2024_gpu_preflight",
        "zip",
        root_dir=OUTPUT_ROOT,
    )
    shutil.copy2(
        OUTPUT_ROOT / "kaggle_run_summary.json",
        "/kaggle/working/kaggle_run_summary.json",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
