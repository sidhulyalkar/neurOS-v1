"""CUDA/Kaggle execution authority for the promoted Kumar2024 study.

This adapter leaves the canonical CPU promoted path unchanged.  It derives a
separate CUDA-bound no-model authority, executes only archived EEGNet shards,
and wraps every GPU result in a systems receipt.  The compatibility scope is
process-local and sequential: it temporarily projects the canonical promoted
configuration onto ``device='cuda'`` while calling the already-qualified binder
and worker implementations.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import kumar2024 as base
from . import kumar2024_promoted_binding as promoted_binding
from . import kumar2024_promoted_worker as promoted_worker
from .kumar2024_comparison import Kumar2024ComparisonPlan, promoted_external_floor_plan

GPU_BINDING_SCHEMA = "neuros.nsq_kumar2024_gpu_binding.v1"
GPU_WORKER_SCHEMA = "neuros.nsq_kumar2024_gpu_worker.v1"
CUDA_DEVICE = "cuda"
KAGGLE_MACHINE_SHAPE = "NvidiaTeslaT4"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
UV_VERSION = "0.12.10"


def _require_sha256(name: str, value: Any) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _cuda_config(plan: Kumar2024ComparisonPlan | None = None) -> base.Kumar2024StudyConfig:
    plan = plan or promoted_external_floor_plan()
    return replace(promoted_binding.promoted_materialization_config(plan), device=CUDA_DEVICE)


@contextlib.contextmanager
def promoted_cuda_scope() -> Iterator[None]:
    """Project the canonical promoted config onto CUDA for one sequential process.

    The canonical CPU functions remain untouched outside this context.  Both the
    binding module and worker module hold direct references to the config helper,
    so both references are replaced and restored together.
    """

    original_binding = promoted_binding.promoted_materialization_config
    original_worker = promoted_worker.promoted_materialization_config

    def cuda_config(plan: Kumar2024ComparisonPlan | None = None):
        return replace(original_binding(plan), device=CUDA_DEVICE)

    promoted_binding.promoted_materialization_config = cuda_config
    promoted_worker.promoted_materialization_config = cuda_config
    try:
        yield
    finally:
        promoted_binding.promoted_materialization_config = original_binding
        promoted_worker.promoted_materialization_config = original_worker


def configure_cuda_determinism(*, require_available: bool) -> dict[str, Any]:
    """Apply the CUDA determinism policy used by both binder and worker."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - cloud/runtime integration
        raise ImportError("Kumar2024 GPU authority requires torch") from exc

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False

    available = bool(torch.cuda.is_available())
    if require_available and not available:
        raise RuntimeError("CUDA-bound Kumar2024 worker requires torch.cuda.is_available()")

    observation: dict[str, Any] = {
        "requested_device": CUDA_DEVICE,
        "cuda_available": available,
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda or "none"),
        "cudnn_runtime": str(torch.backends.cudnn.version() or "none"),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        observation["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        observation["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    if available:
        observation.update(
            {
                "device_count": int(torch.cuda.device_count()),
                "device_name": str(torch.cuda.get_device_name(0)),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return observation


def _assert_cuda_binding_payload(binding_root: Path) -> dict[str, Any]:
    manifest = json.loads((binding_root / "binding_manifest.json").read_text(encoding="utf-8"))
    materialization = json.loads((binding_root / "materialization.json").read_text(encoding="utf-8"))
    methods = json.loads((binding_root / "method_specs.json").read_text(encoding="utf-8"))

    config = manifest.get("materialization_config") or {}
    braindecode = config.get("braindecode") or {}
    if braindecode.get("device") != CUDA_DEVICE:
        raise ValueError("GPU binding materialization config is not CUDA-bound")

    environment = materialization["authority"]["environment"]
    accelerator = environment.get("accelerator_runtime") or {}
    flags = environment.get("deterministic_flags") or {}
    if accelerator.get("requested_device") != CUDA_DEVICE:
        raise ValueError("GPU binding environment does not request CUDA")
    expected_flags = {
        "torch_deterministic_algorithms": "true",
        "cudnn_deterministic": "true",
        "cudnn_benchmark": "false",
    }
    drift = {
        key: {"expected": value, "observed": flags.get(key)}
        for key, value in expected_flags.items()
        if flags.get(key) != value
    }
    if drift:
        raise ValueError(f"GPU binding determinism flags drifted: {drift}")

    eegnet = [
        item for item in methods["method_specs"]
        if str(item["realization_key"]).startswith("braindecode-eegnet/")
    ]
    if len(eegnet) != 3:
        raise ValueError("GPU binding requires exactly three EEGNet realizations")
    if any(item["method_spec"]["metadata"].get("device") != CUDA_DEVICE for item in eegnet):
        raise ValueError("GPU binding EEGNet method specs are not CUDA-bound")
    return {
        "manifest": manifest,
        "environment": environment,
        "eegnet_method_specs": eegnet,
    }


def _gpu_binding_identity(authority_payload: Mapping[str, Any]) -> str:
    return base._identity_sha256(GPU_BINDING_SCHEMA, authority_payload)


def run_gpu_binding(output: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Create a CUDA-bound no-model promoted binding plus GPU policy seal."""

    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty GPU binding output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    binding_root = root / "binding"

    binding_runtime_observation = configure_cuda_determinism(require_available=False)
    with promoted_cuda_scope():
        result = promoted_binding.run_promoted_binding(binding_root, overwrite=overwrite)
        verified = promoted_binding.verify_promoted_binding_bundle(binding_root)

    checked = _assert_cuda_binding_payload(binding_root)
    source_revision = str(checked["manifest"]["source_revision"])
    authority_payload = {
        "schema_version": 1,
        "artifact_kind": "cuda_no_model_promoted_binding_authority",
        "source_revision": source_revision,
        "binding_bundle_sha256": _require_sha256("binding bundle SHA", verified["bundle_sha256"]),
        "comparison_plan_sha256": _require_sha256(
            "comparison plan SHA", checked["manifest"]["comparison_plan_sha256"]
        ),
        "execution_plan_sha256": _require_sha256(
            "execution plan SHA", checked["manifest"]["execution_plan_sha256"]
        ),
        "environment_authority_sha256": _require_sha256(
            "environment SHA", checked["manifest"]["environment_authority_sha256"]
        ),
        "cuda_policy": {
            "requested_device": CUDA_DEVICE,
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "binding_runtime_policy_observation": binding_runtime_observation,
        "cloud_execution": {
            "provider": "kaggle",
            "machine_shape": KAGGLE_MACHINE_SHAPE,
            "bootstrap_uv_version": UV_VERSION,
            "python_version": checked["environment"]["python"]["version"],
            "internet_required": True,
        },
        "model_execution_performed": False,
        "final_assessment_predictions_generated": False,
        "final_assessment_metrics_generated": False,
        "claim_boundary": (
            "CUDA/cloud execution authority only; this artifact contains no model result and "
            "does not strengthen efficacy, method-ranking, or external-floor claims"
        ),
    }
    gpu_binding_sha = _gpu_binding_identity(authority_payload)
    authority = {**authority_payload, "gpu_binding_sha256": gpu_binding_sha}
    base._json_dump(root / "gpu_authority.json", authority)
    hashes = {
        "schema_version": 1,
        "binding_bundle_sha256": verified["bundle_sha256"],
        "gpu_authority_file_sha256": base._file_sha256(root / "gpu_authority.json"),
        "gpu_binding_sha256": gpu_binding_sha,
    }
    base._json_dump(root / "gpu_artifact_hashes.json", hashes)
    return {
        **result,
        "gpu_binding_sha256": gpu_binding_sha,
        "gpu_binding_root": str(root),
        "binding_root": str(binding_root),
    }


def verify_gpu_binding(output: str | Path) -> dict[str, Any]:
    root = Path(output).resolve()
    binding_root = root / "binding"
    with promoted_cuda_scope():
        inner = promoted_binding.verify_promoted_binding_bundle(binding_root)
    checked = _assert_cuda_binding_payload(binding_root)
    authority = json.loads((root / "gpu_authority.json").read_text(encoding="utf-8"))
    hashes = json.loads((root / "gpu_artifact_hashes.json").read_text(encoding="utf-8"))
    payload = dict(authority)
    declared = _require_sha256("gpu_binding_sha256", payload.pop("gpu_binding_sha256", ""))
    expected = _gpu_binding_identity(payload)
    if declared != expected or hashes.get("gpu_binding_sha256") != expected:
        raise ValueError("GPU binding root identity mismatch")
    if hashes.get("binding_bundle_sha256") != inner["bundle_sha256"]:
        raise ValueError("GPU wrapper and inner promoted binding disagree")
    if hashes.get("gpu_authority_file_sha256") != base._file_sha256(root / "gpu_authority.json"):
        raise ValueError("GPU authority file hash mismatch")
    if authority.get("source_revision") != checked["manifest"].get("source_revision"):
        raise ValueError("GPU authority source revision differs from promoted binding")
    if authority.get("environment_authority_sha256") != checked["manifest"].get(
        "environment_authority_sha256"
    ):
        raise ValueError("GPU authority environment differs from promoted binding")
    policy_observation = authority.get("binding_runtime_policy_observation") or {}
    policy = authority.get("cuda_policy") or {}
    for key in (
        "cublas_workspace_config",
        "torch_deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    ):
        if policy_observation.get(key) != policy.get(key):
            raise ValueError(f"GPU binding runtime policy observation differs for {key}")
    return {
        "verified": True,
        "gpu_binding_sha256": expected,
        "binding_bundle_sha256": inner["bundle_sha256"],
        "execution_plan_sha256": inner["execution_plan_sha256"],
        "source_revision": authority["source_revision"],
        "machine_shape": authority["cloud_execution"]["machine_shape"],
        "binding_root": str(binding_root),
    }


def select_preflight_shard(gpu_binding_root: str | Path) -> dict[str, Any]:
    """Select one fixed, non-adaptive EEGNet shard for systems preflight."""

    verified = verify_gpu_binding(gpu_binding_root)
    execution = json.loads(
        (Path(verified["binding_root"]) / "execution_plan.json").read_text(encoding="utf-8")
    )
    matches = [
        shard for shard in execution["template"]["shards"]
        if int(shard["subject"]) == 1
        and str(shard["target_session"]) == "5"
        and int(shard["split_seed"]) == 2026
        and str(shard["method_id"]) == "braindecode-eegnet"
        and int(shard.get("model_seed")) == 31415
    ]
    if len(matches) != 1:
        raise ValueError("fixed GPU systems-preflight shard is not unique in binding")
    shard = matches[0]
    if tuple(shard.get("budgets_per_class", ())) != (0, 1, 2, 5, 10):
        raise ValueError("GPU preflight shard does not contain the complete budget frontier")
    return shard


def _gpu_worker_identity(payload: Mapping[str, Any]) -> str:
    return base._identity_sha256(GPU_WORKER_SCHEMA, payload)


def _require_t4(observation: Mapping[str, Any]) -> None:
    name = str(observation.get("device_name", ""))
    if "T4" not in name.upper():
        raise RuntimeError(
            f"Kaggle GPU preflight requires a T4-class device; observed {name!r}"
        )


def run_gpu_worker(
    gpu_binding_root: str | Path,
    shard_spec_sha256: str,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one archived EEGNet shard under the sealed CUDA/Kaggle authority."""

    gpu_binding = verify_gpu_binding(gpu_binding_root)
    observation = configure_cuda_determinism(require_available=True)
    _require_t4(observation)
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty GPU worker output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    inner_root = root / "worker"

    with promoted_cuda_scope():
        assignment = promoted_worker.load_promoted_worker_assignment(
            gpu_binding["binding_root"], shard_spec_sha256
        )
        if assignment.shard.method_id != "braindecode-eegnet":
            raise ValueError("GPU cloud worker accepts only archived EEGNet shards")
        started = time.monotonic()
        result = promoted_worker.run_promoted_worker(
            gpu_binding["binding_root"],
            shard_spec_sha256,
            inner_root,
            overwrite=overwrite,
        )
        elapsed = time.monotonic() - started
        inner_verified = promoted_worker.verify_promoted_worker_bundle(
            inner_root, binding_root=gpu_binding["binding_root"]
        )

    receipt_payload = {
        "schema_version": 1,
        "artifact_kind": "kaggle_t4_promoted_worker_systems_receipt",
        "gpu_binding_sha256": gpu_binding["gpu_binding_sha256"],
        "binding_bundle_sha256": gpu_binding["binding_bundle_sha256"],
        "source_revision": gpu_binding["source_revision"],
        "shard_spec_sha256": _require_sha256("shard_spec_sha256", shard_spec_sha256),
        "worker_bundle_sha256": inner_verified["worker_bundle_sha256"],
        "shard_result_sha256": inner_verified["shard_result_sha256"],
        "accelerator_observation": observation,
        "elapsed_seconds": elapsed,
        "attempted_budgets": inner_verified["attempted_budgets"],
        "statuses": inner_verified["statuses"],
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
        "go_no_go_inputs": [
            "worker_bundle_verification",
            "wall_clock_seconds",
            "accelerator_identity",
            "environment_identity",
        ],
        "forbidden_go_no_go_inputs": [
            "balanced_accuracy",
            "scientific_score",
            "method_ranking",
            "final_assessment_metric",
        ],
        "claim_boundary": (
            "systems qualification of one preregistered GPU worker frontier only; numerical "
            "results remain quarantined and cannot guide continuation, efficacy, ranking, "
            "external-floor, or ORION decisions"
        ),
    }
    receipt_sha = _gpu_worker_identity(receipt_payload)
    receipt = {**receipt_payload, "gpu_worker_receipt_sha256": receipt_sha}
    base._json_dump(root / "gpu_receipt.json", receipt)
    hashes = {
        "schema_version": 1,
        "gpu_worker_receipt_sha256": receipt_sha,
        "gpu_receipt_file_sha256": base._file_sha256(root / "gpu_receipt.json"),
        "worker_bundle_sha256": inner_verified["worker_bundle_sha256"],
    }
    base._json_dump(root / "gpu_artifact_hashes.json", hashes)
    return {
        "verified": True,
        "gpu_worker_receipt_sha256": receipt_sha,
        "worker_bundle_sha256": inner_verified["worker_bundle_sha256"],
        "gpu_binding_sha256": gpu_binding["gpu_binding_sha256"],
        "shard_spec_sha256": shard_spec_sha256,
        "elapsed_seconds": elapsed,
        "device_name": observation["device_name"],
        "numerical_result_interpretable": False,
        "output": str(root),
        **{key: result[key] for key in ("attempted_budgets", "statuses")},
    }


def verify_gpu_worker(
    output: str | Path,
    *,
    gpu_binding_root: str | Path,
) -> dict[str, Any]:
    root = Path(output).resolve()
    gpu_binding = verify_gpu_binding(gpu_binding_root)
    receipt = json.loads((root / "gpu_receipt.json").read_text(encoding="utf-8"))
    hashes = json.loads((root / "gpu_artifact_hashes.json").read_text(encoding="utf-8"))
    payload = dict(receipt)
    declared = _require_sha256(
        "gpu_worker_receipt_sha256", payload.pop("gpu_worker_receipt_sha256", "")
    )
    expected = _gpu_worker_identity(payload)
    if declared != expected or hashes.get("gpu_worker_receipt_sha256") != expected:
        raise ValueError("GPU worker receipt root identity mismatch")
    with promoted_cuda_scope():
        inner = promoted_worker.verify_promoted_worker_bundle(
            root / "worker", binding_root=gpu_binding["binding_root"]
        )
    if hashes.get("worker_bundle_sha256") != inner["worker_bundle_sha256"]:
        raise ValueError("GPU receipt and inner worker bundle disagree")
    if receipt.get("gpu_binding_sha256") != gpu_binding["gpu_binding_sha256"]:
        raise ValueError("GPU worker names a different GPU binding")
    if receipt.get("numerical_result_interpretable") is not False:
        raise ValueError("GPU systems preflight may not promote numerical interpretation")
    observation = receipt.get("accelerator_observation") or {}
    _require_t4(observation)
    authority = json.loads(
        (Path(gpu_binding_root).resolve() / "gpu_authority.json").read_text(encoding="utf-8")
    )
    policy = authority.get("cuda_policy") or {}
    for key in (
        "cublas_workspace_config",
        "torch_deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    ):
        if observation.get(key) != policy.get(key):
            raise ValueError(f"GPU worker runtime policy differs for {key}")
    return {
        "verified": True,
        "gpu_worker_receipt_sha256": expected,
        "worker_bundle_sha256": inner["worker_bundle_sha256"],
        "shard_spec_sha256": inner["shard_spec_sha256"],
        "elapsed_seconds": float(receipt["elapsed_seconds"]),
        "device_name": receipt["accelerator_observation"]["device_name"],
        "numerical_result_interpretable": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kumar2024 CUDA/Kaggle authority adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    bind = sub.add_parser("bind")
    bind.add_argument("--output", required=True)
    bind.add_argument("--overwrite", action="store_true")

    verify_binding = sub.add_parser("verify-binding")
    verify_binding.add_argument("--output", required=True)

    select = sub.add_parser("select-preflight")
    select.add_argument("--binding", required=True)

    worker = sub.add_parser("run-worker")
    worker.add_argument("--binding", required=True)
    worker.add_argument("--shard-spec-sha256", required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument("--overwrite", action="store_true")

    verify_worker = sub.add_parser("verify-worker")
    verify_worker.add_argument("--binding", required=True)
    verify_worker.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bind":
        result = run_gpu_binding(args.output, overwrite=args.overwrite)
    elif args.command == "verify-binding":
        result = verify_gpu_binding(args.output)
    elif args.command == "select-preflight":
        result = select_preflight_shard(args.binding)
    elif args.command == "run-worker":
        result = run_gpu_worker(
            args.binding,
            args.shard_spec_sha256,
            args.output,
            overwrite=args.overwrite,
        )
    else:
        result = verify_gpu_worker(args.output, gpu_binding_root=args.binding)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_DEVICE",
    "GPU_BINDING_SCHEMA",
    "GPU_WORKER_SCHEMA",
    "KAGGLE_MACHINE_SHAPE",
    "UV_VERSION",
    "configure_cuda_determinism",
    "main",
    "promoted_cuda_scope",
    "run_gpu_binding",
    "run_gpu_worker",
    "select_preflight_shard",
    "verify_gpu_binding",
    "verify_gpu_worker",
]
