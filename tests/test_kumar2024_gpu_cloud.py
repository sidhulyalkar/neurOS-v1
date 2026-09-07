from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuros.evidence import kumar2024_gpu_cloud as gpu
from neuros.evidence import kumar2024_promoted_binding as binding
from neuros.evidence import kumar2024_promoted_worker as worker


def test_cuda_config_is_explicit_without_mutating_cpu_default():
    cpu = binding.promoted_materialization_config()
    cuda = gpu._cuda_config()
    assert cpu.device == "cpu"
    assert cuda.device == "cuda"
    assert cpu.to_dict()["braindecode"]["device"] == "cpu"
    assert cuda.to_dict()["braindecode"]["device"] == "cuda"
    assert cpu.sha256 != cuda.sha256


def test_cuda_scope_updates_binder_and_worker_and_restores_on_error():
    original_binding = binding.promoted_materialization_config
    original_worker = worker.promoted_materialization_config
    with pytest.raises(RuntimeError, match="probe"):
        with gpu.promoted_cuda_scope():
            assert binding.promoted_materialization_config().device == "cuda"
            assert worker.promoted_materialization_config().device == "cuda"
            raise RuntimeError("probe")
    assert binding.promoted_materialization_config is original_binding
    assert worker.promoted_materialization_config is original_worker


def test_fixed_preflight_selection_is_non_adaptive(tmp_path: Path, monkeypatch):
    binding_root = tmp_path / "binding"
    binding_root.mkdir()
    target = {
        "subject": 1,
        "target_session": "5",
        "split_seed": 2026,
        "method_id": "braindecode-eegnet",
        "model_seed": 31415,
        "budgets_per_class": [0, 1, 2, 5, 10],
        "shard_spec_sha256": "a" * 64,
    }
    decoy = {**target, "subject": 10, "shard_spec_sha256": "b" * 64}
    (binding_root / "execution_plan.json").write_text(
        json.dumps({"template": {"shards": [decoy, target]}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        gpu,
        "verify_gpu_binding",
        lambda _: {"binding_root": str(binding_root)},
    )
    assert gpu.select_preflight_shard(tmp_path) == target


def test_preflight_selection_rejects_ambiguous_match(tmp_path: Path, monkeypatch):
    binding_root = tmp_path / "binding"
    binding_root.mkdir()
    target = {
        "subject": 1,
        "target_session": "5",
        "split_seed": 2026,
        "method_id": "braindecode-eegnet",
        "model_seed": 31415,
        "budgets_per_class": [0, 1, 2, 5, 10],
        "shard_spec_sha256": "a" * 64,
    }
    (binding_root / "execution_plan.json").write_text(
        json.dumps({"template": {"shards": [target, target]}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        gpu,
        "verify_gpu_binding",
        lambda _: {"binding_root": str(binding_root)},
    )
    with pytest.raises(ValueError, match="not unique"):
        gpu.select_preflight_shard(tmp_path)


def test_t4_policy_rejects_other_accelerators():
    gpu._require_t4({"device_name": "Tesla T4"})
    with pytest.raises(RuntimeError, match="T4-class"):
        gpu._require_t4({"device_name": "NVIDIA A100-SXM4-40GB"})


def test_gpu_worker_receipt_identity_is_order_independent_for_mappings():
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}
    assert gpu._gpu_worker_identity(first) == gpu._gpu_worker_identity(second)
