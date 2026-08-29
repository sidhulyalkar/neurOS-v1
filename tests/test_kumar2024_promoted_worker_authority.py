from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuros.evidence import kumar2024 as base
from neuros.evidence.kumar2024_promoted_binding import _seal_binding_bundle
from neuros.evidence.kumar2024_promoted_execution import PromotedShardResult
from neuros.evidence.kumar2024_promoted_worker import (
    _assert_runtime_authority,
    _factory_for_assignment,
    _seal_worker_bundle,
    _verify_subject_raw_materialization,
    load_promoted_worker_assignment,
    verify_promoted_worker_bundle,
)
from neuros.foundation_models.qualification import ExternalDecoderMethodSpec

_FIXTURE_PATH = Path(__file__).with_name("test_kumar2024_promoted_binding_authority.py")
_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_kumar2024_promoted_binding_test_fixture", _FIXTURE_PATH
)
if _FIXTURE_SPEC is None or _FIXTURE_SPEC.loader is None:
    raise RuntimeError("unable to load promoted binding synthetic fixture module")
_FIXTURE_MODULE = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURE_MODULE)
_write_synthetic_bundle = _FIXTURE_MODULE._write_synthetic_bundle


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding(root):
    execution = _write_synthetic_bundle(root)
    path = root / "materialization.json"
    payload = json.loads(path.read_text())
    payload["raw_selection"] = {
        "schema_version": 1,
        "loader_contract": "fixture-loader-contract",
        "selections": [
            {
                "schema_version": 1,
                "subject": 1,
                "raw_subject": 1,
                "original_protocol": "GR",
                "session": "0",
                "run": "0",
                "logical_path": "fixture/consumed.gdf",
            }
        ],
    }
    base._json_dump(path, payload)
    _seal_binding_bundle(root)
    return execution


def _mne_shard(execution):
    return next(
        shard
        for shard in execution.template.shards
        if shard.subject == 1 and shard.method_id == "mne-csp-lda"
    )


def test_worker_consumes_exact_archived_shard_case_and_method(tmp_path):
    execution = _binding(tmp_path)
    shard = _mne_shard(execution)
    assignment = load_promoted_worker_assignment(tmp_path, shard.sha256)
    assert assignment.shard.sha256 == shard.sha256
    assert assignment.execution_plan.sha256 == execution.sha256
    assert assignment.case_authority.authority_sha256 == execution.expected_case_authority_sha256(shard)
    assert assignment.archived_method_spec.sha256 == execution.expected_method_spec_sha256(shard)
    assert assignment.archived_processed_shard["shard_id"] == "subject=1"
    assert [item["logical_path"] for item in assignment.archived_raw_files] == [
        "fixture/consumed.gdf"
    ]


def test_worker_rejects_scheduler_supplied_foreign_shard(tmp_path):
    _binding(tmp_path)
    with pytest.raises(ValueError, match="not present in the promoted binding"):
        load_promoted_worker_assignment(tmp_path, _sha("foreign-shard"))


def test_worker_runtime_fails_closed_on_source_revision_drift(tmp_path, monkeypatch):
    execution = _binding(tmp_path)
    assignment = load_promoted_worker_assignment(tmp_path, _mne_shard(execution).sha256)
    monkeypatch.setattr(base, "_git_revision", lambda: "b" * 40)
    with pytest.raises(RuntimeError, match="source revision differs"):
        _assert_runtime_authority(assignment)


def test_worker_runtime_requires_exact_environment_sha(tmp_path, monkeypatch):
    execution = _binding(tmp_path)
    assignment = load_promoted_worker_assignment(tmp_path, _mne_shard(execution).sha256)
    monkeypatch.setattr(base, "_git_revision", lambda: "a" * 40)
    import neuros.evidence.kumar2024_promoted_worker as worker

    monkeypatch.setattr(
        worker,
        "_runtime_authority",
        lambda _config: SimpleNamespace(sha256=_sha("wrong-environment")),
    )
    with pytest.raises(RuntimeError, match="environment authority differs"):
        _assert_runtime_authority(assignment)


def test_worker_rehashes_only_consumed_subject_raw_files(tmp_path):
    execution = _binding(tmp_path)
    assignment = load_promoted_worker_assignment(tmp_path, _mne_shard(execution).sha256)

    class Record:
        def __init__(self, payload):
            self.payload = dict(payload)

        def to_dict(self):
            return dict(self.payload)

    current = SimpleNamespace(
        loader_contract="fixture-loader-contract",
        selections=[Record(item) for item in assignment.archived_raw_selections],
        authority=SimpleNamespace(files=[Record(item) for item in assignment.archived_raw_files]),
    )
    _verify_subject_raw_materialization(assignment, current)
    changed = dict(assignment.archived_raw_files[0])
    changed["sha256"] = _sha("changed-consumed-byte-content")
    current.authority.files = [Record(changed)]
    with pytest.raises(RuntimeError, match="consumed raw subject bytes differ"):
        _verify_subject_raw_materialization(assignment, current)


def test_worker_factory_must_reproduce_archived_method_spec_exactly(tmp_path, monkeypatch):
    execution = _binding(tmp_path)
    assignment = load_promoted_worker_assignment(tmp_path, _mne_shard(execution).sha256)
    import neuros.foundation_models.qualification_baselines as baselines

    monkeypatch.setattr(
        baselines,
        "MNECSPLDAFactory",
        lambda **_kwargs: SimpleNamespace(method_spec=assignment.archived_method_spec),
    )
    assert _factory_for_assignment(assignment, 512.0).method_spec.sha256 == (
        assignment.expected_method_spec_sha256
    )

    wrong = ExternalDecoderMethodSpec(
        method_id="mne-csp-lda",
        implementation="fixture.drifted",
        implementation_version="fixture=2",
        input_axes=("sample", "channel", "time"),
        probability_semantics="uncalibrated_probability",
        metadata={"fixture": "drift"},
    )
    monkeypatch.setattr(
        baselines,
        "MNECSPLDAFactory",
        lambda **_kwargs: SimpleNamespace(method_spec=wrong),
    )
    with pytest.raises(RuntimeError, match="does not reproduce the archived method-spec"):
        _factory_for_assignment(assignment, 512.0)


def _write_worker_bundle(root, assignment):
    shard = assignment.shard
    binding = assignment.execution_plan.binding
    rows, nsq_rows, role_rows = [], [], []
    for budget in shard.budgets_per_class:
        row_sha = _sha(f"qualification-budget-row|{shard.shard_id}|{budget}")
        rows.append(
            {
                "method_id": shard.method_id,
                "subject": shard.subject,
                "held_out_session": shard.target_session,
                "split_seed": shard.split_seed,
                "model_seed": shard.model_seed,
                "calibration_per_class": budget,
                "case_authority_sha256": assignment.expected_case_authority_sha256,
                "original_protocol": "GR",
                "status": "failed",
                "balanced_accuracy": None,
                "qualification_result_row_sha256": row_sha,
            }
        )
        nsq_rows.append(
            {
                "result_sha256": row_sha,
                "status": "failed",
                "method_id": shard.method_id,
                "calibration_per_class": budget,
                "protocol_sha256": binding.protocol_sha256,
                "case_authority_sha256": assignment.expected_case_authority_sha256,
                "method_spec_sha256": assignment.expected_method_spec_sha256,
                "score": None,
            }
        )
        role_rows.append(
            {
                "calibration_per_class": budget,
                "qualification_result_row_sha256": row_sha,
            }
        )
    row_hashes = {}
    for item in nsq_rows:
        payload = dict(item)
        payload.pop("result_sha256")
        digest = base._identity_sha256("neuros.qualification_budget_result.v3", payload)
        item["result_sha256"] = digest
        row_hashes[int(item["calibration_per_class"])] = digest
    for item in rows:
        item["qualification_result_row_sha256"] = row_hashes[
            int(item["calibration_per_class"])
        ]
    for item in role_rows:
        item["qualification_result_row_sha256"] = row_hashes[
            int(item["calibration_per_class"])
        ]

    shard_result = PromotedShardResult(
        execution_plan_sha256=assignment.execution_plan.sha256,
        shard_spec_sha256=shard.sha256,
        comparison_plan_sha256=assignment.comparison_plan.sha256,
        study_materialization_sha256=binding.study_materialization_sha256,
        environment_authority_sha256=binding.environment_authority_sha256,
        raw_materialization_sha256=binding.raw_materialization_sha256,
        dataset_lineage_sha256=binding.dataset_lineage_sha256,
        protocol_sha256=binding.protocol_sha256,
        preprocessing_authority_sha256=binding.preprocessing_authority_sha256,
        case_authority_sha256=assignment.expected_case_authority_sha256,
        method_spec_sha256=assignment.expected_method_spec_sha256,
        rows=tuple(rows),
    )
    result_payload = {
        "schema_version": 3,
        "protocol_sha256": binding.protocol_sha256,
        "case_authority_sha256": assignment.expected_case_authority_sha256,
        "method_spec_sha256": assignment.expected_method_spec_sha256,
        "execution_context_sha256": _sha("fixture-context"),
        "metric_scorecard_sha256": _sha("fixture-scorecard"),
        "rows": nsq_rows,
    }
    result_sha = base._identity_sha256(
        "neuros.qualification_case_result.v3", result_payload
    )
    result_payload["result_sha256"] = result_sha
    base._json_dump(
        root / "worker_manifest.json",
        {
            "schema_version": 1,
            "binding_bundle_sha256": assignment.binding_bundle_sha256,
            "source_revision": binding.source_revision,
            "execution_plan_sha256": assignment.execution_plan.sha256,
            "shard_spec_sha256": shard.sha256,
            "result_sha256": result_sha,
        },
    )
    base._json_dump(
        root / "case_result.json",
        {
            "schema_version": 1,
            "subject": shard.subject,
            "held_out_session": shard.target_session,
            "split_seed": shard.split_seed,
            "model_seed": shard.model_seed,
            "method_realization_key": shard.method_realization_key,
            "result": result_payload,
        },
    )
    base._json_dump(
        root / "observation_roles.json",
        {"schema_version": 1, "shard_spec_sha256": shard.sha256, "rows": role_rows},
    )
    base._json_dump(
        root / "shard_result.json",
        {**shard_result.to_dict(), "shard_result_sha256": shard_result.sha256},
    )
    _seal_worker_bundle(root)
    return shard_result


def test_worker_bundle_binds_back_to_assignment_and_atomic_frontier(tmp_path):
    binding_root = tmp_path / "binding"
    worker_root = tmp_path / "worker"
    binding_root.mkdir()
    worker_root.mkdir()
    execution = _binding(binding_root)
    assignment = load_promoted_worker_assignment(binding_root, _mne_shard(execution).sha256)
    shard_result = _write_worker_bundle(worker_root, assignment)
    verified = verify_promoted_worker_bundle(worker_root, binding_root=binding_root)
    assert verified["verified"] is True
    assert verified["binding_bundle_sha256"] == assignment.binding_bundle_sha256
    assert verified["shard_result_sha256"] == shard_result.sha256
    assert verified["attempted_budgets"] == list(assignment.shard.budgets_per_class)
    assert verified["statuses"] == ["failed"] * len(assignment.shard.budgets_per_class)


def test_worker_bundle_rejects_resealed_cross_file_result_drift(tmp_path):
    binding_root = tmp_path / "binding"
    worker_root = tmp_path / "worker"
    binding_root.mkdir()
    worker_root.mkdir()
    execution = _binding(binding_root)
    assignment = load_promoted_worker_assignment(binding_root, _mne_shard(execution).sha256)
    _write_worker_bundle(worker_root, assignment)
    case_path = worker_root / "case_result.json"
    case_payload = json.loads(case_path.read_text())
    row = case_payload["result"]["rows"][0]
    row["status"] = "oom"
    row_for_hash = dict(row)
    row_for_hash.pop("result_sha256")
    row["result_sha256"] = base._identity_sha256(
        "neuros.qualification_budget_result.v3", row_for_hash
    )
    result_for_hash = dict(case_payload["result"])
    result_for_hash.pop("result_sha256")
    case_payload["result"]["result_sha256"] = base._identity_sha256(
        "neuros.qualification_case_result.v3", result_for_hash
    )
    base._json_dump(case_path, case_payload)
    manifest_path = worker_root / "worker_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["result_sha256"] = case_payload["result"]["result_sha256"]
    base._json_dump(manifest_path, manifest)
    _seal_worker_bundle(worker_root)
    with pytest.raises(ValueError, match="flattened row does not bind"):
        verify_promoted_worker_bundle(worker_root, binding_root=binding_root)


def test_worker_bundle_rejects_resealed_nested_nsq_payload_drift(tmp_path):
    binding_root = tmp_path / "binding"
    worker_root = tmp_path / "worker"
    binding_root.mkdir()
    worker_root.mkdir()
    execution = _binding(binding_root)
    assignment = load_promoted_worker_assignment(binding_root, _mne_shard(execution).sha256)
    _write_worker_bundle(worker_root, assignment)
    path = worker_root / "case_result.json"
    payload = json.loads(path.read_text())
    payload["result"]["rows"][0]["qualification_model_state"] = {
        "metadata": {"tampered_after_execution": True}
    }
    base._json_dump(path, payload)
    _seal_worker_bundle(worker_root)
    with pytest.raises(ValueError, match="serialized NSQ budget-result SHA mismatch"):
        verify_promoted_worker_bundle(worker_root, binding_root=binding_root)
