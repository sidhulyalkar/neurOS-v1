from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from neuros.research import swarm_live_eval
from neuros.research._canonical import canonical_sha256
from neuros.research.nim import NimCallRecord
from neuros.research.nim_observed import NimTokenUsage
from neuros.research.nim_provider import DOCUMENTED_NVIDIA_CHAT_MODELS
from neuros.research.swarm import run_council
from neuros.research.swarm_benchmark_cases import build_benchmark_task
from neuros.research.swarm_live_eval import (
    SCHEDULE_POLICY,
    SMOKE_CASE_IDS,
    EvaluationRunManifest,
    ObservedNvidiaCouncilTransport,
    build_counterbalanced_schedule,
    build_evaluation_configurations,
    council_run_from_dict,
)

REV = "4" * 64
REPO = "sidhulyalkar/neurOS-v1"
ENDPOINT = "https://integrate.api.nvidia.com/v1"
REFERENCE_MODEL = DOCUMENTED_NVIDIA_CHAT_MODELS[0]


def _provider_qualification() -> dict[str, object]:
    probes = []
    for index, model in enumerate(DOCUMENTED_NVIDIA_CHAT_MODELS):
        if index == 0:
            probes.append(
                {
                    "model": model,
                    "status": "qualified",
                    "status_code": None,
                    "response_sha256": "6" * 64,
                    "error_excerpt": None,
                }
            )
        else:
            probes.append(
                {
                    "model": model,
                    "status": "transport_error",
                    "status_code": None,
                    "response_sha256": None,
                    "error_excerpt": "synthetic offline unavailable route",
                }
            )
    payload: dict[str, object] = {
        "schema_version": 1,
        "endpoint": ENDPOINT,
        "discovery_mode": "synthetic+bounded_chat_probe",
        "documented_candidates": list(DOCUMENTED_NVIDIA_CHAT_MODELS),
        "catalog_models_sha256": None,
        "catalog_error": None,
        "probes": probes,
        "qualified_models": [REFERENCE_MODEL],
        "discovery_budget": {
            "timeout_seconds_per_attempt": 20.0,
            "max_attempts_per_route": 2,
        },
        "authority_boundary": "synthetic test qualification",
    }
    payload["fingerprint"] = canonical_sha256(payload)
    return payload


QUALIFICATION = _provider_qualification()
PROVIDER = str(QUALIFICATION["fingerprint"])


def _finding(defect_id="D01"):
    return {
        "finding_id": defect_id,
        "severity": "high",
        "category": "reproducibility",
        "claim": "specific concern",
        "evidence": "benchmark stimulus",
        "falsification_test": "run deterministic check",
        "proposed_repair": "repair",
        "confidence": 0.9,
        "requires_human_judgment": False,
        "reference": "benchmark",
    }


class _Client:
    endpoint = ENDPOINT

    def __init__(self, *, fail_role=None, invalid_role=None, unknown_role=None):  # type: ignore[no-untyped-def]
        self.fail_role = fail_role
        self.invalid_role = invalid_role
        self.unknown_role = unknown_role
        self.calls = []

    def chat_json_observed(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        if kwargs["role"] == self.fail_role:
            raise TimeoutError("synthetic timeout")
        if kwargs["role"] == self.invalid_role:
            payload = {"findings": [{"unexpected": True}]}
        elif kwargs["role"] == self.unknown_role:
            payload = {"findings": [_finding("D99")]}
        else:
            payload = {"findings": [_finding()]}
        index = len(self.calls)
        request_payload = {
            "model": kwargs["model"],
            "messages": [
                {"role": "system", "content": kwargs["system_prompt"]},
                {"role": "user", "content": kwargs["user_prompt"]},
            ],
            "temperature": float(kwargs["temperature"]),
            "max_tokens": int(kwargs["max_tokens"]),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        combined_prompt = f'{kwargs["system_prompt"]}\n\n{kwargs["user_prompt"]}'
        record = NimCallRecord(
            role=kwargs["role"],
            model=kwargs["model"],
            endpoint=self.endpoint,
            prompt_sha256=sha256(combined_prompt.encode("utf-8")).hexdigest(),
            request_sha256=canonical_sha256(request_payload),
            response_sha256=f"{index + 200:064x}",
            response_text="{}",
        )
        return (
            payload,
            record,
            NimTokenUsage(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                usage_sha256=f"{index + 300:064x}",
            ),
        )


def _single_configuration():
    return build_evaluation_configurations((REFERENCE_MODEL,))[0]


def _task(case_id="case-001"):
    return build_benchmark_task(case_id, repository=REPO, source_revision=REV)


def _successful_single_manifest() -> EvaluationRunManifest:
    configuration = _single_configuration()
    transport = ObservedNvidiaCouncilTransport(
        _Client(),
        provider_qualification_fingerprint=PROVIDER,
    )
    run = asyncio.run(run_council(_task(), configuration.members, transport, require_all=False))
    return EvaluationRunManifest(
        repository=REPO,
        source_revision=REV,
        configuration=configuration,
        provider_qualification_fingerprint=PROVIDER,
        case_ids=("case-001",),
        runs=(("case-001", run),),
        receipts=transport.receipts,
        case_wall_latency_ms=(("case-001", 7),),
    )


def _successful_smoke_single_manifest() -> EvaluationRunManifest:
    configuration = _single_configuration()
    transport = ObservedNvidiaCouncilTransport(
        _Client(),
        provider_qualification_fingerprint=PROVIDER,
    )
    runs = []
    timings = []
    for index, case_id in enumerate(SMOKE_CASE_IDS, start=1):
        run = asyncio.run(
            run_council(_task(case_id), configuration.members, transport, require_all=False)
        )
        runs.append((case_id, run))
        timings.append((case_id, index * 7))
    return EvaluationRunManifest(
        repository=REPO,
        source_revision=REV,
        configuration=configuration,
        provider_qualification_fingerprint=PROVIDER,
        case_ids=SMOKE_CASE_IDS,
        runs=tuple(runs),
        receipts=transport.receipts,
        case_wall_latency_ms=tuple(timings),
    )


def _scorer_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "06_score_nim_swarm_live_eval.py"


def _scorer_module():
    path = _scorer_path()
    spec = importlib.util.spec_from_file_location("neuros_score_live_eval_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load live-evaluation scorer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_identity() -> dict[str, object]:
    return {
        "repository": REPO,
        "source_revision": REV,
        "provider_qualification": QUALIFICATION,
    }


def _rehash_receipt_and_manifest(manifest: dict[str, object], receipt: dict[str, object]) -> None:
    receipt_unhashed = dict(receipt)
    del receipt_unhashed["receipt_sha256"]
    receipt["receipt_sha256"] = canonical_sha256(receipt_unhashed)
    manifest_unhashed = dict(manifest)
    del manifest_unhashed["manifest_sha256"]
    manifest["manifest_sha256"] = canonical_sha256(manifest_unhashed)


def test_configuration_plan_freezes_reference_before_outcomes():
    configs = build_evaluation_configurations(DOCUMENTED_NVIDIA_CHAT_MODELS)
    assert [config.kind for config in configs] == [
        "single_reference",
        "homogeneous_five_role",
        "heterogeneous_five_role",
    ]
    assert configs[0].members[0].model == REFERENCE_MODEL
    assert {member.model for member in configs[1].members} == {REFERENCE_MODEL}
    assert len({member.model for member in configs[2].members}) == 3


def test_one_qualified_route_omits_fake_heterogeneous_configuration():
    configs = build_evaluation_configurations((REFERENCE_MODEL,))
    assert [config.kind for config in configs] == [
        "single_reference",
        "homogeneous_five_role",
    ]


def test_counterbalanced_schedule_rotates_configuration_order_without_outcomes():
    configs = build_evaluation_configurations(DOCUMENTED_NVIDIA_CHAT_MODELS[:2])
    schedule = build_counterbalanced_schedule(configs)
    assert SCHEDULE_POLICY == "deterministic_rotating_configuration_order_v1"
    width = len(configs)
    expected = [config.configuration_id for config in configs]
    assert [configuration_id for _, configuration_id in schedule[:width]] == expected
    assert [
        configuration_id for _, configuration_id in schedule[width : 2 * width]
    ] == expected[1:] + expected[:1]
    assert [
        configuration_id for _, configuration_id in schedule[2 * width : 3 * width]
    ] == expected[2:] + expected[:2]


def test_smoke_slice_is_fixed_public_and_small():
    assert len(SMOKE_CASE_IDS) == 3
    assert len(set(SMOKE_CASE_IDS)) == 3
    for case_id in SMOKE_CASE_IDS:
        assert _task(case_id).to_dict()["public_context"]["case"]["case_id"] == case_id


def test_model_facing_live_eval_module_never_imports_scorer_answer_key():
    source = inspect.getsource(swarm_live_eval)
    assert "from .swarm_benchmark import" not in source
    assert "_GROUND_TRUTH" not in source


def test_scorer_has_no_top_level_ground_truth_import():
    tree = ast.parse(_scorer_path().read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.module != "neuros.research.swarm_benchmark"


def test_successful_transport_receipt_binds_exact_review_and_wall_timing():
    manifest = _successful_single_manifest()
    run = manifest.runs[0][1]
    receipt = manifest.receipts[0]
    assert len(run.reviews) == 1
    assert receipt.outcome == "success"
    assert receipt.parsed_response_sha256 == run.reviews[0].response_sha256
    assert receipt.token_usage.total_tokens == 12
    assert manifest.to_dict()["case_wall_latency_ms"] == {"case-001": 7}
    assert manifest.to_dict()["scientific_promotion_authority"] is False


def test_manifest_rejects_missing_or_negative_case_wall_timing():
    manifest = _successful_single_manifest()
    with pytest.raises(ValueError, match="case wall timings"):
        EvaluationRunManifest(
            repository=manifest.repository,
            source_revision=manifest.source_revision,
            configuration=manifest.configuration,
            provider_qualification_fingerprint=manifest.provider_qualification_fingerprint,
            case_ids=manifest.case_ids,
            runs=manifest.runs,
            receipts=manifest.receipts,
            case_wall_latency_ms=(),
        )
    with pytest.raises(ValueError, match="non-negative"):
        EvaluationRunManifest(
            repository=manifest.repository,
            source_revision=manifest.source_revision,
            configuration=manifest.configuration,
            provider_qualification_fingerprint=manifest.provider_qualification_fingerprint,
            case_ids=manifest.case_ids,
            runs=manifest.runs,
            receipts=manifest.receipts,
            case_wall_latency_ms=(("case-001", -1),),
        )


def test_manifest_rejects_tampered_parsed_response_identity():
    manifest = _successful_single_manifest()
    forged = replace(manifest.receipts[0], parsed_response_sha256="9" * 64)
    with pytest.raises(ValueError, match="parsed-response identity mismatch"):
        EvaluationRunManifest(
            repository=manifest.repository,
            source_revision=manifest.source_revision,
            configuration=manifest.configuration,
            provider_qualification_fingerprint=manifest.provider_qualification_fingerprint,
            case_ids=manifest.case_ids,
            runs=manifest.runs,
            receipts=(forged,),
            case_wall_latency_ms=manifest.case_wall_latency_ms,
        )


def test_provider_failure_preserves_failure_without_inventing_call_evidence():
    configuration = build_evaluation_configurations((REFERENCE_MODEL,))[1]
    transport = ObservedNvidiaCouncilTransport(
        _Client(fail_role="swarm:implementation"),
        provider_qualification_fingerprint=PROVIDER,
    )
    run = asyncio.run(run_council(_task(), configuration.members, transport, require_all=False))
    assert len(run.reviews) == 4
    assert run.failed_members == ("implementation:4:TimeoutError",)
    failed = [receipt for receipt in transport.receipts if receipt.outcome != "success"]
    assert len(failed) == 1
    assert failed[0].outcome == "provider_failure"
    assert failed[0].request_sha256 is None
    assert failed[0].token_usage.provider_reported is False


def test_review_validation_failure_preserves_hashes_and_consumed_tokens():
    configuration = _single_configuration()
    transport = ObservedNvidiaCouncilTransport(
        _Client(invalid_role="swarm:generalist"),
        provider_qualification_fingerprint=PROVIDER,
    )
    run = asyncio.run(run_council(_task(), configuration.members, transport, require_all=False))
    assert run.reviews == ()
    assert run.failed_members == ("generalist:1:ValueError",)
    receipt = transport.receipts[0]
    assert receipt.outcome == "review_validation_failure"
    assert receipt.request_sha256 is not None
    assert receipt.response_sha256 is not None
    assert receipt.parsed_response_sha256 is not None
    assert receipt.token_usage.total_tokens == 12


def test_unknown_public_taxonomy_label_becomes_review_validation_failure():
    configuration = _single_configuration()
    transport = ObservedNvidiaCouncilTransport(
        _Client(unknown_role="swarm:generalist"),
        provider_qualification_fingerprint=PROVIDER,
    )
    run = asyncio.run(run_council(_task(), configuration.members, transport, require_all=False))
    assert run.failed_members == ("generalist:1:ValueError",)
    assert transport.receipts[0].outcome == "review_validation_failure"


def test_manifest_rejects_duplicate_attempt_receipts():
    manifest = _successful_single_manifest()
    receipt = manifest.receipts[0]
    with pytest.raises(ValueError, match="exactly one attempt"):
        EvaluationRunManifest(
            repository=manifest.repository,
            source_revision=manifest.source_revision,
            configuration=manifest.configuration,
            provider_qualification_fingerprint=manifest.provider_qualification_fingerprint,
            case_ids=manifest.case_ids,
            runs=manifest.runs,
            receipts=(receipt, receipt),
            case_wall_latency_ms=manifest.case_wall_latency_ms,
        )


def test_scorer_keeps_exact_three_case_boundary():
    scorer = _scorer_module()
    with pytest.raises(ValueError, match="case slice mismatch"):
        scorer._validated_manifest(
            _successful_single_manifest().to_dict(),
            raw=_raw_identity(),
            expected_configuration=_single_configuration(),
        )


def test_scorer_rejects_rehashed_receipt_member_substitution():
    scorer = _scorer_module()
    manifest = _successful_smoke_single_manifest().to_dict()
    receipt = manifest["receipts"][0]
    receipt["role"] = "forged-role"
    _rehash_receipt_and_manifest(manifest, receipt)
    with pytest.raises(ValueError, match="receipt member identity differs"):
        scorer._validated_manifest(
            manifest,
            raw=_raw_identity(),
            expected_configuration=_single_configuration(),
        )


def test_scorer_rejects_rehashed_parsed_response_substitution():
    scorer = _scorer_module()
    manifest = _successful_smoke_single_manifest().to_dict()
    receipt = manifest["receipts"][0]
    receipt["parsed_response_sha256"] = "9" * 64
    _rehash_receipt_and_manifest(manifest, receipt)
    with pytest.raises(ValueError, match="parsed-response identity differs"):
        scorer._validated_manifest(
            manifest,
            raw=_raw_identity(),
            expected_configuration=_single_configuration(),
        )


def test_scorer_rejects_rehashed_request_substitution():
    scorer = _scorer_module()
    manifest = _successful_smoke_single_manifest().to_dict()
    receipt = manifest["receipts"][0]
    receipt["request_sha256"] = "8" * 64
    _rehash_receipt_and_manifest(manifest, receipt)
    with pytest.raises(ValueError, match="request identity differs"):
        scorer._validated_manifest(
            manifest,
            raw=_raw_identity(),
            expected_configuration=_single_configuration(),
        )


def test_scorer_rejects_semantically_impossible_rehashed_provider_failure():
    scorer = _scorer_module()
    manifest = _successful_smoke_single_manifest().to_dict()
    receipt = manifest["receipts"][0]
    receipt["outcome"] = "provider_failure"
    receipt["error_class"] = "TimeoutError"
    _rehash_receipt_and_manifest(manifest, receipt)
    with pytest.raises(ValueError, match="provider failure cannot claim"):
        scorer._validated_manifest(
            manifest,
            raw=_raw_identity(),
            expected_configuration=_single_configuration(),
        )


def test_scorer_rejects_self_hashed_fake_provider_roster():
    scorer = _scorer_module()
    forged = dict(QUALIFICATION)
    forged["documented_candidates"] = ["invented/model"]
    forged_unhashed = dict(forged)
    del forged_unhashed["fingerprint"]
    forged["fingerprint"] = canonical_sha256(forged_unhashed)
    with pytest.raises(ValueError, match="candidate roster differs"):
        scorer._validated_provider_qualification(forged)


def test_council_run_serialization_round_trip_revalidates_hash():
    run = _successful_single_manifest().runs[0][1]
    restored = council_run_from_dict(run.to_dict())
    assert restored.to_dict() == run.to_dict()
    forged = run.to_dict()
    forged["run_sha256"] = canonical_sha256({"forged": True})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        council_run_from_dict(forged)
