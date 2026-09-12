from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "evidence"
    / "admit_kumar2024_external_qualification.py"
)
SPEC = importlib.util.spec_from_file_location(
    "neuros_kumar2024_external_admission_test",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def verified(**overrides):
    values = {
        "verified": True,
        "transport_provider": "local-linux",
        "binding_input_mode": "verified_archive",
        "transport_source_revision": "a" * 40,
        "transport_script_sha256": sha("transport-script"),
        "external_bundle_sha256": sha("outer-bundle"),
        "qualification_sha256": sha("qualification"),
        "worker_bundle_sha256": sha("worker-bundle"),
        "shard_result_sha256": sha("shard-result"),
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
    }
    values.update(overrides)
    return values


def test_admission_is_deterministic_score_blind_and_frozen():
    first = admission.build_admission(
        verified(), verifier_script_sha256=sha("verifier")
    )
    second = admission.build_admission(
        verified(), verifier_script_sha256=sha("verifier")
    )
    assert first == second
    assert first["admission_sha256"] == second["admission_sha256"]
    assert first["source_revision"] == admission.verifier.SOURCE_REVISION
    assert (
        first["environment_authority_sha256"]
        == admission.verifier.ENVIRONMENT_AUTHORITY_SHA256
    )
    assert first["shard_spec_sha256"] == admission.verifier.SHARD_SPEC_SHA256
    assert first["transport_structurally_qualified"] is True
    assert first["scientific_outcomes_inspected"] is False
    assert first["numerical_result_interpretable"] is False
    assert first["external_floor_claim_generated"] is False
    assert first["production_fleet_authorized"] is False
    assert first["orion_comparison_permitted"] is False


def test_admission_rejects_verifier_contract_widening_or_claim_promotion():
    widened = verified()
    widened["accuracy"] = 0.99
    with pytest.raises(ValueError, match="output contract drifted"):
        admission.build_admission(widened, verifier_script_sha256=sha("verifier"))

    with pytest.raises(ValueError, match="numerical_result_interpretable=false"):
        admission.build_admission(
            verified(numerical_result_interpretable=True),
            verifier_script_sha256=sha("verifier"),
        )


def test_admission_rejects_unverified_or_malformed_transport_identity():
    with pytest.raises(ValueError, match="not independently verified"):
        admission.build_admission(
            verified(verified=False), verifier_script_sha256=sha("verifier")
        )
    with pytest.raises(ValueError, match="40-character"):
        admission.build_admission(
            verified(transport_source_revision="not-a-revision"),
            verifier_script_sha256=sha("verifier"),
        )


def test_admit_calls_independent_verifier_and_writes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    transport = tmp_path / "transport.sh"
    transport.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    transport_sha = hashlib.sha256(transport.read_bytes()).hexdigest()
    expected_revision = "b" * 40
    expected = verified(
        transport_source_revision=expected_revision,
        transport_script_sha256=transport_sha,
    )
    observed: dict[str, object] = {}

    def fake_verify(root, *, transport_script, expected_transport_revision):
        observed["root"] = Path(root)
        observed["transport_script"] = Path(transport_script)
        observed["revision"] = expected_transport_revision
        return expected

    monkeypatch.setattr(admission.verifier, "verify", fake_verify)
    output = tmp_path / "admission.json"
    receipt = admission.admit(
        tmp_path / "qualification",
        transport_script=transport,
        expected_transport_revision=expected_revision,
        output=output,
    )
    assert output.is_file()
    assert observed["transport_script"] == transport
    assert observed["revision"] == expected_revision
    assert receipt["transport_script_sha256"] == transport_sha

    with pytest.raises(FileExistsError):
        admission.admit(
            tmp_path / "qualification",
            transport_script=transport,
            expected_transport_revision=expected_revision,
            output=output,
        )


def test_verify_admission_recomputes_identity_and_binds_current_verifier(tmp_path: Path):
    receipt = admission.build_admission(
        verified(),
        verifier_script_sha256=admission._file_sha256(admission.VERIFIER_PATH),
    )
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    checked = admission.verify_admission(path)
    assert checked["admission_sha256"] == receipt["admission_sha256"]

    tampered = dict(receipt)
    tampered["transport_provider"] = "nvidia-brev"
    path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        admission.verify_admission(path)


def test_historical_verifier_requires_explicit_opt_out(tmp_path: Path):
    receipt = admission.build_admission(
        verified(), verifier_script_sha256=sha("historical-verifier")
    )
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="different verifier byte identity"):
        admission.verify_admission(path)
    assert (
        admission.verify_admission(path, require_current_verifier=False)[
            "admission_sha256"
        ]
        == receipt["admission_sha256"]
    )
