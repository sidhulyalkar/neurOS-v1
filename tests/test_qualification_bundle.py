from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuros.contracts import DecoderOutput
from neuros.qualification import (
    QUALIFICATION_SCHEMA_VERSION,
    _OutputDigest,
    qualify_config,
    reproduce_qualification,
    verify_qualification_bundle,
)


@pytest.mark.asyncio
async def test_semantic_output_digest_ignores_latency_not_decision():
    first = _OutputDigest()
    second = _OutputDigest()
    changed = _OutputDigest()

    await first(
        DecoderOutput(
            prediction=1,
            model_id="semantic-test",
            model_version="1",
            inference_time_ns=100,
            metadata={"score": 0.75},
        )
    )
    await second(
        DecoderOutput(
            prediction=1,
            model_id="semantic-test",
            model_version="1",
            inference_time_ns=999999,
            metadata={"score": 0.75},
        )
    )
    await changed(
        DecoderOutput(
            prediction=0,
            model_id="semantic-test",
            model_version="1",
            inference_time_ns=100,
            metadata={"score": 0.75},
        )
    )

    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["sha256"] != changed.to_dict()["sha256"]


@pytest.mark.asyncio
async def test_qualification_bundle_is_sealed_and_reproducible(tmp_path: Path):
    config = Path("configs/examples/mock_bci.yaml").resolve()
    bundle = tmp_path / "qualification"

    result = await qualify_config(
        config,
        bundle,
        session_id="ci-qualification",
        duration_s=0.05,
    )

    assert result["schema_version"] == QUALIFICATION_SCHEMA_VERSION
    assert result["status"] == "complete"
    assert result["evidence_tier"] == "integration"
    assert result["decoder_outputs"]["count"] > 0
    assert result["claim_boundary"]["runtime_record_replay_qualified"] is True
    assert result["claim_boundary"]["hardware_qualified"] is False
    assert result["claim_boundary"]["closed_loop_qualified"] is False
    assert result["claim_boundary"]["clinical_qualified"] is False

    required = {
        "manifest.json",
        "artifact_hashes.json",
        "config.yaml",
        "config.json",
        "environment.json",
        "compatibility.json",
        "devices.json",
        "clocks.json",
        "model.json",
        "runtime.json",
        "decoder_outputs.json",
        "session/manifest.json",
    }
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    assert required <= actual

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["qualification_scope"] == "runtime-record-replay"
    assert manifest["reproducibility"]["archive_integrity_verified"] is True
    assert manifest["reproducibility"]["replay_completed"] is True
    assert manifest["reproducibility"]["decoder_output_digest_exact"] is True
    assert manifest["config_semantic_hash"] == manifest["archive_config_hash"]
    assert manifest["record_summary"]["archive"] == "session"
    assert ".staging-" not in json.dumps(manifest, sort_keys=True)

    model = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
    assert model["artifact_bound"] is False
    assert "learned-weight identity is not claimed" in model["limitation"]

    verification = verify_qualification_bundle(bundle)
    assert verification["integrity"] == "verified"
    assert verification["frame_count"] > 0

    reproduced = await reproduce_qualification(bundle)
    assert reproduced["reproduced"] is True
    assert reproduced["runtime_state"] == "stopped"
    assert reproduced["decoder_outputs"] == result["decoder_outputs"]


@pytest.mark.asyncio
async def test_qualification_refuses_existing_output_without_overwrite(tmp_path: Path):
    config = Path("configs/examples/mock_bci.yaml").resolve()
    bundle = tmp_path / "qualification"
    bundle.mkdir()
    (bundle / "sentinel.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        await qualify_config(config, bundle, duration_s=0.01)

    assert (bundle / "sentinel.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
async def test_qualification_explicit_overwrite_replaces_directory(tmp_path: Path):
    config = Path("configs/examples/mock_bci.yaml").resolve()
    bundle = tmp_path / "qualification"
    bundle.mkdir()
    (bundle / "obsolete.txt").write_text("old", encoding="utf-8")

    await qualify_config(config, bundle, duration_s=0.03, overwrite=True)

    assert not (bundle / "obsolete.txt").exists()
    assert verify_qualification_bundle(bundle)["integrity"] == "verified"


@pytest.mark.asyncio
async def test_qualification_detects_tampering_before_reproduction(tmp_path: Path):
    config = Path("configs/examples/mock_bci.yaml").resolve()
    bundle = tmp_path / "qualification"
    await qualify_config(config, bundle, duration_s=0.03)

    runtime_path = bundle / "runtime.json"
    runtime_path.write_text(runtime_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(IOError, match="artifact (size|hash) mismatch"):
        verify_qualification_bundle(bundle)

    with pytest.raises(IOError, match="artifact (size|hash) mismatch"):
        await reproduce_qualification(bundle)
