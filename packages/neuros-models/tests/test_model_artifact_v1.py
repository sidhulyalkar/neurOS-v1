from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neuros.contracts import StreamDescriptor
from neuros.models import (
    EEGNetModel,
    InterpretabilityManifest,
    ModelInputContract,
    ModelOutputContract,
    export_model_artifact,
    load_model_artifact,
    verify_model_artifact,
)

GIT_SHA = "a" * 40
TRAIN_SHA = "b" * 64
EVAL_SHA = "c" * 64
STUDY_SHA = "d" * 64


def _trained_eegnet() -> tuple[EEGNetModel, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(16, 2, 64)).astype("float32")
    y = (X[:, 0, :].mean(axis=1) > 0).astype("int64")
    model = EEGNetModel(
        n_channels=2,
        n_classes=2,
        temporal_filters=4,
        depth_multiplier=1,
        separable_filters=4,
        temporal_kernel=7,
        separable_kernel=3,
        n_epochs=1,
        batch_size=8,
        device="cpu",
        random_state=7,
    )
    model.train(X, y)
    return model, X, y


def _descriptor() -> StreamDescriptor:
    return StreamDescriptor(
        stream_id="fixture-eeg",
        modality="eeg",
        sample_rate_hz=256.0,
        channel_names=("C3", "C4"),
        units=("uV", "uV"),
    )


def _contract(*, bind_descriptor: bool = False) -> ModelInputContract:
    descriptor = _descriptor()
    return ModelInputContract(
        axes=("batch", "channel", "time"),
        shape=(None, 2, 64),
        dtype="float32",
        channel_names=("C3", "C4"),
        sample_rate_hz=256.0,
        signal_unit="uV",
        stream_descriptor_sha256=descriptor.fingerprint() if bind_descriptor else None,
        metadata={"montage": "fixture-two-channel"},
    )


def _export(path: Path, *, contract: ModelInputContract | None = None):
    model, X, y = _trained_eegnet()
    manifest = export_model_artifact(
        model,
        path,
        artifact_id="fixture-eegnet-v1",
        input_contract=_contract() if contract is None else contract,
        git_sha=GIT_SHA,
        training_authority_sha256s=(TRAIN_SHA,),
        evaluation_authority_sha256s=(EVAL_SHA,),
        scientific_study_sha256=STUDY_SHA,
        metadata={"purpose": "artifact-contract-regression"},
    )
    return model, X, y, manifest


def _write_manifest(path: Path, manifest) -> None:
    (path / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_interpretability_manifest_has_full_immutable_identity():
    notes = {"causal": "held-out intervention only"}
    manifest = InterpretabilityManifest(
        model_type="fixture",
        architecture="fixture",
        backend="fixture",
        input_axes=("batch", "feature"),
        output_semantics="class logits",
        method_notes=notes,
    )
    full = manifest.sha256()
    notes["causal"] = "mutated"
    assert len(full) == 64
    assert manifest.sha256() == full
    assert manifest.fingerprint() == full[:16]
    with pytest.raises(TypeError):
        manifest.method_notes["new"] = "value"  # type: ignore[index]


def test_export_verify_and_reload_are_content_addressed_and_equivalent(tmp_path: Path):
    original, X, _y, exported = _export(tmp_path / "artifact-a")
    verified = verify_model_artifact(tmp_path / "artifact-a")
    loaded = load_model_artifact(tmp_path / "artifact-a", device="cpu")

    assert verified.artifact_sha256 == exported.artifact_sha256
    assert len(verified.artifact_sha256) == 64
    assert len(verified.manifest_sha256) == 64
    assert len(verified.weights_sha256) == 64
    assert len(verified.interpretability_manifest_sha256) == 64
    assert verified.display_fingerprint == verified.artifact_sha256[:16]
    assert verified.training_authority_sha256s == (TRAIN_SHA,)
    assert verified.evaluation_authority_sha256s == (EVAL_SHA,)
    assert verified.scientific_study_sha256 == STUDY_SHA
    assert verified.output_contract.probability_semantics == "uncalibrated_softmax"
    assert verified.output_contract.class_labels == ("0", "1")
    assert {path.name for path in (tmp_path / "artifact-a").iterdir()} == {
        "manifest.json",
        "weights.safetensors",
    }
    assert not list((tmp_path / "artifact-a").glob("*.pkl"))

    np.testing.assert_allclose(
        loaded.predict_logits(X[:4]), original.predict_logits(X[:4]), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        loaded.predict_proba(X[:4]), original.predict_proba(X[:4]), rtol=0, atol=0
    )
    np.testing.assert_allclose(loaded.encode(X[:4]), original.encode(X[:4]), rtol=0, atol=0)
    output = loaded.infer(X[:1])
    assert output.metadata["promoted_artifact"] is True
    assert output.metadata["artifact_sha256"] == verified.artifact_sha256
    assert output.metadata["interpretability_manifest_sha256"] == (
        verified.interpretability_manifest_sha256
    )
    assert output.metadata["probability_semantics"] == "uncalibrated_softmax"
    assert tuple(output.metadata["class_labels"]) == ("0", "1")


def test_same_model_state_and_provenance_have_same_artifact_identity(tmp_path: Path):
    model, _X, _y = _trained_eegnet()
    first = export_model_artifact(
        model,
        tmp_path / "first",
        artifact_id="deterministic",
        input_contract=_contract(),
        git_sha=GIT_SHA,
        training_authority_sha256s=(TRAIN_SHA,),
    )
    second = export_model_artifact(
        model,
        tmp_path / "second",
        artifact_id="deterministic",
        input_contract=_contract(),
        git_sha=GIT_SHA,
        training_authority_sha256s=(TRAIN_SHA,),
    )
    assert first.weights_sha256 == second.weights_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.artifact_sha256 == second.artifact_sha256
    assert (tmp_path / "first" / "weights.safetensors").read_bytes() == (
        tmp_path / "second" / "weights.safetensors"
    ).read_bytes()


def test_promoted_artifact_is_read_only_and_enforces_input_contract(tmp_path: Path):
    _model, X, y, _manifest = _export(tmp_path / "artifact")
    loaded = load_model_artifact(tmp_path / "artifact")
    assert loaded.capabilities.online_fit is False
    with pytest.raises(RuntimeError, match="immutable"):
        loaded.train(X, y)
    with pytest.raises(RuntimeError, match="immutable"):
        loaded.partial_fit(X, y)
    with pytest.raises(RuntimeError, match="immutable"):
        loaded.adapt(X)
    with pytest.raises(ValueError, match="dtype mismatch"):
        loaded.predict(X.astype("float64"))
    with pytest.raises(ValueError, match="shape mismatch"):
        loaded.predict(X[:, :, :-1])
    with pytest.raises(FileExistsError, match="immutable"):
        export_model_artifact(
            _model,
            tmp_path / "artifact",
            artifact_id="overwrite-attempt",
            input_contract=_contract(),
            git_sha=GIT_SHA,
        )


def test_analysis_model_is_a_detached_snapshot_not_live_deployment_state(tmp_path: Path):
    _model, X, _y, manifest = _export(tmp_path / "artifact")
    loaded = load_model_artifact(tmp_path / "artifact")
    before = loaded.predict_logits(X[:3]).copy()

    snapshot = loaded.analysis_model()
    assert all(parameter.requires_grad is False for parameter in snapshot.parameters())
    # Deliberately destroy the detached analysis snapshot.
    for parameter in snapshot.parameters():
        parameter.zero_()

    after = loaded.predict_logits(X[:3])
    np.testing.assert_allclose(after, before, rtol=0, atol=0)
    assert loaded.artifact_manifest.artifact_sha256 == manifest.artifact_sha256


def test_stream_descriptor_authority_can_be_bound_and_checked(tmp_path: Path):
    descriptor = _descriptor()
    _model, X, _y, _manifest = _export(
        tmp_path / "artifact", contract=_contract(bind_descriptor=True)
    )
    loaded = load_model_artifact(tmp_path / "artifact")
    loaded.validate_stream_descriptor(descriptor)
    loaded.predict(X[:1])

    wrong_channels = StreamDescriptor(
        stream_id="fixture-eeg",
        modality="eeg",
        sample_rate_hz=256.0,
        channel_names=("F3", "F4"),
        units=("uV", "uV"),
    )
    with pytest.raises(ValueError, match="descriptor SHA-256"):
        loaded.validate_stream_descriptor(wrong_channels)


def test_builtin_v1_cannot_claim_calibrated_probabilities_without_qualified_factory(tmp_path: Path):
    model, _X, _y = _trained_eegnet()
    calibrated = ModelOutputContract(
        class_labels=("left", "right"),
        probability_semantics="calibrated_probability",
        probability_calibration_method="temperature-scaling",
        probability_calibration_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="uncalibrated softmax"):
        export_model_artifact(
            model,
            tmp_path / "artifact",
            artifact_id="false-calibration-claim",
            input_contract=_contract(),
            output_contract=calibrated,
            git_sha=GIT_SHA,
        )


def test_output_contract_rejects_semantic_contradictions():
    with pytest.raises(ValueError, match="only be declared"):
        ModelOutputContract(
            class_labels=("0", "1"),
            probability_semantics="uncalibrated_softmax",
            probability_calibration_method="temperature-scaling",
        )
    with pytest.raises(ValueError, match="at least two unique"):
        ModelOutputContract(class_labels=("same", "same"))


def test_manifest_tampering_is_rejected_before_model_construction(tmp_path: Path):
    _model, _X, _y, _manifest = _export(tmp_path / "artifact")
    payload = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    payload["artifact_id"] = "forged"
    (tmp_path / "artifact" / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_sha256"):
        verify_model_artifact(tmp_path / "artifact")


def test_weight_tampering_extra_payloads_and_root_symlinks_are_rejected(tmp_path: Path):
    _model, _X, _y, _manifest = _export(tmp_path / "original")

    tampered = tmp_path / "tampered"
    shutil.copytree(tmp_path / "original", tampered)
    weights = tampered / "weights.safetensors"
    content = bytearray(weights.read_bytes())
    content[-1] ^= 0x01
    weights.write_bytes(bytes(content))
    with pytest.raises(ValueError, match="weights SHA-256"):
        verify_model_artifact(tampered)

    extra = tmp_path / "extra"
    shutil.copytree(tmp_path / "original", extra)
    (extra / "payload.pkl").write_bytes(b"not accepted")
    with pytest.raises(ValueError, match="exactly"):
        verify_model_artifact(extra)

    linked = tmp_path / "linked"
    try:
        linked.symlink_to(tmp_path / "original", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="root cannot be a symbolic link"):
        verify_model_artifact(linked)


def test_unknown_factory_cannot_turn_manifest_data_into_an_import(tmp_path: Path):
    _model, _X, _y, manifest = _export(tmp_path / "artifact")
    forged = replace(manifest, factory_id="attacker.module:Payload")
    _write_manifest(tmp_path / "artifact", forged)
    verified = verify_model_artifact(tmp_path / "artifact")
    assert verified.factory_id == "attacker.module:Payload"
    with pytest.raises(ValueError, match="not a built-in safe"):
        load_model_artifact(tmp_path / "artifact")


def test_exact_environment_versions_are_part_of_load_authority(tmp_path: Path):
    _model, _X, _y, manifest = _export(tmp_path / "artifact")
    versions = dict(manifest.package_versions)
    versions["neuros-models"] = "0.0.0-forged"
    incompatible = replace(manifest, package_versions=versions)
    _write_manifest(tmp_path / "artifact", incompatible)
    verify_model_artifact(tmp_path / "artifact")
    with pytest.raises(RuntimeError, match="artifact requires neuros-models"):
        load_model_artifact(tmp_path / "artifact")


def test_input_contract_rejects_ambiguous_channel_and_shape_authority():
    with pytest.raises(ValueError, match="channel_names length"):
        ModelInputContract(
            axes=("batch", "channel", "time"),
            shape=(None, 2, 64),
            channel_names=("C3",),
        )
    with pytest.raises(ValueError, match="integer without coercion"):
        ModelInputContract(
            axes=("batch", "channel", "time"),
            shape=(None, 2.0, 64),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="duplicate names"):
        ModelInputContract(
            axes=("batch", "channel", "channel"),
            shape=(None, 2, 2),
        )
