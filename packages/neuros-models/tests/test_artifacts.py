from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from neuros.models import (
    EEGNetModel,
    TorchDecoderStateSnapshot,
    parameter_state_sha256_from_tensors,
    read_torch_decoder_artifact,
    restore_torch_decoder_state,
    snapshot_torch_decoder_state,
    torch_parameter_state_sha256,
    write_torch_decoder_artifact,
)


def _data(seed: int = 31):
    rng = np.random.default_rng(seed)
    source_x = rng.normal(size=(16, 2, 64)).astype(np.float32)
    source_y = np.asarray([0, 1] * 8, dtype=np.int64)
    calibration_x = rng.normal(size=(8, 2, 64)).astype(np.float32)
    calibration_y = np.asarray([0, 1] * 4, dtype=np.int64)
    evaluation_x = rng.normal(size=(6, 2, 64)).astype(np.float32)
    return source_x, source_y, calibration_x, calibration_y, evaluation_x


def _model(*, temporal_filters: int = 4) -> EEGNetModel:
    return EEGNetModel(
        n_channels=2,
        n_classes=2,
        temporal_filters=temporal_filters,
        depth_multiplier=1,
        separable_filters=4,
        temporal_kernel=15,
        separable_kernel=7,
        dropout=0.25,
        n_epochs=1,
        batch_size=4,
        device="cpu",
        random_state=17,
    )


def _legacy_parameter_hash(model: EEGNetModel) -> str:
    digest = hashlib.sha256()
    state = model.analysis_model().state_dict()
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_parameter_hash_preserves_existing_longitudinal_semantics():
    source_x, source_y, *_ = _data()
    model = _model()
    model.train(source_x, source_y)

    assert torch_parameter_state_sha256(model) == _legacy_parameter_hash(model)
    snapshot = snapshot_torch_decoder_state(model)
    assert snapshot.parameter_state_sha256 == _legacy_parameter_hash(model)
    assert snapshot.parameter_state_sha256 == parameter_state_sha256_from_tensors(
        snapshot.tensors
    )


def test_snapshot_restore_recovers_exact_predictions_and_learning_state():
    source_x, source_y, calibration_x, calibration_y, evaluation_x = _data()
    model = _model()
    model.train(source_x, source_y)
    source = model.snapshot_state(metadata={"stage": "source"})
    source_probability = model.predict_proba(evaluation_x)

    model.train(calibration_x, calibration_y)
    assert model.snapshot_state().learning_state_sha256 != source.learning_state_sha256

    model.restore_state(source)
    restored = model.snapshot_state(metadata={"stage": "source"})
    restored_probability = model.predict_proba(evaluation_x)

    assert restored.parameter_state_sha256 == source.parameter_state_sha256
    assert restored.learning_state_sha256 == source.learning_state_sha256
    assert np.array_equal(restored_probability, source_probability)
    assert restored.training_history == source.training_history


def test_restore_reproduces_future_stochastic_finetuning_trajectory_on_cpu():
    source_x, source_y, calibration_x, calibration_y, evaluation_x = _data()
    model = _model()
    model.train(source_x, source_y)
    source = model.snapshot_state()

    model.train(calibration_x, calibration_y)
    first_adapted = model.snapshot_state()
    first_probability = model.predict_proba(evaluation_x)

    model.restore_state(source)
    model.train(calibration_x, calibration_y)
    second_adapted = model.snapshot_state()
    second_probability = model.predict_proba(evaluation_x)

    assert second_adapted.parameter_state_sha256 == first_adapted.parameter_state_sha256
    assert second_adapted.learning_state_sha256 == first_adapted.learning_state_sha256
    assert np.array_equal(second_probability, first_probability)


def test_incompatible_restore_fails_before_live_state_mutation():
    source_x, source_y, *_ = _data()
    source_model = _model()
    source_model.train(source_x, source_y)
    source = source_model.snapshot_state()

    incompatible = _model(temporal_filters=5)
    incompatible.train(source_x, source_y)
    before = incompatible.snapshot_state()

    with pytest.raises(ValueError, match="configuration differs"):
        incompatible.restore_state(source)

    after = incompatible.snapshot_state()
    assert after.parameter_state_sha256 == before.parameter_state_sha256
    assert after.learning_state_sha256 == before.learning_state_sha256


def test_forged_snapshot_hash_fails_at_construction():
    source_x, source_y, *_ = _data()
    model = _model()
    model.train(source_x, source_y)
    snapshot = model.snapshot_state()
    tensors = {name: np.array(value, copy=True) for name, value in snapshot.tensors.items()}
    first = sorted(tensors)[0]
    mutated = tensors[first].reshape(-1)
    if np.issubdtype(mutated.dtype, np.floating):
        mutated[0] += np.asarray(0.125, dtype=mutated.dtype)
    else:
        mutated[0] += np.asarray(1, dtype=mutated.dtype)

    with pytest.raises(ValueError, match="parameter_state_sha256"):
        TorchDecoderStateSnapshot(
            model_type=snapshot.model_type,
            model_version=snapshot.model_version,
            resolved_config=snapshot.resolved_config,
            analysis_manifest_fingerprint=snapshot.analysis_manifest_fingerprint,
            tensors=tensors,
            cpu_rng_state=snapshot.cpu_rng_state,
            cuda_rng_states=snapshot.cuda_rng_states,
            is_trained=snapshot.is_trained,
            training_history=snapshot.training_history,
            metadata=snapshot.metadata,
            parameter_state_sha256=snapshot.parameter_state_sha256,
            learning_state_sha256=snapshot.learning_state_sha256,
        )


def test_artifact_roundtrip_is_data_only_and_byte_deterministic(tmp_path: Path):
    source_x, source_y, *_ = _data()
    model = _model()
    model.train(source_x, source_y)
    snapshot = model.snapshot_state(
        metadata={
            "input_schema": {"axes": ["batch", "channel", "time"], "dtype": "float32"},
            "provenance": {"dataset": "synthetic-fixture", "calibration_per_class": 0},
        }
    )

    first = write_torch_decoder_artifact(snapshot, tmp_path / "first")
    second = write_torch_decoder_artifact(snapshot, tmp_path / "second")

    assert _tree_bytes(first) == _tree_bytes(second)
    assert set(path.suffix for path in first.rglob("*") if path.is_file()) <= {".json", ".npy"}

    restored = read_torch_decoder_artifact(first)
    assert restored.snapshot_fingerprint == snapshot.snapshot_fingerprint
    assert restored.parameter_state_sha256 == snapshot.parameter_state_sha256
    assert restored.learning_state_sha256 == snapshot.learning_state_sha256
    assert restored.manifest() == snapshot.manifest()

    target = _model()
    restore_torch_decoder_state(target, restored)
    assert target.snapshot_state().learning_state_sha256 == snapshot.learning_state_sha256


def test_corrupt_artifact_tensor_file_fails_integrity_check(tmp_path: Path):
    source_x, source_y, *_ = _data()
    model = _model()
    model.train(source_x, source_y)
    root = model.export_artifact(tmp_path / "artifact")
    tensor_file = sorted((root / "tensors").glob("*.npy"))[0]
    with tensor_file.open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        read_torch_decoder_artifact(root)


def test_manifest_tampering_fails_before_artifact_data_is_trusted(tmp_path: Path):
    source_x, source_y, *_ = _data()
    model = _model()
    model.train(source_x, source_y)
    root = model.export_artifact(tmp_path / "artifact")
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model_type"] = "DifferentModel"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_manifest_sha256"):
        read_torch_decoder_artifact(root)


def test_snapshot_metadata_is_deeply_immutable():
    source_x, source_y, *_ = _data()
    model = _model()
    model.train(source_x, source_y)
    metadata = {"provenance": {"subjects": [1, 2]}}
    snapshot = model.snapshot_state(metadata=metadata)
    fingerprint = snapshot.snapshot_fingerprint

    metadata["provenance"]["subjects"].append(3)
    assert snapshot.snapshot_fingerprint == fingerprint
    assert snapshot.manifest()["metadata"]["provenance"]["subjects"] == [1, 2]
    with pytest.raises(TypeError):
        snapshot.metadata["provenance"]["subjects"] += (3,)
