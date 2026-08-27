from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuros.models import EEGNetModel, ModelArtifactStore, ModelInputContract, export_model_artifact

GIT_SHA = "1" * 40


def _model(seed: int) -> tuple[EEGNetModel, np.ndarray]:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(12, 2, 64)).astype("float32")
    y = (X[:, 0].mean(axis=1) > 0).astype("int64")
    model = EEGNetModel(
        n_channels=2,
        n_classes=2,
        temporal_filters=4,
        depth_multiplier=1,
        separable_filters=4,
        temporal_kernel=7,
        separable_kernel=3,
        n_epochs=1,
        batch_size=6,
        device="cpu",
        random_state=seed,
    )
    model.train(X, y)
    return model, X


def _contract() -> ModelInputContract:
    return ModelInputContract(
        axes=("batch", "channel", "time"),
        shape=(None, 2, 64),
        dtype="float32",
        channel_names=("C3", "C4"),
        sample_rate_hz=256.0,
        signal_unit="uV",
    )


def test_store_publishes_by_content_address_and_rolls_back_only_the_ref(tmp_path: Path):
    first_model, X = _model(3)
    second_model, _ = _model(4)
    first = export_model_artifact(
        first_model,
        tmp_path / "first-export",
        artifact_id="decoder-a",
        input_contract=_contract(),
        git_sha=GIT_SHA,
    )
    second = export_model_artifact(
        second_model,
        tmp_path / "second-export",
        artifact_id="decoder-b",
        input_contract=_contract(),
        git_sha=GIT_SHA,
    )
    assert first.artifact_sha256 != second.artifact_sha256

    store = ModelArtifactStore(tmp_path / "store")
    assert store.publish(tmp_path / "first-export").artifact_sha256 == first.artifact_sha256
    assert store.publish(tmp_path / "second-export").artifact_sha256 == second.artifact_sha256
    assert store.publish(tmp_path / "first-export").artifact_sha256 == first.artifact_sha256

    first_path = store.artifact_path(first.artifact_sha256)
    before_manifest = (first_path / "manifest.json").read_bytes()
    before_weights = (first_path / "weights.safetensors").read_bytes()

    store.activate("active", second.artifact_sha256)
    assert store.active_sha256("active") == second.artifact_sha256
    store.rollback("active", first.artifact_sha256)
    assert store.active_sha256("active") == first.artifact_sha256

    assert (first_path / "manifest.json").read_bytes() == before_manifest
    assert (first_path / "weights.safetensors").read_bytes() == before_weights
    assert store.list_artifacts() == tuple(sorted((first.artifact_sha256, second.artifact_sha256)))

    loaded = store.load("active", device="cpu")
    np.testing.assert_allclose(
        loaded.predict_logits(X[:3]), first_model.predict_logits(X[:3]), rtol=0, atol=0
    )
    assert loaded.artifact_manifest.artifact_sha256 == first.artifact_sha256


def test_store_can_resolve_full_sha_without_a_mutable_ref(tmp_path: Path):
    model, X = _model(5)
    manifest = export_model_artifact(
        model,
        tmp_path / "export",
        artifact_id="direct-sha",
        input_contract=_contract(),
        git_sha=GIT_SHA,
    )
    store = ModelArtifactStore(tmp_path / "store")
    store.publish(tmp_path / "export")
    path, resolved = store.resolve(manifest.artifact_sha256)
    assert path.name == manifest.artifact_sha256
    assert resolved.artifact_sha256 == manifest.artifact_sha256
    np.testing.assert_allclose(
        store.load(manifest.artifact_sha256).predict_logits(X[:2]),
        model.predict_logits(X[:2]),
        rtol=0,
        atol=0,
    )


def test_store_refs_reject_path_traversal_and_sha_ambiguity(tmp_path: Path):
    store = ModelArtifactStore(tmp_path / "store")
    with pytest.raises(ValueError, match="artifact ref"):
        store.activate("../active", "a" * 64)
    with pytest.raises(ValueError, match="SHA-shaped"):
        store.activate("a" * 64, "b" * 64)


def test_store_ref_tampering_fails_closed(tmp_path: Path):
    model, _X = _model(6)
    manifest = export_model_artifact(
        model,
        tmp_path / "export",
        artifact_id="ref-tamper",
        input_contract=_contract(),
        git_sha=GIT_SHA,
    )
    store = ModelArtifactStore(tmp_path / "store")
    store.publish(tmp_path / "export")
    store.activate("active", manifest.artifact_sha256)
    ref = store.refs_dir / "active.json"
    ref.write_text(
        '{"schema_version":1,"ref":"different","artifact_sha256":"'
        + manifest.artifact_sha256
        + '"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requested ref name"):
        store.resolve("active")


def test_store_rejects_symlinked_refs_and_artifact_addresses(tmp_path: Path):
    model, _X = _model(7)
    manifest = export_model_artifact(
        model,
        tmp_path / "export",
        artifact_id="symlink-tamper",
        input_contract=_contract(),
        git_sha=GIT_SHA,
    )
    store = ModelArtifactStore(tmp_path / "store")
    store.publish(tmp_path / "export")
    store.activate("active", manifest.artifact_sha256)

    original_ref = store.refs_dir / "active.json"
    external_ref = tmp_path / "external-ref.json"
    external_ref.write_bytes(original_ref.read_bytes())
    original_ref.unlink()
    try:
        original_ref.symlink_to(external_ref)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="refs cannot be symbolic links"):
        store.resolve("active")

    original_ref.unlink()
    store.activate("active", manifest.artifact_sha256)
    artifact_path = store.artifact_path(manifest.artifact_sha256)
    moved = tmp_path / "moved-artifact"
    artifact_path.rename(moved)
    artifact_path.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        store.resolve(manifest.artifact_sha256)
