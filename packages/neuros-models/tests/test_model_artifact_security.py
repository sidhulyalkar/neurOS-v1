from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neuros.models import (
    EEGNetModel,
    ModelInputContract,
    ModelOutputContract,
    export_model_artifact,
    load_model_artifact,
    verify_model_artifact,
)

GIT_SHA = "9" * 40


def _trained_model() -> EEGNetModel:
    rng = np.random.default_rng(19)
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
        random_state=19,
    )
    model.train(X, y)
    return model


def _contract() -> ModelInputContract:
    return ModelInputContract(
        axes=("batch", "channel", "time"),
        shape=(None, 2, 64),
        dtype="float32",
        channel_names=("C3", "C4"),
        sample_rate_hz=256.0,
        signal_unit="uV",
    )


def _export(path: Path):
    return export_model_artifact(
        _trained_model(),
        path,
        artifact_id="security-fixture",
        input_contract=_contract(),
        git_sha=GIT_SHA,
    )


def _rewrite(path: Path, manifest) -> None:
    (path / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_envelope_verify_remains_separate_from_runtime_authority(tmp_path: Path):
    manifest = _export(tmp_path / "artifact")
    forged = replace(manifest, backend="tensorflow")
    _rewrite(tmp_path / "artifact", forged)

    # Offline integrity inspection remains possible without claiming executable compatibility.
    assert verify_model_artifact(tmp_path / "artifact").backend == "tensorflow"
    with pytest.raises(ValueError, match="backend='pytorch'"):
        load_model_artifact(tmp_path / "artifact")


def test_backend_version_cannot_disagree_with_active_framework_runtime(tmp_path: Path):
    manifest = _export(tmp_path / "artifact")
    forged = replace(manifest, backend_version="0.0.0-forged")
    _rewrite(tmp_path / "artifact", forged)
    verify_model_artifact(tmp_path / "artifact")
    with pytest.raises(RuntimeError, match="backend_version"):
        load_model_artifact(tmp_path / "artifact")


def test_custom_package_versions_cannot_omit_reconstruction_authority(tmp_path: Path):
    baseline = _export(tmp_path / "baseline")
    weakened = dict(baseline.package_versions)
    weakened.pop("numpy")
    destination = tmp_path / "weakened"
    with pytest.raises(ValueError, match="cannot weaken"):
        export_model_artifact(
            _trained_model(),
            destination,
            artifact_id="weakened-runtime-authority",
            input_contract=_contract(),
            git_sha=GIT_SHA,
            package_versions=weakened,
        )
    assert not destination.exists()


def test_custom_package_versions_must_describe_actual_promotion_environment(tmp_path: Path):
    baseline = _export(tmp_path / "baseline")
    forged_versions = dict(baseline.package_versions)
    forged_versions["numpy"] = "0.0.0-forged"
    destination = tmp_path / "forged-environment"
    with pytest.raises(ValueError, match="does not match"):
        export_model_artifact(
            _trained_model(),
            destination,
            artifact_id="forged-environment",
            input_contract=_contract(),
            git_sha=GIT_SHA,
            package_versions=forged_versions,
        )
    assert not destination.exists()


def test_runtime_rejects_manifest_missing_required_package_identity(tmp_path: Path):
    manifest = _export(tmp_path / "artifact")
    weakened = dict(manifest.package_versions)
    weakened.pop("safetensors")
    forged = replace(manifest, package_versions=weakened)
    _rewrite(tmp_path / "artifact", forged)
    verify_model_artifact(tmp_path / "artifact")
    with pytest.raises(ValueError, match="runtime authority is incomplete"):
        load_model_artifact(tmp_path / "artifact")


def test_output_contract_cannot_claim_uncertainty_the_factory_does_not_emit(tmp_path: Path):
    manifest = _export(tmp_path / "artifact")
    lying_output = replace(
        manifest.output_contract,
        uncertainty_semantics="predictive_entropy",
    )
    forged = replace(manifest, output_contract=lying_output)
    _rewrite(tmp_path / "artifact", forged)
    verify_model_artifact(tmp_path / "artifact")
    with pytest.raises(ValueError, match="qualified uncertainty"):
        load_model_artifact(tmp_path / "artifact")

    with pytest.raises(ValueError, match="qualified uncertainty"):
        export_model_artifact(
            _trained_model(),
            tmp_path / "false-uncertainty",
            artifact_id="false-uncertainty",
            input_contract=_contract(),
            output_contract=ModelOutputContract(
                class_labels=("left", "right"),
                uncertainty_semantics="predictive_entropy",
            ),
            git_sha=GIT_SHA,
        )


def test_malicious_constructor_geometry_is_rejected_before_factory_allocation(tmp_path: Path):
    manifest = _export(tmp_path / "artifact")
    config = dict(manifest.model_config)
    config["temporal_filters"] = 10_000_000
    forged = replace(manifest, model_config=config)
    _rewrite(tmp_path / "artifact", forged)

    # The bundle is internally hash-consistent, but not authorized to allocate.
    verify_model_artifact(tmp_path / "artifact")
    with pytest.raises(ValueError, match="resource budget"):
        load_model_artifact(tmp_path / "artifact")


def test_manifest_size_is_bounded_before_json_parse(tmp_path: Path):
    _export(tmp_path / "artifact")
    manifest_path = tmp_path / "artifact" / "manifest.json"
    manifest_path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="manifest exceeds"):
        verify_model_artifact(tmp_path / "artifact")
