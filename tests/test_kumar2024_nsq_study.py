from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from neuros.evidence.kumar2024 import (
    KUMAR2024_ALL_SUBJECTS,
    KUMAR2024_DEFAULT_METHODS,
    Kumar2024StudyConfig,
    _make_case_authority,
    _normalize_methods,
    build_kumar2024_bundle,
    full_config,
    pilot_config,
    verify_kumar2024_bundle,
)
from neuros.foundation_models.longitudinal import get_moabb_longitudinal_spec
from neuros.foundation_models.real_world import GroupedEvaluationData


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _versions() -> dict[str, str]:
    return {
        "moabb": "1.5.0",
        "mne": "1.6.0",
        "neuros": "2.1.0",
        "neuros-orion": "0.1.0",
    }


def _six_session_fixture() -> GroupedEvaluationData:
    trials_per_session = 40
    n_samples = 6 * trials_per_session
    return GroupedEvaluationData(
        dataset_id="moabb-kumar2024",
        X=np.arange(n_samples * 2, dtype=np.float64).reshape(n_samples, 2),
        y=np.asarray(["left", "right"] * (n_samples // 2), dtype=str),
        groups={
            "subject": np.asarray(["1"] * n_samples, dtype=str),
            "session": np.repeat(
                np.asarray(["0", "1", "2", "3", "4", "5"], dtype=str),
                trials_per_session,
            ),
            "trial": np.asarray([f"t{index:03d}" for index in range(n_samples)], dtype=str),
        },
    )


def test_profiles_make_pilot_explicit_and_full_study_predeclared():
    pilot = pilot_config()
    full = full_config()

    assert pilot.profile == "pilot"
    assert pilot.subjects == (1, 10)
    assert pilot.methods == ("mne-csp-lda",)
    assert full.profile == "full"
    assert full.subjects == KUMAR2024_ALL_SUBJECTS

    # Pilot versus full controls study scope, not the scientific quality of a
    # neural method. Both profiles therefore carry the exact same preregistered
    # EEGNet training authority even though the canonical pilot executes CSP only.
    for config in (pilot, full):
        assert config.braindecode_epochs == 1000
        assert config.braindecode_batch_size == 64
        assert config.braindecode_optimizer == "Adam"
        assert config.braindecode_learning_rate == pytest.approx(0.000625)
        assert config.braindecode_weight_decay == pytest.approx(0.0)
        assert config.braindecode_validation_fraction == pytest.approx(0.2)
        assert config.braindecode_validation_seed == 17011
        assert config.braindecode_early_stopping_patience == 300
        assert config.braindecode_model_seed == 31415

    assert pilot.budgets_per_class == (0, 1, 2, 5, 10)
    assert len(pilot.sha256) == 64
    assert pilot.sha256 != full.sha256


def test_preregistered_split_seed_is_literal_and_shared_across_cases():
    data = _six_session_fixture()
    spec = get_moabb_longitudinal_spec("kumar2024")
    config = Kumar2024StudyConfig(
        subjects=(1,),
        methods=("mne-csp-lda",),
        split_seed=2026,
    )

    session_one = _make_case_authority(
        data=data,
        dataset_spec=spec,
        subject=1,
        target_session="1",
        config=config,
    )
    session_five = _make_case_authority(
        data=data,
        dataset_spec=spec,
        subject=1,
        target_session="5",
        config=config,
    )

    assert session_one.seed == 2026
    assert session_five.seed == 2026
    assert session_one.case_metadata["split_seed"] == 2026
    assert session_five.case_metadata["split_seed"] == 2026
    assert session_one.case_id.endswith("/split-2026")
    assert session_five.case_id.endswith("/split-2026")


def test_method_normalization_rejects_unknown_or_duplicate_methods():
    assert _normalize_methods(("mne-csp-lda",)) == ("mne-csp-lda",)
    assert "braindecode-eegconformer" not in KUMAR2024_DEFAULT_METHODS
    with pytest.raises(ValueError, match="unsupported Kumar2024 method"):
        _normalize_methods(("not-a-method",))
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_methods(("mne-csp-lda", "mne-csp-lda"))


def test_bundle_verification_detects_mutation(tmp_path: Path):
    root = build_kumar2024_bundle(
        output_dir=tmp_path / "bundle",
        config=pilot_config(),
        git_revision="a" * 40,
        dataset_lineage={"dataset_id": "moabb-kumar2024", "sha256": _sha("dataset")},
        preprocessing_authority={"sha256": _sha("preprocessing")},
        case_authorities=[{"case_id": "case-1", "authority_sha256": _sha("authority")}],
        results=[{"status": "success", "case_id": "case-1", "method_id": "mne-csp-lda"}],
        analysis={"participant_count": 2, "analysis_sha256": _sha("analysis")},
        runtime_versions=_versions(),
    )

    verified = verify_kumar2024_bundle(root)
    assert verified["verified"] is True
    assert len(verified["bundle_sha256"]) == 64

    results = root / "results.csv"
    results.write_text(results.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_kumar2024_bundle(root)


def test_bundle_root_binds_managed_files_and_manifest(tmp_path: Path):
    root = build_kumar2024_bundle(
        output_dir=tmp_path / "bundle",
        config=pilot_config(),
        git_revision="b" * 40,
        dataset_lineage={"dataset_id": "moabb-kumar2024", "sha256": _sha("dataset")},
        preprocessing_authority={"sha256": _sha("preprocessing")},
        case_authorities=[{"case_id": "case-1", "authority_sha256": _sha("authority")}],
        results=[{"status": "success", "case_id": "case-1", "method_id": "mne-csp-lda"}],
        analysis={"participant_count": 2, "analysis_sha256": _sha("analysis")},
        runtime_versions=_versions(),
    )

    hashes = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert hashes["schema_version"] == 1
    assert len(hashes["bundle_sha256"]) == 64
    assert set(hashes["files"]) == {
        "analysis.json",
        "case_authorities.json",
        "dataset_lineage.json",
        "preprocessing_authority.json",
        "results.csv",
        "study_manifest.json",
    }

    manifest = json.loads((root / "study_manifest.json").read_text(encoding="utf-8"))
    assert manifest["study_config_sha256"] == pilot_config().sha256
    assert manifest["git_revision"] == "b" * 40
