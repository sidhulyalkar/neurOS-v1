from __future__ import annotations

import hashlib
import json
from pathlib import Path

from neuros.evidence.kumar2024 import (
    KUMAR2024_ALL_SUBJECTS,
    Kumar2024PreprocessingSpec,
    Kumar2024StudyConfig,
    _identity_sha256,
    _preprocessing_authority,
    build_dataset_lineage,
    build_protocol,
    full_config,
    pilot_config,
    summarize_rows,
    verify_bundle,
)
from neuros.foundation_models.moabb_epochs import MOABBEpochDescriptor
from neuros.foundation_models.qualification_runner import (
    DEFAULT_CLASSIFICATION_SCORECARD,
)


def _descriptor(n_trials: int = 80) -> MOABBEpochDescriptor:
    return MOABBEpochDescriptor(
        channel_names=("C3", "Cz", "C4"),
        channel_types=("eeg", "eeg", "eeg"),
        sampling_rate_hz=512.0,
        n_times=2560,
        epoch_start_s=0.0,
        epoch_end_s=(2559 / 512.0),
        event_id=(("left_hand", 1), ("right_hand", 2)),
        n_trials=n_trials,
    )


def _versions():
    return {
        "moabb": "1.5.0",
        "mne": "1.6.0",
        "neuros": "2.1.0",
        "neuros-orion": "0.1.0",
    }


def test_profiles_make_pilot_explicit_and_full_study_predeclared():
    pilot = pilot_config()
    full = full_config()

    assert pilot.profile == "pilot"
    assert pilot.subjects == (1, 10)
    assert pilot.braindecode_epochs == 1
    assert full.profile == "full"
    assert full.subjects == KUMAR2024_ALL_SUBJECTS
    assert full.braindecode_epochs == 20
    assert pilot.budgets_per_class == (0, 1, 2, 5, 10)
    assert len(pilot.sha256) == 64
    assert pilot.sha256 != full.sha256


def test_lineage_is_partial_and_does_not_manufacture_raw_content_hash():
    config = pilot_config()
    preprocessing = Kumar2024PreprocessingSpec()
    authority = _preprocessing_authority(preprocessing, _descriptor(), _versions())
    lineage = build_dataset_lineage(
        config=config,
        preprocessing_authority=authority,
        versions=_versions(),
    )

    assert lineage.dataset_id == "moabb-kumar2024"
    assert lineage.content_sha256 is None
    assert lineage.lineage_completeness.value == "partial"
    identity = {item.level: item.identifiers for item in lineage.identity_sets}
    assert identity["participant"] == ("1", "10")
    assert identity["session"] == ("0", "1", "2", "3", "4", "5")
    assert lineage.sampling_assumptions["processed_channel_names"] == (
        "C3",
        "Cz",
        "C4",
    )
    assert lineage.metadata["not_a_reproduction_of_original_online_intervention"] is True
    assert len(lineage.lineage_sha256) == 64


def test_frozen_protocol_binds_production_scorecard_and_preprocessing():
    config = pilot_config()
    preprocessing = _preprocessing_authority(
        Kumar2024PreprocessingSpec(), _descriptor(), _versions()
    )
    lineage = build_dataset_lineage(
        config=config,
        preprocessing_authority=preprocessing,
        versions=_versions(),
    )
    protocol = build_protocol(
        config=config,
        dataset_lineage=lineage,
        preprocessing_authority_sha256=preprocessing["sha256"],
    )

    assert protocol.protocol_status == "frozen"
    assert protocol.independent_unit == "participant"
    assert protocol.grouping_hierarchy == ("participant", "session", "trial")
    assert protocol.calibration_budgets_per_class == (0, 1, 2, 5, 10)
    assert protocol.metric_scorecard_sha256 == DEFAULT_CLASSIFICATION_SCORECARD.sha256
    assert protocol.metadata["unlabeled_target_adaptation"] is False
    assert protocol.metadata["preprocessing_authority_sha256"] == preprocessing["sha256"]
    assert len(protocol.sha256) == 64


def _analysis_rows():
    rows = []
    methods = ("mne-csp-lda", "braindecode-eegnet")
    for method_index, method in enumerate(methods):
        for subject, cohort in ((1, "GR"), (10, "PAR")):
            for session in ("1", "2"):
                for budget in (0, 1):
                    base = 0.55 + 0.05 * budget + 0.02 * method_index
                    rows.append(
                        {
                            "method_id": method,
                            "subject": subject,
                            "original_protocol": cohort,
                            "held_out_session": session,
                            "calibration_per_class": budget,
                            "status": "success",
                            "balanced_accuracy": base + (0.01 if subject == 10 else 0.0),
                        }
                    )
    # Preserve one failed attempt as an explicit extra row; it must affect failure
    # accounting but not participant-level performance as though it succeeded.
    rows.append(
        {
            "method_id": "braindecode-eegnet",
            "subject": 1,
            "original_protocol": "GR",
            "held_out_session": "3",
            "calibration_per_class": 1,
            "status": "oom",
            "balanced_accuracy": None,
        }
    )
    return rows


def test_analysis_uses_participant_as_inferential_unit_and_preserves_failure_counts():
    config = Kumar2024StudyConfig(
        methods=("mne-csp-lda", "braindecode-eegnet"),
        budgets_per_class=(0, 1),
        analysis_bootstrap_replicates=50,
    )
    analysis = summarize_rows(_analysis_rows(), config=config)

    assert analysis["independent_inferential_unit"] == "participant"
    assert "aggregate within participant" in analysis["session_handling"]
    point = next(
        item
        for item in analysis["performance"]
        if item["method_id"] == "braindecode-eegnet"
        and item["calibration_per_class"] == 1
    )
    assert point["failure_status_counts"] == {"oom": 1}
    assert point["participant_level_balanced_accuracy"]["n_participants"] == 2
    paired = next(
        item
        for item in analysis["paired_method_differences"]
        if item["calibration_per_class"] == 0
    )
    assert paired["matched_subject_session_cases"] == 4
    assert paired["left_minus_right_balanced_accuracy"]["n_participants"] == 2
    assert len(analysis["calibration_efficiency"]) == 2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_verification_detects_tampering_without_rerunning_training(tmp_path: Path):
    managed = (
        "study_manifest.json",
        "case_authorities.json",
        "case_results.json",
        "results.csv",
        "analysis.json",
        "report.md",
    )
    for index, name in enumerate(managed):
        (tmp_path / name).write_text(f"fixture-{index}\n", encoding="utf-8")
    files = {name: _sha(tmp_path / name) for name in managed}
    bundle_sha = _identity_sha256(
        "neuros.nsq_kumar2024_bundle.v1", {"files": files}
    )
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps(
            {"schema_version": 1, "files": files, "bundle_sha256": bundle_sha},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    verified = verify_bundle(tmp_path)
    assert verified["verified"] is True
    assert verified["bundle_sha256"] == bundle_sha

    (tmp_path / "results.csv").write_text("tampered\n", encoding="utf-8")
    try:
        verify_bundle(tmp_path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered bundle unexpectedly verified")
