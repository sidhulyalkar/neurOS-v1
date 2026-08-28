from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from neuros.evidence.kumar2024 import (
    KUMAR2024_ALL_SUBJECTS,
    Kumar2024PreprocessingSpec,
    Kumar2024StudyConfig,
    _identity_sha256,
    _make_case_authority,
    _preprocessing_authority,
    build_dataset_lineage,
    build_protocol,
    full_config,
    pilot_config,
    summarize_rows,
    verify_bundle,
)
from neuros.foundation_models.moabb_epochs import MOABBEpochDescriptor
from neuros.foundation_models.moabb_longitudinal import get_moabb_longitudinal_spec
from neuros.foundation_models.qualification_runner import (
    DEFAULT_CLASSIFICATION_SCORECARD,
)
from neuros.foundation_models.real_world import GroupedEvaluationData


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
    assert pilot.braindecode_epochs == 1
    assert full.profile == "full"
    assert full.subjects == KUMAR2024_ALL_SUBJECTS
    assert full.braindecode_epochs == 20
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
    assert session_one.partition_fingerprint != session_five.partition_fingerprint


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


def test_frontier_auc_requires_same_cases_at_every_budget():
    rows = [
        row
        for row in _analysis_rows()
        if row["held_out_session"] in {"1", "2"}
    ]
    rows = [
        row
        for row in rows
        if not (
            row["method_id"] == "braindecode-eegnet"
            and row["subject"] == 1
            and row["held_out_session"] == "2"
            and row["calibration_per_class"] == 1
        )
    ]
    config = Kumar2024StudyConfig(
        methods=("mne-csp-lda", "braindecode-eegnet"),
        target_sessions=("1", "2"),
        budgets_per_class=(0, 1),
        analysis_bootstrap_replicates=25,
    )
    analysis = summarize_rows(rows, config=config)
    eegnet = next(
        item
        for item in analysis["calibration_efficiency"]
        if item["method_id"] == "braindecode-eegnet"
    )
    assert eegnet["complete_frontier_participants"] == [10]
    assert [1, "2"] not in eegnet["complete_frontier_subject_session_cases"]
    paired_auc = analysis["paired_calibration_efficiency"][0]
    assert paired_auc["matched_complete_frontier_participants"] == [10]


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
