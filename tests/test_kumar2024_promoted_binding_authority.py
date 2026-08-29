from __future__ import annotations

import hashlib
import json

import pytest

from neuros.evidence import kumar2024 as base
from neuros.evidence.kumar2024_comparison import promoted_external_floor_plan
from neuros.evidence.kumar2024_materialization import materialization_manifest
from neuros.evidence.kumar2024_promoted_binding import (
    _environment_lock_payload,
    _seal_binding_bundle,
    promoted_materialization_config,
    promoted_method_spec_records,
    verify_promoted_binding_bundle,
)
from neuros.evidence.kumar2024_promoted_execution import (
    bind_promoted_execution_template,
    build_promoted_execution_template,
)
from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
from neuros.foundation_models.materialization_authority import (
    EnvironmentAuthority,
    EnvironmentDistribution,
    ObservationIdentity,
    ObservationIdentityAuthority,
    ProcessedMaterializationShard,
    RawMaterializationAuthority,
    RawMaterializationFile,
    StudyMaterializationAuthority,
)
from neuros.foundation_models.qualification import ExternalDecoderMethodSpec


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fake_materialization(preprocessing_sha: str) -> StudyMaterializationAuthority:
    environment = EnvironmentAuthority(
        python_implementation="CPython",
        python_version="3.11.16",
        platform_system="Linux",
        platform_machine="x86_64",
        distributions=(
            EnvironmentDistribution(name="fixture-external", version="1.2.3"),
            EnvironmentDistribution(name="neuros", version="2.1.0"),
        ),
        source_revision="a" * 40,
        accelerator_runtime=(("requested_device", "cpu"),),
        deterministic_flags=(("fixture", "true"),),
    )
    raw = RawMaterializationAuthority(
        dataset_id=base.KUMAR2024_DATASET_ID,
        files=(
            RawMaterializationFile(
                logical_path="fixture/consumed.gdf",
                size_bytes=123,
                sha256=_sha("raw-gdf"),
            ),
        ),
        upstream_identity=(("fixture", "true"),),
    )
    shards = []
    for subject in range(1, 19):
        observations = tuple(
            ObservationIdentity(
                row_index=index,
                participant=str(subject),
                session=str(index),
                run="0",
                local_epoch=0,
                processed_observation_sha256=_sha(f"observation-{subject}-{index}"),
            )
            for index in range(6)
        )
        identity = ObservationIdentityAuthority(
            dataset_id=base.KUMAR2024_DATASET_ID,
            observations=observations,
        )
        shards.append(
            ProcessedMaterializationShard(
                shard_id=f"subject={subject}",
                processed_data_sha256=_sha(f"processed-{subject}"),
                observation_identity=identity,
                preprocessing_authority_sha256s=(preprocessing_sha,),
            )
        )
    return StudyMaterializationAuthority(
        environment=environment,
        raw_materialization=raw,
        processed_shards=tuple(shards),
    )


def _fake_case_authorities(materialization: StudyMaterializationAuthority):
    processed_by_subject = {
        int(shard.shard_id.split("=", 1)[1]): shard.processed_data_sha256
        for shard in materialization.processed_shards
    }
    plan = promoted_external_floor_plan()
    authorities = []
    case_map = {}
    for subject in plan.subjects:
        for session in plan.target_sessions:
            for split_seed in plan.split_seeds:
                authority = LongitudinalCaseAuthority(
                    dataset_id=base.KUMAR2024_DATASET_ID,
                    case_id=(
                        f"{base.KUMAR2024_DATASET_ID}/subject-{subject}/"
                        f"session-{session}/split-{split_seed}"
                    ),
                    split_unit="session",
                    held_out_values=(session,),
                    history_policy="prior",
                    observed_group_order=base.KUMAR2024_EXPECTED_SESSIONS,
                    source_group_values=tuple(
                        value
                        for value in base.KUMAR2024_EXPECTED_SESSIONS
                        if int(value) < int(session)
                    ),
                    source_train_indices=(0,),
                    evaluation_indices=(5,),
                    calibration_order_by_class={
                        "left_hand": (1, 2),
                        "right_hand": (3, 4),
                    },
                    evaluation_fraction=0.5,
                    seed=split_seed,
                    partition_fingerprint=f"partition-{subject}-{session}-{split_seed}",
                    calibration_split_fingerprint=f"calibration-{subject}-{session}-{split_seed}",
                    processed_data_sha256=processed_by_subject[subject],
                    n_samples=6,
                    input_shape=(6, 1, 1),
                    case_metadata={
                        "subject": subject,
                        "held_out_session": session,
                        "split_seed": split_seed,
                        "original_protocol": "GR" if subject <= 9 else "PAR",
                    },
                )
                key = (subject, session, split_seed)
                authorities.append(authority)
                case_map[key] = authority.authority_sha256
    return tuple(authorities), case_map


def _fake_method_records():
    plan = promoted_external_floor_plan()
    records = []
    method_map = {}
    for policy in plan.method_seed_policies:
        seeds = policy.model_seeds if policy.stochastic else (None,)
        for seed in seeds:
            key = (
                f"{policy.method_id}/deterministic"
                if seed is None
                else f"{policy.method_id}/model-seed-{seed}"
            )
            spec = ExternalDecoderMethodSpec(
                method_id=policy.method_id,
                implementation=f"fixture.{policy.method_id}",
                implementation_version="fixture=1",
                input_axes=("sample", "channel", "time"),
                probability_semantics=(
                    "uncalibrated_softmax"
                    if policy.method_id == "braindecode-eegnet"
                    else "uncalibrated_probability"
                ),
                source_reference="synthetic binding verifier fixture",
                metadata={
                    "model_seed": seed,
                    "final_assessment_used_for_state_selection": False,
                },
            )
            records.append(
                {
                    "realization_key": key,
                    "method_spec": {
                        **spec.to_dict(),
                        "method_spec_sha256": spec.sha256,
                    },
                }
            )
            method_map[key] = spec.sha256
    records.sort(key=lambda item: item["realization_key"])
    return records, method_map


def _write_synthetic_bundle(root):
    plan = promoted_external_floor_plan()
    preprocessing_sha = _sha("preprocessing")
    lineage_sha = _sha("lineage")
    protocol_sha = _sha("protocol")
    materialization = _fake_materialization(preprocessing_sha)
    authorities, case_map = _fake_case_authorities(materialization)
    records, method_map = _fake_method_records()
    template = build_promoted_execution_template(plan)
    execution = bind_promoted_execution_template(
        template,
        study_materialization_sha256=materialization.sha256,
        environment_authority_sha256=materialization.environment.sha256,
        raw_materialization_sha256=materialization.raw_materialization.sha256,
        dataset_lineage_sha256=lineage_sha,
        protocol_sha256=protocol_sha,
        preprocessing_authority_sha256=preprocessing_sha,
        source_revision="a" * 40,
        case_authority_sha256_by_case=case_map,
        method_spec_sha256_by_realization=method_map,
    )
    manifest = {
        "schema_version": 1,
        "study": "nsq-kumar2024-promoted-external-floor-v1",
        "artifact_kind": "no_model_promoted_binding_authority",
        "comparison_plan": plan.to_dict(),
        "comparison_plan_sha256": plan.sha256,
        "preprocessing_authority": {"sha256": preprocessing_sha},
        "dataset_lineage_sha256": lineage_sha,
        "protocol_sha256": protocol_sha,
        "study_materialization_sha256": materialization.sha256,
        "environment_authority_sha256": materialization.environment.sha256,
        "raw_materialization_sha256": materialization.raw_materialization.sha256,
        "source_revision": "a" * 40,
        "execution_plan_sha256": execution.sha256,
        "counts": {
            "participants": 18,
            "case_authorities": 270,
            "method_realizations": 5,
            "execution_shards": 1350,
            "planned_fit_attempts": 6750,
        },
        "model_execution_performed": False,
        "final_assessment_predictions_generated": False,
        "final_assessment_metrics_generated": False,
    }
    base._json_dump(root / "binding_manifest.json", manifest)
    base._json_dump(
        root / "materialization.json",
        materialization_manifest(materialization, raw_selection={"fixture": True}),
    )
    base._json_dump(
        root / "environment_lock.json",
        _environment_lock_payload(materialization.environment),
    )
    base._json_dump(
        root / "case_authorities.json",
        {
            "schema_version": 1,
            "comparison_plan_sha256": plan.sha256,
            "authorities": [item.to_dict() for item in authorities],
        },
    )
    base._json_dump(
        root / "method_specs.json",
        {
            "schema_version": 1,
            "comparison_plan_sha256": plan.sha256,
            "method_specs": records,
        },
    )
    base._json_dump(
        root / "execution_plan.json",
        {**execution.to_dict(), "execution_plan_sha256": execution.sha256},
    )
    _seal_binding_bundle(root)
    return execution


def test_promoted_materialization_config_tracks_full_preregistered_plan():
    plan = promoted_external_floor_plan()
    config = promoted_materialization_config(plan)
    assert config.subjects == plan.subjects
    assert config.target_sessions == plan.target_sessions
    assert config.budgets_per_class == plan.budgets_per_class
    assert config.methods == plan.methods
    assert config.split_seed == plan.split_seeds[0]
    assert config.analysis_seed == plan.analysis_seed
    assert config.analysis_bootstrap_replicates == plan.bootstrap_replicates
    assert config.profile == "full"


def test_method_spec_binding_reads_specs_without_creating_decoders(monkeypatch):
    import neuros.foundation_models.qualification_baselines as baselines

    monkeypatch.setattr(baselines, "_package_version", lambda _name: "fixture")

    def forbidden_create(*_args, **_kwargs):
        raise AssertionError("promoted no-model binding must never create a decoder")

    monkeypatch.setattr(baselines.MNECSPLDAFactory, "create", forbidden_create)
    monkeypatch.setattr(baselines.RiemannianTangentLogRegFactory, "create", forbidden_create)
    monkeypatch.setattr(baselines.UpstreamBraindecodeFactory, "create", forbidden_create)

    plan = promoted_external_floor_plan()
    records = promoted_method_spec_records(
        plan=plan,
        config=promoted_materialization_config(plan),
        sample_rate_hz=512.0,
    )
    assert [item["realization_key"] for item in records] == sorted(
        build_promoted_execution_template(plan).method_realization_keys
    )
    eegnet = [
        item for item in records if item["realization_key"].startswith("braindecode-eegnet/")
    ]
    assert {
        item["method_spec"]["metadata"]["model_seed"]
        for item in eegnet
    } == {31415, 384165836, 3991196546}
    assert all(
        item["method_spec"]["metadata"]["final_assessment_used_for_state_selection"] is False
        for item in eegnet
    )


def test_synthetic_full_binding_bundle_verifies_without_efficacy_rows(tmp_path):
    execution = _write_synthetic_bundle(tmp_path)
    verified = verify_promoted_binding_bundle(tmp_path)
    assert verified["verified"] is True
    assert verified["case_authorities"] == 270
    assert verified["method_realizations"] == 5
    assert verified["expected_shards"] == 1350
    assert verified["expected_fit_attempts"] == 6750
    assert verified["execution_plan_sha256"] == execution.sha256
    assert not (tmp_path / "results.csv").exists()
    assert not (tmp_path / "case_results.json").exists()
    assert not (tmp_path / "analysis.json").exists()


def test_binding_bundle_detects_file_tampering_before_semantic_use(tmp_path):
    _write_synthetic_bundle(tmp_path)
    path = tmp_path / "case_authorities.json"
    payload = json.loads(path.read_text())
    payload["authorities"][0]["case_metadata"]["split_seed"] = 999
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="bundle hash mismatch"):
        verify_promoted_binding_bundle(tmp_path)


def test_binding_bundle_rejects_resealed_cross_file_authority_drift(tmp_path):
    _write_synthetic_bundle(tmp_path)
    manifest_path = tmp_path / "binding_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_lineage_sha256"] = _sha("wrong-lineage")
    base._json_dump(manifest_path, manifest)
    _seal_binding_bundle(tmp_path)
    with pytest.raises(ValueError, match="dataset lineage"):
        verify_promoted_binding_bundle(tmp_path)


def test_binding_bundle_rejects_resealed_environment_lock_drift(tmp_path):
    _write_synthetic_bundle(tmp_path)
    lock_path = tmp_path / "environment_lock.json"
    lock = json.loads(lock_path.read_text())
    lock["external_distributions"].pop()
    base._json_dump(lock_path, lock)
    _seal_binding_bundle(tmp_path)
    with pytest.raises(ValueError, match="environment lock distribution set"):
        verify_promoted_binding_bundle(tmp_path)
