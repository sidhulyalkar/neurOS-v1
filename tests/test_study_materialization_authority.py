from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuros.foundation_models.materialization_authority import (
    EnvironmentAuthority,
    EnvironmentDistribution,
    ObservationIdentity,
    ObservationIdentityAuthority,
    ProcessedMaterializationShard,
    StudyMaterializationAuthority,
    hash_raw_materialization,
    observation_identities_from_grouped_data,
)
from neuros.foundation_models.real_world import GroupedEvaluationData


def _environment(*, numpy_version: str = "2.0.0") -> EnvironmentAuthority:
    # Deliberately supplied out of order. Canonical identity must not depend on
    # installation/enumeration ordering.
    return EnvironmentAuthority(
        python_implementation="CPython",
        python_version="3.11.9",
        platform_system="Linux",
        platform_machine="x86_64",
        distributions=(
            EnvironmentDistribution("scikit_learn", "1.6.0"),
            EnvironmentDistribution("NumPy", numpy_version),
        ),
        source_revision="abc123",
        accelerator_runtime=(
            ("device", "cpu"),
            ("torch", "2.8.0"),
        ),
        deterministic_flags=(("cudnn_deterministic", "true"),),
    )


def _grouped() -> GroupedEvaluationData:
    X = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4)
    y = np.asarray(["left", "right", "left", "right", "left", "right"])
    return GroupedEvaluationData(
        dataset_id="moabb-kumar2024",
        X=X,
        y=y,
        groups={
            "subject": np.asarray(["1", "1", "1", "1", "1", "1"]),
            "session": np.asarray(["0", "0", "0", "1", "1", "1"]),
            "run": np.asarray(["0", "0", "1", "0", "0", "0"]),
        },
        metadata=tuple({"note": f"row-{index}"} for index in range(6)),
    )


def _materialization(tmp_path: Path) -> StudyMaterializationAuthority:
    root = tmp_path / "raw"
    (root / "subject-1").mkdir(parents=True)
    (root / "subject-1" / "session-0.bin").write_bytes(b"abc")
    (root / "subject-1" / "session-1.bin").write_bytes(b"def")
    raw = hash_raw_materialization(
        dataset_id="moabb-kumar2024",
        root=root,
        relative_paths=(
            "subject-1/session-1.bin",
            "subject-1/session-0.bin",
        ),
        upstream_identity={
            "paper_doi": "10.1093/pnasnexus/pgae076",
            "data_doi": "10.5281/zenodo.10694880",
        },
    )
    shard = ProcessedMaterializationShard(
        shard_id="subject=1",
        processed_data_sha256="a" * 64,
        observation_identity=observation_identities_from_grouped_data(_grouped()),
        preprocessing_authority_sha256s=("b" * 64,),
    )
    return StudyMaterializationAuthority(
        environment=_environment(),
        raw_materialization=raw,
        processed_shards=(shard,),
    )


def test_environment_identity_is_order_independent_but_version_sensitive():
    first = _environment()
    reordered = EnvironmentAuthority(
        python_implementation="CPython",
        python_version="3.11.9",
        platform_system="Linux",
        platform_machine="x86_64",
        distributions=tuple(reversed(first.distributions)),
        source_revision="abc123",
        accelerator_runtime=tuple(reversed(first.accelerator_runtime)),
        deterministic_flags=first.deterministic_flags,
    )
    changed = _environment(numpy_version="2.0.1")

    assert first.sha256 == reordered.sha256
    assert first.sha256 != changed.sha256
    payload = first.to_dict()
    assert "hostname" not in payload
    assert "timestamp" not in payload
    assert [item["name"] for item in payload["distributions"]] == [
        "numpy",
        "scikit-learn",
    ]


def test_environment_rejects_conflicting_duplicate_distribution_names():
    with pytest.raises(ValueError, match="duplicate distribution"):
        EnvironmentAuthority(
            python_implementation="CPython",
            python_version="3.11.9",
            platform_system="Linux",
            platform_machine="x86_64",
            distributions=(
                EnvironmentDistribution("scikit-learn", "1.5"),
                EnvironmentDistribution("scikit_learn", "1.6"),
            ),
        )


def test_raw_materialization_is_cache_root_independent_and_byte_sensitive(tmp_path: Path):
    roots = (tmp_path / "cache-a", tmp_path / "cache-b")
    for root in roots:
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "recording.bin").write_bytes(b"same-bytes")

    first = hash_raw_materialization(
        dataset_id="fixture",
        root=roots[0],
        relative_paths=("nested/recording.bin",),
        upstream_identity={"version": "1"},
    )
    second = hash_raw_materialization(
        dataset_id="fixture",
        root=roots[1],
        relative_paths=("nested/recording.bin",),
        upstream_identity={"version": "1"},
    )
    assert first.sha256 == second.sha256
    assert str(roots[0]) not in str(first.to_dict())
    assert first.files[0].logical_path == "nested/recording.bin"

    (roots[1] / "nested" / "recording.bin").write_bytes(b"changed")
    changed = hash_raw_materialization(
        dataset_id="fixture",
        root=roots[1],
        relative_paths=("nested/recording.bin",),
        upstream_identity={"version": "1"},
    )
    assert first.sha256 != changed.sha256


def test_raw_materialization_rejects_escape_missing_and_duplicate_paths(tmp_path: Path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "file.bin").write_bytes(b"x")

    with pytest.raises(ValueError, match="canonical paths"):
        hash_raw_materialization(
            dataset_id="fixture",
            root=root,
            relative_paths=("../escape.bin",),
        )
    with pytest.raises(ValueError, match="does not exist"):
        hash_raw_materialization(
            dataset_id="fixture",
            root=root,
            relative_paths=("missing.bin",),
        )
    with pytest.raises(ValueError, match="duplicate raw logical path"):
        hash_raw_materialization(
            dataset_id="fixture",
            root=root,
            relative_paths=("file.bin", "file.bin"),
        )


def test_grouped_observation_identities_are_human_auditable_and_label_free():
    authority = observation_identities_from_grouped_data(_grouped())

    assert [item.local_epoch for item in authority.observations] == [0, 1, 0, 0, 1, 2]
    assert authority.observations[2].display_id == (
        "participant=1/session=0/run=1/epoch=0"
    )
    serialized = authority.to_dict()
    text = str(serialized)
    assert "left" not in text
    assert "right" not in text
    assert len(authority.sha256) == 64


def test_observation_row_reorder_changes_authority():
    data = _grouped()
    first = observation_identities_from_grouped_data(data)
    order = np.asarray([1, 0, 2, 3, 4, 5])
    reordered = GroupedEvaluationData(
        dataset_id=data.dataset_id,
        X=data.X[order],
        y=data.y[order],
        groups={key: values[order] for key, values in data.groups.items()},
        metadata=tuple(data.metadata[index] for index in order),
    )
    second = observation_identities_from_grouped_data(reordered)
    assert first.sha256 != second.sha256


def test_explicit_ambiguous_observation_identity_fails_closed():
    with pytest.raises(ValueError, match="ambiguous or duplicated"):
        ObservationIdentityAuthority(
            dataset_id="fixture",
            observations=(
                ObservationIdentity(0, "p1", "s1", "r1", 0, "a" * 64),
                ObservationIdentity(1, "p1", "s1", "r1", 0, "b" * 64),
            ),
        )


def test_role_authority_maps_execution_rows_to_human_identities_without_labels():
    observations = observation_identities_from_grouped_data(_grouped())
    source = observations.role(
        "source_history", np.asarray([0, 2, 4], dtype=np.int64)
    )
    reversed_source = observations.role("source_history", (4, 2, 0))

    assert source.row_indices == (0, 2, 4)
    assert source.display_ids == tuple(
        observations.observations[index].display_id for index in (0, 2, 4)
    )
    assert source.observation_identity_authority_sha256 == observations.sha256
    assert source.sha256 != reversed_source.sha256
    assert "left" not in str(source.to_dict())
    assert "right" not in str(source.to_dict())


def test_study_materialization_composes_all_major_authorities(tmp_path: Path):
    study = _materialization(tmp_path)
    payload = study.to_dict()
    shard = study.processed_shards[0]

    assert study.dataset_id == "moabb-kumar2024"
    assert payload["environment_sha256"] == study.environment.sha256
    assert payload["raw_materialization_sha256"] == study.raw_materialization.sha256
    assert payload["processed_shard_sha256s"] == [shard.sha256]
    assert shard.to_dict()["observation_identity_sha256"] == shard.observation_identity.sha256
    assert len(study.sha256) == 64

    changed_environment = StudyMaterializationAuthority(
        environment=_environment(numpy_version="2.0.1"),
        raw_materialization=study.raw_materialization,
        processed_shards=study.processed_shards,
    )
    assert study.sha256 != changed_environment.sha256


def test_study_materialization_shard_order_is_canonical(tmp_path: Path):
    study = _materialization(tmp_path)
    first = study.processed_shards[0]
    second = ProcessedMaterializationShard(
        shard_id="subject=2",
        processed_data_sha256="c" * 64,
        observation_identity=ObservationIdentityAuthority(
            dataset_id=first.dataset_id,
            observations=(
                ObservationIdentity(0, "2", "0", "0", 0, "d" * 64),
            ),
        ),
        preprocessing_authority_sha256s=first.preprocessing_authority_sha256s,
    )
    forward = StudyMaterializationAuthority(
        environment=study.environment,
        raw_materialization=study.raw_materialization,
        processed_shards=(first, second),
    )
    reversed_order = StudyMaterializationAuthority(
        environment=study.environment,
        raw_materialization=study.raw_materialization,
        processed_shards=(second, first),
    )
    assert forward.sha256 == reversed_order.sha256
    assert [item.shard_id for item in forward.processed_shards] == ["subject=1", "subject=2"]


def test_study_materialization_refuses_cross_dataset_composition(tmp_path: Path):
    study = _materialization(tmp_path)
    shard = study.processed_shards[0]
    wrong_observations = ObservationIdentityAuthority(
        dataset_id="different-dataset",
        observations=shard.observation_identity.observations,
    )
    wrong_shard = ProcessedMaterializationShard(
        shard_id=shard.shard_id,
        processed_data_sha256=shard.processed_data_sha256,
        observation_identity=wrong_observations,
        preprocessing_authority_sha256s=shard.preprocessing_authority_sha256s,
    )
    with pytest.raises(ValueError, match="one dataset_id"):
        StudyMaterializationAuthority(
            environment=study.environment,
            raw_materialization=study.raw_materialization,
            processed_shards=(wrong_shard,),
        )
