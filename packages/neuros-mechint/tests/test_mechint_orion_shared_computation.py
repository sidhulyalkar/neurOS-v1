import numpy as np
import pytest

pytest.importorskip("orion")
from orion.contracts import RepresentationBatch

from neuros_mechint.benchmarks import MechanismContext
from neuros_mechint.integrations.orion_study import (
    OrionRepresentationContext,
    run_shared_representation_study,
)


def _context(
    context_id: str,
    architecture: str,
    session: str,
    origin: int,
    *,
    alignment_label: str = "stimulus_onset",
):
    timestamps = np.asarray([origin - 10, origin, origin + 10, origin + 20], dtype=np.int64)
    values = np.asarray(
        [[1.0, 0.1], [2.0, 0.2], [4.0, 0.4], [0.5, 0.05]],
        dtype=np.float64,
    )
    representation = RepresentationBatch(values=values, timestamps_ns=timestamps)

    def scorer(batch: RepresentationBatch) -> float:
        array = np.asarray(batch.values)
        return float(array[:, 0].sum() + array[:, 1].sum())

    return OrionRepresentationContext(
        context=MechanismContext(
            context_id=context_id,
            architecture=architecture,
            dataset_id="synthetic-neural",
            session_id=session,
        ),
        representation=representation,
        scorer=scorer,
        alignment_origin_ns=origin,
        alignment_label=alignment_label,
    )


def _contexts():
    return [
        _context("t-s1", "transformer", "s1", 1_000),
        _context("t-s2", "transformer", "s2", 9_000),
        _context("s-s1", "ssm", "s1", 50_000),
        _context("s-s2", "ssm", "s2", 90_000),
    ]


def test_orion_shared_study_aligns_absolute_sessions_and_generates_hypothesis():
    study = run_shared_representation_study(
        _contexts(),
        window_ns=10,
        stride_ns=10,
        top_k=2,
        include_feature_audits=True,
    )
    maps = [dict(audit.record.effect_map) for audit in study.audits]
    assert all(effect_map == maps[0] for effect_map in maps[1:])
    assert set(maps[0]) == {
        "representation_relative[-10:0]",
        "representation_relative[0:10]",
        "representation_relative[10:20]",
        "representation_relative[20:30]",
    }
    assert all(audit.alignment_label == "stimulus_onset" for audit in study.audits)
    cross = study.analysis.comparison.axis_stability["cross_architecture"]
    assert cross.median_spearman_r == pytest.approx(1.0)
    hypothesis_ids = {item.hypothesis_id for item in study.analysis.hypotheses}
    assert "shared-causal-temporal-structure" in hypothesis_ids
    assert all(audit.feature_result is not None for audit in study.audits)
    assert all(len(audit.feature_result.effects) == 2 for audit in study.audits)
    assert all(len(audit.feature_result.controls) == 2 for audit in study.audits)
    payload = study.to_dict()
    assert payload["parameters"]["window_ns"] == 10
    assert len(payload["study_fingerprint"]) == 64
    assert payload["study_hash"] == payload["study_fingerprint"]
    assert len(payload["run_hash"]) == 64
    assert set(payload["context_manifest_hashes"]) == {"t-s1", "t-s2", "s-s1", "s-s2"}


def test_scientific_fingerprint_is_reproducible_across_timestamped_runs():
    kwargs = {
        "window_ns": 10,
        "stride_ns": 10,
        "top_k": 2,
        "include_feature_audits": True,
    }
    first = run_shared_representation_study(_contexts(), **kwargs)
    second = run_shared_representation_study(_contexts(), **kwargs)
    assert first.study_fingerprint == second.study_fingerprint
    assert first.study_hash == second.study_hash
    assert first.run_hash != second.run_hash


def test_explicit_alignment_labels_must_match():
    contexts = [
        _context("a", "transformer", "s1", 1_000, alignment_label="stimulus_onset"),
        _context("b", "ssm", "s1", 2_000, alignment_label="movement_onset"),
    ]
    with pytest.raises(ValueError, match="same semantic alignment event"):
        run_shared_representation_study(contexts, window_ns=10)
