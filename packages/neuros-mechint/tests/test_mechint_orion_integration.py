import numpy as np
import pytest

orion = pytest.importorskip("orion")
from orion.contracts import NeuroTokenBatch, RepresentationBatch

from neuros_mechint import EvidenceTier
from neuros_mechint.integrations.orion import (
    RepresentationFeatureAblation,
    RepresentationTimeWindowAblation,
    RepresentationTimeWindowShuffle,
    SideFeatureAblation,
    TokenTimeWindowMask,
    TokenTimeWindowShuffle,
    TokenTypeAblation,
    representation_window_audit,
    temporal_window_audit,
)


def _batch():
    return NeuroTokenBatch(
        token_ids=np.array([1, 2, 3, 2, 4], dtype=np.int64),
        timestamps_ns=np.array([0, 10, 20, 30, 40], dtype=np.int64),
        side_features={"rate": np.array([1, 2, 3, 4, 5], dtype=np.float32)},
    )


def _representation():
    return RepresentationBatch(
        values=np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            dtype=np.float32,
        ),
        timestamps_ns=np.array([0, 10, 20, 30], dtype=np.int64),
    )


def test_orion_interventions_preserve_reference_and_target_only_requested_content():
    batch = _batch()
    masked = TokenTimeWindowMask(10, 30, replacement_token_id=0).apply(batch)
    assert batch.token_ids.tolist() == [1, 2, 3, 2, 4]
    assert masked.token_ids.tolist() == [1, 0, 0, 2, 4]

    types = TokenTypeAblation([2], replacement_token_id=9).apply(batch)
    assert types.token_ids.tolist() == [1, 9, 3, 9, 4]

    side = SideFeatureAblation("rate").apply(batch)
    assert np.all(side.side_features["rate"] == 0)
    assert side.token_ids.tolist() == batch.token_ids.tolist()


def test_shuffle_control_is_deterministic():
    batch = _batch()
    control = TokenTimeWindowShuffle(0, 50, seed=7)
    assert np.array_equal(control.apply(batch).token_ids, control.apply(batch).token_ids)


def test_temporal_window_audit_records_orion_effects_and_evidence():
    batch = _batch()
    result = temporal_window_audit(
        batch,
        lambda value: float(np.asarray(value.token_ids).sum()),
        window_ns=20,
        replacement_token_id=0,
        evidence_tier=EvidenceTier.SCIENTIFIC_SYNTHETIC,
        seed=3,
    )
    assert len(result.effects) == 3
    assert len(result.controls) == 3
    assert result.manifest.evidence_tier is EvidenceTier.SCIENTIFIC_SYNTHETIC
    assert all(effect.name == "orion_token_time_window_mask" for effect in result.effects)


def test_representation_interventions_target_time_and_features_independently():
    batch = _representation()
    window = RepresentationTimeWindowAblation(10, 30).apply(batch)
    assert np.array_equal(batch.values[:, 0], np.array([1, 2, 3, 4]))
    assert np.all(window.values[1:3] == 0)
    assert np.array_equal(window.values[[0, 3]], batch.values[[0, 3]])

    feature = RepresentationFeatureAblation([1]).apply(batch)
    assert np.all(feature.values[:, 1] == 0)
    assert np.array_equal(feature.values[:, 0], batch.values[:, 0])

    shuffle = RepresentationTimeWindowShuffle(0, 40, seed=11)
    assert np.array_equal(shuffle.apply(batch).values, shuffle.apply(batch).values)


def test_representation_window_audit_records_effects_and_controls():
    batch = _representation()
    result = representation_window_audit(
        batch,
        lambda value: float(np.asarray(value.values).sum()),
        window_ns=20,
        evidence_tier=EvidenceTier.SCIENTIFIC_SYNTHETIC,
        seed=5,
    )
    assert len(result.effects) == 2
    assert len(result.controls) == 2
    assert all(
        effect.name == "orion_representation_time_window_ablation"
        for effect in result.effects
    )
