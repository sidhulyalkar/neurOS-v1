import numpy as np
import pytest

from orion import AdaptationProposal, NeuroTokenBatch, RepresentationBatch, TokenizerManifest


def test_token_batch_requires_aligned_timestamps():
    with pytest.raises(ValueError):
        NeuroTokenBatch(
            token_ids=np.array([1, 2, 3]),
            timestamps_ns=np.array([10, 20]),
        )


def test_token_batch_rejects_lossy_identity_and_time_coercion():
    with pytest.raises(ValueError, match="integer token identities"):
        NeuroTokenBatch(
            token_ids=np.array([1.0, 2.0]),
            timestamps_ns=np.array([10, 20], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="integer nanosecond timestamps"):
        NeuroTokenBatch(
            token_ids=np.array([1, 2], dtype=np.int64),
            timestamps_ns=np.array([10.0, 20.0]),
        )


def test_token_batch_preserves_simultaneity_but_rejects_time_reversal():
    batch = NeuroTokenBatch(
        token_ids=np.array([1, 2, 3], dtype=np.int64),
        timestamps_ns=np.array([10, 10, 20], dtype=np.int64),
    )
    assert batch.timestamps_ns.tolist() == [10, 10, 20]

    with pytest.raises(ValueError, match="nondecreasing"):
        NeuroTokenBatch(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            timestamps_ns=np.array([10, 9, 20], dtype=np.int64),
        )


def test_token_batch_requires_side_features_and_mask_to_align():
    with pytest.raises(ValueError, match="leading axis"):
        NeuroTokenBatch(
            token_ids=np.array([1, 2], dtype=np.int64),
            timestamps_ns=np.array([10, 20], dtype=np.int64),
            side_features={"count": np.array([1])},
        )
    with pytest.raises(ValueError, match="mask must contain boolean"):
        NeuroTokenBatch(
            token_ids=np.array([1, 2], dtype=np.int64),
            timestamps_ns=np.array([10, 20], dtype=np.int64),
            mask=np.array([1, 0], dtype=np.int64),
        )


def test_token_batch_copies_and_freezes_numpy_buffers():
    token_ids = np.array([1, 2], dtype=np.int64)
    timestamps = np.array([10, 20], dtype=np.int64)
    counts = np.array([3, 4], dtype=np.int64)
    batch = NeuroTokenBatch(
        token_ids=token_ids,
        timestamps_ns=timestamps,
        side_features={"count": counts},
    )

    token_ids[0] = 99
    timestamps[0] = 99
    counts[0] = 99
    assert batch.token_ids.tolist() == [1, 2]
    assert batch.timestamps_ns.tolist() == [10, 20]
    assert batch.side_features["count"].tolist() == [3, 4]
    assert batch.token_ids.flags.writeable is False
    assert batch.timestamps_ns.flags.writeable is False
    assert batch.side_features["count"].flags.writeable is False
    with pytest.raises(ValueError):
        batch.token_ids[0] = 7


def test_representation_batch_requires_finite_floating_values_and_valid_mask():
    with pytest.raises(ValueError, match="real floating dtype"):
        RepresentationBatch(
            values=np.array([[1, 2], [3, 4]], dtype=np.int64),
            timestamps_ns=np.array([10, 20], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="must be finite"):
        RepresentationBatch(
            values=np.array([[1.0, np.nan], [3.0, 4.0]]),
            timestamps_ns=np.array([10, 20], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="mask length"):
        RepresentationBatch(
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
            timestamps_ns=np.array([10, 20], dtype=np.int64),
            mask=np.array([True], dtype=bool),
        )


def test_representation_batch_copies_and_freezes_values_time_and_mask():
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    timestamps = np.array([10, 20], dtype=np.int64)
    mask = np.array([True, False], dtype=bool)
    batch = RepresentationBatch(values=values, timestamps_ns=timestamps, mask=mask)

    values[0, 0] = 99.0
    timestamps[0] = 99
    mask[0] = False
    assert batch.values[0, 0] == pytest.approx(1.0)
    assert batch.timestamps_ns.tolist() == [10, 20]
    assert batch.mask is not None and batch.mask.tolist() == [True, False]
    assert batch.values.flags.writeable is False
    assert batch.timestamps_ns.flags.writeable is False
    assert batch.mask.flags.writeable is False


def test_tokenizer_manifest_is_immutable_at_boundary():
    params = {"bin_ms": 5}
    manifest = TokenizerManifest(
        tokenizer_id="event",
        version="0.1",
        parameters=params,
    )
    params["bin_ms"] = 10
    assert manifest.parameters["bin_ms"] == 5


def test_adaptation_proposal_records_evidence():
    proposal = AdaptationProposal(
        reason="decoder drift",
        changes={"learning_rate": 0.001},
        evidence={"calibration_error": 0.18},
    )
    assert proposal.evidence["calibration_error"] == pytest.approx(0.18)
