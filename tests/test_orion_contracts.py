import numpy as np
import pytest

from orion import AdaptationProposal, NeuroTokenBatch, TokenizerManifest


def test_token_batch_requires_aligned_timestamps():
    with pytest.raises(ValueError):
        NeuroTokenBatch(
            token_ids=np.array([1, 2, 3]),
            timestamps_ns=np.array([10, 20]),
        )


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
