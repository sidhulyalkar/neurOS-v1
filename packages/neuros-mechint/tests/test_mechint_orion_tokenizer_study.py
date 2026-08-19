import numpy as np
import pytest

pytest.importorskip("orion")
from orion.contracts import NeuroTokenBatch

from neuros_mechint.benchmarks import TokenizerMechanismContext
from neuros_mechint.integrations.orion_token_study import (
    OrionTokenizerStudyContext,
    run_orion_tokenizer_study,
)


def _context(context_id: str, tokenizer_id: str, origin: int, scale: int):
    batch = NeuroTokenBatch(
        token_ids=np.asarray([1, 2, 4, 1], dtype=np.int64) * scale,
        timestamps_ns=np.asarray(
            [origin - 10, origin, origin + 10, origin + 20],
            dtype=np.int64,
        ),
    )

    def scorer(value: NeuroTokenBatch) -> float:
        return float(np.asarray(value.token_ids, dtype=np.float64).sum())

    return OrionTokenizerStudyContext(
        context=TokenizerMechanismContext(
            context_id=context_id,
            tokenizer_id=tokenizer_id,
            downstream_model_id="matched-decoder",
            dataset_id="synthetic-neural",
            session_id="s1",
            subject_id="mouse-1",
        ),
        token_batch=batch,
        scorer=scorer,
        alignment_origin_ns=origin,
        alignment_label="movement_onset",
    )


def test_orion_tokenizer_study_aligns_time_and_detects_invariant_profile():
    result = run_orion_tokenizer_study(
        [
            _context("event", "event", 1_000, 1),
            _context("isi", "relative-isi", 90_000, 3),
        ],
        window_ns=10,
        stride_ns=10,
        top_k=2,
    )
    aggregate = result.comparison.isolated_tokenizer_stability
    assert aggregate is not None
    assert aggregate.median_spearman_r == pytest.approx(1.0)
    ids = {item.hypothesis_id for item in result.comparison.hypotheses}
    assert "tokenization-invariant-causal-profile" in ids
    assert len(result.study_fingerprint) == 64
    assert len(result.run_hash) == 64
    maps = [dict(audit.record.effect_map) for audit in result.audits]
    assert list(maps[0]) == list(maps[1])


def test_orion_tokenizer_study_rejects_incompatible_semantic_alignment():
    left = _context("event", "event", 1_000, 1)
    right = _context("isi", "relative-isi", 90_000, 1)
    right = OrionTokenizerStudyContext(
        context=right.context,
        token_batch=right.token_batch,
        scorer=right.scorer,
        alignment_origin_ns=right.alignment_origin_ns,
        alignment_label="reward",
    )
    with pytest.raises(ValueError, match="alignment_label"):
        run_orion_tokenizer_study([left, right], window_ns=10)
