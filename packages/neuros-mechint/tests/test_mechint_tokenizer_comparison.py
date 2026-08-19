import pytest

from neuros_mechint.benchmarks import (
    TokenizerEffectRecord,
    TokenizerMechanismContext,
    compare_tokenizer_mechanisms,
)


def _record(context_id: str, tokenizer_id: str, effects: dict[str, float]):
    return TokenizerEffectRecord(
        context=TokenizerMechanismContext(
            context_id=context_id,
            tokenizer_id=tokenizer_id,
            downstream_model_id="decoder-a",
            dataset_id="synthetic",
            session_id="s1",
            subject_id="mouse-1",
        ),
        baseline_metric=1.0,
        effect_map=effects,
        control_map={key: 0.0 for key in effects},
    )


def test_tokenizer_comparison_detects_invariant_causal_profile():
    report = compare_tokenizer_mechanisms(
        [
            _record("event", "event", {"w0": -1.0, "w1": -2.0, "w2": -4.0}),
            _record("isi", "relative-isi", {"w0": -0.8, "w1": -1.8, "w2": -3.5}),
        ],
        top_k=2,
    )
    aggregate = report.isolated_tokenizer_stability
    assert aggregate is not None
    assert aggregate.median_spearman_r == pytest.approx(1.0)
    ids = {item.hypothesis_id for item in report.hypotheses}
    assert "tokenization-invariant-causal-profile" in ids


def test_tokenizer_comparison_refuses_to_call_confound_change_isolated():
    left = _record("event", "event", {"w0": -1.0, "w1": -2.0, "w2": -4.0})
    right = TokenizerEffectRecord(
        context=TokenizerMechanismContext(
            context_id="isi-other-session",
            tokenizer_id="relative-isi",
            downstream_model_id="decoder-a",
            dataset_id="synthetic",
            session_id="s2",
            subject_id="mouse-1",
        ),
        baseline_metric=1.0,
        effect_map={"w0": -1.0, "w1": -2.0, "w2": -4.0},
    )
    report = compare_tokenizer_mechanisms([left, right])
    assert report.isolated_tokenizer_stability is None
    assert report.hypotheses == ()
