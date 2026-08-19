import pytest

from neuros_mechint.benchmarks import compare_effect_maps


def test_effect_map_stability_reports_rank_sign_and_top_k_agreement():
    report = compare_effect_maps(
        {"early": 1.0, "middle": 0.5, "late": -0.2},
        {"early": 0.9, "middle": 0.4, "late": -0.1},
        top_k=2,
    )
    assert report.shared_targets == 3
    assert report.pearson_r == pytest.approx(1.0, abs=0.03)
    assert report.spearman_r == 1.0
    assert report.sign_agreement == 1.0
    assert report.top_k_jaccard == 1.0
    assert report.mean_absolute_delta == pytest.approx(0.1)


def test_effect_map_stability_does_not_treat_missing_targets_as_zero():
    report = compare_effect_maps({"a": 1.0, "b": 2.0}, {"b": 2.5, "c": 4.0})
    assert report.shared_targets == 1
    assert report.pearson_r is None
    assert report.spearman_r is None
    assert report.top_k == 1


def test_effect_map_stability_requires_shared_targets():
    with pytest.raises(ValueError, match="no shared targets"):
        compare_effect_maps({"a": 1.0}, {"b": 1.0})
