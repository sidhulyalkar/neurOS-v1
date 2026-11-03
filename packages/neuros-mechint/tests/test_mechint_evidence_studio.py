"""Tests for truthful, portable, hash-checked MechInt evidence presentation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from neuros_mechint.core.manifest import stable_hash
from neuros_mechint.evidence_studio import (
    _unique_json_keys,
    render_paired_selectivity_html,
    write_paired_selectivity_html,
)

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "26_evidence_studio_showcase.py"
spec = importlib.util.spec_from_file_location("evidence_studio_demo", SCRIPT)
assert spec is not None and spec.loader is not None
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_three_synthetic_stories_generate_portable_evidence(tmp_path):
    outcomes = demo.main(tmp_path)
    assert outcomes["01-planted-path"]["status"] == "SUPPORTED_DIAGNOSTIC_ONLY"
    assert outcomes["02-generic-degradation"]["status"] == "REJECTED_NO_SELECTIVE_EFFECT"
    assert outcomes["03-incomplete-experiment"]["status"] == (
        "NONESTIMABLE_MISSING_OBSERVATIONS"
    )
    for name, summary in outcomes.items():
        report = json.loads((tmp_path / summary["source"]).read_text(encoding="utf-8"))
        page = (tmp_path / summary["report"]).read_text(encoding="utf-8")
        assert render_paired_selectivity_html(
            report,
            title={
                "01-planted-path": "Planted selective route | synthetic demonstration",
                "02-generic-degradation": "Generic degradation trap | synthetic demonstration",
                "03-incomplete-experiment": "Missing intervention case | synthetic demonstration",
            }[name],
        ) == page
        assert "Synthetic diagnostic" in page
        assert "not independently reproduced" in page
        if name != "03-incomplete-experiment":
            assert "Not a calibrated p-value" in page
        assert "default-src &#" not in page
        assert "default-src 'none'" in page
        assert "<script" not in page
        assert "https://" not in page
        assert summary["status"] in page
    assert '<svg viewBox' in (tmp_path / "01-planted-path.html").read_text()
    assert '<svg viewBox' not in (tmp_path / "03-incomplete-experiment.html").read_text()
    assert "no effect size" in (tmp_path / "03-incomplete-experiment.html").read_text()
    assert "Missing intervention case" in (tmp_path / "03-incomplete-experiment.html").read_text()


def test_report_does_not_invent_metrics_from_partial_observations():
    result = demo.synthetic_case(candidate_effect=3, incomplete=True)
    page = render_paired_selectivity_html(result)
    assert "Effects unavailable" in page
    assert "No chart is drawn" in page
    assert "Candidate effect" not in page
    assert "Per-trial effect summaries are unavailable" in page
    assert "Evidence ladder" in page
    assert "Hash verified" in page


def test_integrity_tampering_and_forced_authority_fail_closed():
    result = demo.synthetic_case(candidate_effect=3)
    tampered = dict(result)
    tampered["mean_candidate_difference_in_differences"] = 900
    with pytest.raises(ValueError, match="result_hash mismatch"):
        render_paired_selectivity_html(tampered)

    forged = {**result, "real_model_evidence": True}
    forged["result_hash"] = stable_hash(
        {key: value for key, value in forged.items() if key != "result_hash"}
    )
    with pytest.raises(ValueError, match="unexpected authority flag"):
        render_paired_selectivity_html(forged)

    mislabeled = {**result, "supports_selective_effect": False}
    mislabeled["result_hash"] = stable_hash(
        {key: value for key, value in mislabeled.items() if key != "result_hash"}
    )
    with pytest.raises(ValueError, match="contradictory status"):
        render_paired_selectivity_html(mislabeled)


def test_summary_and_pair_mean_must_agree_even_with_recomputed_hash():
    result = demo.synthetic_case(candidate_effect=3)
    result["pair_evidence"][0]["candidate_effect"] = 999.0
    result["result_hash"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_hash"}
    )
    with pytest.raises(ValueError, match="summary disagrees"):
        render_paired_selectivity_html(result)


def test_html_escapes_untrusted_identifiers_and_title():
    result = demo.synthetic_case(candidate_effect=3)
    malicious = '<img src=x onerror="alert(1)">'
    result["pair_evidence"][0]["pair_id"] = malicious
    result["random_control_mean_effects"][0]["circuit_id"] = malicious
    result["result_hash"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_hash"}
    )
    page = render_paired_selectivity_html(result, title="<script>alert(1)</script>")
    assert malicious not in page
    assert "&lt;img" in page
    assert "&lt;script&gt;" in page
    assert "<script>" not in page


def test_nonfinite_injected_values_and_drifted_rank_rejected():
    result = demo.synthetic_case(candidate_effect=3)
    result["pair_evidence"][0]["candidate_effect"] = float("nan")
    result["result_hash"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_hash"}
    )
    with pytest.raises(ValueError, match="must be finite"):
        render_paired_selectivity_html(result)

    result = demo.synthetic_case(candidate_effect=3)
    result["random_control_plus_one_rank_fraction"] = 1.0
    result["result_hash"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_hash"}
    )
    with pytest.raises(ValueError, match="rank disagrees"):
        render_paired_selectivity_html(result)


def test_duplicate_json_keys_rejected_and_writer_deterministic(tmp_path):
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads('{"status":"A","status":"B"}', object_pairs_hook=_unique_json_keys)
    result = demo.synthetic_case(candidate_effect=3)
    one = write_paired_selectivity_html(result, tmp_path / "a.html")
    two = write_paired_selectivity_html(result, tmp_path / "b.html")
    assert one.read_bytes() == two.read_bytes()


def test_missing_case_label_is_html_escaped():
    result = demo.synthetic_case(candidate_effect=3, incomplete=True)
    result["missing"][0]["pair_id"] = '<svg onload="bad()">'
    result["result_hash"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_hash"}
    )
    page = render_paired_selectivity_html(result)
    assert "<svg onload=" not in page
    assert "&lt;svg" in page


def test_nonestimable_cannot_carry_fabricated_effect_summary():
    result = demo.synthetic_case(candidate_effect=3, incomplete=True)
    result["mean_candidate_difference_in_differences"] = 0.0
    result["result_hash"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_hash"}
    )
    with pytest.raises(ValueError, match="cannot include completed effects"):
        render_paired_selectivity_html(result)
