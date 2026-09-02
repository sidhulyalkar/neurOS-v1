from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "03_nim_evidence_tournament.py"
    spec = importlib.util.spec_from_file_location("test_nim_evidence_tournament_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load NIM evidence tournament example")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "title": "Prospective geometry screen",
        "statement": "Frozen geometry predicts later matched validation gain.",
        "rationale": "This tests whether development geometry has prospective screening value.",
        "family": "neural_geometry",
        "changed_variables": ["representation.geometry_family"],
        "required_payload_classes": ["source_code", "aggregate_metrics"],
        "development_metrics": ["prospective_geometry_gain_spearman"],
        "primary_metric": "prospective_geometry_gain_spearman",
        "claim_relation": "prospective_prediction",
        "control_description": "Freeze five geometry scores before matched validation deltas reveal.",
        "supports_if": [
            {
                "metric": "prospective_geometry_gain_spearman",
                "operator": ">=",
                "threshold": 0.5,
                "rationale": "positive prospective rank relation",
            }
        ],
        "rejects_if": [
            {
                "metric": "prospective_geometry_gain_spearman",
                "operator": "<=",
                "threshold": 0.0,
                "rationale": "no positive prospective relation",
            }
        ],
        "falsification_test": "Reject if the frozen geometry ranking does not predict later gain.",
        "estimated_compute_tier": "low",
    }


def test_shape_gate_reports_all_missing_fields_across_candidates() -> None:
    module = _load_module()
    first = _candidate("first")
    second = _candidate("second")
    first.pop("title")
    first.pop("required_payload_classes")
    second.pop("claim_relation")
    second.pop("supports_if")

    with pytest.raises(ValueError) as excinfo:
        module._require_complete_semantic_candidate_shape(
            {"candidates": [first, second, _candidate("third"), _candidate("fourth"), _candidate("fifth")]}
        )

    message = str(excinfo.value)
    assert "candidate[0] missing fields: title, required_payload_classes" in message
    assert "candidate[1] missing fields: claim_relation, supports_if" in message


def test_shape_gate_never_fills_missing_scientific_content() -> None:
    module = _load_module()
    candidate = _candidate("first")
    candidate.pop("control_description")
    original = dict(candidate)
    with pytest.raises(ValueError, match="control_description"):
        module._require_complete_semantic_candidate_shape(
            {
                "candidates": [
                    candidate,
                    _candidate("second"),
                    _candidate("third"),
                    _candidate("fourth"),
                    _candidate("fifth"),
                ]
            }
        )
    assert candidate == original
    assert "control_description" not in candidate


def test_repair_schema_contract_repeats_every_required_field_and_original_contract() -> None:
    module = _load_module()
    original_contract = "OUTPUT_SCHEMA_EXAMPLE={candidate contract sentinel}"
    prompt = module._repair_schema_contract(original_contract)
    for field in module._REQUIRED_SEMANTIC_CANDIDATE_FIELDS:
        assert field in prompt
    assert original_contract in prompt
    assert "Do not synthesize hidden outcomes" in prompt


def test_wrapped_parser_runs_shape_gate_before_semantic_parser() -> None:
    module = _load_module()
    calls: list[dict[str, object]] = []

    def parser(payload, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(payload)
        return kwargs

    wrapped = module._wrap_semantic_parser(parser)
    bad = _candidate("first")
    bad.pop("title")
    with pytest.raises(ValueError, match="title"):
        wrapped(
            {
                "candidates": [
                    bad,
                    _candidate("second"),
                    _candidate("third"),
                    _candidate("fourth"),
                    _candidate("fifth"),
                ]
            },
            sentinel=True,
        )
    assert calls == []

    payload = {
        "candidates": [
            _candidate("first"),
            _candidate("second"),
            _candidate("third"),
            _candidate("fourth"),
            _candidate("fifth"),
        ]
    }
    assert wrapped(payload, sentinel=True) == {"sentinel": True}
    assert calls == [payload]
