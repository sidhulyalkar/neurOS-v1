from __future__ import annotations

from neuros.drivers.unicorn_compatibility import run_unicorn_compatibility_suite


def test_unicorn_compatibility_suite_passes_declared_synthetic_contracts():
    report = run_unicorn_compatibility_suite(seed=41)
    assert report.passed, report.to_dict()
    payload = report.to_dict()
    assert payload["schema"] == "neuros.unicorn_hybrid_black_sim.compatibility.v1"
    assert payload["synthetic"] is True
    surfaces = {surface["name"]: surface for surface in payload["surfaces"]}
    assert surfaces["raw_udp17_wire"]["observations"]["auxiliary_tail"] == ["BAT", "CNT", "VALID"]
    assert surfaces["direct_api17_scan"]["observations"]["channels"] == 17
    assert surfaces["recorder19_fields"]["observations"]["fields"] == 19
    assert surfaces["bandpower70_reference"]["evidence_class"] == "reference_implementation"
    assert surfaces["motion_and_battery_stress_policy"]["evidence_class"] == "synthetic_assumption"
    assert "cannot qualify" in payload["evidence_boundary"]
