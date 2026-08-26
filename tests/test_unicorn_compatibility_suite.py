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
    assert surfaces["acquisition_availability_delay"]["evidence_class"] == "reference_implementation"
    assert surfaces["motion_and_battery_stress_policy"]["evidence_class"] == "synthetic_assumption"
    assert "cannot qualify" in payload["evidence_boundary"]


def test_every_vendor_backed_unicorn_surface_has_frozen_upstream_provenance():
    payload = run_unicorn_compatibility_suite(seed=43).to_dict()
    references = payload["upstream_contract_snapshot"]
    assert references

    reference_ids = {reference["reference_id"] for reference in references}
    assert {
        "unicorn-python-api-reference",
        "unicorn-raw-udp-interface",
        "unicorn-recorder-hybrid-black",
        "unicorn-bandpower-hybrid-black",
    } <= reference_ids

    covered_surfaces = {
        surface_name
        for reference in references
        for surface_name in reference["supports"]
    }
    vendor_backed_surfaces = {
        surface["name"]
        for surface in payload["surfaces"]
        if surface["evidence_class"] != "synthetic_assumption"
    }
    assert vendor_backed_surfaces <= covered_surfaces

    github_references = [
        reference
        for reference in references
        if "github.com/unicorn-bi/" in reference["locator"]
    ]
    assert github_references
    assert all(reference["revision"].startswith("git-blob:") for reference in github_references)
    assert all(reference["retrieved_date"] == "2026-08-26" for reference in references)
