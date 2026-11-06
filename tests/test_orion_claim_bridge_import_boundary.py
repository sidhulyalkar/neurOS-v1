from __future__ import annotations

import builtins

import pytest

import neuros
from neuros.evidence import (
    EvidenceRelation,
    EvidenceRequirement,
    EvidenceTier,
    ScientificClaimSpec,
)
from neuros.evidence.orion_bridge import bind_orion_study_claim


def _claim() -> ScientificClaimSpec:
    return ScientificClaimSpec(
        claim_id="import-boundary-fixture",
        statement="The import-boundary fixture is only used to test fail-closed dependency handling.",
        domain="task_utility",
        scope="import-boundary regression",
        inference_unit="fixture",
        target_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        requirements=(
            EvidenceRequirement(
                requirement_id="fixture-contract",
                authority_type="test.fixture",
                description="Exercise ORION import failure semantics without asserting scientific evidence.",
                required_tier=EvidenceTier.SOFTWARE_CONTRACT,
            ),
        ),
    )


def _failing_import(real_import, *, missing_name: str):
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "orion.scientific_authority":
            exc = ModuleNotFoundError(f"No module named {missing_name!r}")
            exc.name = missing_name
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def test_missing_top_level_orion_is_reported_as_optional_dependency(monkeypatch) -> None:
    monkeypatch.setattr(
        builtins,
        "__import__",
        _failing_import(builtins.__import__, missing_name="orion"),
    )

    with pytest.raises(RuntimeError, match="ORION claim binding requires"):
        bind_orion_study_claim(
            _claim(),
            object(),
            relation=EvidenceRelation.CONTEXT,
            declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        )


def test_broken_installed_orion_dependency_is_not_masked(monkeypatch) -> None:
    monkeypatch.setattr(
        builtins,
        "__import__",
        _failing_import(builtins.__import__, missing_name="orion_broken_dependency"),
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        bind_orion_study_claim(
            _claim(),
            object(),
            relation=EvidenceRelation.CONTEXT,
            declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        )

    assert exc_info.value.name == "orion_broken_dependency"


def test_lazy_sdk_export_reports_genuinely_missing_optional_distribution(monkeypatch) -> None:
    def missing_driver_distribution(module_name: str):
        assert module_name == "neuros.drivers.base_driver"
        exc = ModuleNotFoundError("No module named 'neuros.drivers'")
        exc.name = "neuros.drivers"
        raise exc

    monkeypatch.setattr(neuros, "import_module", missing_driver_distribution)

    with pytest.raises(ImportError, match="requires the corresponding neurOS driver/model"):
        neuros.__getattr__("BaseDriver")


def test_lazy_sdk_export_does_not_mask_internal_dependency_failure(monkeypatch) -> None:
    def broken_driver_distribution(module_name: str):
        assert module_name == "neuros.drivers.base_driver"
        exc = ModuleNotFoundError("No module named 'driver_internal_dependency'")
        exc.name = "driver_internal_dependency"
        raise exc

    monkeypatch.setattr(neuros, "import_module", broken_driver_distribution)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        neuros.__getattr__("BaseDriver")

    assert exc_info.value.name == "driver_internal_dependency"
