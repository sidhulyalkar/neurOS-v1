from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "evidence" / "validate_promoted_environment.py"
_SPEC = importlib.util.spec_from_file_location("_promoted_environment_validator", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load promoted environment validator")
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)


def test_exact_constraint_loader_is_normalized_and_fail_closed(tmp_path: Path):
    path = tmp_path / "constraints.txt"
    path.write_text("# fixture\nNumPy==2.4.6\nscikit_learn==1.9.0\n", encoding="utf-8")
    assert validator.load_exact_constraints(path) == {
        "numpy": "2.4.6",
        "scikit-learn": "1.9.0",
    }

    path.write_text("numpy>=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly name==version"):
        validator.load_exact_constraints(path)


def test_promoted_environment_accepts_only_exact_external_frontier(monkeypatch):
    monkeypatch.setattr(
        validator,
        "observed_distributions",
        lambda: {
            "neuros": "2.1.0",
            "numpy": "2.4.6",
            "torch": "2.13.0",
        },
    )
    result = validator.validate_promoted_environment(
        {"numpy": "2.4.6", "torch": "2.13.0"},
        local_distributions={"neuros"},
        required_external={"numpy", "torch"},
    )
    assert result["verified"] is True
    assert result["installed_external_count"] == 2


def test_promoted_environment_rejects_unexpected_distribution(monkeypatch):
    monkeypatch.setattr(
        validator,
        "observed_distributions",
        lambda: {"numpy": "2.4.6", "foreign": "1"},
    )
    with pytest.raises(RuntimeError, match="unexpected=.*foreign"):
        validator.validate_promoted_environment(
            {"numpy": "2.4.6"},
            local_distributions=set(),
            required_external={"numpy"},
        )


def test_promoted_environment_rejects_version_drift(monkeypatch):
    monkeypatch.setattr(
        validator,
        "observed_distributions",
        lambda: {"numpy": "2.4.7"},
    )
    with pytest.raises(RuntimeError, match="mismatched=.*2.4.6"):
        validator.validate_promoted_environment(
            {"numpy": "2.4.6"},
            local_distributions=set(),
            required_external={"numpy"},
        )


def test_promoted_environment_rejects_missing_required_distribution(monkeypatch):
    monkeypatch.setattr(validator, "observed_distributions", lambda: {"numpy": "2.4.6"})
    with pytest.raises(RuntimeError, match="missing_required=.*torch"):
        validator.validate_promoted_environment(
            {"numpy": "2.4.6", "torch": "2.13.0"},
            local_distributions=set(),
            required_external={"numpy", "torch"},
        )
