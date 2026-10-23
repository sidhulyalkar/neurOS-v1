"""A public entrypoint must actually run and reject the decoy circuit."""

import importlib.util
from pathlib import Path

import pytest

from neuros_mechint.benchmarks import read_evidence_pack_artifact


@pytest.mark.parametrize("label,expected", [("known-route", True), ("nuisance-route", False)])
def test_circuit_auditor_quickstart_exports_honest_artifacts(tmp_path, label, expected):
    script = Path(__file__).resolve().parents[1] / "examples" / "23_circuit_auditor_quickstart.py"
    spec = importlib.util.spec_from_file_location("circuit_auditor_quickstart", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outcomes = module.main(tmp_path)
    assert outcomes[label]["passed"] is expected
    assert (tmp_path / f"{label}.json").is_file()
    artifact = read_evidence_pack_artifact(tmp_path / f"{label}.json")
    assert artifact["study_fingerprint"] == outcomes[label]["study_fingerprint"]
    assert artifact["promotion"]["passed"] is expected
