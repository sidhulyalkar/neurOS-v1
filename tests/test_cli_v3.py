import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from neuros.cli import _parse_args
from neuros.cli.config_commands import execute_config, validate_config
from neuros.cli.diagnostics import doctor, plugin_inventory
from neuros.cli.entrypoint import _run_init
from neuros.cli.project_commands import init_project
from neuros.errors import ConfigurationError


CONFIG = Path("configs/examples/mock_bci.yaml")


def test_v3_parser_preserves_legacy_run_defaults():
    with patch.object(sys, "argv", ["neuros", "run"]):
        args = _parse_args()
    assert args.command == "run"
    assert args.config is None
    assert args.duration == 5.0


def test_v3_parser_accepts_config_run():
    with patch.object(sys, "argv", ["neuros", "run", str(CONFIG), "--duration", "0.1"]):
        args = _parse_args()
    assert args.config == str(CONFIG)
    assert args.duration == 0.1


def test_validate_config_resolves_executable_plugin_graph():
    result = validate_config(CONFIG)
    assert result["decoder"] == "threshold"
    assert result["streams"] == ["eeg"]
    assert "transform:eeg:0" in result["graph"]["nodes"]


@pytest.mark.asyncio
async def test_execute_config_runs_native_runtime():
    result = await execute_config(CONFIG, duration_s=0.05)
    assert result["state"] == "stopped"
    assert result["nodes"]["decoder:primary"]["processed"] > 0


def test_doctor_and_plugin_inventory_are_machine_readable():
    report = doctor()
    json.dumps(report)
    assert report["python"]["supported"] is True
    plugins = plugin_inventory()
    assert any(item["kind"] == "source" and item["name"] == "mock" for item in plugins)
    assert any(item["kind"] == "decoder" and item["name"] == "threshold" for item in plugins)


def test_init_project_creates_a_production_validated_starter(tmp_path: Path):
    root = tmp_path / "starter"
    result = init_project(root)

    assert result["template"] == "mock-bci"
    assert result["replaced"] == []
    assert set(result["created"]) == {".gitignore", "README.md", "neuros.yaml"}
    assert "software/runtime evidence only" in result["evidence_boundary"]

    resolved = validate_config(root / "neuros.yaml")
    assert resolved["decoder"] == "threshold"
    assert resolved["streams"] == ["eeg"]
    assert "neuros qualify neuros.yaml" in (root / "README.md").read_text(encoding="utf-8")


def test_init_project_creates_external_nsq_method_starter(tmp_path: Path):
    root = tmp_path / "method"
    result = init_project(root, template="nsq-method")

    assert result["template"] == "nsq-method"
    assert result["config"] is None
    assert set(result["created"]) == {
        ".gitignore",
        "README.md",
        "demo.py",
        "method.py",
        "pyproject.toml",
        "test_method.py",
    }
    assert result["next_commands"] == ["python demo.py", "pytest -q"]
    assert "numerical scores cannot support" in result["evidence_boundary"]

    method = (root / "method.py").read_text(encoding="utf-8")
    demo = (root / "demo.py").read_text(encoding="utf-8")
    test = (root / "test_method.py").read_text(encoding="utf-8")
    assert "ExternalDecoderMethodSpec" in method
    assert 'state_identity_kind="tensor_sha256"' in method
    assert "unsafe_pickle_used" in method
    assert "run_external_qualification_case" in demo
    assert '"numerical_result_interpretable": False' in demo
    assert "evaluation_indices_sha256" in test


def test_init_project_refuses_to_replace_managed_files_without_force(tmp_path: Path):
    root = tmp_path / "starter"
    init_project(root)
    original = (root / "README.md").read_text(encoding="utf-8")
    (root / "unrelated.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="--force"):
        init_project(root)

    assert (root / "README.md").read_text(encoding="utf-8") == original
    assert (root / "unrelated.txt").read_text(encoding="utf-8") == "keep me"

    result = init_project(root, force=True)
    assert set(result["replaced"]) == {".gitignore", "README.md", "neuros.yaml"}
    assert (root / "unrelated.txt").read_text(encoding="utf-8") == "keep me"


def test_init_entrypoint_emits_machine_readable_project_manifest(tmp_path: Path, capsys):
    root = tmp_path / "starter"
    code = _run_init([str(root), "--json"])
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["project_root"] == str(root.resolve())
    assert payload["config"] == str((root / "neuros.yaml").resolve())
    assert payload["next_commands"][0] == "neuros doctor"


def test_init_entrypoint_supports_nsq_method_template(tmp_path: Path, capsys):
    root = tmp_path / "method"
    code = _run_init([str(root), "--template", "nsq-method", "--json"])
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["template"] == "nsq-method"
    assert payload["config"] is None
    assert payload["next_commands"] == ["python demo.py", "pytest -q"]
