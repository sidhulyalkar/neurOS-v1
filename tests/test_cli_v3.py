import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from neuros.cli import _parse_args
from neuros.cli.config_commands import execute_config, validate_config
from neuros.cli.diagnostics import doctor, plugin_inventory


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
