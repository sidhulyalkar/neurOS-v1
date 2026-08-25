import json
import sys

from neuros.cli.app import main


def test_compatibility_cli_json(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["neuros", "compatibility", "mne", "--json"])

    main()

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["integration_id"] == "mne"
    assert payload[0]["status"] == "supported"
    assert payload[0]["evidence_tier"] == "integration"


def test_compatibility_cli_status_filter(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["neuros", "compatibility", "--status", "planned"])

    main()

    output = capsys.readouterr().out
    assert "neuralbench" in output
    assert "braindecode" not in output
    assert "brainflow" not in output


def test_compatibility_cli_experimental_filter_includes_braindecode(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["neuros", "compatibility", "--status", "experimental"])

    main()

    output = capsys.readouterr().out
    assert "braindecode" in output
    assert "evidence=integration" in output
    assert "neuralbench" not in output
