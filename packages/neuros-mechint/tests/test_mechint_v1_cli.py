import json

from neuros_mechint.cli import main


def test_cli_lists_frozen_v1_schemas(capsys):
    assert main(["schemas", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["family"] for item in payload} == {
        "evidence_pack",
        "factorial",
        "correspondence",
        "replication",
        "dose_response",
    }


def test_cli_release_status_separates_software_from_empirical_closure(capsys):
    assert main(["release-status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["software_contract_ready"] is True
    assert payload["empirical_evidence_complete"] is False
    assert "real-neural-factorial-study" in payload["pending_empirical_requirements"]


def test_cli_v1_ground_truth_is_an_adversarial_contract_gate(capsys):
    assert main(["v1-ground-truth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["duplicate_run_rejected"] is True
    assert payload["decision_flip_rejected"] is True
    assert payload["empirical_overclaim_rejected"] is True
