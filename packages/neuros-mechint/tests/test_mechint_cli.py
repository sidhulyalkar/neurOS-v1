import json

from neuros_mechint.cli import main


def test_cli_methods_json(capsys):
    assert main(["methods", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(item["name"] == "module_activation_patching" for item in payload)
    assert any(item["name"] == "circuit_faithfulness_benchmark" for item in payload)
    assert any(item["name"] == "held_out_circuit_evidence_pack" for item in payload)
    assert any(item["name"] == "factorial_mechanism_design" for item in payload)
    assert any(item["name"] == "feature_correspondence_design" for item in payload)
    assert any(item["name"] == "held_out_causal_feature_substitution" for item in payload)
    assert any(item["name"] == "claim_aware_hierarchical_replication" for item in payload)
    assert any(item["name"] == "intervention_dose_response" for item in payload)


def test_cli_evidence_recipes_lists_all_supported_ecosystems(capsys):
    assert main(["evidence-recipes", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    ecosystems = {item["ecosystem"] for item in payload}
    assert ecosystems == {"circuit_tracer", "nnsight", "sae_lens", "transformer_lens"}
    assert all(item["revision_policy"].startswith("resolve-and-pin") for item in payload)


def test_cli_ground_truth_is_a_scientific_smoke(capsys):
    assert main(["ground-truth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["localization"]["passed_separation"] is True


def test_cli_shared_computation_ground_truth_recovers_both_scenarios(capsys):
    assert main(["shared-computation-ground-truth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["shared_mechanism"]["passed"] is True
    assert payload["architecture_specific"]["passed"] is True


def test_cli_mechanism_emergence_ground_truth_recovers_known_transition(capsys):
    assert main(["mechanism-emergence-ground-truth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["report"]["global_stable_step"] == 200


def test_cli_circuit_faithfulness_ground_truth_recovers_known_circuit(capsys):
    assert main(["circuit-faithfulness-ground-truth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["report"]["sufficiency_fraction"] == 1.0
    assert payload["report"]["necessity_fraction"] == 1.0


def test_cli_held_out_gate_rejects_known_discovery_overfit(capsys):
    assert main(["evidence-pack-generalization-ground-truth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["candidate"]["targets"] == ["route_a"]
    assert payload["discovery_aggregate"]["pass_rate"] == 1.0
    assert payload["validation_aggregate"]["pass_rate"] == 0.0
    assert payload["promotion"]["passed"] is False


def test_cli_factorial_gate_recovers_interaction_and_rejects_invalid_designs(capsys):
    assert main(["factorial-ground-truth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["observed_interactions"] == [-0.5, -0.5]
    assert payload["confound_rejected"] is True
    assert payload["missing_cell_rejected"] is True


def test_cli_correspondence_gate_rejects_predictive_noncausal_decoy(capsys):
    assert main(["correspondence-ground-truth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["true_correspondence_passed"] is True
    assert payload["decoy_similarity_high"] is True
    assert payload["decoy_causal_rejected"] is True
    assert payload["decoy_validation_predictive_r2"] > 0.99
    assert payload["decoy_median_source_effect"] == 0.0


def test_cli_replication_gate_rejects_pseudoreplication(capsys):
    assert main(["replication-ground-truth", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["true_replication_recovered"] is True
    assert payload["independent_seed_count_correct"] is True
    assert payload["pseudoreplication_rejected"] is True
    assert payload["heterogeneous_replication_rejected"] is True
    assert payload["dose_response_recovered"] is True
