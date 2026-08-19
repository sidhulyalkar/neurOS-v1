from neuros_mechint.benchmarks import GroundTruthCausalMLP, make_ground_truth_pair
from neuros_mechint.circuits import ModuleCircuitDiscovery


def test_module_pruning_reports_honest_semantics_and_final_performance():
    model = GroundTruthCausalMLP().eval()
    pair = make_ground_truth_pair()
    target = model(pair.clean).detach()

    discovery = ModuleCircuitDiscovery(model, threshold=2.0, ablation_method="zero")
    circuit = discovery.discover_circuit(pair.clean, target)

    assert circuit.metadata["algorithm"] == "acdc_inspired_module_pruning"
    assert circuit.metadata["faithful_acdc"] is False
    assert set(circuit.metadata["removed_modules"]) == {"source", "causal", "nuisance"}
    # With every leaf module ablated, prediction is 0 rather than target 1,
    # so the final negative-MSE score must be -1 rather than the full-model 0.
    assert circuit.performance == -1.0
    assert circuit.sparsity == 0.0
