from neuros_mechint.benchmarks import GroundTruthCausalMLP, make_ground_truth_pair
from neuros_mechint.circuits import ModuleActivationPatcher, PathPatcher


def _metric(output):
    return output.mean()


def test_activation_patching_and_path_patching_are_distinct_and_correct():
    model = GroundTruthCausalMLP().eval()
    pair = make_ground_truth_pair()

    activation = ModuleActivationPatcher(
        model,
        _metric,
        layers_to_patch=["source", "causal", "nuisance"],
    ).patch_all(pair.clean, pair.corrupted)
    activation_scores = {effect.layer_name: effect.direct_effect for effect in activation.effects}
    assert activation_scores["source"] == 1.0
    assert activation_scores["causal"] == 1.0
    assert activation_scores["nuisance"] == 0.0

    paths = PathPatcher(
        model,
        _metric,
        layers_to_patch=["source", "causal", "nuisance"],
    ).patch_all_paths(
        pair.clean,
        pair.corrupted,
        senders=["source"],
        receivers=["causal", "nuisance"],
    )
    path_scores = {(effect.sender, effect.receiver): effect.mediated_effect for effect in paths.effects}
    assert path_scores[("source", "causal")] == 1.0
    assert path_scores[("source", "nuisance")] == 0.0
