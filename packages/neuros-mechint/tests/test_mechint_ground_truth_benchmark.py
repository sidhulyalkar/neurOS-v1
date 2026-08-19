from neuros_mechint.benchmarks import run_ground_truth_benchmark


def test_ground_truth_benchmark_recovers_known_mechanism_and_rejects_nuisance():
    report = run_ground_truth_benchmark()
    localization = report["localization"]
    assert localization["precision_at_k"] == 1.0
    assert localization["recall_at_k"] == 1.0
    assert localization["average_precision"] == 1.0
    assert localization["passed_separation"] is True
    assert report["scores"]["nuisance"] == 0.0
