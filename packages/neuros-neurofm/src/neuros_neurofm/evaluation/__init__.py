"""
Evaluation utilities for NeuroFM-X.

Provides metrics, benchmarks, and visualization tools.
"""

from neuros_neurofm.evaluation.metrics import (
    r2_score,
    pearson_correlation,
    bits_per_spike,
    EvaluationMetrics,
    evaluate_model,
)
from neuros_neurofm.evaluation.falcon import (
    FALCONBenchmark,
    run_falcon_benchmark,
)
from neuros_neurofm.evaluation.visualization import (
    plot_latent_space,
    plot_behavioral_predictions,
    plot_neural_forecasts,
    summarize_model_performance,
)

__all__ = [
    "r2_score",
    "pearson_correlation",
    "bits_per_spike",
    "EvaluationMetrics",
    "evaluate_model",
    "FALCONBenchmark",
    "run_falcon_benchmark",
    "plot_latent_space",
    "plot_behavioral_predictions",
    "plot_neural_forecasts",
    "summarize_model_performance",
]
