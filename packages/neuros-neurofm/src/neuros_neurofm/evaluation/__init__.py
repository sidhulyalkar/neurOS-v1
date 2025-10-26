"""
Evaluation utilities for NeuroFM-X.

Provides metrics, benchmarks, visualization tools, and comprehensive
evaluation frameworks including task registry, zero-shot, and few-shot evaluation.
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
from neuros_neurofm.evaluation.task_registry import (
    TaskRegistry,
    EvaluationTask,
    TaskMetadata,
    TaskType,
    Species,
    Modality,
    get_global_registry,
    register_task,
    get_task,
    list_tasks,
)
from neuros_neurofm.evaluation.zero_shot import (
    ZeroShotEvaluator,
    ZeroShotConfig,
    LinearProbe,
    run_zero_shot_suite,
)
from neuros_neurofm.evaluation.few_shot_eval import (
    FewShotEvaluator,
    FewShotConfig,
    LoRALayer,
    apply_lora_to_model,
    run_few_shot_suite,
)

__all__ = [
    # Metrics
    "r2_score",
    "pearson_correlation",
    "bits_per_spike",
    "EvaluationMetrics",
    "evaluate_model",
    # FALCON Benchmark
    "FALCONBenchmark",
    "run_falcon_benchmark",
    # Visualization
    "plot_latent_space",
    "plot_behavioral_predictions",
    "plot_neural_forecasts",
    "summarize_model_performance",
    # Task Registry
    "TaskRegistry",
    "EvaluationTask",
    "TaskMetadata",
    "TaskType",
    "Species",
    "Modality",
    "get_global_registry",
    "register_task",
    "get_task",
    "list_tasks",
    # Zero-Shot Evaluation
    "ZeroShotEvaluator",
    "ZeroShotConfig",
    "LinearProbe",
    "run_zero_shot_suite",
    # Few-Shot Evaluation
    "FewShotEvaluator",
    "FewShotConfig",
    "LoRALayer",
    "apply_lora_to_model",
    "run_few_shot_suite",
]
