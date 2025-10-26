"""
Optimization and compression utilities for NeuroFM-X.

Provides tools for model compression, quantization, and hyperparameter tuning.
"""

from neuros_neurofm.optimization.hyperparameter_search import (
    HyperparameterSearch,
    GridSearch,
    create_neurofmx_objective,
    save_best_hyperparameters,
    load_hyperparameters,
)

from neuros_neurofm.optimization.model_compression import (
    ModelQuantizer,
    ModelPruner,
    KnowledgeDistiller,
    TorchScriptExporter,
    MixedPrecisionOptimizer,
    compute_model_size,
    compare_model_sizes,
    save_compression_config,
    load_compression_config,
)

# Ray Tune-based hyperparameter search (optional import)
try:
    from neuros_neurofm.optimization.ray_tune_search import (
        NeuroFMXRayTuner,
        NeuroFMXSearchSpace,
        create_neurofmx_train_fn,
    )
    RAY_TUNE_AVAILABLE = True
except ImportError:
    RAY_TUNE_AVAILABLE = False
    NeuroFMXRayTuner = None
    NeuroFMXSearchSpace = None
    create_neurofmx_train_fn = None

__all__ = [
    # Hyperparameter search (Optuna-based)
    'HyperparameterSearch',
    'GridSearch',
    'create_neurofmx_objective',
    'save_best_hyperparameters',
    'load_hyperparameters',
    # Ray Tune-based hyperparameter search
    'NeuroFMXRayTuner',
    'NeuroFMXSearchSpace',
    'create_neurofmx_train_fn',
    'RAY_TUNE_AVAILABLE',
    # Model compression
    'ModelQuantizer',
    'ModelPruner',
    'KnowledgeDistiller',
    'TorchScriptExporter',
    'MixedPrecisionOptimizer',
    'compute_model_size',
    'compare_model_sizes',
    'save_compression_config',
    'load_compression_config',
]
