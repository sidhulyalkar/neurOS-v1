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

__all__ = [
    # Hyperparameter search
    'HyperparameterSearch',
    'GridSearch',
    'create_neurofmx_objective',
    'save_best_hyperparameters',
    'load_hyperparameters',
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
