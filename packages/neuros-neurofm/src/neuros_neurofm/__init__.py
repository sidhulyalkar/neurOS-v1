"""
NeuroFM-X: Foundation Model for Neural Population Dynamics.

This package provides a state-of-the-art foundation model combining:
- Selective State-Space Models (Mamba/SSM) for linear-complexity sequence modeling
- Perceiver-IO for multi-modal fusion
- Population Transformers (PopT) for neural population aggregation
- Latent Diffusion for generative modeling
- Transfer learning adapters (Unit-ID, LoRA)
- Real-time inference with dynamic batching
- Model compression and optimization
- Production deployment tools
"""

__version__ = "0.1.0"

# Core models
from neuros_neurofm.models.neurofmx import NeuroFMX
from neuros_neurofm.training.trainer import NeuroFMXTrainer

# Datasets
from neuros_neurofm.datasets.nwb_loader import (
    NWBDataset,
    IBLDataset,
    AllenDataset,
    create_nwb_dataloaders,
)

# Tokenizers
from neuros_neurofm.tokenizers.calcium_tokenizer import (
    CalciumTokenizer,
    TwoPhotonTokenizer,
    MiniscopeTokenizer,
)

# Optimization
from neuros_neurofm.optimization.hyperparameter_search import (
    HyperparameterSearch,
    GridSearch,
    create_neurofmx_objective,
)
from neuros_neurofm.optimization.model_compression import (
    ModelQuantizer,
    ModelPruner,
    KnowledgeDistiller,
    TorchScriptExporter,
    MixedPrecisionOptimizer,
)

# Inference
from neuros_neurofm.inference.realtime_pipeline import (
    RealtimeInferencePipeline,
    DynamicBatcher,
    ModelCache,
    LatencyProfiler,
)

__all__ = [
    # Core
    "NeuroFMX",
    "NeuroFMXTrainer",
    "__version__",
    # Datasets
    "NWBDataset",
    "IBLDataset",
    "AllenDataset",
    "create_nwb_dataloaders",
    # Tokenizers
    "CalciumTokenizer",
    "TwoPhotonTokenizer",
    "MiniscopeTokenizer",
    # Optimization
    "HyperparameterSearch",
    "GridSearch",
    "create_neurofmx_objective",
    "ModelQuantizer",
    "ModelPruner",
    "KnowledgeDistiller",
    "TorchScriptExporter",
    "MixedPrecisionOptimizer",
    # Inference
    "RealtimeInferencePipeline",
    "DynamicBatcher",
    "ModelCache",
    "LatencyProfiler",
]
