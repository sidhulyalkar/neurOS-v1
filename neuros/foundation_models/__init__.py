"""
Foundation models for large-scale neural decoding.

This package provides wrappers for state-of-the-art foundation models trained on
large-scale neuroscience datasets, including:

- POYO/POYO+: Multi-session, multi-task neural decoding (Azabou et al., NeurIPS 2023, ICLR 2025)
- NDT2/NDT3: Neural Data Transformers (Ye & Pandarinath, NeurIPS 2023, 2025)
- CEBRA: Learnable latent embeddings (Schneider et al., Nature 2023)
- Neuroformer: Multimodal generative pretraining (Gobryal et al., ICLR 2024)

All foundation models extend the neurOS BaseModel interface for seamless integration.
"""

from __future__ import annotations

from neuros.foundation_models.base_foundation_model import BaseFoundationModel

# POYO models
try:
    from neuros.foundation_models.poyo_model import POYOModel, POYOPlusModel

    POYO_AVAILABLE = True
except ImportError:
    POYO_AVAILABLE = False
    POYOModel = None
    POYOPlusModel = None

# NDT models
try:
    from neuros.foundation_models.ndt_model import NDT2Model, NDT3Model

    NDT_AVAILABLE = True
except ImportError:
    NDT_AVAILABLE = False
    NDT2Model = None
    NDT3Model = None

# CEBRA models
try:
    from neuros.foundation_models.cebra_model import CEBRAModel

    CEBRA_AVAILABLE = True
except ImportError:
    CEBRA_AVAILABLE = False
    CEBRAModel = None

# Neuroformer models
try:
    from neuros.foundation_models.neuroformer_model import NeuroformerModel

    NEUROFORMER_AVAILABLE = True
except ImportError:
    NEUROFORMER_AVAILABLE = False
    NeuroformerModel = None

__all__ = [
    "BaseFoundationModel",
    "POYOModel",
    "POYOPlusModel",
    "POYO_AVAILABLE",
    "NDT2Model",
    "NDT3Model",
    "NDT_AVAILABLE",
    "CEBRAModel",
    "CEBRA_AVAILABLE",
    "NeuroformerModel",
    "NEUROFORMER_AVAILABLE",
]
