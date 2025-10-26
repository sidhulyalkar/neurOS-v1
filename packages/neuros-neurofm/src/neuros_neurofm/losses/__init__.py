"""
Loss Functions for NeuroFMx Multimodal Training

Includes:
- Masked modeling losses (random, block, adaptive masking)
- Multi-horizon forecasting losses
- Denoising diffusion losses
- Tri-modal contrastive loss (neural + behavior + stimulus)
- Domain adversarial loss
- Uncertainty-weighted multi-task loss
- Unified loss registry and configuration
"""

# Contrastive losses
from neuros_neurofm.losses.contrastive_loss import (
    TriModalContrastiveLoss,
    InfoNCELoss,
    TemporalContrastiveLoss
)

# Multi-task learning
from neuros_neurofm.losses.domain_adversarial import DomainAdversarialLoss
from neuros_neurofm.losses.multitask_loss import (
    UncertaintyWeightedLoss,
    GradNormLoss,
    MultiTaskLossManager
)

# Masked modeling
from neuros_neurofm.losses.masked_modeling import (
    MaskedModelingLoss,
    PerModalityMaskedLoss
)

# Forecasting
from neuros_neurofm.losses.forecasting import (
    MultiHorizonForecastingLoss,
    TemporalDistanceWeightedLoss,
    BehavioralForecastingLoss
)

# Diffusion
from neuros_neurofm.losses.diffusion import (
    DiffusionLoss,
    LatentDiffusionLoss,
    NeuralSegmentDiffusionLoss
)

# Loss registry
from neuros_neurofm.losses.loss_registry import (
    LossRegistry,
    LossConfig,
    LossScheduler
)

__all__ = [
    # Contrastive
    'TriModalContrastiveLoss',
    'InfoNCELoss',
    'TemporalContrastiveLoss',
    # Multi-task
    'DomainAdversarialLoss',
    'UncertaintyWeightedLoss',
    'GradNormLoss',
    'MultiTaskLossManager',
    # Masked modeling
    'MaskedModelingLoss',
    'PerModalityMaskedLoss',
    # Forecasting
    'MultiHorizonForecastingLoss',
    'TemporalDistanceWeightedLoss',
    'BehavioralForecastingLoss',
    # Diffusion
    'DiffusionLoss',
    'LatentDiffusionLoss',
    'NeuralSegmentDiffusionLoss',
    # Registry
    'LossRegistry',
    'LossConfig',
    'LossScheduler',
]
