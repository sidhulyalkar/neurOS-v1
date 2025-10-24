"""
Loss Functions for NeuroFMx Multimodal Training

Includes:
- Tri-modal contrastive loss (neural + behavior + stimulus)
- Domain adversarial loss
- Uncertainty-weighted multi-task loss
- Standard reconstruction and prediction losses
"""

from neuros_neurofm.losses.contrastive_loss import TriModalContrastiveLoss, InfoNCELoss
from neuros_neurofm.losses.domain_adversarial import DomainAdversarialLoss
from neuros_neurofm.losses.multitask_loss import UncertaintyWeightedLoss, MultiTaskLossManager

__all__ = [
    'TriModalContrastiveLoss',
    'InfoNCELoss',
    'DomainAdversarialLoss',
    'UncertaintyWeightedLoss',
    'MultiTaskLossManager'
]
