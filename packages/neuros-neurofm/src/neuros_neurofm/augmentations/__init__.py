"""
Neural Data Augmentation Module

Provides mathematically grounded augmentation techniques for:
- Calcium imaging traces
- Astrocyte event data
- Multimodal neural recordings
"""

from neuros_neurofm.augmentations.neural_augmentations import (
    NeuralAugmentor,
    AugmentationConfig,
    MixupAugmentor,
    WindowAugmentor,
    create_augmentor,
)

__all__ = [
    'NeuralAugmentor',
    'AugmentationConfig',
    'MixupAugmentor',
    'WindowAugmentor',
    'create_augmentor',
]
