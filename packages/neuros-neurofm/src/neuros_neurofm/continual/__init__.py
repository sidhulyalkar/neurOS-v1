"""
Continual learning modules.
"""

from .continual_learning import (
    ExperienceReplayBuffer,
    ContinualLearner,
    continual_training_loop
)

__all__ = [
    'ExperienceReplayBuffer',
    'ContinualLearner',
    'continual_training_loop'
]
