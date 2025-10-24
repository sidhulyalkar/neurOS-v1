"""
Meta-learning and few-shot learning modules.
"""

from .few_shot import (
    PrototypicalNetwork,
    MAML,
    TransferAdapter,
    FewShotDataset,
    evaluate_few_shot,
    MetaLearningTrainer
)

__all__ = [
    'PrototypicalNetwork',
    'MAML',
    'TransferAdapter',
    'FewShotDataset',
    'evaluate_few_shot',
    'MetaLearningTrainer'
]
