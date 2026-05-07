"""
Synthetic datasets for ENGRAM-FMx training and evaluation.

These datasets test specific capabilities:
- AssociativeRecallDataset: Memory retrieval
- DelayedCopyDataset: Long-horizon sequence recall
- NeuralDynamicsDataset: Latent dynamical system modeling
"""

from neuros_neurofm.data.synthetic.associative_recall import AssociativeRecallDataset
from neuros_neurofm.data.synthetic.delayed_copy import DelayedCopyDataset
from neuros_neurofm.data.synthetic.neural_dynamics import NeuralDynamicsDataset

__all__ = [
    "AssociativeRecallDataset",
    "DelayedCopyDataset",
    "NeuralDynamicsDataset",
]
