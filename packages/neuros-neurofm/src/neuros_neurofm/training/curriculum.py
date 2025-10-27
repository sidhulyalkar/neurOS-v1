"""Training curriculum management for multi-phase training strategies.

This module defines curriculum scheduling for neurOS training loops,
supporting three-phase strategies commonly used in domain adaptation:
    1. Pretraining on all source domains
    2. Domain-weighted fine-tuning
    3. Target-specific fine-tuning
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Iterator, Tuple
from dataclasses import dataclass


class TrainingPhase(Enum):
    """Enumeration of training phases in curriculum learning."""

    PRETRAIN = auto()
    """Uniform pretraining across all source domains."""

    DOMAIN_WEIGHTED = auto()
    """Domain-weighted fine-tuning with adaptive mixture weights."""

    TARGET_FINE_TUNE = auto()
    """Target-specific fine-tuning on target domain only."""


@dataclass
class Curriculum:
    """Curriculum scheduler for three-phase training.

    Manages the progression through pretraining, domain-weighted adaptation,
    and target fine-tuning phases.

    Args:
        num_pretrain_epochs: Number of epochs for uniform pretraining.
        num_weighted_epochs: Number of epochs for domain-weighted phase.
        num_target_epochs: Number of epochs for target fine-tuning.

    Example:
        >>> curriculum = Curriculum(
        ...     num_pretrain_epochs=10,
        ...     num_weighted_epochs=20,
        ...     num_target_epochs=5,
        ... )
        >>> for phase, n_epochs in curriculum.phases():
        ...     print(f"{phase}: {n_epochs} epochs")
        TrainingPhase.PRETRAIN: 10 epochs
        TrainingPhase.DOMAIN_WEIGHTED: 20 epochs
        TrainingPhase.TARGET_FINE_TUNE: 5 epochs
    """

    num_pretrain_epochs: int
    num_weighted_epochs: int
    num_target_epochs: int

    def __post_init__(self) -> None:
        """Validate curriculum configuration."""
        if self.num_pretrain_epochs < 0:
            raise ValueError("num_pretrain_epochs must be non-negative")
        if self.num_weighted_epochs < 0:
            raise ValueError("num_weighted_epochs must be non-negative")
        if self.num_target_epochs < 0:
            raise ValueError("num_target_epochs must be non-negative")

    def phases(self) -> Iterator[Tuple[TrainingPhase, int]]:
        """Iterate through training phases with their epoch counts.

        Yields:
            Tuples of (phase, num_epochs) for each active phase.
            Phases with zero epochs are skipped.
        """
        if self.num_pretrain_epochs > 0:
            yield TrainingPhase.PRETRAIN, self.num_pretrain_epochs
        if self.num_weighted_epochs > 0:
            yield TrainingPhase.DOMAIN_WEIGHTED, self.num_weighted_epochs
        if self.num_target_epochs > 0:
            yield TrainingPhase.TARGET_FINE_TUNE, self.num_target_epochs

    @property
    def total_epochs(self) -> int:
        """Total number of epochs across all phases."""
        return (
            self.num_pretrain_epochs
            + self.num_weighted_epochs
            + self.num_target_epochs
        )

    def get_phase_at_epoch(self, global_epoch: int) -> TrainingPhase:
        """Determine which phase a given epoch belongs to.

        Args:
            global_epoch: Zero-indexed epoch number across all phases.

        Returns:
            The training phase for the given epoch.

        Raises:
            ValueError: If global_epoch is out of range.

        Example:
            >>> curriculum = Curriculum(10, 20, 5)
            >>> curriculum.get_phase_at_epoch(5)
            <TrainingPhase.PRETRAIN: 1>
            >>> curriculum.get_phase_at_epoch(15)
            <TrainingPhase.DOMAIN_WEIGHTED: 2>
            >>> curriculum.get_phase_at_epoch(32)
            <TrainingPhase.TARGET_FINE_TUNE: 3>
        """
        if global_epoch < 0 or global_epoch >= self.total_epochs:
            raise ValueError(
                f"Epoch {global_epoch} out of range [0, {self.total_epochs})"
            )

        if global_epoch < self.num_pretrain_epochs:
            return TrainingPhase.PRETRAIN
        elif global_epoch < self.num_pretrain_epochs + self.num_weighted_epochs:
            return TrainingPhase.DOMAIN_WEIGHTED
        else:
            return TrainingPhase.TARGET_FINE_TUNE


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning strategies.

    This class encapsulates all configuration parameters for curriculum-based
    training, including phase durations, learning rates, and adaptation
    strategies.

    Args:
        pretrain_epochs: Epochs for pretraining (default: 10)
        weighted_epochs: Epochs for weighted adaptation (default: 20)
        target_epochs: Epochs for target fine-tuning (default: 5)
        pretrain_lr: Learning rate for pretraining (default: 1e-3)
        weighted_lr: Learning rate for weighted phase (default: 5e-4)
        target_lr: Learning rate for target fine-tuning (default: 1e-4)
        weight_update_frequency: How often to recompute mixture weights (default: 1)
        warmup_epochs: Epochs of LR warmup in each phase (default: 2)

    Example:
        >>> config = CurriculumConfig(
        ...     pretrain_epochs=10,
        ...     weighted_epochs=20,
        ...     target_epochs=5,
        ...     pretrain_lr=1e-3,
        ... )
        >>> curriculum = Curriculum(
        ...     config.pretrain_epochs,
        ...     config.weighted_epochs,
        ...     config.target_epochs,
        ... )
    """

    # Phase durations
    pretrain_epochs: int = 10
    weighted_epochs: int = 20
    target_epochs: int = 5

    # Learning rates per phase
    pretrain_lr: float = 1e-3
    weighted_lr: float = 5e-4
    target_lr: float = 1e-4

    # Adaptation parameters
    weight_update_frequency: int = 1  # Recompute weights every N epochs
    warmup_epochs: int = 2

    # Optional regularization
    weight_entropy_penalty: float = 0.0  # Encourage uniform weights
    weight_smoothing: float = 0.0  # Temporal smoothing of weights

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.weight_update_frequency < 1:
            raise ValueError("weight_update_frequency must be >= 1")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.weight_entropy_penalty < 0:
            raise ValueError("weight_entropy_penalty must be non-negative")
        if not (0 <= self.weight_smoothing <= 1):
            raise ValueError("weight_smoothing must be in [0, 1]")

    def create_curriculum(self) -> Curriculum:
        """Create a Curriculum instance from this configuration."""
        return Curriculum(
            num_pretrain_epochs=self.pretrain_epochs,
            num_weighted_epochs=self.weighted_epochs,
            num_target_epochs=self.target_epochs,
        )

    def get_learning_rate(self, phase: TrainingPhase) -> float:
        """Get the learning rate for a given phase."""
        if phase == TrainingPhase.PRETRAIN:
            return self.pretrain_lr
        elif phase == TrainingPhase.DOMAIN_WEIGHTED:
            return self.weighted_lr
        elif phase == TrainingPhase.TARGET_FINE_TUNE:
            return self.target_lr
        else:
            raise ValueError(f"Unknown phase: {phase}")
