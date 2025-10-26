"""
Training Curriculum Scheduler for NeuroFMX
Implements multi-stage training: unimodal → pairwise → full multimodal
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn


class CurriculumStage(Enum):
    """Training curriculum stages"""
    UNIMODAL = "unimodal"  # Train on single modalities
    PAIRWISE = "pairwise"  # Train on pairs of modalities
    MULTIMODAL = "multimodal"  # Train on all modalities


@dataclass
class StageConfig:
    """Configuration for a curriculum stage"""
    name: str
    stage: CurriculumStage
    duration_steps: int  # Number of training steps for this stage
    modalities: List[str]  # Allowed modalities
    modality_pairs: List[Tuple[str, str]] = field(default_factory=list)

    # Loss weights for this stage
    loss_weights: Dict[str, float] = field(default_factory=dict)

    # Fusion strategy
    fusion_enabled: bool = True
    fusion_type: str = "early"  # early, mid, late

    # Learning rate multiplier for this stage
    lr_multiplier: float = 1.0

    # Data augmentation strength
    augmentation_strength: float = 0.5


class CurriculumScheduler:
    """
    Manages multi-stage curriculum learning for NeuroFMX

    Training progresses through stages:
    1. Unimodal: Learn representations for each modality independently
    2. Pairwise: Learn cross-modal alignment for pairs of modalities
    3. Multimodal: Full multi-modal integration

    Example:
        >>> stages = [
        ...     StageConfig(
        ...         name="unimodal",
        ...         stage=CurriculumStage.UNIMODAL,
        ...         duration_steps=10000,
        ...         modalities=["eeg", "spikes", "lfp"],
        ...         loss_weights={"masked_modeling": 1.0, "reconstruction": 0.5}
        ...     ),
        ...     StageConfig(
        ...         name="pairwise",
        ...         stage=CurriculumStage.PAIRWISE,
        ...         duration_steps=10000,
        ...         modality_pairs=[("eeg", "video"), ("spikes", "lfp")],
        ...         loss_weights={"contrastive": 1.0, "masked_modeling": 0.5}
        ...     ),
        ... ]
        >>> scheduler = CurriculumScheduler(stages)
        >>> config = scheduler.get_config(global_step=5000)
    """

    def __init__(
        self,
        stages: List[StageConfig],
        transition_smoothing_steps: int = 500,
    ):
        """
        Args:
            stages: List of curriculum stages
            transition_smoothing_steps: Number of steps to smoothly transition between stages
        """
        self.stages = stages
        self.transition_smoothing_steps = transition_smoothing_steps

        # Compute cumulative step boundaries
        self.stage_boundaries = self._compute_boundaries()

        # Current stage info
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]

    def _compute_boundaries(self) -> List[int]:
        """Compute cumulative step boundaries for stages"""
        boundaries = []
        cumulative = 0
        for stage in self.stages:
            cumulative += stage.duration_steps
            boundaries.append(cumulative)
        return boundaries

    def get_current_stage(self, global_step: int) -> Tuple[StageConfig, float]:
        """
        Get current stage and progress through it

        Args:
            global_step: Current training step

        Returns:
            Tuple of (current_stage, progress) where progress ∈ [0, 1]
        """
        # Find which stage we're in
        for idx, boundary in enumerate(self.stage_boundaries):
            if global_step < boundary:
                stage = self.stages[idx]

                # Compute progress through this stage
                start_step = self.stage_boundaries[idx - 1] if idx > 0 else 0
                progress = (global_step - start_step) / stage.duration_steps

                self.current_stage_idx = idx
                self.current_stage = stage

                return stage, progress

        # If past all stages, return last stage
        self.current_stage_idx = len(self.stages) - 1
        self.current_stage = self.stages[-1]
        return self.stages[-1], 1.0

    def get_config(self, global_step: int) -> Dict[str, Any]:
        """
        Get complete configuration for current step

        Args:
            global_step: Current training step

        Returns:
            Dictionary with all curriculum settings
        """
        stage, progress = self.get_current_stage(global_step)

        # Check if we're in transition period
        in_transition, transition_weight = self._check_transition(global_step)

        if in_transition and self.current_stage_idx < len(self.stages) - 1:
            # Blend current and next stage
            next_stage = self.stages[self.current_stage_idx + 1]
            return self._blend_configs(stage, next_stage, transition_weight)

        return {
            "stage_name": stage.name,
            "stage_type": stage.stage.value,
            "modalities": stage.modalities,
            "modality_pairs": stage.modality_pairs,
            "loss_weights": stage.loss_weights,
            "fusion_enabled": stage.fusion_enabled,
            "fusion_type": stage.fusion_type,
            "lr_multiplier": stage.lr_multiplier,
            "augmentation_strength": stage.augmentation_strength,
            "progress": progress,
        }

    def _check_transition(self, global_step: int) -> Tuple[bool, float]:
        """
        Check if in transition period between stages

        Returns:
            Tuple of (in_transition, weight) where weight ∈ [0, 1]
            weight=0 means fully in current stage, weight=1 means fully in next stage
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False, 0.0

        # Get boundary to next stage
        boundary = self.stage_boundaries[self.current_stage_idx]

        # Check if within transition window
        steps_to_boundary = boundary - global_step

        if steps_to_boundary <= self.transition_smoothing_steps and steps_to_boundary >= 0:
            # In transition period before boundary
            weight = 1.0 - (steps_to_boundary / self.transition_smoothing_steps)
            return True, weight

        return False, 0.0

    def _blend_configs(
        self,
        current_stage: StageConfig,
        next_stage: StageConfig,
        weight: float
    ) -> Dict[str, Any]:
        """
        Blend configurations between two stages during transition

        Args:
            current_stage: Current stage config
            next_stage: Next stage config
            weight: Blending weight (0=current, 1=next)
        """
        # Blend loss weights
        blended_weights = {}
        all_losses = set(current_stage.loss_weights.keys()) | set(next_stage.loss_weights.keys())

        for loss_name in all_losses:
            current_w = current_stage.loss_weights.get(loss_name, 0.0)
            next_w = next_stage.loss_weights.get(loss_name, 0.0)
            blended_weights[loss_name] = (1 - weight) * current_w + weight * next_w

        # Blend other continuous values
        lr_mult = (1 - weight) * current_stage.lr_multiplier + weight * next_stage.lr_multiplier
        aug_strength = (1 - weight) * current_stage.augmentation_strength + weight * next_stage.augmentation_strength

        # For discrete values, use threshold
        fusion_enabled = current_stage.fusion_enabled if weight < 0.5 else next_stage.fusion_enabled
        fusion_type = current_stage.fusion_type if weight < 0.5 else next_stage.fusion_type

        # Combine modalities (union during transition)
        modalities = list(set(current_stage.modalities + next_stage.modalities))
        modality_pairs = current_stage.modality_pairs + next_stage.modality_pairs

        return {
            "stage_name": f"{current_stage.name}→{next_stage.name}",
            "stage_type": "transition",
            "modalities": modalities,
            "modality_pairs": modality_pairs,
            "loss_weights": blended_weights,
            "fusion_enabled": fusion_enabled,
            "fusion_type": fusion_type,
            "lr_multiplier": lr_mult,
            "augmentation_strength": aug_strength,
            "progress": weight,
        }

    def filter_batch(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int
    ) -> Dict[str, torch.Tensor]:
        """
        Filter batch to include only modalities allowed in current stage

        Args:
            batch: Dictionary mapping modality names to tensors
            global_step: Current training step

        Returns:
            Filtered batch dictionary
        """
        config = self.get_config(global_step)
        stage_type = config["stage_type"]

        if stage_type == "unimodal":
            # Sample one random modality
            modalities = config["modalities"]
            available = [m for m in modalities if m in batch]
            if not available:
                return batch  # Return original if none match

            import random
            selected = random.choice(available)
            return {selected: batch[selected]}

        elif stage_type == "pairwise":
            # Sample one random pair
            pairs = config["modality_pairs"]
            available_pairs = [(m1, m2) for m1, m2 in pairs if m1 in batch and m2 in batch]
            if not available_pairs:
                return batch

            import random
            m1, m2 = random.choice(available_pairs)
            return {m1: batch[m1], m2: batch[m2]}

        else:  # multimodal or transition
            # Include all allowed modalities
            modalities = config["modalities"]
            return {m: batch[m] for m in modalities if m in batch}

    def get_loss_weights(self, global_step: int) -> Dict[str, float]:
        """Get loss weights for current step"""
        config = self.get_config(global_step)
        return config["loss_weights"]

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing"""
        return {
            "current_stage_idx": self.current_stage_idx,
            "stage_boundaries": self.stage_boundaries,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint"""
        self.current_stage_idx = state_dict["current_stage_idx"]
        self.stage_boundaries = state_dict["stage_boundaries"]
        self.current_stage = self.stages[self.current_stage_idx]


def create_default_curriculum(
    total_steps: int = 100000,
    num_stages: int = 3,
) -> CurriculumScheduler:
    """
    Create default 3-stage curriculum

    Args:
        total_steps: Total training steps
        num_stages: Number of curriculum stages (2 or 3)

    Returns:
        Configured CurriculumScheduler
    """
    if num_stages == 2:
        # Unimodal → Multimodal
        steps_per_stage = total_steps // 2
        stages = [
            StageConfig(
                name="unimodal",
                stage=CurriculumStage.UNIMODAL,
                duration_steps=steps_per_stage,
                modalities=["eeg", "ecog", "lfp", "spikes", "fmri", "video", "audio"],
                loss_weights={
                    "masked_modeling": 1.0,
                    "reconstruction": 0.5,
                },
                fusion_enabled=False,
                lr_multiplier=1.0,
            ),
            StageConfig(
                name="multimodal",
                stage=CurriculumStage.MULTIMODAL,
                duration_steps=steps_per_stage,
                modalities=["eeg", "ecog", "lfp", "spikes", "fmri", "video", "audio"],
                loss_weights={
                    "masked_modeling": 0.3,
                    "contrastive": 0.8,
                    "forecasting": 0.5,
                    "diffusion": 0.2,
                },
                fusion_enabled=True,
                fusion_type="mid",
                lr_multiplier=0.5,
            ),
        ]
    else:
        # Unimodal → Pairwise → Multimodal
        steps_per_stage = total_steps // 3
        stages = [
            StageConfig(
                name="unimodal",
                stage=CurriculumStage.UNIMODAL,
                duration_steps=steps_per_stage,
                modalities=["eeg", "ecog", "lfp", "spikes", "fmri", "video", "audio"],
                loss_weights={
                    "masked_modeling": 1.0,
                    "reconstruction": 0.5,
                },
                fusion_enabled=False,
                lr_multiplier=1.0,
                augmentation_strength=0.3,
            ),
            StageConfig(
                name="pairwise",
                stage=CurriculumStage.PAIRWISE,
                duration_steps=steps_per_stage,
                modality_pairs=[
                    ("eeg", "video"),
                    ("spikes", "lfp"),
                    ("fmri", "video"),
                    ("ecog", "audio"),
                ],
                loss_weights={
                    "masked_modeling": 0.5,
                    "contrastive": 1.0,
                    "reconstruction": 0.3,
                },
                fusion_enabled=True,
                fusion_type="early",
                lr_multiplier=0.7,
                augmentation_strength=0.5,
            ),
            StageConfig(
                name="multimodal",
                stage=CurriculumStage.MULTIMODAL,
                duration_steps=steps_per_stage,
                modalities=["eeg", "ecog", "lfp", "spikes", "fmri", "video", "audio"],
                loss_weights={
                    "masked_modeling": 0.3,
                    "contrastive": 0.8,
                    "forecasting": 0.5,
                    "diffusion": 0.2,
                },
                fusion_enabled=True,
                fusion_type="mid",
                lr_multiplier=0.5,
                augmentation_strength=0.7,
            ),
        ]

    return CurriculumScheduler(stages, transition_smoothing_steps=1000)


# Example usage
if __name__ == "__main__":
    # Create curriculum
    scheduler = create_default_curriculum(total_steps=30000, num_stages=3)

    # Simulate training
    print("Curriculum Schedule:")
    print("=" * 80)

    for step in [0, 5000, 10000, 10500, 15000, 20000, 20500, 25000, 30000]:
        config = scheduler.get_config(step)
        print(f"\nStep {step:6d}:")
        print(f"  Stage: {config['stage_name']}")
        print(f"  Type: {config['stage_type']}")
        print(f"  Progress: {config['progress']:.2%}")
        print(f"  Modalities: {config['modalities'][:3]}...")  # Show first 3
        print(f"  Loss weights: {config['loss_weights']}")
        print(f"  LR multiplier: {config['lr_multiplier']:.2f}")
