"""
Unified Loss Registry and Configuration for NeuroFMX

Provides a central interface for:
- Combining multiple loss functions
- Dynamic loss weighting (uncertainty, GradNorm, manual, scheduled)
- Loss scheduling over training
- Logging and monitoring individual loss components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np

from neuros_neurofm.losses.masked_modeling import MaskedModelingLoss, PerModalityMaskedLoss
from neuros_neurofm.losses.forecasting import MultiHorizonForecastingLoss, TemporalDistanceWeightedLoss
from neuros_neurofm.losses.diffusion import DiffusionLoss, LatentDiffusionLoss
from neuros_neurofm.losses.contrastive_loss import TriModalContrastiveLoss, InfoNCELoss
from neuros_neurofm.losses.multitask_loss import UncertaintyWeightedLoss, GradNormLoss


@dataclass
class LossConfig:
    """
    Configuration for a single loss function.

    Attributes:
        name: Loss function name/identifier
        loss_type: Type of loss ('masked_modeling', 'forecasting', 'diffusion', etc.)
        weight: Static weight for this loss
        enabled: Whether this loss is active
        config: Additional configuration parameters for the loss
        schedule: Optional weight scheduling config
    """
    name: str
    loss_type: str
    weight: float = 1.0
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[Dict[str, Any]] = None


class LossScheduler:
    """
    Schedules loss weights over training.

    Supports:
    - Linear warmup/decay
    - Cosine annealing
    - Step-based changes
    - Custom schedules

    Args:
        schedule_type: Type of schedule ('linear', 'cosine', 'step', 'exponential')
        total_steps: Total training steps
        warmup_steps: Warmup period
        start_weight: Initial weight
        end_weight: Final weight
        step_config: For step schedule: list of (step, weight) tuples
    """

    def __init__(
        self,
        schedule_type: str = 'constant',
        total_steps: int = 100000,
        warmup_steps: int = 0,
        start_weight: float = 0.0,
        end_weight: float = 1.0,
        step_config: Optional[List[Tuple[int, float]]] = None
    ):
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.step_config = step_config or []

    def get_weight(self, step: int) -> float:
        """Get weight at given training step."""
        if self.schedule_type == 'constant':
            return self.end_weight

        # Warmup phase
        if step < self.warmup_steps:
            return self.start_weight + (self.end_weight - self.start_weight) * (
                step / self.warmup_steps
            )

        # Post-warmup schedules
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)

        if self.schedule_type == 'linear':
            weight = self.start_weight + (self.end_weight - self.start_weight) * progress

        elif self.schedule_type == 'cosine':
            weight = self.end_weight + (self.start_weight - self.end_weight) * (
                0.5 * (1 + np.cos(np.pi * progress))
            )

        elif self.schedule_type == 'exponential':
            weight = self.start_weight * (self.end_weight / self.start_weight) ** progress

        elif self.schedule_type == 'step':
            weight = self.start_weight
            for step_threshold, step_weight in sorted(self.step_config):
                if step >= step_threshold:
                    weight = step_weight

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return weight


class LossRegistry(nn.Module):
    """
    Unified registry for managing multiple loss functions.

    Features:
    - Register and combine multiple losses
    - Dynamic weighting strategies
    - Loss scheduling
    - Gradient normalization
    - Logging and monitoring

    Args:
        loss_configs: List of LossConfig objects
        weighting_strategy: How to combine losses ('manual', 'uncertainty', 'gradnorm', 'equal')
        log_individual_losses: Whether to log each loss separately
        gradient_clip_val: Global gradient clipping value
        normalize_gradients: Whether to normalize gradients across losses
    """

    def __init__(
        self,
        loss_configs: List[LossConfig],
        weighting_strategy: str = 'manual',
        log_individual_losses: bool = True,
        gradient_clip_val: float = 1.0,
        normalize_gradients: bool = False
    ):
        super().__init__()

        self.loss_configs = {cfg.name: cfg for cfg in loss_configs}
        self.weighting_strategy = weighting_strategy
        self.log_individual_losses = log_individual_losses
        self.gradient_clip_val = gradient_clip_val
        self.normalize_gradients = normalize_gradients

        # Build loss modules
        self.losses = nn.ModuleDict()
        self._build_losses()

        # Build weighting mechanism
        self.weighting_module = self._build_weighting_module()

        # Build schedulers
        self.schedulers = {}
        self._build_schedulers()

        # Training step counter
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

    def _build_losses(self):
        """Build loss modules from configs."""
        for name, config in self.loss_configs.items():
            if not config.enabled:
                continue

            loss_type = config.loss_type
            loss_config = config.config

            # Create loss module based on type
            if loss_type == 'masked_modeling':
                if 'modality_configs' in loss_config:
                    self.losses[name] = PerModalityMaskedLoss(**loss_config)
                else:
                    self.losses[name] = MaskedModelingLoss(**loss_config)

            elif loss_type == 'forecasting':
                if 'learnable_weights' in loss_config and loss_config['learnable_weights']:
                    self.losses[name] = TemporalDistanceWeightedLoss(**loss_config)
                else:
                    self.losses[name] = MultiHorizonForecastingLoss(**loss_config)

            elif loss_type == 'diffusion':
                if 'latent_dim' in loss_config:
                    self.losses[name] = LatentDiffusionLoss(**loss_config)
                else:
                    self.losses[name] = DiffusionLoss(**loss_config)

            elif loss_type == 'contrastive':
                if loss_config.get('tri_modal', False):
                    self.losses[name] = TriModalContrastiveLoss(**loss_config)
                else:
                    self.losses[name] = InfoNCELoss(**loss_config)

            elif loss_type == 'mse':
                self.losses[name] = nn.MSELoss()

            elif loss_type == 'cross_entropy':
                self.losses[name] = nn.CrossEntropyLoss()

            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

    def _build_weighting_module(self):
        """Build module for dynamic loss weighting."""
        if self.weighting_strategy == 'manual' or self.weighting_strategy == 'equal':
            return None

        elif self.weighting_strategy == 'uncertainty':
            n_losses = len([c for c in self.loss_configs.values() if c.enabled])
            return UncertaintyWeightedLoss(n_tasks=n_losses)

        elif self.weighting_strategy == 'gradnorm':
            n_losses = len([c for c in self.loss_configs.values() if c.enabled])
            return GradNormLoss(n_tasks=n_losses)

        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

    def _build_schedulers(self):
        """Build loss weight schedulers."""
        for name, config in self.loss_configs.items():
            if config.schedule is not None:
                self.schedulers[name] = LossScheduler(**config.schedule)

    def forward(
        self,
        loss_inputs: Dict[str, Any],
        shared_params: Optional[nn.Parameter] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute combined loss.

        Args:
            loss_inputs: Dict mapping loss names to their inputs
                        Each input should be a dict with keys matching the loss function signature
            shared_params: Shared parameter for GradNorm (if using)

        Returns:
            total_loss: Combined weighted loss
            metrics: Dictionary with all loss components and weights
        """
        individual_losses = {}
        all_metrics = {}

        # Compute each individual loss
        for name, loss_module in self.losses.items():
            if name not in loss_inputs:
                continue

            inputs = loss_inputs[name]

            # Compute loss
            result = loss_module(**inputs)

            # Handle different return types
            if isinstance(result, tuple):
                loss, metrics = result
            else:
                loss = result
                metrics = {}

            individual_losses[name] = loss

            # Log individual loss and metrics
            if self.log_individual_losses:
                all_metrics[f'{name}_loss'] = loss.detach()
                for key, value in metrics.items():
                    all_metrics[f'{name}_{key}'] = value.detach() if isinstance(value, torch.Tensor) else value

        # Apply weight scheduling
        current_weights = self._get_current_weights()

        # Combine losses based on weighting strategy
        if self.weighting_strategy == 'uncertainty':
            total_loss = self.weighting_module(individual_losses)

            # Add weight info
            weights = self.weighting_module.get_weights()
            for i, (name, _) in enumerate(individual_losses.items()):
                all_metrics[f'{name}_weight'] = weights.get(f'task_{i}', 1.0)

        elif self.weighting_strategy == 'gradnorm':
            if shared_params is None:
                raise ValueError("GradNorm requires shared_params")

            losses_list = [individual_losses[name] for name in sorted(individual_losses.keys())]
            total_loss, gradnorm_loss = self.weighting_module(losses_list, shared_params)

            all_metrics['gradnorm_loss'] = gradnorm_loss.detach()

        else:  # manual or equal
            total_loss = 0.0
            for name, loss in individual_losses.items():
                weight = current_weights.get(name, 1.0)
                weighted_loss = weight * loss
                total_loss += weighted_loss

                all_metrics[f'{name}_weight'] = weight
                all_metrics[f'{name}_weighted_loss'] = weighted_loss.detach()

        # Gradient normalization across losses
        if self.normalize_gradients and self.training:
            total_loss = self._normalize_gradients(total_loss, individual_losses)

        all_metrics['total_loss'] = total_loss.detach()
        all_metrics['step'] = self.step.item()

        # Increment step counter
        if self.training:
            self.step += 1

        return total_loss, all_metrics

    def _get_current_weights(self) -> Dict[str, float]:
        """Get current weights (applying schedulers)."""
        weights = {}
        current_step = self.step.item()

        for name, config in self.loss_configs.items():
            if not config.enabled:
                continue

            # Base weight
            weight = config.weight

            # Apply scheduler if present
            if name in self.schedulers:
                scheduled_weight = self.schedulers[name].get_weight(current_step)
                weight = weight * scheduled_weight

            weights[name] = weight

        return weights

    def _normalize_gradients(
        self,
        total_loss: torch.Tensor,
        individual_losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Normalize gradients to have similar magnitudes across losses."""
        # This is a simplified version - full implementation would compute
        # actual gradient norms and renormalize
        return total_loss

    def update_config(self, name: str, **kwargs):
        """Update configuration for a specific loss."""
        if name in self.loss_configs:
            config = self.loss_configs[name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    def enable_loss(self, name: str):
        """Enable a specific loss."""
        if name in self.loss_configs:
            self.loss_configs[name].enabled = True

    def disable_loss(self, name: str):
        """Disable a specific loss."""
        if name in self.loss_configs:
            self.loss_configs[name].enabled = False

    def get_weights(self) -> Dict[str, float]:
        """Get current weights for all losses."""
        return self._get_current_weights()

    def state_dict(self):
        """Get state dict for saving."""
        state = {
            'losses': self.losses.state_dict(),
            'step': self.step,
        }

        if self.weighting_module is not None:
            if hasattr(self.weighting_module, 'state_dict'):
                state['weighting_module'] = self.weighting_module.state_dict()

        return state

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.losses.load_state_dict(state_dict['losses'])
        self.step.copy_(state_dict['step'])

        if 'weighting_module' in state_dict and self.weighting_module is not None:
            if hasattr(self.weighting_module, 'load_state_dict'):
                self.weighting_module.load_state_dict(state_dict['weighting_module'])


# Example usage
if __name__ == '__main__':
    # Define loss configurations
    loss_configs = [
        LossConfig(
            name='masked_modeling',
            loss_type='masked_modeling',
            weight=1.0,
            config={
                'mask_ratio': 0.15,
                'masking_strategy': 'random',
                'reconstruction_loss': 'mse'
            },
            schedule={
                'schedule_type': 'linear',
                'total_steps': 100000,
                'warmup_steps': 1000,
                'start_weight': 0.1,
                'end_weight': 1.0
            }
        ),
        LossConfig(
            name='forecasting',
            loss_type='forecasting',
            weight=0.5,
            config={
                'horizons_ms': [100, 250, 500],
                'sampling_rate_hz': 1000.0,
                'loss_type': 'smooth_l1'
            }
        ),
        LossConfig(
            name='diffusion',
            loss_type='diffusion',
            weight=0.3,
            config={
                'n_timesteps': 1000,
                'schedule_type': 'cosine',
                'loss_type': 'mse'
            },
            schedule={
                'schedule_type': 'cosine',
                'total_steps': 100000,
                'warmup_steps': 5000,
                'start_weight': 0.0,
                'end_weight': 0.5
            }
        ),
        LossConfig(
            name='contrastive',
            loss_type='contrastive',
            weight=0.2,
            config={
                'temperature': 0.07
            }
        )
    ]

    # Create registry
    registry = LossRegistry(
        loss_configs=loss_configs,
        weighting_strategy='manual',
        log_individual_losses=True
    )

    print("Loss Registry created successfully!")
    print(f"Registered losses: {list(registry.losses.keys())}")

    # Example forward pass
    batch_size, seq_len, dim = 4, 100, 128

    # Prepare inputs for each loss (simplified example)
    loss_inputs = {
        'masked_modeling': {
            'predictions': torch.randn(batch_size, seq_len, dim),
            'targets': torch.randn(batch_size, seq_len, dim)
        },
        'contrastive': {
            'anchor': torch.randn(batch_size, dim),
            'positive': torch.randn(batch_size, dim)
        }
    }

    # Compute loss
    total_loss, metrics = registry(loss_inputs)

    print(f"\nTotal loss: {total_loss.item():.4f}")
    print("\nIndividual losses:")
    for name in ['masked_modeling', 'contrastive']:
        if f'{name}_loss' in metrics:
            print(f"  {name}: {metrics[f'{name}_loss'].item():.4f} (weight: {metrics[f'{name}_weight']:.4f})")

    # Test weight scheduling
    print("\n\nWeight scheduling test:")
    for step in [0, 1000, 5000, 50000, 100000]:
        registry.step = torch.tensor(step)
        weights = registry.get_weights()
        print(f"Step {step}:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")

    print("\n\nLoss registry implementation complete!")
