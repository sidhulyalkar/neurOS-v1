"""
Multi-Horizon Forecasting Loss for NeuroFMX

Implements predictive coding losses for neural and behavioral forecasting:
- Multi-horizon prediction (100ms to 1000ms ahead)
- Temporal distance weighting
- Support for both continuous and discrete predictions
- Gradient stabilization for long-horizon predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Literal
import numpy as np


class MultiHorizonForecastingLoss(nn.Module):
    """
    Multi-horizon predictive coding loss.

    Predicts neural/behavioral activity at multiple future timepoints,
    with configurable weighting based on prediction horizon.

    Useful for:
    - Training models to predict future neural activity
    - Learning temporal dynamics and causality
    - Enabling anticipatory decoding

    Args:
        horizons_ms: List of prediction horizons in milliseconds
        sampling_rate_hz: Sampling rate of data (for converting ms to timesteps)
        horizon_weights: Weights for each horizon (None = equal, 'inverse' = 1/t, 'exponential' = exp(-t))
        loss_type: Loss function type ('mse', 'smooth_l1', 'huber')
        reduction: How to reduce across horizons ('mean', 'sum', 'weighted')
        gradient_clip_val: Max gradient norm per horizon (for stability)
        normalize_predictions: Whether to normalize predictions
    """

    def __init__(
        self,
        horizons_ms: List[float] = [100, 250, 500, 1000],
        sampling_rate_hz: float = 1000.0,
        horizon_weights: Optional[List[float]] = None,
        loss_type: Literal['mse', 'smooth_l1', 'huber'] = 'smooth_l1',
        reduction: Literal['mean', 'sum', 'weighted'] = 'weighted',
        gradient_clip_val: float = 1.0,
        normalize_predictions: bool = False
    ):
        super().__init__()

        self.horizons_ms = horizons_ms
        self.sampling_rate_hz = sampling_rate_hz
        self.loss_type = loss_type
        self.reduction = reduction
        self.gradient_clip_val = gradient_clip_val
        self.normalize_predictions = normalize_predictions

        # Convert horizons from ms to timesteps
        self.horizons_steps = [
            int(h_ms * sampling_rate_hz / 1000.0) for h_ms in horizons_ms
        ]

        # Set up horizon weights
        if horizon_weights is None:
            # Default: inverse temporal distance weighting
            horizon_weights = [1.0 / (h_ms / 100.0) for h_ms in horizons_ms]
        elif horizon_weights == 'inverse':
            horizon_weights = [1.0 / max(1, h_ms / 100.0) for h_ms in horizons_ms]
        elif horizon_weights == 'exponential':
            horizon_weights = [np.exp(-h_ms / 500.0) for h_ms in horizons_ms]
        elif horizon_weights == 'equal':
            horizon_weights = [1.0] * len(horizons_ms)

        # Normalize weights to sum to 1
        total = sum(horizon_weights)
        self.horizon_weights = torch.tensor([w / total for w in horizon_weights])

    def forward(
        self,
        predictions: Dict[int, torch.Tensor],
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-horizon forecasting loss.

        Args:
            predictions: Dict mapping horizon (in steps) to predictions
                        Each tensor has shape (batch, seq_len, dim)
            targets: Ground truth sequence, shape (batch, seq_len, dim)
            attention_mask: Valid position mask, shape (batch, seq_len)

        Returns:
            loss: Combined loss across horizons
            metrics: Dictionary with per-horizon losses and statistics
        """
        device = targets.device
        batch_size, seq_len, dim = targets.shape

        horizon_losses = []
        metrics = {}

        # Compute loss for each horizon
        for i, horizon_steps in enumerate(self.horizons_steps):
            horizon_ms = self.horizons_ms[i]

            # Get predictions for this horizon
            if horizon_steps not in predictions:
                raise ValueError(f"Predictions missing for horizon {horizon_steps} steps ({horizon_ms}ms)")

            pred = predictions[horizon_steps]  # (batch, seq_len, dim)

            # Compute loss for this horizon
            loss, horizon_metrics = self._compute_horizon_loss(
                pred, targets, horizon_steps, attention_mask
            )

            # Apply gradient clipping if needed
            if self.gradient_clip_val > 0 and self.training:
                loss = loss * torch.clamp(
                    torch.ones_like(loss),
                    max=self.gradient_clip_val
                )

            horizon_losses.append(loss)

            # Store metrics
            metrics[f'horizon_{horizon_ms}ms_loss'] = loss
            for key, value in horizon_metrics.items():
                metrics[f'horizon_{horizon_ms}ms_{key}'] = value

        # Combine horizon losses
        horizon_losses = torch.stack(horizon_losses)  # (n_horizons,)
        weights = self.horizon_weights.to(device)

        if self.reduction == 'mean':
            total_loss = horizon_losses.mean()
        elif self.reduction == 'sum':
            total_loss = horizon_losses.sum()
        elif self.reduction == 'weighted':
            total_loss = (horizon_losses * weights).sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        metrics['total_loss'] = total_loss
        metrics['per_horizon_losses'] = horizon_losses

        return total_loss, metrics

    def _compute_horizon_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        horizon_steps: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss for a single prediction horizon.

        Args:
            predictions: Predictions, shape (batch, seq_len, dim)
            targets: Targets, shape (batch, seq_len, dim)
            horizon_steps: Number of steps ahead being predicted
            attention_mask: Valid positions, shape (batch, seq_len)

        Returns:
            loss: Loss for this horizon
            metrics: Additional metrics
        """
        batch_size, seq_len, dim = targets.shape

        # Shift targets by horizon to get future targets
        # predictions[:, t] should predict targets[:, t + horizon_steps]
        if horizon_steps >= seq_len:
            # Horizon too large for sequence
            return torch.tensor(0.0, device=targets.device), {}

        # Extract valid prediction-target pairs
        pred_valid = predictions[:, :-horizon_steps]  # (batch, seq_len - horizon, dim)
        target_future = targets[:, horizon_steps:]  # (batch, seq_len - horizon, dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Both current and future positions must be valid
            mask_current = attention_mask[:, :-horizon_steps]
            mask_future = attention_mask[:, horizon_steps:]
            valid_mask = mask_current & mask_future  # (batch, seq_len - horizon)

            # Expand mask for broadcasting
            valid_mask = valid_mask.unsqueeze(-1)  # (batch, seq_len - horizon, 1)

            # Mask out invalid positions
            pred_valid = pred_valid * valid_mask
            target_future = target_future * valid_mask

            num_valid = valid_mask.sum()
        else:
            num_valid = pred_valid.numel()

        # Normalize if requested
        if self.normalize_predictions:
            pred_valid = F.layer_norm(pred_valid, (dim,))
            target_future = F.layer_norm(target_future, (dim,))

        # Compute loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_valid, target_future, reduction='sum')
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_valid, target_future, reduction='sum')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(pred_valid, target_future, reduction='sum')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Normalize by number of valid elements
        if num_valid > 0:
            loss = loss / num_valid
        else:
            loss = torch.tensor(0.0, device=targets.device)

        # Compute metrics
        metrics = {}
        with torch.no_grad():
            mae = (pred_valid - target_future).abs().mean()
            metrics['mae'] = mae

            # Correlation (for monitoring)
            if num_valid > 0:
                pred_flat = pred_valid.reshape(-1)
                target_flat = target_future.reshape(-1)
                correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                metrics['correlation'] = correlation

        return loss, metrics


class TemporalDistanceWeightedLoss(nn.Module):
    """
    Forecasting loss with learnable temporal distance weighting.

    Learns optimal weights for different prediction horizons during training.

    Args:
        horizons_ms: List of prediction horizons
        sampling_rate_hz: Data sampling rate
        loss_type: Base loss function type
        learnable_weights: Whether weights are learnable
        init_weights: Initial weight values
    """

    def __init__(
        self,
        horizons_ms: List[float] = [100, 250, 500, 1000],
        sampling_rate_hz: float = 1000.0,
        loss_type: Literal['mse', 'smooth_l1', 'huber'] = 'smooth_l1',
        learnable_weights: bool = True,
        init_weights: Optional[List[float]] = None
    ):
        super().__init__()

        self.horizons_ms = horizons_ms
        self.n_horizons = len(horizons_ms)

        # Base forecasting loss
        self.base_loss = MultiHorizonForecastingLoss(
            horizons_ms=horizons_ms,
            sampling_rate_hz=sampling_rate_hz,
            loss_type=loss_type,
            reduction='sum',  # We'll weight manually
        )

        # Learnable weights (in log space for stability)
        if init_weights is None:
            # Initialize with inverse temporal distance
            init_weights = [1.0 / (h / 100.0) for h in horizons_ms]

        if learnable_weights:
            self.log_weights = nn.Parameter(
                torch.log(torch.tensor(init_weights, dtype=torch.float32))
            )
        else:
            self.register_buffer(
                'log_weights',
                torch.log(torch.tensor(init_weights, dtype=torch.float32))
            )

    def forward(
        self,
        predictions: Dict[int, torch.Tensor],
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute temporally-weighted forecasting loss.

        Args:
            predictions: Dict of predictions per horizon
            targets: Ground truth targets
            attention_mask: Valid position mask

        Returns:
            loss: Weighted loss
            metrics: Loss metrics and learned weights
        """
        # Get base loss with all horizons
        base_loss, metrics = self.base_loss(predictions, targets, attention_mask)

        # Extract per-horizon losses
        horizon_losses = []
        for horizon_ms in self.horizons_ms:
            horizon_losses.append(metrics[f'horizon_{horizon_ms}ms_loss'])
        horizon_losses = torch.stack(horizon_losses)

        # Compute normalized weights (softmax in log space)
        weights = F.softmax(self.log_weights, dim=0)

        # Weighted combination
        total_loss = (horizon_losses * weights).sum()

        # Add weight information to metrics
        for i, horizon_ms in enumerate(self.horizons_ms):
            metrics[f'horizon_{horizon_ms}ms_weight'] = weights[i]

        metrics['total_loss'] = total_loss

        return total_loss, metrics

    def get_weights(self) -> Dict[str, float]:
        """Get current learned weights."""
        weights = F.softmax(self.log_weights, dim=0)
        return {
            f"{self.horizons_ms[i]}ms": weights[i].item()
            for i in range(self.n_horizons)
        }


class BehavioralForecastingLoss(nn.Module):
    """
    Specialized forecasting loss for behavioral predictions.

    Handles both continuous (velocity, position) and discrete (choice) behavioral variables.

    Args:
        horizons_ms: Prediction horizons
        sampling_rate_hz: Sampling rate
        continuous_vars: List of continuous variable names
        discrete_vars: List of discrete variable names
        horizon_weights: Weighting scheme
    """

    def __init__(
        self,
        horizons_ms: List[float] = [100, 250, 500, 1000],
        sampling_rate_hz: float = 1000.0,
        continuous_vars: Optional[List[str]] = None,
        discrete_vars: Optional[List[str]] = None,
        horizon_weights: Optional[List[float]] = None
    ):
        super().__init__()

        self.continuous_vars = continuous_vars or []
        self.discrete_vars = discrete_vars or []

        # Continuous variable loss (MSE/Smooth L1)
        self.continuous_loss = MultiHorizonForecastingLoss(
            horizons_ms=horizons_ms,
            sampling_rate_hz=sampling_rate_hz,
            loss_type='smooth_l1',
            horizon_weights=horizon_weights
        )

        # Discrete variable loss (Cross-entropy)
        # Note: Will need special handling in forward pass

    def forward(
        self,
        predictions: Dict[str, Dict[int, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute behavioral forecasting loss.

        Args:
            predictions: Dict of {variable_name: {horizon: predictions}}
            targets: Dict of {variable_name: targets}
            attention_mask: Valid positions

        Returns:
            loss: Combined loss
            metrics: Per-variable and per-horizon metrics
        """
        total_loss = 0.0
        all_metrics = {}

        # Continuous variables
        for var in self.continuous_vars:
            if var not in predictions or var not in targets:
                continue

            var_pred = predictions[var]
            var_target = targets[var]

            loss, metrics = self.continuous_loss(var_pred, var_target, attention_mask)
            total_loss += loss

            # Store metrics
            for key, value in metrics.items():
                all_metrics[f'{var}_{key}'] = value

        # Discrete variables (e.g., choice, action)
        for var in self.discrete_vars:
            if var not in predictions or var not in targets:
                continue

            # For discrete variables, use cross-entropy
            var_pred = predictions[var]
            var_target = targets[var]  # Should be class indices

            discrete_loss = self._compute_discrete_forecast_loss(
                var_pred, var_target, attention_mask
            )
            total_loss += discrete_loss

            all_metrics[f'{var}_loss'] = discrete_loss

        all_metrics['total_loss'] = total_loss

        return total_loss, all_metrics

    def _compute_discrete_forecast_loss(
        self,
        predictions: Dict[int, torch.Tensor],
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross-entropy loss for discrete forecasting."""
        # Simplified implementation - combine all horizons
        total_loss = 0.0

        for horizon_steps, pred in predictions.items():
            if horizon_steps >= targets.shape[1]:
                continue

            # Shift targets
            pred_valid = pred[:, :-horizon_steps]
            target_future = targets[:, horizon_steps:]

            # Cross-entropy
            loss = F.cross_entropy(
                pred_valid.reshape(-1, pred_valid.shape[-1]),
                target_future.long().reshape(-1),
                reduction='mean'
            )

            total_loss += loss

        return total_loss / len(predictions)


# Example usage
if __name__ == '__main__':
    batch_size, seq_len, dim = 4, 200, 128
    horizons_ms = [100, 250, 500, 1000]
    sampling_rate = 1000.0  # 1kHz

    # Convert horizons to steps
    horizons_steps = [int(h * sampling_rate / 1000) for h in horizons_ms]

    # Create predictions for each horizon
    predictions = {
        h: torch.randn(batch_size, seq_len, dim) for h in horizons_steps
    }

    targets = torch.randn(batch_size, seq_len, dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Test basic multi-horizon loss
    loss_fn = MultiHorizonForecastingLoss(
        horizons_ms=horizons_ms,
        sampling_rate_hz=sampling_rate
    )

    loss, metrics = loss_fn(predictions, targets, attention_mask)
    print(f"Multi-horizon forecasting loss: {loss.item():.4f}")
    for horizon_ms in horizons_ms:
        print(f"  {horizon_ms}ms loss: {metrics[f'horizon_{horizon_ms}ms_loss'].item():.4f}")

    # Test learnable temporal weighting
    learnable_loss = TemporalDistanceWeightedLoss(
        horizons_ms=horizons_ms,
        sampling_rate_hz=sampling_rate,
        learnable_weights=True
    )

    loss, metrics = learnable_loss(predictions, targets, attention_mask)
    print(f"\nLearnable weighted loss: {loss.item():.4f}")
    weights = learnable_loss.get_weights()
    print("Learned weights:")
    for horizon, weight in weights.items():
        print(f"  {horizon}: {weight:.4f}")

    # Test behavioral forecasting
    continuous_vars = ['velocity', 'position']
    behavioral_predictions = {
        'velocity': {h: torch.randn(batch_size, seq_len, 2) for h in horizons_steps},
        'position': {h: torch.randn(batch_size, seq_len, 2) for h in horizons_steps}
    }
    behavioral_targets = {
        'velocity': torch.randn(batch_size, seq_len, 2),
        'position': torch.randn(batch_size, seq_len, 2)
    }

    behavioral_loss = BehavioralForecastingLoss(
        horizons_ms=horizons_ms,
        sampling_rate_hz=sampling_rate,
        continuous_vars=continuous_vars
    )

    loss, metrics = behavioral_loss(behavioral_predictions, behavioral_targets, attention_mask)
    print(f"\nBehavioral forecasting loss: {loss.item():.4f}")
    for var in continuous_vars:
        print(f"  {var} loss: {metrics[f'{var}_total_loss'].item():.4f}")
