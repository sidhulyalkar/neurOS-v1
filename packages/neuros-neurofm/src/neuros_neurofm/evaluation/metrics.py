"""
Evaluation metrics for NeuroFM-X.

Provides comprehensive metrics for neural decoding, encoding, and generation.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Coefficient of determination (R²).

    Parameters
    ----------
    y_true : torch.Tensor
        True values, shape (n_samples, n_features).
    y_pred : torch.Tensor
        Predicted values, same shape.

    Returns
    -------
    torch.Tensor
        R² score (scalar or per-feature).
    """
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2.mean()  # Average over features


def pearson_correlation(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Pearson correlation coefficient.

    Parameters
    ----------
    y_true : torch.Tensor
        True values, shape (n_samples, n_features).
    y_pred : torch.Tensor
        Predicted values, same shape.

    Returns
    -------
    torch.Tensor
        Correlation (averaged over features).
    """
    y_true_centered = y_true - y_true.mean(dim=0, keepdim=True)
    y_pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)

    numerator = (y_true_centered * y_pred_centered).sum(dim=0)
    denominator = torch.sqrt(
        (y_true_centered ** 2).sum(dim=0) * (y_pred_centered ** 2).sum(dim=0)
    ) + 1e-8

    corr = numerator / denominator
    return corr.mean()


def bits_per_spike(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    dt: float = 0.01,
) -> torch.Tensor:
    """Bits per spike (BPS) for neural encoding evaluation.

    Measures information content in spike predictions using Poisson likelihood.

    Parameters
    ----------
    y_true : torch.Tensor
        True spike counts, shape (n_samples, n_neurons).
    y_pred : torch.Tensor
        Predicted firing rates (Hz), same shape.
    dt : float, optional
        Time bin size in seconds.
        Default: 0.01 (10ms).

    Returns
    -------
    torch.Tensor
        Bits per spike (higher is better).
    """
    # Convert rates to expected spike counts
    lambda_pred = y_pred * dt
    lambda_pred = torch.clamp(lambda_pred, min=1e-8)

    # Poisson log-likelihood
    log_likelihood = y_true * torch.log(lambda_pred) - lambda_pred

    # Mean firing rate for normalization
    mean_rate = y_true.mean(dim=0) / dt
    mean_rate = torch.clamp(mean_rate, min=1e-8)
    lambda_mean = mean_rate * dt

    # Log-likelihood under mean rate (baseline)
    log_likelihood_mean = y_true * torch.log(lambda_mean) - lambda_mean

    # BPS = (LL - LL_mean) / (log(2) * total_spikes)
    bps = (log_likelihood - log_likelihood_mean).sum() / (np.log(2) * y_true.sum() + 1e-8)

    return bps


def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return (y_true - y_pred).abs().mean()


def root_mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Root mean squared error."""
    return torch.sqrt(((y_true - y_pred) ** 2).mean())


class EvaluationMetrics:
    """Container for computing multiple evaluation metrics.

    Parameters
    ----------
    task_type : str
        Type of task ('decoder', 'encoder', 'forecast').
    """

    def __init__(self, task_type: str = 'decoder'):
        self.task_type = task_type
        self.reset()

    def reset(self):
        """Reset accumulated predictions."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Add batch of predictions and targets.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions.
        targets : torch.Tensor
            Ground truth targets.
        """
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns
        -------
        dict
            Dictionary of metric names and values.
        """
        if len(self.predictions) == 0:
            return {}

        # Concatenate all batches
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)

        metrics = {}

        if self.task_type == 'decoder':
            # Behavioral decoding metrics
            metrics['r2'] = r2_score(targets, predictions).item()
            metrics['correlation'] = pearson_correlation(targets, predictions).item()
            metrics['mae'] = mean_absolute_error(targets, predictions).item()
            metrics['rmse'] = root_mean_squared_error(targets, predictions).item()

        elif self.task_type == 'encoder':
            # Neural encoding metrics
            metrics['correlation'] = pearson_correlation(targets, predictions).item()
            metrics['bps'] = bits_per_spike(targets, predictions).item()
            metrics['mae'] = mean_absolute_error(targets, predictions).item()

        elif self.task_type == 'forecast':
            # Forecasting metrics
            metrics['mae'] = mean_absolute_error(targets, predictions).item()
            metrics['rmse'] = root_mean_squared_error(targets, predictions).item()
            # Per-step metrics (if multi-step)
            if len(targets.shape) == 3:  # (batch, steps, dim)
                for step in range(targets.shape[1]):
                    step_mae = mean_absolute_error(
                        targets[:, step, :],
                        predictions[:, step, :]
                    ).item()
                    metrics[f'mae_step_{step+1}'] = step_mae

        return metrics


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    task: str = 'decoder',
    device: str = 'cpu',
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Parameters
    ----------
    model : nn.Module
        NeuroFM-X model.
    dataloader : DataLoader
        Evaluation dataloader.
    task : str, optional
        Task to evaluate ('decoder', 'encoder', 'forecast').
        Default: 'decoder'.
    device : str, optional
        Device to evaluate on.
        Default: 'cpu'.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    model.eval()
    model.to(device)

    metrics_tracker = EvaluationMetrics(task_type=task)

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            if hasattr(model, 'tokenizer'):
                tokens, mask = model.tokenizer(batch['spikes'])
            else:
                tokens = batch['spikes']
                mask = None

            latents = model.encode(tokens, mask) if hasattr(model, 'encode') else model(tokens)

            # Get predictions based on task
            if task == 'decoder':
                pooled = latents.mean(dim=1) if len(latents.shape) == 3 else latents
                predictions = model.heads(pooled, task='decoder')
                targets = batch['behavior_target']

            elif task == 'encoder':
                pooled = latents.mean(dim=1) if len(latents.shape) == 3 else latents
                predictions = model.heads(pooled, task='encoder')
                targets = batch['neural']

            elif task == 'forecast':
                # Assume model has forecast method
                if hasattr(model, 'forecast'):
                    predictions = model.forecast(latents)
                    targets = batch.get('future_spikes', batch['spikes'])
                else:
                    continue

            # Update metrics
            metrics_tracker.update(predictions, targets)

    # Compute final metrics
    return metrics_tracker.compute()
