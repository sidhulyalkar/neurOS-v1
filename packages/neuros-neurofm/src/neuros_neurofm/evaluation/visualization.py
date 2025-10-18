"""
Visualization utilities for NeuroFM-X.

Provides plotting functions for model outputs, latent spaces, and metrics.
"""

from typing import Optional, Tuple

import numpy as np
import torch


def plot_latent_space(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'pca',
    title: str = 'Latent Space',
):
    """Plot 2D projection of latent space.

    Parameters
    ----------
    latents : np.ndarray
        Latent vectors, shape (n_samples, latent_dim).
    labels : np.ndarray, optional
        Labels for coloring points.
    method : str, optional
        Dimensionality reduction method ('pca', 'tsne', 'umap').
        Default: 'pca'.
    title : str, optional
        Plot title.

    Returns
    -------
    dict
        Plot data for external plotting.
    """
    from sklearn.decomposition import PCA

    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
        latents_2d = reducer.fit_transform(latents)
        explained_var = reducer.explained_variance_ratio_
    else:
        # For demo, just use PCA
        reducer = PCA(n_components=2)
        latents_2d = reducer.fit_transform(latents)
        explained_var = reducer.explained_variance_ratio_

    return {
        'latents_2d': latents_2d,
        'labels': labels,
        'title': title,
        'explained_variance': explained_var if method == 'pca' else None,
    }


def plot_behavioral_predictions(
    true_behavior: np.ndarray,
    pred_behavior: np.ndarray,
    time_axis: Optional[np.ndarray] = None,
):
    """Plot behavioral predictions vs ground truth.

    Parameters
    ----------
    true_behavior : np.ndarray
        True behavioral variables, shape (n_samples, n_dims).
    pred_behavior : np.ndarray
        Predicted behavioral variables, same shape.
    time_axis : np.ndarray, optional
        Time axis for x-axis.

    Returns
    -------
    dict
        Plot data.
    """
    if time_axis is None:
        time_axis = np.arange(len(true_behavior))

    return {
        'time': time_axis,
        'true': true_behavior,
        'pred': pred_behavior,
        'mse': np.mean((true_behavior - pred_behavior) ** 2, axis=0),
    }


def plot_neural_forecasts(
    true_activity: np.ndarray,
    predicted_activity: np.ndarray,
    context_length: int = 100,
):
    """Plot neural activity forecasts.

    Parameters
    ----------
    true_activity : np.ndarray
        True neural activity, shape (time, n_neurons).
    predicted_activity : np.ndarray
        Predicted future activity, shape (forecast_steps, n_neurons).
    context_length : int, optional
        Length of context window.

    Returns
    -------
    dict
        Plot data.
    """
    return {
        'context': true_activity[:context_length],
        'true_future': true_activity[context_length:],
        'predicted_future': predicted_activity,
    }


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_metrics: Optional[dict] = None,
    val_metrics: Optional[dict] = None,
):
    """Plot training curves.

    Parameters
    ----------
    train_losses : list
        Training losses per epoch.
    val_losses : list
        Validation losses per epoch.
    train_metrics : dict, optional
        Training metrics per epoch.
    val_metrics : dict, optional
        Validation metrics per epoch.

    Returns
    -------
    dict
        Plot data.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    plot_data = {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
    }

    if train_metrics:
        plot_data['train_metrics'] = train_metrics
    if val_metrics:
        plot_data['val_metrics'] = val_metrics

    return plot_data


def plot_tuning_curves(
    neural_activity: np.ndarray,
    behavioral_variable: np.ndarray,
    n_bins: int = 20,
):
    """Plot neural tuning curves.

    Parameters
    ----------
    neural_activity : np.ndarray
        Neural activity, shape (n_samples, n_neurons).
    behavioral_variable : np.ndarray
        Behavioral variable, shape (n_samples,) or (n_samples, 1).
    n_bins : int, optional
        Number of bins for behavioral variable.

    Returns
    -------
    dict
        Tuning curve data.
    """
    if len(behavioral_variable.shape) > 1:
        behavioral_variable = behavioral_variable[:, 0]

    # Bin behavioral variable
    bins = np.linspace(
        behavioral_variable.min(),
        behavioral_variable.max(),
        n_bins + 1
    )
    bin_indices = np.digitize(behavioral_variable, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute mean activity per bin
    bin_centers = (bins[:-1] + bins[1:]) / 2
    tuning_curves = np.zeros((n_bins, neural_activity.shape[1]))

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            tuning_curves[i] = neural_activity[mask].mean(axis=0)

    return {
        'bin_centers': bin_centers,
        'tuning_curves': tuning_curves,
    }


def summarize_model_performance(
    metrics_dict: dict,
    task: str = 'decoder',
) -> str:
    """Create text summary of model performance.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics.
    task : str, optional
        Task type.

    Returns
    -------
    str
        Formatted summary.
    """
    lines = [f"Model Performance ({task}):", "=" * 50]

    for key, value in sorted(metrics_dict.items()):
        if isinstance(value, float):
            lines.append(f"  {key:20s}: {value:.4f}")
        else:
            lines.append(f"  {key:20s}: {value}")

    lines.append("=" * 50)
    return "\n".join(lines)
