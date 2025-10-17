"""
Visualization utilities for EEG signals, features, and model performance.

Provides convenient plotting functions for BCI data analysis and debugging.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_eeg_signals(data: np.ndarray, fs: float = 250.0,
                    channel_names: Optional[List[str]] = None,
                    title: str = "EEG Signals",
                    figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot multi-channel EEG signals.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples) or (n_samples,) for single channel
    fs : float
        Sampling frequency in Hz
    channel_names : list of str
        Channel labels
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    """
    check_matplotlib()

    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_channels, n_samples = data.shape
    time = np.arange(n_samples) / fs

    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        ax.plot(time, data[i], 'b-', linewidth=0.5)
        ax.set_ylabel(ch_name, rotation=0, ha='right')
        ax.grid(True, alpha=0.3)

        # Remove x-axis labels except for bottom plot
        if i < n_channels - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_power_spectrum(data: np.ndarray, fs: float = 250.0,
                       channel_names: Optional[List[str]] = None,
                       freq_range: Tuple[float, float] = (0, 50),
                       figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot power spectral density for each channel.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples)
    fs : float
        Sampling frequency
    channel_names : list of str
        Channel labels
    freq_range : tuple
        Frequency range to display (low, high) in Hz
    figsize : tuple
        Figure size
    """
    check_matplotlib()
    from scipy.signal import welch

    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_channels = data.shape[0]

    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        f, Pxx = welch(data[i], fs=fs, nperseg=min(256, data.shape[1]))

        # Plot only specified frequency range
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])

        ax.semilogy(f[freq_mask], Pxx[freq_mask])
        ax.set_ylabel(f'{ch_name}\nPower', rotation=0, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3)

        if i < n_channels - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel('Frequency (Hz)')
    plt.suptitle('Power Spectral Density')
    plt.tight_layout()
    plt.show()


def plot_band_powers(features: np.ndarray, band_names: List[str],
                    channel_names: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot band power features as heatmap.

    Parameters
    ----------
    features : np.ndarray
        Shape (n_channels * n_bands,) or (n_channels, n_bands)
    band_names : list of str
        Names of frequency bands
    channel_names : list of str
        Channel labels
    figsize : tuple
        Figure size
    """
    check_matplotlib()

    if features.ndim == 1:
        n_bands = len(band_names)
        n_channels = len(features) // n_bands
        features = features.reshape(n_channels, n_bands)
    else:
        n_channels, n_bands = features.shape

    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(features, aspect='auto', cmap='viridis')

    # Set ticks
    ax.set_xticks(np.arange(n_bands))
    ax.set_yticks(np.arange(n_channels))
    ax.set_xticklabels(band_names)
    ax.set_yticklabels(channel_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power', rotation=270, labelpad=20)

    ax.set_title('Band Power Features')
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Channel')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                          normalize: bool = True,
                          figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes, n_classes)
    class_names : list of str
        Class labels
    normalize : bool
        Normalize to percentages
    figsize : tuple
        Figure size
    """
    check_matplotlib()

    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_norm
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if normalize:
        cbar.set_label('Proportion', rotation=270, labelpad=20)
    else:
        cbar.set_label('Count', rotation=270, labelpad=20)

    # Set ticks
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            if normalize:
                text = f'{cm_display[i, j]:.1%}\n({cm[i, j]})'
            else:
                text = f'{cm[i, j]}'

            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm_display[i, j] > thresh else "black",
                   fontsize=10)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    plt.show()


def plot_model_comparison(model_names: List[str], metrics: Dict[str, List[float]],
                         figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot bar chart comparing model performance.

    Parameters
    ----------
    model_names : list of str
        Model names
    metrics : dict
        Dictionary mapping metric names to lists of values
    figsize : tuple
        Figure size
    """
    check_matplotlib()

    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=figsize)

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = width * (i - n_metrics / 2 + 0.5)
        ax.bar(x + offset, values, width, label=metric_name)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_learning_curve(train_scores: np.ndarray, val_scores: np.ndarray,
                       train_sizes: np.ndarray,
                       figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot learning curve showing training vs validation performance.

    Parameters
    ----------
    train_scores : np.ndarray
        Training scores for different training sizes
    val_scores : np.ndarray
        Validation scores for different training sizes
    train_sizes : np.ndarray
        Number of training samples used
    figsize : tuple
        Figure size
    """
    check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-',
           label='Training score', linewidth=2)
    ax.plot(train_sizes, np.mean(val_scores, axis=1), 'o-',
           label='Validation score', linewidth=2)

    # Add standard deviation bands
    train_std = np.std(train_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.fill_between(train_sizes,
                    np.mean(train_scores, axis=1) - train_std,
                    np.mean(train_scores, axis=1) + train_std,
                    alpha=0.1)
    ax.fill_between(train_sizes,
                    np.mean(val_scores, axis=1) - val_std,
                    np.mean(val_scores, axis=1) + val_std,
                    alpha=0.1)

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_latency_distribution(latencies: np.ndarray,
                              figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot latency distribution histogram.

    Parameters
    ----------
    latencies : np.ndarray
        Array of latency values in milliseconds
    figsize : tuple
        Figure size
    """
    check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(latencies, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(latencies), color='r', linestyle='--',
               label=f'Mean: {np.mean(latencies):.2f}ms')
    ax1.axvline(np.percentile(latencies, 95), color='orange', linestyle='--',
               label=f'P95: {np.percentile(latencies, 95):.2f}ms')
    ax1.axvline(np.percentile(latencies, 99), color='red', linestyle='--',
               label=f'P99: {np.percentile(latencies, 99):.2f}ms')
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Count')
    ax1.set_title('Latency Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(latencies, vert=True)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Statistics')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_topomap(values: np.ndarray, channel_positions: np.ndarray,
                figsize: Tuple[int, int] = (8, 8)) -> None:
    """
    Plot topographic map of channel values.

    Parameters
    ----------
    values : np.ndarray
        Values for each channel
    channel_positions : np.ndarray
        2D positions (x, y) for each channel
    figsize : tuple
        Figure size
    """
    check_matplotlib()
    from scipy.interpolate import griddata

    fig, ax = plt.subplots(figsize=figsize)

    # Create interpolation grid
    xi = np.linspace(channel_positions[:, 0].min(), channel_positions[:, 0].max(), 100)
    yi = np.linspace(channel_positions[:, 1].min(), channel_positions[:, 1].max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate
    Zi = griddata(channel_positions, values, (Xi, Yi), method='cubic')

    # Plot
    im = ax.contourf(Xi, Yi, Zi, levels=20, cmap='RdBu_r')
    ax.scatter(channel_positions[:, 0], channel_positions[:, 1],
              c='black', s=50, zorder=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=20)

    # Draw head outline
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title('Topographic Map')
    ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_csp_patterns(patterns: np.ndarray, n_components: int = 4,
                     figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot CSP spatial patterns.

    Parameters
    ----------
    patterns : np.ndarray
        CSP patterns (n_channels, n_components)
    n_components : int
        Number of components to plot
    figsize : tuple
        Figure size
    """
    check_matplotlib()

    n_channels = patterns.shape[0]
    n_plot = min(n_components, patterns.shape[1])

    fig, axes = plt.subplots(1, n_plot, figsize=figsize)
    if n_plot == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.bar(range(n_channels), patterns[:, i])
        ax.set_title(f'CSP Component {i+1}')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)

    plt.suptitle('CSP Spatial Patterns')
    plt.tight_layout()
    plt.show()
