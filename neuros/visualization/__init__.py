"""Visualization utilities for neurOS."""

try:
    from .plots import (
        plot_eeg_signals,
        plot_power_spectrum,
        plot_band_powers,
        plot_confusion_matrix,
        plot_model_comparison,
        plot_learning_curve,
        plot_latency_distribution,
        plot_topomap,
        plot_csp_patterns
    )
    __all__ = [
        'plot_eeg_signals',
        'plot_power_spectrum',
        'plot_band_powers',
        'plot_confusion_matrix',
        'plot_model_comparison',
        'plot_learning_curve',
        'plot_latency_distribution',
        'plot_topomap',
        'plot_csp_patterns'
    ]
except ImportError:
    # Matplotlib not available
    __all__ = []
