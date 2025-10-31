"""
Slow Feature Analysis (SFA) for Temporal Hierarchy Discovery.

Discovers slowly varying features in neural network activations, revealing
temporal structure and hierarchical processing.

Key Principles:
- Fast-changing inputs → Slow-changing features
- Slowness = Temporal predictability
- Hierarchical extraction of time scales
- Unsupervised temporal structure discovery

Based on:
- Wiskott & Sejnowski (2002): Slow Feature Analysis
- Berkes & Wiskott (2005): Slow feature analysis yields a rich repertoire
- Turner & Sahani (2007): A maximum-likelihood interpretation of SFA
- Franzius et al. (2007): Invariant object recognition

Example:
    >>> # Extract slow features
    >>> sfa = SlowFeatureAnalyzer(expansion_degree=2)
    >>>
    >>> # Analyze activation timeseries
    >>> result = sfa.analyze_timeseries(
    ...     activations=layer_activations,
    ...     n_slow_features=10
    ... )
    >>>
    >>> # Check slowness
    >>> print(f"Δ-values: {result.delta_values}")
    >>> print(f"Explained slowness: {result.explained_slowness_ratio}")
    >>>
    >>> # Visualize slow features
    >>> result.visualize_slow_features(use_bokeh=True)

Author: NeuroS Team
Date: 2025-10-31
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import HoverTool
    from bokeh.palettes import Category20_20
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from neuros_mechint.results import DynamicsResult

logger = logging.getLogger(__name__)


@dataclass
class SlowFeatureResult:
    """Results from Slow Feature Analysis."""

    # Slow features
    slow_features: np.ndarray  # (n_timesteps, n_features)
    weights: np.ndarray  # Transformation weights (input_dim, n_features)

    # Slowness metrics
    delta_values: np.ndarray  # Δ-values for each feature (smaller = slower)
    explained_slowness_ratio: np.ndarray  # Cumulative explained slowness

    # Time scales
    characteristic_times: np.ndarray  # Estimated time constant for each feature

    # Metadata
    n_features: int = 0
    original_dim: int = 0
    expansion_degree: int = 1

    def visualize_slow_features(
        self,
        use_bokeh: bool = True,
        save_path: Optional[str] = None,
        n_features_to_plot: int = 5
    ) -> Any:
        """Visualize slow features over time."""
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_bokeh(save_path, n_features_to_plot)
        else:
            return self._visualize_matplotlib(save_path, n_features_to_plot)

    def _visualize_bokeh(self, save_path: Optional[str], n_features: int) -> Any:
        """Bokeh visualization."""
        n_features = min(n_features, self.n_features)

        plots = []

        # Plot 1: Slow features over time
        time = np.arange(len(self.slow_features))

        p1 = figure(
            title='Slow Features Over Time',
            width=1000,
            height=400,
            x_axis_label='Time',
            y_axis_label='Feature Value'
        )

        colors = Category20_20[:n_features]
        for i in range(n_features):
            p1.line(time, self.slow_features[:, i],
                   legend_label=f'SF{i+1} (Δ={self.delta_values[i]:.3f})',
                   line_width=2, alpha=0.7, color=colors[i])

        p1.legend.location = "top_right"
        p1.legend.click_policy = "hide"
        plots.append(p1)

        # Plot 2: Delta values
        p2 = figure(
            title='Δ-values (Slowness) per Feature',
            width=500,
            height=400,
            x_axis_label='Feature Index',
            y_axis_label='Δ-value (lower = slower)'
        )

        features = list(range(1, self.n_features + 1))
        p2.vbar(x=features, top=self.delta_values, width=0.7, alpha=0.7, color='navy')
        plots.append(p2)

        # Plot 3: Explained slowness ratio
        p3 = figure(
            title='Cumulative Explained Slowness',
            width=500,
            height=400,
            x_axis_label='Number of Features',
            y_axis_label='Explained Slowness Ratio'
        )

        p3.line(features, self.explained_slowness_ratio,
               line_width=3, color='green', alpha=0.7)
        p3.circle(features, self.explained_slowness_ratio,
                 size=8, color='green', alpha=0.7)
        plots.append(p3)

        layout = column(plots[0], row(plots[1], plots[2]))

        if save_path:
            output_file(save_path)
            save(layout)

        return layout

    def _visualize_matplotlib(self, save_path: Optional[str], n_features: int) -> Any:
        """Matplotlib visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        n_features = min(n_features, self.n_features)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Slow features over time
        ax1 = axes[0, 0]
        time = np.arange(len(self.slow_features))
        for i in range(n_features):
            ax1.plot(time, self.slow_features[:, i],
                    label=f'SF{i+1} (Δ={self.delta_values[i]:.3f})',
                    alpha=0.7, linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Feature Value')
        ax1.set_title('Slow Features Over Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Delta values
        ax2 = axes[0, 1]
        features = np.arange(1, self.n_features + 1)
        ax2.bar(features, self.delta_values, alpha=0.7, color='navy')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Δ-value (lower = slower)')
        ax2.set_title('Δ-values (Slowness) per Feature', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Explained slowness ratio
        ax3 = axes[1, 0]
        ax3.plot(features, self.explained_slowness_ratio,
                marker='o', linewidth=2, color='green', alpha=0.7)
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Explained Slowness Ratio')
        ax3.set_title('Cumulative Explained Slowness', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Characteristic times
        ax4 = axes[1, 1]
        ax4.bar(features, self.characteristic_times, alpha=0.7, color='teal')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Characteristic Time (steps)')
        ax4.set_title('Feature Time Constants', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class SlowFeatureAnalyzer:
    """
    Extract slowly varying features from temporal data.

    SFA finds features y = g(x) such that:
    1. y has zero mean
    2. y has unit variance
    3. Δ(y) = <ẏ²> is minimized (slowness objective)

    Args:
        expansion_degree: Polynomial expansion degree (1=linear, 2=quadratic, etc.)
        whitening: Apply whitening before SFA
        n_components_pca: Number of PCA components for preprocessing (None = all)
        verbose: Enable verbose logging

    Example:
        >>> analyzer = SlowFeatureAnalyzer(expansion_degree=2)
        >>> result = analyzer.analyze_timeseries(activations, n_slow_features=10)
    """

    def __init__(
        self,
        expansion_degree: int = 2,
        whitening: bool = True,
        n_components_pca: Optional[int] = None,
        verbose: bool = True
    ):
        self.expansion_degree = expansion_degree
        self.whitening = whitening
        self.n_components_pca = n_components_pca
        self.verbose = verbose

        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available. Some features may be limited.")

        self._log("Initialized SlowFeatureAnalyzer")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[SlowFeatureAnalyzer] {message}")

    def analyze_timeseries(
        self,
        activations: np.ndarray,
        n_slow_features: int = 10
    ) -> SlowFeatureResult:
        """
        Extract slow features from activation timeseries.

        Args:
            activations: Time series data (n_timesteps, n_features)
            n_slow_features: Number of slow features to extract

        Returns:
            SlowFeatureResult with extracted features
        """
        self._log(f"Analyzing timeseries: shape {activations.shape}")

        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()

        n_timesteps, n_dims = activations.shape

        # Preprocess: center and optionally whiten
        activations_processed = self._preprocess(activations)

        # Expand features (polynomial)
        activations_expanded = self._expand_features(activations_processed)

        self._log(f"Expanded to {activations_expanded.shape[1]} dimensions")

        # Compute slow features
        slow_features, weights, delta_values = self._compute_slow_features(
            activations_expanded,
            n_slow_features
        )

        # Compute characteristic times
        characteristic_times = self._estimate_characteristic_times(slow_features)

        # Compute explained slowness ratio
        total_slowness = 1.0 / (delta_values.sum() + 1e-10)
        individual_slowness = 1.0 / (delta_values + 1e-10)
        explained_slowness_ratio = np.cumsum(individual_slowness) / (individual_slowness.sum() + 1e-10)

        result = SlowFeatureResult(
            slow_features=slow_features,
            weights=weights,
            delta_values=delta_values,
            explained_slowness_ratio=explained_slowness_ratio,
            characteristic_times=characteristic_times,
            n_features=n_slow_features,
            original_dim=n_dims,
            expansion_degree=self.expansion_degree
        )

        self._log(f"Extracted {n_slow_features} slow features")
        self._log(f"Δ-values range: [{delta_values.min():.4f}, {delta_values.max():.4f}]")

        return result

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess data: center, whiten, and optionally apply PCA."""
        # Center
        data_centered = data - data.mean(axis=0, keepdims=True)

        if not SKLEARN_AVAILABLE:
            return data_centered

        # Optional PCA dimension reduction
        if self.n_components_pca and self.n_components_pca < data.shape[1]:
            pca = PCA(n_components=self.n_components_pca)
            data_centered = pca.fit_transform(data_centered)

        # Whitening
        if self.whitening:
            # Whiten via eigenvalue decomposition
            cov = np.cov(data_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Prevent division by zero
            eigenvalues = np.maximum(eigenvalues, 1e-10)

            # Whitening matrix
            D = np.diag(1.0 / np.sqrt(eigenvalues))
            whitening_matrix = eigenvectors @ D @ eigenvectors.T

            data_whitened = data_centered @ whitening_matrix
            return data_whitened

        return data_centered

    def _expand_features(self, data: np.ndarray) -> np.ndarray:
        """Polynomial expansion of features."""
        if self.expansion_degree == 1:
            return data

        n_timesteps, n_dims = data.shape
        expanded_features = [data]

        # Quadratic terms
        if self.expansion_degree >= 2:
            # All pairwise products (including squares)
            quadratic = []
            for i in range(n_dims):
                for j in range(i, n_dims):
                    quadratic.append(data[:, i] * data[:, j])
            expanded_features.append(np.column_stack(quadratic))

        # Cubic terms (if requested)
        if self.expansion_degree >= 3:
            cubic = []
            for i in range(n_dims):
                for j in range(i, n_dims):
                    for k in range(j, n_dims):
                        cubic.append(data[:, i] * data[:, j] * data[:, k])
            expanded_features.append(np.column_stack(cubic))

        return np.concatenate(expanded_features, axis=1)

    def _compute_slow_features(
        self,
        data: np.ndarray,
        n_features: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute slow features via generalized eigenvalue problem.

        Solves: Ċ w = λ C w

        where:
        - C is the covariance matrix of x
        - Ċ is the covariance matrix of ẋ (time derivative)
        - λ are the Δ-values (slowness)
        """
        n_timesteps, n_dims = data.shape

        # Estimate time derivatives (finite differences)
        derivatives = np.diff(data, axis=0)

        # Adjust data to match derivative length
        data_aligned = data[:-1]

        # Covariance matrices
        C = np.cov(data_aligned.T)
        C_dot = np.cov(derivatives.T)

        # Regularization
        C += 1e-6 * np.eye(C.shape[0])
        C_dot += 1e-6 * np.eye(C_dot.shape[0])

        # Generalized eigenvalue problem
        try:
            eigenvalues, eigenvectors = scipy.linalg.eigh(C_dot, C)
        except:
            # Fallback: standard eigenvalue problem
            C_inv = np.linalg.inv(C + 1e-4 * np.eye(C.shape[0]))
            eigenvalues, eigenvectors = np.linalg.eig(C_inv @ C_dot)

            # Sort by eigenvalues (ascending = slowest first)
            idx = np.argsort(eigenvalues.real)
            eigenvalues = eigenvalues[idx].real
            eigenvectors = eigenvectors[:, idx].real

        # Extract slowest features
        n_features = min(n_features, len(eigenvalues))

        weights = eigenvectors[:, :n_features]
        delta_values = eigenvalues[:n_features]

        # Project data onto slow features
        slow_features = data @ weights

        # Normalize slow features (zero mean, unit variance)
        slow_features = (slow_features - slow_features.mean(axis=0, keepdims=True)) / (
            slow_features.std(axis=0, keepdims=True) + 1e-10
        )

        return slow_features, weights, delta_values

    def _estimate_characteristic_times(self, slow_features: np.ndarray) -> np.ndarray:
        """Estimate characteristic time scale for each slow feature."""
        n_features = slow_features.shape[1]
        characteristic_times = np.zeros(n_features)

        for i in range(n_features):
            # Compute autocorrelation
            feature = slow_features[:, i]
            autocorr = np.correlate(feature, feature, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Normalize
            autocorr = autocorr / autocorr[0]

            # Find where autocorrelation drops to 1/e
            threshold = 1.0 / np.e
            idx = np.where(autocorr < threshold)[0]

            if len(idx) > 0:
                characteristic_times[i] = idx[0]
            else:
                characteristic_times[i] = len(autocorr)

        return characteristic_times


# Import scipy if available
try:
    import scipy.linalg
except ImportError:
    pass


__all__ = [
    'SlowFeatureResult',
    'SlowFeatureAnalyzer',
]
