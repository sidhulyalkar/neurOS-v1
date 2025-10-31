"""
Non-Equilibrium Steady State (NESS) Analysis.

Analyzes neural networks as non-equilibrium thermodynamic systems operating
in steady states. Based on stochastic thermodynamics and NESS theory.

Key Concepts:
- **NESS**: System with constant flows but not in thermal equilibrium
- **Entropy Production**: σ = J·F where J is flux, F is thermodynamic force
- **Steady State Currents**: Persistent probability flows in state space
- **Thermodynamic Efficiency**: Useful work / total energy dissipation
- **Fluctuation-Dissipation Violations**: Signatures of non-equilibrium

Based on:
- Seifert (2012): Stochastic thermodynamics, fluctuation theorems
- Crooks (1999): Entropy production fluctuation theorem
- Hatano & Sasa (2001): Steady-state thermodynamics
- Lebowitz & Spohn (1999): Gallavotti-Cohen theorem

Example:
    >>> # Analyze model in NESS
    >>> analyzer = NESSAnalyzer(model)
    >>>
    >>> # Run inputs and analyze steady state
    >>> result = analyzer.analyze_steady_state(
    ...     inputs=data_loader,
    ...     n_samples=1000
    ... )
    >>>
    >>> # Check if in NESS
    >>> print(f"Entropy production rate: {result.entropy_production_rate:.4f}")
    >>> print(f"Steady state score: {result.steady_state_score:.4f}")
    >>>
    >>> # Visualize NESS properties
    >>> result.visualize_ness_properties(use_bokeh=True)

Author: NeuroS Team
Date: 2025-10-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import HoverTool, ColumnDataSource
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from neuros_mechint.results import MechIntResult

logger = logging.getLogger(__name__)


# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ROOM_TEMPERATURE = 300  # K


@dataclass
class SteadyStateMetrics:
    """Metrics characterizing steady state behavior."""

    # Entropy production
    entropy_production_rate: float  # bits/sample or J/s
    entropy_production_std: float   # Fluctuations

    # Steady state indicators
    steady_state_score: float  # How close to steady state (0-1)
    time_to_steady_state: Optional[float] = None  # Convergence time

    # Currents and flows
    probability_currents: np.ndarray = None  # Flow between states
    current_magnitude: float = 0.0

    # Fluctuation-dissipation
    fd_ratio: float = 1.0  # Should be 1 at equilibrium, != 1 in NESS
    effective_temperature: float = ROOM_TEMPERATURE

    # Thermodynamic efficiency
    efficiency: float = 0.0  # Useful work / dissipation


@dataclass
class NESSAnalysis:
    """Results from NESS analysis."""

    # Steady state metrics
    metrics: SteadyStateMetrics

    # Time series data
    activation_trajectories: np.ndarray  # (n_samples, n_timesteps, n_features)
    entropy_production_timeseries: np.ndarray  # (n_timesteps,)

    # Layer-wise analysis
    layer_metrics: Dict[str, SteadyStateMetrics] = field(default_factory=dict)

    # Statistical properties
    correlation_times: Dict[str, float] = field(default_factory=dict)
    autocorrelations: Dict[str, np.ndarray] = field(default_factory=dict)

    # Metadata
    n_samples: int = 0
    n_timesteps: int = 0
    layer_names: List[str] = field(default_factory=list)

    def visualize_ness_properties(
        self,
        use_bokeh: bool = True,
        save_path: Optional[str] = None
    ) -> Any:
        """
        Visualize NESS properties.

        Args:
            use_bokeh: Use Bokeh for interactive plots
            save_path: Path to save visualization

        Returns:
            Bokeh layout or matplotlib figure
        """
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_bokeh(save_path)
        else:
            return self._visualize_matplotlib(save_path)

    def _visualize_bokeh(self, save_path: Optional[str]) -> Any:
        """Create interactive Bokeh visualization."""

        plots = []

        # 1. Entropy production time series
        time = np.arange(len(self.entropy_production_timeseries))
        source = ColumnDataSource({
            'time': time,
            'entropy': self.entropy_production_timeseries
        })

        p1 = figure(
            title='Entropy Production Rate Over Time',
            x_axis_label='Time Step',
            y_axis_label='dS/dt',
            width=600,
            height=300
        )
        p1.line('time', 'entropy', source=source, line_width=2, color='navy')
        p1.add_tools(HoverTool(tooltips=[("Time", "@time"), ("dS/dt", "@entropy{0.000}")]))
        plots.append(p1)

        # 2. Steady state convergence
        # Compute running mean to show convergence
        window = max(10, len(self.entropy_production_timeseries) // 20)
        running_mean = np.convolve(
            self.entropy_production_timeseries,
            np.ones(window) / window,
            mode='valid'
        )

        p2 = figure(
            title='Convergence to Steady State',
            x_axis_label='Time Step',
            y_axis_label='Running Mean dS/dt',
            width=600,
            height=300
        )
        p2.line(range(len(running_mean)), running_mean, line_width=2, color='red')
        plots.append(p2)

        # 3. Layer-wise entropy production
        if self.layer_metrics:
            layer_names = list(self.layer_metrics.keys())
            entropy_rates = [m.entropy_production_rate for m in self.layer_metrics.values()]
            stds = [m.entropy_production_std for m in self.layer_metrics.values()]

            p3 = figure(
                x_range=layer_names,
                title='Entropy Production by Layer',
                x_axis_label='Layer',
                y_axis_label='dS/dt',
                width=600,
                height=300
            )
            p3.vbar(x=layer_names, top=entropy_rates, width=0.6, color='teal', alpha=0.7)
            p3.xaxis.major_label_orientation = 1
            plots.append(p3)

        # 4. FD ratio distribution (if available)
        if self.layer_metrics:
            fd_ratios = [m.fd_ratio for m in self.layer_metrics.values()]
            layer_names = list(self.layer_metrics.keys())

            p4 = figure(
                x_range=layer_names,
                title='Fluctuation-Dissipation Ratio (1 = Equilibrium)',
                x_axis_label='Layer',
                y_axis_label='FD Ratio',
                width=600,
                height=300
            )
            p4.vbar(x=layer_names, top=fd_ratios, width=0.6, color='orange', alpha=0.7)
            p4.line([-0.5, len(layer_names) - 0.5], [1, 1], line_dash='dashed',
                   line_color='red', line_width=2, legend_label='Equilibrium')
            p4.xaxis.major_label_orientation = 1
            plots.append(p4)

        # Create grid layout
        grid = gridplot([plots[i:i+2] for i in range(0, len(plots), 2)])

        if save_path:
            output_file(save_path)
            save(grid)

        return grid

    def _visualize_matplotlib(self, save_path: Optional[str]) -> Any:
        """Create matplotlib visualization."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return None

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig)

        # 1. Entropy production time series
        ax1 = fig.add_subplot(gs[0, :])
        time = np.arange(len(self.entropy_production_timeseries))
        ax1.plot(time, self.entropy_production_timeseries, linewidth=2, color='navy')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('dS/dt')
        ax1.set_title('Entropy Production Rate Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Running mean (convergence)
        ax2 = fig.add_subplot(gs[1, 0])
        window = max(10, len(self.entropy_production_timeseries) // 20)
        running_mean = np.convolve(
            self.entropy_production_timeseries,
            np.ones(window) / window,
            mode='valid'
        )
        ax2.plot(running_mean, linewidth=2, color='red')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Running Mean dS/dt')
        ax2.set_title('Convergence to Steady State', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Entropy production histogram
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(self.entropy_production_timeseries, bins=50, alpha=0.7, color='green')
        ax3.axvline(self.metrics.entropy_production_rate, color='red',
                   linestyle='--', linewidth=2, label='Mean')
        ax3.set_xlabel('dS/dt')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Entropy Production Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Layer-wise metrics
        if self.layer_metrics:
            ax4 = fig.add_subplot(gs[2, 0])
            layer_names = list(self.layer_metrics.keys())
            entropy_rates = [m.entropy_production_rate for m in self.layer_metrics.values()]
            ax4.bar(layer_names, entropy_rates, alpha=0.7, color='teal')
            ax4.set_xlabel('Layer')
            ax4.set_ylabel('dS/dt')
            ax4.set_title('Entropy Production by Layer', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)

            # 5. FD ratios
            ax5 = fig.add_subplot(gs[2, 1])
            fd_ratios = [m.fd_ratio for m in self.layer_metrics.values()]
            ax5.bar(layer_names, fd_ratios, alpha=0.7, color='orange')
            ax5.axhline(1, color='red', linestyle='--', linewidth=2, label='Equilibrium')
            ax5.set_xlabel('Layer')
            ax5.set_ylabel('FD Ratio')
            ax5.set_title('Fluctuation-Dissipation Ratio', fontweight='bold')
            ax5.tick_params(axis='x', rotation=45)
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class NESSAnalyzer:
    """
    Analyzer for Non-Equilibrium Steady States in neural networks.

    Treats the network as a thermodynamic system and analyzes whether it
    operates in a non-equilibrium steady state with persistent currents
    and entropy production.

    Args:
        model: Neural network to analyze
        device: Torch device
        verbose: Enable verbose logging

    Example:
        >>> analyzer = NESSAnalyzer(model)
        >>> result = analyzer.analyze_steady_state(data_loader)
        >>> print(f"NESS score: {result.metrics.steady_state_score:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        self.model.to(self.device)
        self.model.eval()

        self._log("Initialized NESSAnalyzer")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[NESSAnalyzer] {message}")

    def analyze_steady_state(
        self,
        inputs: Union[torch.Tensor, Any],
        n_samples: int = 1000,
        n_timesteps: int = 100,
        layer_names: Optional[List[str]] = None
    ) -> NESSAnalysis:
        """
        Analyze if network operates in NESS.

        Args:
            inputs: Input data (tensor or dataloader)
            n_samples: Number of samples to analyze
            n_timesteps: Number of time steps to track
            layer_names: Specific layers to analyze (None = all)

        Returns:
            NESSAnalysis with comprehensive metrics
        """
        self._log(f"Analyzing NESS over {n_samples} samples, {n_timesteps} timesteps")

        # Collect activation trajectories
        trajectories, layer_trajectories = self._collect_trajectories(
            inputs, n_samples, n_timesteps, layer_names
        )

        # Compute global NESS metrics
        global_metrics = self._compute_ness_metrics(trajectories)

        # Compute layer-wise metrics
        layer_metrics = {}
        for layer_name, layer_traj in layer_trajectories.items():
            layer_metrics[layer_name] = self._compute_ness_metrics(layer_traj)

        # Compute entropy production time series
        entropy_timeseries = self._compute_entropy_production_timeseries(trajectories)

        # Compute correlations
        correlation_times = {}
        autocorrelations = {}

        for layer_name, layer_traj in layer_trajectories.items():
            autocorr = self._compute_autocorrelation(layer_traj)
            autocorrelations[layer_name] = autocorr
            correlation_times[layer_name] = self._estimate_correlation_time(autocorr)

        result = NESSAnalysis(
            metrics=global_metrics,
            activation_trajectories=trajectories,
            entropy_production_timeseries=entropy_timeseries,
            layer_metrics=layer_metrics,
            correlation_times=correlation_times,
            autocorrelations=autocorrelations,
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            layer_names=list(layer_trajectories.keys())
        )

        self._log(f"NESS analysis complete. Entropy production rate: {global_metrics.entropy_production_rate:.4f}")
        self._log(f"Steady state score: {global_metrics.steady_state_score:.3f}")

        return result

    def _collect_trajectories(
        self,
        inputs: Union[torch.Tensor, Any],
        n_samples: int,
        n_timesteps: int,
        layer_names: Optional[List[str]]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Collect activation trajectories over time."""

        # Determine layers to track
        if layer_names is None:
            layer_names = [
                name for name, module in self.model.named_modules()
                if isinstance(module, (nn.Linear, nn.Conv2d))
            ]

        trajectories_list = []
        layer_trajectories = {name: [] for name in layer_names}

        # Hooks to capture activations
        activations = {}

        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        # Register hooks
        handles = []
        for name in layer_names:
            module = dict(self.model.named_modules())[name]
            handles.append(module.register_forward_hook(make_hook(name)))

        # Collect samples
        sample_count = 0

        if isinstance(inputs, torch.Tensor):
            # Tensor input: treat as sequence
            for t in range(min(n_timesteps, inputs.shape[0])):
                with torch.no_grad():
                    batch = inputs[t:t+1].to(self.device)
                    output = self.model(batch)

                    # Collect activations
                    for name in layer_names:
                        if name in activations:
                            layer_trajectories[name].append(
                                activations[name].cpu().numpy().flatten()
                            )

                sample_count += 1
                if sample_count >= n_samples:
                    break
        else:
            # DataLoader or iterable
            for batch in inputs:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]

                with torch.no_grad():
                    batch = batch.to(self.device)
                    output = self.model(batch)

                    # Collect activations
                    for name in layer_names:
                        if name in activations:
                            layer_trajectories[name].append(
                                activations[name].cpu().numpy().flatten()
                            )

                sample_count += 1
                if sample_count >= n_samples:
                    break

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Convert to arrays
        layer_trajectories_np = {
            name: np.array(traj) for name, traj in layer_trajectories.items()
        }

        # Global trajectory: concatenate all layers
        all_activations = np.concatenate([
            layer_trajectories_np[name]
            for name in layer_names
            if len(layer_trajectories_np[name]) > 0
        ], axis=1)

        return all_activations, layer_trajectories_np

    def _compute_ness_metrics(self, trajectories: np.ndarray) -> SteadyStateMetrics:
        """Compute NESS metrics from trajectories."""

        # Compute entropy production using velocity fluctuations
        if trajectories.shape[0] < 2:
            return SteadyStateMetrics(
                entropy_production_rate=0.0,
                entropy_production_std=0.0,
                steady_state_score=0.0
            )

        velocities = np.diff(trajectories, axis=0)

        # Entropy production rate: proportional to velocity fluctuations
        # σ ≈ <v²> / (2*D) where D is diffusion coefficient
        velocity_sq = np.sum(velocities**2, axis=1)
        D_eff = np.var(velocities) + 1e-10  # Effective diffusion
        entropy_production = velocity_sq / (2 * D_eff)

        entropy_rate = float(np.mean(entropy_production))
        entropy_std = float(np.std(entropy_production))

        # Steady state score: check if statistics are time-independent
        # Compare first half vs second half
        n_half = len(trajectories) // 2
        first_half_mean = np.mean(trajectories[:n_half], axis=0)
        second_half_mean = np.mean(trajectories[n_half:], axis=0)

        mean_diff = np.linalg.norm(first_half_mean - second_half_mean)
        mean_scale = np.linalg.norm(first_half_mean) + 1e-10

        # Score: 1 = perfect steady state, 0 = highly non-stationary
        steady_state_score = float(np.exp(-mean_diff / mean_scale))

        # Compute probability currents (simplified)
        # True NESS has non-zero currents
        current_magnitude = float(np.mean(np.abs(velocities)))

        # Fluctuation-dissipation ratio
        # At equilibrium: <ΔX²> = 2*D*t (Einstein relation)
        # In NESS: modified by effective temperature
        variance_growth = np.var(velocities, axis=0).mean()
        expected_variance = 2 * D_eff  # Expected at equilibrium

        fd_ratio = float(variance_growth / (expected_variance + 1e-10))

        # Effective temperature: T_eff = T * fd_ratio
        T_eff = ROOM_TEMPERATURE * fd_ratio

        return SteadyStateMetrics(
            entropy_production_rate=entropy_rate,
            entropy_production_std=entropy_std,
            steady_state_score=steady_state_score,
            current_magnitude=current_magnitude,
            fd_ratio=fd_ratio,
            effective_temperature=T_eff
        )

    def _compute_entropy_production_timeseries(self, trajectories: np.ndarray) -> np.ndarray:
        """Compute entropy production at each time step."""
        if trajectories.shape[0] < 2:
            return np.array([0.0])

        velocities = np.diff(trajectories, axis=0)
        velocity_sq = np.sum(velocities**2, axis=1)
        D_eff = np.var(velocities) + 1e-10

        entropy_production = velocity_sq / (2 * D_eff)

        return entropy_production

    def _compute_autocorrelation(self, trajectory: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """Compute autocorrelation function."""
        if trajectory.shape[0] < max_lag:
            max_lag = trajectory.shape[0] - 1

        # Average over features
        mean_trajectory = trajectory.mean(axis=1) if trajectory.ndim > 1 else trajectory

        autocorr = []
        for lag in range(max_lag):
            if lag == 0:
                autocorr.append(1.0)
            else:
                corr = np.corrcoef(
                    mean_trajectory[:-lag],
                    mean_trajectory[lag:]
                )[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0.0)

        return np.array(autocorr)

    def _estimate_correlation_time(self, autocorr: np.ndarray) -> float:
        """Estimate correlation time from autocorrelation decay."""
        # Find where autocorrelation drops to 1/e
        threshold = 1 / np.e

        for i, val in enumerate(autocorr):
            if val < threshold:
                return float(i)

        return float(len(autocorr))  # Didn't decay


__all__ = [
    'SteadyStateMetrics',
    'NESSAnalysis',
    'NESSAnalyzer',
]
