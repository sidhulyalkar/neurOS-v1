"""
Energy Cascade Analysis for Neural Networks.

Analyzes hierarchical energy flow through network layers, inspired by
turbulent energy cascades in physics. Tracks how computational "energy"
flows from large scales (early layers) to fine scales (deep layers).

Key Principles:
- Energy injection at input layer
- Forward cascade through layers
- Dissipation at each layer (via nonlinearities)
- Spectral analysis of activation patterns
- Richardson-Kolmogorov cascade theory

Based on:
- Richardson (1922): Weather Prediction by Numerical Process
- Kolmogorov (1941): The local structure of turbulence
- Frisch (1995): Turbulence: The Legacy of A.N. Kolmogorov
- Saxe et al. (2014): Exact solutions to the nonlinear dynamics of learning

Example:
    >>> # Analyze energy cascade through network
    >>> cascade = EnergyCascadeAnalyzer(model)
    >>>
    >>> # Run forward pass and track energy
    >>> result = cascade.analyze_cascade(
    ...     inputs=batch_inputs,
    ...     track_spectrum=True
    ... )
    >>>
    >>> # Visualize cascade
    >>> result.visualize_cascade(use_bokeh=True)
    >>>
    >>> # Check energy conservation
    >>> print(f"Total dissipation: {result.total_dissipation:.4f}")
    >>> print(f"Energy balance: {result.energy_balance:.4f}")

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
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import HoverTool
    from bokeh.palettes import Viridis256, Spectral11
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from neuros_mechint.results import DynamicsResult

logger = logging.getLogger(__name__)


@dataclass
class LayerEnergetics:
    """Energy analysis for a single layer."""

    layer_name: str

    # Energy metrics
    input_energy: float = 0.0
    output_energy: float = 0.0
    dissipation: float = 0.0
    transfer_efficiency: float = 0.0

    # Spectral metrics
    spectral_entropy: float = 0.0
    peak_frequency: float = 0.0
    power_law_exponent: Optional[float] = None

    # Activity patterns
    activation_variance: float = 0.0
    sparsity: float = 0.0

    # Flow metrics
    cascade_rate: float = 0.0  # Rate of energy transfer to next layer


@dataclass
class EnergyCascadeResult:
    """Results from Energy Cascade Analysis."""

    # Per-layer energetics
    layer_energetics: List[LayerEnergetics]

    # Global metrics
    total_input_energy: float = 0.0
    total_dissipation: float = 0.0
    total_output_energy: float = 0.0
    energy_balance: float = 0.0  # Should be close to 0

    # Cascade properties
    cascade_exponent: Optional[float] = None  # Kolmogorov-like exponent
    dissipation_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    transfer_rate: np.ndarray = field(default_factory=lambda: np.array([]))

    # Spectral analysis
    spectral_densities: Optional[Dict[str, np.ndarray]] = None
    frequencies: Optional[np.ndarray] = None

    def visualize_cascade(
        self,
        use_bokeh: bool = True,
        save_path: Optional[str] = None
    ) -> Any:
        """Visualize energy cascade through layers."""
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_bokeh(save_path)
        else:
            return self._visualize_matplotlib(save_path)

    def _visualize_bokeh(self, save_path: Optional[str]) -> Any:
        """Bokeh visualization."""
        plots = []

        # Extract layer names and metrics
        layer_names = [le.layer_name for le in self.layer_energetics]
        layer_indices = list(range(len(layer_names)))

        # Plot 1: Energy flow through layers
        p1 = figure(
            title='Energy Flow Through Network Layers',
            width=1000,
            height=400,
            x_range=layer_names,
            x_axis_label='Layer',
            y_axis_label='Energy'
        )

        input_energies = [le.input_energy for le in self.layer_energetics]
        output_energies = [le.output_energy for le in self.layer_energetics]
        dissipations = [le.dissipation for le in self.layer_energetics]

        p1.line(layer_names, input_energies, legend_label='Input Energy',
               line_width=3, color='green', alpha=0.8)
        p1.line(layer_names, output_energies, legend_label='Output Energy',
               line_width=3, color='blue', alpha=0.8)
        p1.vbar(x=layer_names, top=dissipations, width=0.5,
               legend_label='Dissipation', color='red', alpha=0.6)

        p1.xaxis.major_label_orientation = 0.785  # 45 degrees
        p1.legend.location = "top_right"
        p1.legend.click_policy = "hide"
        plots.append(p1)

        # Plot 2: Cascade rates
        p2 = figure(
            title='Energy Cascade & Transfer Rates',
            width=500,
            height=400,
            x_range=layer_names,
            x_axis_label='Layer',
            y_axis_label='Rate'
        )

        cascade_rates = [le.cascade_rate for le in self.layer_energetics]
        transfer_efficiencies = [le.transfer_efficiency for le in self.layer_energetics]

        p2.line(layer_names, cascade_rates, legend_label='Cascade Rate',
               line_width=3, color='purple', alpha=0.8)
        p2.circle(layer_names, transfer_efficiencies, legend_label='Transfer Efficiency',
                 size=10, color='orange', alpha=0.8)

        p2.xaxis.major_label_orientation = 0.785
        p2.legend.location = "top_right"
        plots.append(p2)

        # Plot 3: Spectral entropy
        p3 = figure(
            title='Spectral Entropy per Layer',
            width=500,
            height=400,
            x_range=layer_names,
            x_axis_label='Layer',
            y_axis_label='Spectral Entropy'
        )

        spectral_entropies = [le.spectral_entropy for le in self.layer_energetics]

        p3.vbar(x=layer_names, top=spectral_entropies, width=0.7,
               color='teal', alpha=0.7)

        p3.xaxis.major_label_orientation = 0.785
        plots.append(p3)

        # Plot 4: Dissipation rate over layers
        if len(self.dissipation_rate) > 0:
            p4 = figure(
                title='Dissipation Rate Progression',
                width=1000,
                height=400,
                x_axis_label='Layer Index',
                y_axis_label='Dissipation Rate'
            )

            p4.line(layer_indices, self.dissipation_rate,
                   line_width=3, color='darkred', alpha=0.8)
            p4.circle(layer_indices, self.dissipation_rate,
                     size=8, color='darkred', alpha=0.8)

            plots.append(p4)

        layout = column(plots[0], row(plots[1], plots[2]), *plots[3:])

        if save_path:
            output_file(save_path)
            save(layout)

        return layout

    def _visualize_matplotlib(self, save_path: Optional[str]) -> Any:
        """Matplotlib visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        n_plots = 4 if len(self.dissipation_rate) > 0 else 3
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        layer_names = [le.layer_name for le in self.layer_energetics]
        layer_indices = np.arange(len(layer_names))

        # Plot 1: Energy flow
        ax1 = axes[0]
        input_energies = [le.input_energy for le in self.layer_energetics]
        output_energies = [le.output_energy for le in self.layer_energetics]
        dissipations = [le.dissipation for le in self.layer_energetics]

        ax1.plot(layer_indices, input_energies, 'o-', label='Input Energy',
                linewidth=2, markersize=6, color='green', alpha=0.7)
        ax1.plot(layer_indices, output_energies, 'o-', label='Output Energy',
                linewidth=2, markersize=6, color='blue', alpha=0.7)
        ax1.bar(layer_indices, dissipations, label='Dissipation',
               alpha=0.5, color='red', width=0.4)

        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Flow Through Layers', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cascade rates
        ax2 = axes[1]
        cascade_rates = [le.cascade_rate for le in self.layer_energetics]
        transfer_efficiencies = [le.transfer_efficiency for le in self.layer_energetics]

        ax2.plot(layer_indices, cascade_rates, 'o-', label='Cascade Rate',
                linewidth=2, markersize=6, color='purple', alpha=0.7)
        ax2.plot(layer_indices, transfer_efficiencies, 's-', label='Transfer Efficiency',
                linewidth=2, markersize=6, color='orange', alpha=0.7)

        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Rate')
        ax2.set_title('Energy Cascade & Transfer Rates', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Spectral entropy
        ax3 = axes[2]
        spectral_entropies = [le.spectral_entropy for le in self.layer_energetics]

        ax3.bar(layer_indices, spectral_entropies, alpha=0.7, color='teal')
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Spectral Entropy')
        ax3.set_title('Spectral Entropy per Layer', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Dissipation rate
        ax4 = axes[3]
        if len(self.dissipation_rate) > 0:
            ax4.plot(layer_indices, self.dissipation_rate, 'o-',
                    linewidth=2, markersize=6, color='darkred', alpha=0.7)
            ax4.set_xlabel('Layer Index')
            ax4.set_ylabel('Dissipation Rate')
            ax4.set_title('Dissipation Rate Progression', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class EnergyCascadeAnalyzer:
    """
    Analyze energy cascades through neural network layers.

    Tracks how computational energy flows from input to output,
    measuring dissipation, transfer efficiency, and spectral properties
    at each layer.

    Args:
        model: Neural network model to analyze
        energy_metric: How to compute energy ('variance', 'l2_norm', 'frobenius')
        track_spectrum: Compute spectral analysis for each layer
        verbose: Enable verbose logging

    Example:
        >>> analyzer = EnergyCascadeAnalyzer(model, energy_metric='variance')
        >>> result = analyzer.analyze_cascade(inputs, track_spectrum=True)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        energy_metric: str = 'variance',
        track_spectrum: bool = True,
        verbose: bool = True
    ):
        self.model = model
        self.energy_metric = energy_metric
        self.track_spectrum = track_spectrum
        self.verbose = verbose

        self._activations = {}
        self._hooks = []

        self._log("Initialized EnergyCascadeAnalyzer")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[EnergyCascadeAnalyzer] {message}")

    def analyze_cascade(
        self,
        inputs: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> EnergyCascadeResult:
        """
        Analyze energy cascade through network.

        Args:
            inputs: Input tensor (batch_size, ...)
            layer_names: Specific layers to track (None = all)

        Returns:
            EnergyCascadeResult with complete cascade analysis
        """
        self._log(f"Analyzing cascade for input shape {inputs.shape}")

        # Register hooks to capture activations
        self._register_hooks(layer_names)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs)

        # Remove hooks
        self._remove_hooks()

        # Compute energetics for each layer
        layer_energetics = []
        prev_energy = self._compute_energy(inputs)

        for layer_name, activation in self._activations.items():
            energetics = self._analyze_layer(
                layer_name,
                activation,
                prev_energy
            )
            layer_energetics.append(energetics)
            prev_energy = energetics.output_energy

        # Compute global metrics
        total_input_energy = self._compute_energy(inputs)
        total_output_energy = self._compute_energy(outputs)
        total_dissipation = sum(le.dissipation for le in layer_energetics)
        energy_balance = total_input_energy - (total_output_energy + total_dissipation)

        # Compute cascade properties
        dissipation_rate = np.array([le.dissipation for le in layer_energetics])
        transfer_rate = np.array([le.cascade_rate for le in layer_energetics])
        cascade_exponent = self._estimate_cascade_exponent(dissipation_rate)

        # Spectral analysis
        spectral_densities = None
        frequencies = None
        if self.track_spectrum:
            spectral_densities, frequencies = self._compute_spectral_densities()

        result = EnergyCascadeResult(
            layer_energetics=layer_energetics,
            total_input_energy=total_input_energy,
            total_dissipation=total_dissipation,
            total_output_energy=total_output_energy,
            energy_balance=energy_balance,
            cascade_exponent=cascade_exponent,
            dissipation_rate=dissipation_rate,
            transfer_rate=transfer_rate,
            spectral_densities=spectral_densities,
            frequencies=frequencies
        )

        self._log(f"Cascade analysis complete: {len(layer_energetics)} layers")
        self._log(f"Energy balance: {energy_balance:.6f}")

        return result

    def _register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to capture activations."""
        self._activations = {}

        def make_hook(name):
            def hook(module, input, output):
                self._activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                if len(list(module.children())) == 0:  # Leaf modules only
                    hook = module.register_forward_hook(make_hook(name))
                    self._hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _compute_energy(self, tensor: torch.Tensor) -> float:
        """Compute energy of a tensor based on chosen metric."""
        if self.energy_metric == 'variance':
            return float(tensor.var().item())
        elif self.energy_metric == 'l2_norm':
            return float(tensor.norm(p=2).item() ** 2)
        elif self.energy_metric == 'frobenius':
            return float((tensor ** 2).sum().item())
        else:
            raise ValueError(f"Unknown energy metric: {self.energy_metric}")

    def _analyze_layer(
        self,
        layer_name: str,
        activation: torch.Tensor,
        prev_energy: float
    ) -> LayerEnergetics:
        """Analyze energetics of a single layer."""
        # Energy metrics
        output_energy = self._compute_energy(activation)
        dissipation = max(0.0, prev_energy - output_energy)
        transfer_efficiency = output_energy / (prev_energy + 1e-10)

        # Spectral analysis
        spectral_entropy = 0.0
        peak_frequency = 0.0
        power_law_exponent = None

        if self.track_spectrum:
            spectral_entropy, peak_frequency, power_law_exponent = self._compute_spectral_metrics(
                activation
            )

        # Activity patterns
        activation_variance = float(activation.var().item())
        sparsity = float((activation.abs() < 1e-3).float().mean().item())

        # Cascade rate (energy transfer to next layer)
        cascade_rate = output_energy / (prev_energy + 1e-10)

        return LayerEnergetics(
            layer_name=layer_name,
            input_energy=prev_energy,
            output_energy=output_energy,
            dissipation=dissipation,
            transfer_efficiency=transfer_efficiency,
            spectral_entropy=spectral_entropy,
            peak_frequency=peak_frequency,
            power_law_exponent=power_law_exponent,
            activation_variance=activation_variance,
            sparsity=sparsity,
            cascade_rate=cascade_rate
        )

    def _compute_spectral_metrics(
        self,
        activation: torch.Tensor
    ) -> Tuple[float, float, Optional[float]]:
        """Compute spectral metrics for activation."""
        # Flatten to 1D for FFT
        act_flat = activation.cpu().numpy().flatten()

        # Compute FFT
        fft = np.fft.fft(act_flat)
        power_spectrum = np.abs(fft) ** 2

        # Normalize
        power_spectrum = power_spectrum / (power_spectrum.sum() + 1e-10)

        # Spectral entropy
        spectral_entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))

        # Peak frequency
        peak_idx = np.argmax(power_spectrum)
        peak_frequency = float(peak_idx / len(power_spectrum))

        # Power law exponent (fit log-log slope)
        power_law_exponent = None
        try:
            # Use first half of spectrum (positive frequencies)
            half_len = len(power_spectrum) // 2
            freqs = np.arange(1, half_len)
            powers = power_spectrum[1:half_len]

            # Log-log fit
            log_freqs = np.log(freqs + 1e-10)
            log_powers = np.log(powers + 1e-10)

            # Linear regression
            coeffs = np.polyfit(log_freqs, log_powers, deg=1)
            power_law_exponent = float(coeffs[0])
        except:
            pass

        return float(spectral_entropy), peak_frequency, power_law_exponent

    def _compute_spectral_densities(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Compute power spectral densities for all layers."""
        spectral_densities = {}
        frequencies = None

        for layer_name, activation in self._activations.items():
            act_flat = activation.cpu().numpy().flatten()

            fft = np.fft.fft(act_flat)
            power_spectrum = np.abs(fft) ** 2

            # Store first half (positive frequencies)
            half_len = len(power_spectrum) // 2
            spectral_densities[layer_name] = power_spectrum[:half_len]

            if frequencies is None:
                frequencies = np.fft.fftfreq(len(power_spectrum))[:half_len]

        return spectral_densities, frequencies

    def _estimate_cascade_exponent(self, dissipation_rate: np.ndarray) -> Optional[float]:
        """Estimate cascade exponent (Kolmogorov-like)."""
        if len(dissipation_rate) < 3:
            return None

        try:
            # Fit power law to dissipation rate
            layers = np.arange(len(dissipation_rate))
            log_layers = np.log(layers[1:] + 1)  # Avoid log(0)
            log_dissipation = np.log(dissipation_rate[1:] + 1e-10)

            # Linear regression in log-log space
            coeffs = np.polyfit(log_layers, log_dissipation, deg=1)
            exponent = float(coeffs[0])

            return exponent
        except:
            return None


__all__ = [
    'LayerEnergetics',
    'EnergyCascadeResult',
    'EnergyCascadeAnalyzer',
]
