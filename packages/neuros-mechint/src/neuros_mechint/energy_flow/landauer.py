"""
Landauer's Principle and Thermodynamics of Computation.

Applies Landauer's principle from thermodynamics to analyze the energy
cost of neural network computations.

Key Insight (Landauer 1961):
    Erasing one bit of information requires minimum energy dissipation of:
    E_min = kT ln(2) ≈ 2.9 × 10^-21 J  (at room temperature)

This fundamental limit applies to ALL computation, including neural networks.

Applications to Neural Networks:
    1. Measure how much "information is erased" during forward pass
    2. Compute theoretical minimum energy required
    3. Compare to actual energy use (if known)
    4. Identify thermodynamically inefficient operations

Physics Background:
    - Second law of thermodynamics: entropy cannot decrease
    - Computation = information processing
    - Irreversible operations (like bit erasure) must dissipate heat
    - Reversible operations can (in principle) be done with zero energy

References:
    - Landauer (1961): "Irreversibility and heat generation in the computing process"
    - Bennett (1973): "Logical reversibility of computation"
    - Parrondo et al. (2015): "Thermodynamics of information"
    - Wolpert (2019): "Stochastic thermodynamics of computation"

Author: NeuroS Team
Date: 2025-10-30
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ROOM_TEMPERATURE = 300  # K (≈27°C)
KT_ROOM = BOLTZMANN_CONSTANT * ROOM_TEMPERATURE  # ≈4.14 × 10^-21 J
LANDAUER_LIMIT = KT_ROOM * np.log(2)  # ≈2.87 × 10^-21 J per bit


@dataclass
class LandauerAnalysis:
    """
    Results from Landauer analysis.

    Attributes:
        total_bits_erased: Total bits erased during computation
        minimum_energy_joules: Theoretical minimum energy (J)
        minimum_energy_flops: Minimum energy per FLOP
        entropy_produced: Total entropy produced (bits)
        irreversible_operations: Number of irreversible ops
        reversibility_score: How reversible is the computation (0=irreversible, 1=reversible)
        layer_analysis: Per-layer analysis
    """
    total_bits_erased: float
    minimum_energy_joules: float
    minimum_energy_flops: float
    entropy_produced: float
    irreversible_operations: int
    reversibility_score: float
    layer_analysis: Dict[str, Dict[str, float]]


class LandauerAnalyzer:
    """
    Analyze neural networks through the lens of Landauer's principle.

    Estimates the thermodynamic cost of computation by measuring
    information erasure and entropy production.

    Args:
        model: Neural network to analyze
        temperature: Temperature in Kelvin (default: room temp)
        device: Torch device

    Example:
        >>> analyzer = LandauerAnalyzer(model)
        >>> analysis = analyzer.analyze_forward_pass(inputs)
        >>> print(f"Minimum energy: {analysis.minimum_energy_joules:.2e} J")
        >>> print(f"Bits erased: {analysis.total_bits_erased:.0f}")
    """

    def __init__(
        self,
        model: nn.Module,
        temperature: float = ROOM_TEMPERATURE,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.model = model
        self.temperature = temperature
        self.kT = BOLTZMANN_CONSTANT * temperature
        self.landauer_limit = self.kT * np.log(2)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # For tracking activations
        self.activations = {}
        self.hooks = []

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[Landauer] {message}")

    def _estimate_entropy(self, tensor: torch.Tensor) -> float:
        """
        Estimate Shannon entropy of tensor values.

        Uses histogram binning to estimate probability distribution,
        then computes H = -∑ p_i log2(p_i)

        Args:
            tensor: Input tensor

        Returns:
            Entropy in bits
        """
        # Flatten tensor
        flat = tensor.detach().cpu().flatten().numpy()

        if len(flat) == 0:
            return 0.0

        # Bin values to estimate distribution
        # Number of bins based on Sturges' rule
        n_bins = int(np.ceil(np.log2(len(flat)) + 1))
        n_bins = max(n_bins, 10)  # At least 10 bins

        # Compute histogram
        counts, _ = np.histogram(flat, bins=n_bins)

        # Convert to probabilities
        probs = counts / counts.sum()

        # Filter out zero probabilities
        probs = probs[probs > 0]

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))

        return entropy

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_activation(name):
            def hook(module, input, output):
                # Store activation
                self.activations[name] = output.detach()
            return hook

        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, nn.Tanh)):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def _analyze_layer(
        self,
        name: str,
        pre_activation: Optional[torch.Tensor],
        post_activation: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze a single layer's thermodynamic cost.

        Args:
            name: Layer name
            pre_activation: Input to layer (if available)
            post_activation: Output of layer

        Returns:
            Dictionary with layer analysis
        """
        # Estimate entropy of output
        entropy_out = self._estimate_entropy(post_activation)

        # If we have input, estimate information erased
        if pre_activation is not None:
            entropy_in = self._estimate_entropy(pre_activation)
            # Information erased = input entropy - output entropy
            # (Assuming input info not fully preserved)
            bits_erased = max(0, entropy_in - entropy_out)
        else:
            # Approximate: assume some information is erased
            # based on output entropy
            bits_erased = entropy_out * 0.1  # Conservative estimate

        # Minimum energy for this layer
        min_energy = bits_erased * self.landauer_limit

        # Count irreversible operations
        # Operations that can't be reversed: ReLU, max pooling, dropout
        is_irreversible = 1 if 'relu' in name.lower() or 'max' in name.lower() else 0

        return {
            'bits_erased': bits_erased,
            'min_energy_joules': min_energy,
            'entropy_in': entropy_in if pre_activation is not None else 0.0,
            'entropy_out': entropy_out,
            'is_irreversible': is_irreversible
        }

    def analyze_forward_pass(
        self,
        inputs: torch.Tensor,
        return_layer_details: bool = False
    ) -> Union[LandauerAnalysis, Tuple[LandauerAnalysis, Dict]]:
        """
        Analyze thermodynamic cost of a forward pass.

        Args:
            inputs: Input data
            return_layer_details: Whether to return detailed layer analysis

        Returns:
            LandauerAnalysis object (and optionally layer details)
        """
        self._log("Starting Landauer analysis...")

        # Register hooks
        self._register_hooks()

        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs)

        # Analyze each layer
        layer_results = {}
        total_bits_erased = 0.0
        total_min_energy = 0.0
        n_irreversible = 0

        # Get ordered layer names
        layer_names = list(self.activations.keys())

        for i, name in enumerate(layer_names):
            post_act = self.activations[name]

            # Try to get pre-activation (output of previous layer)
            pre_act = self.activations[layer_names[i-1]] if i > 0 else inputs

            # Analyze layer
            layer_analysis = self._analyze_layer(name, pre_act, post_act)
            layer_results[name] = layer_analysis

            # Accumulate totals
            total_bits_erased += layer_analysis['bits_erased']
            total_min_energy += layer_analysis['min_energy_joules']
            n_irreversible += layer_analysis['is_irreversible']

        # Remove hooks
        self._remove_hooks()

        # Compute overall metrics
        n_layers = len(layer_results)

        # Reversibility score: fraction of reversible operations
        reversibility = 1.0 - (n_irreversible / n_layers) if n_layers > 0 else 0.0

        # Total entropy produced (in bits)
        entropy_produced = total_bits_erased  # Each bit erased produces entropy

        # Energy per FLOP (approximate)
        # Rough estimate: each parameter involves ~2 FLOPs (multiply + add)
        n_params = sum(p.numel() for p in self.model.parameters())
        n_flops = n_params * 2  # Rough estimate
        energy_per_flop = total_min_energy / n_flops if n_flops > 0 else 0.0

        analysis = LandauerAnalysis(
            total_bits_erased=total_bits_erased,
            minimum_energy_joules=total_min_energy,
            minimum_energy_flops=energy_per_flop,
            entropy_produced=entropy_produced,
            irreversible_operations=n_irreversible,
            reversibility_score=reversibility,
            layer_analysis=layer_results
        )

        self._log(f"Analysis complete!")
        self._log(f"  Total bits erased: {total_bits_erased:.1f}")
        self._log(f"  Minimum energy: {total_min_energy:.2e} J")
        self._log(f"  Reversibility: {reversibility:.1%}")

        if return_layer_details:
            return analysis, layer_results
        return analysis

    def compare_architectures(
        self,
        models: Dict[str, nn.Module],
        inputs: torch.Tensor
    ) -> Dict[str, LandauerAnalysis]:
        """
        Compare thermodynamic efficiency of different architectures.

        Args:
            models: Dictionary of model_name → model
            inputs: Input data (same for all models)

        Returns:
            Dictionary of model_name → LandauerAnalysis
        """
        results = {}

        for name, model in models.items():
            self._log(f"\nAnalyzing {name}...")

            # Temporarily switch model
            old_model = self.model
            self.model = model

            # Analyze
            analysis = self.analyze_forward_pass(inputs)
            results[name] = analysis

            # Restore
            self.model = old_model

        return results

    def visualize_layer_costs(
        self,
        analysis: LandauerAnalysis,
        save_path: Optional[str] = None
    ):
        """
        Visualize per-layer thermodynamic costs.

        Args:
            analysis: LandauerAnalysis result
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self._log("matplotlib required for visualization")
            return

        layer_names = list(analysis.layer_analysis.keys())
        bits_erased = [analysis.layer_analysis[name]['bits_erased']
                      for name in layer_names]
        energies = [analysis.layer_analysis[name]['min_energy_joules']
                   for name in layer_names]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Bits erased per layer
        ax = axes[0]
        ax.bar(range(len(layer_names)), bits_erased, alpha=0.7, color='steelblue')
        ax.set_ylabel('Bits Erased')
        ax.set_title('Information Erasure per Layer')
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Energy cost per layer
        ax = axes[1]
        ax.bar(range(len(layer_names)), energies, alpha=0.7, color='coral')
        ax.set_ylabel('Minimum Energy (J)')
        ax.set_xlabel('Layer')
        ax.set_title('Thermodynamic Cost per Layer (Landauer Limit)')
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Visualization saved to {save_path}")

        return fig


__all__ = [
    'LANDAUER_LIMIT',
    'LandauerAnalysis',
    'LandauerAnalyzer',
]
