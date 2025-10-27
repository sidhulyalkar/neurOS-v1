"""
Ablation Studies for Neural Networks

Tools for systematically ablating (removing or zeroing) components of neural
networks to understand their causal importance for model behavior.

Includes neuron ablation, layer ablation, and systematic ablation studies.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Callable, Tuple, Any, Union
from dataclasses import dataclass
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Results from an ablation experiment.

    Args:
        baseline_metric: Performance without ablation
        ablated_metric: Performance with ablation
        delta: Change in metric (ablated - baseline)
        relative_change: Relative change (delta / baseline)
        component_name: Name of ablated component
    """
    baseline_metric: float
    ablated_metric: float
    delta: float
    relative_change: float
    component_name: str

    def __repr__(self) -> str:
        return (f"AblationResult({self.component_name}: "
                f"baseline={self.baseline_metric:.4f}, "
                f"ablated={self.ablated_metric:.4f}, "
                f"delta={self.delta:.4f}, "
                f"relative={self.relative_change:.4f})")


class NeuronAblation:
    """
    Ablate individual neurons or groups of neurons.

    Systematically zeros out neurons to measure their causal importance.

    Args:
        model: Neural network model
        device: Torch device

    Example:
        >>> ablator = NeuronAblation(model)
        >>> # Ablate neurons 10-20 in layer 6
        >>> result = ablator.ablate_neurons(
        ...     input_data, layer_name='layer_6.mlp',
        ...     neuron_indices=[10, 11, 12, ..., 20],
        ...     metric_fn=accuracy_fn
        ... )
        >>> print(f"Impact: {result.delta:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.hooks = []

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def ablate_neurons(
        self,
        input_data: Tensor,
        layer_name: str,
        neuron_indices: List[int],
        metric_fn: Callable[[Tensor], float],
        ablation_type: str = 'zero',
    ) -> AblationResult:
        """
        Ablate specific neurons in a layer.

        Args:
            input_data: Input tensor
            layer_name: Name of layer containing neurons
            neuron_indices: List of neuron indices to ablate
            metric_fn: Function to compute performance metric
            ablation_type: 'zero' (set to 0) or 'mean' (set to mean activation)

        Returns:
            AblationResult with performance change
        """
        # Baseline: no ablation
        self.model.eval()
        with torch.no_grad():
            baseline_output = self.model(input_data.to(self.device))
            baseline_metric = metric_fn(baseline_output)

        # Ablated: zero out neurons
        if ablation_type == 'mean':
            # First, compute mean activations
            mean_acts = {}
            def capture_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    mean_acts[name] = output.mean(dim=(0, 1))  # Mean over batch and seq
                return hook

            for name, module in self.model.named_modules():
                if name == layer_name:
                    h = module.register_forward_hook(capture_hook(name))
                    self.hooks.append(h)

            with torch.no_grad():
                _ = self.model(input_data.to(self.device))

            self._remove_hooks()
            mean_activation = mean_acts[layer_name]
        else:
            mean_activation = None

        # Now ablate
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                output_tensor = output[0]
                is_tuple = True
            else:
                output_tensor = output
                is_tuple = False

            ablated = output_tensor.clone()

            if ablation_type == 'zero':
                # Zero out specified neurons
                if ablated.dim() == 3:  # [batch, seq, features]
                    ablated[:, :, neuron_indices] = 0
                elif ablated.dim() == 2:  # [batch, features]
                    ablated[:, neuron_indices] = 0
            elif ablation_type == 'mean':
                # Set to mean activation
                if ablated.dim() == 3:
                    ablated[:, :, neuron_indices] = mean_activation[neuron_indices]
                elif ablated.dim() == 2:
                    ablated[:, neuron_indices] = mean_activation[neuron_indices]

            if is_tuple:
                return (ablated,) + output[1:]
            else:
                return ablated

        # Register ablation hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                h = module.register_forward_hook(ablation_hook)
                self.hooks.append(h)

        # Forward with ablation
        with torch.no_grad():
            ablated_output = self.model(input_data.to(self.device))
            ablated_metric = metric_fn(ablated_output)

        self._remove_hooks()

        # Compute delta
        delta = ablated_metric - baseline_metric
        relative_change = delta / (baseline_metric + 1e-10)

        return AblationResult(
            baseline_metric=baseline_metric,
            ablated_metric=ablated_metric,
            delta=delta,
            relative_change=relative_change,
            component_name=f"{layer_name}_neurons_{len(neuron_indices)}",
        )

    def scan_neurons(
        self,
        input_data: Tensor,
        layer_name: str,
        n_neurons: int,
        metric_fn: Callable[[Tensor], float],
        ablation_type: str = 'zero',
        top_k: int = 20,
    ) -> Dict[int, AblationResult]:
        """
        Scan individual neurons to find most important ones.

        Args:
            input_data: Input data
            layer_name: Layer to scan
            n_neurons: Total number of neurons
            metric_fn: Performance metric
            ablation_type: Ablation method
            top_k: Number of top neurons to return

        Returns:
            Dictionary mapping neuron indices to ablation results (top-k)
        """
        results = {}

        # Sample if too many neurons
        if n_neurons > 100:
            sample_indices = torch.randperm(n_neurons)[:100].tolist()
        else:
            sample_indices = list(range(n_neurons))

        for neuron_idx in sample_indices:
            result = self.ablate_neurons(
                input_data, layer_name, [neuron_idx], metric_fn, ablation_type
            )
            results[neuron_idx] = result

        # Sort by absolute impact and return top-k
        sorted_results = sorted(
            results.items(),
            key=lambda x: abs(x[1].delta),
            reverse=True
        )

        return dict(sorted_results[:top_k])


class LayerAblation:
    """
    Ablate entire layers or layer components.

    Args:
        model: Neural network model
        device: Torch device

    Example:
        >>> ablator = LayerAblation(model)
        >>> result = ablator.ablate_layer(input_data, 'layer_6', metric_fn)
        >>> print(f"Layer 6 impact: {result.delta:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.hooks = []

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def ablate_layer(
        self,
        input_data: Tensor,
        layer_name: str,
        metric_fn: Callable[[Tensor], float],
        ablation_type: str = 'zero',
    ) -> AblationResult:
        """
        Ablate an entire layer.

        Args:
            input_data: Input data
            layer_name: Layer to ablate
            metric_fn: Performance metric
            ablation_type: 'zero', 'mean', or 'identity' (pass input through)

        Returns:
            AblationResult
        """
        # Baseline
        self.model.eval()
        with torch.no_grad():
            baseline_output = self.model(input_data.to(self.device))
            baseline_metric = metric_fn(baseline_output)

        # Compute mean if needed
        if ablation_type == 'mean':
            mean_acts = {}
            def capture_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    mean_acts[name] = output.mean(dim=(0, 1))
                return hook

            for name, module in self.model.named_modules():
                if name == layer_name:
                    h = module.register_forward_hook(capture_hook(name))
                    self.hooks.append(h)

            with torch.no_grad():
                _ = self.model(input_data.to(self.device))

            self._remove_hooks()
            mean_activation = mean_acts[layer_name]
        else:
            mean_activation = None

        # Ablation hook
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                output_tensor = output[0]
                is_tuple = True
            else:
                output_tensor = output
                is_tuple = False

            if ablation_type == 'zero':
                ablated = torch.zeros_like(output_tensor)
            elif ablation_type == 'mean':
                ablated = mean_activation.expand_as(output_tensor)
            elif ablation_type == 'identity':
                # Pass input through (skip layer computation)
                if isinstance(input, tuple):
                    ablated = input[0]
                else:
                    ablated = input
            else:
                raise ValueError(f"Unknown ablation_type: {ablation_type}")

            if is_tuple:
                return (ablated,) + output[1:]
            else:
                return ablated

        # Register hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                h = module.register_forward_hook(ablation_hook)
                self.hooks.append(h)

        # Forward with ablation
        with torch.no_grad():
            ablated_output = self.model(input_data.to(self.device))
            ablated_metric = metric_fn(ablated_output)

        self._remove_hooks()

        delta = ablated_metric - baseline_metric
        relative_change = delta / (baseline_metric + 1e-10)

        return AblationResult(
            baseline_metric=baseline_metric,
            ablated_metric=ablated_metric,
            delta=delta,
            relative_change=relative_change,
            component_name=layer_name,
        )

    def scan_layers(
        self,
        input_data: Tensor,
        layer_names: List[str],
        metric_fn: Callable[[Tensor], float],
        ablation_type: str = 'zero',
    ) -> Dict[str, AblationResult]:
        """Scan all specified layers."""
        results = {}

        for layer_name in layer_names:
            result = self.ablate_layer(input_data, layer_name, metric_fn, ablation_type)
            results[layer_name] = result

        return results


class ComponentAblation:
    """
    Ablate specific components (attention, MLP, etc.) within layers.

    For transformer models, separately ablates attention and MLP components.

    Args:
        model: Neural network model
        device: Torch device
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.hooks = []

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def ablate_component(
        self,
        input_data: Tensor,
        component_name: str,
        metric_fn: Callable[[Tensor], float],
        ablation_type: str = 'zero',
    ) -> AblationResult:
        """
        Ablate a specific component (e.g., 'layer_6.attn', 'layer_6.mlp').

        Args:
            input_data: Input data
            component_name: Full component name (e.g., 'layer_6.attn')
            metric_fn: Performance metric
            ablation_type: Ablation method

        Returns:
            AblationResult
        """
        # Baseline
        self.model.eval()
        with torch.no_grad():
            baseline_output = self.model(input_data.to(self.device))
            baseline_metric = metric_fn(baseline_output)

        # Ablation hook
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                output_tensor = output[0]
                is_tuple = True
            else:
                output_tensor = output
                is_tuple = False

            if ablation_type == 'zero':
                ablated = torch.zeros_like(output_tensor)
            elif ablation_type == 'identity':
                if isinstance(input, tuple):
                    ablated = input[0]
                else:
                    ablated = input
            else:
                raise ValueError(f"Unknown ablation_type: {ablation_type}")

            if is_tuple:
                return (ablated,) + output[1:]
            else:
                return ablated

        # Register hook
        for name, module in self.model.named_modules():
            if name == component_name:
                h = module.register_forward_hook(ablation_hook)
                self.hooks.append(h)

        # Forward with ablation
        with torch.no_grad():
            ablated_output = self.model(input_data.to(self.device))
            ablated_metric = metric_fn(ablated_output)

        self._remove_hooks()

        delta = ablated_metric - baseline_metric
        relative_change = delta / (baseline_metric + 1e-10)

        return AblationResult(
            baseline_metric=baseline_metric,
            ablated_metric=ablated_metric,
            delta=delta,
            relative_change=relative_change,
            component_name=component_name,
        )


class AblationStudy:
    """
    Comprehensive ablation study combining multiple ablation methods.

    Provides high-level interface for systematic ablation experiments,
    including hierarchical ablation (layers → components → neurons).

    Args:
        model: Neural network model
        device: Torch device

    Example:
        >>> study = AblationStudy(model)
        >>> results = study.hierarchical_ablation(
        ...     input_data,
        ...     layer_names=['layer_4', 'layer_6', 'layer_8'],
        ...     metric_fn=accuracy_fn,
        ... )
        >>> study.plot_results(results)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.neuron_ablator = NeuronAblation(model, device)
        self.layer_ablator = LayerAblation(model, device)
        self.component_ablator = ComponentAblation(model, device)

    def hierarchical_ablation(
        self,
        input_data: Tensor,
        layer_names: List[str],
        metric_fn: Callable[[Tensor], float],
        ablation_type: str = 'zero',
        ablate_components: bool = True,
        ablate_neurons: bool = False,
        top_k_neurons: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform hierarchical ablation: layers → components → neurons.

        Args:
            input_data: Input data
            layer_names: Layers to ablate
            metric_fn: Performance metric
            ablation_type: Ablation method
            ablate_components: Whether to ablate components (attn, mlp)
            ablate_neurons: Whether to ablate individual neurons
            top_k_neurons: Number of top neurons to ablate per component

        Returns:
            Dictionary with hierarchical results
        """
        results = {
            'layers': {},
            'components': {},
            'neurons': {},
        }

        # Layer-level ablation
        logger.info("Performing layer-level ablation...")
        results['layers'] = self.layer_ablator.scan_layers(
            input_data, layer_names, metric_fn, ablation_type
        )

        # Component-level ablation
        if ablate_components:
            logger.info("Performing component-level ablation...")
            for layer_name in layer_names:
                for component in ['attn', 'mlp']:
                    comp_name = f"{layer_name}.{component}"
                    try:
                        result = self.component_ablator.ablate_component(
                            input_data, comp_name, metric_fn, ablation_type
                        )
                        results['components'][comp_name] = result
                    except Exception as e:
                        logger.warning(f"Could not ablate {comp_name}: {e}")

        # Neuron-level ablation (for most important components)
        if ablate_neurons:
            logger.info("Performing neuron-level ablation...")
            # Find top components by impact
            if results['components']:
                sorted_comps = sorted(
                    results['components'].items(),
                    key=lambda x: abs(x[1].delta),
                    reverse=True
                )
                top_components = [name for name, _ in sorted_comps[:3]]

                for comp_name in top_components:
                    try:
                        # Assume component has 100 neurons (adjust as needed)
                        neuron_results = self.neuron_ablator.scan_neurons(
                            input_data, comp_name, n_neurons=100,
                            metric_fn=metric_fn, ablation_type=ablation_type,
                            top_k=top_k_neurons,
                        )
                        results['neurons'][comp_name] = neuron_results
                    except Exception as e:
                        logger.warning(f"Could not ablate neurons in {comp_name}: {e}")

        return results

    def plot_results(self, results: Dict[str, Any]):
        """
        Plot ablation study results.

        Args:
            results: Results from hierarchical_ablation

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not available for plotting")
            return None

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Layer ablation
        if results['layers']:
            layer_names = list(results['layers'].keys())
            layer_deltas = [r.delta for r in results['layers'].values()]
            axes[0].barh(layer_names, layer_deltas)
            axes[0].set_xlabel('Performance Change')
            axes[0].set_title('Layer Ablation')
            axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

        # Component ablation
        if results['components']:
            comp_names = list(results['components'].keys())
            comp_deltas = [r.delta for r in results['components'].values()]
            axes[1].barh(comp_names, comp_deltas)
            axes[1].set_xlabel('Performance Change')
            axes[1].set_title('Component Ablation')
            axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

        # Neuron ablation (aggregate if multiple components)
        if results['neurons']:
            all_neuron_deltas = []
            all_neuron_names = []
            for comp_name, neuron_results in results['neurons'].items():
                for neuron_idx, result in neuron_results.items():
                    all_neuron_deltas.append(result.delta)
                    all_neuron_names.append(f"{comp_name}_{neuron_idx}")

            # Show top 10
            sorted_pairs = sorted(zip(all_neuron_names, all_neuron_deltas),
                                key=lambda x: abs(x[1]), reverse=True)[:10]
            names, deltas = zip(*sorted_pairs) if sorted_pairs else ([], [])

            axes[2].barh(names, deltas)
            axes[2].set_xlabel('Performance Change')
            axes[2].set_title('Top Neuron Ablations')
            axes[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def summarize(self, results: Dict[str, Any]) -> str:
        """Generate text summary of ablation results."""
        summary = []
        summary.append("=" * 60)
        summary.append("ABLATION STUDY SUMMARY")
        summary.append("=" * 60)

        # Layer results
        if results['layers']:
            summary.append("\nMost Important Layers:")
            sorted_layers = sorted(
                results['layers'].items(),
                key=lambda x: abs(x[1].delta),
                reverse=True
            )
            for name, result in sorted_layers[:5]:
                summary.append(f"  {name}: delta={result.delta:.4f} "
                             f"({result.relative_change*100:.1f}%)")

        # Component results
        if results['components']:
            summary.append("\nMost Important Components:")
            sorted_comps = sorted(
                results['components'].items(),
                key=lambda x: abs(x[1].delta),
                reverse=True
            )
            for name, result in sorted_comps[:5]:
                summary.append(f"  {name}: delta={result.delta:.4f} "
                             f"({result.relative_change*100:.1f}%)")

        # Neuron results
        if results['neurons']:
            summary.append("\nMost Important Neurons:")
            all_neurons = []
            for comp_name, neuron_results in results['neurons'].items():
                for neuron_idx, result in neuron_results.items():
                    all_neurons.append((comp_name, neuron_idx, result))

            sorted_neurons = sorted(all_neurons, key=lambda x: abs(x[2].delta), reverse=True)
            for comp_name, neuron_idx, result in sorted_neurons[:10]:
                summary.append(f"  {comp_name}[{neuron_idx}]: delta={result.delta:.4f}")

        summary.append("=" * 60)
        return "\n".join(summary)
