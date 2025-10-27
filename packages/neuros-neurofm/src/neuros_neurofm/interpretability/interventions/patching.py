"""
Activation Patching for Causal Analysis

Activation patching (also known as causal tracing or interchange intervention)
is a technique for understanding causal relationships in neural networks by
replacing activations from one forward pass with activations from another.

This reveals which components are causally important for specific behaviors.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Callable, Tuple, Any, Union
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PatchSpec:
    """Specification for a single patch operation.

    Args:
        layer_name: Name of layer to patch (e.g., 'layer_6', 'attn_0')
        component: Component within layer ('residual', 'attn', 'mlp', 'all')
        positions: Token positions to patch (None = all positions)
        features: Feature indices to patch (None = all features)
        source: 'clean' or 'corrupted' (which run to take patched activations from)
    """
    layer_name: str
    component: str = 'all'
    positions: Optional[List[int]] = None
    features: Optional[List[int]] = None
    source: str = 'clean'


class ActivationPatcher:
    """
    General-purpose activation patching tool.

    Performs causal interventions by replacing activations from a "corrupted"
    input with activations from a "clean" input at specific layers/positions.

    This reveals which activations are causally responsible for differences
    in model behavior between clean and corrupted inputs.

    Args:
        model: Neural network model to analyze
        device: Torch device

    Example:
        >>> patcher = ActivationPatcher(model)
        >>> # Setup inputs
        >>> clean_input = ...  # Normal input
        >>> corrupted_input = ...  # Modified input (e.g., masked token)
        >>> # Patch layer 6 activations
        >>> patch = PatchSpec(layer_name='layer_6', component='residual')
        >>> result = patcher.patch(clean_input, corrupted_input, [patch])
        >>> print(f"Recovery: {result['recovery_score']:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Storage for activations
        self.activations: Dict[str, Dict[str, Tensor]] = {
            'clean': {},
            'corrupted': {},
        }

        # Hooks
        self.hooks = []

    def _register_hooks(self, layer_names: List[str]):
        """Register forward hooks to capture activations."""
        self._remove_hooks()

        def make_hook(name: str, run_type: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[run_type][name] = output.detach()
            return hook

        # Register hooks for specified layers
        for name, module in self.model.named_modules():
            if name in layer_names:
                for run_type in ['clean', 'corrupted']:
                    hook = module.register_forward_hook(make_hook(name, run_type))
                    self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def patch(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        patches: List[PatchSpec],
        metric_fn: Optional[Callable[[Tensor], float]] = None,
    ) -> Dict[str, Any]:
        """
        Perform activation patching experiment.

        Args:
            clean_input: Clean input tensor
            corrupted_input: Corrupted input tensor
            patches: List of patch specifications
            metric_fn: Function to compute metric from model output
                      (default: mean of output)

        Returns:
            Dictionary with:
                - clean_output: Model output on clean input
                - corrupted_output: Model output on corrupted input
                - patched_output: Model output with patching
                - clean_metric: Metric on clean output
                - corrupted_metric: Metric on corrupted output
                - patched_metric: Metric on patched output
                - recovery_score: (patched - corrupted) / (clean - corrupted)
        """
        if metric_fn is None:
            metric_fn = lambda x: x.mean().item()

        # Collect unique layer names
        layer_names = list(set(p.layer_name for p in patches))
        self._register_hooks(layer_names)

        # Run clean and corrupted forward passes
        self.model.eval()
        with torch.no_grad():
            clean_output = self.model(clean_input.to(self.device))
            corrupted_output = self.model(corrupted_input.to(self.device))

        # Compute metrics
        clean_metric = metric_fn(clean_output)
        corrupted_metric = metric_fn(corrupted_output)

        # Perform patched forward pass
        patched_output = self._patched_forward(corrupted_input, patches)
        patched_metric = metric_fn(patched_output)

        # Compute recovery score
        if abs(clean_metric - corrupted_metric) < 1e-8:
            recovery_score = 0.0
        else:
            recovery_score = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)

        self._remove_hooks()

        return {
            'clean_output': clean_output,
            'corrupted_output': corrupted_output,
            'patched_output': patched_output,
            'clean_metric': clean_metric,
            'corrupted_metric': corrupted_metric,
            'patched_metric': patched_metric,
            'recovery_score': recovery_score,
        }

    def _patched_forward(
        self,
        input: Tensor,
        patches: List[PatchSpec],
    ) -> Tensor:
        """
        Forward pass with patching interventions.

        During forward pass, replace specified activations with saved
        activations from clean or corrupted run.
        """
        patch_hooks = []

        def make_patch_hook(patch: PatchSpec):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output_tensor = output[0]
                    is_tuple = True
                else:
                    output_tensor = output
                    is_tuple = False

                # Get source activations
                source_act = self.activations[patch.source][patch.layer_name]

                # Create patched output
                patched = output_tensor.clone()

                # Determine what to patch
                if patch.positions is None and patch.features is None:
                    # Patch everything
                    patched = source_act
                elif patch.positions is not None and patch.features is None:
                    # Patch specific positions, all features
                    if patched.dim() == 3:  # [batch, seq, features]
                        patched[:, patch.positions, :] = source_act[:, patch.positions, :]
                    elif patched.dim() == 2:  # [batch, features]
                        patched[:, patch.positions] = source_act[:, patch.positions]
                elif patch.positions is None and patch.features is not None:
                    # Patch all positions, specific features
                    if patched.dim() == 3:
                        patched[:, :, patch.features] = source_act[:, :, patch.features]
                    elif patched.dim() == 2:
                        patched[:, patch.features] = source_act[:, patch.features]
                else:
                    # Patch specific positions and features
                    if patched.dim() == 3:
                        patched[:, patch.positions, :][:, :, patch.features] = \
                            source_act[:, patch.positions, :][:, :, patch.features]

                if is_tuple:
                    return (patched,) + output[1:]
                else:
                    return patched

            return hook

        # Register patch hooks
        for patch in patches:
            for name, module in self.model.named_modules():
                if name == patch.layer_name:
                    hook = module.register_forward_hook(make_patch_hook(patch))
                    patch_hooks.append(hook)

        # Forward pass with patching
        with torch.no_grad():
            output = self.model(input.to(self.device))

        # Remove patch hooks
        for hook in patch_hooks:
            hook.remove()

        return output

    def systematic_patch_scan(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        layer_names: List[str],
        metric_fn: Optional[Callable[[Tensor], float]] = None,
        component: str = 'all',
    ) -> Dict[str, float]:
        """
        Systematically patch each layer and measure recovery.

        Args:
            clean_input: Clean input
            corrupted_input: Corrupted input
            layer_names: List of layers to scan
            metric_fn: Metric function
            component: Component to patch ('all', 'attn', 'mlp')

        Returns:
            Dictionary mapping layer names to recovery scores
        """
        results = {}

        for layer_name in layer_names:
            patch = PatchSpec(
                layer_name=layer_name,
                component=component,
                source='clean',
            )
            result = self.patch(clean_input, corrupted_input, [patch], metric_fn)
            results[layer_name] = result['recovery_score']

        return results


class ResidualStreamPatcher(ActivationPatcher):
    """
    Specialized patcher for transformer residual stream.

    Provides convenient methods for patching residual stream activations
    at specific positions in transformer models.
    """

    def patch_position(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        layer_idx: int,
        position: int,
        metric_fn: Optional[Callable[[Tensor], float]] = None,
    ) -> Dict[str, Any]:
        """Patch residual stream at a specific layer and position."""
        patch = PatchSpec(
            layer_name=f'layer_{layer_idx}',
            component='residual',
            positions=[position],
            source='clean',
        )
        return self.patch(clean_input, corrupted_input, [patch], metric_fn)

    def scan_positions(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        layer_idx: int,
        seq_len: int,
        metric_fn: Optional[Callable[[Tensor], float]] = None,
    ) -> Dict[int, float]:
        """Scan all positions in a layer."""
        results = {}

        for pos in range(seq_len):
            result = self.patch_position(
                clean_input, corrupted_input,
                layer_idx, pos, metric_fn
            )
            results[pos] = result['recovery_score']

        return results


class AttentionPatcher(ActivationPatcher):
    """
    Specialized patcher for attention mechanisms.

    Enables patching of attention patterns, values, or outputs to understand
    the role of attention in model computations.
    """

    def patch_attention_pattern(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        layer_idx: int,
        head_idx: Optional[int] = None,
        metric_fn: Optional[Callable[[Tensor], float]] = None,
    ) -> Dict[str, Any]:
        """
        Patch attention patterns (QK^T scores).

        Args:
            clean_input: Clean input
            corrupted_input: Corrupted input
            layer_idx: Layer index
            head_idx: Specific head to patch (None = all heads)
            metric_fn: Metric function
        """
        layer_name = f'layer_{layer_idx}.attn'

        patch = PatchSpec(
            layer_name=layer_name,
            component='pattern',
            features=[head_idx] if head_idx is not None else None,
            source='clean',
        )

        return self.patch(clean_input, corrupted_input, [patch], metric_fn)

    def scan_heads(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        layer_idx: int,
        n_heads: int,
        metric_fn: Optional[Callable[[Tensor], float]] = None,
    ) -> Dict[int, float]:
        """Scan all attention heads in a layer."""
        results = {}

        for head_idx in range(n_heads):
            result = self.patch_attention_pattern(
                clean_input, corrupted_input,
                layer_idx, head_idx, metric_fn
            )
            results[head_idx] = result['recovery_score']

        return results


class MLPPatcher(ActivationPatcher):
    """
    Specialized patcher for MLP/FFN layers.

    Enables patching of MLP activations and individual neurons to understand
    their computational role.
    """

    def patch_mlp_layer(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        layer_idx: int,
        metric_fn: Optional[Callable[[Tensor], float]] = None,
    ) -> Dict[str, Any]:
        """Patch entire MLP layer output."""
        patch = PatchSpec(
            layer_name=f'layer_{layer_idx}.mlp',
            component='all',
            source='clean',
        )
        return self.patch(clean_input, corrupted_input, [patch], metric_fn)

    def patch_mlp_neurons(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        layer_idx: int,
        neuron_indices: List[int],
        metric_fn: Optional[Callable[[Tensor], float]] = None,
    ) -> Dict[str, Any]:
        """Patch specific MLP neurons."""
        patch = PatchSpec(
            layer_name=f'layer_{layer_idx}.mlp',
            component='neurons',
            features=neuron_indices,
            source='clean',
        )
        return self.patch(clean_input, corrupted_input, [patch], metric_fn)

    def scan_neurons(
        self,
        clean_input: Tensor,
        corrupted_input: Tensor,
        layer_idx: int,
        n_neurons: int,
        metric_fn: Optional[Callable[[Tensor], float]] = None,
        top_k: int = 20,
    ) -> Dict[int, float]:
        """
        Scan individual neurons and return top-k most important.

        Args:
            clean_input: Clean input
            corrupted_input: Corrupted input
            layer_idx: Layer index
            n_neurons: Total number of neurons
            metric_fn: Metric function
            top_k: Number of top neurons to return

        Returns:
            Dictionary mapping neuron indices to recovery scores (top-k only)
        """
        results = {}

        # Sample neurons if too many
        if n_neurons > 100:
            sample_indices = torch.randperm(n_neurons)[:100].tolist()
        else:
            sample_indices = list(range(n_neurons))

        for neuron_idx in sample_indices:
            result = self.patch_mlp_neurons(
                clean_input, corrupted_input,
                layer_idx, [neuron_idx], metric_fn
            )
            results[neuron_idx] = result['recovery_score']

        # Return top-k
        sorted_neurons = sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_neurons[:top_k])
