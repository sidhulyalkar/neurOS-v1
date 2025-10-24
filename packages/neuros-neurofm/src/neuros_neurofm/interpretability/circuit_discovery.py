"""
Circuit Discovery via Causal Interventions

Implements methods to discover computational circuits in NeuroFMx:
- Activation patching
- Path patching
- Ablation experiments
- Minimal circuit identification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import copy


class CircuitDiscovery:
    """
    Discover computational circuits through causal interventions.

    Techniques inspired by mechanistic interpretability research:
    - Activation patching to identify causal neurons
    - Path patching to trace information flow
    - Greedy search for minimal circuits
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: NeuroFMx model to analyze
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Cache for activations
        self.activation_cache = {}
        self.hooks = []

    def activation_patching(
        self,
        clean_input: Dict[str, torch.Tensor],
        corrupted_input: Dict[str, torch.Tensor],
        layer_name: str,
        neuron_indices: Optional[List[int]] = None,
        target_output_fn: Optional[Callable] = None
    ) -> float:
        """
        Activation patching: Replace corrupted activations with clean ones.

        Measures how much replacing specific neurons' activations recovers
        performance from corrupted to clean input.

        Args:
            clean_input: Clean modality dict
            corrupted_input: Corrupted modality dict
            layer_name: Name of layer to patch
            neuron_indices: Which neurons to patch (None = all)
            target_output_fn: Function that extracts target metric from outputs

        Returns:
            recovery: Fraction of performance recovered by patching
        """
        # Default target: decoder output mean
        if target_output_fn is None:
            target_output_fn = lambda outputs: outputs['decoder'].mean()

        # 1. Get clean activations and target
        clean_cache = self._forward_with_cache(clean_input, [layer_name])
        clean_output = self.model(clean_input)
        clean_target = target_output_fn(clean_output).item()

        # 2. Get corrupted activations and target
        corrupted_cache = self._forward_with_cache(corrupted_input, [layer_name])
        corrupted_output = self.model(corrupted_input)
        corrupted_target = target_output_fn(corrupted_output).item()

        # 3. Patch: replace corrupted with clean for specific neurons
        patched_cache = copy.deepcopy(corrupted_cache)

        if layer_name in clean_cache:
            if neuron_indices is not None:
                # Patch specific neurons
                patched_cache[layer_name][..., neuron_indices] = \
                    clean_cache[layer_name][..., neuron_indices]
            else:
                # Patch all
                patched_cache[layer_name] = clean_cache[layer_name]

        # 4. Forward from patched activations
        patched_output = self._forward_from_cache(
            corrupted_input, layer_name, patched_cache
        )
        patched_target = target_output_fn(patched_output).item()

        # 5. Compute recovery
        if abs(clean_target - corrupted_target) > 1e-8:
            recovery = (patched_target - corrupted_target) / \
                      (clean_target - corrupted_target)
        else:
            recovery = 0.0

        return recovery

    def _forward_with_cache(
        self,
        inputs: Dict[str, torch.Tensor],
        layer_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass while caching specified layer activations.
        """
        cache = {}

        # Register hooks
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                cache[name] = output.detach().clone()
            return hook

        for name, module in self.model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Forward
        with torch.no_grad():
            _ = self.model(inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return cache

    def _forward_from_cache(
        self,
        inputs: Dict[str, torch.Tensor],
        layer_name: str,
        cache: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Forward from a cached activation.

        Note: This is a simplified version. Full implementation would need
        to intercept the forward pass at the specified layer.
        """
        # For now, just do a regular forward
        # TODO: Implement proper intervention
        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs

    def discover_minimal_circuit(
        self,
        input_data: Dict[str, torch.Tensor],
        target_behavior: torch.Tensor,
        layer_names: List[str],
        search_method: str = 'greedy',
        threshold: float = 0.9
    ) -> Dict[str, List[int]]:
        """
        Find minimal set of neurons sufficient for target behavior.

        Args:
            input_data: Input modality dict
            target_behavior: Target output to preserve
            layer_names: Layers to search over
            search_method: 'greedy' or 'random'
            threshold: Minimum performance to maintain (fraction of baseline)

        Returns:
            circuit: Dict mapping layer_name -> list of neuron indices
        """
        # Get baseline performance
        with torch.no_grad():
            baseline_output = self.model(input_data)

        baseline_perf = self._compute_performance(baseline_output, target_behavior)

        print(f"Baseline performance: {baseline_perf:.4f}")

        # Initialize: all neurons active
        circuit = {}

        for layer_name in tqdm(layer_names, desc="Searching layers"):
            # Get layer size
            cache = self._forward_with_cache(input_data, [layer_name])

            if layer_name not in cache:
                continue

            n_neurons = cache[layer_name].shape[-1]
            active_neurons = set(range(n_neurons))

            # Greedy ablation
            if search_method == 'greedy':
                for neuron_id in range(n_neurons):
                    # Try removing this neuron
                    test_neurons = active_neurons - {neuron_id}

                    # Test performance
                    perf = self._evaluate_with_ablation(
                        input_data, target_behavior,
                        layer_name, list(active_neurons - test_neurons)
                    )

                    # Keep removed if performance OK
                    if perf >= threshold * baseline_perf:
                        active_neurons = test_neurons
                        print(f"  Removed neuron {neuron_id} from {layer_name}, perf: {perf:.4f}")

            circuit[layer_name] = sorted(list(active_neurons))

        print(f"\nMinimal circuit found:")
        for layer, neurons in circuit.items():
            print(f"  {layer}: {len(neurons)} neurons")

        return circuit

    def _compute_performance(
        self,
        outputs: Dict,
        target: torch.Tensor
    ) -> float:
        """Compute performance metric (e.g., correlation, accuracy)."""
        if 'decoder' in outputs:
            pred = outputs['decoder']

            # Flatten and compute correlation
            pred_flat = pred.flatten()
            target_flat = target.flatten()

            # Pearson correlation
            if len(pred_flat) > 1:
                corr = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                return corr.item()

        return 0.0

    def _evaluate_with_ablation(
        self,
        inputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        layer_name: str,
        ablate_neurons: List[int]
    ) -> float:
        """
        Evaluate model with specific neurons ablated (set to zero).
        """
        # TODO: Implement proper ablation
        # For now, just return baseline performance
        with torch.no_grad():
            outputs = self.model(inputs)

        return self._compute_performance(outputs, target)

    def path_patching(
        self,
        clean_input: Dict[str, torch.Tensor],
        corrupted_input: Dict[str, torch.Tensor],
        source_layer: str,
        target_layer: str
    ) -> Dict:
        """
        Path patching: Trace information flow from source to target layer.

        Identifies which paths through the network are causally important.

        Args:
            clean_input: Clean input
            corrupted_input: Corrupted input
            source_layer: Starting layer
            target_layer: Ending layer

        Returns:
            path_importance: Dict with importance scores for paths
        """
        # Get activations for both inputs
        clean_cache = self._forward_with_cache(
            clean_input, [source_layer, target_layer]
        )
        corrupted_cache = self._forward_with_cache(
            corrupted_input, [source_layer, target_layer]
        )

        # Compute differences
        source_diff = (clean_cache[source_layer] - corrupted_cache[source_layer]).abs().mean()
        target_diff = (clean_cache[target_layer] - corrupted_cache[target_layer]).abs().mean()

        path_importance = {
            'source_diff': source_diff.item(),
            'target_diff': target_diff.item(),
            'amplification': (target_diff / (source_diff + 1e-8)).item()
        }

        return path_importance


# Example usage
if __name__ == '__main__':
    from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX

    # Create model
    model = MultiModalNeuroFMX(d_model=256, n_mamba_blocks=4, n_latents=32)

    # Circuit discovery
    discoverer = CircuitDiscovery(model, device='cpu')

    # Dummy inputs
    clean_input = {
        'spike': torch.randn(2, 50, 100)
    }
    corrupted_input = {
        'spike': torch.randn(2, 50, 100) * 0.1  # Noisy
    }

    # Activation patching
    recovery = discoverer.activation_patching(
        clean_input, corrupted_input,
        layer_name='popt',
        neuron_indices=list(range(10))
    )

    print(f"Recovery from patching: {recovery:.3f}")
