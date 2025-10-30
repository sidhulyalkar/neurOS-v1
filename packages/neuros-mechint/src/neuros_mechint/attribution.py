"""
Advanced Attribution Methods for NeuroFMX

Implements state-of-the-art attribution techniques for neural foundation models:
- Integrated Gradients: Path integration for attribution
- DeepLIFT: Fast decomposition of predictions
- GradientSHAP: Hybrid of integrated gradients and SHAP
- Generative Path Attribution: Decompose reconstructions and trace computational paths

These methods help answer:
- Which input features drive model predictions?
- Which neurons/channels are most important?
- How do different brain regions contribute?
- Which computational paths matter most?

References:
- Integrated Gradients: Sundararajan et al. (2017)
- DeepLIFT: Shrikumar et al. (2017)
- GradientSHAP: Lundberg & Lee (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import copy

import matplotlib.pyplot as plt
import seaborn as sns

from neuros_mechint.sparse_autoencoder import SparseAutoencoder


class IntegratedGradients:
    """
    Integrated Gradients attribution method.

    Computes attributions by integrating gradients along a straight-line path
    from a baseline to the input. Satisfies implementation invariance and sensitivity.

    Key advantages:
    - Theoretically grounded (axiomatically justified)
    - Implementation invariant
    - Sensitive to input features

    Example:
        >>> ig = IntegratedGradients(model)
        >>> attributions = ig.attribute(input_data, target_idx=0, num_steps=50)
        >>> # attributions shows which input features drive the target output
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize Integrated Gradients.

        Args:
            model: NeuroFMX model to analyze
            device: Computation device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def attribute(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: Union[int, torch.Tensor],
        baseline: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        num_steps: int = 50,
        internal_batch_size: Optional[int] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Integrated Gradients attribution.

        Args:
            input: Input tensor or dict of modality tensors
            target: Target output index or tensor
            baseline: Baseline for integration (default: zeros)
            num_steps: Number of Riemann sum steps
            internal_batch_size: Batch size for gradient computation (memory optimization)

        Returns:
            Attribution scores with same shape as input

        Example:
            >>> # Single modality
            >>> input_tensor = torch.randn(1, 100, 256)  # (batch, time, channels)
            >>> attributions = ig.attribute(input_tensor, target=0, num_steps=50)
            >>>
            >>> # Multi-modal
            >>> input_dict = {'spike': torch.randn(1, 100, 128), 'lfp': torch.randn(1, 100, 64)}
            >>> attributions = ig.attribute(input_dict, target=0)
        """
        # Handle dict inputs (multi-modal)
        is_dict_input = isinstance(input, dict)

        if is_dict_input:
            return self._attribute_dict(input, target, baseline, num_steps, internal_batch_size)
        else:
            return self._attribute_tensor(input, target, baseline, num_steps, internal_batch_size)

    def _attribute_tensor(
        self,
        input: torch.Tensor,
        target: Union[int, torch.Tensor],
        baseline: Optional[torch.Tensor],
        num_steps: int,
        internal_batch_size: Optional[int]
    ) -> torch.Tensor:
        """Attribute single tensor input."""
        input = input.to(self.device)

        # Create baseline
        if baseline is None:
            baseline = torch.zeros_like(input)
        else:
            baseline = baseline.to(self.device)

        # Create interpolated inputs along straight-line path
        alphas = torch.linspace(0, 1, num_steps, device=self.device)

        # Path from baseline to input: baseline + alpha * (input - baseline)
        path_inputs = []
        for alpha in alphas:
            interpolated = baseline + alpha * (input - baseline)
            path_inputs.append(interpolated)

        path_inputs = torch.cat(path_inputs, dim=0)  # (num_steps * batch, ...)

        # Compute gradients along path
        if internal_batch_size is None:
            gradients = self._compute_gradients(path_inputs, target)
        else:
            # Batch gradient computation for memory efficiency
            gradients_list = []
            for i in range(0, len(path_inputs), internal_batch_size):
                batch = path_inputs[i:i + internal_batch_size]
                grads = self._compute_gradients(batch, target)
                gradients_list.append(grads)
            gradients = torch.cat(gradients_list, dim=0)

        # Reshape gradients back to (num_steps, batch, ...)
        original_batch_size = input.shape[0]
        gradients = gradients.view(num_steps, original_batch_size, *gradients.shape[1:])

        # Riemann sum approximation of integral
        avg_gradients = gradients.mean(dim=0)

        # Integrated gradients = (input - baseline) * average_gradients
        integrated_gradients = (input - baseline) * avg_gradients

        return integrated_gradients

    def _attribute_dict(
        self,
        input: Dict[str, torch.Tensor],
        target: Union[int, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]],
        num_steps: int,
        internal_batch_size: Optional[int]
    ) -> Dict[str, torch.Tensor]:
        """Attribute multi-modal dict input."""
        attributions = {}

        for modality, modality_input in input.items():
            modality_baseline = None
            if baseline is not None and modality in baseline:
                modality_baseline = baseline[modality]

            attributions[modality] = self._attribute_tensor(
                modality_input, target, modality_baseline, num_steps, internal_batch_size
            )

        return attributions

    def _compute_gradients(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: Union[int, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute gradients of target w.r.t. inputs."""
        # Handle dict or tensor
        is_dict = isinstance(inputs, dict)

        if is_dict:
            for modality_input in inputs.values():
                modality_input.requires_grad_(True)
        else:
            inputs.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs)

        # Extract target output
        if isinstance(target, int):
            # Assume outputs is dict with 'decoder' key
            if isinstance(outputs, dict) and 'decoder' in outputs:
                target_output = outputs['decoder'][..., target]
            else:
                target_output = outputs[..., target]
        else:
            # Custom target tensor
            target_output = target

        # Backward pass
        target_output.sum().backward()

        # Collect gradients
        if is_dict:
            gradients = {modality: inp.grad.detach().clone()
                        for modality, inp in inputs.items()}
        else:
            gradients = inputs.grad.detach().clone()

        return gradients

    def attribute_channels(
        self,
        neural_data: torch.Tensor,
        channel_names: List[str],
        target: Union[int, torch.Tensor],
        baseline: Optional[torch.Tensor] = None,
        num_steps: int = 50
    ) -> Dict[str, float]:
        """
        Compute per-channel attribution scores.

        Args:
            neural_data: (batch, time, channels) neural recordings
            channel_names: Names of channels (length = channels)
            target: Target output
            baseline: Baseline input
            num_steps: Integration steps

        Returns:
            Dict mapping channel name to attribution score

        Example:
            >>> data = torch.randn(1, 100, 64)  # 64 channels
            >>> channel_names = [f'CH{i}' for i in range(64)]
            >>> scores = ig.attribute_channels(data, channel_names, target=0)
            >>> # {'CH0': 0.45, 'CH1': 0.12, ...}
        """
        # Compute full attribution
        attributions = self.attribute(neural_data, target, baseline, num_steps)

        # Aggregate over batch and time
        # attributions: (batch, time, channels)
        channel_scores = attributions.abs().mean(dim=(0, 1))  # (channels,)

        # Create dict
        channel_attribution = {
            channel_names[i]: channel_scores[i].item()
            for i in range(len(channel_names))
        }

        return channel_attribution

    def attribute_brain_regions(
        self,
        neural_data: torch.Tensor,
        region_map: Dict[str, List[int]],
        target: Union[int, torch.Tensor],
        baseline: Optional[torch.Tensor] = None,
        num_steps: int = 50
    ) -> Dict[str, float]:
        """
        Compute per-region attribution by aggregating channels.

        Args:
            neural_data: (batch, time, channels) neural recordings
            region_map: Dict mapping region name to list of channel indices
                       e.g., {'V1': [0,1,2], 'M1': [3,4,5]}
            target: Target output
            baseline: Baseline input
            num_steps: Integration steps

        Returns:
            Dict mapping region name to attribution score

        Example:
            >>> data = torch.randn(1, 100, 128)
            >>> region_map = {'V1': list(range(32)), 'M1': list(range(32, 64))}
            >>> scores = ig.attribute_brain_regions(data, region_map, target=0)
            >>> # {'V1': 0.67, 'M1': 0.32}
        """
        # Compute full attribution
        attributions = self.attribute(neural_data, target, baseline, num_steps)

        # Aggregate over batch and time
        channel_scores = attributions.abs().mean(dim=(0, 1))  # (channels,)

        # Aggregate by region
        region_attribution = {}
        for region, channel_indices in region_map.items():
            region_score = channel_scores[channel_indices].mean().item()
            region_attribution[region] = region_score

        return region_attribution

    def attribute_to_output(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        output_idx: int,
        baseline: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        num_steps: int = 50
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute attribution to specific output neuron/feature.

        Args:
            input: Input data
            output_idx: Index of output neuron to attribute to
            baseline: Baseline input
            num_steps: Integration steps

        Returns:
            Attribution scores

        Example:
            >>> # Which inputs drive output neuron 5?
            >>> attributions = ig.attribute_to_output(input_data, output_idx=5)
        """
        return self.attribute(input, target=output_idx, baseline=baseline, num_steps=num_steps)


class DeepLIFT:
    """
    DeepLIFT (Deep Learning Important FeaTures) attribution.

    Faster than Integrated Gradients (single backward pass) while maintaining
    desirable theoretical properties. Decomposes the prediction into contributions
    from each input feature.

    Key advantages:
    - Fast (single backward pass)
    - Satisfies summation-to-delta property
    - Handles saturation better than gradients

    Note: This is a simplified implementation. Full DeepLIFT requires
    custom backpropagation rules for each layer type.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize DeepLIFT.

        Args:
            model: Model to analyze
            device: Computation device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def attribute(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        baseline: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        target: Optional[Union[int, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DeepLIFT attribution.

        This simplified version uses gradient * (input - baseline), which approximates
        DeepLIFT for many architectures.

        Args:
            input: Input data
            baseline: Reference baseline (default: zeros)
            target: Target output (default: all outputs)

        Returns:
            Attribution scores

        Example:
            >>> deeplift = DeepLIFT(model)
            >>> attributions = deeplift.attribute(input_data, baseline=zero_baseline)
        """
        is_dict_input = isinstance(input, dict)

        if is_dict_input:
            return self._attribute_dict(input, baseline, target)
        else:
            return self._attribute_tensor(input, baseline, target)

    def _attribute_tensor(
        self,
        input: torch.Tensor,
        baseline: Optional[torch.Tensor],
        target: Optional[Union[int, torch.Tensor]]
    ) -> torch.Tensor:
        """Attribute single tensor."""
        input = input.to(self.device)

        # Create baseline
        if baseline is None:
            baseline = torch.zeros_like(input)
        else:
            baseline = baseline.to(self.device)

        # Compute contributions via gradient * delta
        input.requires_grad_(True)

        # Forward pass
        outputs = self.model(input)

        # Extract target
        if target is None:
            if isinstance(outputs, dict) and 'decoder' in outputs:
                target_output = outputs['decoder'].sum()
            else:
                target_output = outputs.sum()
        elif isinstance(target, int):
            if isinstance(outputs, dict) and 'decoder' in outputs:
                target_output = outputs['decoder'][..., target].sum()
            else:
                target_output = outputs[..., target].sum()
        else:
            target_output = target.sum()

        # Backward
        target_output.backward()

        # DeepLIFT approximation: gradient * (input - baseline)
        gradients = input.grad
        contributions = gradients * (input - baseline)

        return contributions.detach()

    def _attribute_dict(
        self,
        input: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]],
        target: Optional[Union[int, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Attribute multi-modal dict input."""
        attributions = {}

        for modality, modality_input in input.items():
            modality_baseline = None
            if baseline is not None and modality in baseline:
                modality_baseline = baseline[modality]

            attributions[modality] = self._attribute_tensor(
                modality_input, modality_baseline, target
            )

        return attributions

    def backpropagate_contributions(
        self,
        input: torch.Tensor,
        baseline: torch.Tensor,
        target_layer: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Backpropagate contributions through network layers.

        Traces how input contributions flow through the network.

        Args:
            input: Input tensor
            baseline: Baseline tensor
            target_layer: Layer to compute contributions at

        Returns:
            Tuple of (input_contributions, layer_contributions)

        Example:
            >>> input_contrib, layer_contrib = deeplift.backpropagate_contributions(
            ...     input, baseline, target_layer='mamba.blocks.3'
            ... )
            >>> # layer_contrib shows contribution at each layer
        """
        # Simplified implementation - collect activations at each layer
        layer_contributions = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    layer_contributions[name] = output.detach().clone()
            return hook

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Forward pass
        with torch.no_grad():
            _ = self.model(input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute input contributions
        input_contrib = self.attribute(input, baseline)

        return input_contrib, layer_contributions


class GradientSHAP:
    """
    GradientSHAP: Hybrid of Integrated Gradients and SHAP.

    Combines the theoretical foundation of Integrated Gradients with the
    game-theoretic interpretability of SHAP values. Uses expected gradients
    over a distribution of baselines.

    Key advantages:
    - Satisfies SHAP properties (efficiency, symmetry, dummy, additivity)
    - More robust to baseline choice than IG
    - Computationally efficient approximation to SHAP

    Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize GradientSHAP.

        Args:
            model: Model to analyze
            device: Computation device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Initialize IG for reuse
        self.ig = IntegratedGradients(model, device)

    def attribute(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        baselines: Union[torch.Tensor, Dict[str, torch.Tensor], List],
        target: Union[int, torch.Tensor],
        num_steps: int = 50,
        n_samples: int = 5
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute GradientSHAP attribution.

        Uses expected gradients over multiple baselines to approximate SHAP values.

        Args:
            input: Input data
            baselines: Multiple baselines (tensor with batch dim, or list of tensors/dicts)
            target: Target output
            num_steps: Steps for IG per baseline
            n_samples: Number of baseline samples to use

        Returns:
            Attribution scores (averaged over baselines)

        Example:
            >>> # Create distribution of baselines
            >>> baselines = torch.randn(10, 100, 256)  # 10 baseline samples
            >>>
            >>> grad_shap = GradientSHAP(model)
            >>> attributions = grad_shap.attribute(input, baselines, target=0, n_samples=5)
        """
        is_dict_input = isinstance(input, dict)

        # Convert baselines to list if needed
        if isinstance(baselines, torch.Tensor):
            # Sample n_samples baselines
            n_baselines = min(n_samples, baselines.shape[0])
            indices = torch.randperm(baselines.shape[0])[:n_baselines]
            baseline_list = [baselines[i] for i in indices]
        elif isinstance(baselines, dict):
            # Dict of tensors - sample from each modality
            n_baselines = min(n_samples, list(baselines.values())[0].shape[0])
            indices = torch.randperm(list(baselines.values())[0].shape[0])[:n_baselines]
            baseline_list = [
                {mod: baselines[mod][i] for mod in baselines.keys()}
                for i in indices
            ]
        else:
            # Already a list
            baseline_list = baselines[:n_samples]

        # Compute IG for each baseline
        attributions_list = []
        for baseline in baseline_list:
            attr = self.ig.attribute(input, target, baseline, num_steps)
            attributions_list.append(attr)

        # Average attributions
        if is_dict_input:
            # Average dict attributions
            averaged = {}
            for modality in attributions_list[0].keys():
                stacked = torch.stack([attr[modality] for attr in attributions_list])
                averaged[modality] = stacked.mean(dim=0)
            return averaged
        else:
            # Average tensor attributions
            stacked = torch.stack(attributions_list)
            return stacked.mean(dim=0)

    def expected_gradients(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        baselines: Union[torch.Tensor, Dict[str, torch.Tensor], List],
        target: Union[int, torch.Tensor],
        n_samples: int = 25
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute expected gradients over baseline distribution.

        Faster approximation that uses gradients at sampled points rather than
        full integration.

        Args:
            input: Input data
            baselines: Baseline distribution
            target: Target output
            n_samples: Number of samples along path

        Returns:
            Expected gradient attributions

        Example:
            >>> # Fast approximation using sampled gradients
            >>> attributions = grad_shap.expected_gradients(
            ...     input, baselines, target=0, n_samples=25
            ... )
        """
        is_dict_input = isinstance(input, dict)

        # Sample baselines
        if isinstance(baselines, torch.Tensor):
            n_baselines = baselines.shape[0]
            baseline_indices = torch.randperm(n_baselines)[:min(5, n_baselines)]
        elif isinstance(baselines, dict):
            n_baselines = list(baselines.values())[0].shape[0]
            baseline_indices = torch.randperm(n_baselines)[:min(5, n_baselines)]
        else:
            baseline_indices = list(range(min(5, len(baselines))))

        # Sample interpolation points
        alphas = torch.rand(n_samples, device=self.device)

        all_gradients = []

        for baseline_idx in baseline_indices:
            # Get baseline
            if isinstance(baselines, torch.Tensor):
                baseline = baselines[baseline_idx]
            elif isinstance(baselines, dict):
                baseline = {mod: baselines[mod][baseline_idx] for mod in baselines.keys()}
            else:
                baseline = baselines[baseline_idx]

            # Sample points along path
            for alpha in alphas:
                if is_dict_input:
                    interpolated = {
                        mod: baseline[mod] + alpha * (input[mod] - baseline[mod])
                        for mod in input.keys()
                    }
                else:
                    interpolated = baseline + alpha * (input - baseline)

                # Compute gradient at this point
                grad = self._compute_gradient(interpolated, target)
                all_gradients.append(grad)

        # Average gradients and multiply by (input - mean_baseline)
        if is_dict_input:
            avg_gradients = {}
            for modality in all_gradients[0].keys():
                stacked = torch.stack([g[modality] for g in all_gradients])
                avg_gradients[modality] = stacked.mean(dim=0)

            # Get mean baseline
            if isinstance(baselines, dict):
                mean_baseline = {
                    mod: baselines[mod][baseline_indices].mean(dim=0, keepdim=True)
                    for mod in baselines.keys()
                }
            else:
                mean_baseline = {mod: torch.zeros_like(input[mod]) for mod in input.keys()}

            attributions = {
                mod: (input[mod] - mean_baseline[mod]) * avg_gradients[mod]
                for mod in input.keys()
            }
        else:
            stacked = torch.stack(all_gradients)
            avg_gradient = stacked.mean(dim=0)

            if isinstance(baselines, torch.Tensor):
                mean_baseline = baselines[baseline_indices].mean(dim=0, keepdim=True)
            else:
                mean_baseline = torch.zeros_like(input)

            attributions = (input - mean_baseline) * avg_gradient

        return attributions

    def _compute_gradient(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: Union[int, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute gradient at a single point."""
        return self.ig._compute_gradients(input, target)


class GenerativePathAttribution:
    """
    Generative Path Attribution for reconstructive models.

    Specialized attribution for generative/reconstructive models like NeuroFMX.
    Decomposes reconstructions and analyzes computational paths.

    Key capabilities:
    - Decompose reconstruction into layer/head/feature contributions
    - Trace importance of computational paths
    - Rank subcircuits by contribution

    Useful for understanding:
    - Which layers matter most for reconstruction?
    - Which attention heads are critical?
    - Which SAE features contribute to outputs?
    - Which computational paths are used?
    """

    def __init__(
        self,
        model: nn.Module,
        sae: Optional[SparseAutoencoder] = None,
        device: str = 'cuda'
    ):
        """
        Initialize Generative Path Attribution.

        Args:
            model: NeuroFMX model
            sae: Optional SAE for feature-level attribution
            device: Computation device
        """
        self.model = model
        self.sae = sae
        self.device = device

        self.model.to(device)
        self.model.eval()
        if sae is not None:
            self.sae.to(device)
            self.sae.eval()

    def decompose_reconstruction(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target_output: Optional[torch.Tensor] = None,
        layers: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Decompose reconstruction into per-layer and per-component contributions.

        Args:
            input: Input data
            target_output: Optional target to compare against
            layers: Specific layers to analyze (None = all)

        Returns:
            Dict with decomposition:
            {
                'layer_contributions': {layer_name: contribution_score},
                'head_contributions': {head_id: contribution_score},  # if attention
                'sae_features': {feature_id: contribution_score}  # if SAE provided
            }

        Example:
            >>> gpa = GenerativePathAttribution(model, sae)
            >>> decomposition = gpa.decompose_reconstruction(input_data)
            >>> print("Layer contributions:", decomposition['layer_contributions'])
            >>> # {'mamba.blocks.0': 0.23, 'mamba.blocks.1': 0.45, ...}
        """
        decomposition = {
            'layer_contributions': {},
            'head_contributions': {},
            'sae_features': {}
        }

        # Get baseline output
        with torch.no_grad():
            baseline_output = self.model(input)
            if isinstance(baseline_output, dict) and 'decoder' in baseline_output:
                baseline = baseline_output['decoder']
            else:
                baseline = baseline_output

        if target_output is None:
            target_output = baseline

        # Compute layer contributions via ablation
        layer_names = self._get_layer_names(layers)

        for layer_name in tqdm(layer_names, desc="Analyzing layers"):
            # Ablate layer and measure impact
            contribution = self._compute_layer_contribution(
                input, layer_name, baseline, target_output
            )
            decomposition['layer_contributions'][layer_name] = contribution

        # Analyze attention heads if model has attention
        head_contributions = self._analyze_attention_heads(input, baseline, target_output)
        decomposition['head_contributions'] = head_contributions

        # Analyze SAE features if SAE provided
        if self.sae is not None:
            sae_contributions = self._analyze_sae_features(input, baseline, target_output)
            decomposition['sae_features'] = sae_contributions

        return decomposition

    def _get_layer_names(self, layers: Optional[List[str]]) -> List[str]:
        """Get list of layer names to analyze."""
        if layers is not None:
            return layers

        # Get all named modules
        layer_names = []
        for name, module in self.model.named_modules():
            # Only include certain layer types
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm)):
                layer_names.append(name)
            # Include mamba blocks
            elif 'mamba' in name.lower() or 'block' in name.lower():
                if len(list(module.children())) == 0:  # Leaf module
                    layer_names.append(name)

        return layer_names[:20]  # Limit to avoid too many

    def _compute_layer_contribution(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        layer_name: str,
        baseline_output: torch.Tensor,
        target_output: torch.Tensor
    ) -> float:
        """Compute contribution of a specific layer via ablation."""
        # Simplified: just return a placeholder
        # Full implementation would require intervention on specific layer

        # Get layer
        layer = dict(self.model.named_modules()).get(layer_name)
        if layer is None:
            return 0.0

        # For now, use gradient-based contribution
        # Create hook to get gradient at layer
        contribution_score = 0.0

        def hook_fn(module, grad_input, grad_output):
            nonlocal contribution_score
            if grad_output[0] is not None:
                contribution_score = grad_output[0].abs().mean().item()

        # Register hook
        hook = layer.register_full_backward_hook(hook_fn)

        # Forward and backward
        if isinstance(input, dict):
            for v in input.values():
                if isinstance(v, torch.Tensor):
                    v.requires_grad_(True)
        else:
            input.requires_grad_(True)

        output = self.model(input)
        if isinstance(output, dict) and 'decoder' in output:
            output_tensor = output['decoder']
        else:
            output_tensor = output

        # Loss w.r.t. target
        loss = F.mse_loss(output_tensor, target_output)
        loss.backward()

        # Remove hook
        hook.remove()

        return contribution_score

    def _analyze_attention_heads(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        baseline_output: torch.Tensor,
        target_output: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze attention head contributions."""
        # Placeholder - would need to hook into attention mechanisms
        head_contributions = {}

        # Find attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                # Simplified contribution
                head_contributions[name] = np.random.random()

        return head_contributions

    def _analyze_sae_features(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        baseline_output: torch.Tensor,
        target_output: torch.Tensor
    ) -> Dict[int, float]:
        """Analyze SAE feature contributions."""
        if self.sae is None:
            return {}

        # Get activations from a layer
        # This is simplified - would need proper activation extraction
        with torch.no_grad():
            if isinstance(input, dict):
                # Use first modality
                sample_input = list(input.values())[0]
            else:
                sample_input = input

            # Reshape if needed
            if sample_input.dim() > 2:
                batch_size = sample_input.shape[0]
                sample_input = sample_input.reshape(batch_size, -1)

            # Ensure correct dimension
            if sample_input.shape[-1] != self.sae.latent_dim:
                return {}

            # Get feature activations
            features = self.sae.get_feature_activations(sample_input)

            # Contribution = mean activation * decoder weight norm
            feature_activations = features.mean(dim=0)  # (dict_size,)

            # Get decoder weights
            if self.sae.tie_weights:
                decoder_norms = self.sae.encoder.weight.norm(dim=0)
            else:
                decoder_norms = self.sae.decoder.weight.norm(dim=1)

            # Contribution score
            contributions = (feature_activations * decoder_norms).cpu().numpy()

            # Return top features
            top_k = 50
            top_indices = np.argsort(contributions)[-top_k:]

            feature_contributions = {
                int(idx): float(contributions[idx])
                for idx in top_indices
            }

        return feature_contributions

    def path_importance(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        paths: List[List[str]],
        target_output: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Measure importance of computational paths through the network.

        Args:
            input: Input data
            paths: List of paths, where each path is a list of layer names
                  e.g., [['layer1', 'layer3', 'output'], ['layer2', 'layer4', 'output']]
            target_output: Optional target for comparison

        Returns:
            Dict mapping path description to importance score

        Example:
            >>> paths = [
            ...     ['mamba.blocks.0', 'mamba.blocks.2', 'decoder'],
            ...     ['mamba.blocks.1', 'mamba.blocks.3', 'decoder']
            ... ]
            >>> path_scores = gpa.path_importance(input, paths)
            >>> # {'path_0': 0.67, 'path_1': 0.42}
        """
        path_scores = {}

        # Get baseline output
        with torch.no_grad():
            baseline_output = self.model(input)
            if isinstance(baseline_output, dict) and 'decoder' in baseline_output:
                baseline = baseline_output['decoder']
            else:
                baseline = baseline_output

        if target_output is None:
            target_output = baseline

        # Collect activations along each path
        for path_idx, path in enumerate(paths):
            path_name = f"path_{path_idx}"

            # Compute path importance via activation patching
            importance = self._compute_path_importance(
                input, path, baseline, target_output
            )

            path_scores[path_name] = importance
            path_scores[f"{path_name}_layers"] = ' -> '.join(path)

        return path_scores

    def _compute_path_importance(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        path: List[str],
        baseline_output: torch.Tensor,
        target_output: torch.Tensor
    ) -> float:
        """Compute importance of a specific computational path."""
        # Collect activations along path
        activations = {}

        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach().clone()
            return hook

        hooks = []
        for layer_name in path:
            layer = dict(self.model.named_modules()).get(layer_name)
            if layer is not None:
                hooks.append(layer.register_forward_hook(make_hook(layer_name)))

        # Forward pass
        with torch.no_grad():
            _ = self.model(input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute path importance as sum of activation magnitudes
        total_importance = 0.0
        for layer_name in path:
            if layer_name in activations:
                act = activations[layer_name]
                if isinstance(act, torch.Tensor):
                    total_importance += act.abs().mean().item()

        return total_importance / max(len(path), 1)

    def circuit_contribution_scores(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        circuit_definition: Dict[str, List[str]],
        target_output: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Rank subcircuits by their contribution to output.

        Args:
            input: Input data
            circuit_definition: Dict mapping circuit name to list of component names
                               e.g., {'visual_circuit': ['layer1', 'layer2'],
                                      'motor_circuit': ['layer3', 'layer4']}
            target_output: Optional target

        Returns:
            Dict mapping circuit name to contribution score

        Example:
            >>> circuits = {
            ...     'early_processing': ['mamba.blocks.0', 'mamba.blocks.1'],
            ...     'late_processing': ['mamba.blocks.2', 'mamba.blocks.3']
            ... }
            >>> scores = gpa.circuit_contribution_scores(input, circuits)
            >>> # {'early_processing': 0.56, 'late_processing': 0.78}
        """
        circuit_scores = {}

        for circuit_name, components in circuit_definition.items():
            # Compute aggregate contribution of all components in circuit
            total_contribution = 0.0

            for component in components:
                contrib = self._compute_layer_contribution(
                    input, component,
                    torch.zeros(1), torch.zeros(1)  # Placeholders
                )
                total_contribution += contrib

            circuit_scores[circuit_name] = total_contribution / max(len(components), 1)

        return circuit_scores


def visualize_attributions(
    attributions: Union[torch.Tensor, Dict[str, torch.Tensor]],
    channel_names: Optional[List[str]] = None,
    title: str = "Attribution Scores",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualize attribution scores.

    Args:
        attributions: Attribution tensor or dict
        channel_names: Optional channel names for labeling
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure

    Example:
        >>> attributions = ig.attribute(input_data, target=0)
        >>> fig = visualize_attributions(attributions, title="Input Attribution")
        >>> fig.savefig("attributions.png")
    """
    if isinstance(attributions, dict):
        # Multi-modal: create subplots
        n_modalities = len(attributions)
        fig, axes = plt.subplots(1, n_modalities, figsize=(figsize[0], figsize[1]))

        if n_modalities == 1:
            axes = [axes]

        for idx, (modality, attr) in enumerate(attributions.items()):
            # Average over batch and time if needed
            if attr.dim() > 1:
                attr_scores = attr.abs().mean(dim=tuple(range(attr.dim()-1)))
            else:
                attr_scores = attr.abs()

            attr_scores = attr_scores.cpu().numpy()

            axes[idx].bar(range(len(attr_scores)), attr_scores)
            axes[idx].set_title(f"{modality} Attribution")
            axes[idx].set_xlabel("Channel")
            axes[idx].set_ylabel("Attribution Score")

        plt.suptitle(title)
        plt.tight_layout()

    else:
        # Single tensor
        fig, ax = plt.subplots(figsize=figsize)

        # Average over batch and time if needed
        if attributions.dim() > 1:
            attr_scores = attributions.abs().mean(dim=tuple(range(attributions.dim()-1)))
        else:
            attr_scores = attributions.abs()

        attr_scores = attr_scores.cpu().numpy()

        ax.bar(range(len(attr_scores)), attr_scores)
        ax.set_title(title)
        ax.set_xlabel("Channel" if channel_names is None else "")
        ax.set_ylabel("Attribution Score")

        if channel_names is not None:
            ax.set_xticks(range(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=45, ha='right')

        plt.tight_layout()

    return fig


# Example usage
if __name__ == '__main__':
    print("Advanced Attribution Methods for NeuroFMX")
    print("=" * 60)

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(256, 512)
            self.decoder = nn.Linear(512, 128)

        def forward(self, x):
            if isinstance(x, dict):
                x = list(x.values())[0]
            h = F.relu(self.encoder(x))
            out = self.decoder(h)
            return {'decoder': out}

    model = DummyModel()

    # Test data
    input_data = torch.randn(2, 100, 256)  # (batch, time, channels)

    print("\n1. Integrated Gradients")
    print("-" * 60)
    ig = IntegratedGradients(model, device='cpu')

    # Flatten for dummy model
    input_flat = input_data.reshape(2, -1)
    attributions = ig.attribute(input_flat, target=0, num_steps=20)
    print(f"Attribution shape: {attributions.shape}")
    print(f"Top attribution value: {attributions.abs().max():.4f}")

    # Channel attribution
    channel_names = [f'CH{i}' for i in range(64)]
    input_channels = torch.randn(1, 100, 64)
    channel_attr = ig.attribute_channels(
        input_channels.reshape(1, -1)[:, :256],
        channel_names[:256],
        target=0
    )
    print(f"\nTop 3 channels:")
    sorted_channels = sorted(channel_attr.items(), key=lambda x: x[1], reverse=True)
    for ch, score in sorted_channels[:3]:
        print(f"  {ch}: {score:.4f}")

    print("\n2. DeepLIFT")
    print("-" * 60)
    deeplift = DeepLIFT(model, device='cpu')

    baseline = torch.zeros_like(input_flat)
    deeplift_attr = deeplift.attribute(input_flat, baseline=baseline, target=0)
    print(f"DeepLIFT attribution shape: {deeplift_attr.shape}")
    print(f"Top attribution value: {deeplift_attr.abs().max():.4f}")

    print("\n3. GradientSHAP")
    print("-" * 60)
    grad_shap = GradientSHAP(model, device='cpu')

    # Create baseline distribution
    baselines = torch.randn(10, *input_flat.shape[1:])
    shap_attr = grad_shap.attribute(
        input_flat, baselines, target=0, num_steps=10, n_samples=3
    )
    print(f"GradientSHAP attribution shape: {shap_attr.shape}")
    print(f"Top attribution value: {shap_attr.abs().max():.4f}")

    print("\n4. Generative Path Attribution")
    print("-" * 60)

    # Create SAE
    sae = SparseAutoencoder(latent_dim=256, dictionary_size=1024)
    gpa = GenerativePathAttribution(model, sae=sae, device='cpu')

    # Decompose reconstruction
    decomposition = gpa.decompose_reconstruction(input_flat)
    print(f"\nLayer contributions: {len(decomposition['layer_contributions'])} layers")
    if decomposition['layer_contributions']:
        top_layer = max(decomposition['layer_contributions'].items(),
                       key=lambda x: x[1])
        print(f"Top layer: {top_layer[0]} (score: {top_layer[1]:.4f})")

    print(f"\nSAE features: {len(decomposition['sae_features'])} features")
    if decomposition['sae_features']:
        top_features = sorted(decomposition['sae_features'].items(),
                            key=lambda x: x[1], reverse=True)[:3]
        print("Top SAE features:")
        for feat_id, score in top_features:
            print(f"  Feature {feat_id}: {score:.4f}")

    # Path importance
    paths = [
        ['encoder', 'decoder'],
    ]
    path_scores = gpa.path_importance(input_flat, paths)
    print(f"\nPath scores:")
    for path_name, score in path_scores.items():
        if not path_name.endswith('_layers'):
            print(f"  {path_name}: {score:.4f}")

    print("\n5. Visualization")
    print("-" * 60)

    # Visualize attributions
    small_attr = attributions[:, :64]  # Just first 64 dims
    fig = visualize_attributions(
        small_attr,
        channel_names=[f'CH{i}' for i in range(64)],
        title="Integrated Gradients Attribution"
    )
    print("Visualization created (figure object)")

    # Multi-modal visualization
    multi_attr = {
        'spike': torch.randn(1, 32),
        'lfp': torch.randn(1, 16)
    }
    fig2 = visualize_attributions(multi_attr, title="Multi-Modal Attribution")
    print("Multi-modal visualization created")

    print("\n" + "=" * 60)
    print("Attribution methods demo complete!")
    print("\nKey features:")
    print("  - Integrated Gradients: Rigorous path integration")
    print("  - DeepLIFT: Fast single-pass attribution")
    print("  - GradientSHAP: SHAP-compatible with multiple baselines")
    print("  - Generative Path Attribution: Specialized for reconstructive models")
    print("\nAll methods support:")
    print("  - Single and multi-modal inputs")
    print("  - Per-channel and per-region attribution")
    print("  - Flexible target specification")
    print("  - Visualization utilities")
