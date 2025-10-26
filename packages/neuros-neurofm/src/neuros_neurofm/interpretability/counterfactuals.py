"""
Counterfactual Analysis: Latent Surgery and Do-Calculus Interventions
Perform targeted edits to latent states and measure causal effects
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


@dataclass
class CounterfactualResult:
    """Result of a counterfactual intervention"""
    intervention: Dict[str, Any]  # What was changed
    original_output: torch.Tensor
    counterfactual_output: torch.Tensor
    effect_size: float  # Magnitude of change
    metric_changes: Dict[str, float]  # Per-metric changes


class LatentSurgery:
    """
    Targeted edits to hidden states

    Allows precise interventions on specific latent dimensions/layers
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []

    def edit_latent(
        self,
        input_data: torch.Tensor,
        layer: str,
        edit_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply edit_fn to latent at specified layer

        Args:
            input_data: Model input
            layer: Layer name to intervene on
            edit_fn: Function to modify activations

        Returns:
            Model output with edited latent
        """
        intervention_applied = [False]  # Mutable flag

        def hook_fn(module, input, output):
            if intervention_applied[0]:
                return output

            # Apply edit
            if isinstance(output, tuple):
                edited = list(output)
                edited[0] = edit_fn(edited[0])
                intervention_applied[0] = True
                return tuple(edited)
            else:
                intervention_applied[0] = True
                return edit_fn(output)

        # Register hook
        target_module = dict(self.model.named_modules())[layer]
        handle = target_module.register_forward_hook(hook_fn)
        self.hooks.append(handle)

        # Forward pass
        with torch.no_grad():
            output = self.model(input_data)

        # Remove hook
        handle.remove()
        if handle in self.hooks:
            self.hooks.remove(handle)

        return output

    def swap_latent_dimension(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        layer: str,
        dims: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Swap specific dimensions between two latents

        Useful for disentanglement analysis

        Args:
            input1, input2: Two inputs
            layer: Layer to intervene on
            dims: Which dimensions to swap

        Returns:
            (output1_swapped, output2_swapped)
        """
        # Cache activations
        cached_activations = {}

        def cache_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    cached_activations[name] = output[0].clone()
                else:
                    cached_activations[name] = output.clone()
            return hook_fn

        # Get activations for both inputs
        target_module = dict(self.model.named_modules())[layer]

        # Input 1
        handle = target_module.register_forward_hook(cache_hook('input1'))
        with torch.no_grad():
            self.model(input1)
        handle.remove()

        # Input 2
        handle = target_module.register_forward_hook(cache_hook('input2'))
        with torch.no_grad():
            self.model(input2)
        handle.remove()

        # Swap dimensions
        latent1 = cached_activations['input1']
        latent2 = cached_activations['input2']

        latent1_swapped = latent1.clone()
        latent2_swapped = latent2.clone()

        for dim in dims:
            latent1_swapped[..., dim] = latent2[..., dim]
            latent2_swapped[..., dim] = latent1[..., dim]

        # Re-run with swapped latents
        def swap_hook1(module, input, output):
            return latent1_swapped

        def swap_hook2(module, input, output):
            return latent2_swapped

        # Output 1 with swap
        handle = target_module.register_forward_hook(swap_hook1)
        with torch.no_grad():
            output1_swapped = self.model(input1)
        handle.remove()

        # Output 2 with swap
        handle = target_module.register_forward_hook(swap_hook2)
        with torch.no_grad():
            output2_swapped = self.model(input2)
        handle.remove()

        return output1_swapped, output2_swapped

    def interpolate_latents(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        layer: str,
        num_steps: int = 10
    ) -> List[torch.Tensor]:
        """
        Spherical interpolation between latents

        Args:
            input1, input2: Start and end inputs
            layer: Layer to interpolate
            num_steps: Number of interpolation steps

        Returns:
            List of outputs along interpolation path
        """
        # Get latents
        cached_latents = {}

        def cache_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    cached_latents[name] = output[0].clone()
                else:
                    cached_latents[name] = output.clone()
            return hook_fn

        target_module = dict(self.model.named_modules())[layer]

        # Get latent1
        handle = target_module.register_forward_hook(cache_hook('latent1'))
        with torch.no_grad():
            self.model(input1)
        handle.remove()

        # Get latent2
        handle = target_module.register_forward_hook(cache_hook('latent2'))
        with torch.no_grad():
            self.model(input2)
        handle.remove()

        latent1 = cached_latents['latent1']
        latent2 = cached_latents['latent2']

        # Spherical interpolation (slerp)
        alphas = np.linspace(0, 1, num_steps)
        outputs = []

        for alpha in alphas:
            # Slerp
            interp_latent = self._slerp(latent1, latent2, alpha)

            # Generate output with interpolated latent
            def interp_hook(module, input, output):
                return interp_latent

            handle = target_module.register_forward_hook(interp_hook)
            with torch.no_grad():
                output = self.model(input1)  # Use input1 as base
            handle.remove()

            outputs.append(output)

        return outputs

    def _slerp(
        self,
        v0: torch.Tensor,
        v1: torch.Tensor,
        t: float,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Spherical linear interpolation

        Args:
            v0, v1: Start and end vectors
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated vector
        """
        # Normalize
        v0_norm = F.normalize(v0, dim=-1)
        v1_norm = F.normalize(v1, dim=-1)

        # Dot product
        dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True)

        # Clamp for numerical stability
        dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)

        # Angle
        theta = torch.acos(dot)

        # Slerp formula
        sin_theta = torch.sin(theta)
        s0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
        s1 = torch.sin(t * theta) / (sin_theta + eps)

        return s0 * v0 + s1 * v1


class DoCalculusInterventions:
    """
    Causal interventions using do-calculus

    Estimate P(Y | do(Z_k = z)) by setting latent Z_k to value z
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.surgery = LatentSurgery(model)

    def estimate_causal_effect(
        self,
        input_data: torch.Tensor,
        intervention: Dict[str, torch.Tensor],  # {layer: value}
        outcome_fn: Callable[[torch.Tensor], float]
    ) -> float:
        """
        Estimate P(Y | do(Z_k = z))

        Args:
            input_data: Input data
            intervention: Which layer to intervene on and what value
            outcome_fn: Function to compute outcome from model output

        Returns:
            Outcome under intervention
        """
        layer, value = list(intervention.items())[0]

        # Define edit function
        def set_to_value(latent):
            return value.expand_as(latent)

        # Apply intervention
        output = self.surgery.edit_latent(input_data, layer, set_to_value)

        # Compute outcome
        return outcome_fn(output)

    def causal_response_curve(
        self,
        input_data: torch.Tensor,
        layer: str,
        dim: int,
        values: np.ndarray,
        outcome_fn: Callable[[torch.Tensor], float]
    ) -> np.ndarray:
        """
        Sweep do(Z_k[dim] = v) for v in values

        Args:
            input_data: Input data
            layer: Layer to intervene on
            dim: Specific dimension to vary
            values: Range of values to test
            outcome_fn: Outcome function

        Returns:
            Array of outcomes Y(v)
        """
        outcomes = []

        for v in tqdm(values, desc="Computing causal response"):
            # Define edit function
            def set_dim(latent):
                edited = latent.clone()
                edited[..., dim] = v
                return edited

            # Apply intervention
            output = self.surgery.edit_latent(input_data, layer, set_dim)

            # Compute outcome
            outcome = outcome_fn(output)
            outcomes.append(outcome)

        return np.array(outcomes)

    def estimate_ate(
        self,
        input_data: torch.Tensor,
        layer: str,
        dim: int,
        treatment_value: float,
        control_value: float,
        outcome_fn: Callable[[torch.Tensor], float]
    ) -> float:
        """
        Estimate Average Treatment Effect (ATE)

        ATE = E[Y | do(Z=treatment)] - E[Y | do(Z=control)]

        Args:
            input_data: Input data (multiple samples)
            layer: Layer to intervene on
            dim: Dimension to intervene on
            treatment_value: Treatment value
            control_value: Control value
            outcome_fn: Outcome function

        Returns:
            ATE estimate
        """
        # Treatment outcomes
        def set_treatment(latent):
            edited = latent.clone()
            edited[..., dim] = treatment_value
            return edited

        output_treatment = self.surgery.edit_latent(input_data, layer, set_treatment)
        outcome_treatment = outcome_fn(output_treatment)

        # Control outcomes
        def set_control(latent):
            edited = latent.clone()
            edited[..., dim] = control_value
            return edited

        output_control = self.surgery.edit_latent(input_data, layer, set_control)
        outcome_control = outcome_fn(output_control)

        # ATE
        ate = outcome_treatment - outcome_control

        return float(ate)


class SyntheticLesions:
    """
    Knock-out heads/blocks/circuits

    Measure compensation and recovery
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def lesion_heads(
        self,
        layer: int,
        heads: List[int]
    ) -> nn.Module:
        """
        Zero-out attention heads or SSM blocks

        Args:
            layer: Layer index
            heads: Which heads to lesion

        Returns:
            Modified model
        """
        # Clone model
        lesioned_model = self._clone_model()

        # Find attention/SSM layers
        for name, module in lesioned_model.named_modules():
            if f"layers.{layer}" in name:
                if hasattr(module, "attn"):
                    # Attention layer
                    self._lesion_attention_heads(module.attn, heads)
                elif hasattr(module, "mamba_block"):
                    # SSM block
                    self._lesion_ssm_blocks(module.mamba_block, heads)

        return lesioned_model

    def _lesion_attention_heads(self, attn_module: nn.Module, heads: List[int]):
        """Zero out specific attention heads"""
        # Assuming standard MultiheadAttention
        if hasattr(attn_module, "out_proj"):
            # Zero out output projection weights for these heads
            embed_dim = attn_module.embed_dim
            num_heads = attn_module.num_heads
            head_dim = embed_dim // num_heads

            with torch.no_grad():
                for head_idx in heads:
                    start = head_idx * head_dim
                    end = (head_idx + 1) * head_dim
                    attn_module.out_proj.weight[:, start:end] = 0

    def _lesion_ssm_blocks(self, ssm_module: nn.Module, blocks: List[int]):
        """Zero out SSM blocks"""
        # Zero out output weights
        if hasattr(ssm_module, "out_proj"):
            with torch.no_grad():
                # Assuming block structure
                d_model = ssm_module.out_proj.weight.shape[0]
                block_size = d_model // len(blocks)

                for block_idx in blocks:
                    start = block_idx * block_size
                    end = (block_idx + 1) * block_size
                    ssm_module.out_proj.weight[start:end] = 0

    def _clone_model(self) -> nn.Module:
        """Create a copy of the model"""
        import copy
        return copy.deepcopy(self.model)

    def measure_compensation(
        self,
        original_model: nn.Module,
        lesioned_model: nn.Module,
        data: torch.utils.data.DataLoader,
        metric: Callable[[torch.Tensor, torch.Tensor], float],
        finetune_steps: int = 100
    ) -> Dict[str, float]:
        """
        Measure how network compensates for lesion

        Args:
            original_model: Original model
            lesioned_model: Model with lesion
            data: Evaluation data
            metric: Performance metric
            finetune_steps: Steps to finetune after lesion

        Returns:
            Dictionary with compensation metrics
        """
        # Baseline performance (original model)
        baseline_perf = self._evaluate(original_model, data, metric)

        # Immediate post-lesion performance
        immediate_perf = self._evaluate(lesioned_model, data, metric)

        # Finetune lesioned model
        finetuned_model = self._finetune(lesioned_model, data, finetune_steps)

        # Post-finetune performance
        recovered_perf = self._evaluate(finetuned_model, data, metric)

        # Compute metrics
        immediate_drop = baseline_perf - immediate_perf
        recovery = recovered_perf - immediate_perf
        compensation_score = recovery / (immediate_drop + 1e-8)

        return {
            'baseline': float(baseline_perf),
            'immediate_drop': float(immediate_drop),
            'immediate_performance': float(immediate_perf),
            'recovered_performance': float(recovered_perf),
            'recovery_amount': float(recovery),
            'compensation_score': float(compensation_score)
        }

    def _evaluate(
        self,
        model: nn.Module,
        data: torch.utils.data.DataLoader,
        metric: Callable
    ) -> float:
        """Evaluate model on data"""
        model.eval()
        total_metric = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, batch

                outputs = model(inputs)
                total_metric += metric(outputs, targets)
                num_batches += 1

        return total_metric / num_batches

    def _finetune(
        self,
        model: nn.Module,
        data: torch.utils.data.DataLoader,
        steps: int
    ) -> nn.Module:
        """Finetune model for a few steps"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for step in range(steps):
            for batch in data:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, batch

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                if step >= steps:
                    break

        return model


# Example usage
if __name__ == "__main__":
    print("Counterfactual Analysis")
    print("=" * 80)

    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    # Latent surgery
    surgery = LatentSurgery(model)

    # Test data
    x = torch.randn(4, 10)

    # Edit latent at layer 0
    def zero_first_dim(latent):
        edited = latent.clone()
        edited[:, 0] = 0
        return edited

    output_edited = surgery.edit_latent(x, "0", zero_first_dim)

    print(f"\nLatent surgery applied:")
    print(f"  Output shape: {output_edited.shape}")

    # Do-calculus
    do_calc = DoCalculusInterventions(model)

    def outcome_fn(output):
        return output.mean().item()

    # Causal response curve
    values = np.linspace(-2, 2, 20)
    response = do_calc.causal_response_curve(x, "0", dim=0, values=values, outcome_fn=outcome_fn)

    print(f"\nCausal response curve computed:")
    print(f"  Values tested: {len(values)}")
    print(f"  Response range: [{response.min():.3f}, {response.max():.3f}]")
