"""
Feature Visualization for Neural Circuits

Optimal input synthesis and activation maximization to understand what
neurons and circuits compute. Generates interpretable visualizations of
learned features.

Based on Olah et al. (2018) and optimization-based visualization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureVisualizer:
    """
    Visualize what neurons and layers have learned via gradient-based optimization.

    Synthesizes optimal inputs that maximally activate specific neurons or
    feature directions, revealing what the model has learned to detect.

    Args:
        model: Neural network to visualize
        device: Torch device

    Example:
        >>> visualizer = FeatureVisualizer(model)
        >>> optimal_input = visualizer.visualize_neuron(
        ...     layer_name='layer_6',
        ...     neuron_idx=42,
        ...     n_steps=500
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Store activations during forward pass
        self.activations = {}
        self.hooks = []

    def register_hooks(self, layer_names: List[str]):
        """Register forward hooks to capture activations."""
        self.activations = {name: None for name in layer_names}

        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        # Register hooks
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def visualize_neuron(
        self,
        layer_name: str,
        neuron_idx: int,
        input_shape: Tuple[int, ...],
        n_steps: int = 500,
        learning_rate: float = 0.05,
        regularization: Dict[str, float] = None,
    ) -> Tensor:
        """
        Generate optimal input that maximally activates a specific neuron.

        Args:
            layer_name: Name of layer containing the neuron
            neuron_idx: Index of neuron to visualize
            input_shape: Shape of input to optimize (excluding batch dimension)
            n_steps: Number of optimization steps
            learning_rate: Learning rate for optimization
            regularization: Dictionary of regularization weights
                          {'l2': 0.01, 'tv': 0.001, 'blur': 0.0}

        Returns:
            Optimal input [1, *input_shape]
        """
        if regularization is None:
            regularization = {'l2': 0.01, 'tv': 0.001, 'blur': 0.0}

        # Register hook for target layer
        self.register_hooks([layer_name])

        # Initialize input (random noise or zeros)
        optimal_input = torch.randn(1, *input_shape, device=self.device) * 0.1
        optimal_input.requires_grad = True

        # Optimizer
        optimizer = torch.optim.Adam([optimal_input], lr=learning_rate)

        for step in range(n_steps):
            optimizer.zero_grad()

            # Forward pass
            _ = self.model(optimal_input)

            # Get activation of target neuron
            activation = self.activations[layer_name]

            # Handle different activation shapes
            if activation.dim() == 3:  # [batch, seq, features]
                neuron_activation = activation[0, :, neuron_idx].mean()
            elif activation.dim() == 2:  # [batch, features]
                neuron_activation = activation[0, neuron_idx]
            else:
                raise ValueError(f"Unexpected activation shape: {activation.shape}")

            # Objective: maximize neuron activation
            loss = -neuron_activation

            # Regularization
            if regularization.get('l2', 0) > 0:
                loss += regularization['l2'] * (optimal_input ** 2).mean()

            if regularization.get('tv', 0) > 0:
                # Total variation (encourages smoothness)
                tv_loss = self._total_variation(optimal_input)
                loss += regularization['tv'] * tv_loss

            if regularization.get('blur', 0) > 0:
                # Blur regularization
                blurred = self._blur(optimal_input)
                loss += regularization['blur'] * ((optimal_input - blurred) ** 2).mean()

            # Backward pass
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                logger.info(f"Step {step}/{n_steps} - Activation: {-loss.item():.4f}")

        self.remove_hooks()

        return optimal_input.detach()

    def visualize_direction(
        self,
        layer_name: str,
        direction: Tensor,
        input_shape: Tuple[int, ...],
        n_steps: int = 500,
        learning_rate: float = 0.05,
    ) -> Tensor:
        """
        Visualize what activates a specific direction in feature space.

        Args:
            layer_name: Layer to visualize
            direction: Direction vector in feature space [n_features]
            input_shape: Shape of input
            n_steps: Optimization steps
            learning_rate: Learning rate

        Returns:
            Optimal input
        """
        direction = direction.to(self.device)
        direction = direction / (direction.norm() + 1e-8)

        self.register_hooks([layer_name])

        optimal_input = torch.randn(1, *input_shape, device=self.device) * 0.1
        optimal_input.requires_grad = True

        optimizer = torch.optim.Adam([optimal_input], lr=learning_rate)

        for step in range(n_steps):
            optimizer.zero_grad()

            _ = self.model(optimal_input)
            activation = self.activations[layer_name]

            # Project activation onto direction
            if activation.dim() == 3:
                activation_flat = activation[0].mean(dim=0)  # Average over time
            else:
                activation_flat = activation[0]

            projection = (activation_flat * direction).sum()

            # Maximize projection
            loss = -projection + 0.01 * (optimal_input ** 2).mean()

            loss.backward()
            optimizer.step()

        self.remove_hooks()

        return optimal_input.detach()

    def _total_variation(self, x: Tensor) -> Tensor:
        """Compute total variation for smoothness regularization."""
        if x.dim() == 3:  # [batch, seq, features]
            tv = ((x[:, 1:, :] - x[:, :-1, :]) ** 2).mean()
        else:
            tv = torch.tensor(0.0, device=x.device)
        return tv

    def _blur(self, x: Tensor, kernel_size: int = 3) -> Tensor:
        """Apply Gaussian blur."""
        # Simple box blur approximation
        if x.dim() == 3:
            kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
            x_padded = F.pad(x.unsqueeze(1), (kernel_size // 2, kernel_size // 2), mode='reflect')
            blurred = F.conv1d(x_padded, kernel).squeeze(1)
            return blurred
        return x


class OptimalStimulus:
    """
    Find optimal stimuli for neurons using constrained optimization.

    Discovers what input patterns maximally drive specific neural responses,
    with biological constraints (e.g., naturalistic statistics).

    Args:
        model: Neural network
        constraint_type: Type of constraint ('none', 'naturalistic', 'sparse')
        device: Torch device

    Example:
        >>> optimizer = OptimalStimulus(model, constraint_type='naturalistic')
        >>> stimulus = optimizer.find_optimal(
        ...     layer_name='encoder_layer_3',
        ...     neuron_idx=10
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        constraint_type: str = 'naturalistic',
        device: Optional[str] = None,
    ):
        self.model = model
        self.constraint_type = constraint_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def find_optimal(
        self,
        layer_name: str,
        neuron_idx: int,
        input_shape: Tuple[int, ...],
        n_iterations: int = 1000,
        constraint_strength: float = 0.1,
    ) -> Tensor:
        """
        Find optimal stimulus with constraints.

        Args:
            layer_name: Target layer
            neuron_idx: Target neuron index
            input_shape: Input shape
            n_iterations: Number of optimization iterations
            constraint_strength: Strength of constraint

        Returns:
            Optimal stimulus
        """
        # Initialize
        stimulus = torch.randn(1, *input_shape, device=self.device) * 0.1
        stimulus.requires_grad = True

        optimizer = torch.optim.Adam([stimulus], lr=0.02)

        # Register hook
        activation_value = {}

        def hook(module, input, output):
            activation_value['activation'] = output

        # Find module
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Layer {layer_name} not found")

        handle = target_module.register_forward_hook(hook)

        for iteration in range(n_iterations):
            optimizer.zero_grad()

            # Forward pass
            _ = self.model(stimulus)

            # Get activation
            activation = activation_value['activation']
            if activation.dim() == 3:
                neuron_response = activation[0, :, neuron_idx].mean()
            else:
                neuron_response = activation[0, neuron_idx]

            # Objective
            loss = -neuron_response

            # Add constraint
            if self.constraint_type == 'naturalistic':
                # 1/f spectrum constraint (naturalistic images/signals have 1/f spectrum)
                loss += constraint_strength * self._spectral_constraint(stimulus)
            elif self.constraint_type == 'sparse':
                # Sparsity constraint
                loss += constraint_strength * stimulus.abs().mean()

            loss.backward()
            optimizer.step()

            # Project back to valid range (optional)
            with torch.no_grad():
                stimulus.clamp_(-3, 3)

        handle.remove()

        return stimulus.detach()

    def _spectral_constraint(self, x: Tensor) -> Tensor:
        """Penalize deviation from 1/f spectrum (naturalistic)."""
        # FFT
        x_fft = torch.fft.rfft(x, dim=1)
        power_spectrum = x_fft.abs() ** 2

        # Expected 1/f: power ~ 1/f
        freqs = torch.fft.rfftfreq(x.size(1), device=x.device)
        expected_power = 1.0 / (freqs + 0.1)  # Add offset to avoid division by zero

        # Loss: deviation from expected
        loss = ((power_spectrum.mean(dim=0) - expected_power) ** 2).mean()

        return loss


class ActivationMaximization:
    """
    Activation maximization with multiple objectives and constraints.

    Advanced feature visualization supporting:
        - Multi-objective optimization (activate multiple neurons)
        - Diversity regularization (find multiple diverse optima)
        - Transformation robustness (rotation, translation invariance)

    Args:
        model: Neural network
        device: Torch device

    Example:
        >>> maximizer = ActivationMaximization(model)
        >>> diverse_stimuli = maximizer.find_diverse_optima(
        ...     layer_name='layer_8',
        ...     neuron_idx=15,
        ...     n_optima=5
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def maximize_activation(
        self,
        layer_name: str,
        objective_fn: Callable[[Tensor], Tensor],
        input_shape: Tuple[int, ...],
        n_steps: int = 500,
        learning_rate: float = 0.05,
        diversity_term: Optional[List[Tensor]] = None,
        diversity_weight: float = 0.1,
    ) -> Tensor:
        """
        Maximize activation with custom objective function.

        Args:
            layer_name: Target layer
            objective_fn: Function mapping activation tensor to scalar objective
            input_shape: Input shape
            n_steps: Optimization steps
            learning_rate: Learning rate
            diversity_term: Previous optima for diversity regularization
            diversity_weight: Weight for diversity penalty

        Returns:
            Optimal input
        """
        # Register hook
        activation_store = {}

        def hook(module, input, output):
            activation_store['activation'] = output

        target_module = dict(self.model.named_modules())[layer_name]
        handle = target_module.register_forward_hook(hook)

        # Initialize
        optimal_input = torch.randn(1, *input_shape, device=self.device) * 0.1
        optimal_input.requires_grad = True

        optimizer = torch.optim.Adam([optimal_input], lr=learning_rate)

        for step in range(n_steps):
            optimizer.zero_grad()

            # Forward
            _ = self.model(optimal_input)
            activation = activation_store['activation']

            # Objective
            objective = objective_fn(activation)
            loss = -objective

            # Diversity regularization
            if diversity_term is not None and len(diversity_term) > 0:
                for prev_input in diversity_term:
                    similarity = F.cosine_similarity(
                        optimal_input.flatten(),
                        prev_input.flatten(),
                        dim=0
                    )
                    loss -= diversity_weight * similarity  # Penalize similarity

            # Regularization
            loss += 0.01 * (optimal_input ** 2).mean()

            loss.backward()
            optimizer.step()

        handle.remove()

        return optimal_input.detach()

    def find_diverse_optima(
        self,
        layer_name: str,
        neuron_idx: int,
        input_shape: Tuple[int, ...],
        n_optima: int = 5,
        n_steps: int = 500,
    ) -> List[Tensor]:
        """
        Find multiple diverse inputs that activate the same neuron.

        Reveals different "modes" of activation for a neuron.

        Args:
            layer_name: Target layer
            neuron_idx: Target neuron
            input_shape: Input shape
            n_optima: Number of diverse optima to find
            n_steps: Optimization steps per optimum

        Returns:
            List of optimal inputs
        """
        def objective_fn(activation):
            if activation.dim() == 3:
                return activation[0, :, neuron_idx].mean()
            else:
                return activation[0, neuron_idx]

        optima = []

        for i in range(n_optima):
            logger.info(f"Finding optimum {i+1}/{n_optima}")

            # Find next optimum with diversity from previous ones
            optimum = self.maximize_activation(
                layer_name=layer_name,
                objective_fn=objective_fn,
                input_shape=input_shape,
                n_steps=n_steps,
                diversity_term=optima,
                diversity_weight=0.2,
            )

            optima.append(optimum)

        return optima

    def transformation_robustness(
        self,
        layer_name: str,
        neuron_idx: int,
        input_shape: Tuple[int, ...],
        transformations: List[str] = None,
        n_steps: int = 500,
    ) -> Tensor:
        """
        Find stimulus that robustly activates neuron under transformations.

        Tests for translation/rotation invariance.

        Args:
            layer_name: Target layer
            neuron_idx: Target neuron
            input_shape: Input shape
            transformations: List of transformations to apply
                           ['shift', 'scale', 'noise']
            n_steps: Optimization steps

        Returns:
            Robust optimal input
        """
        if transformations is None:
            transformations = ['shift', 'noise']

        # Register hook
        activation_store = {}

        def hook(module, input, output):
            activation_store['activation'] = output

        target_module = dict(self.model.named_modules())[layer_name]
        handle = target_module.register_forward_hook(hook)

        # Initialize
        optimal_input = torch.randn(1, *input_shape, device=self.device) * 0.1
        optimal_input.requires_grad = True

        optimizer = torch.optim.Adam([optimal_input], lr=0.05)

        for step in range(n_steps):
            optimizer.zero_grad()

            total_activation = 0.0
            n_transforms = len(transformations) + 1  # Include original

            # Original
            _ = self.model(optimal_input)
            activation = activation_store['activation']
            if activation.dim() == 3:
                total_activation += activation[0, :, neuron_idx].mean()
            else:
                total_activation += activation[0, neuron_idx]

            # Transformed versions
            for transform in transformations:
                if transform == 'shift':
                    # Temporal shift
                    shifted = torch.roll(optimal_input, shifts=5, dims=1)
                    _ = self.model(shifted)
                elif transform == 'noise':
                    # Add noise
                    noisy = optimal_input + torch.randn_like(optimal_input) * 0.1
                    _ = self.model(noisy)
                elif transform == 'scale':
                    # Scale
                    scaled = optimal_input * torch.randn(1, device=self.device).abs()
                    _ = self.model(scaled)

                activation = activation_store['activation']
                if activation.dim() == 3:
                    total_activation += activation[0, :, neuron_idx].mean()
                else:
                    total_activation += activation[0, neuron_idx]

            # Average activation across transformations
            avg_activation = total_activation / n_transforms

            # Maximize
            loss = -avg_activation + 0.01 * (optimal_input ** 2).mean()

            loss.backward()
            optimizer.step()

        handle.remove()

        return optimal_input.detach()
