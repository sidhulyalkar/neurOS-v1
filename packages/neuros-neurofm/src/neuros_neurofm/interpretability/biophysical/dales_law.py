"""
Dale's Law Enforcement for Neural Networks

Constrain networks to maintain biological plausibility with excitatory/inhibitory separation.
Dale's principle: A neuron releases the same neurotransmitter at all synapses,
so all outgoing connections must have the same sign (all positive or all negative).

This module provides layers and constraints to enforce Dale's law in neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DalesLawConstraint:
    """
    Enforce excitatory/inhibitory separation in weight matrices.

    Excitatory neurons: all outgoing weights ≥ 0
    Inhibitory neurons: all outgoing weights ≤ 0

    Args:
        n_neurons: Total number of neurons
        ei_ratio: Fraction of excitatory neurons (default: 0.8 → 80% E, 20% I)

    Example:
        >>> constraint = DalesLawConstraint(n_neurons=100, ei_ratio=0.8)
        >>> # During training:
        >>> constraint.apply(model.weight)
    """

    def __init__(self, n_neurons: int, ei_ratio: float = 0.8):
        if not (0 < ei_ratio < 1):
            raise ValueError("ei_ratio must be between 0 and 1")

        self.n_neurons = n_neurons
        self.ei_ratio = ei_ratio
        self.n_exc = int(n_neurons * ei_ratio)
        self.n_inh = n_neurons - self.n_exc

        logger.info(f"Dale's law: {self.n_exc} excitatory, {self.n_inh} inhibitory neurons")

    def apply(self, weight_matrix: nn.Parameter):
        """
        Apply Dale's law constraint to weight matrix in-place.

        Args:
            weight_matrix: Weight matrix [n_out, n_in] where n_in = n_neurons
                          Each row corresponds to outgoing weights from one neuron
        """
        with torch.no_grad():
            # Excitatory neurons: rows 0 to n_exc-1
            # Clamp outgoing weights to be non-negative
            weight_matrix[:self.n_exc, :] = weight_matrix[:self.n_exc, :].clamp(min=0)

            # Inhibitory neurons: rows n_exc to end
            # Clamp outgoing weights to be non-positive
            weight_matrix[self.n_exc:, :] = weight_matrix[self.n_exc:, :].clamp(max=0)

    def get_neuron_types(self) -> Tensor:
        """
        Get boolean tensor indicating neuron types.

        Returns:
            Boolean tensor [n_neurons] where True = excitatory, False = inhibitory
        """
        neuron_types = torch.zeros(self.n_neurons, dtype=torch.bool)
        neuron_types[:self.n_exc] = True
        return neuron_types


class DalesLinear(nn.Module):
    """
    Linear layer with Dale's law constraint.

    Automatically enforces E/I separation during forward pass.

    Args:
        in_features: Number of input features (must equal number of neurons for Dale's law)
        out_features: Number of output features
        ei_ratio: Fraction of excitatory neurons
        bias: Whether to include bias term
        device: Torch device

    Example:
        >>> layer = DalesLinear(in_features=100, out_features=50, ei_ratio=0.8)
        >>> output = layer(input)
        >>> # Weights automatically constrained
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ei_ratio: float = 0.8,
        bias: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ei_ratio = ei_ratio

        # Weight matrix
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Dale's law constraint
        self.constraint = DalesLawConstraint(in_features, ei_ratio)

        # Initialize weights
        self._initialize_weights()

        if device:
            self.to(device)

    def _initialize_weights(self):
        """Initialize weights with correct signs for Dale's law."""
        with torch.no_grad():
            # Initialize with Xavier and apply Dale's law
            nn.init.xavier_uniform_(self.weight)
            self.constraint.apply(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with Dale's law enforcement.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Apply Dale's law constraint
        self.constraint.apply(self.weight)

        # Linear transformation
        output = x @ self.weight

        if self.bias is not None:
            output = output + self.bias

        return output


class EINetworkClassifier(nn.Module):
    """
    Multi-layer classifier with Dale's law in hidden layers.

    Creates a biologically plausible neural network where intermediate
    layers maintain E/I separation.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer sizes
        output_dim: Output dimension
        ei_ratio: E/I ratio for hidden layers
        activation: Activation function
        device: Torch device

    Example:
        >>> classifier = EINetworkClassifier(
        ...     input_dim=784,
        ...     hidden_dims=[200, 100],
        ...     output_dim=10,
        ...     ei_ratio=0.8
        ... )
        >>> logits = classifier(images)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        ei_ratio: float = 0.8,
        activation: str = 'relu',
        device: Optional[str] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.ei_ratio = ei_ratio

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []

        # Input layer (no Dale's law)
        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers with Dale's law
        for i in range(len(hidden_dims) - 1):
            layers.append(DalesLinear(
                in_features=hidden_dims[i],
                out_features=hidden_dims[i + 1],
                ei_ratio=ei_ratio,
            ))

        # Output layer (no Dale's law - allows both positive and negative outputs)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.layers = nn.ModuleList(layers)

        if device:
            self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through E/I network.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Output logits [batch, output_dim]
        """
        # First layer
        x = self.activation(self.layers[0](x))

        # Hidden layers with Dale's law
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))

        # Output layer
        x = self.layers[-1](x)

        return x

    def get_ei_statistics(self) -> dict:
        """
        Get statistics about E/I neurons in the network.

        Returns:
            Dictionary with E/I statistics for each Dale's law layer
        """
        stats = {}

        for i, layer in enumerate(self.layers):
            if isinstance(layer, DalesLinear):
                with torch.no_grad():
                    neuron_types = layer.constraint.get_neuron_types()

                    # Average weights
                    exc_weights = layer.weight[:layer.constraint.n_exc, :].abs().mean().item()
                    inh_weights = layer.weight[layer.constraint.n_exc:, :].abs().mean().item()

                    stats[f'layer_{i}'] = {
                        'n_excitatory': layer.constraint.n_exc,
                        'n_inhibitory': layer.constraint.n_inh,
                        'ei_ratio': layer.constraint.ei_ratio,
                        'avg_exc_weight': exc_weights,
                        'avg_inh_weight': inh_weights,
                    }

        return stats


class RecurrentDalesNetwork(nn.Module):
    """
    Recurrent network with Dale's law constraints.

    RNN with explicit E/I neuron separation for modeling
    biologically plausible dynamics.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden state dimension
        output_dim: Output dimension
        ei_ratio: E/I ratio
        device: Torch device

    Example:
        >>> rnn = RecurrentDalesNetwork(
        ...     input_dim=10,
        ...     hidden_dim=100,
        ...     output_dim=5,
        ...     ei_ratio=0.8
        ... )
        >>> outputs, hidden = rnn(inputs, seq_len=50)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        ei_ratio: float = 0.8,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ei_ratio = ei_ratio

        # Input weights
        self.W_in = nn.Linear(input_dim, hidden_dim)

        # Recurrent weights with Dale's law
        self.W_rec = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

        # Output weights
        self.W_out = nn.Linear(hidden_dim, output_dim)

        # Dale's law constraint for recurrent weights
        self.dales_constraint = DalesLawConstraint(hidden_dim, ei_ratio)

        if device:
            self.to(device)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through recurrent network.

        Args:
            x: Input sequence [batch, seq_len, input_dim]
            hidden: Initial hidden state [batch, hidden_dim]

        Returns:
            outputs: Output sequence [batch, seq_len, output_dim]
            hidden: Final hidden state [batch, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Apply Dale's law to recurrent weights
        self.dales_constraint.apply(self.W_rec)

        outputs = []

        for t in range(seq_len):
            # RNN update
            input_contrib = self.W_in(x[:, t, :])
            recurrent_contrib = hidden @ self.W_rec.T

            hidden = torch.tanh(input_contrib + recurrent_contrib)

            # Output
            output = self.W_out(hidden)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden


class DalesLossRegularizer(nn.Module):
    """
    Soft regularization to encourage Dale's law compliance.

    Instead of hard constraints, adds a penalty for violations.
    Useful when exact Dale's law is not required but encouraged.

    Args:
        weight: Regularization weight
        ei_ratio: Target E/I ratio

    Example:
        >>> regularizer = DalesLossRegularizer(weight=0.01, ei_ratio=0.8)
        >>> loss = base_loss + regularizer(model.recurrent_weights, neuron_types)
    """

    def __init__(self, weight: float = 0.01, ei_ratio: float = 0.8):
        super().__init__()
        self.weight = weight
        self.ei_ratio = ei_ratio

    def forward(self, weight_matrix: Tensor, neuron_types: Tensor) -> Tensor:
        """
        Compute Dale's law violation penalty.

        Args:
            weight_matrix: Weight matrix [n_neurons, n_neurons]
            neuron_types: Boolean tensor [n_neurons] (True=E, False=I)

        Returns:
            Regularization loss scalar
        """
        # Excitatory neurons should have positive outgoing weights
        exc_violation = F.relu(-weight_matrix[neuron_types, :]).sum()

        # Inhibitory neurons should have negative outgoing weights
        inh_violation = F.relu(weight_matrix[~neuron_types, :]).sum()

        total_violation = exc_violation + inh_violation

        return self.weight * total_violation


def analyze_dale_compliance(weight_matrix: Tensor, neuron_types: Tensor) -> dict:
    """
    Analyze how well a weight matrix complies with Dale's law.

    Args:
        weight_matrix: Weight matrix [n_neurons, ...]
        neuron_types: Boolean tensor [n_neurons] (True=E, False=I)

    Returns:
        Dictionary with compliance statistics
    """
    with torch.no_grad():
        # Excitatory compliance
        exc_weights = weight_matrix[neuron_types, :]
        exc_positive = (exc_weights >= 0).float().mean().item()

        # Inhibitory compliance
        inh_weights = weight_matrix[~neuron_types, :]
        inh_negative = (inh_weights <= 0).float().mean().item()

        # Overall compliance
        total_compliance = (exc_positive * neuron_types.sum() + inh_negative * (~neuron_types).sum()) / len(neuron_types)

        # Average magnitudes
        exc_magnitude = exc_weights.abs().mean().item()
        inh_magnitude = inh_weights.abs().mean().item()

        return {
            'excitatory_compliance': exc_positive,
            'inhibitory_compliance': inh_negative,
            'overall_compliance': total_compliance.item(),
            'avg_exc_magnitude': exc_magnitude,
            'avg_inh_magnitude': inh_magnitude,
            'ei_balance': exc_magnitude / (inh_magnitude + 1e-8),
        }
