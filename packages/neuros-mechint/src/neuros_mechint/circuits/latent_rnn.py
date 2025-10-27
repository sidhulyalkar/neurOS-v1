"""
Latent Circuit Inference from High-Dimensional Neural Representations

Extract interpretable low-dimensional RNN circuits from learned representations.
Based on Langdon & Engel (2025) and Sussillo & Barak (2013).

This module enables extracting minimal computational circuits that explain
complex neural dynamics, providing deep interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LatentCircuitModel(nn.Module):
    """
    Low-dimensional RNN that explains high-dimensional neural responses.

    Fits a minimal RNN circuit to observed neural activity, revealing the
    underlying computational structure. The circuit has far fewer units than
    the observed responses, forcing extraction of core computations.

    Args:
        n_latent: Number of latent RNN units (typically 5-20)
        n_observed: Number of observed neural responses
        enforce_dales: Whether to enforce excitatory/inhibitory separation
        ei_ratio: Fraction of excitatory neurons (default: 0.8)
        nonlinearity: RNN nonlinearity ('tanh', 'relu', 'none')
        device: Torch device

    Example:
        >>> circuit = LatentCircuitModel(n_latent=10, n_observed=100)
        >>> outputs, hidden = circuit(inputs)
        >>> connectivity = circuit.extract_circuit()
    """

    def __init__(
        self,
        n_latent: int,
        n_observed: int,
        enforce_dales: bool = True,
        ei_ratio: float = 0.8,
        nonlinearity: str = 'tanh',
        device: Optional[str] = None,
    ):
        super().__init__()

        self.n_latent = n_latent
        self.n_observed = n_observed
        self.enforce_dales = enforce_dales
        self.ei_ratio = ei_ratio
        self.nonlinearity = nonlinearity
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Latent RNN weights
        self.W_rec = nn.Parameter(torch.randn(n_latent, n_latent) * 0.1)
        self.W_in = nn.Parameter(torch.randn(n_latent, n_latent) * 0.1)  # Assuming input dim = n_latent for now
        self.W_out = nn.Parameter(torch.randn(n_observed, n_latent) * 0.1)

        # Biases
        self.b_rec = nn.Parameter(torch.zeros(n_latent))
        self.b_out = nn.Parameter(torch.zeros(n_observed))

        # E/I neuron mask if Dale's law is enforced
        if enforce_dales:
            n_exc = int(n_latent * ei_ratio)
            self.register_buffer('ei_mask', self._create_ei_mask(n_exc, n_latent))
            self.register_buffer('neuron_types', torch.cat([
                torch.ones(n_exc, dtype=torch.bool),
                torch.zeros(n_latent - n_exc, dtype=torch.bool)
            ]))
        else:
            self.register_buffer('ei_mask', None)
            self.register_buffer('neuron_types', None)

        self.to(self.device)

    def _create_ei_mask(self, n_exc: int, n_total: int) -> Tensor:
        """
        Create mask for Dale's law enforcement.

        Excitatory neurons (first n_exc): all outgoing weights >= 0
        Inhibitory neurons (remaining): all outgoing weights <= 0
        """
        mask = torch.ones(n_total, n_total)
        # Sign for each row based on neuron type
        for i in range(n_total):
            if i < n_exc:
                # Excitatory: positive weights
                mask[i, :] = 1.0
            else:
                # Inhibitory: negative weights
                mask[i, :] = -1.0
        return mask

    def _apply_dales_law(self):
        """Apply Dale's law constraints to recurrent weights."""
        if self.enforce_dales and self.ei_mask is not None:
            with torch.no_grad():
                # Take absolute value and multiply by sign mask
                self.W_rec.data = self.W_rec.data.abs() * self.ei_mask

    def _apply_nonlinearity(self, x: Tensor) -> Tensor:
        """Apply RNN nonlinearity."""
        if self.nonlinearity == 'tanh':
            return torch.tanh(x)
        elif self.nonlinearity == 'relu':
            return F.relu(x)
        elif self.nonlinearity == 'none':
            return x
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")

    def forward(
        self,
        inputs: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through latent RNN.

        Args:
            inputs: Input sequence [batch, seq_len, input_dim]
            hidden: Initial hidden state [batch, n_latent] (default: zeros)

        Returns:
            outputs: Reconstructed observations [batch, seq_len, n_observed]
            hidden_states: Latent states over time [batch, seq_len, n_latent]
        """
        batch_size, seq_len, _ = inputs.shape

        # Initialize hidden state
        if hidden is None:
            hidden = torch.zeros(batch_size, self.n_latent, device=self.device)

        # Apply Dale's law constraints
        if self.enforce_dales:
            self._apply_dales_law()

        # Collect hidden states and outputs
        hidden_states = []
        outputs = []

        for t in range(seq_len):
            # RNN update: h_t = f(W_rec @ h_{t-1} + W_in @ u_t + b_rec)
            inp_t = inputs[:, t, :self.n_latent]  # Take first n_latent dims
            pre_activation = (
                hidden @ self.W_rec.T
                + inp_t @ self.W_in.T
                + self.b_rec
            )
            hidden = self._apply_nonlinearity(pre_activation)
            hidden_states.append(hidden)

            # Readout: x_t = W_out @ h_t + b_out
            output = hidden @ self.W_out.T + self.b_out
            outputs.append(output)

        hidden_states = torch.stack(hidden_states, dim=1)  # [batch, seq_len, n_latent]
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, n_observed]

        return outputs, hidden_states

    def extract_circuit(self) -> Dict[str, Any]:
        """
        Extract interpretable circuit structure.

        Returns:
            Dictionary with:
                - W_rec: Recurrent weight matrix [n_latent, n_latent]
                - W_in: Input weights
                - W_out: Output weights
                - neuron_types: Excitatory (True) / Inhibitory (False)
                - eigenvalues: Eigenvalues of W_rec (stability analysis)
                - connection_strength: Effective connectivity strength
        """
        W_rec = self.W_rec.detach().cpu()

        # Compute eigenvalues for stability analysis
        eigenvalues = torch.linalg.eigvals(W_rec)

        # Connection strength matrix (absolute values)
        connection_strength = W_rec.abs()

        result = {
            'W_rec': W_rec,
            'W_in': self.W_in.detach().cpu(),
            'W_out': self.W_out.detach().cpu(),
            'eigenvalues': eigenvalues,
            'connection_strength': connection_strength,
        }

        if self.neuron_types is not None:
            result['neuron_types'] = self.neuron_types.cpu()
            result['n_excitatory'] = self.neuron_types.sum().item()
            result['n_inhibitory'] = (~self.neuron_types).sum().item()

        return result

    def get_effective_connectivity(self, threshold: float = 0.1) -> Tensor:
        """
        Get effective connectivity graph (thresholded weights).

        Args:
            threshold: Minimum absolute weight to consider connected

        Returns:
            Binary connectivity matrix [n_latent, n_latent]
        """
        W = self.W_rec.detach().abs()
        return (W > threshold).float()


class CircuitFitter:
    """
    Fit latent circuit models to observed neural data.

    Optimizes a low-dimensional RNN to match high-dimensional neural responses,
    extracting minimal computational circuits.

    Args:
        n_latent: Number of latent units
        enforce_dales: Whether to enforce E/I separation
        learning_rate: Optimization learning rate (default: 1e-3)
        sparsity_weight: L1 penalty on recurrent weights (default: 0.01)
        device: Torch device

    Example:
        >>> fitter = CircuitFitter(n_latent=10, enforce_dales=True)
        >>> circuit = fitter.fit(neural_responses, stimuli, n_epochs=1000)
        >>> print(f"Final loss: {fitter.losses[-1]:.4f}")
    """

    def __init__(
        self,
        n_latent: int,
        enforce_dales: bool = True,
        learning_rate: float = 1e-3,
        sparsity_weight: float = 0.01,
        device: Optional[str] = None,
    ):
        self.n_latent = n_latent
        self.enforce_dales = enforce_dales
        self.learning_rate = learning_rate
        self.sparsity_weight = sparsity_weight
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.losses: List[float] = []

    def fit(
        self,
        neural_responses: Tensor,
        stimuli: Tensor,
        n_epochs: int = 1000,
        verbose: bool = True,
    ) -> LatentCircuitModel:
        """
        Fit latent circuit to neural data.

        Args:
            neural_responses: Observed neural activity [batch, seq_len, n_neurons]
            stimuli: Input stimuli [batch, seq_len, stim_dim]
            n_epochs: Number of training epochs
            verbose: Whether to print progress

        Returns:
            Fitted LatentCircuitModel
        """
        batch_size, seq_len, n_observed = neural_responses.shape
        _, _, stim_dim = stimuli.shape

        # Create model
        model = LatentCircuitModel(
            n_latent=self.n_latent,
            n_observed=n_observed,
            enforce_dales=self.enforce_dales,
            device=self.device,
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Move data to device
        neural_responses = neural_responses.to(self.device)
        stimuli = stimuli.to(self.device)

        # Training loop
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Forward pass
            predicted, hidden_states = model(stimuli)

            # Reconstruction loss
            recon_loss = F.mse_loss(predicted, neural_responses)

            # Sparsity penalty on recurrent weights
            sparsity_loss = self.sparsity_weight * model.W_rec.abs().mean()

            # Total loss
            loss = recon_loss + sparsity_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            self.losses.append(loss.item())

            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch}/{n_epochs} - Loss: {loss.item():.4f} "
                      f"(Recon: {recon_loss.item():.4f}, Sparsity: {sparsity_loss.item():.4f})")

        return model

    def evaluate(
        self,
        model: LatentCircuitModel,
        neural_responses: Tensor,
        stimuli: Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate fitted circuit on data.

        Args:
            model: Fitted LatentCircuitModel
            neural_responses: True neural responses
            stimuli: Input stimuli

        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        neural_responses = neural_responses.to(self.device)
        stimuli = stimuli.to(self.device)

        with torch.no_grad():
            predicted, _ = model(stimuli)

            # MSE
            mse = F.mse_loss(predicted, neural_responses).item()

            # R² score
            ss_res = ((neural_responses - predicted) ** 2).sum().item()
            ss_tot = ((neural_responses - neural_responses.mean()) ** 2).sum().item()
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            # Correlation (averaged across neurons)
            correlations = []
            for n in range(neural_responses.size(2)):
                true_n = neural_responses[:, :, n].flatten()
                pred_n = predicted[:, :, n].flatten()
                corr = torch.corrcoef(torch.stack([true_n, pred_n]))[0, 1]
                correlations.append(corr.item())

            mean_corr = np.mean(correlations)

        return {
            'mse': mse,
            'r2': r2,
            'mean_correlation': mean_corr,
            'neuron_correlations': correlations,
        }


class RecurrentDynamicsAnalyzer:
    """
    Analyze dynamics of fitted latent circuits.

    Provides tools for understanding the computational properties of
    extracted circuits: fixed points, stability, attractors, etc.

    Args:
        device: Torch device

    Example:
        >>> analyzer = RecurrentDynamicsAnalyzer()
        >>> circuit_info = circuit.extract_circuit()
        >>> analysis = analyzer.analyze(circuit_info)
        >>> print(f"Number of stable fixed points: {len(analysis['stable_fixed_points'])}")
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze(self, circuit_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive dynamical analysis of circuit.

        Args:
            circuit_info: Output from LatentCircuitModel.extract_circuit()

        Returns:
            Dictionary with dynamical analysis results
        """
        W_rec = circuit_info['W_rec']
        eigenvalues = circuit_info['eigenvalues']

        # 1. Stability analysis from eigenvalues
        max_eigenvalue = eigenvalues.abs().max().item()
        is_stable = max_eigenvalue < 1.0

        # 2. Find fixed points (numerical search)
        fixed_points = self._find_fixed_points(W_rec)

        # 3. Classify fixed points
        stable_fps = []
        unstable_fps = []

        for fp in fixed_points:
            jacobian = self._compute_jacobian(W_rec, fp)
            jac_eigenvalues = torch.linalg.eigvals(jacobian)
            if jac_eigenvalues.abs().max() < 1.0:
                stable_fps.append(fp)
            else:
                unstable_fps.append(fp)

        return {
            'max_eigenvalue': max_eigenvalue,
            'is_stable': is_stable,
            'fixed_points': fixed_points,
            'stable_fixed_points': stable_fps,
            'unstable_fixed_points': unstable_fps,
            'n_fixed_points': len(fixed_points),
        }

    def _find_fixed_points(
        self,
        W_rec: Tensor,
        n_initializations: int = 100,
        tolerance: float = 1e-4,
    ) -> List[Tensor]:
        """
        Find fixed points via random initialization + optimization.

        Fixed point: h* = tanh(W_rec @ h*)
        """
        fixed_points = []

        for _ in range(n_initializations):
            # Random initialization
            h = torch.randn(W_rec.size(0)) * 0.5
            h.requires_grad = True

            # Optimize to find fixed point
            optimizer = torch.optim.LBFGS([h], max_iter=20)

            def closure():
                optimizer.zero_grad()
                # Fixed point condition: h - tanh(W @ h) = 0
                residual = h - torch.tanh(W_rec @ h)
                loss = (residual ** 2).sum()
                loss.backward()
                return loss

            optimizer.step(closure)

            # Check if this is a valid fixed point
            with torch.no_grad():
                residual = h - torch.tanh(W_rec @ h)
                if residual.abs().max() < tolerance:
                    # Check if we already found this fixed point
                    is_new = True
                    for existing_fp in fixed_points:
                        if (h - existing_fp).abs().max() < tolerance:
                            is_new = False
                            break

                    if is_new:
                        fixed_points.append(h.detach().clone())

        return fixed_points

    def _compute_jacobian(self, W_rec: Tensor, fixed_point: Tensor) -> Tensor:
        """
        Compute Jacobian at a fixed point.

        Jacobian = W_rec * diag(1 - tanh²(W_rec @ h*))
        """
        h_star = fixed_point
        pre_activation = W_rec @ h_star
        tanh_deriv = 1 - torch.tanh(pre_activation) ** 2
        jacobian = W_rec * tanh_deriv.unsqueeze(1)
        return jacobian

    def compute_dimensionality(self, hidden_states: Tensor) -> Dict[str, float]:
        """
        Compute effective dimensionality of latent dynamics.

        Args:
            hidden_states: Latent states over time [batch, seq_len, n_latent]

        Returns:
            Dictionary with dimensionality metrics
        """
        # Flatten batch and time
        flat_states = hidden_states.reshape(-1, hidden_states.size(2))  # [batch*seq, n_latent]

        # PCA
        flat_states_centered = flat_states - flat_states.mean(dim=0, keepdim=True)
        cov = flat_states_centered.T @ flat_states_centered / flat_states_centered.size(0)
        eigenvalues, _ = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(0)  # Descending order

        # Participation ratio (effective dimensionality)
        eigenvalues_normalized = eigenvalues / eigenvalues.sum()
        participation_ratio = 1.0 / (eigenvalues_normalized ** 2).sum()

        # Explained variance
        explained_var_90 = 0
        cumsum = 0
        for i, ev in enumerate(eigenvalues_normalized):
            cumsum += ev
            if cumsum >= 0.9:
                explained_var_90 = i + 1
                break

        return {
            'participation_ratio': participation_ratio.item(),
            'dims_for_90pct_var': explained_var_90,
            'eigenvalues': eigenvalues.cpu().numpy(),
        }
