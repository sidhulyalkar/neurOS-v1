"""
DUNL: Deconvolutional Unrolled Neural Learning for Mixed Selectivity Decomposition

Disentangle mixed selectivity into interpretable components using iterative
sparse coding with learned dictionaries. Based on unrolled optimization methods.

Mixed selectivity neurons encode multiple task variables simultaneously (e.g.,
stimulus identity + task context + motor preparation). DUNL separates these
into distinct, interpretable factors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DUNLModel(nn.Module):
    """
    Deconvolutional Unrolled Neural Learning for sparse coding.

    Implements iterative soft-thresholding (ISTA) unrolled into a neural network.
    Each layer performs one iteration of sparse coding, with learned dictionaries
    and thresholds.

    The model solves: x ≈ D @ s, where D is the dictionary and s is sparse.

    Args:
        n_neurons: Number of input neurons
        n_factors: Number of dictionary atoms (sparse code dimension)
        n_iterations: Number of unrolling iterations (network depth)
        sparsity: Initial sparsity threshold
        learn_threshold: Whether to learn thresholds per iteration
        normalize_dict: Whether to normalize dictionary columns
        device: Torch device

    Example:
        >>> dunl = DUNLModel(n_neurons=100, n_factors=20, n_iterations=10)
        >>> sparse_codes, reconstructed = dunl(neural_activity)
        >>> factors = dunl.extract_factors()
    """

    def __init__(
        self,
        n_neurons: int,
        n_factors: int = 20,
        n_iterations: int = 10,
        sparsity: float = 0.1,
        learn_threshold: bool = True,
        normalize_dict: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.sparsity = sparsity
        self.learn_threshold = learn_threshold
        self.normalize_dict = normalize_dict
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Dictionary for each iteration (allows learning different dictionaries per layer)
        self.dictionaries = nn.ModuleList([
            nn.Linear(n_factors, n_neurons, bias=False)
            for _ in range(n_iterations)
        ])

        # Initialize dictionaries with random orthogonal matrices
        for dictionary in self.dictionaries:
            nn.init.orthogonal_(dictionary.weight)

        # Thresholds for soft-thresholding (one per iteration)
        if learn_threshold:
            self.thresholds = nn.Parameter(torch.ones(n_iterations) * sparsity)
        else:
            self.register_buffer('thresholds', torch.ones(n_iterations) * sparsity)

        # Step sizes for gradient descent
        self.step_sizes = nn.Parameter(torch.ones(n_iterations) * 0.1)

        self.to(self.device)

    def _normalize_dictionaries(self):
        """Normalize dictionary columns to unit norm."""
        if self.normalize_dict:
            for dictionary in self.dictionaries:
                with torch.no_grad():
                    norms = dictionary.weight.norm(dim=1, keepdim=True)
                    dictionary.weight.data = dictionary.weight.data / (norms + 1e-8)

    def _soft_threshold(self, x: Tensor, threshold: float) -> Tensor:
        """Soft-thresholding operator for sparsity."""
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)

    def forward(
        self,
        neural_activity: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: iterative sparse coding.

        Args:
            neural_activity: Neural responses [batch, n_neurons] or [batch, time, n_neurons]

        Returns:
            sparse_codes: Sparse factor codes [batch, n_factors] or [batch, time, n_factors]
            reconstructed: Reconstructed activity [batch, n_neurons] or [batch, time, n_neurons]
        """
        # Normalize dictionaries
        self._normalize_dictionaries()

        # Handle temporal dimension
        if neural_activity.dim() == 3:
            batch_size, seq_len, n_neurons = neural_activity.shape
            neural_activity_flat = neural_activity.reshape(batch_size * seq_len, n_neurons)
            was_temporal = True
        else:
            neural_activity_flat = neural_activity
            was_temporal = False

        # Initialize sparse codes
        sparse_codes = torch.zeros(
            neural_activity_flat.size(0), self.n_factors,
            device=self.device
        )

        # Iterative sparse coding (unrolled ISTA)
        for t in range(self.n_iterations):
            # Reconstruction from current codes
            reconstructed = self.dictionaries[t](sparse_codes)

            # Residual
            residual = neural_activity_flat - reconstructed

            # Gradient step: s = s + step_size * D^T @ residual
            # (D^T is the transpose/input direction of the linear layer)
            gradient = F.linear(residual, self.dictionaries[t].weight.T)
            sparse_codes = sparse_codes + self.step_sizes[t] * gradient

            # Soft-thresholding for sparsity
            sparse_codes = self._soft_threshold(sparse_codes, self.thresholds[t])

        # Final reconstruction
        reconstructed = self.dictionaries[-1](sparse_codes)

        # Reshape back if temporal
        if was_temporal:
            sparse_codes = sparse_codes.reshape(batch_size, seq_len, self.n_factors)
            reconstructed = reconstructed.reshape(batch_size, seq_len, n_neurons)

        return sparse_codes, reconstructed

    def extract_factors(self) -> Dict[str, Tensor]:
        """
        Extract learned dictionary factors.

        Returns:
            Dictionary with:
                - dictionaries: List of dictionary matrices (one per iteration)
                - final_dictionary: Final iteration dictionary [n_neurons, n_factors]
                - thresholds: Learned sparsity thresholds
        """
        return {
            'dictionaries': [d.weight.detach().cpu() for d in self.dictionaries],
            'final_dictionary': self.dictionaries[-1].weight.detach().cpu(),
            'thresholds': self.thresholds.detach().cpu() if isinstance(self.thresholds, nn.Parameter) else self.thresholds.cpu(),
            'step_sizes': self.step_sizes.detach().cpu(),
        }

    def compute_sparsity(self, sparse_codes: Tensor) -> Dict[str, float]:
        """
        Compute sparsity statistics of learned codes.

        Args:
            sparse_codes: Sparse codes from forward pass

        Returns:
            Dictionary with sparsity metrics
        """
        # Fraction of non-zero elements
        nonzero_fraction = (sparse_codes.abs() > 1e-6).float().mean().item()

        # L0 pseudo-norm (count of non-zeros)
        l0_norm = (sparse_codes.abs() > 1e-6).float().sum(dim=-1).mean().item()

        # L1 norm
        l1_norm = sparse_codes.abs().sum(dim=-1).mean().item()

        # Gini coefficient (measure of sparsity concentration)
        sorted_codes = torch.sort(sparse_codes.abs().flatten())[0]
        n = len(sorted_codes)
        index = torch.arange(1, n + 1, device=sorted_codes.device).float()
        gini = (2 * (index * sorted_codes).sum()) / (n * sorted_codes.sum() + 1e-10) - (n + 1) / n

        return {
            'nonzero_fraction': nonzero_fraction,
            'l0_norm': l0_norm,
            'l1_norm': l1_norm,
            'gini_coefficient': gini.item(),
        }


class MixedSelectivityAnalyzer:
    """
    Analyze and decompose mixed selectivity in neural responses.

    Identifies task-relevant factors and quantifies how neurons encode
    multiple variables simultaneously.

    Args:
        device: Torch device

    Example:
        >>> analyzer = MixedSelectivityAnalyzer()
        >>> decomposition = analyzer.decompose(responses, task_conditions)
        >>> print(f"Task variance: {decomposition['task_variance']:.2%}")
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def decompose(
        self,
        responses: Tensor,
        conditions: Dict[str, Tensor],
    ) -> Dict[str, Any]:
        """
        Decompose neural responses into task-relevant components.

        Args:
            responses: Neural activity [n_trials, n_neurons]
            conditions: Dictionary of condition labels
                       e.g., {'stimulus': [n_trials], 'context': [n_trials], 'choice': [n_trials]}

        Returns:
            Dictionary with variance decomposition and component loadings
        """
        responses = responses.to(self.device)

        # Center responses
        responses_centered = responses - responses.mean(dim=0, keepdim=True)

        # Total variance
        total_var = (responses_centered ** 2).sum().item()

        # Variance explained by each condition
        variance_explained = {}
        component_loadings = {}

        for cond_name, cond_labels in conditions.items():
            cond_labels = cond_labels.to(self.device)

            # Get unique conditions
            unique_conds = torch.unique(cond_labels)

            # Compute condition means
            cond_means = []
            for cond in unique_conds:
                mask = cond_labels == cond
                cond_mean = responses[mask].mean(dim=0)
                cond_means.append(cond_mean)

            cond_means = torch.stack(cond_means, dim=0)  # [n_conds, n_neurons]

            # Variance explained by this condition (between-condition variance)
            # For each trial, find distance to condition mean
            explained_var = 0.0
            for cond in unique_conds:
                mask = cond_labels == cond
                cond_idx = (unique_conds == cond).nonzero(as_tuple=True)[0].item()
                cond_mean = cond_means[cond_idx]

                # Distance from global mean to condition mean
                explained_var += ((cond_mean - responses.mean(dim=0)) ** 2).sum().item() * mask.sum().item()

            variance_explained[cond_name] = explained_var / total_var

            # Component loadings (how much each neuron encodes this condition)
            # Use discriminant analysis: loadings = cov^{-1} @ (mean_diff)
            # Simplified: just use difference in means
            if len(unique_conds) == 2:
                mean_diff = cond_means[1] - cond_means[0]
                component_loadings[cond_name] = mean_diff.cpu()
            else:
                # Multi-class: use first PC of condition means
                cond_means_centered = cond_means - cond_means.mean(dim=0, keepdim=True)
                U, S, V = torch.svd(cond_means_centered.T)
                component_loadings[cond_name] = V[:, 0].cpu()  # First PC

        # Interaction terms (variance not explained by any single factor)
        explained_total = sum(variance_explained.values())
        interaction_variance = max(0, 1.0 - explained_total)

        return {
            'variance_explained': variance_explained,
            'component_loadings': component_loadings,
            'interaction_variance': interaction_variance,
            'total_variance': total_var,
        }

    def compute_selectivity_index(
        self,
        responses: Tensor,
        condition_labels: Tensor,
    ) -> Tensor:
        """
        Compute selectivity index for each neuron.

        Selectivity = (max_response - mean_other) / (max_response + mean_other)

        Args:
            responses: Neural responses [n_trials, n_neurons]
            condition_labels: Condition labels [n_trials]

        Returns:
            Selectivity indices [n_neurons]
        """
        responses = responses.to(self.device)
        condition_labels = condition_labels.to(self.device)

        unique_conds = torch.unique(condition_labels)
        n_neurons = responses.size(1)

        selectivity = torch.zeros(n_neurons, device=self.device)

        for neuron in range(n_neurons):
            neuron_responses = responses[:, neuron]

            # Mean response for each condition
            cond_means = []
            for cond in unique_conds:
                mask = condition_labels == cond
                cond_means.append(neuron_responses[mask].mean())

            cond_means = torch.stack(cond_means)

            # Selectivity
            max_response = cond_means.max()
            other_mean = cond_means[cond_means != max_response].mean()

            selectivity[neuron] = (max_response - other_mean) / (max_response + other_mean + 1e-10)

        return selectivity


class FactorDecomposition:
    """
    Decompose neural activity into interpretable factors using various methods.

    Supports:
        - PCA (principal components)
        - ICA (independent components)
        - NMF (non-negative matrix factorization)
        - DUNL (learned sparse dictionary)

    Args:
        method: Decomposition method ('pca', 'ica', 'nmf', 'dunl')
        n_components: Number of components/factors
        device: Torch device

    Example:
        >>> decomposer = FactorDecomposition(method='dunl', n_components=15)
        >>> factors, loadings = decomposer.fit_transform(neural_data)
    """

    def __init__(
        self,
        method: str = 'dunl',
        n_components: int = 15,
        device: Optional[str] = None,
    ):
        self.method = method
        self.n_components = n_components
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if method == 'dunl':
            self.model = None  # Will be created in fit

    def fit_transform(
        self,
        neural_data: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Fit decomposition and transform data.

        Args:
            neural_data: Neural activity [n_samples, n_neurons] or [batch, time, n_neurons]

        Returns:
            factors: Low-dimensional factors [n_samples, n_components]
            loadings: Neuron loadings on factors [n_neurons, n_components]
        """
        neural_data = neural_data.to(self.device)

        if self.method == 'pca':
            return self._pca(neural_data)
        elif self.method == 'ica':
            return self._ica(neural_data)
        elif self.method == 'nmf':
            return self._nmf(neural_data)
        elif self.method == 'dunl':
            return self._dunl(neural_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _pca(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """Principal Component Analysis."""
        # Handle temporal dimension
        if data.dim() == 3:
            batch_size, seq_len, n_neurons = data.shape
            data = data.reshape(batch_size * seq_len, n_neurons)

        # Center data
        data_centered = data - data.mean(dim=0, keepdim=True)

        # SVD
        U, S, Vt = torch.svd(data_centered)

        # Factors (scores) and loadings
        factors = U[:, :self.n_components] * S[:self.n_components].unsqueeze(0)
        loadings = Vt[:, :self.n_components]  # [n_neurons, n_components]

        return factors, loadings

    def _ica(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """Independent Component Analysis (FastICA)."""
        # Simplified ICA using torch
        # First whiten with PCA
        factors_pca, loadings_pca = self._pca(data)

        # FastICA iterations (simplified)
        W = torch.randn(self.n_components, self.n_components, device=self.device)
        W = torch.linalg.qr(W)[0]  # Orthogonalize

        for _ in range(100):
            # Non-linear transformation (tanh)
            WX = factors_pca @ W.T
            g = torch.tanh(WX)
            g_prime = 1 - g ** 2

            # Update rule
            W_new = (g @ factors_pca).T / factors_pca.size(0) - g_prime.mean(dim=0, keepdim=True).T * W
            W = torch.linalg.qr(W_new)[0]

        # Independent components
        factors = factors_pca @ W.T
        loadings = loadings_pca @ W

        return factors, loadings

    def _nmf(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """Non-negative Matrix Factorization."""
        # Handle temporal dimension
        if data.dim() == 3:
            batch_size, seq_len, n_neurons = data.shape
            data = data.reshape(batch_size * seq_len, n_neurons)

        # Ensure non-negative
        data = F.relu(data)

        # Initialize
        n_samples, n_neurons = data.shape
        W = torch.rand(n_samples, self.n_components, device=self.device)
        H = torch.rand(self.n_components, n_neurons, device=self.device)

        # Multiplicative update rules
        for _ in range(100):
            # Update H
            H = H * ((W.T @ data) / (W.T @ W @ H + 1e-10))

            # Update W
            W = W * ((data @ H.T) / (W @ H @ H.T + 1e-10))

        factors = W
        loadings = H.T

        return factors, loadings

    def _dunl(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """DUNL sparse dictionary learning."""
        # Handle temporal dimension
        original_shape = data.shape
        if data.dim() == 3:
            batch_size, seq_len, n_neurons = data.shape
            data_flat = data.reshape(batch_size * seq_len, n_neurons)
        else:
            data_flat = data

        n_neurons = data_flat.size(1)

        # Create and train DUNL model
        if self.model is None:
            self.model = DUNLModel(
                n_neurons=n_neurons,
                n_factors=self.n_components,
                n_iterations=10,
                device=self.device,
            )

        # Train (simplified - in practice would use proper training loop)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(100):
            optimizer.zero_grad()
            sparse_codes, reconstructed = self.model(data_flat)
            loss = F.mse_loss(reconstructed, data_flat)
            loss.backward()
            optimizer.step()

        # Get final factors and loadings
        with torch.no_grad():
            factors, _ = self.model(data_flat)

        factor_dict = self.model.extract_factors()
        loadings = factor_dict['final_dictionary'].T.to(self.device)  # [n_neurons, n_factors]

        # Reshape factors if needed
        if len(original_shape) == 3:
            factors = factors.reshape(original_shape[0], original_shape[1], self.n_components)

        return factors, loadings
