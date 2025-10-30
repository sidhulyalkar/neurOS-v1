"""
Fractal Regularizers for Training

Training losses that enforce fractal properties in learned representations.
These priors encourage biologically plausible dynamics (1/f noise, scale-free structure).

Usage in training:
    >>> from neuros_neurofm.losses import CombinedLoss
    >>> from neuros_mechint.fractals import SpectralPrior
    >>>
    >>> loss_fn = CombinedLoss([
    >>>     MaskedModelingLoss(),
    >>>     SpectralPrior(target_beta=1.0, weight=0.01),
    >>> ])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import logging

from .metrics import SpectralSlope, MultifractalSpectrum, GraphFractalDimension

logger = logging.getLogger(__name__)


class SpectralPrior(nn.Module):
    """
    1/f^β spectral prior for latent dynamics.

    Encourages learned representations to exhibit scale-free temporal dynamics
    matching biological neural activity (typically β ≈ 1 for pink noise).

    Loss = weight * |β_observed - β_target|^2

    Args:
        target_beta: Target spectral exponent (default: 1.0 for 1/f pink noise)
        weight: Loss weight (default: 0.01)
        freq_range: Frequency range for fitting (Hz)
        sampling_rate: Sampling rate (Hz)

    Example:
        >>> prior = SpectralPrior(target_beta=1.0, weight=0.01)
        >>> latents = model.encode(batch)  # [batch, time, features]
        >>> loss = prior(latents)
    """

    def __init__(
        self,
        target_beta: float = 1.0,
        weight: float = 0.01,
        freq_range: Tuple[float, float] = (1.0, 50.0),
        sampling_rate: float = 100.0,
        apply_per_feature: bool = False,
    ):
        super().__init__()
        self.target_beta = target_beta
        self.weight = weight
        self.apply_per_feature = apply_per_feature

        self.spectral_estimator = SpectralSlope(
            freq_range=freq_range,
            sampling_rate=sampling_rate,
        )

    def forward(self, latents: Tensor) -> Tensor:
        """
        Compute spectral prior loss.

        Args:
            latents: Latent representations [batch, time, features] or [batch, time]

        Returns:
            Loss scalar
        """
        if latents.dim() == 3:
            batch_size, seq_len, n_features = latents.shape

            if self.apply_per_feature:
                # Compute spectral slope for each feature independently
                latents_2d = latents.permute(0, 2, 1).reshape(batch_size * n_features, seq_len)
            else:
                # Average across features
                latents_2d = latents.mean(dim=2)  # [batch, time]
        else:
            latents_2d = latents

        # Estimate spectral slope
        beta_observed, _, _ = self.spectral_estimator.compute(latents_2d)

        # Loss: penalize deviation from target
        loss = ((beta_observed - self.target_beta) ** 2).mean()

        return self.weight * loss


class MultifractalSmoothness(nn.Module):
    """
    Encourage smooth multifractal spectra.

    Penalizes highly irregular f(α) curves, promoting smooth scaling properties.
    Biological systems often exhibit smooth multifractal spectra.

    Loss = weight * (curvature of f(α))

    Args:
        weight: Loss weight (default: 0.005)
        q_range: Range of moments for multifractal analysis

    Example:
        >>> smoothness = MultifractalSmoothness(weight=0.005)
        >>> latents = model.encode(batch)
        >>> loss = smoothness(latents)
    """

    def __init__(
        self,
        weight: float = 0.005,
        q_range: Tuple[float, float] = (-3.0, 3.0),
        n_q: int = 15,
    ):
        super().__init__()
        self.weight = weight
        self.mf_estimator = MultifractalSpectrum(q_range=q_range, n_q=n_q)

    def forward(self, latents: Tensor) -> Tensor:
        """
        Compute multifractal smoothness loss.

        Args:
            latents: Latent representations [batch, time, features] or [batch, time]

        Returns:
            Loss scalar
        """
        if latents.dim() == 3:
            # Average across features
            latents = latents.mean(dim=2)

        # Compute multifractal spectrum
        mf_results = self.mf_estimator.compute(latents)
        f_alpha = mf_results['f_alpha']  # [batch, n_q]

        # Compute second derivative (curvature) of f(α)
        # Higher curvature = more irregular spectrum
        d2f = torch.diff(f_alpha, n=2, dim=1)  # Second-order differences
        curvature = (d2f ** 2).mean()

        return self.weight * curvature


class GraphFractalityPrior(nn.Module):
    """
    Encourage fractal graph structure in attention weights.

    Biological neural networks often exhibit fractal connectivity patterns.
    This prior encourages attention weights to form fractal graphs.

    Loss = weight * |d_B_observed - d_B_target|^2

    Args:
        target_dim: Target graph fractal dimension (default: 2.0)
        weight: Loss weight (default: 0.005)
        threshold: Edge weight threshold for binarization (default: 0.1)

    Example:
        >>> prior = GraphFractalityPrior(target_dim=2.0, weight=0.005)
        >>> attention_weights = model.get_attention_weights()  # [batch, n, n]
        >>> loss = prior(attention_weights)
    """

    def __init__(
        self,
        target_dim: float = 2.0,
        weight: float = 0.005,
        threshold: float = 0.1,
        min_box: int = 2,
        max_box: int = 8,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.weight = weight

        self.graph_fd_estimator = GraphFractalDimension(
            min_box=min_box,
            max_box=max_box,
            threshold=threshold,
        )

    def forward(self, attention_weights: Tensor) -> Tensor:
        """
        Compute graph fractality prior loss.

        Args:
            attention_weights: Attention weights [batch, n_heads, n, n] or [batch, n, n]

        Returns:
            Loss scalar
        """
        if attention_weights.dim() == 4:
            # Average across heads
            batch_size, n_heads, n, _ = attention_weights.shape
            attention_weights = attention_weights.mean(dim=1)  # [batch, n, n]

        # Compute graph fractal dimension
        d_B_observed = self.graph_fd_estimator.compute(attention_weights)

        # Loss: penalize deviation from target
        loss = ((d_B_observed - self.target_dim) ** 2).mean()

        return self.weight * loss


class TemporalScaleInvariance(nn.Module):
    """
    Encourage temporal scale invariance in representations.

    Enforces that representations at different temporal resolutions maintain
    similar statistical properties (self-similarity).

    Loss = weight * KL(P_coarse || P_fine)

    Args:
        weight: Loss weight (default: 0.01)
        scales: Downsampling scales to compare (default: [2, 4, 8])

    Example:
        >>> scale_inv = TemporalScaleInvariance(weight=0.01, scales=[2, 4])
        >>> latents = model.encode(batch)
        >>> loss = scale_inv(latents)
    """

    def __init__(
        self,
        weight: float = 0.01,
        scales: list = None,
    ):
        super().__init__()
        self.weight = weight
        self.scales = scales or [2, 4, 8]

    def forward(self, latents: Tensor) -> Tensor:
        """
        Compute temporal scale invariance loss.

        Args:
            latents: Latent representations [batch, time, features]

        Returns:
            Loss scalar
        """
        if latents.dim() == 2:
            latents = latents.unsqueeze(2)

        batch_size, seq_len, n_features = latents.shape

        # Compute statistics at original scale
        mu_orig = latents.mean(dim=1)  # [batch, features]
        sigma_orig = latents.std(dim=1)  # [batch, features]

        total_loss = 0.0

        for scale in self.scales:
            if seq_len // scale < 10:  # Skip if too short
                continue

            # Downsample
            downsampled = F.avg_pool1d(
                latents.permute(0, 2, 1),  # [batch, features, time]
                kernel_size=scale,
                stride=scale,
            ).permute(0, 2, 1)  # [batch, time', features]

            # Compute statistics
            mu_scaled = downsampled.mean(dim=1)
            sigma_scaled = downsampled.std(dim=1)

            # KL divergence between Gaussians
            # KL(N(μ1,σ1) || N(μ2,σ2)) = log(σ2/σ1) + (σ1^2 + (μ1-μ2)^2)/(2σ2^2) - 0.5
            kl = (
                torch.log((sigma_orig + 1e-6) / (sigma_scaled + 1e-6))
                + (sigma_scaled ** 2 + (mu_scaled - mu_orig) ** 2) / (2 * (sigma_orig ** 2 + 1e-6))
                - 0.5
            )

            total_loss += kl.mean()

        return self.weight * total_loss / len(self.scales)


class FractalRegularizationLoss(nn.Module):
    """
    Combined fractal regularization loss.

    Combines multiple fractal priors for comprehensive biological plausibility.

    Args:
        spectral_weight: Weight for spectral prior (default: 0.01)
        multifractal_weight: Weight for multifractal smoothness (default: 0.005)
        graph_weight: Weight for graph fractality (default: 0.005)
        scale_inv_weight: Weight for scale invariance (default: 0.01)
        target_beta: Target spectral exponent (default: 1.0)
        target_graph_dim: Target graph fractal dimension (default: 2.0)

    Example:
        >>> fractal_loss = FractalRegularizationLoss(
        >>>     spectral_weight=0.01,
        >>>     multifractal_weight=0.005,
        >>> )
        >>> latents = model.encode(batch)
        >>> attention = model.get_attention_weights()
        >>> loss = fractal_loss(latents, attention_weights=attention)
    """

    def __init__(
        self,
        spectral_weight: float = 0.01,
        multifractal_weight: float = 0.005,
        graph_weight: float = 0.005,
        scale_inv_weight: float = 0.01,
        target_beta: float = 1.0,
        target_graph_dim: float = 2.0,
    ):
        super().__init__()

        self.spectral_prior = SpectralPrior(
            target_beta=target_beta,
            weight=spectral_weight,
        ) if spectral_weight > 0 else None

        self.multifractal_smoothness = MultifractalSmoothness(
            weight=multifractal_weight,
        ) if multifractal_weight > 0 else None

        self.graph_prior = GraphFractalityPrior(
            target_dim=target_graph_dim,
            weight=graph_weight,
        ) if graph_weight > 0 else None

        self.scale_invariance = TemporalScaleInvariance(
            weight=scale_inv_weight,
        ) if scale_inv_weight > 0 else None

    def forward(
        self,
        latents: Tensor,
        attention_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute combined fractal regularization loss.

        Args:
            latents: Latent representations [batch, time, features]
            attention_weights: Optional attention weights for graph prior

        Returns:
            Total loss scalar
        """
        total_loss = 0.0

        if self.spectral_prior is not None:
            total_loss += self.spectral_prior(latents)

        if self.multifractal_smoothness is not None:
            total_loss += self.multifractal_smoothness(latents)

        if self.graph_prior is not None and attention_weights is not None:
            total_loss += self.graph_prior(attention_weights)

        if self.scale_invariance is not None:
            total_loss += self.scale_invariance(latents)

        return total_loss


class AdaptiveFractalPrior(nn.Module):
    """
    Adaptive fractal prior that learns target parameters.

    Instead of fixed target β or dimension, learns optimal values from data.

    Args:
        initial_beta: Initial spectral exponent (default: 1.0)
        initial_graph_dim: Initial graph dimension (default: 2.0)
        learnable: Whether to make targets learnable (default: True)
        weight: Loss weight (default: 0.01)

    Example:
        >>> adaptive_prior = AdaptiveFractalPrior(learnable=True)
        >>> # During training, targets will be optimized
        >>> loss = adaptive_prior(latents, attention_weights)
        >>> print(f"Learned beta: {adaptive_prior.target_beta.item()}")
    """

    def __init__(
        self,
        initial_beta: float = 1.0,
        initial_graph_dim: float = 2.0,
        learnable: bool = True,
        weight: float = 0.01,
    ):
        super().__init__()
        self.weight = weight

        if learnable:
            self.target_beta = nn.Parameter(torch.tensor(initial_beta))
            self.target_graph_dim = nn.Parameter(torch.tensor(initial_graph_dim))
        else:
            self.register_buffer('target_beta', torch.tensor(initial_beta))
            self.register_buffer('target_graph_dim', torch.tensor(initial_graph_dim))

        self.spectral_estimator = SpectralSlope()
        self.graph_estimator = GraphFractalDimension()

    def forward(
        self,
        latents: Tensor,
        attention_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute adaptive fractal prior loss."""
        total_loss = 0.0

        # Spectral prior
        if latents.dim() == 3:
            latents_2d = latents.mean(dim=2)
        else:
            latents_2d = latents

        beta_obs, _, _ = self.spectral_estimator.compute(latents_2d)
        spectral_loss = ((beta_obs - self.target_beta) ** 2).mean()
        total_loss += spectral_loss

        # Graph prior
        if attention_weights is not None:
            if attention_weights.dim() == 4:
                attention_weights = attention_weights.mean(dim=1)

            graph_dim_obs = self.graph_estimator.compute(attention_weights)
            graph_loss = ((graph_dim_obs - self.target_graph_dim) ** 2).mean()
            total_loss += graph_loss

        return self.weight * total_loss

    def get_learned_parameters(self) -> dict:
        """Get learned fractal parameters."""
        return {
            'target_beta': self.target_beta.item(),
            'target_graph_dim': self.target_graph_dim.item(),
        }
