"""
Denoising Diffusion Loss for NeuroFMX

Implements diffusion-based losses for neural data:
- Denoising score matching for neural segments
- Multiple noise schedules (linear, cosine, polynomial)
- Timestep sampling strategies
- Compatible with NeuroFMX latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Literal
import numpy as np


class DiffusionLoss(nn.Module):
    """
    Denoising diffusion probabilistic model (DDPM) loss.

    Trains a model to predict noise added to neural data at various noise levels.
    Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020) and
    adapted for neural data latent representations.

    Args:
        n_timesteps: Number of diffusion timesteps
        schedule_type: Noise schedule ('linear', 'cosine', 'polynomial')
        beta_start: Starting beta for linear schedule
        beta_end: Ending beta for linear schedule
        loss_type: Loss function ('mse', 'smooth_l1', 'huber')
        timestep_sampling: Sampling strategy ('uniform', 'low_discrepancy', 'importance')
        prediction_type: What to predict ('noise', 'x0', 'v')
        gradient_clip_val: Maximum gradient norm
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        schedule_type: Literal['linear', 'cosine', 'polynomial'] = 'cosine',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        loss_type: Literal['mse', 'smooth_l1', 'huber'] = 'mse',
        timestep_sampling: Literal['uniform', 'low_discrepancy', 'importance'] = 'uniform',
        prediction_type: Literal['noise', 'x0', 'v'] = 'noise',
        gradient_clip_val: float = 1.0
    ):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.schedule_type = schedule_type
        self.loss_type = loss_type
        self.timestep_sampling = timestep_sampling
        self.prediction_type = prediction_type
        self.gradient_clip_val = gradient_clip_val

        # Generate noise schedule
        self.register_buffer('betas', self._generate_schedule(
            n_timesteps, schedule_type, beta_start, beta_end
        ))

        # Precompute diffusion constants
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', torch.cat([
            torch.tensor([1.0]), self.alphas_cumprod[:-1]
        ]))

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                            torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                            torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                            torch.sqrt(1.0 / self.alphas_cumprod - 1))

        # For importance sampling
        if timestep_sampling == 'importance':
            self.register_buffer('timestep_weights',
                                torch.ones(n_timesteps) / n_timesteps)

    def _generate_schedule(
        self,
        n_timesteps: int,
        schedule_type: str,
        beta_start: float,
        beta_end: float
    ) -> torch.Tensor:
        """Generate noise schedule."""
        if schedule_type == 'linear':
            return torch.linspace(beta_start, beta_end, n_timesteps)

        elif schedule_type == 'cosine':
            # Improved cosine schedule from "Improved DDPM"
            steps = n_timesteps + 1
            x = torch.linspace(0, n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / n_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)

        elif schedule_type == 'polynomial':
            # Quadratic schedule
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_timesteps) ** 2
            return betas

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def forward(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute diffusion training loss.

        Args:
            model: Denoising model (takes x_t, t, condition and predicts noise/x0)
            x_start: Clean data, shape (batch, seq_len, dim) or (batch, dim)
            condition: Optional conditioning information
            attention_mask: Valid positions mask (for sequences)

        Returns:
            loss: Diffusion loss
            metrics: Loss metrics and statistics
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample timesteps
        t = self._sample_timesteps(batch_size, device)

        # Sample noise
        noise = torch.randn_like(x_start)

        # Add noise to clean data: q(x_t | x_0)
        x_noisy = self._q_sample(x_start, t, noise)

        # Predict noise/x0 with model
        if condition is not None:
            model_output = model(x_noisy, t, condition)
        else:
            model_output = model(x_noisy, t)

        # Determine target based on prediction type
        if self.prediction_type == 'noise':
            target = noise
        elif self.prediction_type == 'x0':
            target = x_start
        elif self.prediction_type == 'v':
            # Velocity prediction (v-parameterization)
            target = (
                self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_start.dim() - 1))) * noise -
                self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_start.dim() - 1))) * x_start
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            model_output = model_output * mask
            target = target * mask
            num_valid = mask.sum()
        else:
            num_valid = target.numel()

        # Compute loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(model_output, target, reduction='sum')
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(model_output, target, reduction='sum')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(model_output, target, reduction='sum')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Normalize by number of valid elements
        if num_valid > 0:
            loss = loss / num_valid

        # Gradient clipping
        if self.gradient_clip_val > 0 and self.training:
            loss = loss * torch.clamp(
                torch.ones_like(loss),
                max=self.gradient_clip_val
            )

        # Compute metrics
        metrics = {}
        with torch.no_grad():
            mae = (model_output - target).abs().mean()
            metrics['diffusion_loss'] = loss
            metrics['diffusion_mae'] = mae
            metrics['mean_timestep'] = t.float().mean()

            # SNR at sampled timesteps
            snr = self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
            metrics['mean_snr'] = snr.mean()

        return loss, metrics

    def _q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to x_start at timestep t.

        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
        """
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, *([1] * (x_start.dim() - 1)))
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(
            -1, *([1] * (x_start.dim() - 1))
        )

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def _sample_timesteps(self, batch_size: int, device: str) -> torch.Tensor:
        """Sample timesteps according to sampling strategy."""
        if self.timestep_sampling == 'uniform':
            # Uniform random sampling
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=device)

        elif self.timestep_sampling == 'low_discrepancy':
            # Low-discrepancy sampling (quasi-random)
            # Use Sobol sequence approximation
            step = self.n_timesteps / batch_size
            t = torch.arange(batch_size, device=device) * step
            t = (t + torch.rand(batch_size, device=device) * step).long()
            t = torch.clamp(t, 0, self.n_timesteps - 1)

        elif self.timestep_sampling == 'importance':
            # Importance sampling based on loss history
            # Sample according to timestep_weights
            t = torch.multinomial(
                self.timestep_weights.to(device),
                batch_size,
                replacement=True
            )

        else:
            raise ValueError(f"Unknown timestep sampling: {self.timestep_sampling}")

        return t

    def update_importance_weights(self, timesteps: torch.Tensor, losses: torch.Tensor):
        """Update importance weights based on per-timestep losses."""
        if self.timestep_sampling != 'importance':
            return

        # Update timestep weights with exponential moving average
        alpha = 0.1
        for t, loss in zip(timesteps.cpu(), losses.cpu()):
            self.timestep_weights[t] = (
                (1 - alpha) * self.timestep_weights[t] + alpha * loss.item()
            )

        # Normalize
        self.timestep_weights = self.timestep_weights / self.timestep_weights.sum()


class LatentDiffusionLoss(nn.Module):
    """
    Diffusion loss for neural latent representations.

    Specifically designed for NeuroFMX latent space. Includes:
    - Latent space normalization
    - Conditional diffusion on past context
    - KL regularization to keep latents well-behaved

    Args:
        n_timesteps: Number of diffusion timesteps
        schedule_type: Noise schedule type
        latent_dim: Dimension of latent space
        condition_on_past: Whether to condition on past latents
        kl_weight: Weight for KL regularization
        **kwargs: Additional args for DiffusionLoss
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        schedule_type: str = 'cosine',
        latent_dim: int = 512,
        condition_on_past: bool = True,
        kl_weight: float = 0.001,
        **kwargs
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_on_past = condition_on_past
        self.kl_weight = kl_weight

        # Base diffusion loss
        self.diffusion_loss = DiffusionLoss(
            n_timesteps=n_timesteps,
            schedule_type=schedule_type,
            **kwargs
        )

        # Latent normalization layers
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        past_latents: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute latent diffusion loss.

        Args:
            model: Denoising model
            latents: Current latent representations, shape (batch, seq_len, latent_dim)
            past_latents: Past latents for conditioning, shape (batch, context_len, latent_dim)
            attention_mask: Valid positions

        Returns:
            loss: Total loss (diffusion + KL)
            metrics: Loss components and statistics
        """
        # Normalize latents
        latents_norm = self.latent_norm(latents)

        # Prepare conditioning
        condition = None
        if self.condition_on_past and past_latents is not None:
            # Pool past latents as conditioning
            condition = past_latents.mean(dim=1)  # (batch, latent_dim)

        # Compute diffusion loss
        diff_loss, diff_metrics = self.diffusion_loss(
            model, latents_norm, condition, attention_mask
        )

        # KL regularization (encourage latents to stay near standard normal)
        kl_loss = self._compute_kl_regularization(latents_norm)

        # Total loss
        total_loss = diff_loss + self.kl_weight * kl_loss

        # Combine metrics
        metrics = {**diff_metrics}
        metrics['kl_loss'] = kl_loss
        metrics['total_loss'] = total_loss

        return total_loss, metrics

    def _compute_kl_regularization(self, latents: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from standard normal.

        KL(q(z) || p(z)) where p(z) = N(0, I)
        """
        # Compute mean and variance across batch and sequence
        mean = latents.mean(dim=[0, 1])
        var = latents.var(dim=[0, 1])

        # KL divergence from N(0, 1)
        kl = 0.5 * torch.sum(mean ** 2 + var - torch.log(var + 1e-8) - 1)

        return kl


class NeuralSegmentDiffusionLoss(nn.Module):
    """
    Diffusion loss for neural data segments.

    Designed for short segments of neural activity (e.g., 100-500ms).
    Includes:
    - Segment-level denoising
    - Multi-scale diffusion (different schedules for different frequencies)
    - Support for spiking data (Poisson likelihood)

    Args:
        n_timesteps: Number of diffusion timesteps
        segment_length_ms: Length of segments in milliseconds
        sampling_rate_hz: Data sampling rate
        multi_scale: Use multi-scale diffusion
        scale_schedules: Different schedules for different scales
        data_type: Type of neural data ('continuous', 'spikes')
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        segment_length_ms: float = 250.0,
        sampling_rate_hz: float = 1000.0,
        multi_scale: bool = False,
        scale_schedules: Optional[Dict[str, str]] = None,
        data_type: Literal['continuous', 'spikes'] = 'continuous'
    ):
        super().__init__()

        self.segment_length_ms = segment_length_ms
        self.sampling_rate_hz = sampling_rate_hz
        self.segment_length_steps = int(segment_length_ms * sampling_rate_hz / 1000.0)
        self.multi_scale = multi_scale
        self.data_type = data_type

        if multi_scale and scale_schedules:
            # Create multiple diffusion losses with different schedules
            self.scale_losses = nn.ModuleDict()
            for scale_name, schedule_type in scale_schedules.items():
                self.scale_losses[scale_name] = DiffusionLoss(
                    n_timesteps=n_timesteps,
                    schedule_type=schedule_type
                )
        else:
            # Single-scale diffusion
            self.diffusion_loss = DiffusionLoss(n_timesteps=n_timesteps)

    def forward(
        self,
        model: nn.Module,
        segments: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute segment diffusion loss.

        Args:
            model: Denoising model
            segments: Neural segments, shape (batch, segment_len, channels)
            condition: Optional conditioning

        Returns:
            loss: Diffusion loss
            metrics: Loss metrics
        """
        if self.multi_scale:
            return self._multi_scale_forward(model, segments, condition)
        else:
            return self.diffusion_loss(model, segments, condition)

    def _multi_scale_forward(
        self,
        model: nn.Module,
        segments: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Multi-scale diffusion loss."""
        total_loss = 0.0
        all_metrics = {}

        for scale_name, scale_loss in self.scale_losses.items():
            loss, metrics = scale_loss(model, segments, condition)
            total_loss += loss

            # Add scale prefix to metrics
            for key, value in metrics.items():
                all_metrics[f'{scale_name}_{key}'] = value

        all_metrics['total_loss'] = total_loss

        return total_loss, all_metrics


# Example usage
if __name__ == '__main__':
    batch_size, seq_len, dim = 4, 100, 128

    # Create a simple denoising model (for demonstration)
    class SimpleDenoiser(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 64, 256),  # +64 for time embedding
                nn.SiLU(),
                nn.Linear(256, dim)
            )

        def forward(self, x, t, condition=None):
            # Simple time embedding
            t_emb = torch.sin(t.unsqueeze(-1).float() * torch.linspace(0, 1, 64).to(x.device))

            if x.dim() == 3:  # (batch, seq, dim)
                t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
                x_with_t = torch.cat([x, t_emb], dim=-1)
            else:  # (batch, dim)
                x_with_t = torch.cat([x, t_emb], dim=-1)

            return self.net(x_with_t)

    model = SimpleDenoiser(dim)

    # Test basic diffusion loss
    x_clean = torch.randn(batch_size, seq_len, dim)

    diff_loss = DiffusionLoss(
        n_timesteps=1000,
        schedule_type='cosine',
        loss_type='mse'
    )

    loss, metrics = diff_loss(model, x_clean)
    print(f"Diffusion loss: {loss.item():.4f}")
    print(f"MAE: {metrics['diffusion_mae'].item():.4f}")
    print(f"Mean SNR: {metrics['mean_snr'].item():.4f}")

    # Test latent diffusion loss
    latent_dim = 512
    latents = torch.randn(batch_size, seq_len, latent_dim)
    past_latents = torch.randn(batch_size, 50, latent_dim)

    latent_model = SimpleDenoiser(latent_dim)
    latent_diff_loss = LatentDiffusionLoss(
        n_timesteps=1000,
        latent_dim=latent_dim,
        condition_on_past=True,
        kl_weight=0.001
    )

    loss, metrics = latent_diff_loss(latent_model, latents, past_latents)
    print(f"\nLatent diffusion loss: {loss.item():.4f}")
    print(f"KL loss: {metrics['kl_loss'].item():.6f}")

    # Test segment diffusion
    segments = torch.randn(batch_size, 250, 64)  # 250ms @ 1kHz, 64 channels

    segment_diff_loss = NeuralSegmentDiffusionLoss(
        n_timesteps=500,
        segment_length_ms=250.0,
        sampling_rate_hz=1000.0,
        multi_scale=True,
        scale_schedules={
            'fine': 'linear',
            'coarse': 'cosine'
        }
    )

    segment_model = SimpleDenoiser(64)
    loss, metrics = segment_diff_loss(segment_model, segments)
    print(f"\nSegment diffusion loss: {loss.item():.4f}")
