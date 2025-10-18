"""
Latent diffusion for NeuroFM-X.

Implements latent diffusion models for neural forecasting and generation,
inspired by LDNS (Latent Diffusion for Neural Spiking) and GNOCCHI.

This enables 1-2 second ahead neural activity forecasting and imputation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffusionSchedule:
    """Noise schedule for diffusion process.

    Parameters
    ----------
    n_timesteps : int, optional
        Number of diffusion timesteps.
        Default: 1000.
    schedule_type : str, optional
        Type of schedule ('linear', 'cosine', 'quadratic').
        Default: 'cosine'.
    beta_start : float, optional
        Starting beta value for linear schedule.
        Default: 0.0001.
    beta_end : float, optional
        Ending beta value for linear schedule.
        Default: 0.02.
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        schedule_type: str = 'cosine',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.n_timesteps = n_timesteps
        self.schedule_type = schedule_type

        # Generate beta schedule
        if schedule_type == 'linear':
            betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif schedule_type == 'cosine':
            # Improved cosine schedule from Improved DDPM paper
            steps = n_timesteps + 1
            x = torch.linspace(0, n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / n_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        elif schedule_type == 'quadratic':
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Precompute useful quantities
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion process (add noise).

        Parameters
        ----------
        x_start : torch.Tensor
            Clean data, shape (batch, *).
        t : torch.Tensor
            Timestep indices, shape (batch,).
        noise : torch.Tensor, optional
            Noise to add (if None, sample from N(0,I)).

        Returns
        -------
        torch.Tensor
            Noisy data at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_start.dim() - 1)))
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_start.dim() - 1)))

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reverse diffusion process (denoise one step).

        Parameters
        ----------
        model : nn.Module
            Denoising model.
        x_t : torch.Tensor
            Noisy data at timestep t.
        t : torch.Tensor
            Timestep indices.
        condition : torch.Tensor, optional
            Conditioning information.

        Returns
        -------
        torch.Tensor
            Denoised data at timestep t-1.
        """
        # Predict noise
        if condition is not None:
            predicted_noise = model(x_t, t, condition)
        else:
            predicted_noise = model(x_t, t)

        # Compute coefficients
        alpha_t = self.alphas[t].view(-1, *([1] * (x_t.dim() - 1)))
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, *([1] * (x_t.dim() - 1)))
        beta_t = self.betas[t].view(-1, *([1] * (x_t.dim() - 1)))
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_t.dim() - 1)))

        # Compute x_{t-1}
        model_mean = (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t) / torch.sqrt(alpha_t)

        if t[0] > 0:
            # Add noise for t > 0
            posterior_variance_t = self.posterior_variance[t].view(-1, *([1] * (x_t.dim() - 1)))
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean


class SimpleUNet(nn.Module):
    """Simple UNet for diffusion denoising.

    Parameters
    ----------
    dim : int
        Base dimension.
    dim_mults : tuple, optional
        Dimension multipliers for each level.
        Default: (1, 2, 4).
    n_heads : int, optional
        Number of attention heads.
        Default: 8.
    condition_dim : int, optional
        Dimension of conditioning vector.
        Default: None.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        dim: int = 256,
        dim_mults: tuple = (1, 2, 4),
        n_heads: int = 8,
        condition_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition embedding (if provided)
        if condition_dim is not None:
            self.condition_proj = nn.Linear(condition_dim, time_dim)
        else:
            self.condition_proj = None

        # Down blocks
        self.downs = nn.ModuleList([])
        in_dim = dim
        for mult in dim_mults:
            out_dim = dim * mult
            self.downs.append(nn.Sequential(
                nn.Linear(in_dim + time_dim, out_dim),
                nn.SiLU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout),
            ))
            in_dim = out_dim

        # Middle
        mid_dim = in_dim
        self.mid = nn.Sequential(
            nn.Linear(mid_dim + time_dim, mid_dim),
            nn.SiLU(),
            nn.LayerNorm(mid_dim),
        )

        # Up blocks (process in reverse order, matching skip connections)
        self.ups = nn.ModuleList([])
        # Build list of dims for up path
        up_dims = [dim * mult for mult in reversed(dim_mults)]

        for i, out_dim in enumerate(up_dims):
            # Input dimension: current h + skip connection (which matches output of corresponding down block)
            skip_dim = dim * list(reversed(dim_mults))[i]
            self.ups.append(nn.Sequential(
                nn.Linear(in_dim + skip_dim + time_dim, out_dim),
                nn.SiLU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout),
            ))
            in_dim = out_dim

        # Final projection
        self.final = nn.Linear(in_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input, shape (batch, dim).
        t : torch.Tensor
            Timestep, shape (batch,).
        condition : torch.Tensor, optional
            Conditioning, shape (batch, condition_dim).

        Returns
        -------
        torch.Tensor
            Predicted noise, shape (batch, dim).
        """
        # Time embedding with sinusoidal encoding
        t_emb = self._sinusoidal_embedding(t, self.dim)
        t_emb = self.time_mlp(t_emb)

        # Add condition if provided
        if condition is not None and self.condition_proj is not None:
            t_emb = t_emb + self.condition_proj(condition)

        # Downsampling
        skip_connections = []
        h = x
        for down_block in self.downs:
            # Concatenate with time embedding
            h_with_time = torch.cat([h, t_emb], dim=-1)
            h = down_block(h_with_time)
            skip_connections.append(h)

        # Middle
        h_with_time = torch.cat([h, t_emb], dim=-1)
        h = self.mid(h_with_time)

        # Upsampling with skip connections
        for up_block, skip in zip(self.ups, reversed(skip_connections)):
            h_with_skip_time = torch.cat([h, skip, t_emb], dim=-1)
            h = up_block(h_with_skip_time)

        # Final projection
        return self.final(h)

    def _sinusoidal_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal time embeddings."""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class LatentDiffusionModel(nn.Module):
    """Latent diffusion model for neural forecasting.

    Parameters
    ----------
    latent_dim : int
        Dimension of latent space.
    n_timesteps : int, optional
        Number of diffusion timesteps.
        Default: 1000.
    schedule_type : str, optional
        Noise schedule type.
        Default: 'cosine'.
    condition_dim : int, optional
        Dimension of conditioning (e.g., from past latents).
        Default: None.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        n_timesteps: int = 1000,
        schedule_type: str = 'cosine',
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_timesteps = n_timesteps

        # Diffusion schedule
        self.schedule = DiffusionSchedule(
            n_timesteps=n_timesteps,
            schedule_type=schedule_type,
        )

        # Denoising network (UNet)
        self.denoiser = SimpleUNet(
            dim=latent_dim,
            dim_mults=(1, 2, 4),
            condition_dim=condition_dim,
        )

    def forward(
        self,
        x_start: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Parameters
        ----------
        x_start : torch.Tensor
            Clean latent vectors, shape (batch, latent_dim).
        condition : torch.Tensor, optional
            Conditioning information.

        Returns
        -------
        loss : torch.Tensor
            Diffusion loss (MSE).
        predicted_noise : torch.Tensor
            Predicted noise for inspection.
        """
        batch_size = x_start.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_start.device)

        # Sample noise
        noise = torch.randn_like(x_start)

        # Add noise to x_start
        x_noisy = self.schedule.q_sample(x_start, t, noise)

        # Predict noise
        predicted_noise = self.denoiser(x_noisy, t, condition)

        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss, predicted_noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: Optional[torch.Tensor] = None,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """Generate samples from noise.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        condition : torch.Tensor, optional
            Conditioning information.
        device : str, optional
            Device to generate on.

        Returns
        -------
        torch.Tensor
            Generated latent vectors, shape (batch, latent_dim).
        """
        # Start from pure noise
        x = torch.randn(batch_size, self.latent_dim, device=device)

        # Iteratively denoise
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.schedule.p_sample(self.denoiser, x, t_batch, condition)

        return x

    @torch.no_grad()
    def forecast(
        self,
        context: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Forecast future latent states.

        Parameters
        ----------
        context : torch.Tensor
            Past latent states for conditioning, shape (batch, latent_dim).
        n_steps : int, optional
            Number of future steps to forecast.
            Default: 10.

        Returns
        -------
        torch.Tensor
            Forecasted latent states, shape (batch, n_steps, latent_dim).
        """
        batch_size = context.shape[0]
        forecasts = []

        # Use context as conditioning
        current_condition = context

        for _ in range(n_steps):
            # Generate next step
            next_latent = self.sample(
                batch_size=batch_size,
                condition=current_condition,
                device=context.device,
            )
            forecasts.append(next_latent)

            # Update condition (use generated as new context)
            current_condition = next_latent

        # Stack forecasts
        forecasts = torch.stack(forecasts, dim=1)
        return forecasts
