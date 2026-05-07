"""
Spectral Operator Dynamics for ENGRAM-FMx.

Implements neural-operator style latent field update using
FFT-based spectral mixing over latent slots.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SpectralOperatorDynamics(nn.Module):
    """Spectral neural operator for latent dynamics.

    Applies FFT-based spectral mixing over the latent slot dimension:
        z_fft = FFT(z, dim=slot)
        z_fft[:modes] = R @ z_fft[:modes]  # learned mode transform
        z' = IFFT(z_fft)
        output = z + MLP(LayerNorm(z'))

    This gives the latent workspace a global structured update
    that is different from attention.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension (D).
    num_latents : int
        Number of latent slots (K). Default: 64.
    operator_modes : int
        Number of spectral modes to transform. Default: 16.
    dropout : float
        Dropout probability. Default: 0.1.
    layer_norm_eps : float
        LayerNorm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_latents: int = 64,
        operator_modes: int = 16,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.operator_modes = min(operator_modes, num_latents // 2 + 1)

        # Learnable complex weights for spectral modes
        # For real FFT, we have K//2 + 1 frequency bins
        # We learn separate real and imaginary transforms
        self.mode_weights_real = nn.Parameter(
            torch.randn(self.operator_modes, hidden_dim, hidden_dim) * 0.02
        )
        self.mode_weights_imag = nn.Parameter(
            torch.randn(self.operator_modes, hidden_dim, hidden_dim) * 0.02
        )

        # Layer norm and MLP for residual path
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass: apply spectral operator dynamics.

        Parameters
        ----------
        z : torch.Tensor
            Latent states, shape [B, K, D].

        Returns
        -------
        Tuple[torch.Tensor, dict]
            - Output tensor [B, K, D]
            - Diagnostics dict
        """
        B, K, D = z.shape

        # Apply real FFT over latent slot dimension
        # z: [B, K, D] -> z_fft: [B, K_freq, D] where K_freq = K//2 + 1
        z_fft = torch.fft.rfft(z, dim=1)  # Complex tensor [B, K_freq, D]
        K_freq = z_fft.shape[1]

        # Determine how many modes to transform
        modes_to_use = min(self.operator_modes, K_freq)

        # Extract modes to transform
        z_fft_low = z_fft[:, :modes_to_use, :]  # [B, modes, D]

        # Apply learned transform to each mode
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        z_real = z_fft_low.real  # [B, modes, D]
        z_imag = z_fft_low.imag  # [B, modes, D]

        # Transform: W_real and W_imag are [modes, D, D]
        # For each mode m: out_real = W_real[m] @ z_real - W_imag[m] @ z_imag
        #                  out_imag = W_real[m] @ z_imag + W_imag[m] @ z_real
        out_real = torch.einsum("bmd,mde->bme", z_real, self.mode_weights_real[:modes_to_use]) - \
                   torch.einsum("bmd,mde->bme", z_imag, self.mode_weights_imag[:modes_to_use])
        out_imag = torch.einsum("bmd,mde->bme", z_imag, self.mode_weights_real[:modes_to_use]) + \
                   torch.einsum("bmd,mde->bme", z_real, self.mode_weights_imag[:modes_to_use])

        # Reconstruct complex tensor
        z_fft_transformed = torch.complex(out_real, out_imag)

        # Replace low modes in original spectrum
        z_fft_out = z_fft.clone()
        z_fft_out[:, :modes_to_use, :] = z_fft_transformed

        # Inverse FFT back to spatial domain
        z_ifft = torch.fft.irfft(z_fft_out, n=K, dim=1)  # [B, K, D]

        # Residual MLP
        z_out = z + self.mlp(self.norm(z_ifft))

        # Compute diagnostics
        # Spectral energy by mode
        mode_energy = (z_fft.abs() ** 2).mean(dim=(0, 2))  # [K_freq]
        total_energy = mode_energy.sum().item()
        low_mode_energy = mode_energy[:modes_to_use].sum().item()
        energy_ratio = low_mode_energy / (total_energy + 1e-10)

        diagnostics = {
            "operator_total_spectral_energy": total_energy,
            "operator_low_mode_energy_ratio": energy_ratio,
            "operator_output_norm": z_out.norm(dim=-1).mean().item(),
        }

        return z_out, diagnostics
