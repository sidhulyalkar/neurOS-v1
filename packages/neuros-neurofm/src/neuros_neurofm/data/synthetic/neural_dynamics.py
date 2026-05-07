"""
Neural Dynamics Dataset for ENGRAM-FMx.

Generates synthetic neural activity from latent dynamical systems.
Tests the model's ability to learn and forecast neural dynamics.

Supports multiple dynamical systems:
- Linear oscillator
- Lorenz attractor
- Random RNN dynamics
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Literal
import math


class NeuralDynamicsDataset(Dataset):
    """Neural dynamics task dataset.

    Generates neural activity from latent dynamical systems.
    Task: Predict future neural activity given past activity.

    Parameters
    ----------
    num_samples : int
        Number of samples (trajectories) to generate. Default: 10000.
    seq_length : int
        Total sequence length. Default: 256.
    hidden_dim : int
        Neural activity dimension (number of "neurons"). Default: 128.
    latent_dim : int
        Latent dynamical system dimension. Default: 16.
    forecast_horizon : int
        Number of timesteps to forecast. Default: 16.
    dynamics_type : str
        Type of dynamics: "oscillator", "lorenz", "rnn". Default: "oscillator".
    noise_std : float
        Observation noise standard deviation. Default: 0.1.
    dt : float
        Time step for dynamics integration. Default: 0.05.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 256,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        forecast_horizon: int = 16,
        dynamics_type: Literal["oscillator", "lorenz", "rnn"] = "oscillator",
        noise_std: float = 0.1,
        dt: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.forecast_horizon = forecast_horizon
        self.dynamics_type = dynamics_type
        self.noise_std = noise_std
        self.dt = dt

        if seed is not None:
            torch.manual_seed(seed)

        # Observation matrix: latent -> neural activity
        self.C = torch.randn(hidden_dim, latent_dim) / math.sqrt(latent_dim)

        # Initialize dynamics parameters
        self._init_dynamics()

        # Pre-generate samples
        self._generate_samples()

    def _init_dynamics(self):
        """Initialize dynamics-specific parameters."""
        if self.dynamics_type == "oscillator":
            # Coupled oscillator dynamics
            # Create block-diagonal rotation matrices with different frequencies
            self.A = torch.zeros(self.latent_dim, self.latent_dim)
            num_oscillators = self.latent_dim // 2

            for i in range(num_oscillators):
                freq = 0.5 + i * 0.3  # Different frequencies
                idx = i * 2
                # 2D rotation matrix
                self.A[idx, idx] = math.cos(freq * self.dt)
                self.A[idx, idx + 1] = -math.sin(freq * self.dt)
                self.A[idx + 1, idx] = math.sin(freq * self.dt)
                self.A[idx + 1, idx + 1] = math.cos(freq * self.dt)

            # Add slight damping
            self.A = self.A * 0.999

        elif self.dynamics_type == "lorenz":
            # Lorenz attractor parameters (only use first 3 dims, rest are driven)
            self.sigma = 10.0
            self.rho = 28.0
            self.beta = 8.0 / 3.0

        elif self.dynamics_type == "rnn":
            # Random RNN dynamics
            self.W = torch.randn(self.latent_dim, self.latent_dim) / math.sqrt(self.latent_dim)
            # Ensure stability (spectral radius < 1)
            eigvals = torch.linalg.eigvals(self.W)
            spectral_radius = eigvals.abs().max().item()
            self.W = self.W / (spectral_radius + 0.1) * 0.95

    def _step_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """Advance latent state by one timestep.

        Parameters
        ----------
        z : torch.Tensor
            Current latent state [latent_dim] or [batch, latent_dim].

        Returns
        -------
        torch.Tensor
            Next latent state.
        """
        if self.dynamics_type == "oscillator":
            return z @ self.A.T

        elif self.dynamics_type == "lorenz":
            # Lorenz for first 3 dims
            x, y, zz = z[..., 0], z[..., 1], z[..., 2]

            dx = self.sigma * (y - x) * self.dt
            dy = (x * (self.rho - zz) - y) * self.dt
            dzz = (x * y - self.beta * zz) * self.dt

            z_new = z.clone()
            z_new[..., 0] = x + dx
            z_new[..., 1] = y + dy
            z_new[..., 2] = zz + dzz

            # Other dims: simple decay
            if self.latent_dim > 3:
                z_new[..., 3:] = z[..., 3:] * 0.99

            return z_new

        elif self.dynamics_type == "rnn":
            return torch.tanh(z @ self.W.T)

        else:
            raise ValueError(f"Unknown dynamics type: {self.dynamics_type}")

    def _generate_samples(self):
        """Pre-generate all trajectories."""
        self.trajectories = []

        for _ in range(self.num_samples):
            # Random initial condition
            z = torch.randn(self.latent_dim)

            if self.dynamics_type == "lorenz":
                # Scale initial condition for Lorenz
                z = z * 10.0

            # Generate trajectory
            latent_trajectory = torch.zeros(self.seq_length, self.latent_dim)

            for t in range(self.seq_length):
                latent_trajectory[t] = z
                z = self._step_dynamics(z)

            # Project to neural activity and add noise
            neural_activity = latent_trajectory @ self.C.T
            neural_activity = neural_activity + torch.randn_like(neural_activity) * self.noise_std

            self.trajectories.append(neural_activity)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - input_sequence: [seq_length, hidden_dim]
            - target_sequence: [seq_length, hidden_dim] (shifted by forecast_horizon)
            - mask: [seq_length] - True where prediction should be made
        """
        trajectory = self.trajectories[idx]

        # Input: full trajectory
        input_seq = trajectory.clone()

        # Target: shifted trajectory for forecasting
        target_seq = torch.zeros_like(trajectory)
        mask = torch.zeros(self.seq_length, dtype=torch.bool)

        # For each position t, predict position t + forecast_horizon
        for t in range(self.seq_length - self.forecast_horizon):
            target_seq[t] = trajectory[t + self.forecast_horizon]
            mask[t] = True

        return input_seq, target_seq, mask

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        inputs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        masks = torch.stack([b[2] for b in batch])
        return inputs, targets, masks
