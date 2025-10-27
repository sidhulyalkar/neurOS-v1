"""
Fractal Simulators for Biophysical Modeling

Biophysically-inspired fractal dynamical systems for synthetic data generation.
Includes fractional Ornstein-Uhlenbeck processes, fractal dendrite growth,
and scale-free network models.

These simulators generate realistic synthetic neural data with known fractal
properties, useful for validation, testing, and data augmentation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List, Dict
from collections import namedtuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FractionalOU:
    """
    Fractional Ornstein-Uhlenbeck process simulator.

    Models neural dynamics with long-range temporal correlations via
    fractional noise. The process satisfies:

        dX_t = θ(μ - X_t)dt + σ dB^H_t

    where B^H is fractional Brownian motion with Hurst exponent H.

    Args:
        alpha: Fractional exponent (0 < α ≤ 1). Related to H by α ≈ H for small increments.
        theta: Mean reversion rate (larger = faster return to mean)
        mu: Long-term mean level
        sigma: Noise intensity
        device: Torch device

    Example:
        >>> fou = FractionalOU(alpha=0.8, theta=1.0, sigma=0.1)
        >>> trajectory = fou.simulate(n_steps=10000, dt=0.001, batch_size=16)
        >>> print(trajectory.shape)  # torch.Size([16, 10000])
    """

    def __init__(
        self,
        alpha: float = 0.8,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 0.1,
        device: Optional[str] = None,
    ):
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        if theta <= 0:
            raise ValueError("theta must be positive")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")

        self.alpha = alpha
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def simulate(
        self,
        n_steps: int,
        dt: float = 0.001,
        batch_size: int = 1,
        x0: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Simulate fractional OU process using Euler-Maruyama with fractional noise.

        Args:
            n_steps: Number of time steps
            dt: Time step size
            batch_size: Number of independent trajectories
            x0: Initial condition [batch_size] or scalar (default: mu)

        Returns:
            Trajectories [batch_size, n_steps]
        """
        # Initialize
        if x0 is None:
            x = torch.full((batch_size,), self.mu, device=self.device, dtype=torch.float32)
        else:
            x = x0.to(self.device)
            if x.dim() == 0:
                x = x.expand(batch_size)

        # Generate fractional Gaussian noise (fGn) increments
        noise = torch.randn(batch_size, n_steps, device=self.device)
        noise_fft = torch.fft.rfft(noise, dim=1)
        freqs = torch.fft.rfftfreq(n_steps, device=self.device)

        # Fractional noise spectrum
        spectrum = torch.zeros_like(freqs)
        spectrum[1:] = freqs[1:].pow(-self.alpha - 0.5)
        spectrum[0] = 0

        # Apply spectrum
        weighted_fft = noise_fft * spectrum.unsqueeze(0).sqrt()
        fgn = torch.fft.irfft(weighted_fft, n=n_steps, dim=1)

        # Euler-Maruyama integration
        trajectory = torch.zeros(batch_size, n_steps, device=self.device)
        trajectory[:, 0] = x

        for t in range(1, n_steps):
            drift = self.theta * (self.mu - trajectory[:, t-1]) * dt
            diffusion = self.sigma * fgn[:, t] * np.sqrt(dt)
            trajectory[:, t] = trajectory[:, t-1] + drift + diffusion

        return trajectory


class DendriteGrowthSimulator:
    """
    Fractal dendrite growth simulator using stochastic L-systems.

    Generates dendritic trees with target fractal dimension via
    probabilistic branching rules.

    Args:
        target_fd: Target fractal dimension (typically 1.5-1.8)
        branching_prob: Probability of branching at each step
        branch_angle: Angle between branches (radians)
        length_decay: How much branches shrink per generation
        device: Torch device
    """

    def __init__(
        self,
        target_fd: float = 1.7,
        branching_prob: float = 0.3,
        branch_angle: float = np.pi / 4,
        length_decay: float = 0.8,
        device: Optional[str] = None,
    ):
        self.target_fd = target_fd
        self.branching_prob = branching_prob
        self.branch_angle = branch_angle
        self.length_decay = length_decay
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def grow(self, n_iterations: int, initial_length: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Grow dendritic tree using stochastic L-system."""
        positions = [torch.zeros(3, device=self.device)]
        connectivity_list = []

        BranchTip = namedtuple('BranchTip', ['pos', 'direction', 'length', 'parent'])
        tips = [BranchTip(
            pos=torch.zeros(3, device=self.device),
            direction=torch.tensor([0.0, 0.0, 1.0], device=self.device),
            length=initial_length,
            parent=0,
        )]

        for iteration in range(n_iterations):
            new_tips = []
            current_length = initial_length * (self.length_decay ** iteration)

            for tip in tips:
                new_pos = tip.pos + tip.direction * current_length
                new_idx = len(positions)
                positions.append(new_pos)
                connectivity_list.append((tip.parent, new_idx))

                if torch.rand(1).item() < self.branching_prob:
                    for angle_sign in [-1, 1]:
                        rotation_axis = torch.randn(3, device=self.device)
                        rotation_axis = rotation_axis / (rotation_axis.norm() + 1e-8)
                        angle = angle_sign * self.branch_angle
                        rotated_dir = (
                            tip.direction * np.cos(angle)
                            + torch.cross(rotation_axis, tip.direction) * np.sin(angle)
                        )
                        rotated_dir = rotated_dir / (rotated_dir.norm() + 1e-8)
                        new_tips.append(BranchTip(
                            pos=new_pos, direction=rotated_dir,
                            length=current_length * self.length_decay, parent=new_idx
                        ))
                else:
                    new_tips.append(BranchTip(
                        pos=new_pos, direction=tip.direction,
                        length=current_length * self.length_decay, parent=new_idx
                    ))

            tips = new_tips
            if not tips:
                break

        positions_tensor = torch.stack(positions, dim=0)
        n_nodes = len(positions)
        connectivity = torch.zeros(n_nodes, n_nodes, device=self.device)
        for parent, child in connectivity_list:
            connectivity[parent, child] = 1.0
            connectivity[child, parent] = 1.0

        return positions_tensor, connectivity


class FractalNetworkModel:
    """
    Scale-free network generator with power-law degree distribution.

    Args:
        n_nodes: Number of nodes
        gamma: Power-law exponent (typical: 2-3)
        method: 'preferential_attachment' or 'configuration'
        m: Edges to attach from new node (PA model)
        device: Torch device
    """

    def __init__(
        self,
        n_nodes: int,
        gamma: float = 2.5,
        method: str = 'preferential_attachment',
        m: int = 2,
        device: Optional[str] = None,
    ):
        self.n_nodes = n_nodes
        self.gamma = gamma
        self.method = method
        self.m = m
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def generate(self) -> Tensor:
        """Generate scale-free network."""
        if self.method == 'preferential_attachment':
            return self._preferential_attachment()
        elif self.method == 'configuration':
            return self._configuration_model()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _preferential_attachment(self) -> Tensor:
        """Barabási-Albert preferential attachment model."""
        adj = torch.zeros(self.n_nodes, self.n_nodes, device=self.device)
        for i in range(self.m + 1):
            for j in range(i + 1, self.m + 1):
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        for new_node in range(self.m + 1, self.n_nodes):
            degrees = adj.sum(dim=1)[:new_node]
            probs = degrees / degrees.sum() if degrees.sum() > 0 else torch.ones(new_node, device=self.device) / new_node
            targets = torch.multinomial(probs, self.m, replacement=False)
            for target in targets:
                adj[new_node, target] = 1.0
                adj[target, new_node] = 1.0

        return adj

    def _configuration_model(self) -> Tensor:
        """Configuration model with power-law degree sequence."""
        k_min, k_max = 1, self.n_nodes // 2
        degrees = []
        for _ in range(self.n_nodes):
            u = torch.rand(1).item()
            k = int(k_min * (1 - u) ** (-1.0 / (self.gamma - 1)))
            degrees.append(min(k, k_max))
        degrees = torch.tensor(degrees, device=self.device, dtype=torch.long)
        if degrees.sum() % 2 == 1:
            degrees[0] += 1

        stubs = []
        for node, deg in enumerate(degrees):
            stubs.extend([node] * deg.item())
        stubs = torch.tensor(stubs, device=self.device)
        perm = torch.randperm(len(stubs), device=self.device)
        stubs = stubs[perm]

        adj = torch.zeros(self.n_nodes, self.n_nodes, device=self.device)
        for i in range(0, len(stubs) - 1, 2):
            u, v = stubs[i].item(), stubs[i+1].item()
            if u != v:
                adj[u, v] = 1.0
                adj[v, u] = 1.0

        return adj
