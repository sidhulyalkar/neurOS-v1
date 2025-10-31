"""
Fractal Stimulus Generation

Generate fractal test signals for probing model responses and data augmentation.
Includes fBm, colored noise, multiplicative cascades, and fractal patterns.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FractionalBrownianMotion:
    """
    Fractional Brownian motion (fBm) generator.

    fBm is a generalization of Brownian motion with tunable Hurst exponent H.
    - H = 0.5: Standard Brownian motion
    - H < 0.5: Anti-persistent (mean-reverting)
    - H > 0.5: Persistent (trending)

    Uses Davies-Harte method for exact fBm generation.

    Args:
        H: Hurst exponent (default: 0.7)
        device: torch device

    Example:
        >>> fbm = FractionalBrownianMotion(H=0.7)
        >>> signal = fbm.generate(n_samples=1000, batch_size=32)
        >>> print(signal.shape)  # torch.Size([32, 1000])
    """

    def __init__(self, H: float = 0.7, device: Optional[str] = None):
        self.H = H
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def generate(
        self,
        n_samples: int,
        batch_size: int = 1,
        n_dims: int = 1,
    ) -> Tensor:
        """
        Generate fBm using spectral synthesis.

        Args:
            n_samples: Number of time points
            batch_size: Number of independent realizations
            n_dims: Number of dimensions (channels)

        Returns:
            fBm signals [batch_size, n_samples, n_dims] or [batch_size, n_samples] if n_dims=1
        """
        # Generate Gaussian white noise
        noise = torch.randn(batch_size, n_samples, n_dims, device=self.device)

        # FFT of noise
        noise_fft = torch.fft.rfft(noise, dim=1)

        # Frequency vector
        freqs = torch.fft.rfftfreq(n_samples, device=self.device)

        # Power spectrum: S(f) ∝ f^(-2H-1)
        # Avoid division by zero at f=0
        spectrum = torch.zeros_like(freqs)
        spectrum[1:] = freqs[1:].pow(-self.H - 0.5)
        spectrum[0] = 0  # DC component

        # Apply spectrum
        weighted_fft = noise_fft * spectrum.view(1, -1, 1).sqrt()

        # Inverse FFT
        fbm = torch.fft.irfft(weighted_fft, n=n_samples, dim=1)

        # Cumulative sum to get fBm from fractional Gaussian noise
        fbm = torch.cumsum(fbm, dim=1)

        if n_dims == 1:
            fbm = fbm.squeeze(2)

        return fbm


class ColoredNoise:
    """
    Colored noise generator with 1/f^β power spectrum.

    - β = 0: White noise
    - β = 1: Pink noise (1/f)
    - β = 2: Brown/red noise

    Uses spectral synthesis method.

    Args:
        beta: Spectral exponent (default: 1.0)
        device: torch device

    Example:
        >>> pink_noise = ColoredNoise(beta=1.0)
        >>> signal = pink_noise.generate(n_samples=5000, batch_size=16)
    """

    def __init__(self, beta: float = 1.0, device: Optional[str] = None):
        self.beta = beta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def generate(
        self,
        n_samples: int,
        batch_size: int = 1,
        n_dims: int = 1,
    ) -> Tensor:
        """
        Generate colored noise.

        Args:
            n_samples: Number of time points
            batch_size: Number of independent realizations
            n_dims: Number of dimensions

        Returns:
            Colored noise [batch_size, n_samples, n_dims] or [batch_size, n_samples]
        """
        # Generate white noise
        noise = torch.randn(batch_size, n_samples, n_dims, device=self.device)

        # FFT
        noise_fft = torch.fft.rfft(noise, dim=1)

        # Frequency vector
        freqs = torch.fft.rfftfreq(n_samples, device=self.device)

        # Power spectrum: S(f) ∝ 1/f^β
        spectrum = torch.zeros_like(freqs)
        spectrum[1:] = freqs[1:].pow(-self.beta / 2.0)
        spectrum[0] = 1.0  # DC component

        # Apply spectrum
        weighted_fft = noise_fft * spectrum.view(1, -1, 1)

        # Inverse FFT
        colored = torch.fft.irfft(weighted_fft, n=n_samples, dim=1)

        if n_dims == 1:
            colored = colored.squeeze(2)

        return colored


class MultiplicativeCascade:
    """
    Multiplicative cascade for multifractal signal generation.

    Iteratively multiplies random weights at multiple scales to create
    heterogeneous scaling properties.

    Args:
        n_levels: Number of cascade levels (default: 10)
        branching: Branching factor (default: 2)
        device: torch device

    Example:
        >>> cascade = MultiplicativeCascade(n_levels=10, branching=2)
        >>> signal = cascade.generate(batch_size=8)
        >>> print(signal.shape)  # torch.Size([8, 1024])  (2^10 = 1024)
    """

    def __init__(
        self,
        n_levels: int = 10,
        branching: int = 2,
        device: Optional[str] = None,
    ):
        self.n_levels = n_levels
        self.branching = branching
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def generate(
        self,
        batch_size: int = 1,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generate multiplicative cascade.

        Args:
            batch_size: Number of independent cascades
            weights: Optional custom weights [batch, branching]
                    If None, uses uniform weights normalized to sum to 1

        Returns:
            Cascade measure [batch_size, branching^n_levels]
        """
        n_points = self.branching ** self.n_levels

        # Initialize with uniform measure
        measure = torch.ones(batch_size, 1, device=self.device)

        for level in range(self.n_levels):
            # Generate random weights for this level
            if weights is None:
                # Random weights from log-normal distribution
                W = torch.randn(batch_size, self.branching, device=self.device).exp()
                # Normalize to sum to 1
                W = W / W.sum(dim=1, keepdim=True)
            else:
                W = weights

            # Expand measure by multiplying with weights
            # Old measure: [batch, n]
            # New measure: [batch, n * branching]
            expanded = measure.unsqueeze(2) * W.unsqueeze(1)  # [batch, n, branching]
            measure = expanded.reshape(batch_size, -1)  # [batch, n * branching]

        return measure


class FractalPatterns:
    """
    2D fractal pattern generators for visual stimuli.

    Includes Mandelbrot set, Julia sets, IFS fractals, etc.

    Example:
        >>> patterns = FractalPatterns()
        >>> mandelbrot = patterns.generate_mandelbrot(size=256, zoom=1.0)
        >>> print(mandelbrot.shape)  # torch.Size([1, 256, 256])
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_mandelbrot(
        self,
        size: int = 256,
        zoom: float = 1.0,
        max_iter: int = 100,
        batch_size: int = 1,
    ) -> Tensor:
        """
        Generate Mandelbrot set fractal.

        Args:
            size: Image size (size x size)
            zoom: Zoom factor (default: 1.0)
            max_iter: Maximum iterations (default: 100)
            batch_size: Number of patterns with random perturbations

        Returns:
            Fractal patterns [batch, size, size]
        """
        # Create coordinate grid
        x = torch.linspace(-2.0 / zoom, 1.0 / zoom, size, device=self.device)
        y = torch.linspace(-1.5 / zoom, 1.5 / zoom, size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='xy')

        fractals = []

        for b in range(batch_size):
            # Small random perturbation for variation
            perturb = torch.randn(2, device=self.device) * 0.1
            C = X + 1j * Y + complex(perturb[0].item(), perturb[1].item())

            Z = torch.zeros_like(C)
            M = torch.zeros(size, size, device=self.device)

            for i in range(max_iter):
                mask = Z.abs() <= 2
                Z[mask] = Z[mask] ** 2 + C[mask]
                M[mask] = i

            fractals.append(M)

        return torch.stack(fractals, dim=0)

    def generate_julia(
        self,
        c: complex = complex(-0.7, 0.27015),
        size: int = 256,
        zoom: float = 1.0,
        max_iter: int = 100,
        batch_size: int = 1,
    ) -> Tensor:
        """
        Generate Julia set fractal.

        Args:
            c: Complex parameter for Julia set
            size: Image size
            zoom: Zoom factor
            max_iter: Maximum iterations
            batch_size: Number of patterns

        Returns:
            Fractal patterns [batch, size, size]
        """
        x = torch.linspace(-2.0 / zoom, 2.0 / zoom, size, device=self.device)
        y = torch.linspace(-2.0 / zoom, 2.0 / zoom, size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='xy')

        fractals = []

        for b in range(batch_size):
            Z = X + 1j * Y
            M = torch.zeros(size, size, device=self.device)

            for i in range(max_iter):
                mask = Z.abs() <= 2
                Z[mask] = Z[mask] ** 2 + c
                M[mask] = i

            fractals.append(M)

        return torch.stack(fractals, dim=0)

    def generate_sierpinski(
        self,
        size: int = 256,
        n_iterations: int = 8,
        batch_size: int = 1,
    ) -> Tensor:
        """
        Generate Sierpinski triangle using chaos game.

        Args:
            size: Image size
            n_iterations: Number of iterations
            batch_size: Number of patterns

        Returns:
            Fractal patterns [batch, size, size]
        """
        fractals = []

        # Triangle vertices
        vertices = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ], device=self.device)

        for b in range(batch_size):
            canvas = torch.zeros(size, size, device=self.device)

            # Starting point
            point = torch.rand(2, device=self.device)

            n_points = 2 ** n_iterations
            for _ in range(n_points):
                # Choose random vertex
                vertex_idx = torch.randint(0, 3, (1,), device=self.device).item()
                vertex = vertices[vertex_idx]

                # Move halfway to vertex
                point = (point + vertex) / 2.0

                # Mark point on canvas
                x = int(point[0] * (size - 1))
                y = int(point[1] * (size - 1))
                if 0 <= x < size and 0 <= y < size:
                    canvas[y, x] = 1.0

            fractals.append(canvas)

        return torch.stack(fractals, dim=0)


class FractalTimeSeries:
    """
    Combined fractal time series generator with multiple methods.

    Convenience class for generating various fractal signals.

    Example:
        >>> gen = FractalTimeSeries()
        >>> signals = gen.generate_mixed(n_samples=1000, batch_size=16)
        >>> # Returns dict with multiple signal types
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.fbm = FractionalBrownianMotion(device=self.device)
        self.colored_noise = ColoredNoise(device=self.device)
        self.cascade = MultiplicativeCascade(device=self.device)

    def generate_mixed(
        self,
        n_samples: int,
        batch_size: int = 1,
        H_range: Tuple[float, float] = (0.3, 0.9),
        beta_range: Tuple[float, float] = (0.5, 1.5),
    ) -> dict:
        """
        Generate mixed fractal signals with varying parameters.

        Args:
            n_samples: Number of time points
            batch_size: Number of signals
            H_range: Range of Hurst exponents
            beta_range: Range of spectral exponents

        Returns:
            Dictionary with multiple signal types
        """
        signals = {}

        # Generate fBm with random H for each sample
        H_values = torch.rand(batch_size, device=self.device) * (H_range[1] - H_range[0]) + H_range[0]
        fbm_signals = []
        for h in H_values:
            self.fbm.H = h.item()
            fbm_signals.append(self.fbm.generate(n_samples, batch_size=1))
        signals['fbm'] = torch.cat(fbm_signals, dim=0)

        # Generate colored noise with random β
        beta_values = torch.rand(batch_size, device=self.device) * (beta_range[1] - beta_range[0]) + beta_range[0]
        colored_signals = []
        for beta in beta_values:
            self.colored_noise.beta = beta.item()
            colored_signals.append(self.colored_noise.generate(n_samples, batch_size=1))
        signals['colored_noise'] = torch.cat(colored_signals, dim=0)

        # Pad cascade to n_samples if needed
        cascade_raw = self.cascade.generate(batch_size=batch_size)
        if cascade_raw.size(1) < n_samples:
            # Interpolate to desired length
            cascade_interp = torch.nn.functional.interpolate(
                cascade_raw.unsqueeze(1),
                size=n_samples,
                mode='linear',
                align_corners=False,
            ).squeeze(1)
        else:
            cascade_interp = cascade_raw[:, :n_samples]

        signals['cascade'] = cascade_interp

        return signals

    def generate_benchmark_suite(
        self,
        n_samples: int = 5000,
        batch_size: int = 10,
    ) -> dict:
        """
        Generate benchmark suite with known fractal properties for validation.

        Returns:
            Dictionary with signals and their ground-truth parameters
        """
        suite = {}

        # White noise (H=0.5, β=0)
        suite['white_noise'] = {
            'signal': torch.randn(batch_size, n_samples, device=self.device),
            'H': 0.5,
            'beta': 0.0,
            'fractal_dim': 1.5,
        }

        # Pink noise (H≈1, β=1)
        self.colored_noise.beta = 1.0
        suite['pink_noise'] = {
            'signal': self.colored_noise.generate(n_samples, batch_size),
            'H': 1.0,
            'beta': 1.0,
            'fractal_dim': 1.0,
        }

        # fBm with H=0.7
        self.fbm.H = 0.7
        suite['fbm_07'] = {
            'signal': self.fbm.generate(n_samples, batch_size),
            'H': 0.7,
            'beta': 1.4,
            'fractal_dim': 1.3,
        }

        # fBm with H=0.3 (anti-persistent)
        self.fbm.H = 0.3
        suite['fbm_03'] = {
            'signal': self.fbm.generate(n_samples, batch_size),
            'H': 0.3,
            'beta': 0.6,
            'fractal_dim': 1.7,
        }

        return suite
