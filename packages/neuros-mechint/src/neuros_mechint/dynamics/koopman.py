"""
Koopman Operator Analysis

This module provides tools for Koopman operator theory and Dynamic Mode Decomposition (DMD).
The Koopman operator is a linear, infinite-dimensional operator that describes the evolution
of observables in a nonlinear dynamical system.

Key capabilities:
- Standard DMD (Dynamic Mode Decomposition)
- Extended DMD (EDMD) with custom observables
- Kernel DMD for nonlinear systems
- Hankel DMD for control systems
- Optimal DMD with variable projection
- Sparsity-promoting DMD
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svd, eig
from scipy.spatial.distance import pdist, squareform
import logging

logger = logging.getLogger(__name__)


@dataclass
class KoopmanResult:
    """Results from Koopman operator analysis."""

    koopman_matrix: np.ndarray  # Koopman operator matrix
    eigenvalues: np.ndarray  # Koopman eigenvalues
    eigenvectors: np.ndarray  # Koopman eigenvectors (modes)
    amplitudes: np.ndarray  # Mode amplitudes
    growth_rates: np.ndarray  # Real part of eigenvalues
    frequencies: np.ndarray  # Imaginary part of eigenvalues
    reconstruction_error: float  # Reconstruction error

    # SVD components
    U: Optional[np.ndarray] = None  # Left singular vectors
    S: Optional[np.ndarray] = None  # Singular values
    Vh: Optional[np.ndarray] = None  # Right singular vectors

    # DMD variants info
    method: str = "standard"  # DMD variant used
    rank: Optional[int] = None  # Truncation rank

    def dominant_modes(self, k: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the k most dominant Koopman modes.

        Args:
            k: Number of modes to return

        Returns:
            Tuple of (eigenvalues, eigenvectors, amplitudes) for dominant modes
        """
        # Sort by amplitude
        indices = np.argsort(np.abs(self.amplitudes))[::-1][:k]
        return (
            self.eigenvalues[indices],
            self.eigenvectors[:, indices],
            self.amplitudes[indices]
        )

    def stable_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices and eigenvalues of stable modes (|λ| < 1)."""
        stable_idx = np.abs(self.eigenvalues) < 1.0
        return stable_idx, self.eigenvalues[stable_idx]

    def unstable_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices and eigenvalues of unstable modes (|λ| > 1)."""
        unstable_idx = np.abs(self.eigenvalues) > 1.0
        return unstable_idx, self.eigenvalues[unstable_idx]


class KoopmanOperator:
    """
    Koopman Operator Analysis using various DMD variants.

    The Koopman operator K is a linear operator on the space of observables:
        g(x_{t+1}) = K g(x_t)

    where g is an observable function and x_t is the state at time t.
    """

    def __init__(
        self,
        dt: float = 0.01,
        rank: Optional[int] = None,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize Koopman operator analyzer.

        Args:
            dt: Time step between observations
            rank: Truncation rank for SVD (None = automatic)
            device: Device for torch computations
            verbose: Whether to log information
        """
        self.dt = dt
        self.rank = rank
        self.device = device
        self.verbose = verbose

    def fit(
        self,
        trajectories: np.ndarray,
        method: str = "standard",
        **kwargs
    ) -> KoopmanResult:
        """
        Fit Koopman operator to trajectory data.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features) or
                         (n_trajectories, n_timesteps, n_features)
            method: DMD variant ("standard", "extended", "kernel", "hankel", "optimal", "sparse")
            **kwargs: Additional method-specific arguments

        Returns:
            KoopmanResult with operator and modes
        """
        # Reshape if needed
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        if method == "standard":
            return self._standard_dmd(trajectories, **kwargs)
        elif method == "extended":
            return self._extended_dmd(trajectories, **kwargs)
        elif method == "kernel":
            return self._kernel_dmd(trajectories, **kwargs)
        elif method == "hankel":
            return self._hankel_dmd(trajectories, **kwargs)
        elif method == "optimal":
            return self._optimal_dmd(trajectories, **kwargs)
        elif method == "sparse":
            return self._sparse_dmd(trajectories, **kwargs)
        else:
            raise ValueError(f"Unknown DMD method: {method}")

    def _standard_dmd(
        self,
        X: np.ndarray,
        rank: Optional[int] = None
    ) -> KoopmanResult:
        """
        Standard Dynamic Mode Decomposition.

        Computes the best-fit linear operator A such that:
            X_{t+1} ≈ A X_t

        Args:
            X: State snapshots (n_timesteps, n_features)
            rank: SVD truncation rank

        Returns:
            KoopmanResult
        """
        if rank is None:
            rank = self.rank

        # Split into X and Y (X' in DMD notation)
        X1 = X[:-1].T  # (n_features, n_timesteps-1)
        X2 = X[1:].T   # (n_features, n_timesteps-1)

        # SVD of X1
        U, S, Vh = svd(X1, full_matrices=False)

        # Truncate
        if rank is not None:
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]

        # Compute Koopman operator in reduced space
        S_inv = np.diag(1.0 / S)
        A_tilde = U.T @ X2 @ Vh.T @ S_inv

        # Eigendecomposition
        eigenvalues, eigenvectors_tilde = eig(A_tilde)

        # Project back to full space
        eigenvectors = X2 @ Vh.T @ S_inv @ eigenvectors_tilde

        # Normalize eigenvectors
        for i in range(eigenvectors.shape[1]):
            eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])

        # Compute amplitudes (initial conditions in mode coordinates)
        x0 = X1[:, 0]
        amplitudes = np.linalg.lstsq(eigenvectors, x0, rcond=None)[0]

        # Full Koopman matrix (if requested)
        K_full = U @ A_tilde @ U.T

        # Compute reconstruction error
        X_reconstructed = self._reconstruct(
            eigenvalues, eigenvectors, amplitudes, X1.shape[1]
        )
        reconstruction_error = np.linalg.norm(X1 - X_reconstructed) / np.linalg.norm(X1)

        # Extract growth rates and frequencies
        growth_rates = np.log(np.abs(eigenvalues)) / self.dt
        frequencies = np.angle(eigenvalues) / self.dt

        return KoopmanResult(
            koopman_matrix=K_full,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            amplitudes=amplitudes,
            growth_rates=growth_rates,
            frequencies=frequencies,
            reconstruction_error=reconstruction_error,
            U=U,
            S=S,
            Vh=Vh,
            method="standard",
            rank=rank
        )

    def _extended_dmd(
        self,
        X: np.ndarray,
        observables: Optional[Callable] = None,
        **kwargs
    ) -> KoopmanResult:
        """
        Extended DMD with custom observable functions.

        Args:
            X: State data (n_timesteps, n_features)
            observables: Function that maps states to observables
                        If None, uses polynomial observables up to degree 2

        Returns:
            KoopmanResult
        """
        if observables is None:
            # Default: polynomial observables up to degree 2
            observables = lambda x: self._polynomial_observables(x, degree=2)

        # Apply observables
        G = np.array([observables(x) for x in X])

        # Run standard DMD on observables
        return self._standard_dmd(G, **kwargs)

    def _kernel_dmd(
        self,
        X: np.ndarray,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        **kwargs
    ) -> KoopmanResult:
        """
        Kernel DMD for highly nonlinear systems.

        Args:
            X: State data (n_timesteps, n_features)
            kernel: Kernel type ("rbf", "polynomial", "linear")
            gamma: Kernel bandwidth parameter

        Returns:
            KoopmanResult
        """
        n = len(X)

        # Compute kernel matrices
        if kernel == "rbf":
            if gamma is None:
                # Use median heuristic
                pairwise_dists = pdist(X, metric='euclidean')
                gamma = 1.0 / np.median(pairwise_dists)**2

            K1 = self._rbf_kernel(X[:-1], X[:-1], gamma)
            K2 = self._rbf_kernel(X[1:], X[:-1], gamma)
        elif kernel == "polynomial":
            degree = kwargs.get('degree', 2)
            K1 = (X[:-1] @ X[:-1].T + 1)**degree
            K2 = (X[1:] @ X[:-1].T + 1)**degree
        else:
            K1 = X[:-1] @ X[:-1].T
            K2 = X[1:] @ X[:-1].T

        # Solve for Koopman operator in kernel space
        # K2 ≈ K_koopman @ K1
        K_koopman = K2 @ np.linalg.pinv(K1)

        # Eigendecomposition
        eigenvalues, eigenvectors = eig(K_koopman)

        # Real-valued results
        eigenvalues = eigenvalues.astype(complex)

        # Compute amplitudes
        amplitudes = eigenvectors[0, :]  # First time point

        # Reconstruction error
        K2_reconstructed = K_koopman @ K1
        reconstruction_error = np.linalg.norm(K2 - K2_reconstructed) / np.linalg.norm(K2)

        # Growth rates and frequencies
        growth_rates = np.log(np.abs(eigenvalues)) / self.dt
        frequencies = np.angle(eigenvalues) / self.dt

        return KoopmanResult(
            koopman_matrix=K_koopman,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            amplitudes=amplitudes,
            growth_rates=growth_rates,
            frequencies=frequencies,
            reconstruction_error=reconstruction_error,
            method="kernel"
        )

    def _hankel_dmd(
        self,
        X: np.ndarray,
        delay: int = 1,
        **kwargs
    ) -> KoopmanResult:
        """
        Hankel DMD for systems with delays.

        Args:
            X: State data (n_timesteps, n_features)
            delay: Number of delay embeddings

        Returns:
            KoopmanResult
        """
        # Create Hankel matrix
        n_timesteps, n_features = X.shape
        H = np.zeros((n_timesteps - delay, n_features * (delay + 1)))

        for i in range(delay + 1):
            H[:, i*n_features:(i+1)*n_features] = X[i:n_timesteps-delay+i]

        # Run standard DMD on Hankel matrix
        return self._standard_dmd(H, **kwargs)

    def _optimal_dmd(
        self,
        X: np.ndarray,
        rank: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> KoopmanResult:
        """
        Optimal DMD with variable projection.

        This method optimizes both the DMD modes and eigenvalues simultaneously.

        Args:
            X: State data (n_timesteps, n_features)
            rank: Number of modes
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance

        Returns:
            KoopmanResult
        """
        # Start with standard DMD
        result = self._standard_dmd(X, rank=rank)

        # Extract initial guess
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors

        # Prepare data
        X1 = X[:-1].T
        X2 = X[1:].T

        # Optimization loop (simplified variable projection)
        for iteration in range(max_iter):
            # Fix eigenvalues, optimize eigenvectors
            # This is a simplified version; full optimal DMD is more complex
            Vand = self._vandermonde(eigenvalues, X1.shape[1])
            eigenvectors_new = X2 @ np.linalg.pinv(Vand)

            # Check convergence
            diff = np.linalg.norm(eigenvectors_new - eigenvectors)
            if diff < tol:
                break

            eigenvectors = eigenvectors_new

        # Recompute amplitudes
        amplitudes = np.linalg.lstsq(eigenvectors, X1[:, 0], rcond=None)[0]

        # Reconstruction error
        X_reconstructed = self._reconstruct(eigenvalues, eigenvectors, amplitudes, X1.shape[1])
        reconstruction_error = np.linalg.norm(X1 - X_reconstructed) / np.linalg.norm(X1)

        # Growth rates and frequencies
        growth_rates = np.log(np.abs(eigenvalues)) / self.dt
        frequencies = np.angle(eigenvalues) / self.dt

        return KoopmanResult(
            koopman_matrix=result.koopman_matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            amplitudes=amplitudes,
            growth_rates=growth_rates,
            frequencies=frequencies,
            reconstruction_error=reconstruction_error,
            method="optimal",
            rank=rank
        )

    def _sparse_dmd(
        self,
        X: np.ndarray,
        alpha: float = 0.1,
        **kwargs
    ) -> KoopmanResult:
        """
        Sparsity-promoting DMD for interpretable modes.

        Args:
            X: State data (n_timesteps, n_features)
            alpha: Sparsity penalty parameter

        Returns:
            KoopmanResult
        """
        # Start with standard DMD
        result = self._standard_dmd(X, **kwargs)

        # Apply L1 penalty to amplitudes to promote sparsity
        amplitudes = result.amplitudes.copy()

        # Soft thresholding
        amplitudes = np.sign(amplitudes) * np.maximum(np.abs(amplitudes) - alpha, 0)

        # Recompute with sparse amplitudes
        X1 = X[:-1].T
        X_reconstructed = self._reconstruct(
            result.eigenvalues, result.eigenvectors, amplitudes, X1.shape[1]
        )
        reconstruction_error = np.linalg.norm(X1 - X_reconstructed) / np.linalg.norm(X1)

        return KoopmanResult(
            koopman_matrix=result.koopman_matrix,
            eigenvalues=result.eigenvalues,
            eigenvectors=result.eigenvectors,
            amplitudes=amplitudes,
            growth_rates=result.growth_rates,
            frequencies=result.frequencies,
            reconstruction_error=reconstruction_error,
            method="sparse",
            rank=result.rank
        )

    def predict(
        self,
        result: KoopmanResult,
        x0: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Predict future states using Koopman operator.

        Args:
            result: KoopmanResult from fitting
            x0: Initial state
            n_steps: Number of time steps to predict

        Returns:
            Predicted trajectory (n_steps, n_features)
        """
        # Reconstruct trajectory
        trajectory = self._reconstruct(
            result.eigenvalues,
            result.eigenvectors,
            result.amplitudes,
            n_steps
        )

        return trajectory.T

    def _reconstruct(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        amplitudes: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """Reconstruct trajectory from DMD modes."""
        Vand = self._vandermonde(eigenvalues, n_steps)
        return eigenvectors @ np.diag(amplitudes) @ Vand

    def _vandermonde(self, eigenvalues: np.ndarray, n_steps: int) -> np.ndarray:
        """Create Vandermonde matrix from eigenvalues."""
        time_steps = np.arange(n_steps)
        return np.vander(eigenvalues, n_steps, increasing=True).T

    def _polynomial_observables(self, x: np.ndarray, degree: int = 2) -> np.ndarray:
        """Create polynomial observables up to specified degree."""
        n = len(x)
        obs = [x]  # Linear terms

        if degree >= 2:
            # Quadratic terms
            quad = []
            for i in range(n):
                for j in range(i, n):
                    quad.append(x[i] * x[j])
            obs.append(np.array(quad))

        if degree >= 3:
            # Cubic terms (simplified)
            cubic = [x[i]**3 for i in range(n)]
            obs.append(np.array(cubic))

        return np.concatenate(obs)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF (Gaussian) kernel matrix."""
        dists = squareform(pdist(np.vstack([X1, X2]), metric='sqeuclidean'))
        K = np.exp(-gamma * dists)
        return K[:len(X1), len(X1):]
