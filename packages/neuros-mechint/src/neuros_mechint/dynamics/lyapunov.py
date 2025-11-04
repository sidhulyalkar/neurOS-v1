"""
Lyapunov Analysis for Stability and Chaos

This module provides comprehensive tools for computing Lyapunov exponents,
Lyapunov functions, and stability analysis of dynamical systems.

Key capabilities:
- Lyapunov exponent computation (multiple methods)
- Maximum Lyapunov exponent for chaos detection
- Lyapunov spectrum for full system characterization
- Lyapunov function estimation
- Finite-time Lyapunov exponents (FTLE)
- Local and global stability analysis
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_continuous_lyapunov, qr, norm
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger(__name__)


@dataclass
class LyapunovResult:
    """Results from Lyapunov analysis."""

    # Lyapunov exponents
    max_exponent: float  # Maximum Lyapunov exponent
    exponents: np.ndarray  # Full Lyapunov spectrum
    lyapunov_dimension: float  # Kaplan-Yorke dimension
    kolmogorov_entropy: float  # Kolmogorov-Sinai entropy

    # Classification
    is_chaotic: bool  # Whether system exhibits chaos
    is_stable: bool  # Whether system is stable

    # Additional metrics
    divergence_rate: Optional[float] = None  # Average divergence rate
    predictability_time: Optional[float] = None  # Time until predictions become unreliable

    # Method info
    method: str = "orthogonalization"  # Method used
    n_timesteps: Optional[int] = None  # Number of timesteps analyzed


@dataclass
class LyapunovFunctionResult:
    """Results from Lyapunov function estimation."""

    lyapunov_function: Callable  # Estimated Lyapunov function V(x)
    lyapunov_derivative: Callable  # Time derivative dV/dt
    stability_region: Optional[np.ndarray] = None  # Region where V̇ < 0
    max_level_set: Optional[float] = None  # Largest stable level set


class LyapunovAnalyzer:
    """
    Comprehensive Lyapunov analysis for dynamical systems.

    Lyapunov exponents quantify the rate of separation of infinitesimally
    close trajectories, providing a measure of chaos and predictability.
    """

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize Lyapunov analyzer.

        Args:
            dt: Time step between observations
            device: Device for torch computations
            verbose: Whether to log information
        """
        self.dt = dt
        self.device = device
        self.verbose = verbose

    def compute_exponents(
        self,
        trajectories: np.ndarray,
        method: str = "orthogonalization",
        **kwargs
    ) -> LyapunovResult:
        """
        Compute Lyapunov exponents.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features) or
                         (n_trajectories, n_timesteps, n_features)
            method: Computation method ("orthogonalization", "jacobian", "divergence", "ftle")
            **kwargs: Method-specific parameters

        Returns:
            LyapunovResult with exponents and characterization
        """
        # Reshape if needed
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        if method == "orthogonalization":
            return self._orthogonalization_method(trajectories, **kwargs)
        elif method == "jacobian":
            return self._jacobian_method(trajectories, **kwargs)
        elif method == "divergence":
            return self._divergence_method(trajectories, **kwargs)
        elif method == "ftle":
            return self._ftle_method(trajectories, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _orthogonalization_method(
        self,
        X: np.ndarray,
        n_neighbors: int = 10,
        evolution_time: int = 10
    ) -> LyapunovResult:
        """
        Compute Lyapunov exponents using orthogonalization method.

        This is the standard method that tracks the evolution of an orthonormal
        basis using QR decomposition.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            n_neighbors: Number of neighbors for Jacobian estimation
            evolution_time: Time steps for evolution

        Returns:
            LyapunovResult
        """
        n_timesteps, n_features = X.shape

        # Initialize orthonormal basis
        Q = np.eye(n_features)

        # Accumulate log of growth rates
        lyap_sum = np.zeros(n_features)
        n_steps = 0

        # Iterate through trajectory
        for t in range(0, n_timesteps - evolution_time, evolution_time):
            # Estimate local Jacobian
            J = self._estimate_jacobian(
                X[t:t+evolution_time+1],
                n_neighbors=n_neighbors
            )

            # Evolve tangent vectors
            Q_evolved = J @ Q

            # QR decomposition to orthogonalize
            Q, R = qr(Q_evolved)

            # Accumulate logarithms of diagonal elements
            lyap_sum += np.log(np.abs(np.diag(R)))
            n_steps += 1

        # Normalize by time
        exponents = lyap_sum / (n_steps * evolution_time * self.dt)

        # Maximum exponent
        max_exponent = exponents[0]

        # Lyapunov dimension (Kaplan-Yorke)
        lyapunov_dim = self._kaplan_yorke_dimension(exponents)

        # Kolmogorov entropy (sum of positive exponents)
        kolmogorov_entropy = np.sum(exponents[exponents > 0])

        # Classification
        is_chaotic = max_exponent > 0
        is_stable = max_exponent < 0

        # Predictability time (inverse of max exponent)
        if max_exponent > 0:
            predictability_time = 1.0 / max_exponent
        else:
            predictability_time = np.inf

        return LyapunovResult(
            max_exponent=max_exponent,
            exponents=exponents,
            lyapunov_dimension=lyapunov_dim,
            kolmogorov_entropy=kolmogorov_entropy,
            is_chaotic=is_chaotic,
            is_stable=is_stable,
            predictability_time=predictability_time,
            method="orthogonalization",
            n_timesteps=n_timesteps
        )

    def _jacobian_method(
        self,
        X: np.ndarray,
        n_neighbors: int = 10
    ) -> LyapunovResult:
        """
        Compute Lyapunov exponents from Jacobian eigenvalues.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            n_neighbors: Number of neighbors for Jacobian estimation

        Returns:
            LyapunovResult
        """
        n_timesteps, n_features = X.shape

        # Estimate Jacobian at multiple points
        exponents_list = []

        for t in range(0, n_timesteps - 1, max(1, n_timesteps // 100)):
            # Estimate local Jacobian
            J = self._estimate_jacobian(
                X[max(0, t-5):min(n_timesteps, t+6)],
                n_neighbors=n_neighbors
            )

            # Eigenvalues of Jacobian
            eigvals = np.linalg.eigvals(J)

            # Lyapunov exponents from eigenvalues
            lyap = np.log(np.abs(eigvals)) / self.dt

            exponents_list.append(lyap)

        # Average over trajectory
        exponents = np.mean(exponents_list, axis=0)
        exponents = np.sort(exponents)[::-1]  # Sort descending

        max_exponent = exponents[0]
        lyapunov_dim = self._kaplan_yorke_dimension(exponents)
        kolmogorov_entropy = np.sum(exponents[exponents > 0])

        is_chaotic = max_exponent > 0
        is_stable = max_exponent < 0

        if max_exponent > 0:
            predictability_time = 1.0 / max_exponent
        else:
            predictability_time = np.inf

        return LyapunovResult(
            max_exponent=max_exponent,
            exponents=exponents,
            lyapunov_dimension=lyapunov_dim,
            kolmogorov_entropy=kolmogorov_entropy,
            is_chaotic=is_chaotic,
            is_stable=is_stable,
            predictability_time=predictability_time,
            method="jacobian",
            n_timesteps=n_timesteps
        )

    def _divergence_method(
        self,
        X: np.ndarray,
        n_neighbors: int = 5,
        epsilon: float = 1e-8
    ) -> LyapunovResult:
        """
        Compute maximum Lyapunov exponent from trajectory divergence.

        This method tracks the divergence of nearby trajectories.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            n_neighbors: Number of neighbors to track
            epsilon: Initial separation threshold

        Returns:
            LyapunovResult
        """
        n_timesteps, n_features = X.shape

        # Find nearby trajectory pairs
        from scipy.spatial import cKDTree
        tree = cKDTree(X[:-1])

        divergence_rates = []

        for t in range(n_timesteps - 1):
            # Find nearest neighbors
            distances, indices = tree.query(X[t], k=n_neighbors + 1)

            # Skip self
            distances = distances[1:]
            indices = indices[1:]

            # Track divergence
            for idx, dist0 in zip(indices, distances):
                if dist0 < epsilon and idx < n_timesteps - 1:
                    # Distance at next time step
                    dist1 = norm(X[t + 1] - X[idx + 1])

                    if dist1 > 1e-10:
                        # Log divergence rate
                        rate = np.log(dist1 / dist0) / self.dt
                        divergence_rates.append(rate)

        if len(divergence_rates) == 0:
            logger.warning("No valid divergence pairs found")
            max_exponent = 0.0
        else:
            max_exponent = np.mean(divergence_rates)

        # For this method, we only get max exponent
        exponents = np.array([max_exponent])

        is_chaotic = max_exponent > 0
        is_stable = max_exponent < 0

        if max_exponent > 0:
            predictability_time = 1.0 / max_exponent
        else:
            predictability_time = np.inf

        return LyapunovResult(
            max_exponent=max_exponent,
            exponents=exponents,
            lyapunov_dimension=1.0 if max_exponent > 0 else 0.0,
            kolmogorov_entropy=max(0, max_exponent),
            is_chaotic=is_chaotic,
            is_stable=is_stable,
            divergence_rate=max_exponent,
            predictability_time=predictability_time,
            method="divergence",
            n_timesteps=n_timesteps
        )

    def _ftle_method(
        self,
        X: np.ndarray,
        integration_time: int = 10
    ) -> LyapunovResult:
        """
        Compute Finite-Time Lyapunov Exponents (FTLE).

        Args:
            X: Trajectory data (n_timesteps, n_features)
            integration_time: Time window for FTLE computation

        Returns:
            LyapunovResult
        """
        n_timesteps, n_features = X.shape

        ftle_values = []

        for t in range(n_timesteps - integration_time):
            # Flow map from t to t + T
            x0 = X[t]
            xT = X[t + integration_time]

            # Estimate flow map Jacobian
            J = self._estimate_flow_jacobian(
                X[t:t+integration_time+1]
            )

            # Cauchy-Green deformation tensor
            C = J.T @ J

            # Eigenvalues
            eigvals = np.linalg.eigvals(C)

            # FTLE (largest)
            ftle = np.log(np.sqrt(np.max(eigvals))) / (integration_time * self.dt)
            ftle_values.append(ftle)

        max_exponent = np.mean(ftle_values)
        exponents = np.array([max_exponent])

        is_chaotic = max_exponent > 0
        is_stable = max_exponent < 0

        if max_exponent > 0:
            predictability_time = 1.0 / max_exponent
        else:
            predictability_time = np.inf

        return LyapunovResult(
            max_exponent=max_exponent,
            exponents=exponents,
            lyapunov_dimension=1.0 if max_exponent > 0 else 0.0,
            kolmogorov_entropy=max(0, max_exponent),
            is_chaotic=is_chaotic,
            is_stable=is_stable,
            predictability_time=predictability_time,
            method="ftle",
            n_timesteps=n_timesteps
        )

    def estimate_lyapunov_function(
        self,
        trajectories: np.ndarray,
        equilibrium: Optional[np.ndarray] = None,
        method: str = "neural"
    ) -> LyapunovFunctionResult:
        """
        Estimate a Lyapunov function for the system.

        Args:
            trajectories: Trajectory data
            equilibrium: Equilibrium point (if None, estimated as mean)
            method: Estimation method ("neural", "quadratic", "sos")

        Returns:
            LyapunovFunctionResult
        """
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        if equilibrium is None:
            equilibrium = np.mean(trajectories, axis=0)

        if method == "neural":
            return self._neural_lyapunov_function(trajectories, equilibrium)
        elif method == "quadratic":
            return self._quadratic_lyapunov_function(trajectories, equilibrium)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _neural_lyapunov_function(
        self,
        X: np.ndarray,
        equilibrium: np.ndarray,
        hidden_dim: int = 64,
        n_epochs: int = 100
    ) -> LyapunovFunctionResult:
        """
        Learn Lyapunov function using neural network.

        Args:
            X: Trajectory data
            equilibrium: Equilibrium point
            hidden_dim: Hidden layer dimension
            n_epochs: Training epochs

        Returns:
            LyapunovFunctionResult
        """
        n_features = X.shape[1]

        # Shift to equilibrium frame
        X_shifted = X - equilibrium

        # Create neural network for V(x)
        class LyapunovNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_features, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                    nn.Softplus()  # Ensure positive
                )

            def forward(self, x):
                return self.net(x)

        model = LyapunovNet().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        X_torch = torch.tensor(X_shifted, dtype=torch.float32, device=self.device)

        # Training loop
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # V(x) should be positive definite
            V = model(X_torch)

            # Compute time derivative
            V_dot = torch.zeros_like(V)
            for i in range(len(X_torch) - 1):
                V_dot[i] = (V[i+1] - V[i]) / self.dt

            # Loss: V > 0, V(0) = 0, V̇ < 0
            loss = (
                -torch.mean(V) +  # Encourage V > 0
                torch.mean(torch.relu(V_dot)) +  # Penalize V̇ > 0
                model(torch.zeros(1, n_features, device=self.device))**2  # V(0) = 0
            )

            loss.backward()
            optimizer.step()

        # Create callable functions
        def lyapunov_function(x):
            x_shifted = x - equilibrium
            x_torch = torch.tensor(x_shifted, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                return model(x_torch.unsqueeze(0)).item()

        def lyapunov_derivative(x):
            # Approximate using finite differences
            dx = 1e-6
            V_x = lyapunov_function(x)
            derivatives = []
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += dx
                V_plus = lyapunov_function(x_plus)
                derivatives.append((V_plus - V_x) / dx)
            return np.array(derivatives)

        return LyapunovFunctionResult(
            lyapunov_function=lyapunov_function,
            lyapunov_derivative=lyapunov_derivative
        )

    def _quadratic_lyapunov_function(
        self,
        X: np.ndarray,
        equilibrium: np.ndarray
    ) -> LyapunovFunctionResult:
        """
        Estimate quadratic Lyapunov function V(x) = (x-x*)ᵀ P (x-x*).

        Args:
            X: Trajectory data
            equilibrium: Equilibrium point

        Returns:
            LyapunovFunctionResult
        """
        # Shift to equilibrium frame
        X_shifted = X - equilibrium

        # Estimate Jacobian at equilibrium
        J = self._estimate_jacobian(X, n_neighbors=10)

        # Solve Lyapunov equation: AᵀP + PA = -Q
        Q = np.eye(J.shape[0])
        try:
            P = solve_continuous_lyapunov(J.T, -Q)
        except:
            logger.warning("Lyapunov equation failed, using identity")
            P = np.eye(J.shape[0])

        # Ensure P is positive definite
        eigvals = np.linalg.eigvals(P)
        if np.any(eigvals <= 0):
            P = P + (np.abs(np.min(eigvals)) + 0.1) * np.eye(P.shape[0])

        def lyapunov_function(x):
            x_shifted = x - equilibrium
            return x_shifted @ P @ x_shifted

        def lyapunov_derivative(x):
            x_shifted = x - equilibrium
            return 2 * P @ x_shifted

        return LyapunovFunctionResult(
            lyapunov_function=lyapunov_function,
            lyapunov_derivative=lyapunov_derivative
        )

    def _estimate_jacobian(
        self,
        X: np.ndarray,
        n_neighbors: int = 10
    ) -> np.ndarray:
        """
        Estimate Jacobian matrix using local linear regression.

        Args:
            X: Trajectory segment (n_timesteps, n_features)
            n_neighbors: Number of neighbors for regression

        Returns:
            Jacobian matrix (n_features, n_features)
        """
        if len(X) < 2:
            return np.eye(X.shape[1])

        # Compute velocities
        velocities = np.diff(X, axis=0) / self.dt

        # Use middle points
        X_mid = X[:-1]
        n_samples = min(len(X_mid), n_neighbors)

        if n_samples < 2:
            return np.eye(X.shape[1])

        # Ridge regression for each output dimension
        J = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            reg = Ridge(alpha=1e-6)
            try:
                reg.fit(X_mid[-n_samples:], velocities[-n_samples:, i])
                J[i] = reg.coef_
            except:
                J[i, i] = 1.0

        return J

    def _estimate_flow_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Estimate Jacobian of flow map."""
        # Simplified: use finite differences
        dx = 1e-6
        n_features = X.shape[1]
        J = np.zeros((n_features, n_features))

        x0 = X[0]
        xT = X[-1]

        for i in range(n_features):
            x_plus = x0.copy()
            x_plus[i] += dx

            # Approximate flow
            xT_plus = xT + dx * (X[-1] - X[0])[i] / np.linalg.norm(X[-1] - X[0])

            J[:, i] = (xT_plus - xT) / dx

        return J

    def _kaplan_yorke_dimension(self, exponents: np.ndarray) -> float:
        """
        Compute Kaplan-Yorke (Lyapunov) dimension.

        D_KY = j + (λ_1 + ... + λ_j) / |λ_{j+1}|

        where j is the largest index such that λ_1 + ... + λ_j >= 0.
        """
        sorted_exps = np.sort(exponents)[::-1]
        cumsum = np.cumsum(sorted_exps)

        j = np.where(cumsum >= 0)[0]
        if len(j) == 0:
            return 0.0

        j = j[-1]

        if j == len(exponents) - 1:
            return float(j + 1)

        return j + 1 + cumsum[j] / np.abs(sorted_exps[j + 1])
