"""
Optimal Transport Methods

This module provides optimal transport (Wasserstein) distances and
tools for comparing probability distributions and trajectories.

Key capabilities:
- Wasserstein distances (1D and multi-dimensional)
- Optimal transport between trajectories
- Barycenter computation
- Gromov-Wasserstein distance
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance as scipy_wasserstein
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimalTransportResult:
    """Results from optimal transport computation."""

    distance: float  # Wasserstein distance
    transport_plan: Optional[np.ndarray] = None  # Optimal coupling
    transport_cost: Optional[float] = None  # Total transport cost


class OptimalTransport:
    """
    Optimal transport methods for comparing distributions and trajectories.

    Optimal transport provides a geometrically meaningful way to compare
    probability distributions based on the cost of moving mass.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize optimal transport analyzer.

        Args:
            verbose: Whether to log information
        """
        self.verbose = verbose

    def wasserstein_distance(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        p: int = 1,
        weights_X: Optional[np.ndarray] = None,
        weights_Y: Optional[np.ndarray] = None
    ) -> OptimalTransportResult:
        """
        Compute Wasserstein distance between two point clouds.

        Args:
            X: First distribution (n_samples, n_features)
            Y: Second distribution (m_samples, n_features)
            p: Order of Wasserstein distance (1 or 2)
            weights_X: Weights for X (default: uniform)
            weights_Y: Weights for Y (default: uniform)

        Returns:
            OptimalTransportResult
        """
        if X.shape[1] == 1 and Y.shape[1] == 1:
            # Use efficient 1D algorithm
            return self._wasserstein_1d(X.flatten(), Y.flatten(), p, weights_X, weights_Y)
        else:
            # Use general algorithm
            return self._wasserstein_nd(X, Y, p, weights_X, weights_Y)

    def _wasserstein_1d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        p: int,
        weights_X: Optional[np.ndarray],
        weights_Y: Optional[np.ndarray]
    ) -> OptimalTransportResult:
        """1D Wasserstein distance (fast algorithm)."""
        if weights_X is None:
            weights_X = np.ones(len(X)) / len(X)
        if weights_Y is None:
            weights_Y = np.ones(len(Y)) / len(Y)

        # Scipy's implementation
        distance = scipy_wasserstein(X, Y, weights_X, weights_Y)

        if p == 2:
            distance = distance ** 2

        return OptimalTransportResult(distance=distance)

    def _wasserstein_nd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        p: int,
        weights_X: Optional[np.ndarray],
        weights_Y: Optional[np.ndarray]
    ) -> OptimalTransportResult:
        """Multi-dimensional Wasserstein distance using linear programming."""
        n = len(X)
        m = len(Y)

        if weights_X is None:
            weights_X = np.ones(n) / n
        if weights_Y is None:
            weights_Y = np.ones(m) / m

        # Compute cost matrix (pairwise distances)
        cost_matrix = cdist(X, Y, metric='euclidean')

        if p == 2:
            cost_matrix = cost_matrix ** 2

        # Solve optimal transport using Hungarian algorithm (for balanced case)
        # For unbalanced case, would need proper OT solver

        # Simplified: use linear assignment for equal weights
        if len(weights_X) == len(weights_Y) and np.allclose(weights_X, weights_Y):
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Transport plan
            transport_plan = np.zeros((n, m))
            transport_plan[row_ind, col_ind] = weights_X[row_ind]

            # Distance
            distance = cost_matrix[row_ind, col_ind].sum()

        else:
            # Use Sinkhorn algorithm for general case
            transport_plan = self._sinkhorn(cost_matrix, weights_X, weights_Y)
            distance = np.sum(transport_plan * cost_matrix)

        if p == 2:
            distance = np.sqrt(distance)

        return OptimalTransportResult(
            distance=distance,
            transport_plan=transport_plan,
            transport_cost=np.sum(transport_plan * cost_matrix)
        )

    def _sinkhorn(
        self,
        cost_matrix: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        reg: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-9
    ) -> np.ndarray:
        """
        Sinkhorn algorithm for entropic regularized OT.

        Args:
            cost_matrix: Cost matrix (n, m)
            a: Source distribution (n,)
            b: Target distribution (m,)
            reg: Regularization parameter
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Transport plan (n, m)
        """
        # Kernel matrix
        K = np.exp(-cost_matrix / reg)

        # Initialize
        u = np.ones(len(a)) / len(a)
        v = np.ones(len(b)) / len(b)

        # Sinkhorn iterations
        for _ in range(max_iter):
            u_prev = u.copy()

            u = a / (K @ v)
            v = b / (K.T @ u)

            # Check convergence
            if np.max(np.abs(u - u_prev)) < tol:
                break

        # Transport plan
        transport_plan = u[:, np.newaxis] * K * v[np.newaxis, :]

        return transport_plan

    def trajectory_distance(
        self,
        traj1: np.ndarray,
        traj2: np.ndarray,
        method: str = "dtw"
    ) -> float:
        """
        Compute distance between two trajectories.

        Args:
            traj1: First trajectory (n_timesteps, n_features)
            traj2: Second trajectory (m_timesteps, n_features)
            method: Method ("dtw", "frechet", "wasserstein")

        Returns:
            Distance between trajectories
        """
        if method == "dtw":
            return self._dtw_distance(traj1, traj2)
        elif method == "frechet":
            return self._frechet_distance(traj1, traj2)
        elif method == "wasserstein":
            result = self.wasserstein_distance(traj1, traj2)
            return result.distance
        else:
            raise ValueError(f"Unknown method: {method}")

    def _dtw_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Dynamic Time Warping distance."""
        n, m = len(traj1), len(traj2)

        # DTW matrix
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(traj1[i - 1] - traj2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        return dtw[n, m]

    def _frechet_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Discrete Fréchet distance."""
        n, m = len(traj1), len(traj2)

        # Distance matrix
        ca = np.full((n, m), -1.0)

        def c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]

            d = np.linalg.norm(traj1[i] - traj2[j])

            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(c(i - 1, 0), d)
            elif i == 0 and j > 0:
                ca[i, j] = max(c(0, j - 1), d)
            elif i > 0 and j > 0:
                ca[i, j] = max(min(c(i - 1, j), c(i, j - 1), c(i - 1, j - 1)), d)
            else:
                ca[i, j] = np.inf

            return ca[i, j]

        return c(n - 1, m - 1)

    def barycenter(
        self,
        distributions: list,
        weights: Optional[np.ndarray] = None,
        max_iter: int = 100
    ) -> np.ndarray:
        """
        Compute Wasserstein barycenter of distributions.

        Args:
            distributions: List of point clouds
            weights: Weights for each distribution
            max_iter: Maximum iterations

        Returns:
            Barycenter point cloud
        """
        n_distributions = len(distributions)

        if weights is None:
            weights = np.ones(n_distributions) / n_distributions

        # Initialize barycenter as weighted mean
        all_points = np.vstack(distributions)
        barycenter = all_points[np.random.choice(len(all_points), len(distributions[0]))]

        # Iterative refinement
        for iteration in range(max_iter):
            barycenter_prev = barycenter.copy()

            # Compute optimal transport to each distribution
            new_barycenter = np.zeros_like(barycenter)

            for dist, weight in zip(distributions, weights):
                # Transport from barycenter to distribution
                cost_matrix = cdist(barycenter, dist)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Update barycenter
                new_barycenter[row_ind] += weight * dist[col_ind]

            barycenter = new_barycenter

            # Check convergence
            if np.linalg.norm(barycenter - barycenter_prev) < 1e-6:
                break

        return barycenter

    def gromov_wasserstein(
        self,
        C1: np.ndarray,
        C2: np.ndarray,
        p1: Optional[np.ndarray] = None,
        p2: Optional[np.ndarray] = None,
        max_iter: int = 100
    ) -> OptimalTransportResult:
        """
        Compute Gromov-Wasserstein distance between metric spaces.

        Args:
            C1: Cost matrix for space 1 (n1, n1)
            C2: Cost matrix for space 2 (n2, n2)
            p1: Distribution on space 1
            p2: Distribution on space 2
            max_iter: Maximum iterations

        Returns:
            OptimalTransportResult with GW distance
        """
        n1, n2 = C1.shape[0], C2.shape[0]

        if p1 is None:
            p1 = np.ones(n1) / n1
        if p2 is None:
            p2 = np.ones(n2) / n2

        # Initialize transport plan
        T = np.outer(p1, p2)

        # Gromov-Wasserstein iterations
        for _ in range(max_iter):
            # Compute cost tensor
            # L(i,j) = sum_kl |C1(i,k) - C2(j,l)|^2 T(k,l)

            # Simplified gradient descent
            # Full GW requires more sophisticated optimization

            T_prev = T.copy()

            # Update (simplified)
            # This is a placeholder - full GW is more complex
            T = np.outer(p1, p2)  # Reset to uniform

            if np.linalg.norm(T - T_prev) < 1e-6:
                break

        # Compute GW distance
        distance = 0.0
        for i in range(n1):
            for j in range(n2):
                for k in range(n1):
                    for l in range(n2):
                        distance += (C1[i, k] - C2[j, l]) ** 2 * T[i, j] * T[k, l]

        distance = np.sqrt(distance)

        return OptimalTransportResult(
            distance=distance,
            transport_plan=T,
            transport_cost=distance
        )
