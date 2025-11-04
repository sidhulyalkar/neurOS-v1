"""
Fixed Point Detection and Analysis

This module provides tools for finding and analyzing fixed points, equilibria,
and periodic orbits in dynamical systems.

Key capabilities:
- Fixed point detection (multiple methods)
- Stability classification (stable, unstable, saddle)
- Periodic orbit detection
- Basin of attraction estimation
- Bifurcation tracking
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import numpy as np
from scipy.optimize import minimize, fsolve
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


@dataclass
class FixedPoint:
    """A single fixed point with properties."""

    location: np.ndarray  # Position in state space
    stability: str  # "stable", "unstable", "saddle", "unknown"
    eigenvalues: Optional[np.ndarray] = None  # Jacobian eigenvalues
    basin_size: Optional[float] = None  # Estimated basin of attraction size
    index: Optional[int] = None  # Index for identification


@dataclass
class PeriodicOrbit:
    """A periodic orbit."""

    points: np.ndarray  # Points on the orbit (n_points, n_features)
    period: float  # Period in time units
    stability: str  # "stable" or "unstable"
    floquet_multipliers: Optional[np.ndarray] = None  # Floquet multipliers


@dataclass
class FixedPointResult:
    """Results from fixed point analysis."""

    fixed_points: List[FixedPoint]  # List of detected fixed points
    periodic_orbits: List[PeriodicOrbit]  # List of detected periodic orbits
    n_stable: int  # Number of stable fixed points
    n_unstable: int  # Number of unstable fixed points
    n_saddles: int  # Number of saddle points


class FixedPointFinder:
    """
    Find and analyze fixed points in dynamical systems.

    Fixed points are states where dx/dt = 0, and are fundamental
    to understanding the long-term behavior of dynamical systems.
    """

    def __init__(
        self,
        dt: float = 0.01,
        velocity_threshold: float = 1e-3,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize fixed point finder.

        Args:
            dt: Time step
            velocity_threshold: Threshold for considering a point fixed
            device: Device for computations
            verbose: Whether to log information
        """
        self.dt = dt
        self.velocity_threshold = velocity_threshold
        self.device = device
        self.verbose = verbose

    def find_fixed_points(
        self,
        trajectories: np.ndarray,
        method: str = "velocity",
        **kwargs
    ) -> FixedPointResult:
        """
        Find fixed points in the system.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features) or
                         (n_trajectories, n_timesteps, n_features)
            method: Detection method ("velocity", "optimization", "recurrence")
            **kwargs: Method-specific parameters

        Returns:
            FixedPointResult with detected fixed points
        """
        # Reshape if needed
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        if method == "velocity":
            fixed_points = self._velocity_method(trajectories, **kwargs)
        elif method == "optimization":
            fixed_points = self._optimization_method(trajectories, **kwargs)
        elif method == "recurrence":
            fixed_points = self._recurrence_method(trajectories, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Classify stability
        for fp in fixed_points:
            self._classify_stability(fp, trajectories)

        # Detect periodic orbits
        periodic_orbits = self._detect_periodic_orbits(trajectories)

        # Count by stability
        n_stable = sum(1 for fp in fixed_points if fp.stability == "stable")
        n_unstable = sum(1 for fp in fixed_points if fp.stability == "unstable")
        n_saddles = sum(1 for fp in fixed_points if fp.stability == "saddle")

        return FixedPointResult(
            fixed_points=fixed_points,
            periodic_orbits=periodic_orbits,
            n_stable=n_stable,
            n_unstable=n_unstable,
            n_saddles=n_saddles
        )

    def _velocity_method(
        self,
        X: np.ndarray,
        min_samples: int = 5,
        eps: float = 0.1
    ) -> List[FixedPoint]:
        """
        Find fixed points by detecting low-velocity regions.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            min_samples: Minimum samples for clustering
            eps: DBSCAN epsilon parameter

        Returns:
            List of FixedPoint objects
        """
        # Compute velocities
        velocities = np.diff(X, axis=0) / self.dt
        speeds = np.linalg.norm(velocities, axis=1)

        # Find points with low velocity
        slow_indices = np.where(speeds < self.velocity_threshold)[0]

        if len(slow_indices) == 0:
            logger.warning("No fixed points found")
            return []

        slow_points = X[slow_indices]

        # Cluster slow points
        if len(slow_points) < min_samples:
            # Not enough points for clustering
            fixed_points = [
                FixedPoint(location=np.mean(slow_points, axis=0), stability="unknown", index=0)
            ]
        else:
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clustering.fit_predict(slow_points)

            # Extract cluster centers as fixed points
            fixed_points = []
            for label in set(labels):
                if label == -1:  # Noise
                    continue

                cluster_points = slow_points[labels == label]
                center = np.mean(cluster_points, axis=0)

                fp = FixedPoint(
                    location=center,
                    stability="unknown",
                    index=len(fixed_points)
                )
                fixed_points.append(fp)

        return fixed_points

    def _optimization_method(
        self,
        X: np.ndarray,
        n_initializations: int = 10,
        velocity_model: Optional[Callable] = None
    ) -> List[FixedPoint]:
        """
        Find fixed points by optimization.

        Minimize ||f(x)|| where f(x) = dx/dt.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            n_initializations: Number of random initializations
            velocity_model: Function that computes velocity given state

        Returns:
            List of FixedPoint objects
        """
        n_features = X.shape[1]

        # If no velocity model provided, estimate from data
        if velocity_model is None:
            from sklearn.linear_model import Ridge

            velocities = np.diff(X, axis=0) / self.dt
            X_train = X[:-1]

            # Train simple model
            model = Ridge(alpha=1.0)
            model.fit(X_train, velocities)

            velocity_model = lambda x: model.predict(x.reshape(1, -1))[0]

        # Objective: minimize velocity magnitude
        def objective(x):
            v = velocity_model(x)
            return np.sum(v ** 2)

        # Multiple random initializations
        fixed_points = []
        seen_points = []

        # Add some initializations from data
        indices = np.random.choice(len(X), min(n_initializations, len(X)), replace=False)

        for idx in indices:
            x0 = X[idx]

            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100}
                )

                if result.success and result.fun < self.velocity_threshold ** 2:
                    x_fp = result.x

                    # Check if we've already found this point
                    is_new = True
                    for seen_point in seen_points:
                        if np.linalg.norm(x_fp - seen_point) < 0.1:
                            is_new = False
                            break

                    if is_new:
                        fp = FixedPoint(
                            location=x_fp,
                            stability="unknown",
                            index=len(fixed_points)
                        )
                        fixed_points.append(fp)
                        seen_points.append(x_fp)

            except Exception as e:
                logger.debug(f"Optimization failed: {e}")
                continue

        return fixed_points

    def _recurrence_method(
        self,
        X: np.ndarray,
        recurrence_threshold: float = 0.1,
        min_recurrence_count: int = 5
    ) -> List[FixedPoint]:
        """
        Find fixed points by detecting recurrent regions.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            recurrence_threshold: Distance threshold for recurrence
            min_recurrence_count: Minimum number of recurrences

        Returns:
            List of FixedPoint objects
        """
        # Compute distance matrix
        dists = distance_matrix(X, X)

        # Count recurrences for each point
        recurrence_counts = np.sum(dists < recurrence_threshold, axis=1)

        # Find highly recurrent points
        recurrent_indices = np.where(recurrence_counts >= min_recurrence_count)[0]

        if len(recurrent_indices) == 0:
            return []

        # Cluster recurrent points
        recurrent_points = X[recurrent_indices]

        clustering = DBSCAN(eps=recurrence_threshold, min_samples=min_recurrence_count)
        labels = clustering.fit_predict(recurrent_points)

        fixed_points = []
        for label in set(labels):
            if label == -1:
                continue

            cluster_points = recurrent_points[labels == label]
            center = np.mean(cluster_points, axis=0)

            fp = FixedPoint(
                location=center,
                stability="unknown",
                index=len(fixed_points)
            )
            fixed_points.append(fp)

        return fixed_points

    def _classify_stability(
        self,
        fixed_point: FixedPoint,
        trajectories: np.ndarray
    ):
        """
        Classify the stability of a fixed point.

        Args:
            fixed_point: FixedPoint to classify
            trajectories: Trajectory data for Jacobian estimation
        """
        # Estimate Jacobian at fixed point
        J = self._estimate_jacobian_at_point(fixed_point.location, trajectories)

        # Eigenvalues determine stability
        eigenvalues = np.linalg.eigvals(J)
        fixed_point.eigenvalues = eigenvalues

        # Classification based on eigenvalues
        real_parts = np.real(eigenvalues)

        if np.all(real_parts < 0):
            fixed_point.stability = "stable"
        elif np.all(real_parts > 0):
            fixed_point.stability = "unstable"
        elif np.any(real_parts < 0) and np.any(real_parts > 0):
            fixed_point.stability = "saddle"
        else:
            fixed_point.stability = "unknown"

    def _estimate_jacobian_at_point(
        self,
        point: np.ndarray,
        trajectories: np.ndarray,
        n_neighbors: int = 20
    ) -> np.ndarray:
        """
        Estimate Jacobian matrix at a specific point.

        Args:
            point: Point at which to estimate Jacobian
            trajectories: Trajectory data
            n_neighbors: Number of neighbors to use

        Returns:
            Jacobian matrix (n_features, n_features)
        """
        # Find nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(trajectories[:-1])
        distances, indices = tree.query(point, k=n_neighbors)

        # Get local data
        X_local = trajectories[indices]
        velocities = np.diff(trajectories, axis=0)[indices] / self.dt

        # Ridge regression
        from sklearn.linear_model import Ridge

        n_features = trajectories.shape[1]
        J = np.zeros((n_features, n_features))

        for i in range(n_features):
            reg = Ridge(alpha=1e-6)
            try:
                reg.fit(X_local, velocities[:, i])
                J[i] = reg.coef_
            except:
                J[i, i] = -0.1  # Default stable

        return J

    def _detect_periodic_orbits(
        self,
        X: np.ndarray,
        recurrence_threshold: float = 0.1,
        min_period: int = 10,
        max_period: int = 1000
    ) -> List[PeriodicOrbit]:
        """
        Detect periodic orbits using recurrence analysis.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            recurrence_threshold: Distance threshold for recurrence
            min_period: Minimum period to consider
            max_period: Maximum period to consider

        Returns:
            List of PeriodicOrbit objects
        """
        periodic_orbits = []

        # For each point, check if it recurs
        for t in range(len(X) - max_period):
            # Compute distances to future points
            future_points = X[t+min_period:t+max_period]
            distances = np.linalg.norm(future_points - X[t], axis=1)

            # Find recurrences
            recurrence_indices = np.where(distances < recurrence_threshold)[0] + t + min_period

            if len(recurrence_indices) > 0:
                # First recurrence gives period
                period_steps = recurrence_indices[0] - t
                period_time = period_steps * self.dt

                # Extract orbit points
                orbit_points = X[t:t+period_steps]

                # Check if we already found this orbit
                is_new = True
                for existing_orbit in periodic_orbits:
                    if np.abs(existing_orbit.period - period_time) < self.dt * 5:
                        is_new = False
                        break

                if is_new and len(orbit_points) > min_period:
                    # Estimate stability (simplified)
                    stability = "unknown"

                    orbit = PeriodicOrbit(
                        points=orbit_points,
                        period=period_time,
                        stability=stability
                    )
                    periodic_orbits.append(orbit)

                    # Skip ahead to avoid duplicate detection
                    t += period_steps

        return periodic_orbits

    def estimate_basin_of_attraction(
        self,
        fixed_point: FixedPoint,
        trajectories: np.ndarray,
        convergence_threshold: float = 0.1
    ) -> float:
        """
        Estimate the size of the basin of attraction for a fixed point.

        Args:
            fixed_point: FixedPoint to analyze
            trajectories: Trajectory data
            convergence_threshold: Distance threshold for convergence

        Returns:
            Estimated basin size (volume estimate)
        """
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        # Find points that converge to this fixed point
        final_points = trajectories[-100:]  # Last 100 points
        distances = np.linalg.norm(final_points - fixed_point.location, axis=1)

        converging_points = trajectories[distances < convergence_threshold]

        if len(converging_points) == 0:
            return 0.0

        # Estimate volume using convex hull or bounding box
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(converging_points)
            basin_size = hull.volume
        except:
            # Fallback: use bounding box volume
            ranges = np.ptp(converging_points, axis=0)
            basin_size = np.prod(ranges)

        fixed_point.basin_size = basin_size
        return basin_size
