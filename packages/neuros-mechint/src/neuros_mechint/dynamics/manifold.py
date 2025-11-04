"""
Manifold Analysis for Dynamical Systems

This module provides tools for analyzing the geometric structure of
state space manifolds in dynamical systems.

Key capabilities:
- Intrinsic dimensionality estimation
- Tangent space estimation
- Curvature computation
- Geodesic distance estimation
- Slow manifold detection
- Inertial manifold identification
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import distance_matrix
from scipy.linalg import svd, eigh
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)


@dataclass
class ManifoldResult:
    """Results from manifold analysis."""

    # Dimensionality
    intrinsic_dimension: int  # Estimated intrinsic dimension
    embedding_dimension: int  # Original embedding dimension
    dimension_estimates: dict  # Multiple estimation methods

    # PCA results
    principal_components: np.ndarray  # Principal components
    explained_variance: np.ndarray  # Explained variance
    explained_variance_ratio: np.ndarray  # Explained variance ratio
    participation_ratio: float  # Effective dimensionality

    # Geometry
    mean_curvature: Optional[float] = None  # Mean curvature
    gaussian_curvature: Optional[float] = None  # Gaussian curvature
    curvature_field: Optional[np.ndarray] = None  # Curvature at each point

    # Tangent spaces
    tangent_spaces: Optional[np.ndarray] = None  # Local tangent spaces

    # Geodesic distances
    geodesic_distances: Optional[np.ndarray] = None  # Geodesic distance matrix


class ManifoldAnalyzer:
    """
    Analyze the manifold structure of dynamical systems.

    Dynamical systems often evolve on low-dimensional manifolds
    embedded in high-dimensional state spaces.
    """

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize manifold analyzer.

        Args:
            dt: Time step
            device: Device for computations
            verbose: Whether to log information
        """
        self.dt = dt
        self.device = device
        self.verbose = verbose

    def analyze(
        self,
        trajectories: np.ndarray,
        compute_curvature: bool = True,
        compute_tangent_spaces: bool = True,
        compute_geodesics: bool = False,
        **kwargs
    ) -> ManifoldResult:
        """
        Comprehensive manifold analysis.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features) or
                         (n_trajectories, n_timesteps, n_features)
            compute_curvature: Whether to compute curvature
            compute_tangent_spaces: Whether to compute tangent spaces
            compute_geodesics: Whether to compute geodesic distances
            **kwargs: Additional parameters

        Returns:
            ManifoldResult with manifold properties
        """
        # Reshape if needed
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        # Estimate intrinsic dimension
        intrinsic_dim, dim_estimates = self._estimate_intrinsic_dimension(
            trajectories, **kwargs
        )

        # PCA analysis
        pca_result = self._pca_analysis(trajectories)

        # Curvature
        if compute_curvature:
            curvature_field = self._estimate_curvature_field(trajectories)
            mean_curvature = np.mean(curvature_field)
            gaussian_curvature = None  # Computed separately if needed
        else:
            curvature_field = None
            mean_curvature = None
            gaussian_curvature = None

        # Tangent spaces
        if compute_tangent_spaces:
            tangent_spaces = self._estimate_tangent_spaces(trajectories, intrinsic_dim)
        else:
            tangent_spaces = None

        # Geodesic distances
        if compute_geodesics:
            geodesic_distances = self._compute_geodesic_distances(trajectories)
        else:
            geodesic_distances = None

        return ManifoldResult(
            intrinsic_dimension=intrinsic_dim,
            embedding_dimension=trajectories.shape[1],
            dimension_estimates=dim_estimates,
            principal_components=pca_result['components'],
            explained_variance=pca_result['explained_variance'],
            explained_variance_ratio=pca_result['explained_variance_ratio'],
            participation_ratio=pca_result['participation_ratio'],
            mean_curvature=mean_curvature,
            gaussian_curvature=gaussian_curvature,
            curvature_field=curvature_field,
            tangent_spaces=tangent_spaces,
            geodesic_distances=geodesic_distances
        )

    def _estimate_intrinsic_dimension(
        self,
        X: np.ndarray,
        methods: list = ['pca', 'mle', 'correlation']
    ) -> Tuple[int, dict]:
        """
        Estimate intrinsic dimension using multiple methods.

        Args:
            X: Data points (n_samples, n_features)
            methods: List of methods to use

        Returns:
            Tuple of (best_estimate, all_estimates)
        """
        estimates = {}

        # PCA-based estimate
        if 'pca' in methods:
            estimates['pca'] = self._intrinsic_dim_pca(X)

        # MLE-based estimate
        if 'mle' in methods:
            try:
                estimates['mle'] = self._intrinsic_dim_mle(X)
            except Exception as e:
                logger.debug(f"MLE dimension estimation failed: {e}")
                estimates['mle'] = estimates.get('pca', X.shape[1])

        # Correlation dimension
        if 'correlation' in methods:
            try:
                estimates['correlation'] = self._intrinsic_dim_correlation(X)
            except Exception as e:
                logger.debug(f"Correlation dimension estimation failed: {e}")
                estimates['correlation'] = estimates.get('pca', X.shape[1])

        # Use median as best estimate
        best_estimate = int(np.median(list(estimates.values())))

        return best_estimate, estimates

    def _intrinsic_dim_pca(
        self,
        X: np.ndarray,
        variance_threshold: float = 0.95
    ) -> int:
        """
        Estimate intrinsic dimension using PCA.

        Counts components needed to explain variance_threshold of variance.

        Args:
            X: Data points (n_samples, n_features)
            variance_threshold: Cumulative variance threshold

        Returns:
            Estimated intrinsic dimension
        """
        pca = PCA()
        pca.fit(X)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.searchsorted(cumulative_variance, variance_threshold) + 1

        return min(intrinsic_dim, X.shape[1])

    def _intrinsic_dim_mle(
        self,
        X: np.ndarray,
        k: int = 20
    ) -> int:
        """
        Estimate intrinsic dimension using Maximum Likelihood Estimation.

        Based on Levina & Bickel (2004) method.

        Args:
            X: Data points (n_samples, n_features)
            k: Number of neighbors

        Returns:
            Estimated intrinsic dimension
        """
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Remove self (first neighbor)
        distances = distances[:, 1:]

        # MLE estimate
        d_estimates = []
        for i in range(len(X)):
            r = distances[i]
            # Estimate: m = (sum log(r_k/r_j))^-1
            if r[-1] > 0:
                ratios = r[-1] / r[:-1]
                log_ratios = np.log(ratios[ratios > 0])
                if len(log_ratios) > 0:
                    m = len(log_ratios) / np.sum(log_ratios)
                    d_estimates.append(m)

        if len(d_estimates) == 0:
            return X.shape[1]

        intrinsic_dim = int(np.median(d_estimates))
        return max(1, min(intrinsic_dim, X.shape[1]))

    def _intrinsic_dim_correlation(
        self,
        X: np.ndarray,
        n_samples: int = 1000
    ) -> int:
        """
        Estimate intrinsic dimension using correlation dimension.

        Args:
            X: Data points (n_samples, n_features)
            n_samples: Number of samples for estimation

        Returns:
            Estimated intrinsic dimension
        """
        # Subsample for efficiency
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Compute pairwise distances
        dists = distance_matrix(X_sample, X_sample)
        dists = dists[np.triu_indices_from(dists, k=1)]

        # Range of radius values
        r_values = np.percentile(dists, np.linspace(10, 50, 10))

        # Count pairs within each radius
        counts = [np.sum(dists < r) for r in r_values]

        # Fit log-log relationship: log(C(r)) ~ d * log(r)
        log_r = np.log(r_values[counts > 0])
        log_C = np.log(np.array(counts)[counts > 0])

        if len(log_r) < 2:
            return X.shape[1]

        # Linear regression
        slope = np.polyfit(log_r, log_C, 1)[0]

        intrinsic_dim = int(np.round(slope))
        return max(1, min(intrinsic_dim, X.shape[1]))

    def _pca_analysis(self, X: np.ndarray) -> dict:
        """
        Perform PCA analysis.

        Args:
            X: Data points (n_samples, n_features)

        Returns:
            Dictionary with PCA results
        """
        pca = PCA()
        pca.fit(X)

        # Participation ratio: (sum λ_i)^2 / sum λ_i^2
        eigenvalues = pca.explained_variance_
        participation_ratio = (
            np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)
        )

        return {
            'components': pca.components_,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'participation_ratio': participation_ratio
        }

    def _estimate_tangent_spaces(
        self,
        X: np.ndarray,
        intrinsic_dim: int,
        k: int = 20
    ) -> np.ndarray:
        """
        Estimate local tangent spaces at each point.

        Args:
            X: Data points (n_samples, n_features)
            intrinsic_dim: Intrinsic dimension
            k: Number of neighbors for local PCA

        Returns:
            Tangent spaces (n_samples, n_features, intrinsic_dim)
        """
        n_samples, n_features = X.shape
        tangent_spaces = np.zeros((n_samples, n_features, intrinsic_dim))

        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Local PCA at each point
        for i in range(n_samples):
            # Get local neighborhood
            neighbors = X[indices[i]]

            # Center
            neighbors_centered = neighbors - np.mean(neighbors, axis=0)

            # Local PCA
            try:
                U, S, Vh = svd(neighbors_centered.T, full_matrices=False)
                tangent_space = U[:, :intrinsic_dim]
                tangent_spaces[i] = tangent_space
            except:
                # Fallback: use global PCA directions
                tangent_spaces[i] = np.eye(n_features)[:, :intrinsic_dim]

        return tangent_spaces

    def _estimate_curvature_field(
        self,
        X: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """
        Estimate curvature at each point using local geometry.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            k: Number of neighbors

        Returns:
            Curvature values (n_timesteps,)
        """
        n_timesteps = len(X)
        curvatures = np.zeros(n_timesteps)

        # Use velocity-based curvature for trajectories
        if n_timesteps >= 3:
            # Compute velocity and acceleration
            velocities = np.gradient(X, axis=0)
            accelerations = np.gradient(velocities, axis=0)

            # Curvature: κ = ||v × a|| / ||v||^3
            for i in range(n_timesteps):
                v = velocities[i]
                a = accelerations[i]

                v_norm = np.linalg.norm(v)
                if v_norm > 1e-10:
                    # For high dimensions, use ||a - (a·v̂)v̂|| / ||v||^2
                    v_hat = v / v_norm
                    a_perp = a - np.dot(a, v_hat) * v_hat
                    curvature = np.linalg.norm(a_perp) / (v_norm ** 2)
                    curvatures[i] = curvature

        return curvatures

    def _compute_geodesic_distances(
        self,
        X: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """
        Compute approximate geodesic distances using graph methods.

        Args:
            X: Data points (n_samples, n_features)
            k: Number of neighbors for graph construction

        Returns:
            Geodesic distance matrix (n_samples, n_samples)
        """
        from sklearn.neighbors import kneighbors_graph
        from scipy.sparse.csgraph import floyd_warshall

        # Build k-NN graph with Euclidean distances as weights
        knn_graph = kneighbors_graph(
            X,
            k,
            mode='distance',
            include_self=False
        )

        # Compute shortest paths (geodesics)
        geodesic_distances = floyd_warshall(
            knn_graph,
            directed=False,
            unweighted=False
        )

        return geodesic_distances

    def identify_slow_manifold(
        self,
        trajectories: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify slow manifold using timescale separation.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            threshold: Threshold for slow/fast separation

        Returns:
            Tuple of (slow_directions, fast_directions)
        """
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        # Compute velocities
        velocities = np.diff(trajectories, axis=0) / self.dt

        # Covariance of velocities
        cov_vel = np.cov(velocities.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(cov_vel)

        # Sort by eigenvalue (ascending = slowest first)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Separate slow and fast
        # Slow modes have small velocity variance
        normalized_eigs = eigenvalues / np.max(eigenvalues)
        slow_idx = normalized_eigs < threshold

        slow_directions = eigenvectors[:, slow_idx]
        fast_directions = eigenvectors[:, ~slow_idx]

        return slow_directions, fast_directions

    def compute_sectional_curvature(
        self,
        tangent_space1: np.ndarray,
        tangent_space2: np.ndarray
    ) -> float:
        """
        Compute sectional curvature between two tangent planes.

        Args:
            tangent_space1: First tangent space basis
            tangent_space2: Second tangent space basis

        Returns:
            Sectional curvature (approximate)
        """
        # Simplified: use angle between subspaces
        # True sectional curvature requires the metric

        # Compute principal angles
        U1, _, _ = svd(tangent_space1, full_matrices=False)
        U2, _, _ = svd(tangent_space2, full_matrices=False)

        # Canonical correlation
        M = U1.T @ U2
        singular_values = svd(M, compute_uv=False)

        # Principal angle
        principal_angle = np.arccos(np.clip(singular_values[0], -1, 1))

        # Curvature ~ change in angle
        # This is a rough approximation
        curvature = principal_angle

        return curvature
