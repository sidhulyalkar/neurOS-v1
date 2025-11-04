"""
Phase Space Analysis

This module provides tools for analyzing phase space structure of dynamical systems.

Key capabilities:
- Phase portrait construction
- Poincaré sections and maps
- Attractor reconstruction
- Basin of attraction estimation
- State space partitioning
- Flow topology analysis
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import numpy as np
from scipy.spatial import distance_matrix, ConvexHull
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoincareSection:
    """Results from Poincaré section analysis."""

    section_points: np.ndarray  # Points on Poincaré section
    return_times: np.ndarray  # Time between successive intersections
    return_map: Optional[np.ndarray] = None  # Poincaré return map
    section_normal: Optional[np.ndarray] = None  # Normal vector to section
    section_point: Optional[np.ndarray] = None  # Point on section plane


@dataclass
class AttractorResult:
    """Results from attractor analysis."""

    attractor_type: str  # "fixed_point", "limit_cycle", "strange", "unknown"
    attractor_points: np.ndarray  # Points on attractor
    basin_volume: Optional[float] = None  # Volume of basin of attraction
    fractal_dimension: Optional[float] = None  # Fractal dimension (if strange)


@dataclass
class PhaseSpaceResult:
    """Results from phase space analysis."""

    # Attractors
    attractors: List[AttractorResult]  # Detected attractors
    n_attractors: int  # Number of attractors

    # Basins
    basin_labels: Optional[np.ndarray] = None  # Basin assignment for each point
    basin_boundaries: Optional[List[np.ndarray]] = None  # Basin boundary points

    # Poincaré analysis
    poincare_sections: Optional[List[PoincareSection]] = None  # Poincaré sections

    # Topology
    topology_type: Optional[str] = None  # Global topology type


class PhaseSpaceAnalyzer:
    """
    Analyze phase space structure of dynamical systems.

    Phase space analysis reveals the geometric and topological
    structure of the system's long-term behavior.
    """

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize phase space analyzer.

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
        detect_attractors: bool = True,
        compute_basins: bool = True,
        compute_poincare: bool = False,
        **kwargs
    ) -> PhaseSpaceResult:
        """
        Comprehensive phase space analysis.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features) or
                         (n_trajectories, n_timesteps, n_features)
            detect_attractors: Whether to detect attractors
            compute_basins: Whether to compute basins of attraction
            compute_poincare: Whether to compute Poincaré sections
            **kwargs: Additional parameters

        Returns:
            PhaseSpaceResult with phase space characterization
        """
        original_shape = trajectories.shape

        # Reshape if needed
        if trajectories.ndim == 3:
            n_traj = trajectories.shape[0]
            trajectories_flat = trajectories.reshape(-1, trajectories.shape[-1])
        else:
            n_traj = 1
            trajectories_flat = trajectories

        # Detect attractors
        if detect_attractors:
            attractors = self._detect_attractors(trajectories_flat, **kwargs)
        else:
            attractors = []

        # Compute basins
        if compute_basins and len(attractors) > 0:
            basin_labels, basin_boundaries = self._estimate_basins(
                trajectories_flat, attractors
            )
        else:
            basin_labels = None
            basin_boundaries = None

        # Poincaré sections
        if compute_poincare:
            poincare_sections = self._compute_poincare_sections(
                trajectories_flat, **kwargs
            )
        else:
            poincare_sections = None

        # Classify topology
        topology_type = self._classify_topology(attractors)

        return PhaseSpaceResult(
            attractors=attractors,
            n_attractors=len(attractors),
            basin_labels=basin_labels,
            basin_boundaries=basin_boundaries,
            poincare_sections=poincare_sections,
            topology_type=topology_type
        )

    def _detect_attractors(
        self,
        X: np.ndarray,
        method: str = "clustering",
        n_attractors: Optional[int] = None
    ) -> List[AttractorResult]:
        """
        Detect attractors in the phase space.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            method: Detection method ("clustering", "recurrence")
            n_attractors: Number of attractors (if known)

        Returns:
            List of AttractorResult objects
        """
        # Focus on late-time behavior (after transients)
        transient_time = len(X) // 4
        X_late = X[transient_time:]

        if method == "clustering":
            return self._detect_attractors_clustering(X_late, n_attractors)
        elif method == "recurrence":
            return self._detect_attractors_recurrence(X_late)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _detect_attractors_clustering(
        self,
        X: np.ndarray,
        n_attractors: Optional[int] = None
    ) -> List[AttractorResult]:
        """
        Detect attractors using clustering of late-time dynamics.

        Args:
            X: Late-time trajectory data
            n_attractors: Number of attractors to find

        Returns:
            List of AttractorResult objects
        """
        # Automatic cluster number determination
        if n_attractors is None:
            # Use elbow method
            inertias = []
            K_range = range(1, min(10, len(X) // 10))

            for k in K_range:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)

            # Find elbow
            if len(inertias) > 2:
                diffs = np.diff(inertias)
                n_attractors = np.argmax(diffs[:-1] - diffs[1:]) + 2
            else:
                n_attractors = 1

            n_attractors = max(1, min(n_attractors, 5))

        # Cluster
        kmeans = KMeans(n_clusters=n_attractors, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        attractors = []
        for i in range(n_attractors):
            cluster_points = X[labels == i]

            if len(cluster_points) == 0:
                continue

            # Classify attractor type
            attractor_type = self._classify_attractor(cluster_points)

            # Compute fractal dimension if strange attractor
            if attractor_type == "strange":
                fractal_dim = self._estimate_fractal_dimension(cluster_points)
            else:
                fractal_dim = None

            # Estimate basin volume
            try:
                hull = ConvexHull(cluster_points)
                basin_volume = hull.volume
            except:
                basin_volume = None

            attractor = AttractorResult(
                attractor_type=attractor_type,
                attractor_points=cluster_points,
                basin_volume=basin_volume,
                fractal_dimension=fractal_dim
            )
            attractors.append(attractor)

        return attractors

    def _detect_attractors_recurrence(
        self,
        X: np.ndarray,
        recurrence_threshold: float = 0.1
    ) -> List[AttractorResult]:
        """
        Detect attractors using recurrence analysis.

        Args:
            X: Late-time trajectory data
            recurrence_threshold: Distance threshold for recurrence

        Returns:
            List of AttractorResult objects
        """
        # Compute recurrence matrix
        dists = distance_matrix(X, X)
        recurrence_matrix = dists < recurrence_threshold

        # Find highly recurrent regions
        recurrence_counts = np.sum(recurrence_matrix, axis=1)

        # Threshold for attractor membership
        threshold = np.percentile(recurrence_counts, 75)
        attractor_indices = recurrence_counts > threshold

        if np.sum(attractor_indices) == 0:
            return []

        attractor_points = X[attractor_indices]

        # Classify attractor
        attractor_type = self._classify_attractor(attractor_points)

        if attractor_type == "strange":
            fractal_dim = self._estimate_fractal_dimension(attractor_points)
        else:
            fractal_dim = None

        try:
            hull = ConvexHull(attractor_points)
            basin_volume = hull.volume
        except:
            basin_volume = None

        attractor = AttractorResult(
            attractor_type=attractor_type,
            attractor_points=attractor_points,
            basin_volume=basin_volume,
            fractal_dimension=fractal_dim
        )

        return [attractor]

    def _classify_attractor(self, points: np.ndarray) -> str:
        """
        Classify attractor type based on geometry.

        Args:
            points: Points on the attractor

        Returns:
            Attractor type string
        """
        # Check if fixed point (very small spread)
        spread = np.std(points, axis=0)
        if np.mean(spread) < 0.01:
            return "fixed_point"

        # Check if limit cycle (1D structure)
        pca_variance = np.var(points, axis=0)
        sorted_var = np.sort(pca_variance)[::-1]

        if sorted_var[0] > 10 * sorted_var[1]:
            return "limit_cycle"

        # Check for strange attractor (fractal structure)
        fractal_dim = self._estimate_fractal_dimension(points)
        if fractal_dim is not None and fractal_dim % 1 > 0.1:
            return "strange"

        return "unknown"

    def _estimate_fractal_dimension(
        self,
        points: np.ndarray,
        method: str = "correlation"
    ) -> Optional[float]:
        """
        Estimate fractal (correlation) dimension.

        Args:
            points: Points on attractor
            method: Estimation method

        Returns:
            Estimated fractal dimension
        """
        if len(points) < 100:
            return None

        # Correlation dimension using Grassberger-Procaccia algorithm
        # Subsample for efficiency
        n_samples = min(1000, len(points))
        indices = np.random.choice(len(points), n_samples, replace=False)
        points_sample = points[indices]

        # Compute pairwise distances
        dists = distance_matrix(points_sample, points_sample)
        dists = dists[np.triu_indices_from(dists, k=1)]

        # Range of radius values
        r_values = np.percentile(dists, np.linspace(5, 50, 10))

        # Correlation sum: C(r) = fraction of pairs with distance < r
        C_r = [np.sum(dists < r) / len(dists) for r in r_values]

        # Fit log-log relationship: log(C(r)) ~ D * log(r)
        log_r = np.log(r_values[np.array(C_r) > 0])
        log_C = np.log(np.array(C_r)[np.array(C_r) > 0])

        if len(log_r) < 2:
            return None

        # Linear regression in log-log space
        slope = np.polyfit(log_r, log_C, 1)[0]

        return max(0, min(slope, points.shape[1]))

    def _estimate_basins(
        self,
        X: np.ndarray,
        attractors: List[AttractorResult]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Estimate basins of attraction.

        Args:
            X: Trajectory data
            attractors: List of detected attractors

        Returns:
            Tuple of (basin_labels, basin_boundaries)
        """
        n_attractors = len(attractors)
        basin_labels = np.zeros(len(X), dtype=int)

        # Assign each point to nearest attractor
        for i, point in enumerate(X):
            min_dist = np.inf
            best_attractor = 0

            for j, attractor in enumerate(attractors):
                # Distance to attractor centroid
                centroid = np.mean(attractor.attractor_points, axis=0)
                dist = np.linalg.norm(point - centroid)

                if dist < min_dist:
                    min_dist = dist
                    best_attractor = j

            basin_labels[i] = best_attractor

        # Find boundary points (points near different basins)
        basin_boundaries = []
        for i in range(n_attractors):
            basin_points = X[basin_labels == i]

            if len(basin_points) == 0:
                continue

            # Find points on the boundary (neighboring other basins)
            # Simplified: just use points far from centroid
            centroid = np.mean(basin_points, axis=0)
            distances = np.linalg.norm(basin_points - centroid, axis=1)
            boundary_threshold = np.percentile(distances, 90)

            boundary_points = basin_points[distances > boundary_threshold]
            basin_boundaries.append(boundary_points)

        return basin_labels, basin_boundaries

    def _compute_poincare_sections(
        self,
        X: np.ndarray,
        section_normal: Optional[np.ndarray] = None,
        section_point: Optional[np.ndarray] = None
    ) -> List[PoincareSection]:
        """
        Compute Poincaré sections.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            section_normal: Normal vector to section plane
            section_point: Point on section plane

        Returns:
            List of PoincareSection objects
        """
        n_features = X.shape[1]

        # Default: use hyperplane perpendicular to first principal component
        if section_normal is None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca.fit(X)
            section_normal = pca.components_[0]

        if section_point is None:
            section_point = np.mean(X, axis=0)

        # Find intersections with section
        section_points = []
        return_times = []

        last_intersection_time = None

        for t in range(len(X) - 1):
            # Check if trajectory crosses section
            p1 = X[t] - section_point
            p2 = X[t + 1] - section_point

            # Signed distances to plane
            d1 = np.dot(p1, section_normal)
            d2 = np.dot(p2, section_normal)

            # Check for crossing (sign change)
            if d1 * d2 < 0:
                # Interpolate intersection point
                alpha = d1 / (d1 - d2)
                intersection = X[t] + alpha * (X[t + 1] - X[t])

                section_points.append(intersection)

                if last_intersection_time is not None:
                    return_time = (t - last_intersection_time) * self.dt
                    return_times.append(return_time)

                last_intersection_time = t

        if len(section_points) == 0:
            return []

        section_points = np.array(section_points)
        return_times = np.array(return_times) if len(return_times) > 0 else np.array([])

        # Construct return map
        if len(section_points) > 1:
            return_map = np.column_stack([
                section_points[:-1],
                section_points[1:]
            ])
        else:
            return_map = None

        poincare = PoincareSection(
            section_points=section_points,
            return_times=return_times,
            return_map=return_map,
            section_normal=section_normal,
            section_point=section_point
        )

        return [poincare]

    def _classify_topology(self, attractors: List[AttractorResult]) -> str:
        """
        Classify global phase space topology.

        Args:
            attractors: List of detected attractors

        Returns:
            Topology type string
        """
        if len(attractors) == 0:
            return "unknown"

        # Single attractor
        if len(attractors) == 1:
            return f"single_{attractors[0].attractor_type}"

        # Multiple attractors
        attractor_types = [a.attractor_type for a in attractors]

        if all(t == "fixed_point" for t in attractor_types):
            return f"multistable_{len(attractors)}_fixed_points"

        if any(t == "strange" for t in attractor_types):
            return "complex_multistable"

        return f"multistable_{len(attractors)}_attractors"

    def delay_embedding(
        self,
        time_series: np.ndarray,
        delay: int = 1,
        embedding_dim: int = 3
    ) -> np.ndarray:
        """
        Perform delay embedding (Takens' theorem).

        Reconstructs phase space from a scalar time series.

        Args:
            time_series: Scalar time series (n_timesteps,)
            delay: Time delay in samples
            embedding_dim: Embedding dimension

        Returns:
            Embedded trajectory (n_timesteps - delay*embedding_dim, embedding_dim)
        """
        n = len(time_series)
        n_vectors = n - delay * (embedding_dim - 1)

        if n_vectors <= 0:
            raise ValueError("Time series too short for embedding")

        embedded = np.zeros((n_vectors, embedding_dim))

        for i in range(embedding_dim):
            embedded[:, i] = time_series[i * delay:i * delay + n_vectors]

        return embedded

    def estimate_optimal_delay(
        self,
        time_series: np.ndarray,
        method: str = "autocorrelation"
    ) -> int:
        """
        Estimate optimal delay for embedding.

        Args:
            time_series: Scalar time series
            method: Method ("autocorrelation", "mutual_information")

        Returns:
            Optimal delay (in samples)
        """
        if method == "autocorrelation":
            # First zero crossing of autocorrelation
            autocorr = np.correlate(time_series, time_series, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]

            # Find first zero crossing
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            if len(zero_crossings) > 0:
                return zero_crossings[0]
            else:
                return len(time_series) // 10

        else:
            raise NotImplementedError(f"Method {method} not implemented")
