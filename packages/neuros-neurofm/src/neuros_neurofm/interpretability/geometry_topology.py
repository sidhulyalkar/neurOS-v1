"""
Manifold Geometry and Topological Data Analysis for NeuroFMX

This module provides tools for analyzing the geometric and topological structure
of neural trajectories in latent space, including:
- Riemannian manifold geometry (curvature, divergence, geodesics)
- Persistent homology and topological features
- Manifold visualization and dimensionality analysis

Author: neurOS Development Team
Date: 2025-10-25
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Core scientific computing
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

# Topological data analysis
import gudhi

# Manifold learning
import umap

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.collections import LineCollection
import seaborn as sns


@dataclass
class PersistenceResults:
    """Results from persistent homology computation."""
    persistence_pairs: List[Tuple[int, Tuple[float, float]]]
    betti_numbers: Dict[int, int]
    birth_death_pairs: Dict[int, np.ndarray]  # dimension -> (birth, death) array
    max_dimension: int


@dataclass
class ManifoldMetrics:
    """Metrics characterizing manifold geometry."""
    intrinsic_dim: float
    participation_ratio: float
    variance_explained_90: int
    mean_curvature: float
    curvature_std: float
    mean_divergence: float


class ManifoldGeometry:
    """
    Analyze Riemannian geometry of neural trajectories.

    Provides methods for estimating curvature, computing divergence,
    identifying slow manifolds, and measuring geodesic distances.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize manifold geometry analyzer.

        Args:
            device: Device for tensor computations ('cpu' or 'cuda')
        """
        self.device = device
        self.manifold_data = {}
        self.isomap_model = None

    def estimate_curvature(
        self,
        trajectories: np.ndarray,
        k_neighbors: int = 10
    ) -> np.ndarray:
        """
        Estimate Riemannian curvature via local PCA analysis.

        Uses the variation in local tangent spaces to approximate curvature.
        High curvature indicates regions where the manifold bends sharply.

        Args:
            trajectories: Array of shape (n_samples, n_dims) or (n_trials, n_timesteps, n_dims)
            k_neighbors: Number of neighbors for local analysis

        Returns:
            curvature: Array of curvature estimates at each point
        """
        # Reshape if needed
        if trajectories.ndim == 3:
            n_trials, n_timesteps, n_dims = trajectories.shape
            trajectories = trajectories.reshape(-1, n_dims)
            reshape_needed = True
        else:
            reshape_needed = False
            n_samples, n_dims = trajectories.shape

        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree')
        nbrs.fit(trajectories)
        distances, indices = nbrs.kneighbors(trajectories)

        curvature = np.zeros(len(trajectories))

        for i in range(len(trajectories)):
            # Get local neighborhood (excluding point itself)
            neighbor_indices = indices[i, 1:]
            local_points = trajectories[neighbor_indices]

            # Center the local points
            centered = local_points - trajectories[i]

            # Perform PCA on local neighborhood
            if len(centered) > 1:
                try:
                    U, S, Vt = svd(centered, full_matrices=False)

                    # Curvature estimate: ratio of smallest to largest singular values
                    # High curvature = high variation in all directions (bent manifold)
                    if S[0] > 1e-10:
                        curvature[i] = 1.0 - (S[-1] / S[0])
                    else:
                        curvature[i] = 0.0
                except:
                    curvature[i] = 0.0

        # Reshape back if needed
        if reshape_needed:
            curvature = curvature.reshape(n_trials, n_timesteps)

        self.manifold_data['curvature'] = curvature
        return curvature

    def compute_divergence(
        self,
        trajectories: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute divergence of latent flow field.

        Divergence measures expansion/contraction of the flow:
        - Positive divergence: expansion (unstable region)
        - Negative divergence: contraction (attractor basin)
        - Zero divergence: volume-preserving flow

        Args:
            trajectories: Array of shape (n_trials, n_timesteps, n_dims)
            dt: Time step between consecutive points

        Returns:
            divergence: Array of shape (n_trials, n_timesteps-1) with divergence estimates
        """
        if trajectories.ndim != 3:
            raise ValueError("Trajectories must be 3D: (n_trials, n_timesteps, n_dims)")

        n_trials, n_timesteps, n_dims = trajectories.shape

        # Compute velocity field
        velocities = np.diff(trajectories, axis=1) / dt

        # Estimate divergence using finite differences
        # div(v) ≈ ∂v_i/∂x_i summed over dimensions
        divergence = np.zeros((n_trials, n_timesteps - 1))

        for trial in range(n_trials):
            for t in range(n_timesteps - 2):
                # Estimate gradient of velocity field
                dv = velocities[trial, t + 1] - velocities[trial, t]
                dx = trajectories[trial, t + 1] - trajectories[trial, t]

                # Avoid division by zero
                norm_dx = np.linalg.norm(dx)
                if norm_dx > 1e-10:
                    # Approximate divergence as rate of change of velocity magnitude
                    divergence[trial, t] = np.dot(dv, dx) / (norm_dx ** 2)

        self.manifold_data['divergence'] = divergence
        return divergence

    def identify_slow_manifold(
        self,
        trajectories: np.ndarray,
        n_components: int = 3,
        return_projection: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Identify low-dimensional slow manifold (attractor).

        Uses PCA to find the subspace that captures most variance,
        which typically corresponds to slow dynamics.

        Args:
            trajectories: Array of shape (n_samples, n_dims) or (n_trials, n_timesteps, n_dims)
            n_components: Number of slow manifold dimensions
            return_projection: Whether to return projected trajectories

        Returns:
            slow_manifold: Projected trajectories on slow manifold
            info: Dictionary with PCA components, variance explained, etc.
        """
        # Reshape if needed
        if trajectories.ndim == 3:
            original_shape = trajectories.shape
            trajectories_flat = trajectories.reshape(-1, trajectories.shape[-1])
        else:
            original_shape = None
            trajectories_flat = trajectories

        # Fit PCA
        pca = PCA(n_components=n_components)
        slow_manifold = pca.fit_transform(trajectories_flat)

        # Reshape back if needed
        if original_shape is not None:
            slow_manifold = slow_manifold.reshape(
                original_shape[0], original_shape[1], n_components
            )

        info = {
            'components': pca.components_,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'mean': pca.mean_,
            'pca_model': pca
        }

        self.manifold_data['slow_manifold'] = slow_manifold
        self.manifold_data['pca_info'] = info

        if return_projection:
            return slow_manifold, info
        else:
            return info

    def geodesic_distance(
        self,
        point1: np.ndarray,
        point2: np.ndarray,
        trajectories: Optional[np.ndarray] = None,
        n_neighbors: int = 10
    ) -> float:
        """
        Compute geodesic distance along manifold using Isomap.

        Args:
            point1: First point (n_dims,)
            point2: Second point (n_dims,)
            trajectories: Reference trajectories for building graph (n_samples, n_dims)
            n_neighbors: Number of neighbors for graph construction

        Returns:
            distance: Geodesic distance between points
        """
        if trajectories is None:
            # Use Euclidean distance if no manifold reference
            return np.linalg.norm(point1 - point2)

        # Flatten trajectories if needed
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        # Add the query points to the dataset
        all_points = np.vstack([trajectories, point1.reshape(1, -1), point2.reshape(1, -1)])

        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        nbrs.fit(all_points)
        distances, indices = nbrs.kneighbors(all_points)

        # Build adjacency matrix
        n_points = len(all_points)
        adj_matrix = np.full((n_points, n_points), np.inf)

        for i in range(n_points):
            adj_matrix[i, indices[i]] = distances[i]
            adj_matrix[indices[i], i] = distances[i]

        # Compute shortest path (geodesic distance)
        graph_distances = shortest_path(
            csr_matrix(adj_matrix),
            method='auto',
            directed=False
        )

        # Get distance between the two query points
        idx1, idx2 = len(trajectories), len(trajectories) + 1
        geodesic_dist = graph_distances[idx1, idx2]

        return geodesic_dist

    def intrinsic_dimensionality(
        self,
        trajectories: np.ndarray,
        variance_threshold: float = 0.90
    ) -> ManifoldMetrics:
        """
        Estimate intrinsic dimensionality using participation ratio and variance threshold.

        Args:
            trajectories: Array of shape (n_samples, n_dims) or (n_trials, n_timesteps, n_dims)
            variance_threshold: Cumulative variance threshold (default 0.90)

        Returns:
            metrics: ManifoldMetrics object with dimensionality estimates
        """
        # Reshape if needed
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        # Compute PCA
        pca = PCA()
        pca.fit(trajectories)

        eigenvalues = pca.explained_variance_
        variance_ratios = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratios)

        # Participation ratio: (sum λ_i)^2 / (sum λ_i^2)
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

        # Number of dimensions to explain threshold variance
        variance_dim = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Estimate curvature if not already computed
        if 'curvature' not in self.manifold_data:
            curvature = self.estimate_curvature(trajectories)
        else:
            curvature = self.manifold_data['curvature']

        # Estimate divergence if not already computed
        if 'divergence' not in self.manifold_data:
            # Reshape for divergence computation
            n_samples = trajectories.shape[0]
            n_timesteps = int(np.sqrt(n_samples))
            if n_timesteps ** 2 == n_samples:
                traj_3d = trajectories.reshape(n_timesteps, n_timesteps, -1)[:1]
                divergence = self.compute_divergence(traj_3d)
            else:
                divergence = np.array([0.0])
        else:
            divergence = self.manifold_data['divergence']

        metrics = ManifoldMetrics(
            intrinsic_dim=participation_ratio,
            participation_ratio=participation_ratio,
            variance_explained_90=variance_dim,
            mean_curvature=np.mean(curvature),
            curvature_std=np.std(curvature),
            mean_divergence=np.mean(divergence)
        )

        self.manifold_data['metrics'] = metrics
        return metrics


class TopologicalAnalysis:
    """
    Persistent homology and topological data analysis.

    Uses Gudhi library to compute persistent homology, Betti numbers,
    and topological features of neural activity manifolds.
    """

    def __init__(self):
        """Initialize topological analysis."""
        self.persistence_data = {}

    def compute_persistence(
        self,
        point_cloud: np.ndarray,
        max_dimension: int = 2,
        max_edge_length: float = 2.0
    ) -> PersistenceResults:
        """
        Compute persistent homology using Gudhi.

        Args:
            point_cloud: Array of shape (n_points, n_dims)
            max_dimension: Maximum homology dimension to compute
            max_edge_length: Maximum edge length for Rips complex

        Returns:
            results: PersistenceResults object with persistence pairs and Betti numbers
        """
        # Flatten if needed
        if point_cloud.ndim == 3:
            point_cloud = point_cloud.reshape(-1, point_cloud.shape[-1])

        # Build Rips complex
        rips_complex = gudhi.RipsComplex(
            points=point_cloud,
            max_edge_length=max_edge_length
        )

        # Create simplex tree
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension + 1)

        # Compute persistence
        persistence = simplex_tree.persistence()

        # Organize persistence pairs by dimension
        birth_death_pairs = {dim: [] for dim in range(max_dimension + 1)}

        for dim, (birth, death) in persistence:
            if dim <= max_dimension:
                birth_death_pairs[dim].append([birth, death])

        # Convert to arrays
        for dim in birth_death_pairs:
            if birth_death_pairs[dim]:
                birth_death_pairs[dim] = np.array(birth_death_pairs[dim])
            else:
                birth_death_pairs[dim] = np.array([]).reshape(0, 2)

        # Compute Betti numbers at infinite time (features that never die)
        betti_numbers = {}
        for dim in range(max_dimension + 1):
            pairs = birth_death_pairs[dim]
            # Count features with infinite death time
            betti_numbers[dim] = np.sum(np.isinf(pairs[:, 1])) if len(pairs) > 0 else 0

        results = PersistenceResults(
            persistence_pairs=persistence,
            betti_numbers=betti_numbers,
            birth_death_pairs=birth_death_pairs,
            max_dimension=max_dimension
        )

        self.persistence_data['results'] = results
        return results

    def betti_numbers(
        self,
        persistence: Optional[PersistenceResults] = None,
        threshold: Optional[float] = None
    ) -> Dict[int, int]:
        """
        Extract Betti numbers from persistence diagram.

        Betti numbers characterize topological features:
        - β0: connected components
        - β1: loops/cycles
        - β2: voids

        Args:
            persistence: PersistenceResults object (uses stored if None)
            threshold: Filtration value to compute Betti numbers (None = infinity)

        Returns:
            betti_numbers: Dictionary mapping dimension to Betti number
        """
        if persistence is None:
            if 'results' not in self.persistence_data:
                raise ValueError("No persistence results available. Run compute_persistence first.")
            persistence = self.persistence_data['results']

        betti_numbers = {}

        for dim in range(persistence.max_dimension + 1):
            pairs = persistence.birth_death_pairs[dim]

            if len(pairs) == 0:
                betti_numbers[dim] = 0
                continue

            if threshold is None:
                # Count features that persist to infinity
                betti_numbers[dim] = np.sum(np.isinf(pairs[:, 1]))
            else:
                # Count features alive at threshold
                alive = (pairs[:, 0] <= threshold) & (
                    (pairs[:, 1] > threshold) | np.isinf(pairs[:, 1])
                )
                betti_numbers[dim] = np.sum(alive)

        return betti_numbers

    def compare_topologies(
        self,
        pers1: PersistenceResults,
        pers2: PersistenceResults,
        dimension: int = 1
    ) -> float:
        """
        Compare topologies using Wasserstein distance between persistence diagrams.

        Args:
            pers1: First persistence results
            pers2: Second persistence results
            dimension: Homology dimension to compare

        Returns:
            distance: Wasserstein distance between diagrams
        """
        # Get persistence diagrams for specified dimension
        diag1 = pers1.birth_death_pairs[dimension]
        diag2 = pers2.birth_death_pairs[dimension]

        # Remove infinite death times for Wasserstein distance
        diag1_finite = diag1[~np.isinf(diag1[:, 1])] if len(diag1) > 0 else np.array([]).reshape(0, 2)
        diag2_finite = diag2[~np.isinf(diag2[:, 1])] if len(diag2) > 0 else np.array([]).reshape(0, 2)

        # Compute Wasserstein distance using Gudhi
        try:
            distance = gudhi.wasserstein.wasserstein_distance(
                diag1_finite,
                diag2_finite,
                order=1.0
            )
        except:
            # Fallback to simple metric if Wasserstein fails
            if len(diag1_finite) == 0 and len(diag2_finite) == 0:
                distance = 0.0
            else:
                # Use difference in persistence
                pers1_sum = np.sum(diag1_finite[:, 1] - diag1_finite[:, 0]) if len(diag1_finite) > 0 else 0
                pers2_sum = np.sum(diag2_finite[:, 1] - diag2_finite[:, 0]) if len(diag2_finite) > 0 else 0
                distance = abs(pers1_sum - pers2_sum)

        return distance

    def persistence_diagram(
        self,
        persistence: Optional[PersistenceResults] = None,
        dimensions: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize persistence diagram (birth vs death plot).

        Args:
            persistence: PersistenceResults object (uses stored if None)
            dimensions: List of dimensions to plot (None = all)
            save_path: Path to save figure (None = don't save)

        Returns:
            fig: Matplotlib figure
        """
        if persistence is None:
            if 'results' not in self.persistence_data:
                raise ValueError("No persistence results available. Run compute_persistence first.")
            persistence = self.persistence_data['results']

        if dimensions is None:
            dimensions = list(range(persistence.max_dimension + 1))

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['C0', 'C1', 'C2', 'C3']
        labels = ['H₀ (components)', 'H₁ (loops)', 'H₂ (voids)', 'H₃']

        max_val = 0

        for dim in dimensions:
            if dim > persistence.max_dimension:
                continue

            pairs = persistence.birth_death_pairs[dim]

            if len(pairs) == 0:
                continue

            # Separate finite and infinite deaths
            finite_mask = ~np.isinf(pairs[:, 1])
            finite_pairs = pairs[finite_mask]
            infinite_pairs = pairs[~finite_mask]

            # Plot finite persistence
            if len(finite_pairs) > 0:
                ax.scatter(
                    finite_pairs[:, 0],
                    finite_pairs[:, 1],
                    c=colors[dim],
                    label=labels[dim],
                    alpha=0.6,
                    s=50
                )
                max_val = max(max_val, np.max(finite_pairs))

            # Plot infinite persistence as triangles at top
            if len(infinite_pairs) > 0:
                y_inf = max_val * 1.1 if max_val > 0 else 1.0
                ax.scatter(
                    infinite_pairs[:, 0],
                    np.full(len(infinite_pairs), y_inf),
                    c=colors[dim],
                    marker='^',
                    s=100,
                    alpha=0.8
                )
                max_val = max(max_val, y_inf)

        # Diagonal line (birth = death)
        if max_val > 0:
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Birth = Death')

        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title('Persistence Diagram', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def barcode_diagram(
        self,
        persistence: Optional[PersistenceResults] = None,
        dimensions: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize persistence barcode.

        Args:
            persistence: PersistenceResults object (uses stored if None)
            dimensions: List of dimensions to plot (None = all)
            save_path: Path to save figure (None = don't save)

        Returns:
            fig: Matplotlib figure
        """
        if persistence is None:
            if 'results' not in self.persistence_data:
                raise ValueError("No persistence results available. Run compute_persistence first.")
            persistence = self.persistence_data['results']

        if dimensions is None:
            dimensions = list(range(persistence.max_dimension + 1))

        fig, axes = plt.subplots(len(dimensions), 1, figsize=(12, 3 * len(dimensions)))

        if len(dimensions) == 1:
            axes = [axes]

        colors = ['C0', 'C1', 'C2', 'C3']
        labels = ['H₀ (components)', 'H₁ (loops)', 'H₂ (voids)', 'H₃']

        for idx, dim in enumerate(dimensions):
            ax = axes[idx]

            if dim > persistence.max_dimension:
                continue

            pairs = persistence.birth_death_pairs[dim]

            if len(pairs) == 0:
                ax.text(0.5, 0.5, 'No features', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(labels[dim], fontsize=12, fontweight='bold')
                continue

            # Sort by persistence (death - birth)
            persistence_vals = pairs[:, 1] - pairs[:, 0]
            persistence_vals[np.isinf(persistence_vals)] = np.max(pairs[~np.isinf(pairs[:, 1]), 1]) * 1.5
            sorted_idx = np.argsort(persistence_vals)[::-1]
            sorted_pairs = pairs[sorted_idx]

            # Plot bars
            for i, (birth, death) in enumerate(sorted_pairs):
                if np.isinf(death):
                    death = np.max(pairs[~np.isinf(pairs[:, 1]), 1]) * 1.2 if np.any(~np.isinf(pairs[:, 1])) else birth + 1
                    ax.barh(i, death - birth, left=birth, height=0.8, color=colors[dim], alpha=0.8, edgecolor='red', linewidth=2)
                else:
                    ax.barh(i, death - birth, left=birth, height=0.8, color=colors[dim], alpha=0.6)

            ax.set_xlabel('Filtration Value', fontsize=11)
            ax.set_ylabel('Feature Index', fontsize=11)
            ax.set_title(labels[dim], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class ManifoldVisualization:
    """
    Visualization tools for neural manifolds.

    Provides UMAP, Isomap embeddings, 3D visualizations,
    and curvature heatmaps.
    """

    def __init__(self):
        """Initialize manifold visualization."""
        self.embeddings = {}

    def umap_embedding(
        self,
        latents: np.ndarray,
        temporal_coloring: bool = True,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute UMAP embedding with temporal or task-based coloring.

        Args:
            latents: Array of shape (n_samples, n_dims) or (n_trials, n_timesteps, n_dims)
            temporal_coloring: If True, color by time; if False, color by labels
            n_components: Number of UMAP dimensions
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance parameter for UMAP
            metric: Distance metric for UMAP
            labels: Task/condition labels for coloring (n_trials,) if temporal_coloring=False

        Returns:
            embedding: UMAP embedding of shape (n_samples, n_components)
            info: Dictionary with UMAP model and metadata
        """
        # Handle 3D input
        original_shape = None
        if latents.ndim == 3:
            original_shape = latents.shape
            n_trials, n_timesteps, n_dims = latents.shape
            latents_flat = latents.reshape(-1, n_dims)
        else:
            latents_flat = latents

        # Fit UMAP
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )

        embedding = umap_model.fit_transform(latents_flat)

        # Prepare coloring
        if temporal_coloring and original_shape is not None:
            # Color by time
            time_colors = np.tile(np.arange(n_timesteps), n_trials)
            trial_labels = np.repeat(np.arange(n_trials), n_timesteps)
        elif labels is not None:
            if original_shape is not None:
                # Expand labels to all timesteps
                time_colors = np.repeat(labels, n_timesteps)
                trial_labels = np.repeat(np.arange(n_trials), n_timesteps)
            else:
                time_colors = labels
                trial_labels = np.arange(len(labels))
        else:
            time_colors = np.arange(len(latents_flat))
            trial_labels = np.arange(len(latents_flat))

        info = {
            'model': umap_model,
            'n_components': n_components,
            'time_colors': time_colors,
            'trial_labels': trial_labels,
            'temporal_coloring': temporal_coloring
        }

        self.embeddings['umap'] = {'embedding': embedding, 'info': info}

        return embedding, info

    def isomap_embedding(
        self,
        latents: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 10
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute Isomap embedding (geodesic distance preservation).

        Args:
            latents: Array of shape (n_samples, n_dims) or (n_trials, n_timesteps, n_dims)
            n_components: Number of embedding dimensions
            n_neighbors: Number of neighbors for graph construction

        Returns:
            embedding: Isomap embedding of shape (n_samples, n_components)
            info: Dictionary with Isomap model and reconstruction error
        """
        # Flatten if needed
        if latents.ndim == 3:
            original_shape = latents.shape
            latents_flat = latents.reshape(-1, latents.shape[-1])
        else:
            latents_flat = latents
            original_shape = None

        # Fit Isomap
        isomap_model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        embedding = isomap_model.fit_transform(latents_flat)

        info = {
            'model': isomap_model,
            'n_components': n_components,
            'reconstruction_error': isomap_model.reconstruction_error(),
            'original_shape': original_shape
        }

        self.embeddings['isomap'] = {'embedding': embedding, 'info': info}

        return embedding, info

    def plot_manifold_3d(
        self,
        embedding: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        title: str = "3D Manifold Visualization",
        color_label: str = "Time",
        save_path: Optional[str] = None,
        use_umap: bool = True
    ) -> go.Figure:
        """
        Create interactive 3D plotly visualization of manifold.

        Args:
            embedding: 3D embedding array (n_samples, 3). If None, uses stored UMAP/Isomap
            colors: Color values for each point
            title: Plot title
            color_label: Label for color scale
            save_path: Path to save HTML file (None = don't save)
            use_umap: If True, use UMAP embedding; else use Isomap

        Returns:
            fig: Plotly figure object
        """
        # Get embedding if not provided
        if embedding is None:
            key = 'umap' if use_umap else 'isomap'
            if key not in self.embeddings:
                raise ValueError(f"No {key} embedding available. Run {key}_embedding first.")

            embedding = self.embeddings[key]['embedding']

            # Get 3D version if needed
            if embedding.shape[1] != 3:
                raise ValueError(f"Stored embedding has {embedding.shape[1]} dimensions, need 3 for 3D plot.")

            # Get colors from info if not provided
            if colors is None and use_umap:
                colors = self.embeddings[key]['info']['time_colors']

        if embedding.shape[1] != 3:
            raise ValueError(f"Embedding must be 3D, got shape {embedding.shape}")

        # Default colors
        if colors is None:
            colors = np.arange(len(embedding))

        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_label),
                opacity=0.7
            ),
            hovertemplate='<b>X:</b> %{x:.3f}<br><b>Y:</b> %{y:.3f}<br><b>Z:</b> %{z:.3f}<br><b>' +
                         color_label + ':</b> %{marker.color:.2f}<extra></extra>'
        )])

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, family='Arial Black')),
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=900,
            height=700,
            hovermode='closest'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_curvature_heatmap(
        self,
        trajectories: np.ndarray,
        curvature: np.ndarray,
        save_path: Optional[str] = None,
        cmap: str = 'hot'
    ) -> plt.Figure:
        """
        Plot curvature distribution along trajectories as heatmap.

        Args:
            trajectories: Array of shape (n_trials, n_timesteps, n_dims)
            curvature: Array of shape (n_trials, n_timesteps) with curvature values
            save_path: Path to save figure (None = don't save)
            cmap: Colormap name

        Returns:
            fig: Matplotlib figure
        """
        if trajectories.ndim != 3:
            raise ValueError("Trajectories must be 3D: (n_trials, n_timesteps, n_dims)")

        if curvature.shape != trajectories.shape[:2]:
            raise ValueError(f"Curvature shape {curvature.shape} doesn't match trajectory shape {trajectories.shape[:2]}")

        n_trials, n_timesteps, n_dims = trajectories.shape

        # Create figure with two subplots
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[20, 1])

        # Heatmap
        ax_heat = fig.add_subplot(gs[0, 0])
        im = ax_heat.imshow(
            curvature,
            aspect='auto',
            cmap=cmap,
            interpolation='nearest'
        )
        ax_heat.set_xlabel('Time Step', fontsize=12)
        ax_heat.set_ylabel('Trial', fontsize=12)
        ax_heat.set_title('Curvature Heatmap Along Trajectories', fontsize=14, fontweight='bold')

        # Colorbar
        ax_cbar = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Curvature', fontsize=11)

        # Average curvature over time
        ax_avg = fig.add_subplot(gs[1, 0])
        mean_curv = np.mean(curvature, axis=0)
        std_curv = np.std(curvature, axis=0)

        time_steps = np.arange(n_timesteps)
        ax_avg.plot(time_steps, mean_curv, 'b-', linewidth=2, label='Mean')
        ax_avg.fill_between(
            time_steps,
            mean_curv - std_curv,
            mean_curv + std_curv,
            alpha=0.3,
            label='±1 SD'
        )
        ax_avg.set_xlabel('Time Step', fontsize=12)
        ax_avg.set_ylabel('Curvature', fontsize=12)
        ax_avg.set_title('Average Curvature Over Time', fontsize=12, fontweight='bold')
        ax_avg.legend()
        ax_avg.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_2d_embedding(
        self,
        embedding: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        title: str = "2D Manifold Embedding",
        color_label: str = "Time",
        save_path: Optional[str] = None,
        use_umap: bool = True,
        show_trajectories: bool = False,
        n_trials: Optional[int] = None
    ) -> plt.Figure:
        """
        Plot 2D embedding with optional trajectory lines.

        Args:
            embedding: 2D embedding array (n_samples, 2)
            colors: Color values for each point
            title: Plot title
            color_label: Label for color scale
            save_path: Path to save figure
            use_umap: If True, use UMAP; else use Isomap
            show_trajectories: If True, draw lines connecting sequential points
            n_trials: Number of trials (needed if show_trajectories=True)

        Returns:
            fig: Matplotlib figure
        """
        # Get embedding if not provided
        if embedding is None:
            key = 'umap' if use_umap else 'isomap'
            if key not in self.embeddings:
                raise ValueError(f"No {key} embedding available.")

            embedding = self.embeddings[key]['embedding']

            if colors is None and use_umap:
                colors = self.embeddings[key]['info']['time_colors']

        if embedding.shape[1] != 2:
            raise ValueError(f"Embedding must be 2D, got shape {embedding.shape}")

        if colors is None:
            colors = np.arange(len(embedding))

        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw trajectories if requested
        if show_trajectories and n_trials is not None:
            n_timesteps = len(embedding) // n_trials
            for trial in range(n_trials):
                start_idx = trial * n_timesteps
                end_idx = (trial + 1) * n_timesteps
                traj = embedding[start_idx:end_idx]
                ax.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.2, linewidth=0.5)

        # Scatter plot
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors,
            cmap='viridis',
            s=30,
            alpha=0.6,
            edgecolors='none'
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_label, fontsize=12)

        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


# ============================================================================
# Example Usage and Demonstrations
# ============================================================================

def example_manifold_analysis():
    """
    Comprehensive example of manifold geometry analysis.
    """
    print("=" * 80)
    print("MANIFOLD GEOMETRY ANALYSIS EXAMPLE")
    print("=" * 80)

    # Generate synthetic neural trajectories
    np.random.seed(42)
    n_trials = 10
    n_timesteps = 100
    latent_dim = 50

    # Create trajectories on a noisy manifold
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    trajectories = np.zeros((n_trials, n_timesteps, latent_dim))

    for trial in range(n_trials):
        # Base trajectory: spiral in low dimensions
        phase = trial * 2 * np.pi / n_trials
        trajectories[trial, :, 0] = np.cos(t + phase) * (1 + 0.5 * t / (4 * np.pi))
        trajectories[trial, :, 1] = np.sin(t + phase) * (1 + 0.5 * t / (4 * np.pi))
        trajectories[trial, :, 2] = t / (4 * np.pi)

        # Add noise in higher dimensions
        trajectories[trial, :, 3:] = np.random.randn(n_timesteps, latent_dim - 3) * 0.1

    print(f"\nGenerated trajectories: {trajectories.shape}")

    # Initialize analyzer
    manifold = ManifoldGeometry()

    # 1. Estimate curvature
    print("\n1. Estimating curvature...")
    curvature = manifold.estimate_curvature(trajectories, k_neighbors=10)
    print(f"   Curvature shape: {curvature.shape}")
    print(f"   Mean curvature: {np.mean(curvature):.4f}")
    print(f"   Max curvature: {np.max(curvature):.4f}")

    # 2. Compute divergence
    print("\n2. Computing divergence...")
    divergence = manifold.compute_divergence(trajectories, dt=1.0)
    print(f"   Divergence shape: {divergence.shape}")
    print(f"   Mean divergence: {np.mean(divergence):.4f}")

    # 3. Identify slow manifold
    print("\n3. Identifying slow manifold...")
    slow_manifold, info = manifold.identify_slow_manifold(trajectories, n_components=3)
    print(f"   Slow manifold shape: {slow_manifold.shape}")
    print(f"   Variance explained: {info['explained_variance_ratio']}")
    print(f"   Cumulative variance: {info['cumulative_variance']}")

    # 4. Intrinsic dimensionality
    print("\n4. Estimating intrinsic dimensionality...")
    metrics = manifold.intrinsic_dimensionality(trajectories)
    print(f"   Participation ratio: {metrics.participation_ratio:.2f}")
    print(f"   Dimensions for 90% variance: {metrics.variance_explained_90}")
    print(f"   Mean curvature: {metrics.mean_curvature:.4f}")

    # 5. Geodesic distance
    print("\n5. Computing geodesic distances...")
    point1 = trajectories[0, 0, :]
    point2 = trajectories[0, -1, :]
    euclidean_dist = np.linalg.norm(point1 - point2)
    geodesic_dist = manifold.geodesic_distance(point1, point2, trajectories)
    print(f"   Euclidean distance: {euclidean_dist:.4f}")
    print(f"   Geodesic distance: {geodesic_dist:.4f}")
    print(f"   Ratio (geodesic/euclidean): {geodesic_dist/euclidean_dist:.4f}")

    return trajectories, manifold, metrics


def example_topological_analysis():
    """
    Comprehensive example of topological data analysis.
    """
    print("\n" + "=" * 80)
    print("TOPOLOGICAL DATA ANALYSIS EXAMPLE")
    print("=" * 80)

    # Generate synthetic point cloud with topological features
    np.random.seed(42)

    # Create a circle (1D loop) in 2D
    n_points = 200
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    circle += np.random.randn(n_points, 2) * 0.1  # Add noise

    # Add some random points
    random_points = np.random.randn(50, 2) * 0.3
    point_cloud = np.vstack([circle, random_points])

    print(f"\nGenerated point cloud: {point_cloud.shape}")

    # Initialize analyzer
    topo = TopologicalAnalysis()

    # 1. Compute persistence
    print("\n1. Computing persistent homology...")
    results = topo.compute_persistence(
        point_cloud,
        max_dimension=2,
        max_edge_length=1.0
    )
    print(f"   Found {len(results.persistence_pairs)} persistence pairs")
    print(f"   Max dimension: {results.max_dimension}")

    # 2. Betti numbers
    print("\n2. Computing Betti numbers...")
    betti_inf = topo.betti_numbers(results, threshold=None)
    betti_05 = topo.betti_numbers(results, threshold=0.5)
    print(f"   Betti numbers (infinite): {betti_inf}")
    print(f"   Betti numbers (threshold=0.5): {betti_05}")
    print(f"   β0: {betti_inf[0]} connected components")
    print(f"   β1: {betti_inf[1]} loops/cycles")

    # 3. Compare topologies
    print("\n3. Comparing topologies...")
    # Create another point cloud (sphere)
    n_sphere = 200
    phi = np.random.uniform(0, 2 * np.pi, n_sphere)
    sphere = np.column_stack([1.5 * np.cos(phi), 1.5 * np.sin(phi)])
    sphere += np.random.randn(n_sphere, 2) * 0.1

    results2 = topo.compute_persistence(sphere, max_dimension=1, max_edge_length=1.0)
    distance = topo.compare_topologies(results, results2, dimension=1)
    print(f"   Wasserstein distance between topologies: {distance:.4f}")

    # 4. Visualize persistence diagram
    print("\n4. Generating persistence diagram...")
    fig_diagram = topo.persistence_diagram(results)

    # 5. Visualize barcode
    print("5. Generating barcode diagram...")
    fig_barcode = topo.barcode_diagram(results)

    plt.show()

    return point_cloud, results, topo


def example_manifold_visualization():
    """
    Comprehensive example of manifold visualization.
    """
    print("\n" + "=" * 80)
    print("MANIFOLD VISUALIZATION EXAMPLE")
    print("=" * 80)

    # Generate synthetic trajectories
    np.random.seed(42)
    n_trials = 20
    n_timesteps = 150
    latent_dim = 100

    # Create complex manifold: double helix
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    trajectories = np.zeros((n_trials, n_timesteps, latent_dim))

    for trial in range(n_trials):
        helix = trial % 2  # Which helix
        phase = (trial // 2) * 2 * np.pi / (n_trials // 2)

        trajectories[trial, :, 0] = np.cos(t + phase) * (1 + helix * 0.5)
        trajectories[trial, :, 1] = np.sin(t + phase) * (1 + helix * 0.5)
        trajectories[trial, :, 2] = t / np.pi + helix * 2

        # Add noise
        trajectories[trial, :, 3:] = np.random.randn(n_timesteps, latent_dim - 3) * 0.05

    print(f"\nGenerated trajectories: {trajectories.shape}")

    # Flatten for embedding
    latents_flat = trajectories.reshape(-1, latent_dim)

    # Initialize visualizer
    viz = ManifoldVisualization()

    # 1. UMAP embedding (2D)
    print("\n1. Computing UMAP embedding (2D)...")
    umap_2d, umap_info = viz.umap_embedding(
        trajectories,
        temporal_coloring=True,
        n_components=2
    )
    print(f"   UMAP 2D shape: {umap_2d.shape}")

    # 2. UMAP embedding (3D)
    print("\n2. Computing UMAP embedding (3D)...")
    umap_3d, _ = viz.umap_embedding(
        trajectories,
        temporal_coloring=True,
        n_components=3
    )
    print(f"   UMAP 3D shape: {umap_3d.shape}")

    # 3. Isomap embedding
    print("\n3. Computing Isomap embedding...")
    isomap_2d, isomap_info = viz.isomap_embedding(
        trajectories,
        n_components=2,
        n_neighbors=15
    )
    print(f"   Isomap shape: {isomap_2d.shape}")
    print(f"   Reconstruction error: {isomap_info['reconstruction_error']:.4f}")

    # 4. Plot 2D embeddings
    print("\n4. Generating 2D embedding plots...")
    fig_umap = viz.plot_2d_embedding(
        umap_2d,
        colors=umap_info['time_colors'],
        title="UMAP Embedding (2D)",
        show_trajectories=True,
        n_trials=n_trials
    )

    fig_isomap = viz.plot_2d_embedding(
        isomap_2d,
        colors=np.tile(np.arange(n_timesteps), n_trials),
        title="Isomap Embedding (2D)",
        use_umap=False,
        show_trajectories=True,
        n_trials=n_trials
    )

    # 5. Plot 3D interactive
    print("\n5. Generating interactive 3D plot...")
    fig_3d = viz.plot_manifold_3d(
        umap_3d,
        colors=umap_info['time_colors'],
        title="UMAP 3D Manifold Visualization"
    )

    # 6. Curvature heatmap
    print("\n6. Generating curvature heatmap...")
    manifold = ManifoldGeometry()
    curvature = manifold.estimate_curvature(trajectories, k_neighbors=10)

    fig_curv = viz.plot_curvature_heatmap(
        trajectories,
        curvature,
        cmap='hot'
    )

    plt.show()

    return viz, trajectories


def run_all_examples():
    """Run all demonstration examples."""
    print("\n" + "#" * 80)
    print("# NEUROFMX MANIFOLD GEOMETRY & TOPOLOGICAL DATA ANALYSIS")
    print("# Comprehensive Demonstration")
    print("#" * 80)

    # Run examples
    trajectories, manifold, metrics = example_manifold_analysis()
    point_cloud, results, topo = example_topological_analysis()
    viz, traj_viz = example_manifold_visualization()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll analyses completed successfully!")
    print("Figures have been generated and are ready for display.")
    print("\nKey capabilities demonstrated:")
    print("  - Riemannian curvature estimation")
    print("  - Divergence computation")
    print("  - Slow manifold identification")
    print("  - Intrinsic dimensionality analysis")
    print("  - Persistent homology computation")
    print("  - Betti number extraction")
    print("  - Topology comparison (Wasserstein distance)")
    print("  - UMAP and Isomap embeddings")
    print("  - Interactive 3D visualizations")
    print("  - Curvature heatmaps")


if __name__ == "__main__":
    # Run comprehensive examples
    run_all_examples()
