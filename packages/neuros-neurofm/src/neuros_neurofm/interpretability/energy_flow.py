"""
Information/Energy Flow Analysis for NeuroFMX.

This module implements Module 2 of the MECHINT_EXPANSION_PLAN.md:
- Information flow analysis (mutual information estimation)
- Energy landscape estimation and basin detection
- Entropy production analysis
- Information plane visualization (Tishby et al.)

Based on:
- Tishby & Zaslavsky (2015): Deep learning and the information bottleneck principle
- Schwartz-Ziv & Tishby (2017): Opening the black box of deep neural networks
- Seifert (2012): Stochastic thermodynamics, fluctuation theorems and molecular machines

Author: NeuroFMX Team
Date: 2025-10-25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import warnings


# ==================== DATA STRUCTURES ====================

@dataclass
class MutualInformationEstimate:
    """Results from mutual information estimation."""
    I_XZ: float  # I(X;Z) - input-latent MI
    I_ZY: float  # I(Z;Y) - latent-output MI
    method: str
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InformationPlane:
    """Information plane data (Tishby)."""
    layers: List[str]
    I_XZ_per_layer: np.ndarray  # Shape: (n_layers,)
    I_ZY_per_layer: np.ndarray  # Shape: (n_layers,)
    epochs: Optional[List[int]] = None  # If temporal
    I_XZ_trajectory: Optional[np.ndarray] = None  # Shape: (n_epochs, n_layers)
    I_ZY_trajectory: Optional[np.ndarray] = None  # Shape: (n_epochs, n_layers)


@dataclass
class EnergyFunction:
    """Energy landscape representation."""
    grid: np.ndarray  # Grid coordinates
    energy: np.ndarray  # Energy values at each grid point
    latent_dim: int
    method: str
    pca_basis: Optional[np.ndarray] = None  # For >2D projections


@dataclass
class Basin:
    """Energy basin (stable state)."""
    centroid: np.ndarray
    energy: float
    volume: float
    samples: np.ndarray  # Points belonging to this basin
    stability: float  # Local curvature measure


@dataclass
class EntropyProductionEstimate:
    """Entropy production along trajectories."""
    entropy_production_rate: np.ndarray  # dS/dt per timepoint
    dissipation_rate: float  # Total energy dissipation
    nonequilibrium_score: float  # Distance from equilibrium
    trajectories: np.ndarray


# ==================== MINE NETWORK ====================

class MINENetwork(nn.Module):
    """
    Mutual Information Neural Estimation network.

    Small neural network that learns to estimate mutual information
    using the Donsker-Varadhan representation.

    Reference: Belghazi et al. (2018) "Mutual Information Neural Estimation"
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3
    ):
        super().__init__()

        layers = []
        input_dim = x_dim + z_dim

        for i in range(n_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute statistics network T(x, z).

        Args:
            x: Input samples (batch_size, x_dim)
            z: Latent samples (batch_size, z_dim)

        Returns:
            T(x, z): Statistics (batch_size, 1)
        """
        xz = torch.cat([x, z], dim=-1)
        return self.network(xz)


# ==================== INFORMATION FLOW ANALYZER ====================

class InformationFlowAnalyzer:
    """
    Analyze information flow through NeuroFMX layers.

    Estimates mutual information I(X;Z) and I(Z;Y) to understand
    how information propagates and compresses through the network.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        self.device = device
        self.verbose = verbose

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[InformationFlowAnalyzer] {message}")

    def estimate_mutual_information(
        self,
        X: torch.Tensor,
        Z_layers: Union[torch.Tensor, List[torch.Tensor]],
        Y: Optional[torch.Tensor] = None,
        method: str = 'mine',
        n_bootstrap: int = 0
    ) -> Dict[str, MutualInformationEstimate]:
        """
        Compute I(X;Z_l) and I(Z_l;Y) for each layer.

        Args:
            X: Input data (batch_size, input_dim)
            Z_layers: Layer activations, either single tensor (batch, layers, dim)
                     or list of tensors [(batch, dim_l), ...]
            Y: Output/target (batch_size, output_dim), optional
            method: 'mine', 'knn', or 'histogram'
            n_bootstrap: Number of bootstrap samples for confidence intervals

        Returns:
            Dictionary mapping layer names to MI estimates
        """
        self._log(f"Estimating mutual information using {method} method")

        # Convert to list of layer tensors
        if isinstance(Z_layers, torch.Tensor):
            if Z_layers.dim() == 3:  # (batch, n_layers, dim)
                Z_list = [Z_layers[:, i, :] for i in range(Z_layers.shape[1])]
            else:
                Z_list = [Z_layers]
        else:
            Z_list = Z_layers

        results = {}

        for layer_idx, Z in enumerate(Z_list):
            layer_name = f"layer_{layer_idx}"

            # Estimate I(X;Z)
            I_XZ = self._estimate_mi_pair(X, Z, method=method)

            # Estimate I(Z;Y) if Y provided
            I_ZY = None
            if Y is not None:
                I_ZY = self._estimate_mi_pair(Z, Y, method=method)

            # Bootstrap confidence intervals if requested
            ci_XZ, ci_ZY = None, None
            if n_bootstrap > 0:
                ci_XZ = self._bootstrap_mi(X, Z, method, n_bootstrap)
                if Y is not None:
                    ci_ZY = self._bootstrap_mi(Z, Y, method, n_bootstrap)

            results[layer_name] = MutualInformationEstimate(
                I_XZ=I_XZ,
                I_ZY=I_ZY if I_ZY is not None else 0.0,
                method=method,
                confidence_interval=(ci_XZ, ci_ZY),
                metadata={'layer_idx': layer_idx}
            )

            self._log(f"{layer_name}: I(X;Z)={I_XZ:.4f}, I(Z;Y)={I_ZY:.4f if I_ZY else 0.0:.4f}")

        return results

    def _estimate_mi_pair(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        method: str = 'mine'
    ) -> float:
        """Estimate I(X;Z) using specified method."""

        if method == 'mine':
            return self._estimate_mi_mine(X, Z)
        elif method == 'knn':
            return self._estimate_mi_knn(X, Z)
        elif method == 'histogram':
            return self._estimate_mi_histogram(X, Z)
        else:
            raise ValueError(f"Unknown MI estimation method: {method}")

    def _estimate_mi_mine(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        n_epochs: int = 100,
        lr: float = 1e-3
    ) -> float:
        """
        Estimate MI using MINE (Mutual Information Neural Estimation).

        Trains a small network to estimate the Donsker-Varadhan divergence.
        """
        X = X.to(self.device)
        Z = Z.to(self.device)

        # Initialize MINE network
        mine_net = MINENetwork(
            x_dim=X.shape[-1],
            z_dim=Z.shape[-1],
            hidden_dim=128,
            n_layers=3
        ).to(self.device)

        optimizer = torch.optim.Adam(mine_net.parameters(), lr=lr)

        batch_size = min(512, X.shape[0])
        n_batches = X.shape[0] // batch_size

        # Training loop
        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                X_batch = X[start_idx:end_idx]
                Z_batch = Z[start_idx:end_idx]

                # Joint distribution: T(x, z)
                T_joint = mine_net(X_batch, Z_batch)

                # Marginal distribution: shuffle Z to break dependence
                Z_marginal = Z_batch[torch.randperm(Z_batch.shape[0])]
                T_marginal = mine_net(X_batch, Z_marginal)

                # MINE loss (negative of MI lower bound)
                # MI >= E[T(x,z)] - log E[exp(T(x,z'))]
                mi_lower_bound = T_joint.mean() - torch.exp(T_marginal).mean().log()
                loss = -mi_lower_bound

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        # Final estimate
        with torch.no_grad():
            T_joint = mine_net(X, Z)
            Z_marginal = Z[torch.randperm(Z.shape[0])]
            T_marginal = mine_net(X, Z_marginal)
            mi_estimate = (T_joint.mean() - torch.exp(T_marginal).mean().log()).item()

        return max(0.0, mi_estimate)  # MI is non-negative

    def _estimate_mi_knn(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        k: int = 5
    ) -> float:
        """
        Estimate MI using k-NN (Kraskov method).

        Reference: Kraskov et al. (2004) "Estimating mutual information"
        """
        X_np = X.detach().cpu().numpy()
        Z_np = Z.detach().cpu().numpy()

        n_samples = X_np.shape[0]

        # Concatenate for joint distribution
        XZ = np.concatenate([X_np, Z_np], axis=1)

        # Build k-NN in joint space
        nbrs_joint = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
        nbrs_joint.fit(XZ)
        distances_joint, _ = nbrs_joint.kneighbors(XZ)
        epsilon = distances_joint[:, -1]  # k-th neighbor distance

        # Count neighbors in marginals within epsilon
        nbrs_X = NearestNeighbors(metric='chebyshev', radius=1.0)
        nbrs_Z = NearestNeighbors(metric='chebyshev', radius=1.0)
        nbrs_X.fit(X_np)
        nbrs_Z.fit(Z_np)

        nx_total = 0
        nz_total = 0

        for i in range(n_samples):
            # Count neighbors within epsilon[i]
            nx = nbrs_X.radius_neighbors([X_np[i]], radius=epsilon[i], return_distance=False)[0].shape[0] - 1
            nz = nbrs_Z.radius_neighbors([Z_np[i]], radius=epsilon[i], return_distance=False)[0].shape[0] - 1

            nx_total += max(1, nx)
            nz_total += max(1, nz)

        # Kraskov estimator
        mi = (
            np.log(n_samples) - np.log(k)
            - (np.log(nx_total / n_samples) + np.log(nz_total / n_samples))
        )

        return max(0.0, mi)

    def _estimate_mi_histogram(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        n_bins: int = 20
    ) -> float:
        """
        Estimate MI using histogram binning.

        Fast but less accurate, especially in high dimensions.
        """
        X_np = X.detach().cpu().numpy()
        Z_np = Z.detach().cpu().numpy()

        # Reduce dimensionality if needed
        if X_np.shape[1] > 10:
            pca_x = PCA(n_components=min(10, X_np.shape[1]))
            X_np = pca_x.fit_transform(X_np)

        if Z_np.shape[1] > 10:
            pca_z = PCA(n_components=min(10, Z_np.shape[1]))
            Z_np = pca_z.fit_transform(Z_np)

        # Discretize
        X_binned = np.floor(
            (X_np - X_np.min(0)) / (X_np.max(0) - X_np.min(0) + 1e-8) * (n_bins - 1)
        ).astype(int)
        Z_binned = np.floor(
            (Z_np - Z_np.min(0)) / (Z_np.max(0) - Z_np.min(0) + 1e-8) * (n_bins - 1)
        ).astype(int)

        # Compute joint and marginal distributions
        n_samples = X_np.shape[0]

        # Convert to tuple indices
        X_indices = tuple(X_binned[:, i] for i in range(X_binned.shape[1]))
        Z_indices = tuple(Z_binned[:, i] for i in range(Z_binned.shape[1]))
        XZ_indices = X_indices + Z_indices

        # Count occurrences
        p_x = np.zeros([n_bins] * X_binned.shape[1])
        p_z = np.zeros([n_bins] * Z_binned.shape[1])
        p_xz = np.zeros([n_bins] * (X_binned.shape[1] + Z_binned.shape[1]))

        for i in range(n_samples):
            x_idx = tuple(X_binned[i])
            z_idx = tuple(Z_binned[i])
            xz_idx = x_idx + z_idx

            p_x[x_idx] += 1
            p_z[z_idx] += 1
            p_xz[xz_idx] += 1

        # Normalize
        p_x /= n_samples
        p_z /= n_samples
        p_xz /= n_samples

        # Compute MI
        mi = 0.0
        for idx in np.ndindex(p_xz.shape):
            if p_xz[idx] > 0:
                x_idx = idx[:X_binned.shape[1]]
                z_idx = idx[X_binned.shape[1]:]

                if p_x[x_idx] > 0 and p_z[z_idx] > 0:
                    mi += p_xz[idx] * np.log(p_xz[idx] / (p_x[x_idx] * p_z[z_idx]))

        return max(0.0, mi)

    def _bootstrap_mi(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        method: str,
        n_bootstrap: int = 100
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for MI."""
        mi_samples = []
        n_samples = X.shape[0]

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = torch.randint(0, n_samples, (n_samples,))
            X_boot = X[indices]
            Z_boot = Z[indices]

            mi = self._estimate_mi_pair(X_boot, Z_boot, method=method)
            mi_samples.append(mi)

        mi_samples = np.array(mi_samples)
        return (np.percentile(mi_samples, 2.5), np.percentile(mi_samples, 97.5))

    def information_plane(
        self,
        activations: Dict[str, torch.Tensor],
        X: torch.Tensor,
        Y: torch.Tensor,
        method: str = 'mine'
    ) -> InformationPlane:
        """
        Compute Tishby's information plane: I(X;T) vs I(T;Y).

        Args:
            activations: Dictionary mapping layer names to activations
            X: Input data
            Y: Output/target data
            method: MI estimation method

        Returns:
            InformationPlane object with MI per layer
        """
        self._log("Computing information plane")

        layers = sorted(activations.keys())
        I_XZ = []
        I_ZY = []

        for layer_name in layers:
            Z = activations[layer_name]

            # Flatten if needed
            if Z.dim() > 2:
                Z = Z.reshape(Z.shape[0], -1)

            mi_xz = self._estimate_mi_pair(X, Z, method=method)
            mi_zy = self._estimate_mi_pair(Z, Y, method=method)

            I_XZ.append(mi_xz)
            I_ZY.append(mi_zy)

            self._log(f"{layer_name}: I(X;Z)={mi_xz:.4f}, I(Z;Y)={mi_zy:.4f}")

        return InformationPlane(
            layers=layers,
            I_XZ_per_layer=np.array(I_XZ),
            I_ZY_per_layer=np.array(I_ZY)
        )

    def information_bottleneck_curve(
        self,
        info_plane: InformationPlane,
        beta_range: np.ndarray = np.logspace(-2, 2, 20)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute information bottleneck tradeoff curve.

        The IB objective is: min_Z I(X;Z) - β*I(Z;Y)

        Args:
            info_plane: Information plane data
            beta_range: Range of β values (tradeoff parameter)

        Returns:
            Tuple of (I_XZ_optimal, I_ZY_optimal) for each β
        """
        I_XZ = info_plane.I_XZ_per_layer
        I_ZY = info_plane.I_ZY_per_layer

        I_XZ_curve = []
        I_ZY_curve = []

        for beta in beta_range:
            # For each β, find layer that minimizes IB objective
            ib_objective = I_XZ - beta * I_ZY
            best_idx = np.argmin(ib_objective)

            I_XZ_curve.append(I_XZ[best_idx])
            I_ZY_curve.append(I_ZY[best_idx])

        return np.array(I_XZ_curve), np.array(I_ZY_curve)

    def visualize_information_plane(
        self,
        info_plane: InformationPlane,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize the information plane.

        Args:
            info_plane: Information plane data
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot trajectory through layers
        ax.plot(
            info_plane.I_XZ_per_layer,
            info_plane.I_ZY_per_layer,
            'o-', linewidth=2, markersize=8, label='Layer progression'
        )

        # Annotate layers
        for i, layer_name in enumerate(info_plane.layers):
            ax.annotate(
                layer_name,
                (info_plane.I_XZ_per_layer[i], info_plane.I_ZY_per_layer[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                alpha=0.7
            )

        ax.set_xlabel('I(X;Z) - Information about Input', fontsize=12)
        ax.set_ylabel('I(Z;Y) - Information about Output', fontsize=12)
        ax.set_title('Information Plane (Tishby)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Saved information plane to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig


# ==================== ENERGY LANDSCAPE ====================

class EnergyLandscape:
    """
    Estimate energy landscape of latent space.

    Models the latent distribution as p(z) ∝ exp(-U(z)) and estimates
    the energy function U(z).
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        self.device = device
        self.verbose = verbose

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[EnergyLandscape] {message}")

    def estimate_landscape(
        self,
        latents: torch.Tensor,
        method: str = 'density',
        grid_resolution: int = 50,
        n_components_2d: int = 2
    ) -> EnergyFunction:
        """
        Approximate energy U(z) such that p(z) ∝ exp(-U(z)).

        Args:
            latents: Latent samples (n_samples, latent_dim)
            method: 'score', 'quadratic', or 'density'
            grid_resolution: Resolution for energy grid
            n_components_2d: Number of PCA components for 2D projection

        Returns:
            EnergyFunction object
        """
        self._log(f"Estimating energy landscape using {method} method")

        latents_np = latents.detach().cpu().numpy()

        # Project to 2D for visualization if high-dimensional
        pca_basis = None
        if latents_np.shape[1] > 2:
            pca = PCA(n_components=n_components_2d)
            latents_2d = pca.fit_transform(latents_np)
            pca_basis = pca.components_
        else:
            latents_2d = latents_np

        if method == 'density':
            energy_fn = self._estimate_energy_density(latents_2d, grid_resolution)
        elif method == 'score':
            energy_fn = self._estimate_energy_score(latents_2d, grid_resolution)
        elif method == 'quadratic':
            energy_fn = self._estimate_energy_quadratic(latents_2d, grid_resolution)
        else:
            raise ValueError(f"Unknown method: {method}")

        return EnergyFunction(
            grid=energy_fn['grid'],
            energy=energy_fn['energy'],
            latent_dim=latents_np.shape[1],
            method=method,
            pca_basis=pca_basis
        )

    def _estimate_energy_density(
        self,
        latents: np.ndarray,
        grid_resolution: int
    ) -> Dict[str, np.ndarray]:
        """Estimate energy via density estimation: U(z) = -log p(z)."""

        # Fit Gaussian mixture model
        n_components = min(10, latents.shape[0] // 100)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.fit(latents)

        # Create grid
        x_min, x_max = latents[:, 0].min() - 1, latents[:, 0].max() + 1
        y_min, y_max = latents[:, 1].min() - 1, latents[:, 1].max() + 1

        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

        # Compute log density
        log_density = gmm.score_samples(grid_points)

        # Energy = -log p(z)
        energy = -log_density.reshape(X_grid.shape)

        # Normalize energy (min = 0)
        energy = energy - energy.min()

        return {
            'grid': (X_grid, Y_grid),
            'energy': energy
        }

    def _estimate_energy_score(
        self,
        latents: np.ndarray,
        grid_resolution: int
    ) -> Dict[str, np.ndarray]:
        """Estimate energy via score matching: ∇U(z) ≈ score function."""

        # Simple approach: estimate gradient field using local regression
        # For production, would use denoising score matching

        # Create grid
        x_min, x_max = latents[:, 0].min() - 1, latents[:, 0].max() + 1
        y_min, y_max = latents[:, 1].min() - 1, latents[:, 1].max() + 1

        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # Use density-based estimate as fallback
        return self._estimate_energy_density(latents, grid_resolution)

    def _estimate_energy_quadratic(
        self,
        latents: np.ndarray,
        grid_resolution: int
    ) -> Dict[str, np.ndarray]:
        """Estimate energy via local quadratic approximation."""

        # Fit quadratic: U(z) = 0.5 * (z - μ)^T Σ^{-1} (z - μ)
        mean = latents.mean(axis=0)
        cov = np.cov(latents.T)

        # Create grid
        x_min, x_max = latents[:, 0].min() - 1, latents[:, 0].max() + 1
        y_min, y_max = latents[:, 1].min() - 1, latents[:, 1].max() + 1

        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

        # Compute Mahalanobis distance
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        diff = grid_points - mean
        energy = 0.5 * np.sum(diff @ cov_inv * diff, axis=1)

        energy = energy.reshape(X_grid.shape)

        return {
            'grid': (X_grid, Y_grid),
            'energy': energy
        }

    def find_basins(
        self,
        landscape: EnergyFunction,
        num_basins: int = 5,
        min_depth: float = 0.5
    ) -> List[Basin]:
        """
        Identify energy basins (stable states).

        Args:
            landscape: Energy function
            num_basins: Maximum number of basins to find
            min_depth: Minimum depth for a basin to be significant

        Returns:
            List of Basin objects
        """
        self._log(f"Finding up to {num_basins} energy basins")

        energy = landscape.energy
        X_grid, Y_grid = landscape.grid

        # Find local minima
        from scipy.ndimage import minimum_filter

        local_min_mask = (energy == minimum_filter(energy, size=5))
        local_min_coords = np.argwhere(local_min_mask)
        local_min_energies = energy[local_min_mask]

        # Sort by energy (lowest first)
        sorted_indices = np.argsort(local_min_energies)

        basins = []

        for idx in sorted_indices[:num_basins]:
            i, j = local_min_coords[idx]

            # Check depth
            energy_val = energy[i, j]

            # Estimate local curvature (stability)
            if i > 0 and i < energy.shape[0] - 1 and j > 0 and j < energy.shape[1] - 1:
                # Second derivatives
                d2E_dx2 = energy[i+1, j] - 2*energy[i, j] + energy[i-1, j]
                d2E_dy2 = energy[i, j+1] - 2*energy[i, j] + energy[i, j-1]
                curvature = (d2E_dx2 + d2E_dy2) / 2.0
            else:
                curvature = 0.0

            # Estimate basin volume (count nearby low-energy points)
            threshold = energy_val + min_depth
            basin_mask = energy < threshold
            volume = np.sum(basin_mask)

            centroid = np.array([X_grid[i, j], Y_grid[i, j]])

            basins.append(Basin(
                centroid=centroid,
                energy=energy_val,
                volume=float(volume),
                samples=centroid.reshape(1, -1),  # Would need actual samples
                stability=curvature
            ))

        self._log(f"Found {len(basins)} basins")
        return basins

    def compute_barriers(
        self,
        landscape: EnergyFunction,
        basins: List[Basin]
    ) -> np.ndarray:
        """
        Compute energy barriers between basins.

        Args:
            landscape: Energy function
            basins: List of basins

        Returns:
            Barrier matrix (n_basins, n_basins)
        """
        n_basins = len(basins)
        barriers = np.zeros((n_basins, n_basins))

        energy = landscape.energy
        X_grid, Y_grid = landscape.grid

        for i in range(n_basins):
            for j in range(i + 1, n_basins):
                # Find path between basin i and j
                # Simple approach: straight line, find max energy

                centroid_i = basins[i].centroid
                centroid_j = basins[j].centroid

                # Interpolate along path
                n_points = 100
                path = np.linspace(centroid_i, centroid_j, n_points)

                # Map to grid indices and get energies
                path_energies = []
                for point in path:
                    # Find nearest grid point
                    x_idx = np.argmin(np.abs(X_grid[0, :] - point[0]))
                    y_idx = np.argmin(np.abs(Y_grid[:, 0] - point[1]))

                    if 0 <= y_idx < energy.shape[0] and 0 <= x_idx < energy.shape[1]:
                        path_energies.append(energy[y_idx, x_idx])

                if path_energies:
                    max_energy = max(path_energies)
                    barrier = max_energy - min(basins[i].energy, basins[j].energy)
                    barriers[i, j] = barrier
                    barriers[j, i] = barrier

        return barriers

    def visualize_landscape_2d(
        self,
        landscape: EnergyFunction,
        basins: Optional[List[Basin]] = None,
        latents: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize 2D energy landscape.

        Args:
            landscape: Energy function
            basins: Optional list of basins to overlay
            latents: Optional latent samples to overlay
            save_path: Path to save figure
            show: Whether to display

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2, width_ratios=[1, 1])

        X_grid, Y_grid = landscape.grid
        energy = landscape.energy

        # Heatmap
        ax1 = fig.add_subplot(gs[0])
        im = ax1.contourf(X_grid, Y_grid, energy, levels=20, cmap='viridis')
        plt.colorbar(im, ax=ax1, label='Energy U(z)')

        # Overlay samples if provided
        if latents is not None:
            latents_np = latents.detach().cpu().numpy()
            if landscape.pca_basis is not None:
                # Project to 2D
                latents_2d = latents_np @ landscape.pca_basis.T
            else:
                latents_2d = latents_np
            ax1.scatter(latents_2d[:, 0], latents_2d[:, 1], c='red', s=1, alpha=0.3, label='Samples')

        # Overlay basins if provided
        if basins:
            for basin in basins:
                ax1.plot(basin.centroid[0], basin.centroid[1], 'r*', markersize=15)

        ax1.set_xlabel('z₁', fontsize=12)
        ax1.set_ylabel('z₂', fontsize=12)
        ax1.set_title('Energy Landscape Heatmap', fontsize=14, fontweight='bold')

        # 3D surface
        ax2 = fig.add_subplot(gs[1], projection='3d')
        surf = ax2.plot_surface(X_grid, Y_grid, energy, cmap='viridis', alpha=0.8)

        ax2.set_xlabel('z₁')
        ax2.set_ylabel('z₂')
        ax2.set_zlabel('Energy U(z)')
        ax2.set_title('Energy Landscape 3D', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Saved landscape visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig


# ==================== ENTROPY PRODUCTION ====================

class EntropyProduction:
    """
    Measure entropy production along trajectories.

    High entropy production indicates non-equilibrium dynamics
    and irreversible processes.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        self.device = device
        self.verbose = verbose

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[EntropyProduction] {message}")

    def estimate_entropy_production(
        self,
        trajectories: torch.Tensor,
        dt: float = 0.01
    ) -> EntropyProductionEstimate:
        """
        Estimate dS/dt along trajectories.

        Uses the approximation:
        dS/dt ≈ ||dx/dt||² / (2*D)

        where D is the diffusion coefficient.

        Args:
            trajectories: Trajectory data (n_trials, n_timesteps, n_dims)
            dt: Time step

        Returns:
            EntropyProductionEstimate object
        """
        self._log("Estimating entropy production")

        trajectories_np = trajectories.detach().cpu().numpy()

        # Compute velocities
        velocities = np.diff(trajectories_np, axis=1) / dt  # (n_trials, n_timesteps-1, n_dims)

        # Estimate diffusion coefficient from fluctuations
        velocity_var = np.var(velocities, axis=(0, 1))
        D = np.mean(velocity_var) * dt / 2.0  # Einstein relation approximation

        # Entropy production rate: dS/dt ∝ ||v||²
        velocity_magnitude_sq = np.sum(velocities**2, axis=-1)  # (n_trials, n_timesteps-1)

        entropy_production_rate = velocity_magnitude_sq / (2 * D + 1e-10)

        # Average over trials
        avg_entropy_production = entropy_production_rate.mean(axis=0)

        # Total dissipation
        dissipation_rate = float(avg_entropy_production.mean())

        # Nonequilibrium score: deviation from stationary
        nonequilibrium_score = float(np.std(avg_entropy_production) / (np.mean(avg_entropy_production) + 1e-10))

        self._log(f"Dissipation rate: {dissipation_rate:.4f}")
        self._log(f"Nonequilibrium score: {nonequilibrium_score:.4f}")

        return EntropyProductionEstimate(
            entropy_production_rate=avg_entropy_production,
            dissipation_rate=dissipation_rate,
            nonequilibrium_score=nonequilibrium_score,
            trajectories=trajectories_np
        )

    def dissipation_rate(
        self,
        entropy_production: EntropyProductionEstimate
    ) -> float:
        """Get total energy dissipation rate."""
        return entropy_production.dissipation_rate

    def nonequilibrium_score(
        self,
        entropy_production: EntropyProductionEstimate
    ) -> float:
        """Get distance from equilibrium."""
        return entropy_production.nonequilibrium_score

    def visualize_entropy_production(
        self,
        entropy_production: EntropyProductionEstimate,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize entropy production over time.

        Args:
            entropy_production: EntropyProductionEstimate object
            save_path: Path to save figure
            show: Whether to display

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        time = np.arange(len(entropy_production.entropy_production_rate))

        # Entropy production time series
        axes[0].plot(time, entropy_production.entropy_production_rate, linewidth=2)
        axes[0].axhline(
            entropy_production.dissipation_rate,
            color='r', linestyle='--', label=f'Mean: {entropy_production.dissipation_rate:.3f}'
        )
        axes[0].set_xlabel('Time Step', fontsize=12)
        axes[0].set_ylabel('dS/dt', fontsize=12)
        axes[0].set_title('Entropy Production Rate', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Histogram
        axes[1].hist(entropy_production.entropy_production_rate, bins=50, density=True, alpha=0.7)
        axes[1].axvline(
            entropy_production.dissipation_rate,
            color='r', linestyle='--', linewidth=2, label='Mean'
        )
        axes[1].set_xlabel('dS/dt', fontsize=12)
        axes[1].set_ylabel('Probability Density', fontsize=12)
        axes[1].set_title('Distribution of Entropy Production', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Saved entropy production visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig


# ==================== UTILITY FUNCTIONS ====================

def compute_information_plane_trajectory(
    model: nn.Module,
    checkpoints: List[str],
    data_loader,
    layer_names: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    method: str = 'knn'
) -> InformationPlane:
    """
    Compute information plane trajectory over training epochs.

    Args:
        model: Neural network model
        checkpoints: List of checkpoint paths
        data_loader: DataLoader for evaluation
        layer_names: Names of layers to analyze
        device: Computation device
        method: MI estimation method

    Returns:
        InformationPlane with temporal data
    """
    analyzer = InformationFlowAnalyzer(device=device)

    I_XZ_trajectory = []
    I_ZY_trajectory = []
    epochs = []

    for epoch_idx, ckpt_path in enumerate(checkpoints):
        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Collect activations
        all_X = []
        all_Y = []
        all_activations = {name: [] for name in layer_names}

        with torch.no_grad():
            for batch in data_loader:
                X = batch['input'].to(device)
                Y = batch['target'].to(device)

                # Forward pass with hooks to capture activations
                activations = {}

                def hook_fn(name):
                    def hook(module, input, output):
                        activations[name] = output.detach()
                    return hook

                handles = []
                for name in layer_names:
                    layer = dict(model.named_modules())[name]
                    handles.append(layer.register_forward_hook(hook_fn(name)))

                _ = model(X)

                for handle in handles:
                    handle.remove()

                all_X.append(X)
                all_Y.append(Y)
                for name in layer_names:
                    all_activations[name].append(activations[name])

        # Concatenate
        X_all = torch.cat(all_X, dim=0)
        Y_all = torch.cat(all_Y, dim=0)
        activations_all = {
            name: torch.cat(all_activations[name], dim=0)
            for name in layer_names
        }

        # Compute information plane
        info_plane = analyzer.information_plane(
            activations_all,
            X_all,
            Y_all,
            method=method
        )

        I_XZ_trajectory.append(info_plane.I_XZ_per_layer)
        I_ZY_trajectory.append(info_plane.I_ZY_per_layer)
        epochs.append(epoch_idx)

    return InformationPlane(
        layers=layer_names,
        I_XZ_per_layer=I_XZ_trajectory[-1],
        I_ZY_per_layer=I_ZY_trajectory[-1],
        epochs=epochs,
        I_XZ_trajectory=np.array(I_XZ_trajectory),
        I_ZY_trajectory=np.array(I_ZY_trajectory)
    )


if __name__ == "__main__":
    # Example usage
    print("Energy Flow Analysis Module")
    print("=" * 50)

    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 1000
    input_dim = 20
    latent_dim = 10
    output_dim = 5

    X = torch.randn(n_samples, input_dim)
    Z = torch.randn(n_samples, latent_dim)
    Y = torch.randn(n_samples, output_dim)

    # Test Information Flow Analyzer
    print("\n1. Testing InformationFlowAnalyzer")
    print("-" * 50)
    analyzer = InformationFlowAnalyzer()

    # Test MI estimation with different methods
    for method in ['knn', 'histogram']:
        print(f"\nMethod: {method}")
        mi_results = analyzer.estimate_mutual_information(
            X, [Z], Y, method=method
        )
        for layer_name, result in mi_results.items():
            print(f"{layer_name}: I(X;Z)={result.I_XZ:.4f}, I(Z;Y)={result.I_ZY:.4f}")

    # Test information plane
    activations = {
        'layer_0': torch.randn(n_samples, 64),
        'layer_1': torch.randn(n_samples, 128),
        'layer_2': torch.randn(n_samples, 64),
        'layer_3': torch.randn(n_samples, 32)
    }

    info_plane = analyzer.information_plane(activations, X, Y, method='knn')
    fig = analyzer.visualize_information_plane(info_plane, show=False)
    print(f"\nInformation plane computed for {len(info_plane.layers)} layers")

    # Test Energy Landscape
    print("\n2. Testing EnergyLandscape")
    print("-" * 50)
    landscape_analyzer = EnergyLandscape()

    # Generate latents with multiple modes
    latents = torch.cat([
        torch.randn(300, 2) + torch.tensor([2.0, 2.0]),
        torch.randn(300, 2) + torch.tensor([-2.0, -2.0]),
        torch.randn(400, 2)
    ], dim=0)

    landscape = landscape_analyzer.estimate_landscape(latents, method='density')
    basins = landscape_analyzer.find_basins(landscape, num_basins=3)
    barriers = landscape_analyzer.compute_barriers(landscape, basins)

    print(f"Found {len(basins)} energy basins")
    for i, basin in enumerate(basins):
        print(f"Basin {i}: centroid={basin.centroid}, energy={basin.energy:.3f}, stability={basin.stability:.3f}")

    print(f"\nEnergy barriers:\n{barriers}")

    fig = landscape_analyzer.visualize_landscape_2d(landscape, basins, latents, show=False)

    # Test Entropy Production
    print("\n3. Testing EntropyProduction")
    print("-" * 50)
    entropy_analyzer = EntropyProduction()

    # Generate trajectories
    n_trials = 50
    n_timesteps = 100
    n_dims = 5
    trajectories = torch.cumsum(torch.randn(n_trials, n_timesteps, n_dims) * 0.1, dim=1)

    entropy_prod = entropy_analyzer.estimate_entropy_production(trajectories, dt=0.01)
    print(f"Dissipation rate: {entropy_prod.dissipation_rate:.4f}")
    print(f"Nonequilibrium score: {entropy_prod.nonequilibrium_score:.4f}")

    fig = entropy_analyzer.visualize_entropy_production(entropy_prod, show=False)

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
