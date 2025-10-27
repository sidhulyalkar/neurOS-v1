"""
Dynamical Systems Analysis for NeuroFMX Latent Representations.

This module provides comprehensive tools for analyzing the dynamics of neural
trajectories in latent space, including:
- Koopman operator analysis
- Lyapunov exponents
- Manifold analysis
- Phase space analysis
- Controllability/observability
- Stability analysis

Author: NeuroFMX Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.linalg import svd, eigh, solve_continuous_lyapunov
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings


class DynamicsAnalyzer:
    """
    Comprehensive dynamical systems analysis for neural trajectories.

    Analyzes latent space dynamics using techniques from nonlinear dynamics,
    control theory, and differential geometry.

    Args:
        dt: Time step between consecutive time points
        device: Torch device for computations
        verbose: Whether to print progress information
    """

    def __init__(
        self,
        dt: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        self.dt = dt
        self.device = device
        self.verbose = verbose

        # Cache for computed quantities
        self._cache = {}

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[DynamicsAnalyzer] {message}")

    def _compute_velocities(
        self,
        trajectories: torch.Tensor,
        method: str = "finite_diff"
    ) -> torch.Tensor:
        """
        Compute velocities from trajectories.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            method: "finite_diff" or "savitzky_golay"

        Returns:
            velocities: (n_trials, n_timesteps-1, n_dims)
        """
        if method == "finite_diff":
            # Central difference for interior points
            velocities = (trajectories[:, 1:, :] - trajectories[:, :-1, :]) / self.dt
        elif method == "savitzky_golay":
            # Smoothed derivative (simple 3-point smoother)
            velocities = torch.zeros_like(trajectories[:, :-1, :])
            velocities[:, 1:-1, :] = (
                trajectories[:, 2:, :] - trajectories[:, :-2, :]
            ) / (2 * self.dt)
            velocities[:, 0, :] = velocities[:, 1, :]
            velocities[:, -1, :] = velocities[:, -2, :]
        else:
            raise ValueError(f"Unknown method: {method}")

        return velocities

    # ==================== KOOPMAN OPERATOR ANALYSIS ====================

    def estimate_koopman_operator(
        self,
        trajectories: torch.Tensor,
        n_delays: int = 1,
        rank: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate Koopman operator via Dynamic Mode Decomposition (DMD).

        The Koopman operator is a linear operator that describes nonlinear
        dynamics in a lifted observable space.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            n_delays: Number of delay embeddings to include
            rank: Rank for truncated SVD (None = full rank)

        Returns:
            Dictionary containing:
                - koopman_matrix: Koopman operator matrix
                - eigenvalues: Koopman eigenvalues
                - eigenvectors: Koopman eigenvectors (modes)
                - growth_rates: Real part of log(eigenvalues) / dt
                - frequencies: Imaginary part of log(eigenvalues) / (2*pi*dt)
                - reconstruction: Reconstructed trajectories
                - reconstruction_error: MSE reconstruction error
        """
        self._log("Estimating Koopman operator via DMD...")

        n_trials, n_timesteps, n_dims = trajectories.shape

        # Create delay embedding if requested
        if n_delays > 1:
            traj_embedded = self._delay_embed(trajectories, n_delays)
        else:
            traj_embedded = trajectories

        # Reshape: (n_trials, n_timesteps, n_dims) -> (n_features, n_snapshots)
        X = traj_embedded[:, :-1, :].reshape(-1, traj_embedded.shape[-1]).T
        Y = traj_embedded[:, 1:, :].reshape(-1, traj_embedded.shape[-1]).T

        # Move to CPU for SVD (more stable)
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()

        # SVD of X
        U, S, Vh = svd(X_np, full_matrices=False)

        # Truncate if rank specified
        if rank is not None:
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]

        # Compute Koopman operator: K = Y * V * S^-1 * U^T
        S_inv = np.diag(1.0 / S)
        K = Y_np @ Vh.T @ S_inv @ U.T

        # Eigendecomposition of K
        eigenvalues, eigenvectors = np.linalg.eig(K)

        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Compute growth rates and frequencies
        log_eigenvalues = np.log(eigenvalues + 1e-10)
        growth_rates = np.real(log_eigenvalues) / self.dt
        frequencies = np.imag(log_eigenvalues) / (2 * np.pi * self.dt)

        # Reconstruction
        X_recon = K @ X_np
        reconstruction_error = np.mean((Y_np - X_recon) ** 2)

        # Convert back to torch
        results = {
            "koopman_matrix": torch.from_numpy(K).to(self.device),
            "eigenvalues": torch.from_numpy(eigenvalues).to(self.device),
            "eigenvectors": torch.from_numpy(eigenvectors).to(self.device),
            "growth_rates": torch.from_numpy(growth_rates).to(self.device),
            "frequencies": torch.from_numpy(frequencies).to(self.device),
            "reconstruction_error": torch.tensor(reconstruction_error).to(self.device)
        }

        self._log(f"Koopman operator estimated. Reconstruction error: {reconstruction_error:.6f}")

        return results

    def identify_dominant_modes(
        self,
        koopman_results: Dict[str, torch.Tensor],
        n_modes: int = 5,
        min_growth_rate: float = -10.0
    ) -> Dict[str, torch.Tensor]:
        """
        Identify dominant Koopman modes.

        Args:
            koopman_results: Output from estimate_koopman_operator
            n_modes: Number of top modes to return
            min_growth_rate: Minimum growth rate to consider

        Returns:
            Dictionary with dominant modes information
        """
        eigenvalues = koopman_results["eigenvalues"]
        eigenvectors = koopman_results["eigenvectors"]
        growth_rates = koopman_results["growth_rates"]
        frequencies = koopman_results["frequencies"]

        # Filter by growth rate
        mask = growth_rates.real > min_growth_rate

        # Sort by magnitude
        magnitudes = torch.abs(eigenvalues[mask])
        idx = torch.argsort(magnitudes, descending=True)[:n_modes]

        # Extract dominant modes
        dominant_eigenvalues = eigenvalues[mask][idx]
        dominant_eigenvectors = eigenvectors[:, mask][:, idx]
        dominant_growth_rates = growth_rates[mask][idx]
        dominant_frequencies = frequencies[mask][idx]

        return {
            "eigenvalues": dominant_eigenvalues,
            "eigenvectors": dominant_eigenvectors,
            "growth_rates": dominant_growth_rates,
            "frequencies": dominant_frequencies,
            "mode_indices": idx
        }

    # ==================== LYAPUNOV EXPONENTS ====================

    def compute_lyapunov_exponents(
        self,
        trajectories: torch.Tensor,
        n_exponents: Optional[int] = None,
        evolution_time: Optional[float] = None,
        n_steps: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Lyapunov exponents using orthogonalization method.

        Lyapunov exponents quantify rates of separation of infinitesimally
        close trajectories, indicating chaos when positive.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            n_exponents: Number of exponents to compute (None = all)
            evolution_time: Total time for evolution (None = auto)
            n_steps: Number of renormalization steps

        Returns:
            Dictionary containing:
                - lyapunov_exponents: Full spectrum
                - max_lyapunov_exponent: Maximum LE (MLE)
                - lyapunov_spectrum: Sorted exponents
                - chaos_detected: Boolean flag (MLE > 0)
                - divergence_rate: Average rate of trajectory divergence
        """
        self._log("Computing Lyapunov exponents...")

        n_trials, n_timesteps, n_dims = trajectories.shape

        if n_exponents is None:
            n_exponents = min(n_dims, 10)  # Limit for efficiency

        if evolution_time is None:
            evolution_time = (n_timesteps - 1) * self.dt

        dt_step = evolution_time / n_steps
        steps_per_renorm = max(1, (n_timesteps - 1) // n_steps)

        # Initialize perturbation matrix (identity)
        W = torch.eye(n_dims, n_exponents, device=self.device)

        # Storage for Lyapunov exponents
        lyap_sum = torch.zeros(n_exponents, device=self.device)

        # Compute velocities
        velocities = self._compute_velocities(trajectories)

        # Average over trials
        for trial_idx in range(n_trials):
            traj = trajectories[trial_idx]
            vel = velocities[trial_idx]

            W_trial = W.clone()

            for step in range(n_steps):
                start_idx = step * steps_per_renorm
                end_idx = min(start_idx + steps_per_renorm, len(vel))

                if start_idx >= len(vel):
                    break

                # Estimate local Jacobian
                J = self._estimate_jacobian(
                    traj[start_idx:end_idx],
                    vel[start_idx:end_idx]
                )

                # Propagate perturbations: W_new = J * W
                W_trial = J @ W_trial

                # QR decomposition for renormalization
                Q, R = torch.linalg.qr(W_trial)
                W_trial = Q

                # Accumulate logarithms of diagonal elements
                lyap_sum += torch.log(torch.abs(torch.diag(R)))

        # Normalize by time and number of trials
        lyapunov_exponents = lyap_sum / (evolution_time * n_trials)

        # Sort in descending order
        lyapunov_spectrum, _ = torch.sort(lyapunov_exponents, descending=True)

        max_lyapunov = lyapunov_spectrum[0]
        chaos_detected = max_lyapunov > 0

        # Estimate divergence rate
        divergence_rate = self._estimate_divergence_rate(trajectories)

        self._log(f"Max Lyapunov exponent: {max_lyapunov:.6f}")
        self._log(f"Chaos detected: {chaos_detected}")

        return {
            "lyapunov_exponents": lyapunov_exponents,
            "max_lyapunov_exponent": max_lyapunov,
            "lyapunov_spectrum": lyapunov_spectrum,
            "chaos_detected": chaos_detected,
            "divergence_rate": divergence_rate
        }

    def _estimate_jacobian(
        self,
        states: torch.Tensor,
        velocities: torch.Tensor,
        regularization: float = 1e-6
    ) -> torch.Tensor:
        """
        Estimate local Jacobian via linear regression.

        Args:
            states: (n_points, n_dims)
            velocities: (n_points, n_dims)
            regularization: Ridge regularization parameter

        Returns:
            jacobian: (n_dims, n_dims)
        """
        # Center the data
        states_mean = states.mean(dim=0, keepdim=True)
        states_centered = states - states_mean

        # Solve: velocities = J * states_centered
        # J = velocities^T * states * (states^T * states + lambda*I)^-1

        X = states_centered
        Y = velocities

        # Ridge regression
        XtX = X.T @ X + regularization * torch.eye(X.shape[1], device=X.device)
        XtY = X.T @ Y

        J = torch.linalg.solve(XtX, XtY).T

        return J

    def _estimate_divergence_rate(
        self,
        trajectories: torch.Tensor,
        n_neighbors: int = 5
    ) -> torch.Tensor:
        """
        Estimate average divergence rate of nearby trajectories.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            n_neighbors: Number of neighbors to consider

        Returns:
            divergence_rate: Scalar tensor
        """
        n_trials, n_timesteps, n_dims = trajectories.shape

        if n_trials < 2:
            return torch.tensor(0.0, device=self.device)

        # Sample initial and final points
        initial_points = trajectories[:, 0, :]  # (n_trials, n_dims)
        final_points = trajectories[:, -1, :]

        # Compute pairwise distances
        initial_dists = torch.cdist(initial_points, initial_points)
        final_dists = torch.cdist(final_points, final_points)

        # Find k nearest neighbors
        _, indices = torch.topk(
            initial_dists,
            k=min(n_neighbors + 1, n_trials),
            largest=False,
            dim=1
        )

        # Exclude self (first element)
        indices = indices[:, 1:]

        # Compute divergence
        divergences = []
        for i in range(n_trials):
            for j in indices[i]:
                initial_dist = initial_dists[i, j]
                final_dist = final_dists[i, j]

                if initial_dist > 1e-8:
                    divergence = torch.log(final_dist / initial_dist)
                    divergences.append(divergence)

        if len(divergences) == 0:
            return torch.tensor(0.0, device=self.device)

        total_time = (n_timesteps - 1) * self.dt
        divergence_rate = torch.stack(divergences).mean() / total_time

        return divergence_rate

    # ==================== MANIFOLD ANALYSIS ====================

    def analyze_manifold(
        self,
        trajectories: torch.Tensor,
        n_components: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze the slow manifold structure of dynamics.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            n_components: Number of PCA components for slow manifold

        Returns:
            Dictionary containing:
                - slow_manifold_components: Principal components
                - explained_variance_ratio: Variance explained by each PC
                - intrinsic_dimensionality: Effective dimensionality
                - participation_ratio: Participation ratio
                - tangent_spaces: Local tangent space bases
                - curvature_estimates: Local curvature estimates
        """
        self._log("Analyzing manifold structure...")

        n_trials, n_timesteps, n_dims = trajectories.shape

        # Flatten trajectories
        X = trajectories.reshape(-1, n_dims)  # (n_trials * n_timesteps, n_dims)

        # Center the data
        X_mean = X.mean(dim=0, keepdim=True)
        X_centered = X - X_mean

        # PCA via SVD
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        # Compute explained variance
        variance = S ** 2 / (X.shape[0] - 1)
        total_variance = variance.sum()
        explained_variance_ratio = variance / total_variance

        # Intrinsic dimensionality (number of PCs to explain 90% variance)
        cumsum_variance = torch.cumsum(explained_variance_ratio, dim=0)
        intrinsic_dim = torch.searchsorted(cumsum_variance, 0.9) + 1

        # Participation ratio
        participation_ratio = (variance.sum() ** 2) / (variance ** 2).sum()

        # Slow manifold components
        slow_components = Vh[:n_components, :]

        # Estimate local tangent spaces
        tangent_spaces = self._estimate_tangent_spaces(trajectories, k=10)

        # Estimate curvature
        curvature_estimates = self._estimate_curvature(trajectories)

        self._log(f"Intrinsic dimensionality: {intrinsic_dim.item()}")
        self._log(f"Participation ratio: {participation_ratio:.2f}")

        return {
            "slow_manifold_components": slow_components,
            "explained_variance_ratio": explained_variance_ratio,
            "intrinsic_dimensionality": intrinsic_dim,
            "participation_ratio": participation_ratio,
            "tangent_spaces": tangent_spaces,
            "curvature_estimates": curvature_estimates,
            "mean": X_mean
        }

    def _estimate_tangent_spaces(
        self,
        trajectories: torch.Tensor,
        k: int = 10
    ) -> torch.Tensor:
        """
        Estimate local tangent spaces via local PCA.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            k: Number of neighbors for local PCA

        Returns:
            tangent_bases: (n_trials, n_timesteps, n_dims, n_tangent_dims)
        """
        n_trials, n_timesteps, n_dims = trajectories.shape

        # Flatten for distance computation
        X = trajectories.reshape(-1, n_dims)

        # Sample subset for efficiency
        n_samples = min(1000, len(X))
        indices = torch.randperm(len(X))[:n_samples]
        X_sample = X[indices]

        # Compute distances
        dists = torch.cdist(X_sample, X_sample)

        # Find k nearest neighbors
        _, neighbor_indices = torch.topk(dists, k=k+1, largest=False, dim=1)
        neighbor_indices = neighbor_indices[:, 1:]  # Exclude self

        # Estimate tangent space for each sample
        tangent_bases = []
        for i in range(len(X_sample)):
            neighbors = X_sample[neighbor_indices[i]]
            neighbors_centered = neighbors - neighbors.mean(dim=0, keepdim=True)

            # Local PCA
            _, _, Vh = torch.linalg.svd(neighbors_centered, full_matrices=False)

            # Take top 3 components as tangent basis
            tangent_basis = Vh[:3, :].T  # (n_dims, 3)
            tangent_bases.append(tangent_basis)

        tangent_bases = torch.stack(tangent_bases)  # (n_samples, n_dims, 3)

        return tangent_bases

    def _estimate_curvature(
        self,
        trajectories: torch.Tensor,
        epsilon: float = 1e-3
    ) -> torch.Tensor:
        """
        Estimate local curvature of trajectories.

        Uses finite differences to estimate second derivatives.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            epsilon: Finite difference step size

        Returns:
            curvature: (n_trials, n_timesteps-2) - magnitude of curvature
        """
        # Second derivative: d²x/dt²
        first_diff = trajectories[:, 1:, :] - trajectories[:, :-1, :]
        second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]

        # Curvature magnitude
        curvature = torch.norm(second_diff, dim=-1) / (self.dt ** 2)

        return curvature

    def compute_geodesic_distance(
        self,
        trajectories: torch.Tensor,
        point_a: torch.Tensor,
        point_b: torch.Tensor,
        n_steps: int = 100
    ) -> torch.Tensor:
        """
        Compute approximate geodesic distance on the manifold.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims) - defines manifold
            point_a: (n_dims,) - start point
            point_b: (n_dims,) - end point
            n_steps: Number of steps for path integration

        Returns:
            geodesic_distance: Scalar tensor
        """
        # Simple approximation: straight line path with correction
        # For true geodesic, would need to solve geodesic equation

        # Flatten trajectories
        X = trajectories.reshape(-1, trajectories.shape[-1])

        # Create path from a to b
        t = torch.linspace(0, 1, n_steps, device=self.device)
        path = point_a[None, :] + t[:, None] * (point_b - point_a)[None, :]

        # Find nearest manifold points
        dists = torch.cdist(path, X)
        _, nearest_indices = torch.min(dists, dim=1)
        manifold_path = X[nearest_indices]

        # Integrate path length
        path_diffs = manifold_path[1:] - manifold_path[:-1]
        path_lengths = torch.norm(path_diffs, dim=1)
        geodesic_distance = path_lengths.sum()

        return geodesic_distance

    # ==================== PHASE SPACE ANALYSIS ====================

    def analyze_phase_space(
        self,
        trajectories: torch.Tensor,
        dimensions: Tuple[int, int] = (0, 1)
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Analyze phase space structure.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            dimensions: Which dimensions to analyze (for visualization)

        Returns:
            Dictionary containing:
                - fixed_points: Detected fixed points
                - fixed_point_stability: Stability of each fixed point
                - limit_cycles: Detected limit cycles
                - attractor_basins: Estimated attractor basins
        """
        self._log("Analyzing phase space...")

        # Detect fixed points
        fixed_points, stability = self._detect_fixed_points(trajectories)

        # Detect limit cycles
        limit_cycles = self._detect_limit_cycles(trajectories)

        # Estimate attractor basins (simplified)
        attractor_basins = self._estimate_attractor_basins(
            trajectories,
            fixed_points
        )

        self._log(f"Found {len(fixed_points)} fixed points")
        self._log(f"Found {len(limit_cycles)} potential limit cycles")

        return {
            "fixed_points": fixed_points,
            "fixed_point_stability": stability,
            "limit_cycles": limit_cycles,
            "attractor_basins": attractor_basins
        }

    def _detect_fixed_points(
        self,
        trajectories: torch.Tensor,
        velocity_threshold: float = 1e-3,
        min_duration: int = 10
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Detect fixed points where velocity is near zero.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            velocity_threshold: Maximum velocity magnitude for fixed point
            min_duration: Minimum time steps near fixed point

        Returns:
            fixed_points: List of fixed point locations
            stability: List of stability classifications
        """
        velocities = self._compute_velocities(trajectories)
        velocity_mag = torch.norm(velocities, dim=-1)

        fixed_points = []
        stability_types = []

        for trial_idx in range(len(trajectories)):
            vel_mag = velocity_mag[trial_idx]
            traj = trajectories[trial_idx]

            # Find regions with low velocity
            low_vel = vel_mag < velocity_threshold

            # Find contiguous regions
            starts = []
            ends = []
            in_region = False

            for t in range(len(low_vel)):
                if low_vel[t] and not in_region:
                    starts.append(t)
                    in_region = True
                elif not low_vel[t] and in_region:
                    ends.append(t)
                    in_region = False

            if in_region:
                ends.append(len(low_vel))

            # Extract fixed points from long enough regions
            for start, end in zip(starts, ends):
                if end - start >= min_duration:
                    fp = traj[start:end].mean(dim=0)
                    fixed_points.append(fp)

                    # Estimate stability
                    stability = self._classify_fixed_point_stability(
                        traj[start:end],
                        velocities[trial_idx, start:end]
                    )
                    stability_types.append(stability)

        return fixed_points, stability_types

    def _classify_fixed_point_stability(
        self,
        states: torch.Tensor,
        velocities: torch.Tensor
    ) -> str:
        """
        Classify fixed point stability via local Jacobian eigenvalues.

        Args:
            states: (n_points, n_dims)
            velocities: (n_points, n_dims)

        Returns:
            stability: "stable", "unstable", or "saddle"
        """
        J = self._estimate_jacobian(states, velocities)
        eigenvalues = torch.linalg.eigvals(J)

        real_parts = eigenvalues.real

        if torch.all(real_parts < 0):
            return "stable"
        elif torch.all(real_parts > 0):
            return "unstable"
        else:
            return "saddle"

    def _detect_limit_cycles(
        self,
        trajectories: torch.Tensor,
        recurrence_threshold: float = 0.1
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Detect potential limit cycles via recurrence analysis.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            recurrence_threshold: Distance threshold for recurrence

        Returns:
            limit_cycles: List of dictionaries with cycle information
        """
        limit_cycles = []

        for trial_idx in range(len(trajectories)):
            traj = trajectories[trial_idx]

            # Compute recurrence matrix
            dists = torch.cdist(traj, traj)

            # Find near-recurrences
            recurrent = dists < recurrence_threshold

            # Look for periodic patterns
            for t in range(len(traj) // 2):
                if t == 0:
                    continue

                # Check if trajectory returns to near starting point
                if recurrent[0, t]:
                    # Potential cycle of period t
                    cycle_segment = traj[:t]

                    limit_cycles.append({
                        "period": t,
                        "trajectory": cycle_segment,
                        "trial": trial_idx
                    })
                    break  # Only take first cycle per trial

        return limit_cycles

    def _estimate_attractor_basins(
        self,
        trajectories: torch.Tensor,
        fixed_points: List[torch.Tensor],
        n_grid_points: int = 20
    ) -> torch.Tensor:
        """
        Estimate basins of attraction (simplified 2D version).

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            fixed_points: List of fixed point locations
            n_grid_points: Grid resolution for basin estimation

        Returns:
            basin_map: (n_grid_points, n_grid_points) - basin assignments
        """
        if len(fixed_points) == 0:
            return torch.zeros(n_grid_points, n_grid_points, device=self.device)

        # Flatten trajectories
        X = trajectories.reshape(-1, trajectories.shape[-1])

        # Get bounds
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()

        # Create grid
        x = torch.linspace(x_min, x_max, n_grid_points, device=self.device)
        y = torch.linspace(y_min, y_max, n_grid_points, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        # For each grid point, find nearest fixed point
        basin_map = torch.zeros(len(grid_points), device=self.device)

        fp_tensor = torch.stack(fixed_points)[:, :2]  # Take first 2 dims

        for i, point in enumerate(grid_points):
            dists = torch.norm(fp_tensor - point[None, :], dim=1)
            basin_map[i] = torch.argmin(dists)

        basin_map = basin_map.reshape(n_grid_points, n_grid_points)

        return basin_map

    def compute_poincare_section(
        self,
        trajectories: torch.Tensor,
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Poincaré section (intersection with hyperplane).

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            plane_normal: (n_dims,) - normal vector to section plane
            plane_point: (n_dims,) - point on section plane

        Returns:
            intersections: (n_intersections, n_dims) - points where trajectory
                          crosses the Poincaré section
        """
        plane_normal = plane_normal / torch.norm(plane_normal)

        intersections = []

        for trial_idx in range(len(trajectories)):
            traj = trajectories[trial_idx]

            # Compute signed distance from plane
            distances = (traj - plane_point[None, :]) @ plane_normal

            # Find zero crossings
            sign_changes = distances[:-1] * distances[1:] < 0

            for t in torch.where(sign_changes)[0]:
                # Linear interpolation to find intersection
                d1, d2 = distances[t], distances[t+1]
                alpha = -d1 / (d2 - d1)

                intersection = traj[t] + alpha * (traj[t+1] - traj[t])
                intersections.append(intersection)

        if len(intersections) == 0:
            return torch.empty(0, trajectories.shape[-1], device=self.device)

        return torch.stack(intersections)

    # ==================== CONTROLLABILITY & OBSERVABILITY ====================

    def compute_controllability(
        self,
        trajectories: torch.Tensor,
        control_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute controllability metrics.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            control_inputs: (n_trials, n_timesteps, n_controls) - if None,
                          assumes identity control matrix

        Returns:
            Dictionary containing:
                - controllability_gramian: Controllability Gramian matrix
                - controllability_index: Trace of Gramian
                - min_energy_controls: Minimum energy control directions
                - modal_controllability: Per-mode controllability
        """
        self._log("Computing controllability metrics...")

        n_trials, n_timesteps, n_dims = trajectories.shape

        # Estimate system matrices A and B
        velocities = self._compute_velocities(trajectories)

        # Average over trials
        X = trajectories[:, :-1, :].reshape(-1, n_dims)
        dX = velocities.reshape(-1, n_dims)

        if control_inputs is not None:
            U = control_inputs[:, :-1, :].reshape(-1, control_inputs.shape[-1])
        else:
            # Assume identity control
            U = torch.eye(n_dims, device=self.device).repeat(len(X), 1)

        # Estimate A and B via least squares: dX = A*X + B*U
        XU = torch.cat([X, U], dim=1)
        AB = torch.linalg.lstsq(XU, dX).solution

        A = AB[:n_dims, :].T
        B = AB[n_dims:, :].T

        # Compute controllability Gramian
        # W_c = integral_0^inf exp(A*t) * B * B^T * exp(A^T*t) dt
        # For discrete time: solve Lyapunov equation
        # W_c = A * W_c * A^T + B * B^T

        A_np = A.cpu().numpy()
        B_np = B.cpu().numpy()
        BBT = B_np @ B_np.T

        try:
            W_c_np = solve_continuous_lyapunov(A_np, -BBT)
            W_c = torch.from_numpy(W_c_np).to(self.device)
        except:
            # If Lyapunov equation fails, use empirical Gramian
            W_c = BBT @ torch.eye(n_dims, device=self.device)
            W_c = torch.from_numpy(W_c).to(self.device)

        # Controllability index (trace of Gramian)
        controllability_index = torch.trace(W_c)

        # Minimum energy control directions (eigenvectors of W_c)
        eigenvalues, eigenvectors = torch.linalg.eigh(W_c)

        # Modal controllability
        modal_controllability = eigenvalues

        self._log(f"Controllability index: {controllability_index:.6f}")

        return {
            "controllability_gramian": W_c,
            "controllability_index": controllability_index,
            "min_energy_controls": eigenvectors,
            "modal_controllability": modal_controllability,
            "A_matrix": A,
            "B_matrix": B
        }

    def compute_observability(
        self,
        trajectories: torch.Tensor,
        observations: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute observability metrics.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            observations: (n_trials, n_timesteps, n_obs) - if None,
                         assumes identity observation matrix

        Returns:
            Dictionary containing:
                - observability_gramian: Observability Gramian matrix
                - observability_index: Trace of Gramian
                - observable_modes: Most observable modes
                - modal_observability: Per-mode observability
        """
        self._log("Computing observability metrics...")

        n_trials, n_timesteps, n_dims = trajectories.shape

        # Estimate system matrices A and C
        velocities = self._compute_velocities(trajectories)

        X = trajectories[:, :-1, :].reshape(-1, n_dims)
        dX = velocities.reshape(-1, n_dims)

        # Estimate A
        A = torch.linalg.lstsq(X, dX).solution.T

        if observations is not None:
            Y = observations[:, :-1, :].reshape(-1, observations.shape[-1])
            # Estimate C: Y = C*X
            C = torch.linalg.lstsq(X, Y).solution.T
        else:
            # Identity observation
            C = torch.eye(n_dims, device=self.device)

        # Compute observability Gramian
        # W_o = integral_0^inf exp(A^T*t) * C^T * C * exp(A*t) dt

        A_np = A.cpu().numpy()
        C_np = C.cpu().numpy()
        CTC = C_np.T @ C_np

        try:
            W_o_np = solve_continuous_lyapunov(A_np.T, -CTC)
            W_o = torch.from_numpy(W_o_np).to(self.device)
        except:
            # If Lyapunov equation fails, use empirical Gramian
            W_o = CTC @ torch.eye(n_dims, device=self.device)
            W_o = torch.from_numpy(W_o).to(self.device)

        # Observability index
        observability_index = torch.trace(W_o)

        # Observable modes
        eigenvalues, eigenvectors = torch.linalg.eigh(W_o)

        # Sort by observability
        idx = torch.argsort(eigenvalues, descending=True)
        modal_observability = eigenvalues[idx]
        observable_modes = eigenvectors[:, idx]

        self._log(f"Observability index: {observability_index:.6f}")

        return {
            "observability_gramian": W_o,
            "observability_index": observability_index,
            "observable_modes": observable_modes,
            "modal_observability": modal_observability,
            "A_matrix": A,
            "C_matrix": C
        }

    # ==================== STABILITY ANALYSIS ====================

    def analyze_stability(
        self,
        trajectories: torch.Tensor,
        perturbation_magnitude: float = 0.1,
        n_perturbations: int = 10
    ) -> Dict[str, Union[torch.Tensor, bool]]:
        """
        Analyze stability of dynamics.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            perturbation_magnitude: Magnitude of perturbations to test
            n_perturbations: Number of perturbations per trial

        Returns:
            Dictionary containing:
                - eigenvalue_stability: Whether all eigenvalues have negative real parts
                - max_eigenvalue_real: Maximum real part of eigenvalues
                - perturbation_growth: Growth of perturbations over time
                - stability_margin: Margin of stability
                - is_stable: Overall stability verdict
        """
        self._log("Analyzing stability...")

        # Estimate system matrix A
        velocities = self._compute_velocities(trajectories)
        X = trajectories[:, :-1, :].reshape(-1, trajectories.shape[-1])
        dX = velocities.reshape(-1, velocities.shape[-1])

        A = torch.linalg.lstsq(X, dX).solution.T

        # Eigenvalue analysis
        eigenvalues = torch.linalg.eigvals(A)
        real_parts = eigenvalues.real
        max_real = real_parts.max()

        eigenvalue_stable = max_real < 0
        stability_margin = -max_real if eigenvalue_stable else torch.tensor(0.0)

        # Test with perturbations
        perturbation_growth = self._test_perturbation_robustness(
            trajectories,
            perturbation_magnitude,
            n_perturbations
        )

        # Overall verdict
        is_stable = eigenvalue_stable and (perturbation_growth.mean() < 1.0)

        self._log(f"Maximum eigenvalue real part: {max_real:.6f}")
        self._log(f"System is {'stable' if is_stable else 'unstable'}")

        return {
            "eigenvalue_stability": eigenvalue_stable,
            "max_eigenvalue_real": max_real,
            "perturbation_growth": perturbation_growth,
            "stability_margin": stability_margin,
            "is_stable": is_stable,
            "eigenvalues": eigenvalues,
            "A_matrix": A
        }

    def _test_perturbation_robustness(
        self,
        trajectories: torch.Tensor,
        perturbation_magnitude: float,
        n_perturbations: int
    ) -> torch.Tensor:
        """
        Test robustness to perturbations.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            perturbation_magnitude: Magnitude of perturbations
            n_perturbations: Number of random perturbations to test

        Returns:
            growth_factors: (n_perturbations,) - ratio of final to initial perturbation
        """
        n_trials, n_timesteps, n_dims = trajectories.shape

        # Estimate Jacobian at various points
        velocities = self._compute_velocities(trajectories)

        growth_factors = []

        for _ in range(n_perturbations):
            # Random initial perturbation
            perturbation = torch.randn(n_dims, device=self.device)
            perturbation = perturbation / torch.norm(perturbation) * perturbation_magnitude

            initial_norm = torch.norm(perturbation)

            # Propagate perturbation through dynamics
            # Sample random trial and timepoint
            trial_idx = torch.randint(0, n_trials, (1,)).item()
            start_t = torch.randint(0, n_timesteps - 10, (1,)).item()

            states = trajectories[trial_idx, start_t:start_t+10]
            vels = velocities[trial_idx, start_t:start_t+10]

            J = self._estimate_jacobian(states, vels)

            # Propagate: perturbation_new = J * perturbation
            final_perturbation = J @ perturbation
            final_norm = torch.norm(final_perturbation)

            growth_factor = final_norm / initial_norm
            growth_factors.append(growth_factor)

        return torch.stack(growth_factors)

    # ==================== VISUALIZATION ====================

    def visualize_dynamics(
        self,
        trajectories: torch.Tensor,
        koopman_results: Optional[Dict] = None,
        manifold_results: Optional[Dict] = None,
        phase_results: Optional[Dict] = None,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive visualization of dynamics analysis.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            koopman_results: Results from estimate_koopman_operator
            manifold_results: Results from analyze_manifold
            phase_results: Results from analyze_phase_space
            save_path: Path to save figure
        """
        n_plots = 2  # Base plots
        if koopman_results is not None:
            n_plots += 2
        if manifold_results is not None:
            n_plots += 1
        if phase_results is not None:
            n_plots += 1

        fig = plt.figure(figsize=(15, 4 * ((n_plots + 1) // 2)))
        gs = GridSpec(((n_plots + 1) // 2), 2, figure=fig)

        plot_idx = 0

        # 1. Phase portrait (2D projection)
        ax1 = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
        self._plot_phase_portrait(ax1, trajectories, phase_results)
        plot_idx += 1

        # 2. Trajectory time series
        ax2 = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
        self._plot_trajectories(ax2, trajectories)
        plot_idx += 1

        # 3. Koopman eigenvalues
        if koopman_results is not None:
            ax3 = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            self._plot_koopman_eigenvalues(ax3, koopman_results)
            plot_idx += 1

            # 4. Koopman modes
            ax4 = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            self._plot_koopman_modes(ax4, koopman_results)
            plot_idx += 1

        # 5. Manifold structure
        if manifold_results is not None:
            ax5 = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            self._plot_manifold_variance(ax5, manifold_results)
            plot_idx += 1

        # 6. Attractor basins
        if phase_results is not None and "attractor_basins" in phase_results:
            ax6 = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            self._plot_attractor_basins(ax6, phase_results)
            plot_idx += 1

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self._log(f"Visualization saved to {save_path}")

        plt.close()

    def _plot_phase_portrait(
        self,
        ax: plt.Axes,
        trajectories: torch.Tensor,
        phase_results: Optional[Dict]
    ):
        """Plot 2D phase portrait."""
        traj_np = trajectories.cpu().numpy()

        # Plot trajectories
        for trial in range(min(10, len(trajectories))):
            ax.plot(
                traj_np[trial, :, 0],
                traj_np[trial, :, 1],
                alpha=0.3,
                linewidth=0.5
            )

        # Plot fixed points if available
        if phase_results and "fixed_points" in phase_results:
            fps = phase_results["fixed_points"]
            stability = phase_results["fixed_point_stability"]

            for fp, stab in zip(fps, stability):
                fp_np = fp.cpu().numpy()
                color = "green" if stab == "stable" else "red" if stab == "unstable" else "orange"
                ax.scatter(fp_np[0], fp_np[1], c=color, s=100, marker="*",
                          edgecolors="black", zorder=10)

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("Phase Portrait")
        ax.grid(True, alpha=0.3)

    def _plot_trajectories(self, ax: plt.Axes, trajectories: torch.Tensor):
        """Plot trajectory time series."""
        traj_np = trajectories.cpu().numpy()

        # Plot first few dimensions
        n_dims_plot = min(3, trajectories.shape[-1])
        time = np.arange(trajectories.shape[1]) * self.dt

        for dim in range(n_dims_plot):
            # Plot mean trajectory
            mean_traj = traj_np[:, :, dim].mean(axis=0)
            std_traj = traj_np[:, :, dim].std(axis=0)

            ax.plot(time, mean_traj, label=f"Dim {dim+1}")
            ax.fill_between(
                time,
                mean_traj - std_traj,
                mean_traj + std_traj,
                alpha=0.2
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("State")
        ax.set_title("Trajectory Time Series")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_koopman_eigenvalues(
        self,
        ax: plt.Axes,
        koopman_results: Dict
    ):
        """Plot Koopman eigenvalues in complex plane."""
        eigenvalues = koopman_results["eigenvalues"].cpu().numpy()

        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label="Unit circle")

        # Plot eigenvalues
        ax.scatter(
            eigenvalues.real,
            eigenvalues.imag,
            c=np.abs(eigenvalues),
            cmap="viridis",
            s=50,
            alpha=0.7
        )

        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title("Koopman Eigenvalues")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_koopman_modes(
        self,
        ax: plt.Axes,
        koopman_results: Dict
    ):
        """Plot Koopman mode growth rates and frequencies."""
        growth_rates = koopman_results["growth_rates"].real.cpu().numpy()
        frequencies = koopman_results["frequencies"].real.cpu().numpy()

        n_modes = min(20, len(growth_rates))

        ax.scatter(
            frequencies[:n_modes],
            growth_rates[:n_modes],
            s=100,
            alpha=0.7,
            c=np.arange(n_modes),
            cmap="viridis"
        )

        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label="Stability boundary")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Growth Rate (1/s)")
        ax.set_title("Koopman Mode Spectrum")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_manifold_variance(
        self,
        ax: plt.Axes,
        manifold_results: Dict
    ):
        """Plot explained variance ratio."""
        var_ratio = manifold_results["explained_variance_ratio"].cpu().numpy()

        n_components = min(20, len(var_ratio))
        cumsum_var = np.cumsum(var_ratio[:n_components])

        ax.bar(range(n_components), var_ratio[:n_components], alpha=0.7, label="Individual")
        ax.plot(range(n_components), cumsum_var, 'r-o', label="Cumulative")

        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label="90% threshold")
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance Ratio")
        ax.set_title("Manifold Variance Explained")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_attractor_basins(
        self,
        ax: plt.Axes,
        phase_results: Dict
    ):
        """Plot attractor basins."""
        basins = phase_results["attractor_basins"].cpu().numpy()

        im = ax.imshow(basins.T, origin="lower", cmap="tab10", alpha=0.5)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("Attractor Basins")
        plt.colorbar(im, ax=ax, label="Basin ID")

    # ==================== UTILITY METHODS ====================

    def _delay_embed(
        self,
        trajectories: torch.Tensor,
        n_delays: int
    ) -> torch.Tensor:
        """
        Create delay embedding of trajectories.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            n_delays: Number of delays

        Returns:
            embedded: (n_trials, n_timesteps - n_delays + 1, n_dims * n_delays)
        """
        n_trials, n_timesteps, n_dims = trajectories.shape

        embedded = []
        for delay in range(n_delays):
            embedded.append(trajectories[:, delay:n_timesteps-n_delays+delay+1, :])

        embedded = torch.cat(embedded, dim=-1)

        return embedded

    def summary_report(
        self,
        trajectories: torch.Tensor,
        include_all: bool = True
    ) -> str:
        """
        Generate comprehensive summary report of dynamics.

        Args:
            trajectories: (n_trials, n_timesteps, n_dims)
            include_all: Whether to run all analyses

        Returns:
            report: String with formatted summary
        """
        self._log("Generating comprehensive dynamics summary...")

        lines = []
        lines.append("="*70)
        lines.append("DYNAMICAL SYSTEMS ANALYSIS SUMMARY")
        lines.append("="*70)

        n_trials, n_timesteps, n_dims = trajectories.shape
        lines.append(f"\nData shape: {n_trials} trials × {n_timesteps} timesteps × {n_dims} dimensions")
        lines.append(f"Time step: {self.dt:.6f} s")
        lines.append(f"Total duration: {(n_timesteps-1)*self.dt:.6f} s")

        if include_all:
            # Koopman analysis
            lines.append("\n" + "-"*70)
            lines.append("KOOPMAN OPERATOR ANALYSIS")
            lines.append("-"*70)
            koopman = self.estimate_koopman_operator(trajectories)
            dominant = self.identify_dominant_modes(koopman, n_modes=5)

            lines.append(f"Reconstruction error: {koopman['reconstruction_error']:.6f}")
            lines.append(f"\nTop 5 dominant modes:")
            for i in range(min(5, len(dominant['eigenvalues']))):
                lines.append(
                    f"  Mode {i+1}: λ={dominant['eigenvalues'][i]:.4f}, "
                    f"γ={dominant['growth_rates'][i]:.4f}/s, "
                    f"f={dominant['frequencies'][i]:.4f}Hz"
                )

            # Lyapunov analysis
            lines.append("\n" + "-"*70)
            lines.append("LYAPUNOV EXPONENTS")
            lines.append("-"*70)
            lyapunov = self.compute_lyapunov_exponents(trajectories)

            lines.append(f"Maximum Lyapunov exponent: {lyapunov['max_lyapunov_exponent']:.6f}")
            lines.append(f"Chaos detected: {lyapunov['chaos_detected']}")
            lines.append(f"Divergence rate: {lyapunov['divergence_rate']:.6f}")

            # Manifold analysis
            lines.append("\n" + "-"*70)
            lines.append("MANIFOLD ANALYSIS")
            lines.append("-"*70)
            manifold = self.analyze_manifold(trajectories)

            lines.append(f"Intrinsic dimensionality: {manifold['intrinsic_dimensionality']}")
            lines.append(f"Participation ratio: {manifold['participation_ratio']:.2f}")
            lines.append(f"Variance explained by top 3 PCs: "
                        f"{manifold['explained_variance_ratio'][:3].sum():.2%}")

            # Phase space analysis
            lines.append("\n" + "-"*70)
            lines.append("PHASE SPACE ANALYSIS")
            lines.append("-"*70)
            phase = self.analyze_phase_space(trajectories)

            lines.append(f"Fixed points detected: {len(phase['fixed_points'])}")
            if len(phase['fixed_points']) > 0:
                stability_counts = {}
                for stab in phase['fixed_point_stability']:
                    stability_counts[stab] = stability_counts.get(stab, 0) + 1
                for stab, count in stability_counts.items():
                    lines.append(f"  {stab}: {count}")

            lines.append(f"Limit cycles detected: {len(phase['limit_cycles'])}")

            # Stability analysis
            lines.append("\n" + "-"*70)
            lines.append("STABILITY ANALYSIS")
            lines.append("-"*70)
            stability = self.analyze_stability(trajectories)

            lines.append(f"System is stable: {stability['is_stable']}")
            lines.append(f"Max eigenvalue (real): {stability['max_eigenvalue_real']:.6f}")
            lines.append(f"Stability margin: {stability['stability_margin']:.6f}")
            lines.append(f"Mean perturbation growth: {stability['perturbation_growth'].mean():.4f}")

        lines.append("\n" + "="*70)

        report = "\n".join(lines)
        print(report)

        return report


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example: Analyze Lorenz attractor dynamics
    print("Testing DynamicsAnalyzer on Lorenz attractor...")

    # Generate Lorenz attractor data
    def lorenz_system(state, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return state + dt * np.array([dx, dy, dz])

    # Generate trajectories
    n_trials = 20
    n_timesteps = 1000
    dt = 0.01

    trajectories = []
    for _ in range(n_trials):
        # Random initial condition
        state = np.random.randn(3) * 5
        traj = [state]

        for _ in range(n_timesteps - 1):
            state = lorenz_system(state, dt=dt)
            traj.append(state)

        trajectories.append(np.array(traj))

    trajectories = torch.from_numpy(np.array(trajectories)).float()

    # Create analyzer
    analyzer = DynamicsAnalyzer(dt=dt, verbose=True)

    # Run comprehensive analysis
    print("\n" + "="*70)
    report = analyzer.summary_report(trajectories, include_all=True)

    print("\n✓ DynamicsAnalyzer test complete!")
