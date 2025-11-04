"""
Synchronization Analysis

This module provides tools for analyzing synchronization in coupled
dynamical systems and networks.

Key capabilities:
- Synchronization measures (phase sync, complete sync, lag sync)
- Kuramoto order parameter
- Phase difference analysis
- Synchronization manifold detection
- Master stability function
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class SynchronizationResult:
    """Results from synchronization analysis."""

    # Global synchronization metrics
    sync_level: float  # Overall synchronization level [0, 1]
    sync_type: str  # "complete", "phase", "lag", "cluster", "none"

    # Phase synchronization
    kuramoto_order_parameter: Optional[float] = None  # Phase coherence [0, 1]
    mean_phase_difference: Optional[float] = None  # Mean phase difference
    phase_locking_value: Optional[float] = None  # Phase locking value

    # Pairwise synchronization
    sync_matrix: Optional[np.ndarray] = None  # Pairwise synchronization strengths

    # Lag synchronization
    optimal_lags: Optional[np.ndarray] = None  # Optimal time lags
    lag_sync_strength: Optional[float] = None  # Lag synchronization strength

    # Cluster synchronization
    n_clusters: Optional[int] = None  # Number of synchronized clusters
    cluster_assignments: Optional[np.ndarray] = None  # Cluster membership


@dataclass
class PhaseResult:
    """Results from phase analysis."""

    phases: np.ndarray  # Instantaneous phases (n_timesteps, n_oscillators)
    frequencies: np.ndarray  # Mean frequencies (n_oscillators,)
    amplitudes: Optional[np.ndarray] = None  # Instantaneous amplitudes


class SynchronizationAnalyzer:
    """
    Analyze synchronization in coupled dynamical systems.

    Synchronization occurs when coupled systems adjust their rhythms
    due to coupling or external forcing.
    """

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize synchronization analyzer.

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
        compute_phases: bool = True,
        detect_clusters: bool = True
    ) -> SynchronizationResult:
        """
        Comprehensive synchronization analysis.

        Args:
            trajectories: Trajectory data
                         Shape: (n_timesteps, n_oscillators) or (n_timesteps, n_oscillators, n_features)
            compute_phases: Whether to compute phase synchronization
            detect_clusters: Whether to detect cluster synchronization

        Returns:
            SynchronizationResult
        """
        # Handle multi-dimensional oscillators
        if trajectories.ndim == 3:
            # Use first feature for phase analysis
            X = trajectories[:, :, 0]
        else:
            X = trajectories

        n_timesteps, n_oscillators = X.shape

        # Extract phases
        if compute_phases:
            phase_result = self._extract_phases(X)
            phases = phase_result.phases

            # Kuramoto order parameter
            kuramoto_r = self._compute_kuramoto_order(phases)

            # Phase locking value
            plv = self._compute_phase_locking_value(phases)

            # Mean phase difference
            mean_phase_diff = self._compute_mean_phase_difference(phases)
        else:
            kuramoto_r = None
            plv = None
            mean_phase_diff = None
            phases = None

        # Pairwise synchronization matrix
        sync_matrix = self._compute_sync_matrix(X)

        # Check for complete synchronization
        is_complete_sync = self._check_complete_sync(X)

        # Check for lag synchronization
        optimal_lags, lag_sync = self._compute_lag_sync(X)

        # Detect synchronized clusters
        if detect_clusters:
            n_clusters, cluster_assignments = self._detect_clusters(sync_matrix)
        else:
            n_clusters = None
            cluster_assignments = None

        # Determine synchronization type
        sync_type = self._classify_sync_type(
            is_complete_sync,
            kuramoto_r,
            lag_sync,
            n_clusters,
            n_oscillators
        )

        # Overall synchronization level
        sync_level = self._compute_overall_sync_level(
            sync_type,
            kuramoto_r,
            lag_sync,
            sync_matrix
        )

        return SynchronizationResult(
            sync_level=sync_level,
            sync_type=sync_type,
            kuramoto_order_parameter=kuramoto_r,
            mean_phase_difference=mean_phase_diff,
            phase_locking_value=plv,
            sync_matrix=sync_matrix,
            optimal_lags=optimal_lags,
            lag_sync_strength=lag_sync,
            n_clusters=n_clusters,
            cluster_assignments=cluster_assignments
        )

    def _extract_phases(self, X: np.ndarray) -> PhaseResult:
        """
        Extract instantaneous phases using Hilbert transform.

        Args:
            X: Time series (n_timesteps, n_oscillators)

        Returns:
            PhaseResult
        """
        n_timesteps, n_oscillators = X.shape

        phases = np.zeros_like(X)
        amplitudes = np.zeros_like(X)
        frequencies = np.zeros(n_oscillators)

        for i in range(n_oscillators):
            # Analytic signal via Hilbert transform
            analytic_signal = hilbert(X[:, i])

            # Phase
            phases[:, i] = np.angle(analytic_signal)

            # Amplitude
            amplitudes[:, i] = np.abs(analytic_signal)

            # Mean frequency
            phase_diff = np.diff(np.unwrap(phases[:, i]))
            frequencies[i] = np.mean(phase_diff) / self.dt / (2 * np.pi)

        return PhaseResult(
            phases=phases,
            frequencies=frequencies,
            amplitudes=amplitudes
        )

    def _compute_kuramoto_order(self, phases: np.ndarray) -> float:
        """
        Compute Kuramoto order parameter.

        R = |⟨exp(i θ_j)⟩_j|

        Args:
            phases: Phase array (n_timesteps, n_oscillators)

        Returns:
            Order parameter [0, 1]
        """
        # Complex representation
        complex_phases = np.exp(1j * phases)

        # Mean over oscillators
        mean_complex = np.mean(complex_phases, axis=1)

        # Order parameter (magnitude)
        R = np.abs(mean_complex)

        # Time average
        R_mean = np.mean(R)

        return R_mean

    def _compute_phase_locking_value(self, phases: np.ndarray) -> float:
        """
        Compute phase locking value (PLV) averaged over all pairs.

        PLV_ij = |⟨exp(i(θ_i - θ_j))⟩_t|

        Args:
            phases: Phase array (n_timesteps, n_oscillators)

        Returns:
            Mean PLV [0, 1]
        """
        n_timesteps, n_oscillators = phases.shape

        plvs = []

        for i in range(n_oscillators):
            for j in range(i + 1, n_oscillators):
                phase_diff = phases[:, i] - phases[:, j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plvs.append(plv)

        return np.mean(plvs) if len(plvs) > 0 else 0.0

    def _compute_mean_phase_difference(self, phases: np.ndarray) -> float:
        """
        Compute mean phase difference between oscillators.

        Args:
            phases: Phase array (n_timesteps, n_oscillators)

        Returns:
            Mean phase difference [0, π]
        """
        n_timesteps, n_oscillators = phases.shape

        phase_diffs = []

        for i in range(n_oscillators):
            for j in range(i + 1, n_oscillators):
                phase_diff = np.abs(phases[:, i] - phases[:, j])
                # Map to [0, π]
                phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
                phase_diffs.append(np.mean(phase_diff))

        return np.mean(phase_diffs) if len(phase_diffs) > 0 else 0.0

    def _compute_sync_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise synchronization matrix using correlation.

        Args:
            X: Time series (n_timesteps, n_oscillators)

        Returns:
            Synchronization matrix (n_oscillators, n_oscillators)
        """
        n_oscillators = X.shape[1]
        sync_matrix = np.zeros((n_oscillators, n_oscillators))

        for i in range(n_oscillators):
            for j in range(n_oscillators):
                # Pearson correlation
                if i == j:
                    sync_matrix[i, j] = 1.0
                else:
                    corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                    sync_matrix[i, j] = np.abs(corr)

        return sync_matrix

    def _check_complete_sync(
        self,
        X: np.ndarray,
        threshold: float = 0.05
    ) -> bool:
        """
        Check if system exhibits complete synchronization.

        Args:
            X: Time series (n_timesteps, n_oscillators)
            threshold: Threshold for synchronization

        Returns:
            True if completely synchronized
        """
        # Check if all trajectories are nearly identical
        mean_trajectory = np.mean(X, axis=1, keepdims=True)
        deviations = np.std(X - mean_trajectory, axis=1)
        mean_deviation = np.mean(deviations)

        # Normalize by signal amplitude
        signal_std = np.std(X)

        if signal_std > 0:
            normalized_deviation = mean_deviation / signal_std
        else:
            normalized_deviation = 0

        return normalized_deviation < threshold

    def _compute_lag_sync(
        self,
        X: np.ndarray,
        max_lag: int = 50
    ) -> Tuple[np.ndarray, float]:
        """
        Compute lag synchronization (time-shifted synchronization).

        Args:
            X: Time series (n_timesteps, n_oscillators)
            max_lag: Maximum lag to search

        Returns:
            Tuple of (optimal_lags, lag_sync_strength)
        """
        n_oscillators = X.shape[1]
        optimal_lags = np.zeros((n_oscillators, n_oscillators), dtype=int)
        lag_sync_strengths = np.zeros((n_oscillators, n_oscillators))

        for i in range(n_oscillators):
            for j in range(i + 1, n_oscillators):
                # Find optimal lag
                best_lag = 0
                best_corr = 0

                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        x1 = X[:lag, i]
                        x2 = X[-lag:, j]
                    elif lag > 0:
                        x1 = X[lag:, i]
                        x2 = X[:-lag, j]
                    else:
                        x1 = X[:, i]
                        x2 = X[:, j]

                    if len(x1) > 0:
                        corr = np.abs(np.corrcoef(x1, x2)[0, 1])
                        if corr > best_corr:
                            best_corr = corr
                            best_lag = lag

                optimal_lags[i, j] = best_lag
                optimal_lags[j, i] = -best_lag
                lag_sync_strengths[i, j] = best_corr
                lag_sync_strengths[j, i] = best_corr

        # Mean lag synchronization strength
        lag_sync_mean = np.mean(lag_sync_strengths[np.triu_indices(n_oscillators, k=1)])

        return optimal_lags, lag_sync_mean

    def _detect_clusters(
        self,
        sync_matrix: np.ndarray,
        threshold: float = 0.7
    ) -> Tuple[int, np.ndarray]:
        """
        Detect synchronized clusters using synchronization matrix.

        Args:
            sync_matrix: Pairwise synchronization strengths
            threshold: Threshold for cluster membership

        Returns:
            Tuple of (n_clusters, cluster_assignments)
        """
        from sklearn.cluster import AgglomerativeClustering

        # Convert synchronization to distance
        distance_matrix = 1 - sync_matrix

        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - threshold,
            metric='precomputed',
            linkage='average'
        )

        cluster_assignments = clustering.fit_predict(distance_matrix)
        n_clusters = len(np.unique(cluster_assignments))

        return n_clusters, cluster_assignments

    def _classify_sync_type(
        self,
        is_complete: bool,
        kuramoto_r: Optional[float],
        lag_sync: Optional[float],
        n_clusters: Optional[int],
        n_oscillators: int
    ) -> str:
        """
        Classify type of synchronization.

        Args:
            is_complete: Complete synchronization flag
            kuramoto_r: Kuramoto order parameter
            lag_sync: Lag synchronization strength
            n_clusters: Number of clusters
            n_oscillators: Total oscillators

        Returns:
            Synchronization type string
        """
        if is_complete:
            return "complete"

        if kuramoto_r is not None and kuramoto_r > 0.8:
            return "phase"

        if lag_sync is not None and lag_sync > 0.8:
            return "lag"

        if n_clusters is not None and 1 < n_clusters < n_oscillators:
            return "cluster"

        return "none"

    def _compute_overall_sync_level(
        self,
        sync_type: str,
        kuramoto_r: Optional[float],
        lag_sync: Optional[float],
        sync_matrix: np.ndarray
    ) -> float:
        """
        Compute overall synchronization level [0, 1].

        Args:
            sync_type: Synchronization type
            kuramoto_r: Kuramoto order parameter
            lag_sync: Lag synchronization strength
            sync_matrix: Pairwise synchronization matrix

        Returns:
            Overall synchronization level
        """
        if sync_type == "complete":
            return 1.0

        # Combine multiple metrics
        levels = []

        if kuramoto_r is not None:
            levels.append(kuramoto_r)

        if lag_sync is not None:
            levels.append(lag_sync)

        # Mean pairwise synchronization
        n = sync_matrix.shape[0]
        if n > 1:
            pairwise_mean = np.mean(sync_matrix[np.triu_indices(n, k=1)])
            levels.append(pairwise_mean)

        if len(levels) > 0:
            return np.mean(levels)
        else:
            return 0.0

    def compute_master_stability_function(
        self,
        coupling_function: callable,
        equilibrium_state: np.ndarray,
        coupling_range: Tuple[float, float] = (0.0, 2.0),
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute master stability function for network synchronization.

        The master stability function determines stability of synchronized
        state as a function of coupling strength.

        Args:
            coupling_function: Coupling function
            equilibrium_state: Synchronized equilibrium state
            coupling_range: Range of coupling strengths
            n_points: Number of points to sample

        Returns:
            Tuple of (coupling_strengths, max_lyapunov_exponents)
        """
        coupling_strengths = np.linspace(coupling_range[0], coupling_range[1], n_points)
        max_lyapunovs = np.zeros(n_points)

        for i, sigma in enumerate(coupling_strengths):
            # Compute largest Lyapunov exponent of variational equation
            # This is a simplified version
            # Full implementation requires integration of variational equations

            # Estimate using Jacobian eigenvalues
            J = self._estimate_variational_jacobian(
                coupling_function,
                equilibrium_state,
                sigma
            )

            eigenvalues = np.linalg.eigvals(J)
            max_lyapunovs[i] = np.max(np.real(eigenvalues))

        return coupling_strengths, max_lyapunovs

    def _estimate_variational_jacobian(
        self,
        coupling_function: callable,
        state: np.ndarray,
        coupling_strength: float,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """
        Estimate Jacobian of variational equation.

        Args:
            coupling_function: Coupling function
            state: State
            coupling_strength: Coupling strength
            epsilon: Finite difference step

        Returns:
            Jacobian matrix
        """
        n = len(state)
        J = np.zeros((n, n))

        f0 = coupling_function(state, coupling_strength)

        for i in range(n):
            state_plus = state.copy()
            state_plus[i] += epsilon

            f_plus = coupling_function(state_plus, coupling_strength)

            J[:, i] = (f_plus - f0) / epsilon

        return J
