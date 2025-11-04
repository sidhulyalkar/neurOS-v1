"""
Transfer Operator Methods

This module provides tools for analyzing stochastic dynamical systems
using transfer operator theory and Perron-Frobenius operators.

Key capabilities:
- Transfer operator estimation
- Dominant eigenvalues and eigenfunctions
- Invariant measures and densities
- Metastability analysis
- Transition path theory
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransferOperatorResult:
    """Results from transfer operator analysis."""

    # Transfer operator
    transfer_matrix: np.ndarray  # Discretized transfer operator
    stationary_distribution: np.ndarray  # Invariant measure

    # Spectral properties
    eigenvalues: np.ndarray  # Leading eigenvalues
    eigenfunctions: np.ndarray  # Leading eigenfunctions
    timescales: np.ndarray  # Implied timescales

    # Metastability
    n_metastable_sets: int  # Number of metastable sets
    metastable_sets: Optional[np.ndarray] = None  # Assignment to metastable sets
    transition_matrix: Optional[np.ndarray] = None  # Coarse-grained transition matrix


@dataclass
class TransitionPath:
    """Transition path between states."""

    source_set: int  # Source metastable set
    target_set: int  # Target metastable set
    committor_forward: np.ndarray  # Forward committor function
    committor_backward: np.ndarray  # Backward committor function
    flux: Optional[np.ndarray] = None  # Reactive flux
    rate: float = 0.0  # Transition rate


class TransferOperator:
    """
    Transfer operator methods for stochastic dynamics.

    The transfer operator (Perron-Frobenius operator) describes the
    evolution of probability densities in stochastic systems.
    """

    def __init__(
        self,
        dt: float = 0.01,
        n_bins: int = 100,
        lag_time: int = 1,
        verbose: bool = True
    ):
        """
        Initialize transfer operator analyzer.

        Args:
            dt: Time step
            n_bins: Number of bins for discretization
            lag_time: Lag time for transition counting
            verbose: Whether to log information
        """
        self.dt = dt
        self.n_bins = n_bins
        self.lag_time = lag_time
        self.verbose = verbose

    def estimate(
        self,
        trajectories: np.ndarray,
        method: str = "ulam",
        n_eigenpairs: int = 10
    ) -> TransferOperatorResult:
        """
        Estimate transfer operator from trajectory data.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            method: Estimation method ("ulam", "galerkin", "edmd")
            n_eigenpairs: Number of eigenpairs to compute

        Returns:
            TransferOperatorResult
        """
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        if method == "ulam":
            result = self._ulam_method(trajectories, n_eigenpairs)
        elif method == "galerkin":
            result = self._galerkin_method(trajectories, n_eigenpairs)
        elif method == "edmd":
            result = self._edmd_method(trajectories, n_eigenpairs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Detect metastability
        n_metastable, metastable_sets, transition_matrix = self._detect_metastability(
            result.transfer_matrix,
            result.eigenfunctions
        )

        result.n_metastable_sets = n_metastable
        result.metastable_sets = metastable_sets
        result.transition_matrix = transition_matrix

        return result

    def _ulam_method(
        self,
        X: np.ndarray,
        n_eigenpairs: int
    ) -> TransferOperatorResult:
        """
        Estimate transfer operator using Ulam's method (box discretization).

        Args:
            X: Trajectory data
            n_eigenpairs: Number of eigenpairs

        Returns:
            TransferOperatorResult
        """
        n_timesteps, n_features = X.shape

        # Discretize state space using uniform grid
        bin_edges = []
        for d in range(n_features):
            edges = np.linspace(X[:, d].min(), X[:, d].max(), self.n_bins + 1)
            bin_edges.append(edges)

        # Assign states to bins
        bin_indices = np.zeros((n_timesteps, n_features), dtype=int)
        for d in range(n_features):
            bin_indices[:, d] = np.digitize(X[:, d], bin_edges[d]) - 1
            bin_indices[:, d] = np.clip(bin_indices[:, d], 0, self.n_bins - 1)

        # Convert to single index
        def multi_to_single(multi_idx):
            single = 0
            for d in range(n_features):
                single += multi_idx[d] * (self.n_bins ** d)
            return single

        state_indices = np.array([multi_to_single(idx) for idx in bin_indices])

        # Count transitions
        n_states = self.n_bins ** n_features
        count_matrix = np.zeros((n_states, n_states))

        for t in range(n_timesteps - self.lag_time):
            i = state_indices[t]
            j = state_indices[t + self.lag_time]
            count_matrix[i, j] += 1

        # Normalize to get transition probabilities
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transfer_matrix = count_matrix / row_sums

        # Stationary distribution (left eigenvector with eigenvalue 1)
        eigenvalues, eigenvectors = eigh(transfer_matrix.T)
        idx = np.argmax(np.abs(eigenvalues))
        stationary = np.abs(eigenvectors[:, idx])
        stationary /= np.sum(stationary)

        # Leading eigenpairs
        eigenvalues_sorted = np.sort(eigenvalues)[::-1][:n_eigenpairs]
        idx_sorted = np.argsort(eigenvalues)[::-1][:n_eigenpairs]
        eigenfunctions = eigenvectors[:, idx_sorted]

        # Implied timescales
        timescales = -self.lag_time * self.dt / np.log(np.abs(eigenvalues_sorted[1:]))

        return TransferOperatorResult(
            transfer_matrix=transfer_matrix,
            stationary_distribution=stationary,
            eigenvalues=eigenvalues_sorted,
            eigenfunctions=eigenfunctions,
            timescales=timescales,
            n_metastable_sets=0
        )

    def _galerkin_method(
        self,
        X: np.ndarray,
        n_eigenpairs: int
    ) -> TransferOperatorResult:
        """
        Estimate transfer operator using Galerkin approximation.

        Args:
            X: Trajectory data
            n_eigenpairs: Number of eigenpairs

        Returns:
            TransferOperatorResult
        """
        # Use radial basis functions as basis
        n_basis = min(100, len(X) // 10)

        # Select basis centers using k-means
        kmeans = KMeans(n_clusters=n_basis, n_init=10, random_state=42)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_

        # Compute basis functions (RBF)
        sigma = np.median(np.std(X, axis=0))

        def basis_function(x, center):
            return np.exp(-np.sum((x - center)**2) / (2 * sigma**2))

        # Evaluate basis functions
        Psi = np.zeros((len(X), n_basis))
        for i, x in enumerate(X):
            for j, center in enumerate(centers):
                Psi[i, j] = basis_function(x, center)

        # Compute transfer operator in basis representation
        # P_ij = ⟨ψ_i, P ψ_j⟩ / ⟨ψ_i, ψ_j⟩

        Psi_t = Psi[:-self.lag_time]
        Psi_t_lag = Psi[self.lag_time:]

        # Covariance matrices
        C_0 = Psi_t.T @ Psi_t / len(Psi_t)
        C_tau = Psi_t.T @ Psi_t_lag / len(Psi_t)

        # Transfer operator: T = C_0^{-1} C_tau
        try:
            T = np.linalg.solve(C_0, C_tau)
        except:
            T = np.linalg.lstsq(C_0, C_tau, rcond=None)[0]

        # Eigendecomposition
        eigenvalues, eigenvectors = eig(T, k=n_eigenpairs)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Sort by eigenvalue
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Project eigenfunctions to data points
        eigenfunctions = Psi @ eigenvectors

        # Stationary distribution (first eigenfunction)
        stationary = np.abs(eigenfunctions[:, 0])
        stationary /= np.sum(stationary)

        # Implied timescales
        timescales = -self.lag_time * self.dt / np.log(np.abs(eigenvalues[1:]))

        return TransferOperatorResult(
            transfer_matrix=T,
            stationary_distribution=stationary,
            eigenvalues=eigenvalues,
            eigenfunctions=eigenfunctions,
            timescales=timescales,
            n_metastable_sets=0
        )

    def _edmd_method(
        self,
        X: np.ndarray,
        n_eigenpairs: int
    ) -> TransferOperatorResult:
        """
        Extended Dynamic Mode Decomposition for transfer operator.

        Args:
            X: Trajectory data
            n_eigenpairs: Number of eigenpairs

        Returns:
            TransferOperatorResult
        """
        # Use polynomial observables
        def observables(x):
            return np.concatenate([x, x**2, np.array([np.prod(x[i:i+2]) for i in range(len(x)-1)])])

        # Compute observables
        G = np.array([observables(x) for x in X])

        # Split into X and Y
        G_t = G[:-self.lag_time]
        G_t_lag = G[self.lag_time:]

        # Compute Koopman operator
        K = np.linalg.lstsq(G_t, G_t_lag, rcond=None)[0]

        # Eigendecomposition
        eigenvalues, eigenvectors = eig(K, k=min(n_eigenpairs, K.shape[0]))
        eigenvalues = np.real(eigenvalues)

        # Sort
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Eigenfunctions at data points
        eigenfunctions = G_t @ eigenvectors

        # Stationary distribution
        stationary = np.abs(eigenfunctions[:, 0])
        stationary /= np.sum(stationary)

        # Extend stationary to all points
        stationary_full = np.abs(G @ eigenvectors[:, 0])
        stationary_full /= np.sum(stationary_full)

        # Timescales
        timescales = -self.lag_time * self.dt / np.log(np.abs(eigenvalues[1:]))

        return TransferOperatorResult(
            transfer_matrix=K,
            stationary_distribution=stationary_full,
            eigenvalues=eigenvalues,
            eigenfunctions=G @ eigenvectors,
            timescales=timescales,
            n_metastable_sets=0
        )

    def _detect_metastability(
        self,
        T: np.ndarray,
        eigenfunctions: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Detect metastable sets using PCCA+ algorithm.

        Args:
            T: Transfer matrix
            eigenfunctions: Eigenfunctions
            n_clusters: Number of metastable sets (auto-detect if None)

        Returns:
            Tuple of (n_sets, assignments, transition_matrix)
        """
        # Auto-detect number of metastable sets from eigenvalue gap
        if n_clusters is None:
            eigenvalues = np.linalg.eigvals(T)
            eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]

            # Find spectral gap
            gaps = np.diff(eigenvalues_sorted[:10])
            n_clusters = np.argmax(gaps) + 2  # +2 because we want clusters before gap

            n_clusters = max(2, min(n_clusters, 5))

        # Use eigenfunctions for clustering
        n_eigenfunctions = min(n_clusters, eigenfunctions.shape[1])
        features = eigenfunctions[:, :n_eigenfunctions]

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        assignments = kmeans.fit_predict(features)

        # Build coarse-grained transition matrix
        transition_matrix = np.zeros((n_clusters, n_clusters))

        for i in range(len(assignments) - 1):
            s_from = assignments[i]
            s_to = assignments[i + 1]
            transition_matrix[s_from, s_to] += 1

        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix /= row_sums

        return n_clusters, assignments, transition_matrix

    def compute_committor(
        self,
        transfer_matrix: np.ndarray,
        source_set: np.ndarray,
        target_set: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward and backward committor functions.

        The committor is the probability of reaching the target before source.

        Args:
            transfer_matrix: Transfer matrix
            source_set: Boolean array marking source states
            target_set: Boolean array marking target states

        Returns:
            Tuple of (forward_committor, backward_committor)
        """
        n_states = len(transfer_matrix)

        # Forward committor: q^+ = probability to reach target before source
        # Boundary conditions: q^+(target) = 1, q^+(source) = 0
        # For intermediate states: q^+ = T q^+

        # Set up linear system
        A = np.eye(n_states) - transfer_matrix
        b = np.zeros(n_states)

        # Boundary conditions
        for i in range(n_states):
            if target_set[i]:
                A[i] = 0
                A[i, i] = 1
                b[i] = 1
            elif source_set[i]:
                A[i] = 0
                A[i, i] = 1
                b[i] = 0

        # Solve
        try:
            forward_committor = np.linalg.solve(A, b)
        except:
            forward_committor = np.linalg.lstsq(A, b, rcond=None)[0]

        # Backward committor: q^- = probability to have come from source
        backward_committor = 1 - forward_committor

        return forward_committor, backward_committor

    def transition_path_theory(
        self,
        result: TransferOperatorResult,
        source_set_idx: int,
        target_set_idx: int
    ) -> TransitionPath:
        """
        Analyze transition paths using Transition Path Theory.

        Args:
            result: TransferOperatorResult
            source_set_idx: Index of source metastable set
            target_set_idx: Index of target metastable set

        Returns:
            TransitionPath object
        """
        # Convert metastable set indices to state indices
        source_states = result.metastable_sets == source_set_idx
        target_states = result.metastable_sets == target_set_idx

        # Compute committors
        q_plus, q_minus = self.compute_committor(
            result.transfer_matrix,
            source_states,
            target_states
        )

        # Compute reactive flux
        # f_ij = q_minus(i) * T_ij * (q_plus(j) - q_plus(i))
        n_states = len(result.transfer_matrix)
        flux = np.zeros((n_states, n_states))

        for i in range(n_states):
            for j in range(n_states):
                if result.transfer_matrix[i, j] > 0:
                    flux[i, j] = (
                        q_minus[i] *
                        result.transfer_matrix[i, j] *
                        (q_plus[j] - q_plus[i])
                    )

        # Total flux (transition rate)
        rate = np.sum(flux)

        return TransitionPath(
            source_set=source_set_idx,
            target_set=target_set_idx,
            committor_forward=q_plus,
            committor_backward=q_minus,
            flux=flux,
            rate=rate
        )
