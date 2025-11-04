"""
Information-Theoretic Analysis

This module provides information-theoretic measures for analyzing
dynamical systems and causal relationships.

Key capabilities:
- Mutual information
- Transfer entropy
- Active information storage
- Information flow
- Complexity measures (statistical complexity, excess entropy)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)


@dataclass
class InformationResult:
    """Results from information-theoretic analysis."""

    # Mutual information
    mutual_information: float  # MI between variables (bits)

    # Transfer entropy
    transfer_entropy: Optional[float] = None  # TE from X to Y (bits)
    transfer_entropy_reverse: Optional[float] = None  # TE from Y to X (bits)

    # Information storage
    active_information_storage: Optional[float] = None  # AIS (bits)

    # Complexity measures
    statistical_complexity: Optional[float] = None  # Statistical complexity
    excess_entropy: Optional[float] = None  # Excess entropy

    # Predictive information
    predictive_information: Optional[float] = None  # PI (bits)


@dataclass
class InformationDecomposition:
    """Partial information decomposition."""

    redundancy: float  # Redundant information
    unique_X: float  # Unique information from X
    unique_Y: float  # Unique information from Y
    synergy: float  # Synergistic information


class InformationAnalyzer:
    """
    Information-theoretic analysis of dynamical systems.

    Information theory provides model-free measures of dependence,
    causality, and complexity.
    """

    def __init__(
        self,
        dt: float = 0.01,
        k_neighbors: int = 3,
        verbose: bool = True
    ):
        """
        Initialize information analyzer.

        Args:
            dt: Time step
            k_neighbors: Number of neighbors for k-NN entropy estimation
            verbose: Whether to log information
        """
        self.dt = dt
        self.k_neighbors = k_neighbors
        self.verbose = verbose

    def analyze(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        compute_transfer_entropy: bool = True,
        compute_complexity: bool = True
    ) -> InformationResult:
        """
        Comprehensive information-theoretic analysis.

        Args:
            X: First time series (n_timesteps,) or (n_timesteps, n_features)
            Y: Second time series (optional, for pairwise analysis)
            compute_transfer_entropy: Whether to compute transfer entropy
            compute_complexity: Whether to compute complexity measures

        Returns:
            InformationResult
        """
        # Ensure 2D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if Y is not None and Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Mutual information
        if Y is not None:
            mi = self.mutual_information(X, Y)
        else:
            # Auto-MI with time lag
            mi = self.mutual_information(X[:-1], X[1:])

        # Transfer entropy
        if compute_transfer_entropy and Y is not None:
            te_xy = self.transfer_entropy(X, Y)
            te_yx = self.transfer_entropy(Y, X)
        else:
            te_xy = None
            te_yx = None

        # Active information storage
        ais = self.active_information_storage(X)

        # Complexity measures
        if compute_complexity:
            stat_complexity = self.statistical_complexity(X)
            excess_entropy = self.excess_entropy(X)
        else:
            stat_complexity = None
            excess_entropy = None

        # Predictive information
        pred_info = self.predictive_information(X, max_lag=10)

        return InformationResult(
            mutual_information=mi,
            transfer_entropy=te_xy,
            transfer_entropy_reverse=te_yx,
            active_information_storage=ais,
            statistical_complexity=stat_complexity,
            excess_entropy=excess_entropy,
            predictive_information=pred_info
        )

    def mutual_information(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        method: str = "knn"
    ) -> float:
        """
        Compute mutual information I(X;Y).

        Args:
            X: First variable (n_samples, n_features_x)
            Y: Second variable (n_samples, n_features_y)
            method: Estimation method ("knn", "binning")

        Returns:
            Mutual information in bits
        """
        if method == "knn":
            return self._mutual_information_knn(X, Y)
        elif method == "binning":
            return self._mutual_information_binning(X, Y)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _mutual_information_knn(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Estimate MI using k-nearest neighbors (Kraskov estimator).

        Args:
            X: First variable (n_samples, n_features_x)
            Y: Second variable (n_samples, n_features_y)

        Returns:
            MI estimate in nats (divide by log(2) for bits)
        """
        n_samples = len(X)

        # Joint space
        XY = np.hstack([X, Y])

        # Build k-NN tree for joint space
        tree_xy = cKDTree(XY)

        # Find k-nearest neighbors in joint space
        distances, _ = tree_xy.query(XY, k=self.k_neighbors + 1)
        epsilon = distances[:, -1]  # Distance to k-th neighbor

        # Count neighbors in marginal spaces within epsilon
        tree_x = cKDTree(X)
        tree_y = cKDTree(Y)

        nx = np.array([len(tree_x.query_ball_point(X[i], r=epsilon[i] - 1e-15)) - 1 for i in range(n_samples)])
        ny = np.array([len(tree_y.query_ball_point(Y[i], r=epsilon[i] - 1e-15)) - 1 for i in range(n_samples)])

        # Kraskov estimator
        mi = digamma(self.k_neighbors) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n_samples)

        # Convert to bits
        mi_bits = mi / np.log(2)

        return max(0, mi_bits)

    def _mutual_information_binning(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Estimate MI using histogram binning.

        Args:
            X: First variable (n_samples, 1)
            Y: Second variable (n_samples, 1)
            n_bins: Number of bins

        Returns:
            MI estimate in bits
        """
        # Only works for 1D variables
        if X.shape[1] > 1 or Y.shape[1] > 1:
            # Use first feature only
            X = X[:, 0:1]
            Y = Y[:, 0:1]

        # Compute joint and marginal histograms
        hist_xy, _, _ = np.histogram2d(X.flatten(), Y.flatten(), bins=n_bins)
        hist_x = np.histogram(X, bins=n_bins)[0]
        hist_y = np.histogram(Y, bins=n_bins)[0]

        # Normalize
        p_xy = hist_xy / np.sum(hist_xy)
        p_x = hist_x / np.sum(hist_x)
        p_y = hist_y / np.sum(hist_y)

        # Compute MI: sum_xy p(x,y) log(p(x,y) / (p(x)p(y)))
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

        return max(0, mi)

    def transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1,
        history_length: int = 1
    ) -> float:
        """
        Compute transfer entropy from source to target.

        TE(S→T) = I(T_future; S_past | T_past)

        Args:
            source: Source time series (n_timesteps, n_features)
            target: Target time series (n_timesteps, n_features)
            lag: Prediction lag
            history_length: Length of history to condition on

        Returns:
            Transfer entropy in bits
        """
        n_samples = len(target) - max(lag, history_length)

        # Construct past and future
        target_future = target[history_length + lag:history_length + lag + n_samples]
        target_past = self._create_history(target, history_length)[:n_samples]
        source_past = self._create_history(source, history_length)[:n_samples]

        # TE = I(target_future; source_past | target_past)
        # = H(target_future | target_past) - H(target_future | target_past, source_past)

        # Conditional entropy estimation using k-NN
        h1 = self._conditional_entropy_knn(target_future, target_past)
        h2 = self._conditional_entropy_knn(
            target_future,
            np.hstack([target_past, source_past])
        )

        te = h1 - h2

        return max(0, te)

    def _create_history(self, X: np.ndarray, history_length: int) -> np.ndarray:
        """
        Create history embeddings.

        Args:
            X: Time series (n_timesteps, n_features)
            history_length: Length of history

        Returns:
            History matrix (n_timesteps - history_length, n_features * history_length)
        """
        n_timesteps, n_features = X.shape
        n_samples = n_timesteps - history_length + 1

        history = np.zeros((n_samples, n_features * history_length))

        for i in range(n_samples):
            for j in range(history_length):
                history[i, j * n_features:(j + 1) * n_features] = X[i + j]

        return history

    def _conditional_entropy_knn(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Estimate conditional entropy H(X|Y) using k-NN.

        Args:
            X: Variable (n_samples, n_features_x)
            Y: Conditioning variable (n_samples, n_features_y)

        Returns:
            Conditional entropy in bits
        """
        n_samples = len(X)

        # Joint space
        XY = np.hstack([X, Y])

        # Build trees
        tree_xy = cKDTree(XY)
        tree_y = cKDTree(Y)

        # Find k-nearest neighbors
        distances_xy, _ = tree_xy.query(XY, k=self.k_neighbors + 1)
        epsilon = distances_xy[:, -1]

        # Count neighbors in Y space
        ny = np.array([len(tree_y.query_ball_point(Y[i], r=epsilon[i] - 1e-15)) - 1 for i in range(n_samples)])

        # Conditional entropy estimator
        h = -digamma(self.k_neighbors) + np.mean(digamma(ny + 1))

        # Convert to bits
        h_bits = h / np.log(2)

        return h_bits

    def active_information_storage(
        self,
        X: np.ndarray,
        history_length: int = 1
    ) -> float:
        """
        Compute active information storage (self-predictive information).

        AIS = I(X_past; X_future)

        Args:
            X: Time series (n_timesteps, n_features)
            history_length: Length of history

        Returns:
            AIS in bits
        """
        # Create past and future
        X_past = self._create_history(X[:-1], history_length)
        X_future = X[history_length:]

        # Trim to same length
        min_len = min(len(X_past), len(X_future))
        X_past = X_past[:min_len]
        X_future = X_future[:min_len]

        # Mutual information between past and future
        ais = self.mutual_information(X_past, X_future)

        return ais

    def statistical_complexity(
        self,
        X: np.ndarray,
        max_history: int = 5
    ) -> float:
        """
        Compute statistical complexity (Crutchfield measure).

        C = H(S) where S is the causal state

        Args:
            X: Time series (n_timesteps, n_features)
            max_history: Maximum history length

        Returns:
            Statistical complexity in bits
        """
        # Discretize time series
        n_bins = 10
        X_discrete = np.digitize(X[:, 0], bins=np.linspace(X[:, 0].min(), X[:, 0].max(), n_bins))

        # Estimate causal states using histories
        histories = {}
        for i in range(max_history, len(X_discrete)):
            history = tuple(X_discrete[i - max_history:i])
            future = X_discrete[i]

            if history not in histories:
                histories[history] = []
            histories[history].append(future)

        # Cluster similar histories (simplified)
        # Full implementation requires epsilon-machine reconstruction

        # For now, use history entropy as proxy
        history_counts = {h: len(futures) for h, futures in histories.items()}
        total = sum(history_counts.values())
        probabilities = np.array(list(history_counts.values())) / total

        complexity = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        return complexity

    def excess_entropy(
        self,
        X: np.ndarray,
        max_history: int = 10
    ) -> float:
        """
        Compute excess entropy (predictive information up to infinite past).

        E = sum_{k=1}^∞ MI(X_past^k; X_future^k)

        Args:
            X: Time series (n_timesteps, n_features)
            max_history: Maximum history to consider

        Returns:
            Excess entropy in bits
        """
        excess_entropy = 0.0

        for k in range(1, max_history + 1):
            # Past and future of length k
            X_past = self._create_history(X[:-k], k)
            X_future = self._create_history(X[k:], k)

            min_len = min(len(X_past), len(X_future))
            X_past = X_past[:min_len]
            X_future = X_future[:min_len]

            if len(X_past) < 10:
                break

            mi = self.mutual_information(X_past, X_future)
            excess_entropy += mi

        return excess_entropy

    def predictive_information(
        self,
        X: np.ndarray,
        max_lag: int = 10
    ) -> float:
        """
        Compute predictive information (mutual information with time-shifted self).

        Args:
            X: Time series (n_timesteps, n_features)
            max_lag: Maximum lag to consider

        Returns:
            Total predictive information in bits
        """
        pred_info = 0.0

        for lag in range(1, max_lag + 1):
            mi = self.mutual_information(X[:-lag], X[lag:])
            pred_info += mi

        return pred_info

    def information_decomposition(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> InformationDecomposition:
        """
        Partial information decomposition (PID).

        Decompose I(X,Y; Z) into redundancy, unique, and synergy.

        Args:
            X: First predictor (n_samples, n_features)
            Y: Second predictor (n_samples, n_features)
            Z: Target (n_samples, n_features)

        Returns:
            InformationDecomposition
        """
        # Compute mutual informations
        I_XZ = self.mutual_information(X, Z)
        I_YZ = self.mutual_information(Y, Z)
        I_XYZ = self._mutual_information_three(X, Y, Z)

        # Simplified PID (Williams-Beer decomposition)
        # Redundancy: min(I(X;Z), I(Y;Z))
        redundancy = min(I_XZ, I_YZ)

        # Unique information
        unique_X = I_XZ - redundancy
        unique_Y = I_YZ - redundancy

        # Synergy
        synergy = I_XYZ - I_XZ - I_YZ + redundancy

        return InformationDecomposition(
            redundancy=redundancy,
            unique_X=unique_X,
            unique_Y=unique_Y,
            synergy=max(0, synergy)
        )

    def _mutual_information_three(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> float:
        """
        Compute I(X,Y; Z) = I(X; Z) + I(Y; Z | X)

        Args:
            X: First variable
            Y: Second variable
            Z: Target variable

        Returns:
            Three-way mutual information in bits
        """
        # I(X,Y; Z) using joint XY
        XY = np.hstack([X, Y])
        return self.mutual_information(XY, Z)
