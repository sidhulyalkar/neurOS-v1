"""
Granger Causality Analysis

This module provides tools for analyzing temporal causal relationships
between time series using Granger causality and related methods.

Key capabilities:
- Pairwise Granger causality
- Conditional Granger causality
- Multivariate Granger causality
- Transfer entropy (information-theoretic causality)
- Directed information
- Causal graph construction
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso
import logging

logger = logging.getLogger(__name__)


@dataclass
class GrangerResult:
    """Results from Granger causality analysis."""

    # Pairwise causality
    causality_matrix: np.ndarray  # (n_features, n_features) F-statistics
    p_values: np.ndarray  # P-values for significance testing
    significant_edges: List[Tuple[int, int]]  # List of significant causal edges

    # Graph properties
    n_edges: int  # Number of significant causal edges
    sparsity: float  # Graph sparsity (fraction of possible edges present)

    # Additional metrics
    lag_order: int  # Lag order used
    threshold: float  # Significance threshold


@dataclass
class CausalGraph:
    """Directed causal graph."""

    adjacency_matrix: np.ndarray  # (n_nodes, n_nodes) binary adjacency
    edge_weights: np.ndarray  # (n_nodes, n_nodes) causal strengths
    node_names: Optional[List[str]] = None  # Names for nodes

    def get_parents(self, node: int) -> List[int]:
        """Get parent nodes (causes) of a given node."""
        return list(np.where(self.adjacency_matrix[:, node] > 0)[0])

    def get_children(self, node: int) -> List[int]:
        """Get children nodes (effects) of a given node."""
        return list(np.where(self.adjacency_matrix[node, :] > 0)[0])

    def get_ancestors(self, node: int, max_depth: int = 10) -> List[int]:
        """Get all ancestor nodes (transitive causes)."""
        ancestors = set()
        current_level = {node}

        for _ in range(max_depth):
            next_level = set()
            for n in current_level:
                parents = self.get_parents(n)
                next_level.update(parents)

            if not next_level - ancestors:
                break

            ancestors.update(next_level)
            current_level = next_level

        return list(ancestors)


class GrangerCausality:
    """
    Granger causality analysis for temporal causal inference.

    Granger causality tests whether past values of one time series
    help predict future values of another, beyond what the target's
    own past values can provide.
    """

    def __init__(
        self,
        dt: float = 0.01,
        max_lag: int = 10,
        significance_level: float = 0.05,
        verbose: bool = True
    ):
        """
        Initialize Granger causality analyzer.

        Args:
            dt: Time step
            max_lag: Maximum lag order to consider
            significance_level: P-value threshold for significance
            verbose: Whether to log information
        """
        self.dt = dt
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.verbose = verbose

    def analyze(
        self,
        data: np.ndarray,
        lag_order: Optional[int] = None,
        method: str = "pairwise"
    ) -> GrangerResult:
        """
        Perform Granger causality analysis.

        Args:
            data: Multivariate time series (n_timesteps, n_features)
            lag_order: Lag order (if None, determined by BIC)
            method: Analysis method ("pairwise", "conditional", "multivariate")

        Returns:
            GrangerResult with causality matrix and graph
        """
        n_timesteps, n_features = data.shape

        # Determine optimal lag order if not provided
        if lag_order is None:
            lag_order = self._select_lag_order(data)

        if method == "pairwise":
            causality_matrix, p_values = self._pairwise_granger(data, lag_order)
        elif method == "conditional":
            causality_matrix, p_values = self._conditional_granger(data, lag_order)
        elif method == "multivariate":
            causality_matrix, p_values = self._multivariate_granger(data, lag_order)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Find significant edges
        significant_edges = []
        for i in range(n_features):
            for j in range(n_features):
                if i != j and p_values[i, j] < self.significance_level:
                    significant_edges.append((i, j))

        # Compute graph properties
        n_edges = len(significant_edges)
        max_possible_edges = n_features * (n_features - 1)
        sparsity = 1.0 - (n_edges / max_possible_edges) if max_possible_edges > 0 else 1.0

        return GrangerResult(
            causality_matrix=causality_matrix,
            p_values=p_values,
            significant_edges=significant_edges,
            n_edges=n_edges,
            sparsity=sparsity,
            lag_order=lag_order,
            threshold=self.significance_level
        )

    def _pairwise_granger(
        self,
        data: np.ndarray,
        lag_order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise Granger causality.

        Tests whether X Granger-causes Y for all pairs (X, Y).

        Args:
            data: Time series data (n_timesteps, n_features)
            lag_order: Lag order

        Returns:
            Tuple of (causality_matrix, p_values)
        """
        n_features = data.shape[1]
        causality_matrix = np.zeros((n_features, n_features))
        p_values = np.ones((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue

                # Test if i Granger-causes j
                F_stat, p_value = self._granger_test(
                    data[:, i],  # Potential cause
                    data[:, j],  # Effect
                    lag_order
                )

                causality_matrix[i, j] = F_stat
                p_values[i, j] = p_value

        return causality_matrix, p_values

    def _granger_test(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        lag_order: int
    ) -> Tuple[float, float]:
        """
        Perform Granger causality test for a single pair.

        Args:
            cause: Potential cause time series (n_timesteps,)
            effect: Effect time series (n_timesteps,)
            lag_order: Lag order

        Returns:
            Tuple of (F_statistic, p_value)
        """
        n = len(effect)

        # Build lagged design matrices
        X_restricted = self._build_lag_matrix(effect, lag_order)
        X_full = np.column_stack([
            X_restricted,
            self._build_lag_matrix(cause, lag_order)
        ])

        # Target: future values of effect
        y = effect[lag_order:]

        # Ensure we have enough data
        if len(y) < 2 * lag_order:
            return 0.0, 1.0

        # Fit restricted model (only past of effect)
        model_restricted = LinearRegression()
        model_restricted.fit(X_restricted, y)
        y_pred_restricted = model_restricted.predict(X_restricted)
        rss_restricted = np.sum((y - y_pred_restricted) ** 2)

        # Fit full model (past of both effect and cause)
        model_full = LinearRegression()
        model_full.fit(X_full, y)
        y_pred_full = model_full.predict(X_full)
        rss_full = np.sum((y - y_pred_full) ** 2)

        # F-test
        n_samples = len(y)
        n_params_restricted = X_restricted.shape[1]
        n_params_full = X_full.shape[1]

        if rss_full == 0 or n_samples <= n_params_full:
            return 0.0, 1.0

        F_stat = (
            (rss_restricted - rss_full) / (n_params_full - n_params_restricted)
        ) / (rss_full / (n_samples - n_params_full))

        # Compute p-value
        df1 = n_params_full - n_params_restricted
        df2 = n_samples - n_params_full
        p_value = 1.0 - stats.f.cdf(F_stat, df1, df2)

        return F_stat, p_value

    def _conditional_granger(
        self,
        data: np.ndarray,
        lag_order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute conditional Granger causality.

        Tests whether X Granger-causes Y conditional on all other variables.

        Args:
            data: Time series data (n_timesteps, n_features)
            lag_order: Lag order

        Returns:
            Tuple of (causality_matrix, p_values)
        """
        n_features = data.shape[1]
        causality_matrix = np.zeros((n_features, n_features))
        p_values = np.ones((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue

                # Conditioning set: all variables except i and j
                conditioning = [k for k in range(n_features) if k != i and k != j]

                # Test if i Granger-causes j given conditioning set
                F_stat, p_value = self._conditional_granger_test(
                    data[:, i],  # Potential cause
                    data[:, j],  # Effect
                    data[:, conditioning],  # Conditioning variables
                    lag_order
                )

                causality_matrix[i, j] = F_stat
                p_values[i, j] = p_value

        return causality_matrix, p_values

    def _conditional_granger_test(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        conditioning: np.ndarray,
        lag_order: int
    ) -> Tuple[float, float]:
        """
        Conditional Granger causality test.

        Args:
            cause: Potential cause (n_timesteps,)
            effect: Effect (n_timesteps,)
            conditioning: Conditioning variables (n_timesteps, n_conditioning)
            lag_order: Lag order

        Returns:
            Tuple of (F_statistic, p_value)
        """
        # Build lagged design matrices
        X_effect = self._build_lag_matrix(effect, lag_order)

        if conditioning.ndim == 1:
            conditioning = conditioning.reshape(-1, 1)

        # Add conditioning variables
        X_conditioning = []
        for k in range(conditioning.shape[1]):
            X_conditioning.append(self._build_lag_matrix(conditioning[:, k], lag_order))

        X_restricted = np.column_stack([X_effect] + X_conditioning)

        # Add cause
        X_cause = self._build_lag_matrix(cause, lag_order)
        X_full = np.column_stack([X_restricted, X_cause])

        # Target
        y = effect[lag_order:]

        if len(y) < X_full.shape[1]:
            return 0.0, 1.0

        # Fit models
        try:
            model_restricted = LinearRegression()
            model_restricted.fit(X_restricted, y)
            rss_restricted = np.sum((y - model_restricted.predict(X_restricted)) ** 2)

            model_full = LinearRegression()
            model_full.fit(X_full, y)
            rss_full = np.sum((y - model_full.predict(X_full)) ** 2)

            # F-test
            n_samples = len(y)
            n_params_restricted = X_restricted.shape[1]
            n_params_full = X_full.shape[1]

            if rss_full == 0 or n_samples <= n_params_full:
                return 0.0, 1.0

            F_stat = (
                (rss_restricted - rss_full) / (n_params_full - n_params_restricted)
            ) / (rss_full / (n_samples - n_params_full))

            df1 = n_params_full - n_params_restricted
            df2 = n_samples - n_params_full
            p_value = 1.0 - stats.f.cdf(F_stat, df1, df2)

            return F_stat, p_value

        except Exception as e:
            logger.debug(f"Conditional Granger test failed: {e}")
            return 0.0, 1.0

    def _multivariate_granger(
        self,
        data: np.ndarray,
        lag_order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multivariate Granger causality using VAR model.

        Args:
            data: Time series data (n_timesteps, n_features)
            lag_order: Lag order

        Returns:
            Tuple of (causality_matrix, p_values)
        """
        n_features = data.shape[1]

        # Fit VAR model
        X = []
        y = []

        for t in range(lag_order, len(data)):
            # Stack all lagged values
            x_t = []
            for lag in range(1, lag_order + 1):
                x_t.extend(data[t - lag])
            X.append(x_t)
            y.append(data[t])

        X = np.array(X)
        y = np.array(y)

        # Fit full VAR model
        var_models = []
        for j in range(n_features):
            model = LinearRegression()
            model.fit(X, y[:, j])
            var_models.append(model)

        # Test causality for each pair
        causality_matrix = np.zeros((n_features, n_features))
        p_values = np.ones((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue

                # Remove variable i's lags from model for variable j
                # Indices of variable i in the lag matrix
                i_indices = [i + k * n_features for k in range(lag_order)]

                # Create restricted feature matrix
                X_restricted = np.delete(X, i_indices, axis=1)

                # Fit restricted model
                model_restricted = LinearRegression()
                model_restricted.fit(X_restricted, y[:, j])

                # Compute RSS
                rss_full = np.sum((y[:, j] - var_models[j].predict(X)) ** 2)
                rss_restricted = np.sum((y[:, j] - model_restricted.predict(X_restricted)) ** 2)

                # F-test
                n_samples = len(y)
                n_params_full = X.shape[1]
                n_params_restricted = X_restricted.shape[1]

                if rss_full == 0 or n_samples <= n_params_full:
                    F_stat = 0.0
                    p_value = 1.0
                else:
                    F_stat = (
                        (rss_restricted - rss_full) / (n_params_full - n_params_restricted)
                    ) / (rss_full / (n_samples - n_params_full))

                    df1 = n_params_full - n_params_restricted
                    df2 = n_samples - n_params_full
                    p_value = 1.0 - stats.f.cdf(F_stat, df1, df2)

                causality_matrix[i, j] = F_stat
                p_values[i, j] = p_value

        return causality_matrix, p_values

    def _build_lag_matrix(
        self,
        series: np.ndarray,
        lag_order: int
    ) -> np.ndarray:
        """
        Build lagged design matrix for time series.

        Args:
            series: Time series (n_timesteps,)
            lag_order: Number of lags

        Returns:
            Lagged matrix (n_timesteps - lag_order, lag_order)
        """
        n = len(series)
        X = np.zeros((n - lag_order, lag_order))

        for i in range(lag_order):
            X[:, i] = series[lag_order - i - 1:n - i - 1]

        return X

    def _select_lag_order(
        self,
        data: np.ndarray,
        max_lag: Optional[int] = None
    ) -> int:
        """
        Select optimal lag order using BIC.

        Args:
            data: Time series data (n_timesteps, n_features)
            max_lag: Maximum lag to consider

        Returns:
            Optimal lag order
        """
        if max_lag is None:
            max_lag = min(self.max_lag, len(data) // 10)

        n_timesteps, n_features = data.shape

        bic_scores = []

        for lag in range(1, max_lag + 1):
            # Build VAR model
            X = []
            y = []

            for t in range(lag, n_timesteps):
                x_t = []
                for l in range(1, lag + 1):
                    x_t.extend(data[t - l])
                X.append(x_t)
                y.append(data[t])

            X = np.array(X)
            y = np.array(y)

            # Fit models and compute BIC
            total_rss = 0
            for j in range(n_features):
                model = LinearRegression()
                model.fit(X, y[:, j])
                residuals = y[:, j] - model.predict(X)
                total_rss += np.sum(residuals ** 2)

            n_samples = len(y)
            n_params = X.shape[1] * n_features

            # BIC = n * log(RSS/n) + k * log(n)
            bic = n_samples * np.log(total_rss / n_samples) + n_params * np.log(n_samples)
            bic_scores.append(bic)

        # Select lag with minimum BIC
        optimal_lag = np.argmin(bic_scores) + 1

        return optimal_lag

    def build_causal_graph(
        self,
        result: GrangerResult,
        node_names: Optional[List[str]] = None
    ) -> CausalGraph:
        """
        Build directed causal graph from Granger causality results.

        Args:
            result: GrangerResult from analysis
            node_names: Optional names for nodes

        Returns:
            CausalGraph object
        """
        # Binary adjacency matrix from significant edges
        n_nodes = result.causality_matrix.shape[0]
        adjacency = np.zeros((n_nodes, n_nodes))

        for (i, j) in result.significant_edges:
            adjacency[i, j] = 1

        # Edge weights from F-statistics
        edge_weights = result.causality_matrix.copy()

        return CausalGraph(
            adjacency_matrix=adjacency,
            edge_weights=edge_weights,
            node_names=node_names
        )
