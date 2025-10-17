"""
Time series alignment utilities using Dynamic Time Warping (DTW).

Provides piecewise linear time warping for aligning neural recordings across
trials, sessions, and subjects. Based on affine warp methods for robust
temporal alignment of neural time series.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class WarpResult:
    """Results from time warping alignment.

    Parameters
    ----------
    warped_data : np.ndarray
        Aligned data after warping.
    warp_functions : list of callable
        Warping functions applied to each trial.
    knot_points : np.ndarray, optional
        Knot points for piecewise linear warping.
    alignment_cost : float, optional
        Final alignment cost/loss.
    """

    warped_data: np.ndarray
    warp_functions: List[Callable]
    knot_points: Optional[np.ndarray] = None
    alignment_cost: Optional[float] = None


def piecewise_linear_warp(
    X: np.ndarray,
    warp_params: np.ndarray,
    n_knots: int = 5,
) -> np.ndarray:
    """Apply piecewise linear time warping to data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_timepoints, n_features) or (n_timepoints,).
    warp_params : np.ndarray
        Warping parameters (shifts at each knot point).
    n_knots : int, default=5
        Number of knot points for piecewise warping.

    Returns
    -------
    np.ndarray
        Warped data with same shape as input.

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> warp_params = np.array([0, 5, -3, 2, 0])  # shifts at 5 knots
    >>> X_warped = piecewise_linear_warp(X, warp_params, n_knots=5)
    """
    n_timepoints = len(X)

    # Create knot points evenly spaced across time
    knot_indices = np.linspace(0, n_timepoints - 1, n_knots)

    # Compute cumulative warp at each knot
    # warp_params represents the shift at each knot
    cumulative_warp = np.cumsum(warp_params)

    # Add boundary conditions (start and end fixed)
    time_original = np.linspace(0, n_timepoints - 1, n_timepoints)

    # Interpolate warp function
    warp_function = interp1d(
        knot_indices,
        knot_indices + cumulative_warp,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate',
    )

    # Apply warping
    warped_time = warp_function(time_original)

    # Clip to valid range
    warped_time = np.clip(warped_time, 0, n_timepoints - 1)

    # Interpolate data at warped time points
    if X.ndim == 1:
        # 1D data
        data_interp = interp1d(
            time_original, X, kind='linear', bounds_error=False, fill_value=0
        )
        X_warped = data_interp(warped_time)
    else:
        # Multi-dimensional data
        X_warped = np.zeros_like(X)
        for feat in range(X.shape[1]):
            data_interp = interp1d(
                time_original, X[:, feat], kind='linear',
                bounds_error=False, fill_value=0
            )
            X_warped[:, feat] = data_interp(warped_time)

    return X_warped


def compute_alignment_loss(
    X_list: List[np.ndarray],
    warp_params_list: List[np.ndarray],
    n_knots: int = 5,
    template: Optional[np.ndarray] = None,
) -> float:
    """Compute alignment loss for a set of warped trials.

    The loss encourages trials to be similar after warping.

    Parameters
    ----------
    X_list : list of np.ndarray
        List of trials to align.
    warp_params_list : list of np.ndarray
        Warping parameters for each trial.
    n_knots : int, default=5
        Number of knot points.
    template : np.ndarray, optional
        Target template to align to. If None, use mean of warped trials.

    Returns
    -------
    float
        Alignment loss (lower is better).
    """
    # Apply warping to each trial
    warped_trials = []
    for X, warp_params in zip(X_list, warp_params_list):
        X_warped = piecewise_linear_warp(X, warp_params, n_knots=n_knots)
        warped_trials.append(X_warped)

    # Stack trials
    warped_stack = np.stack(warped_trials, axis=0)

    # Compute template if not provided
    if template is None:
        template = np.mean(warped_stack, axis=0)

    # Compute loss as sum of squared differences from template
    loss = 0.0
    for X_warped in warped_trials:
        loss += np.sum((X_warped - template) ** 2)

    # Add smoothness penalty on warp parameters
    smoothness_penalty = 0.1
    for warp_params in warp_params_list:
        # Penalize large changes in warp
        loss += smoothness_penalty * np.sum(warp_params ** 2)

    return loss


def align_trials(
    X_list: List[np.ndarray],
    n_knots: int = 5,
    max_iter: int = 100,
    template: Optional[np.ndarray] = None,
    return_template: bool = False,
) -> WarpResult:
    """Align multiple trials using piecewise linear time warping.

    Parameters
    ----------
    X_list : list of np.ndarray
        List of trials to align, each of shape (n_timepoints, n_features).
    n_knots : int, default=5
        Number of knot points for piecewise warping.
    max_iter : int, default=100
        Maximum iterations for optimization.
    template : np.ndarray, optional
        Fixed template to align to. If None, jointly optimize template and warps.
    return_template : bool, default=False
        Whether to include the template in the result.

    Returns
    -------
    WarpResult
        Results containing warped data and warp functions.

    Examples
    --------
    >>> # Generate synthetic trials with temporal jitter
    >>> trials = []
    >>> for i in range(10):
    ...     t = np.linspace(0, 2*np.pi, 100)
    ...     trial = np.sin(t + np.random.randn()*0.3)[:, np.newaxis]
    ...     trials.append(trial)
    >>> result = align_trials(trials, n_knots=5)
    >>> print(f"Aligned {len(trials)} trials, cost: {result.alignment_cost:.2f}")
    """
    n_trials = len(X_list)

    # Initialize warp parameters (start with identity warping)
    warp_params_list = [np.zeros(n_knots) for _ in range(n_trials)]

    logger.info(f"Aligning {n_trials} trials with {n_knots} knots")

    # Flatten parameters for optimization
    def pack_params(warp_params_list):
        return np.concatenate(warp_params_list)

    def unpack_params(flat_params):
        return [flat_params[i*n_knots:(i+1)*n_knots] for i in range(n_trials)]

    # Define objective function
    def objective(flat_params):
        warp_params_list = unpack_params(flat_params)
        return compute_alignment_loss(X_list, warp_params_list, n_knots, template)

    # Initial parameters
    initial_params = pack_params(warp_params_list)

    # Optimize
    logger.info("Optimizing alignment...")
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False},
    )

    # Extract optimized parameters
    optimal_params = unpack_params(result.x)

    # Apply optimal warping
    warped_trials = []
    warp_functions = []

    for X, warp_params in zip(X_list, optimal_params):
        X_warped = piecewise_linear_warp(X, warp_params, n_knots=n_knots)
        warped_trials.append(X_warped)

        # Store warp function
        n_timepoints = len(X)
        knot_indices = np.linspace(0, n_timepoints - 1, n_knots)
        cumulative_warp = np.cumsum(warp_params)

        warp_fn = interp1d(
            knot_indices,
            knot_indices + cumulative_warp,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate',
        )
        warp_functions.append(warp_fn)

    warped_data = np.stack(warped_trials, axis=0)

    # Compute knot points for reference
    knot_points = np.linspace(0, len(X_list[0]) - 1, n_knots)

    logger.info(f"Alignment complete. Final cost: {result.fun:.2f}")

    warp_result = WarpResult(
        warped_data=warped_data,
        warp_functions=warp_functions,
        knot_points=knot_points,
        alignment_cost=result.fun,
    )

    return warp_result


def dynamic_time_warping_distance(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute DTW distance between two time series.

    Classic dynamic programming DTW for measuring similarity between
    temporal sequences that may vary in speed.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape (n_timepoints,) or (n_timepoints, n_features).
    y : np.ndarray
        Second time series, shape (m_timepoints,) or (m_timepoints, n_features).

    Returns
    -------
    distance : float
        DTW distance.
    path : np.ndarray
        Optimal alignment path, shape (path_length, 2).

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 2*np.pi, 100))
    >>> y = np.sin(np.linspace(0, 2*np.pi, 120))  # Different length
    >>> distance, path = dynamic_time_warping_distance(x, y)
    >>> print(f"DTW distance: {distance:.3f}")
    """
    # Ensure 2D
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    n, m = len(x), len(y)

    # Initialize cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sum((x[i - 1] - y[j - 1]) ** 2)
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Backtrack to find path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append([i - 1, j - 1])

        # Find minimum of three predecessors
        candidates = [dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]]
        argmin = np.argmin(candidates)

        if argmin == 0:
            i -= 1
        elif argmin == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()
    path = np.array(path)

    distance = np.sqrt(dtw[n, m])

    return distance, path


def apply_warp_to_new_data(
    X_new: np.ndarray,
    warp_function: Callable,
) -> np.ndarray:
    """Apply a learned warp function to new data.

    Parameters
    ----------
    X_new : np.ndarray
        New data to warp, shape (n_timepoints, n_features).
    warp_function : callable
        Warp function from align_trials().

    Returns
    -------
    np.ndarray
        Warped data.

    Examples
    --------
    >>> # Learn alignment from training trials
    >>> result = align_trials(train_trials)
    >>> # Apply to test trial
    >>> test_warped = apply_warp_to_new_data(test_trial, result.warp_functions[0])
    """
    n_timepoints = len(X_new)
    time_original = np.linspace(0, n_timepoints - 1, n_timepoints)

    # Apply warp function
    warped_time = warp_function(time_original)
    warped_time = np.clip(warped_time, 0, n_timepoints - 1)

    # Interpolate
    if X_new.ndim == 1:
        data_interp = interp1d(
            time_original, X_new, kind='linear', bounds_error=False, fill_value=0
        )
        X_warped = data_interp(warped_time)
    else:
        X_warped = np.zeros_like(X_new)
        for feat in range(X_new.shape[1]):
            data_interp = interp1d(
                time_original, X_new[:, feat], kind='linear',
                bounds_error=False, fill_value=0
            )
            X_warped[:, feat] = data_interp(warped_time)

    return X_warped


def estimate_template(
    X_list: List[np.ndarray],
    method: str = 'mean',
) -> np.ndarray:
    """Estimate template from multiple trials.

    Parameters
    ----------
    X_list : list of np.ndarray
        List of trials.
    method : str, default='mean'
        Method for template estimation ('mean', 'median', 'pca').

    Returns
    -------
    np.ndarray
        Template time series.

    Examples
    --------
    >>> trials = [np.random.randn(100, 10) for _ in range(20)]
    >>> template = estimate_template(trials, method='median')
    """
    # Stack trials
    X_stack = np.stack(X_list, axis=0)

    if method == 'mean':
        template = np.mean(X_stack, axis=0)
    elif method == 'median':
        template = np.median(X_stack, axis=0)
    elif method == 'pca':
        # Use first principal component as template
        from sklearn.decomposition import PCA

        n_trials, n_timepoints, n_features = X_stack.shape
        X_reshaped = X_stack.reshape(n_trials, n_timepoints * n_features)

        pca = PCA(n_components=1)
        pca.fit(X_reshaped)

        template = pca.components_[0].reshape(n_timepoints, n_features)
    else:
        raise ValueError(f"Unknown method: {method}")

    return template
