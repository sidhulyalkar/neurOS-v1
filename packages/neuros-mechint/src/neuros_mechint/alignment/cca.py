"""
Canonical Correlation Analysis (CCA) for model-to-brain alignment.

This module implements various CCA methods for finding shared representations
between neural network activations and brain recordings.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union
from sklearn.cross_decomposition import CCA as SklearnCCA
from sklearn.model_selection import KFold
import warnings


class CCA:
    """
    Standard Canonical Correlation Analysis.

    Finds linear projections that maximize correlation between two sets of variables.

    Examples:
        >>> import torch
        >>> # Model activations: (samples, features1)
        >>> X = torch.randn(100, 50)
        >>> # Brain recordings: (samples, features2)
        >>> Y = torch.randn(100, 30)
        >>>
        >>> cca = CCA(n_components=10)
        >>> cca.fit(X, Y)
        >>> correlations = cca.canonical_correlations_
        >>> X_c, Y_c = cca.transform(X, Y)  # Canonical variates
    """

    def __init__(
        self,
        n_components: int = 10,
        reg: float = 0.0,
        device: Optional[str] = None
    ):
        """
        Initialize CCA.

        Args:
            n_components: Number of canonical components to compute
            reg: Regularization parameter (0 = no regularization)
            device: Device for computation ('cuda' or 'cpu')
        """
        self.n_components = n_components
        self.reg = reg
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Learned parameters
        self.weights_x_: Optional[torch.Tensor] = None
        self.weights_y_: Optional[torch.Tensor] = None
        self.canonical_correlations_: Optional[torch.Tensor] = None
        self.mean_x_: Optional[torch.Tensor] = None
        self.mean_y_: Optional[torch.Tensor] = None

    def _to_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor on correct device."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        return X.to(self.device)

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> 'CCA':
        """
        Fit CCA model.

        Args:
            X: First data matrix (n_samples, n_features_x)
            Y: Second data matrix (n_samples, n_features_y)

        Returns:
            self: Fitted CCA object
        """
        X = self._to_tensor(X)
        Y = self._to_tensor(Y)

        n_samples = X.shape[0]
        if Y.shape[0] != n_samples:
            raise ValueError("X and Y must have same number of samples")

        # Center the data
        self.mean_x_ = X.mean(dim=0, keepdim=True)
        self.mean_y_ = Y.mean(dim=0, keepdim=True)

        X_centered = X - self.mean_x_
        Y_centered = Y - self.mean_y_

        # Compute covariance matrices
        Cxx = (X_centered.T @ X_centered) / (n_samples - 1)
        Cyy = (Y_centered.T @ Y_centered) / (n_samples - 1)
        Cxy = (X_centered.T @ Y_centered) / (n_samples - 1)

        # Add regularization
        if self.reg > 0:
            Cxx = Cxx + self.reg * torch.eye(Cxx.shape[0], device=self.device)
            Cyy = Cyy + self.reg * torch.eye(Cyy.shape[0], device=self.device)

        # Solve generalized eigenvalue problem
        # (Cxy @ inv(Cyy) @ Cyx) @ wx = lambda^2 @ Cxx @ wx
        try:
            Cxx_inv_sqrt = torch.linalg.inv(torch.linalg.cholesky(Cxx))
            Cyy_inv_sqrt = torch.linalg.inv(torch.linalg.cholesky(Cyy))
        except RuntimeError:
            # Use pseudo-inverse if singular
            warnings.warn("Covariance matrices are singular, using pseudo-inverse")
            U_x, S_x, Vh_x = torch.linalg.svd(Cxx)
            Cxx_inv_sqrt = (Vh_x.T @ torch.diag(1.0 / torch.sqrt(S_x + 1e-10)) @ U_x.T)

            U_y, S_y, Vh_y = torch.linalg.svd(Cyy)
            Cyy_inv_sqrt = (Vh_y.T @ torch.diag(1.0 / torch.sqrt(S_y + 1e-10)) @ U_y.T)

        # Compute matrix for eigendecomposition
        M = Cxx_inv_sqrt.T @ Cxy @ Cyy_inv_sqrt

        # SVD of M
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Keep top n_components
        n_comp = min(self.n_components, S.shape[0])
        U = U[:, :n_comp]
        S = S[:n_comp]
        Vh = Vh[:n_comp, :]

        # Canonical correlations
        self.canonical_correlations_ = S

        # Canonical weights
        self.weights_x_ = Cxx_inv_sqrt @ U
        self.weights_y_ = Cyy_inv_sqrt @ Vh.T

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transform data to canonical space.

        Args:
            X: First data matrix (n_samples, n_features_x)
            Y: Second data matrix (n_samples, n_features_y), optional

        Returns:
            X_c: Canonical variates for X
            Y_c: Canonical variates for Y (if Y provided)
        """
        if self.weights_x_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._to_tensor(X)
        X_centered = X - self.mean_x_
        X_c = X_centered @ self.weights_x_

        if Y is not None:
            Y = self._to_tensor(Y)
            Y_centered = Y - self.mean_y_
            Y_c = Y_centered @ self.weights_y_
            return X_c, Y_c

        return X_c

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit and transform in one step."""
        self.fit(X, Y)
        return self.transform(X, Y)

    def score(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute mean canonical correlation on new data.

        Args:
            X: First data matrix
            Y: Second data matrix

        Returns:
            Mean canonical correlation
        """
        X_c, Y_c = self.transform(X, Y)

        # Compute correlations for each component
        correlations = []
        for i in range(X_c.shape[1]):
            corr = torch.corrcoef(torch.stack([X_c[:, i], Y_c[:, i]]))[0, 1]
            correlations.append(corr.item())

        return np.mean(correlations)


class RegularizedCCA(CCA):
    """
    Regularized CCA with automatic regularization parameter selection.

    Uses cross-validation to select optimal regularization strength.

    Examples:
        >>> rcca = RegularizedCCA(n_components=10, reg_params=[1e-3, 1e-2, 1e-1])
        >>> rcca.fit(X, Y)
        >>> print(f"Best regularization: {rcca.best_reg_}")
    """

    def __init__(
        self,
        n_components: int = 10,
        reg_params: List[float] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        cv: int = 5,
        device: Optional[str] = None
    ):
        """
        Initialize RegularizedCCA.

        Args:
            n_components: Number of canonical components
            reg_params: List of regularization parameters to try
            cv: Number of cross-validation folds
            device: Device for computation
        """
        super().__init__(n_components=n_components, device=device)
        self.reg_params = reg_params
        self.cv = cv
        self.best_reg_: Optional[float] = None
        self.cv_scores_: Optional[Dict[float, float]] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> 'RegularizedCCA':
        """Fit with cross-validated regularization selection."""
        X = self._to_tensor(X)
        Y = self._to_tensor(Y)

        # Cross-validation for regularization parameter
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        cv_scores = {reg: [] for reg in self.reg_params}

        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()

        for train_idx, val_idx in kf.split(X_np):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            for reg in self.reg_params:
                self.reg = reg
                super().fit(X_train, Y_train)
                score = self.score(X_val, Y_val)
                cv_scores[reg].append(score)

        # Select best regularization
        mean_scores = {reg: np.mean(scores) for reg, scores in cv_scores.items()}
        self.best_reg_ = max(mean_scores, key=mean_scores.get)
        self.cv_scores_ = mean_scores

        # Fit final model with best regularization
        self.reg = self.best_reg_
        super().fit(X, Y)

        return self


class KernelCCA:
    """
    Kernel CCA for non-linear alignment.

    Uses kernel trick to find non-linear canonical correlations.

    Examples:
        >>> kcca = KernelCCA(n_components=10, kernel='rbf', gamma=0.1)
        >>> kcca.fit(X, Y)
        >>> X_c, Y_c = kcca.transform(X, Y)
    """

    def __init__(
        self,
        n_components: int = 10,
        kernel: str = 'rbf',
        gamma: float = 0.1,
        reg: float = 1e-3,
        device: Optional[str] = None
    ):
        """
        Initialize KernelCCA.

        Args:
            n_components: Number of canonical components
            kernel: Kernel type ('rbf', 'linear', 'poly')
            gamma: Kernel coefficient for rbf/poly
            reg: Regularization parameter
            device: Device for computation
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.reg = reg
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.alphas_x_: Optional[torch.Tensor] = None
        self.alphas_y_: Optional[torch.Tensor] = None
        self.canonical_correlations_: Optional[torch.Tensor] = None
        self.X_train_: Optional[torch.Tensor] = None
        self.Y_train_: Optional[torch.Tensor] = None

    def _to_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        return X.to(self.device)

    def _compute_kernel(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute kernel matrix."""
        if Y is None:
            Y = X

        if self.kernel == 'linear':
            return X @ Y.T
        elif self.kernel == 'rbf':
            # Compute pairwise distances
            X_norm = (X ** 2).sum(1).view(-1, 1)
            Y_norm = (Y ** 2).sum(1).view(1, -1)
            dist = X_norm + Y_norm - 2.0 * X @ Y.T
            return torch.exp(-self.gamma * dist)
        elif self.kernel == 'poly':
            return (self.gamma * (X @ Y.T) + 1) ** 2
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> 'KernelCCA':
        """Fit Kernel CCA."""
        X = self._to_tensor(X)
        Y = self._to_tensor(Y)

        self.X_train_ = X
        self.Y_train_ = Y

        n_samples = X.shape[0]

        # Compute kernel matrices
        Kx = self._compute_kernel(X)
        Ky = self._compute_kernel(Y)

        # Center kernel matrices
        N = torch.ones(n_samples, n_samples, device=self.device) / n_samples
        Kx = Kx - N @ Kx - Kx @ N + N @ Kx @ N
        Ky = Ky - N @ Ky - Ky @ N + N @ Ky @ N

        # Add regularization
        Kx = Kx + self.reg * torch.eye(n_samples, device=self.device)
        Ky = Ky + self.reg * torch.eye(n_samples, device=self.device)

        # Solve generalized eigenvalue problem
        Kx_inv_sqrt = torch.linalg.inv(torch.linalg.cholesky(Kx))
        Ky_inv_sqrt = torch.linalg.inv(torch.linalg.cholesky(Ky))

        M = Kx_inv_sqrt.T @ Kx @ Ky @ Kx_inv_sqrt

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(M)

        # Sort by eigenvalue (descending)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep top components
        n_comp = min(self.n_components, eigenvectors.shape[1])
        self.canonical_correlations_ = torch.sqrt(eigenvalues[:n_comp])

        # Compute dual coefficients
        self.alphas_x_ = Kx_inv_sqrt @ eigenvectors[:, :n_comp]

        # For Y
        M_y = Ky_inv_sqrt.T @ Ky @ Kx @ Ky_inv_sqrt
        eigenvalues_y, eigenvectors_y = torch.linalg.eigh(M_y)
        idx_y = torch.argsort(eigenvalues_y, descending=True)
        eigenvectors_y = eigenvectors_y[:, idx_y]
        self.alphas_y_ = Ky_inv_sqrt @ eigenvectors_y[:, :n_comp]

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Transform data to canonical space."""
        if self.alphas_x_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._to_tensor(X)

        # Compute kernel between X and training data
        Kx = self._compute_kernel(X, self.X_train_)
        X_c = Kx @ self.alphas_x_

        if Y is not None:
            Y = self._to_tensor(Y)
            Ky = self._compute_kernel(Y, self.Y_train_)
            Y_c = Ky @ self.alphas_y_
            return X_c, Y_c

        return X_c

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit and transform in one step."""
        self.fit(X, Y)
        return self.transform(X, Y)


class TimeVaryingCCA:
    """
    Time-varying CCA using sliding windows.

    Computes CCA in sliding windows to track how alignment changes over time.

    Examples:
        >>> # Time series data: (samples, time, features)
        >>> X_time = torch.randn(50, 100, 30)
        >>> Y_time = torch.randn(50, 100, 20)
        >>>
        >>> tvcca = TimeVaryingCCA(n_components=5, window_size=20, stride=5)
        >>> results = tvcca.fit_transform(X_time, Y_time)
        >>> print(results['correlations'].shape)  # (n_windows, n_components)
    """

    def __init__(
        self,
        n_components: int = 10,
        window_size: int = 50,
        stride: int = 10,
        reg: float = 1e-3,
        device: Optional[str] = None
    ):
        """
        Initialize TimeVaryingCCA.

        Args:
            n_components: Number of canonical components
            window_size: Size of sliding window (in time steps)
            stride: Stride between windows
            reg: Regularization parameter
            device: Device for computation
        """
        self.n_components = n_components
        self.window_size = window_size
        self.stride = stride
        self.reg = reg
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.window_ccas_: Optional[List[CCA]] = None
        self.window_correlations_: Optional[torch.Tensor] = None

    def _to_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        return X.to(self.device)

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Fit time-varying CCA and transform data.

        Args:
            X: Time series data (n_samples, n_timepoints, n_features_x)
            Y: Time series data (n_samples, n_timepoints, n_features_y)

        Returns:
            Dictionary with:
                - correlations: Canonical correlations per window
                - X_canonical: Transformed X data
                - Y_canonical: Transformed Y data
                - window_times: Center time of each window
        """
        X = self._to_tensor(X)
        Y = self._to_tensor(Y)

        n_samples, n_timepoints, n_features_x = X.shape
        _, _, n_features_y = Y.shape

        # Compute windows
        windows = []
        window_times = []
        for start in range(0, n_timepoints - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append((start, end))
            window_times.append((start + end) / 2)

        n_windows = len(windows)

        # Fit CCA for each window
        self.window_ccas_ = []
        correlations = []
        X_canonical_all = []
        Y_canonical_all = []

        for start, end in windows:
            # Extract window data (flatten samples and time)
            X_window = X[:, start:end, :].reshape(-1, n_features_x)
            Y_window = Y[:, start:end, :].reshape(-1, n_features_y)

            # Fit CCA
            cca = CCA(n_components=self.n_components, reg=self.reg, device=self.device)
            cca.fit(X_window, Y_window)

            # Transform
            X_c, Y_c = cca.transform(X_window, Y_window)

            self.window_ccas_.append(cca)
            correlations.append(cca.canonical_correlations_)
            X_canonical_all.append(X_c.reshape(n_samples, self.window_size, -1))
            Y_canonical_all.append(Y_c.reshape(n_samples, self.window_size, -1))

        self.window_correlations_ = torch.stack(correlations)

        return {
            'correlations': self.window_correlations_,
            'X_canonical': X_canonical_all,
            'Y_canonical': Y_canonical_all,
            'window_times': torch.tensor(window_times, device=self.device)
        }


def select_cca_dimensions(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    max_components: int = 50,
    cv: int = 5,
    device: Optional[str] = None
) -> Dict[str, any]:
    """
    Select optimal number of CCA components via cross-validation.

    Args:
        X: First data matrix (n_samples, n_features_x)
        Y: Second data matrix (n_samples, n_features_y)
        max_components: Maximum number of components to try
        cv: Number of cross-validation folds
        device: Device for computation

    Returns:
        Dictionary with:
            - best_n_components: Optimal number of components
            - cv_scores: Cross-validation scores for each number
            - std_scores: Standard deviation of scores

    Examples:
        >>> result = select_cca_dimensions(X, Y, max_components=20)
        >>> print(f"Best n_components: {result['best_n_components']}")
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()

    # Try different numbers of components
    n_components_range = range(1, min(max_components, X.shape[1], Y.shape[1]) + 1, 2)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = {n: [] for n in n_components_range}

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        for n_comp in n_components_range:
            cca = CCA(n_components=n_comp, reg=1e-3, device=device)
            cca.fit(X_train, Y_train)
            score = cca.score(X_val, Y_val)
            cv_scores[n_comp].append(score)

    # Compute mean and std
    mean_scores = {n: np.mean(scores) for n, scores in cv_scores.items()}
    std_scores = {n: np.std(scores) for n, scores in cv_scores.items()}

    best_n_components = max(mean_scores, key=mean_scores.get)

    return {
        'best_n_components': best_n_components,
        'cv_scores': mean_scores,
        'std_scores': std_scores
    }


if __name__ == "__main__":
    # Example usage
    print("Testing CCA implementations...")

    # Generate synthetic data
    n_samples = 200
    n_features_x = 50
    n_features_y = 30
    n_components = 10

    X = torch.randn(n_samples, n_features_x)
    Y = torch.randn(n_samples, n_features_y)

    # Add correlation
    shared = torch.randn(n_samples, n_components)
    X[:, :n_components] = shared + 0.1 * torch.randn(n_samples, n_components)
    Y[:, :n_components] = shared + 0.1 * torch.randn(n_samples, n_components)

    print("\n1. Standard CCA")
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    print(f"Canonical correlations: {cca.canonical_correlations_[:5].cpu().numpy()}")
    print(f"Score: {cca.score(X, Y):.4f}")

    print("\n2. Regularized CCA")
    rcca = RegularizedCCA(n_components=n_components)
    rcca.fit(X, Y)
    print(f"Best regularization: {rcca.best_reg_}")
    print(f"CV scores: {rcca.cv_scores_}")

    print("\n3. Kernel CCA")
    kcca = KernelCCA(n_components=5, kernel='rbf', gamma=0.01)
    kcca.fit(X[:100], Y[:100])  # Use subset for speed
    X_c, Y_c = kcca.transform(X[:100], Y[:100])
    print(f"Transformed shapes: {X_c.shape}, {Y_c.shape}")

    print("\n4. Time-Varying CCA")
    X_time = torch.randn(20, 100, 30)
    Y_time = torch.randn(20, 100, 20)
    tvcca = TimeVaryingCCA(n_components=5, window_size=20, stride=10)
    results = tvcca.fit_transform(X_time, Y_time)
    print(f"Number of windows: {results['correlations'].shape[0]}")
    print(f"Mean correlation per window: {results['correlations'].mean(dim=1)[:5].cpu().numpy()}")

    print("\n5. Dimension Selection")
    result = select_cca_dimensions(X, Y, max_components=20, cv=3)
    print(f"Best n_components: {result['best_n_components']}")

    print("\nAll CCA tests passed!")
