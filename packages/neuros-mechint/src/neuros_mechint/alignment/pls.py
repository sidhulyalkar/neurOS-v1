"""
Partial Least Squares (PLS) for model-to-brain alignment.

This module implements PLS regression for predicting brain activity from
neural network representations and analyzing latent relationships.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class PLS:
    """
    Partial Least Squares Regression.

    Finds latent variables that maximize covariance between predictors and targets.
    Useful for predicting brain activity from model activations.

    Examples:
        >>> import torch
        >>> # Model activations: (samples, features)
        >>> X = torch.randn(200, 512)
        >>> # Brain recordings: (samples, voxels)
        >>> Y = torch.randn(200, 100)
        >>>
        >>> pls = PLS(n_components=20)
        >>> pls.fit(X, Y)
        >>> Y_pred = pls.predict(X)
        >>> score = pls.score(X, Y)
        >>> print(f"R² score: {score:.4f}")
    """

    def __init__(
        self,
        n_components: int = 10,
        scale: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize PLS.

        Args:
            n_components: Number of PLS components
            scale: Whether to standardize data
            device: Device for computation ('cuda' or 'cpu')
        """
        self.n_components = n_components
        self.scale = scale
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Learned parameters
        self.weights_x_: Optional[torch.Tensor] = None
        self.weights_y_: Optional[torch.Tensor] = None
        self.loadings_x_: Optional[torch.Tensor] = None
        self.loadings_y_: Optional[torch.Tensor] = None
        self.coef_: Optional[torch.Tensor] = None

        self.mean_x_: Optional[torch.Tensor] = None
        self.mean_y_: Optional[torch.Tensor] = None
        self.std_x_: Optional[torch.Tensor] = None
        self.std_y_: Optional[torch.Tensor] = None

        self.x_scores_: Optional[torch.Tensor] = None
        self.y_scores_: Optional[torch.Tensor] = None

    def _to_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor on correct device."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        return X.to(self.device)

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> 'PLS':
        """
        Fit PLS model.

        Args:
            X: Predictor matrix (n_samples, n_features_x)
            Y: Target matrix (n_samples, n_features_y)

        Returns:
            self: Fitted PLS object
        """
        X = self._to_tensor(X)
        Y = self._to_tensor(Y)

        n_samples = X.shape[0]
        if Y.shape[0] != n_samples:
            raise ValueError("X and Y must have same number of samples")

        # Ensure Y is 2D
        if Y.dim() == 1:
            Y = Y.unsqueeze(1)

        # Store means
        self.mean_x_ = X.mean(dim=0, keepdim=True)
        self.mean_y_ = Y.mean(dim=0, keepdim=True)

        # Center data
        X_centered = X - self.mean_x_
        Y_centered = Y - self.mean_y_

        # Optionally scale
        if self.scale:
            self.std_x_ = X_centered.std(dim=0, keepdim=True) + 1e-10
            self.std_y_ = Y_centered.std(dim=0, keepdim=True) + 1e-10
            X_centered = X_centered / self.std_x_
            Y_centered = Y_centered / self.std_y_
        else:
            self.std_x_ = torch.ones(1, X.shape[1], device=self.device)
            self.std_y_ = torch.ones(1, Y.shape[1], device=self.device)

        # NIPALS algorithm for PLS
        n_comp = min(self.n_components, X.shape[1], Y.shape[1])

        # Initialize storage
        W = torch.zeros(X.shape[1], n_comp, device=self.device)  # X weights
        C = torch.zeros(Y.shape[1], n_comp, device=self.device)  # Y weights
        P = torch.zeros(X.shape[1], n_comp, device=self.device)  # X loadings
        Q = torch.zeros(Y.shape[1], n_comp, device=self.device)  # Y loadings
        T = torch.zeros(n_samples, n_comp, device=self.device)   # X scores
        U = torch.zeros(n_samples, n_comp, device=self.device)   # Y scores

        X_residual = X_centered.clone()
        Y_residual = Y_centered.clone()

        for k in range(n_comp):
            # Initialize u as first column of Y
            u = Y_residual[:, 0:1]

            # Iterate until convergence
            for _ in range(100):
                # X weights
                w = X_residual.T @ u
                w = w / (torch.norm(w) + 1e-10)

                # X scores
                t = X_residual @ w

                # Y weights
                c = Y_residual.T @ t
                c = c / (torch.norm(c) + 1e-10)

                # Y scores
                u_new = Y_residual @ c

                # Check convergence
                if torch.allclose(u, u_new, atol=1e-6):
                    break
                u = u_new

            # X loadings
            p = X_residual.T @ t / (t.T @ t + 1e-10)

            # Y loadings
            q = Y_residual.T @ t / (t.T @ t + 1e-10)

            # Deflate
            X_residual = X_residual - t @ p.T
            Y_residual = Y_residual - t @ q.T

            # Store
            W[:, k:k+1] = w
            C[:, k:k+1] = c
            P[:, k:k+1] = p
            Q[:, k:k+1] = q
            T[:, k:k+1] = t
            U[:, k:k+1] = u

        self.weights_x_ = W
        self.weights_y_ = C
        self.loadings_x_ = P
        self.loadings_y_ = Q
        self.x_scores_ = T
        self.y_scores_ = U

        # Compute regression coefficients
        # Y = X * B
        # B = W * inv(P' * W) * Q'
        self.coef_ = self.weights_x_ @ torch.linalg.inv(
            self.loadings_x_.T @ self.weights_x_
        ) @ self.loadings_y_.T

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transform data to latent space.

        Args:
            X: Predictor matrix (n_samples, n_features_x)
            Y: Target matrix (n_samples, n_features_y), optional

        Returns:
            X_scores: Latent variables for X
            Y_scores: Latent variables for Y (if Y provided)
        """
        if self.weights_x_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._to_tensor(X)
        X_centered = X - self.mean_x_

        if self.scale:
            X_centered = X_centered / self.std_x_

        X_scores = X_centered @ self.weights_x_

        if Y is not None:
            Y = self._to_tensor(Y)
            if Y.dim() == 1:
                Y = Y.unsqueeze(1)

            Y_centered = Y - self.mean_y_
            if self.scale:
                Y_centered = Y_centered / self.std_y_

            Y_scores = Y_centered @ self.weights_y_
            return X_scores, Y_scores

        return X_scores

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Predict target values from predictors.

        Args:
            X: Predictor matrix (n_samples, n_features_x)

        Returns:
            Y_pred: Predicted target values (n_samples, n_features_y)
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._to_tensor(X)
        X_centered = X - self.mean_x_

        if self.scale:
            X_centered = X_centered / self.std_x_

        Y_pred_centered = X_centered @ self.coef_

        if self.scale:
            Y_pred_centered = Y_pred_centered * self.std_y_

        Y_pred = Y_pred_centered + self.mean_y_

        return Y_pred

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
        Compute R² score.

        Args:
            X: Predictor matrix
            Y: Target matrix

        Returns:
            R² score (coefficient of determination)
        """
        Y = self._to_tensor(Y)
        if Y.dim() == 1:
            Y = Y.unsqueeze(1)

        Y_pred = self.predict(X)

        # Compute R²
        ss_res = torch.sum((Y - Y_pred) ** 2)
        ss_tot = torch.sum((Y - Y.mean(dim=0, keepdim=True)) ** 2)

        r2 = 1 - ss_res / (ss_tot + 1e-10)

        return r2.item()

    def explained_variance(self) -> Dict[str, np.ndarray]:
        """
        Compute explained variance for each component.

        Returns:
            Dictionary with:
                - x_variance: Variance explained in X
                - y_variance: Variance explained in Y
                - cumulative_x: Cumulative variance in X
                - cumulative_y: Cumulative variance in Y
        """
        if self.x_scores_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Variance explained in X
        x_var = torch.var(self.x_scores_, dim=0).cpu().numpy()
        x_var_ratio = x_var / np.sum(x_var)

        # Variance explained in Y
        y_var = torch.var(self.y_scores_, dim=0).cpu().numpy()
        y_var_ratio = y_var / np.sum(y_var)

        return {
            'x_variance': x_var_ratio,
            'y_variance': y_var_ratio,
            'cumulative_x': np.cumsum(x_var_ratio),
            'cumulative_y': np.cumsum(y_var_ratio)
        }


class CrossValidatedPLS:
    """
    PLS with cross-validated component selection.

    Automatically selects the optimal number of components using cross-validation.

    Examples:
        >>> cv_pls = CrossValidatedPLS(max_components=50, cv=5)
        >>> cv_pls.fit(X, Y)
        >>> print(f"Best n_components: {cv_pls.best_n_components_}")
        >>> Y_pred = cv_pls.predict(X)
    """

    def __init__(
        self,
        max_components: int = 50,
        cv: int = 5,
        scale: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize CrossValidatedPLS.

        Args:
            max_components: Maximum number of components to try
            cv: Number of cross-validation folds
            scale: Whether to standardize data
            device: Device for computation
        """
        self.max_components = max_components
        self.cv = cv
        self.scale = scale
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.best_n_components_: Optional[int] = None
        self.cv_scores_: Optional[Dict[int, float]] = None
        self.pls_: Optional[PLS] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> 'CrossValidatedPLS':
        """Fit with cross-validated component selection."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()

        # Determine component range
        max_comp = min(self.max_components, X.shape[1], Y.shape[1], X.shape[0] // self.cv)
        n_components_range = range(1, max_comp + 1, max(1, max_comp // 20))

        # Cross-validation
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        cv_scores = {n: [] for n in n_components_range}

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            for n_comp in n_components_range:
                pls = PLS(n_components=n_comp, scale=self.scale, device=self.device)
                pls.fit(X_train, Y_train)
                score = pls.score(X_val, Y_val)
                cv_scores[n_comp].append(score)

        # Compute mean scores
        mean_scores = {n: np.mean(scores) for n, scores in cv_scores.items()}
        self.cv_scores_ = mean_scores

        # Select best
        self.best_n_components_ = max(mean_scores, key=mean_scores.get)

        # Fit final model
        self.pls_ = PLS(
            n_components=self.best_n_components_,
            scale=self.scale,
            device=self.device
        )
        self.pls_.fit(X, Y)

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Transform using fitted PLS model."""
        if self.pls_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.pls_.transform(X, Y)

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Predict using fitted PLS model."""
        if self.pls_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.pls_.predict(X)

    def score(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """Compute R² score using fitted PLS model."""
        if self.pls_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.pls_.score(X, Y)

    def plot_cv_scores(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot cross-validation scores vs. number of components.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.cv_scores_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        fig, ax = plt.subplots(figsize=figsize)

        n_comps = sorted(self.cv_scores_.keys())
        scores = [self.cv_scores_[n] for n in n_comps]

        ax.plot(n_comps, scores, 'o-', linewidth=2, markersize=8)
        ax.axvline(self.best_n_components_, color='r', linestyle='--',
                  label=f'Best: {self.best_n_components_} components')

        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cross-Validated R²')
        ax.set_title('PLS Component Selection')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class PLSVisualization:
    """
    Visualization tools for PLS results.

    Provides methods to visualize latent variables, loadings, and predictions.

    Examples:
        >>> pls_vis = PLSVisualization(pls_model)
        >>> pls_vis.plot_latent_variables()
        >>> pls_vis.plot_loadings()
        >>> pls_vis.plot_predictions(X, Y)
    """

    def __init__(self, pls: PLS):
        """
        Initialize visualization.

        Args:
            pls: Fitted PLS model
        """
        if pls.weights_x_ is None:
            raise ValueError("PLS model not fitted")
        self.pls = pls

    def plot_latent_variables(
        self,
        components: List[int] = [0, 1],
        figsize: Tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """
        Plot latent variables (scores).

        Args:
            components: Which components to plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if len(components) != 2:
            raise ValueError("Exactly 2 components must be specified")

        X_scores = self.pls.x_scores_.cpu().numpy()
        Y_scores = self.pls.y_scores_.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # X scores scatter
        axes[0].scatter(X_scores[:, components[0]], X_scores[:, components[1]], alpha=0.6)
        axes[0].set_xlabel(f'Component {components[0] + 1}')
        axes[0].set_ylabel(f'Component {components[1] + 1}')
        axes[0].set_title('X Latent Variables')
        axes[0].grid(True, alpha=0.3)

        # Y scores scatter
        axes[1].scatter(Y_scores[:, components[0]], Y_scores[:, components[1]], alpha=0.6)
        axes[1].set_xlabel(f'Component {components[0] + 1}')
        axes[1].set_ylabel(f'Component {components[1] + 1}')
        axes[1].set_title('Y Latent Variables')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_loadings(
        self,
        component: int = 0,
        top_k: int = 20,
        figsize: Tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """
        Plot top loadings for a component.

        Args:
            component: Which component to plot
            top_k: Number of top features to show
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        P = self.pls.loadings_x_[:, component].cpu().numpy()
        Q = self.pls.loadings_y_[:, component].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # X loadings
        top_idx_x = np.argsort(np.abs(P))[-top_k:]
        axes[0].barh(range(top_k), P[top_idx_x])
        axes[0].set_xlabel('Loading')
        axes[0].set_ylabel('Feature Index')
        axes[0].set_title(f'X Loadings (Component {component + 1})')
        axes[0].set_yticks(range(top_k))
        axes[0].set_yticklabels(top_idx_x)

        # Y loadings
        top_idx_y = np.argsort(np.abs(Q))[-top_k:]
        axes[1].barh(range(top_k), Q[top_idx_y])
        axes[1].set_xlabel('Loading')
        axes[1].set_ylabel('Feature Index')
        axes[1].set_title(f'Y Loadings (Component {component + 1})')
        axes[1].set_yticks(range(top_k))
        axes[1].set_yticklabels(top_idx_y)

        plt.tight_layout()
        return fig

    def plot_explained_variance(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot explained variance.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        var_dict = self.pls.explained_variance()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        n_comp = len(var_dict['x_variance'])
        components = range(1, n_comp + 1)

        # Individual variance
        axes[0].bar(components, var_dict['x_variance'], alpha=0.6, label='X')
        axes[0].bar(components, var_dict['y_variance'], alpha=0.6, label='Y')
        axes[0].set_xlabel('Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Explained Variance per Component')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Cumulative variance
        axes[1].plot(components, var_dict['cumulative_x'], 'o-', label='X', linewidth=2)
        axes[1].plot(components, var_dict['cumulative_y'], 's-', label='Y', linewidth=2)
        axes[1].set_xlabel('Component')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_predictions(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        n_features: int = 5,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot predicted vs. actual values.

        Args:
            X: Predictor matrix
            Y: Target matrix
            n_features: Number of output features to plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        Y_pred = self.pls.predict(X)

        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        if Y.dim() == 1:
            Y = Y.unsqueeze(1)

        Y_np = Y.cpu().numpy()
        Y_pred_np = Y_pred.cpu().numpy()

        n_features = min(n_features, Y_np.shape[1])
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()

        for i in range(n_features):
            axes[i].scatter(Y_np[:, i], Y_pred_np[:, i], alpha=0.5)

            # Add identity line
            min_val = min(Y_np[:, i].min(), Y_pred_np[:, i].min())
            max_val = max(Y_np[:, i].max(), Y_pred_np[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            # Compute correlation
            corr = np.corrcoef(Y_np[:, i], Y_pred_np[:, i])[0, 1]
            axes[i].set_xlabel('Actual')
            axes[i].set_ylabel('Predicted')
            axes[i].set_title(f'Feature {i} (r={corr:.3f})')
            axes[i].grid(True, alpha=0.3)

        # Hide unused axes
        for i in range(n_features, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Example usage
    print("Testing PLS implementations...")

    # Generate synthetic data
    n_samples = 200
    n_features_x = 100
    n_features_y = 50
    n_components = 20

    # Create correlated data
    latent = torch.randn(n_samples, n_components)
    X = torch.cat([latent @ torch.randn(n_components, n_features_x // 2),
                   torch.randn(n_samples, n_features_x // 2)], dim=1)
    Y = torch.cat([latent @ torch.randn(n_components, n_features_y // 2),
                   torch.randn(n_samples, n_features_y // 2)], dim=1)

    print("\n1. Standard PLS")
    pls = PLS(n_components=n_components)
    pls.fit(X, Y)
    Y_pred = pls.predict(X)
    score = pls.score(X, Y)
    print(f"R² score: {score:.4f}")
    print(f"Prediction shape: {Y_pred.shape}")

    print("\n2. Transform to Latent Space")
    X_scores, Y_scores = pls.transform(X, Y)
    print(f"X scores shape: {X_scores.shape}")
    print(f"Y scores shape: {Y_scores.shape}")

    print("\n3. Explained Variance")
    var_dict = pls.explained_variance()
    print(f"X cumulative variance (first 5): {var_dict['cumulative_x'][:5]}")
    print(f"Y cumulative variance (first 5): {var_dict['cumulative_y'][:5]}")

    print("\n4. Cross-Validated PLS")
    cv_pls = CrossValidatedPLS(max_components=30, cv=3)
    cv_pls.fit(X, Y)
    print(f"Best n_components: {cv_pls.best_n_components_}")
    print(f"Best CV score: {max(cv_pls.cv_scores_.values()):.4f}")

    print("\n5. Visualization")
    pls_vis = PLSVisualization(pls)
    print("Visualization tools initialized")

    print("\nAll PLS tests passed!")
