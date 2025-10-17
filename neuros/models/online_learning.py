"""
Online learning and adaptive model updates for non-stationary BCI data.

This module provides models that can adapt to changing signal characteristics
over time, crucial for long-term BCI applications where user states evolve.
"""

from __future__ import annotations

from typing import Optional, Dict, List
import numpy as np
from collections import deque
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from neuros.models.base_model import BaseModel


class OnlineLinearClassifier(BaseModel):
    """
    Online linear classifier using stochastic gradient descent.

    Adapts to new data in real-time using incremental learning.
    Ideal for non-stationary environments where user states change.

    Parameters
    ----------
    learning_rate : str or float
        Learning rate schedule ('optimal', 'constant', 'invscaling', 'adaptive')
    alpha : float
        Regularization parameter (default: 0.0001)
    max_iter : int
        Maximum iterations for initial training (default: 1000)
    window_size : int
        Size of sliding window for recent samples (default: 500)
    """

    def __init__(self, learning_rate: str = 'optimal', alpha: float = 0.0001,
                 max_iter: int = 1000, window_size: int = 500):
        super().__init__()
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.max_iter = max_iter
        self.window_size = window_size

        self.clf = SGDClassifier(
            loss='log_loss',
            learning_rate=learning_rate,
            alpha=alpha,
            max_iter=max_iter,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Keep recent samples for adaptation
        self.recent_X = deque(maxlen=window_size)
        self.recent_y = deque(maxlen=window_size)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initial training on batch data."""
        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Initial fit
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        self.is_trained = True

        # Store initial samples
        for x, label in zip(X, y):
            self.recent_X.append(x)
            self.recent_y.append(label)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained")

        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained")

        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Update model with new data (online learning).

        Parameters
        ----------
        X : np.ndarray
            New feature samples
        y : np.ndarray
            New labels (can be predictions or user feedback)
        """
        if not self.is_fitted:
            # If not yet trained, use as initial training
            self.train(X, y)
            return

        # Update scaler incrementally (running statistics)
        X_scaled = self.scaler.transform(X)

        # Partial fit (online update)
        self.clf.partial_fit(X_scaled, y)

        # Add to recent samples
        for x, label in zip(X, y):
            self.recent_X.append(x)
            self.recent_y.append(label)

    def retrain_on_recent(self) -> None:
        """Retrain on recent samples from sliding window."""
        if len(self.recent_X) == 0:
            return

        X_recent = np.array(list(self.recent_X))
        y_recent = np.array(list(self.recent_y))

        # Refit scaler and model on recent data
        self.scaler.fit(X_recent)
        X_scaled = self.scaler.transform(X_recent)
        self.clf.fit(X_scaled, y_recent)


class AdaptiveEnsemble(BaseModel):
    """
    Adaptive ensemble that weights models based on recent performance.

    Maintains multiple models and dynamically adjusts their weights
    based on validation performance over a sliding window.

    Parameters
    ----------
    base_models : list
        List of base model instances
    window_size : int
        Size of sliding window for performance tracking (default: 100)
    adaptation_rate : float
        Rate of weight adaptation (0-1, default: 0.1)
    """

    def __init__(self, base_models: List[BaseModel], window_size: int = 100,
                 adaptation_rate: float = 0.1):
        super().__init__()
        self.base_models = base_models
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate

        # Initialize equal weights
        self.weights = np.ones(len(base_models)) / len(base_models)

        # Track recent performance
        self.recent_performance = [deque(maxlen=window_size)
                                  for _ in base_models]

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train all base models."""
        for model in self.base_models:
            model.train(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction."""
        if not self.is_trained:
            raise RuntimeError("Models not trained")

        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.base_models])

        # Weighted vote
        n_samples = X.shape[0]
        n_classes = len(np.unique(predictions))

        weighted_probs = np.zeros((n_samples, n_classes))

        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            for j in range(n_samples):
                weighted_probs[j, pred[j]] += weight

        return np.argmax(weighted_probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability prediction."""
        if not self.is_trained:
            raise RuntimeError("Models not trained")

        # Get probabilities from all models
        all_probs = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                all_probs.append(model.predict_proba(X))
            else:
                # Convert predictions to one-hot probabilities
                pred = model.predict(X)
                n_classes = len(np.unique(pred))
                probs = np.zeros((len(pred), n_classes))
                for i, p in enumerate(pred):
                    probs[i, p] = 1.0
                all_probs.append(probs)

        # Weighted average
        weighted_probs = np.average(all_probs, axis=0, weights=self.weights)
        return weighted_probs

    def update_weights(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Update model weights based on recent performance.

        Parameters
        ----------
        X : np.ndarray
            Validation samples
        y_true : np.ndarray
            True labels
        """
        # Compute accuracy for each model
        accuracies = []
        for i, model in enumerate(self.base_models):
            y_pred = model.predict(X)
            acc = np.mean(y_pred == y_true)
            accuracies.append(acc)
            self.recent_performance[i].append(acc)

        # Update weights based on recent average performance
        avg_performance = np.array([
            np.mean(list(perf)) if len(perf) > 0 else 0.5
            for perf in self.recent_performance
        ])

        # Softmax to convert to weights
        exp_perf = np.exp(5 * avg_performance)  # Temperature = 5
        new_weights = exp_perf / np.sum(exp_perf)

        # Smooth update
        self.weights = ((1 - self.adaptation_rate) * self.weights +
                       self.adaptation_rate * new_weights)

        # Renormalize
        self.weights /= np.sum(self.weights)


class AdaptiveThresholdClassifier(BaseModel):
    """
    Classifier with adaptive decision thresholds.

    Adjusts classification thresholds based on recent prediction
    confidence and correctness to handle non-stationary data.

    Parameters
    ----------
    base_model : BaseModel
        Base classifier model
    window_size : int
        Size of adaptation window (default: 50)
    confidence_threshold : float
        Initial confidence threshold (default: 0.6)
    adaptation_rate : float
        Rate of threshold adaptation (default: 0.05)
    """

    def __init__(self, base_model: BaseModel, window_size: int = 50,
                 confidence_threshold: float = 0.6, adaptation_rate: float = 0.05):
        super().__init__()
        self.base_model = base_model
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.adaptation_rate = adaptation_rate

        # Track recent predictions and feedback
        self.recent_confidence = deque(maxlen=window_size)
        self.recent_correct = deque(maxlen=window_size)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train base model."""
        self.base_model.train(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with adaptive thresholds."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        probs = self.base_model.predict_proba(X)
        max_probs = np.max(probs, axis=1)

        # Apply adaptive threshold
        predictions = []
        for i, (prob_dist, confidence) in enumerate(zip(probs, max_probs)):
            if confidence >= self.confidence_threshold:
                # High confidence: use model prediction
                predictions.append(np.argmax(prob_dist))
            else:
                # Low confidence: use conservative strategy (e.g., most common class)
                # or could trigger user feedback
                predictions.append(np.argmax(prob_dist))  # Still use prediction

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.base_model.predict_proba(X)

    def update_threshold(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Update confidence threshold based on recent performance.

        Parameters
        ----------
        X : np.ndarray
            Recent samples
        y_true : np.ndarray
            True labels (from feedback)
        """
        probs = self.base_model.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        y_pred = np.argmax(probs, axis=1)

        correct = (y_pred == y_true)

        # Store recent performance
        for conf, corr in zip(max_probs, correct):
            self.recent_confidence.append(conf)
            self.recent_correct.append(corr)

        if len(self.recent_correct) < 10:
            return

        # Calculate optimal threshold based on recent data
        confidences = np.array(list(self.recent_confidence))
        correctness = np.array(list(self.recent_correct))

        # Find threshold that maximizes accuracy
        thresholds = np.linspace(0.5, 1.0, 20)
        accuracies = []

        for thresh in thresholds:
            high_conf_mask = confidences >= thresh
            if np.sum(high_conf_mask) > 0:
                acc = np.mean(correctness[high_conf_mask])
            else:
                acc = 0.5
            accuracies.append(acc)

        # Update threshold toward optimal
        optimal_idx = np.argmax(accuracies)
        optimal_thresh = thresholds[optimal_idx]

        self.confidence_threshold = ((1 - self.adaptation_rate) *
                                    self.confidence_threshold +
                                    self.adaptation_rate * optimal_thresh)


class IncrementalCSP(BaseEstimator, ClassifierMixin):
    """
    Incremental Common Spatial Patterns for online adaptation.

    Updates CSP filters incrementally as new data arrives,
    allowing adaptation to changing signal characteristics.

    Parameters
    ----------
    n_components : int
        Number of CSP components (default: 4)
    update_interval : int
        Number of new samples before updating filters (default: 50)
    forgetting_factor : float
        Factor for exponential forgetting (0-1, default: 0.95)
    """

    def __init__(self, n_components: int = 4, update_interval: int = 50,
                 forgetting_factor: float = 0.95):
        self.n_components = n_components
        self.update_interval = update_interval
        self.forgetting_factor = forgetting_factor

        self.filters_ = None
        self.cov_0_ = None
        self.cov_1_ = None
        self.n_updates = 0
        self.buffer_X = []
        self.buffer_y = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Initial fit on batch data."""
        from neuros.processing.advanced_features import CommonSpatialPatterns

        csp = CommonSpatialPatterns(n_components=self.n_components)
        csp.fit(X, y)

        self.filters_ = csp.filters_

        # Store initial covariances
        X_0 = X[y == 0]
        X_1 = X[y == 1]

        self.cov_0_ = self._compute_covariance(X_0)
        self.cov_1_ = self._compute_covariance(X_1)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using current CSP filters."""
        if self.filters_ is None:
            raise RuntimeError("IncrementalCSP not fitted")

        n_trials = X.shape[0]
        features = np.zeros((n_trials, self.n_components))

        for i in range(n_trials):
            X_filtered = self.filters_[:self.n_components] @ X[i]
            variances = np.var(X_filtered, axis=1)
            features[i] = variances / np.sum(variances)

        return np.log(features + 1e-8)

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Update CSP filters with new data."""
        # Add to buffer
        self.buffer_X.extend(X)
        self.buffer_y.extend(y)

        # Check if it's time to update
        if len(self.buffer_X) < self.update_interval:
            return self

        # Convert buffer to arrays
        X_new = np.array(self.buffer_X)
        y_new = np.array(self.buffer_y)

        # Update covariances with exponential forgetting
        X_0_new = X_new[y_new == 0]
        X_1_new = X_new[y_new == 1]

        if len(X_0_new) > 0:
            cov_0_new = self._compute_covariance(X_0_new)
            self.cov_0_ = (self.forgetting_factor * self.cov_0_ +
                          (1 - self.forgetting_factor) * cov_0_new)

        if len(X_1_new) > 0:
            cov_1_new = self._compute_covariance(X_1_new)
            self.cov_1_ = (self.forgetting_factor * self.cov_1_ +
                          (1 - self.forgetting_factor) * cov_1_new)

        # Recompute filters
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(self.cov_0_, self.cov_0_ + self.cov_1_)
        ix = np.argsort(eigenvalues)[::-1]
        self.filters_ = eigenvectors[:, ix].T

        # Clear buffer
        self.buffer_X = []
        self.buffer_y = []
        self.n_updates += 1

        return self

    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute average normalized covariance."""
        n_trials, n_channels, n_samples = X.shape
        cov = np.zeros((n_channels, n_channels))

        for trial in X:
            trial_cov = np.cov(trial)
            cov += trial_cov / np.trace(trial_cov)

        return cov / n_trials
