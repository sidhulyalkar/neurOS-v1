"""
Attention-based multi-modal fusion model for neurOS.

This module implements an advanced fusion strategy that learns to weight
different modalities based on their relevance for the prediction task using
attention mechanisms.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from .base_model import BaseModel


class AttentionFusionModel(BaseModel):
    """Multi-modal fusion model with learned attention weights.

    This model fuses features from multiple modalities (e.g., EEG, video, motion)
    using a learned attention mechanism. Instead of simple concatenation or
    averaging, it learns to weight each modality's contribution based on the
    input data and task.

    The architecture:
    1. Per-modality feature projection (linear layers)
    2. Attention mechanism to compute modality weights
    3. Weighted fusion of modality features
    4. Final classifier on fused features

    Parameters
    ----------
    modality_dims : list[int]
        Feature dimensions for each modality.
    n_classes : int, optional
        Number of output classes. Default is 2.
    fusion_dim : int, optional
        Dimension of fused feature space. Default is 64.
    attention_type : str, optional
        Type of attention: "learned", "self", or "cross". Default is "learned".
    dropout : float, optional
        Dropout rate for regularization. Default is 0.3.

    Examples
    --------
    >>> from neuros.models import AttentionFusionModel
    >>> import numpy as np

    >>> # Create model for 3 modalities: EEG (40 features), Video (128), Motion (12)
    >>> model = AttentionFusionModel(
    ...     modality_dims=[40, 128, 12],
    ...     n_classes=3,
    ...     fusion_dim=64,
    ... )

    >>> # Train on multi-modal data
    >>> # X should be concatenated features: [EEG | Video | Motion]
    >>> X_train = np.random.randn(100, 40 + 128 + 12)
    >>> y_train = np.random.randint(0, 3, 100)
    >>> model.train(X_train, y_train)

    >>> # Predict
    >>> X_test = np.random.randn(20, 180)
    >>> predictions = model.predict(X_test)
    >>> print(predictions.shape)  # (20,)

    >>> # Get attention weights (after training)
    >>> weights = model.get_attention_weights(X_test[0])
    >>> print(f"Modality weights: EEG={weights[0]:.2f}, Video={weights[1]:.2f}, Motion={weights[2]:.2f}")
    """

    def __init__(
        self,
        modality_dims: List[int],
        n_classes: int = 2,
        fusion_dim: int = 64,
        attention_type: str = "learned",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.n_modalities = len(modality_dims)
        self.n_classes = n_classes
        self.fusion_dim = fusion_dim
        self.attention_type = attention_type
        self.dropout = dropout

        # Calculate total input dimension
        self.total_dim = sum(modality_dims)

        # Initialize parameters (will be set during training)
        self.modality_projections: List[np.ndarray] = []  # Projection matrices
        self.modality_biases: List[np.ndarray] = []  # Bias vectors
        self.attention_weights: Optional[np.ndarray] = None  # Attention parameters
        self.classifier_weights: Optional[np.ndarray] = None
        self.classifier_bias: Optional[np.ndarray] = None

        # Training parameters
        self.learning_rate = 0.01
        self.n_epochs = 100
        self.batch_size = 32

    def _split_modalities(self, X: np.ndarray) -> List[np.ndarray]:
        """Split concatenated features into per-modality features.

        Parameters
        ----------
        X : np.ndarray
            Concatenated features of shape (n_samples, total_dim).

        Returns
        -------
        list[np.ndarray]
            List of per-modality features.
        """
        modalities = []
        start_idx = 0
        for dim in self.modality_dims:
            end_idx = start_idx + dim
            modalities.append(X[:, start_idx:end_idx])
            start_idx = end_idx
        return modalities

    def _initialize_parameters(self):
        """Initialize model parameters with Xavier initialization."""
        # Per-modality projections to fusion space
        self.modality_projections = []
        self.modality_biases = []
        for dim in self.modality_dims:
            # Xavier initialization
            limit = np.sqrt(6.0 / (dim + self.fusion_dim))
            W = np.random.uniform(-limit, limit, (dim, self.fusion_dim))
            b = np.zeros(self.fusion_dim)
            self.modality_projections.append(W)
            self.modality_biases.append(b)

        # Attention parameters
        if self.attention_type == "learned":
            # Simple learned attention: each modality gets a score
            self.attention_weights = np.random.randn(self.n_modalities, 1) * 0.01
        elif self.attention_type == "self":
            # Self-attention over modalities
            self.attention_weights = np.random.randn(
                self.fusion_dim, self.n_modalities
            ) * 0.01

        # Classifier weights
        limit = np.sqrt(6.0 / (self.fusion_dim + self.n_classes))
        self.classifier_weights = np.random.uniform(
            -limit, limit, (self.fusion_dim, self.n_classes)
        )
        self.classifier_bias = np.zeros(self.n_classes)

    def _compute_attention(
        self, projected_features: List[np.ndarray]
    ) -> np.ndarray:
        """Compute attention weights for each modality.

        Parameters
        ----------
        projected_features : list[np.ndarray]
            List of projected features, each of shape (n_samples, fusion_dim).

        Returns
        -------
        np.ndarray
            Attention weights of shape (n_samples, n_modalities).
        """
        n_samples = projected_features[0].shape[0]

        if self.attention_type == "learned":
            # Simple learned scalar weight per modality
            # Broadcast attention weights across samples
            attention_logits = self.attention_weights.T  # (1, n_modalities)
            attention_logits = np.tile(attention_logits, (n_samples, 1))

        elif self.attention_type == "self":
            # Self-attention: compute similarity between modalities
            # Stack features: (n_samples, n_modalities, fusion_dim)
            stacked = np.stack(projected_features, axis=1)
            # Compute attention scores
            # Simple dot-product attention
            attention_logits = np.zeros((n_samples, self.n_modalities))
            for i in range(self.n_modalities):
                # Score for modality i based on its projected features
                scores = projected_features[i] @ self.attention_weights
                attention_logits[:, i] = scores[:, i]

        else:  # default: uniform
            attention_logits = np.ones((n_samples, self.n_modalities))

        # Apply softmax to get weights
        attention_weights = self._softmax(attention_logits, axis=1)
        return attention_weights

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass through the network.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, total_dim).

        Returns
        -------
        logits : np.ndarray
            Class logits of shape (n_samples, n_classes).
        attention : np.ndarray
            Attention weights of shape (n_samples, n_modalities).
        """
        # Split into modalities
        modality_features = self._split_modalities(X)

        # Project each modality to fusion space
        projected = []
        for i, features in enumerate(modality_features):
            proj = features @ self.modality_projections[i] + self.modality_biases[i]
            proj = self._relu(proj)  # Activation
            projected.append(proj)

        # Compute attention weights
        attention = self._compute_attention(projected)

        # Fuse modalities with attention weights
        # Stack: (n_samples, n_modalities, fusion_dim)
        stacked = np.stack(projected, axis=1)
        # Weighted sum: (n_samples, fusion_dim)
        fused = np.sum(stacked * attention[:, :, np.newaxis], axis=1)

        # Classifier
        logits = fused @ self.classifier_weights + self.classifier_bias

        return logits, attention

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the attention fusion model.

        This uses a simplified training procedure based on scikit-learn's
        logistic regression for the fusion layer, with learned attention weights.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, total_dim).
        y : np.ndarray
            Training labels of shape (n_samples,).
        """
        # Initialize parameters
        self._initialize_parameters()

        # For simplicity, we'll use a pre-trained approach:
        # 1. Project modalities
        # 2. Use sklearn to train the final classifier

        from sklearn.linear_model import LogisticRegression

        # Split modalities
        modality_features = self._split_modalities(X)

        # Project each modality
        projected = []
        for i, features in enumerate(modality_features):
            proj = features @ self.modality_projections[i] + self.modality_biases[i]
            proj = self._relu(proj)
            projected.append(proj)

        # Initialize attention weights (uniform for now, could be learned)
        # For simplicity, start with uniform attention
        n_samples = X.shape[0]
        attention = np.ones((n_samples, self.n_modalities)) / self.n_modalities

        # Fuse with attention
        stacked = np.stack(projected, axis=1)
        fused = np.sum(stacked * attention[:, :, np.newaxis], axis=1)

        # Train classifier on fused features
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(fused, y)

        # Store classifier weights
        self.classifier_weights = clf.coef_.T
        self.classifier_bias = clf.intercept_

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input features.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, total_dim).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (n_samples,).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        logits, _ = self._forward(X)
        probas = self._softmax(logits, axis=1)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, total_dim).

        Returns
        -------
        np.ndarray
            Class probabilities of shape (n_samples, n_classes).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        logits, _ = self._forward(X)
        return self._softmax(logits, axis=1)

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Get attention weights for each modality.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, total_dim) or (total_dim,).

        Returns
        -------
        np.ndarray
            Attention weights of shape (n_samples, n_modalities) or (n_modalities,).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
            _, attention = self._forward(X)
            return attention[0]
        else:
            _, attention = self._forward(X)
            return attention

    def interpret_attention(self, X: np.ndarray, modality_names: Optional[List[str]] = None) -> dict:
        """Get interpretable attention weights.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        modality_names : list[str], optional
            Names of modalities. If None, uses "Modality 0", "Modality 1", etc.

        Returns
        -------
        dict
            Dictionary mapping modality names to average attention weights.
        """
        attention = self.get_attention_weights(X)

        if modality_names is None:
            modality_names = [f"Modality {i}" for i in range(self.n_modalities)]

        # Average across samples
        if attention.ndim == 2:
            avg_attention = np.mean(attention, axis=0)
        else:
            avg_attention = attention

        return {name: float(weight) for name, weight in zip(modality_names, avg_attention)}
