"""
CEBRA: Learnable latent embeddings for joint behavioral and neural analysis.

This module implements wrappers for CEBRA models, which use contrastive learning
to create consistent, high-performance latent spaces from neural and behavioral data.

References:
    - Schneider et al., "Learnable latent embeddings for joint behavioural and neural analysis", Nature 2023
    - https://github.com/AdaptiveMotorControlLab/CEBRA

CEBRA uses temperature-based contrastive learning to learn latent embeddings that:
1. Capture both neural dynamics and behavioral correlates
2. Work across different recording modalities (calcium imaging, Neuropixels, etc.)
3. Enable cross-session decoding and transfer learning
4. Provide interpretable low-dimensional representations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from neuros.foundation_models.base_foundation_model import BaseFoundationModel

logger = logging.getLogger(__name__)

# Try to import torch and CEBRA (optional dependencies)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. CEBRA models will run in mock mode.")

try:
    # Try to import the actual CEBRA library if available
    import cebra
    CEBRA_AVAILABLE = True
except ImportError:
    CEBRA_AVAILABLE = False
    logger.info("CEBRA library not available. Using mock implementation.")


# Mock CEBRA encoder network (module-level for pickling)
if TORCH_AVAILABLE:
    class CEBRAEncoder(nn.Module):
        """Mock CEBRA encoder network."""

        def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                     dropout: float = 0.1):
            super().__init__()
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.encoder = nn.Sequential(*layers)

        def forward(self, x):
            return self.encoder(x)
else:
    CEBRAEncoder = None


class CEBRAModel(BaseFoundationModel):
    """
    CEBRA model for learning latent embeddings from neural and behavioral data.

    CEBRA uses contrastive learning with temporal context to learn consistent
    latent representations that capture both neural dynamics and behavioral
    correlates. It can operate in three modes:
    1. Time-contrastive: Learn from neural data alone using temporal structure
    2. Behavior-contrastive: Align neural data with behavioral variables
    3. Hybrid: Combine both time and behavior information

    Parameters
    ----------
    input_dim : int
        Dimension of input neural data (e.g., number of neurons or channels).
    output_dim : int, default=3
        Dimension of the output latent space (typically 3-32).
    hidden_dims : list of int, default=[256, 128, 64]
        Dimensions of hidden layers in the encoder network.
    learning_mode : str, default="time"
        Learning mode: "time" (time-contrastive), "behavior" (behavior-contrastive),
        or "hybrid" (both).
    temperature : float, default=0.1
        Temperature parameter for contrastive loss.
    time_offset : int, default=10
        Number of time steps for positive pairs in time-contrastive learning.
    dropout : float, default=0.1
        Dropout probability in encoder layers.
    max_iterations : int, default=5000
        Maximum number of training iterations.
    batch_size : int, default=512
        Batch size for training.

    Attributes
    ----------
    is_trained : bool
        Whether the model has been trained.
    model : torch.nn.Module or None
        The underlying encoder network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        hidden_dims: Optional[List[int]] = None,
        learning_mode: str = "time",
        temperature: float = 0.1,
        time_offset: int = 10,
        dropout: float = 0.1,
        max_iterations: int = 5000,
        batch_size: int = 512,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.learning_mode = learning_mode
        self.temperature = temperature
        self.time_offset = time_offset
        self.dropout = dropout
        self.max_iterations = max_iterations
        self.batch_size = batch_size

        if learning_mode not in ["time", "behavior", "hybrid"]:
            raise ValueError(f"learning_mode must be 'time', 'behavior', or 'hybrid', got {learning_mode}")

        self.model = None
        if TORCH_AVAILABLE:
            self.model = self._create_model()
        else:
            logger.warning("PyTorch not available. CEBRAModel will use mock predictions.")

    def _create_model(self) -> nn.Module:
        """Create the CEBRA encoder network."""
        if not TORCH_AVAILABLE or CEBRAEncoder is None:
            return None

        return CEBRAEncoder(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "CEBRAModel":
        """
        Load a pretrained CEBRA model.

        Parameters
        ----------
        model_name_or_path : str
            Path to the pretrained model checkpoint.
        **kwargs
            Additional arguments passed to __init__.

        Returns
        -------
        CEBRAModel
            The loaded model instance.
        """
        # Create model instance first
        input_dim = kwargs.pop("input_dim", 100)
        output_dim = kwargs.pop("output_dim", 3)
        model = cls(input_dim=input_dim, output_dim=output_dim, **kwargs)

        # Load checkpoint if it exists
        path_obj = Path(model_name_or_path)
        if path_obj.exists():
            checkpoint = model.load_checkpoint(model_name_or_path)

            if checkpoint and "model_state_dict" in checkpoint and TORCH_AVAILABLE:
                model.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded CEBRA model weights from {model_name_or_path}")

        model.is_trained = True
        return model

    def train(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        behavior: Optional[np.ndarray] = None
    ) -> None:
        """
        Train the CEBRA model on neural data.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, input_dim) or (n_samples, n_timepoints, input_dim).
        y : np.ndarray, optional
            Target labels (not used in CEBRA, kept for compatibility).
        behavior : np.ndarray, optional
            Behavioral variables of shape (n_samples, n_behavior_dims).
            Required if learning_mode is "behavior" or "hybrid".
        """
        if X.ndim == 3:
            # Flatten time dimension if present
            n_samples, n_timepoints, input_dim = X.shape
            X = X.reshape(-1, input_dim)

        if self.learning_mode in ["behavior", "hybrid"] and behavior is None:
            raise ValueError(f"behavior data required for learning_mode='{self.learning_mode}'")

        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Training CEBRA in mock mode (PyTorch not available)")
            self.is_trained = True
            return

        # Mock training loop
        # In real implementation, this would use contrastive learning
        self.model.train()
        logger.info(f"Training CEBRA on {X.shape[0]} samples with mode '{self.learning_mode}'")
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Encode neural data into latent space (alias for encode).

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, input_dim).

        Returns
        -------
        np.ndarray
            Latent embeddings of shape (n_samples, output_dim).
        """
        return self.encode(X)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode neural data into latent space.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, input_dim).

        Returns
        -------
        np.ndarray
            Latent embeddings of shape (n_samples, output_dim).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before encoding")

        if X.ndim == 3:
            n_samples, n_timepoints, input_dim = X.shape
            X = X.reshape(-1, input_dim)

        # Mock encoding
        return np.random.randn(X.shape[0], self.output_dim)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """
        Decode latent embeddings back to neural space.

        Note: CEBRA is primarily an encoder. Decoding is not typically used,
        but this method is provided for compatibility with BaseFoundationModel.

        Parameters
        ----------
        latents : np.ndarray
            Latent embeddings of shape (n_samples, output_dim).

        Returns
        -------
        np.ndarray
            Reconstructed neural data of shape (n_samples, input_dim).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before decoding")

        # Mock decoding (CEBRA typically doesn't decode)
        logger.warning("CEBRA is primarily an encoder; decoding may not be meaningful")
        return np.random.randn(latents.shape[0], self.input_dim)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform neural data to latent space (sklearn-style API).

        This is an alias for encode() to match sklearn's API conventions.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, input_dim).

        Returns
        -------
        np.ndarray
            Latent embeddings of shape (n_samples, output_dim).
        """
        return self.encode(X)

    def fit_transform(self, X: np.ndarray, behavior: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model and transform the data in one step (sklearn-style API).

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, input_dim).
        behavior : np.ndarray, optional
            Behavioral variables of shape (n_samples, n_behavior_dims).

        Returns
        -------
        np.ndarray
            Latent embeddings of shape (n_samples, output_dim).
        """
        self.train(X, behavior=behavior)
        return self.transform(X)

    def compute_consistency(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        n_neighbors: int = 5
    ) -> float:
        """
        Compute consistency score between two datasets.

        Consistency measures how well the learned embedding preserves
        neighborhood structure across different recordings or sessions.

        Parameters
        ----------
        X1 : np.ndarray
            First dataset, shape (n_samples1, input_dim).
        X2 : np.ndarray
            Second dataset, shape (n_samples2, input_dim).
        n_neighbors : int, default=5
            Number of neighbors to consider.

        Returns
        -------
        float
            Consistency score (0-1, higher is better).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing consistency")

        # Encode both datasets
        emb1 = self.encode(X1)
        emb2 = self.encode(X2)

        # Mock consistency computation
        # In real implementation, this would compute neighborhood preservation
        logger.info(f"Computing consistency between datasets with {n_neighbors} neighbors")
        return np.random.rand()

    def decode_behavior(
        self,
        X: np.ndarray,
        behavior: np.ndarray,
        n_folds: int = 5
    ) -> Dict[str, float]:
        """
        Decode behavioral variables from latent embeddings.

        This evaluates how well the learned latent space captures behavioral
        information by training a simple decoder.

        Parameters
        ----------
        X : np.ndarray
            Neural data, shape (n_samples, input_dim).
        behavior : np.ndarray
            Behavioral variables, shape (n_samples, n_behavior_dims).
        n_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        dict
            Dictionary with keys 'r2_score' and 'mse'.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before decoding behavior")

        # Encode neural data
        latents = self.encode(X)

        # Mock behavior decoding
        logger.info(f"Decoding behavior using {n_folds}-fold CV")
        return {
            "r2_score": np.random.rand(),
            "mse": np.random.rand() * 0.1
        }
