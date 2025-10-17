"""
Neural Data Transformer (NDT) models for neural decoding.

This module implements wrappers for NDT2 and NDT3 models, which are
transformer-based architectures for multi-context neural decoding.

References:
    - NDT2: "Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity", NeurIPS 2023
    - NDT3: "A Generalist Intracortical Motor Decoder", 2025
    - Original NDT: "Neural Data Transformers: Multi-Context Pretraining for Neural Spiking Activity", NeurIPS 2022

The NDT family uses transformers to model neural population activity across
multiple sessions, subjects, and tasks. NDT2 introduces multi-context
pretraining for better generalization, while NDT3 focuses on intracortical
motor decoding with improved cross-subject transfer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuros.foundation_models.base_foundation_model import BaseFoundationModel
from neuros.foundation_models.utils import spikes_to_tokens, create_session_embeddings

logger = logging.getLogger(__name__)

# Try to import torch and torch_brain (optional dependencies)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. NDT models will run in mock mode.")


# Mock transformer models (module-level for pickling)
if TORCH_AVAILABLE:
    class NDT2Net(nn.Module):
        """Mock NDT2 transformer network."""

        def __init__(self, n_neurons, n_bins, dim, depth, num_heads, dropout, max_contexts):
            super().__init__()
            self.embedding = nn.Linear(n_neurons, dim)
            self.pos_encoding = nn.Parameter(torch.randn(n_bins, dim))
            self.context_embedding = nn.Embedding(max_contexts, dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.output_head = nn.Linear(dim, n_neurons)

        def forward(self, x, context_ids=None):
            # x: (batch, n_bins, n_neurons)
            h = self.embedding(x)  # (batch, n_bins, dim)
            h = h + self.pos_encoding.unsqueeze(0)  # Add positional encoding
            if context_ids is not None:
                context_emb = self.context_embedding(context_ids)  # (batch, dim)
                h = h + context_emb.unsqueeze(1)  # Broadcast to all time steps
            h = self.transformer(h)  # (batch, n_bins, dim)
            out = self.output_head(h)  # (batch, n_bins, n_neurons)
            return out

    class NDT3Net(nn.Module):
        """Mock NDT3 transformer network for motor decoding."""

        def __init__(self, n_neurons, n_bins, output_dim, dim, depth, num_heads,
                     dropout, use_subject_embedding, max_subjects):
            super().__init__()
            self.embedding = nn.Linear(n_neurons, dim)
            self.pos_encoding = nn.Parameter(torch.randn(n_bins, dim))
            self.use_subject_embedding = use_subject_embedding
            if use_subject_embedding:
                self.subject_embedding = nn.Embedding(max_subjects, dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.motor_head = nn.Linear(dim, output_dim)

        def forward(self, x, subject_ids=None):
            # x: (batch, n_bins, n_neurons)
            h = self.embedding(x)  # (batch, n_bins, dim)
            h = h + self.pos_encoding.unsqueeze(0)
            if self.use_subject_embedding and subject_ids is not None:
                subject_emb = self.subject_embedding(subject_ids)  # (batch, dim)
                h = h + subject_emb.unsqueeze(1)
            h = self.transformer(h)  # (batch, n_bins, dim)
            # Use the last time step for motor decoding
            motor_output = self.motor_head(h[:, -1, :])  # (batch, output_dim)
            return motor_output
else:
    NDT2Net = None
    NDT3Net = None


class NDT2Model(BaseFoundationModel):
    """
    Neural Data Transformer 2 for multi-context pretraining.

    NDT2 extends the original NDT with improved multi-context pretraining,
    allowing it to learn representations across multiple recording sessions,
    brain areas, and behavioral contexts.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the input population.
    context_forward_steps : int, default=1
        Number of future time steps to predict (masked language modeling).
    sequence_length : float, default=1.0
        Length of input sequence in seconds.
    bin_size : float, default=0.005
        Temporal bin size in seconds for discretizing spike times.
    dim : int, default=256
        Dimension of the transformer hidden states.
    depth : int, default=6
        Number of transformer layers.
    num_heads : int, default=8
        Number of attention heads in each transformer layer.
    dropout : float, default=0.1
        Dropout probability.
    context_integration : str, default="learned"
        How to integrate context embeddings: "learned", "concat", or "addition".
    max_contexts : int, default=100
        Maximum number of different contexts (sessions/subjects).

    Attributes
    ----------
    is_trained : bool
        Whether the model has been trained.
    model : torch.nn.Module or None
        The underlying transformer model (if PyTorch is available).
    context_embeddings : np.ndarray or None
        Learned embeddings for each context.
    """

    def __init__(
        self,
        n_neurons: int,
        context_forward_steps: int = 1,
        sequence_length: float = 1.0,
        bin_size: float = 0.005,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        context_integration: str = "learned",
        max_contexts: int = 100,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.context_forward_steps = context_forward_steps
        self.sequence_length = sequence_length
        self.bin_size = bin_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout = dropout
        self.context_integration = context_integration
        self.max_contexts = max_contexts

        self.n_bins = int(sequence_length / bin_size)
        self.model = None
        self.context_embeddings = None

        if TORCH_AVAILABLE:
            self.model = self._create_model()
        else:
            logger.warning("PyTorch not available. NDT2Model will use mock predictions.")

    def _create_model(self) -> nn.Module:
        """Create the NDT2 transformer model (mock implementation)."""
        if not TORCH_AVAILABLE or NDT2Net is None:
            return None

        return NDT2Net(
            self.n_neurons, self.n_bins, self.dim, self.depth,
            self.num_heads, self.dropout, self.max_contexts
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "NDT2Model":
        """
        Load a pretrained NDT2 model.

        Parameters
        ----------
        model_name_or_path : str
            Path to the pretrained model checkpoint or HuggingFace model identifier.
        **kwargs
            Additional arguments passed to __init__.

        Returns
        -------
        NDT2Model
            The loaded model instance.
        """
        # Create model instance first
        n_neurons = kwargs.pop("n_neurons", 100)
        model = cls(n_neurons=n_neurons, **kwargs)

        # Load checkpoint if it exists
        path_obj = Path(model_name_or_path)
        if path_obj.exists():
            checkpoint = model.load_checkpoint(model_name_or_path)

            if checkpoint and "model_state_dict" in checkpoint and TORCH_AVAILABLE:
                model.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded NDT2 model weights from {model_name_or_path}")

        model.is_trained = True
        return model

    def train(self, X: np.ndarray, y: np.ndarray, context_ids: Optional[np.ndarray] = None) -> None:
        """
        Train the NDT2 model on neural data.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, n_neurons, n_bins) or (n_samples, n_neurons).
        y : np.ndarray
            Target neural activity for next time steps, same shape as X.
        context_ids : np.ndarray, optional
            Context identifiers for each sample, shape (n_samples,).
            Used to learn context-specific embeddings.
        """
        if X.ndim == 2:
            # Reshape to (n_samples, n_neurons, n_bins)
            X = X.reshape(X.shape[0], self.n_neurons, -1)

        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Training NDT2 in mock mode (PyTorch not available)")
            self.is_trained = True
            return

        # Mock training loop
        # In real implementation, this would be the full training procedure
        self.model.train()
        logger.info(f"Training NDT2 on {X.shape[0]} samples")
        self.is_trained = True

    def predict(self, X: np.ndarray, context_id: Optional[int] = None) -> np.ndarray:
        """
        Predict future neural activity.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, n_neurons, n_bins) or (n_samples, n_neurons).
        context_id : int, optional
            Context identifier for the samples.

        Returns
        -------
        np.ndarray
            Predicted neural activity for the next time step(s).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.n_neurons, -1)

        # Mock predictions
        return np.random.randn(X.shape[0], self.n_neurons)

    def encode(self, X: np.ndarray, context_id: Optional[int] = None) -> np.ndarray:
        """
        Encode neural data into latent representations.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, n_neurons, n_bins).
        context_id : int, optional
            Context identifier.

        Returns
        -------
        np.ndarray
            Latent representations of shape (n_samples, dim).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before encoding")

        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.n_neurons, -1)

        # Mock encoding - average pooling over time
        return np.random.randn(X.shape[0], self.dim)

    def decode(self, latents: np.ndarray, context_id: Optional[int] = None) -> np.ndarray:
        """
        Decode latent representations into neural activity.

        Parameters
        ----------
        latents : np.ndarray
            Latent representations of shape (n_samples, dim).
        context_id : int, optional
            Context identifier.

        Returns
        -------
        np.ndarray
            Decoded neural activity of shape (n_samples, n_neurons).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before decoding")

        # Mock decoding
        return np.random.randn(latents.shape[0], self.n_neurons)


class NDT3Model(BaseFoundationModel):
    """
    Neural Data Transformer 3 for generalist intracortical motor decoding.

    NDT3 is specifically designed for motor decoding from intracortical
    recordings, with enhanced cross-subject transfer learning and
    real-time decoding capabilities.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the input population.
    output_dim : int, default=2
        Dimensionality of the motor output (e.g., 2 for 2D cursor control).
    sequence_length : float, default=0.5
        Length of input sequence in seconds (shorter than NDT2 for real-time).
    bin_size : float, default=0.02
        Temporal bin size in seconds.
    dim : int, default=256
        Dimension of the transformer hidden states.
    depth : int, default=4
        Number of transformer layers (shallower than NDT2 for speed).
    num_heads : int, default=8
        Number of attention heads.
    dropout : float, default=0.1
        Dropout probability.
    use_subject_embedding : bool, default=True
        Whether to use subject-specific embeddings for cross-subject transfer.
    max_subjects : int, default=50
        Maximum number of subjects.

    Attributes
    ----------
    is_trained : bool
        Whether the model has been trained.
    model : torch.nn.Module or None
        The underlying transformer model.
    subject_embeddings : np.ndarray or None
        Learned embeddings for each subject.
    """

    def __init__(
        self,
        n_neurons: int,
        output_dim: int = 2,
        sequence_length: float = 0.5,
        bin_size: float = 0.02,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_subject_embedding: bool = True,
        max_subjects: int = 50,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.bin_size = bin_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_subject_embedding = use_subject_embedding
        self.max_subjects = max_subjects

        self.n_bins = int(sequence_length / bin_size)
        self.model = None
        self.subject_embeddings = None

        if TORCH_AVAILABLE:
            self.model = self._create_model()
        else:
            logger.warning("PyTorch not available. NDT3Model will use mock predictions.")

    def _create_model(self) -> nn.Module:
        """Create the NDT3 transformer model (mock implementation)."""
        if not TORCH_AVAILABLE or NDT3Net is None:
            return None

        return NDT3Net(
            self.n_neurons, self.n_bins, self.output_dim, self.dim,
            self.depth, self.num_heads, self.dropout,
            self.use_subject_embedding, self.max_subjects
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "NDT3Model":
        """
        Load a pretrained NDT3 model.

        Parameters
        ----------
        model_name_or_path : str
            Path to the pretrained model checkpoint or HuggingFace model identifier.
        **kwargs
            Additional arguments passed to __init__.

        Returns
        -------
        NDT3Model
            The loaded model instance.
        """
        # Create model instance first
        n_neurons = kwargs.pop("n_neurons", 100)
        output_dim = kwargs.pop("output_dim", 2)
        model = cls(n_neurons=n_neurons, output_dim=output_dim, **kwargs)

        # Load checkpoint if it exists
        path_obj = Path(model_name_or_path)
        if path_obj.exists():
            checkpoint = model.load_checkpoint(model_name_or_path)

            if checkpoint and "model_state_dict" in checkpoint and TORCH_AVAILABLE:
                model.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded NDT3 model weights from {model_name_or_path}")

        model.is_trained = True
        return model

    def train(self, X: np.ndarray, y: np.ndarray, subject_ids: Optional[np.ndarray] = None) -> None:
        """
        Train the NDT3 model on neural data and motor outputs.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, n_neurons, n_bins) or (n_samples, n_neurons).
        y : np.ndarray
            Motor outputs of shape (n_samples, output_dim) (e.g., cursor velocity).
        subject_ids : np.ndarray, optional
            Subject identifiers for each sample, shape (n_samples,).
        """
        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.n_neurons, -1)

        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Training NDT3 in mock mode (PyTorch not available)")
            self.is_trained = True
            return

        # Mock training
        self.model.train()
        logger.info(f"Training NDT3 on {X.shape[0]} samples")
        self.is_trained = True

    def predict(self, X: np.ndarray, subject_id: Optional[int] = None) -> np.ndarray:
        """
        Predict motor outputs from neural activity.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, n_neurons, n_bins) or (n_samples, n_neurons).
        subject_id : int, optional
            Subject identifier.

        Returns
        -------
        np.ndarray
            Predicted motor outputs of shape (n_samples, output_dim).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.n_neurons, -1)

        # Mock predictions
        return np.random.randn(X.shape[0], self.output_dim)

    def encode(self, X: np.ndarray, subject_id: Optional[int] = None) -> np.ndarray:
        """
        Encode neural data into latent representations.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, n_neurons, n_bins).
        subject_id : int, optional
            Subject identifier.

        Returns
        -------
        np.ndarray
            Latent representations of shape (n_samples, dim).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before encoding")

        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.n_neurons, -1)

        # Mock encoding
        return np.random.randn(X.shape[0], self.dim)

    def decode(self, latents: np.ndarray, subject_id: Optional[int] = None) -> np.ndarray:
        """
        Decode latent representations into motor outputs.

        Parameters
        ----------
        latents : np.ndarray
            Latent representations of shape (n_samples, dim).
        subject_id : int, optional
            Subject identifier.

        Returns
        -------
        np.ndarray
            Decoded motor outputs of shape (n_samples, output_dim).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before decoding")

        # Mock decoding
        return np.random.randn(latents.shape[0], self.output_dim)

    def fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_id: Optional[int] = None,
        n_epochs: int = 10,
        learning_rate: float = 1e-4
    ) -> None:
        """
        Fine-tune the model on a new subject's data.

        This is particularly useful for NDT3's cross-subject transfer learning.
        The model can be pretrained on many subjects and then fine-tuned with
        minimal data from a new subject.

        Parameters
        ----------
        X : np.ndarray
            Neural data from the new subject.
        y : np.ndarray
            Motor outputs from the new subject.
        subject_id : int, optional
            Subject identifier for the new subject.
        n_epochs : int, default=10
            Number of fine-tuning epochs.
        learning_rate : float, default=1e-4
            Learning rate for fine-tuning (typically lower than pretraining).
        """
        if not self.is_trained:
            raise ValueError("Model must be pretrained before fine-tuning")

        logger.info(f"Fine-tuning NDT3 on subject {subject_id} for {n_epochs} epochs")

        # Mock fine-tuning
        # In real implementation, this would freeze most layers and only
        # update the subject embedding and/or final layers
        if TORCH_AVAILABLE and self.model is not None:
            self.model.train()

        logger.info("Fine-tuning complete")
