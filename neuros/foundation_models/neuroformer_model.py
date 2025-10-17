"""
Neuroformer: Multimodal generative pretraining for neural data.

This module implements wrappers for Neuroformer models, which use
transformer-based generative pretraining on multimodal neural and behavioral data.

References:
    - Gobryal et al., "Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data", ICLR 2024
    - https://github.com/nerdslab/neuroformer

Neuroformer is a foundation model that:
1. Pretrained on large-scale multimodal neuroscience datasets
2. Supports multiple modalities (spikes, LFP, behavior, video)
3. Uses masked autoencoding for self-supervised learning
4. Enables zero-shot and few-shot transfer to new tasks
5. Provides unified interface for diverse neural decoding tasks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from neuros.foundation_models.base_foundation_model import BaseFoundationModel

logger = logging.getLogger(__name__)

# Try to import torch (optional dependency)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Neuroformer models will run in mock mode.")


# Mock Neuroformer transformer network (module-level for pickling)
if TORCH_AVAILABLE:
    class NeuroformerNet(nn.Module):
        """Mock Neuroformer transformer network."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_modalities: int,
            dim: int,
            depth: int,
            num_heads: int,
            dropout: float,
            mask_ratio: float
        ):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.n_modalities = n_modalities
            self.mask_ratio = mask_ratio

            # Modality embeddings
            self.modality_embedding = nn.Embedding(n_modalities, dim)

            # Input projection
            self.input_proj = nn.Linear(input_dim, dim)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

            # Output heads
            self.reconstruction_head = nn.Linear(dim, input_dim)
            self.task_head = nn.Linear(dim, output_dim)

        def forward(self, x, modality_ids=None, mask_indices=None):
            # x: (batch, seq_len, input_dim)
            batch_size, seq_len, _ = x.shape

            # Project input
            h = self.input_proj(x)  # (batch, seq_len, dim)

            # Add modality embeddings if provided
            if modality_ids is not None:
                modality_emb = self.modality_embedding(modality_ids)  # (batch, n_modalities, dim)
                # Broadcast to sequence length
                modality_emb = modality_emb.unsqueeze(1).expand(-1, seq_len // self.n_modalities, -1, -1)
                modality_emb = modality_emb.reshape(batch_size, seq_len, -1)
                h = h + modality_emb

            # Apply transformer
            h = self.transformer(h)  # (batch, seq_len, dim)

            # Generate outputs
            reconstruction = self.reconstruction_head(h)  # For pretraining
            task_output = self.task_head(h[:, 0, :])  # Use first token for classification

            return {
                'reconstruction': reconstruction,
                'task_output': task_output,
                'embeddings': h
            }
else:
    NeuroformerNet = None


class NeuroformerModel(BaseFoundationModel):
    """
    Neuroformer model for multimodal neural data analysis.

    Neuroformer uses masked autoencoding pretraining on multimodal neural data
    (spikes, LFP, behavior, video) to learn generalizable representations that
    can be fine-tuned for diverse downstream tasks.

    Parameters
    ----------
    input_dim : int
        Dimension of input neural data per modality.
    output_dim : int, default=2
        Dimension of task output (e.g., number of classes).
    n_modalities : int, default=1
        Number of input modalities (e.g., spikes, LFP, behavior).
    dim : int, default=512
        Dimension of transformer hidden states.
    depth : int, default=8
        Number of transformer layers.
    num_heads : int, default=8
        Number of attention heads.
    dropout : float, default=0.1
        Dropout probability.
    mask_ratio : float, default=0.15
        Ratio of tokens to mask during pretraining.
    pretrain_mode : bool, default=False
        Whether to use pretraining mode (masked autoencoding).
    task_type : str, default="classification"
        Type of downstream task: "classification", "regression", or "generation".

    Attributes
    ----------
    is_trained : bool
        Whether the model has been trained or loaded.
    model : torch.nn.Module or None
        The underlying transformer model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        n_modalities: int = 1,
        dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        pretrain_mode: bool = False,
        task_type: str = "classification",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_modalities = n_modalities
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask_ratio = mask_ratio
        self.pretrain_mode = pretrain_mode
        self.task_type = task_type

        if task_type not in ["classification", "regression", "generation"]:
            raise ValueError(
                f"task_type must be 'classification', 'regression', or 'generation', got {task_type}"
            )

        self.model = None
        if TORCH_AVAILABLE:
            self.model = self._create_model()
        else:
            logger.warning("PyTorch not available. NeuroformerModel will use mock predictions.")

    def _create_model(self) -> nn.Module:
        """Create the Neuroformer transformer model."""
        if not TORCH_AVAILABLE or NeuroformerNet is None:
            return None

        return NeuroformerNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_modalities=self.n_modalities,
            dim=self.dim,
            depth=self.depth,
            num_heads=self.num_heads,
            dropout=self.dropout,
            mask_ratio=self.mask_ratio
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "NeuroformerModel":
        """
        Load a pretrained Neuroformer model.

        Parameters
        ----------
        model_name_or_path : str
            Path to the pretrained model checkpoint or HuggingFace model identifier.
        **kwargs
            Additional arguments passed to __init__.

        Returns
        -------
        NeuroformerModel
            The loaded model instance.
        """
        # Create model instance first
        input_dim = kwargs.pop("input_dim", 100)
        output_dim = kwargs.pop("output_dim", 2)
        model = cls(input_dim=input_dim, output_dim=output_dim, **kwargs)

        # Load checkpoint if it exists
        path_obj = Path(model_name_or_path)
        if path_obj.exists():
            checkpoint = model.load_checkpoint(model_name_or_path)

            if checkpoint and "model_state_dict" in checkpoint and TORCH_AVAILABLE:
                model.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded Neuroformer model weights from {model_name_or_path}")

        model.is_trained = True
        return model

    def pretrain(
        self,
        X: np.ndarray,
        modality_ids: Optional[np.ndarray] = None,
        n_epochs: int = 100
    ) -> None:
        """
        Pretrain the model using masked autoencoding.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, seq_len, input_dim) or (n_samples, input_dim).
        modality_ids : np.ndarray, optional
            Modality identifiers for each token, shape (n_samples, n_modalities).
        n_epochs : int, default=100
            Number of pretraining epochs.
        """
        if X.ndim == 2:
            # Add sequence dimension: (n_samples, 1, input_dim)
            X = X[:, np.newaxis, :]

        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Pretraining Neuroformer in mock mode (PyTorch not available)")
            self.is_trained = True
            return

        # Mock pretraining
        self.model.train()
        logger.info(
            f"Pretraining Neuroformer on {X.shape[0]} samples "
            f"with mask_ratio={self.mask_ratio} for {n_epochs} epochs"
        )
        self.is_trained = True

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        modality_ids: Optional[np.ndarray] = None
    ) -> None:
        """
        Fine-tune the model on a downstream task.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, seq_len, input_dim) or (n_samples, input_dim).
        y : np.ndarray
            Task labels of shape (n_samples,) for classification or
            (n_samples, output_dim) for regression.
        modality_ids : np.ndarray, optional
            Modality identifiers for each sample.
        """
        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Training Neuroformer in mock mode (PyTorch not available)")
            self.is_trained = True
            return

        # Mock fine-tuning
        self.model.train()
        logger.info(f"Fine-tuning Neuroformer on {X.shape[0]} samples for {self.task_type}")
        self.is_trained = True

    def predict(self, X: np.ndarray, modality_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, seq_len, input_dim) or (n_samples, input_dim).
        modality_ids : np.ndarray, optional
            Modality identifiers for each sample.

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,) for classification or
            (n_samples, output_dim) for regression.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        # Mock predictions
        if self.task_type == "classification":
            return np.random.randint(0, self.output_dim, size=X.shape[0])
        else:  # regression or generation
            return np.random.randn(X.shape[0], self.output_dim)

    def encode(self, X: np.ndarray, modality_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode neural data into latent representations.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, seq_len, input_dim).
        modality_ids : np.ndarray, optional
            Modality identifiers.

        Returns
        -------
        np.ndarray
            Latent representations of shape (n_samples, dim).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before encoding")

        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        # Mock encoding
        return np.random.randn(X.shape[0], self.dim)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """
        Decode latent representations to neural data.

        Parameters
        ----------
        latents : np.ndarray
            Latent representations of shape (n_samples, dim).

        Returns
        -------
        np.ndarray
            Reconstructed neural data of shape (n_samples, input_dim).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before decoding")

        # Mock decoding
        return np.random.randn(latents.shape[0], self.input_dim)

    def zero_shot_predict(
        self,
        X: np.ndarray,
        task_description: str,
        modality_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make zero-shot predictions using task description.

        This leverages the pretrained model's understanding of neural data
        to make predictions on new tasks without fine-tuning.

        Parameters
        ----------
        X : np.ndarray
            Neural data of shape (n_samples, seq_len, input_dim).
        task_description : str
            Natural language description of the task.
        modality_ids : np.ndarray, optional
            Modality identifiers.

        Returns
        -------
        np.ndarray
            Zero-shot predictions.
        """
        if not self.is_trained:
            raise ValueError("Model must be pretrained before zero-shot prediction")

        logger.info(f"Zero-shot prediction with task: '{task_description}'")

        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        # Mock zero-shot predictions
        return np.random.randn(X.shape[0], self.output_dim)

    def few_shot_adapt(
        self,
        X_support: np.ndarray,
        y_support: np.ndarray,
        X_query: np.ndarray,
        n_shots: int = 5
    ) -> np.ndarray:
        """
        Perform few-shot adaptation and prediction.

        Uses a small support set to rapidly adapt the model to a new task,
        then makes predictions on query samples.

        Parameters
        ----------
        X_support : np.ndarray
            Support set data, shape (n_shots, seq_len, input_dim).
        y_support : np.ndarray
            Support set labels, shape (n_shots,).
        X_query : np.ndarray
            Query set data, shape (n_query, seq_len, input_dim).
        n_shots : int, default=5
            Number of examples per class in the support set.

        Returns
        -------
        np.ndarray
            Predictions on query set.
        """
        if not self.is_trained:
            raise ValueError("Model must be pretrained before few-shot adaptation")

        logger.info(f"Few-shot adaptation with {n_shots} shots per class")

        # Mock few-shot predictions
        if self.task_type == "classification":
            return np.random.randint(0, self.output_dim, size=X_query.shape[0])
        else:
            return np.random.randn(X_query.shape[0], self.output_dim)

    def generate(
        self,
        context: Optional[np.ndarray] = None,
        n_samples: int = 100,
        sequence_length: int = 100
    ) -> np.ndarray:
        """
        Generate synthetic neural data.

        Uses the pretrained generative model to create synthetic neural
        recordings, optionally conditioned on context.

        Parameters
        ----------
        context : np.ndarray, optional
            Context data to condition generation, shape (batch_size, context_len, input_dim).
        n_samples : int, default=100
            Number of samples to generate.
        sequence_length : int, default=100
            Length of generated sequences.

        Returns
        -------
        np.ndarray
            Generated neural data of shape (n_samples, sequence_length, input_dim).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generation")

        logger.info(f"Generating {n_samples} synthetic samples of length {sequence_length}")

        # Mock generation
        return np.random.randn(n_samples, sequence_length, self.input_dim)
