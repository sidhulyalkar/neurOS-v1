"""
Base class for foundation models.

Provides common interface and utilities for integrating large-scale pretrained
neural decoding models into neurOS.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np

from neuros.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BaseFoundationModel(BaseModel):
    """Abstract base class for foundation models.

    Foundation models are large-scale pretrained models that can be fine-tuned
    for specific tasks. They typically:
    - Support transfer learning across sessions, subjects, brain regions
    - Handle multi-modal inputs (spikes, LFP, calcium imaging, etc.)
    - Provide rich latent representations
    - Support multi-task decoding

    Parameters
    ----------
    pretrained : bool, default=False
        Whether to load pretrained weights.
    pretrained_path : str, optional
        Path to pretrained model checkpoint.
    device : str, default='cpu'
        Device to run model on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.device = device
        self._is_trained = False

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs,
    ) -> "BaseFoundationModel":
        """Load a pretrained foundation model.

        Parameters
        ----------
        model_name_or_path : str
            Model name (e.g., 'poyo-base') or path to checkpoint.
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        BaseFoundationModel
            Loaded model instance.

        Examples
        --------
        >>> model = POYOModel.from_pretrained('poyo-base')
        >>> predictions = model.predict(neural_data)
        """
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        X: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Encode neural data into latent representations.

        Parameters
        ----------
        X : np.ndarray
            Neural data (format depends on specific model).
        **kwargs
            Model-specific encoding arguments.

        Returns
        -------
        np.ndarray
            Latent representations.

        Examples
        --------
        >>> latents = model.encode(spike_times)
        >>> print(f"Latent shape: {latents.shape}")
        """
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        latents: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Decode predictions from latent representations.

        Parameters
        ----------
        latents : np.ndarray
            Latent representations from encode().
        **kwargs
            Model-specific decoding arguments.

        Returns
        -------
        np.ndarray
            Decoded predictions.
        """
        raise NotImplementedError

    def fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_epochs: int = 10,
        learning_rate: float = 1e-4,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fine-tune pretrained model on new data.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Training labels.
        n_epochs : int, default=10
            Number of fine-tuning epochs.
        learning_rate : float, default=1e-4
            Learning rate for fine-tuning.
        **kwargs
            Model-specific fine-tuning arguments.

        Returns
        -------
        dict
            Training history and metrics.

        Examples
        --------
        >>> history = model.fine_tune(new_session_data, labels, n_epochs=5)
        >>> print(f"Final loss: {history['loss'][-1]:.4f}")
        """
        # Default implementation: just call train()
        logger.warning(
            f"{self.__class__.__name__} does not implement fine_tune(), "
            "falling back to train()"
        )
        self.train(X, y)
        return {"message": "Used train() method instead of fine_tune()"}

    def save_checkpoint(
        self,
        path: str,
        *,
        save_optimizer: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        path : str
            Path to save checkpoint.
        save_optimizer : bool, default=True
            Whether to save optimizer state.
        metadata : dict, optional
            Additional metadata to save.
        """
        import pickle

        checkpoint = {
            "model_state": self.__dict__,
            "metadata": metadata or {},
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(
        self,
        path: str,
        *,
        load_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """Load model checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint.
        load_optimizer : bool, default=True
            Whether to load optimizer state.

        Returns
        -------
        dict
            Checkpoint metadata.
        """
        import pickle
        from pathlib import Path

        path_obj = Path(path)

        if not path_obj.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return {}

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        # Restore model state
        for key, value in checkpoint["model_state"].items():
            if hasattr(self, key):
                setattr(self, key, value)

        logger.info(f"Loaded checkpoint from {path}")

        return checkpoint.get("metadata", {})

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns
        -------
        dict
            Model configuration dictionary.
        """
        return {
            "model_class": self.__class__.__name__,
            "pretrained": self.pretrained,
            "pretrained_path": self.pretrained_path,
            "device": self.device,
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        config = self.get_config()
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        return f"{self.__class__.__name__}({config_str})"
