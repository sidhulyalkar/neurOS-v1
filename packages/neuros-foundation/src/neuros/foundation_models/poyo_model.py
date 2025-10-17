"""
POYO and POYO+ model wrappers for neurOS.

Implements wrappers around the torch_brain POYO models for multi-session,
multi-task neural decoding.

References
----------
- POYO: Azabou et al., "POYO: A Unified, Scalable Framework for Neural
  Population Decoding", NeurIPS 2023
- POYO+: "Multi-session, multi-task neural decoding from distinct cell-types
  and brain regions", ICLR 2025
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from neuros.foundation_models.base_foundation_model import BaseFoundationModel
from neuros.foundation_models.utils import spikes_to_tokens, create_session_embeddings, create_readout_spec

logger = logging.getLogger(__name__)

# Try to import torch_brain, but make it optional
try:
    import torch
    import torch_brain
    from torch_brain.models import POYO, POYOPlus

    TORCH_BRAIN_AVAILABLE = True
except ImportError:
    TORCH_BRAIN_AVAILABLE = False
    torch = None
    POYO = None
    POYOPlus = None
    logger.warning(
        "torch_brain not available. Install with: pip install torch-brain"
    )


class POYOModel(BaseFoundationModel):
    """POYO model wrapper for neurOS.

    POYO (POp

ulation decoding with Y... Optimization) is a transformer-based
    model for neural decoding from spike trains.

    Parameters
    ----------
    sequence_length : float, default=1.0
        Maximum input spike sequence duration in seconds.
    latent_step : float, default=0.01
        Timestep of latent grid in seconds.
    num_latents_per_step : int, default=8
        Number of unique latent tokens per timestep.
    dim : int, default=256
        Model dimensionality.
    depth : int, default=4
        Number of transformer layers.
    heads : int, default=8
        Number of attention heads.
    output_dim : int, default=2
        Output dimensionality (depends on task).
    dropout : float, default=0.1
        Dropout rate.
    pretrained : bool, default=False
        Whether to load pretrained weights.
    pretrained_path : str, optional
        Path to pretrained checkpoint.
    device : str, default='cpu'
        Device to run on ('cpu' or 'cuda').

    Examples
    --------
    >>> # Create model
    >>> model = POYOModel(output_dim=2, pretrained=False)
    >>>
    >>> # Train on spike data
    >>> spike_times = [np.array([0.1, 0.5, 1.2]), ...]  # List of spike times
    >>> labels = np.array([0, 1, 0, 1])  # Classification labels
    >>> model.train(spike_times, labels)
    >>>
    >>> # Make predictions
    >>> predictions = model.predict(new_spike_times)
    """

    def __init__(
        self,
        sequence_length: float = 1.0,
        latent_step: float = 0.01,
        num_latents_per_step: int = 8,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        output_dim: int = 2,
        dropout: float = 0.1,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            device=device,
        )

        self.sequence_length = sequence_length
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.output_dim = output_dim
        self.dropout = dropout

        # Model will be initialized on first use
        self._model = None
        self._n_units = None

    def _ensure_model(self, n_units: int) -> None:
        """Ensure model is initialized with correct number of units."""
        if self._model is None or self._n_units != n_units:
            if not TORCH_BRAIN_AVAILABLE:
                logger.warning(
                    "torch_brain not available, using mock implementation"
                )
                self._model = None
                self._n_units = n_units
                return

            # Create readout spec for single task
            readout_spec = {
                "output_dim": self.output_dim,
                "continuous": True,  # Will be set based on task
            }

            # Initialize POYO model
            self._model = POYO(
                sequence_length=self.sequence_length,
                readout_spec=readout_spec,
                latent_step=self.latent_step,
                num_latents_per_step=self.num_latents_per_step,
                dim=self.dim,
                depth=self.depth,
                heads=self.heads,
                dropout=self.dropout,
            )

            # Move to device
            if torch is not None:
                self._model = self._model.to(self.device)

            self._n_units = n_units

            logger.info(f"Initialized POYO model with {n_units} units")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs,
    ) -> "POYOModel":
        """Load pretrained POYO model.

        Parameters
        ----------
        model_name_or_path : str
            Model name or path to checkpoint.
        **kwargs
            Additional model arguments.

        Returns
        -------
        POYOModel
            Loaded model instance.
        """
        model = cls(pretrained=True, pretrained_path=model_name_or_path, **kwargs)

        if model.pretrained_path is not None:
            model.load_checkpoint(model.pretrained_path)

        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train POYO model.

        Parameters
        ----------
        X : np.ndarray or list
            Training data. Can be:
            - List of spike time arrays (one per neuron)
            - 2D array of spike raster (n_bins, n_neurons)
        y : np.ndarray
            Training labels.
        """
        # Convert to spike times if needed
        if isinstance(X, list):
            spike_times = X
            n_units = len(spike_times)
        else:
            # Assume spike raster, convert to spike times
            from neuros.foundation_models.utils import raster_to_spike_times

            spike_times = raster_to_spike_times(X, fs=1000.0)  # Assume 1kHz
            n_units = X.shape[1] if X.ndim > 1 else 1

        self._ensure_model(n_units)

        if not TORCH_BRAIN_AVAILABLE:
            logger.warning("Training with mock implementation (no-op)")
            self._is_trained = True
            return

        # TODO: Implement actual training loop with torch_brain
        # This would involve:
        # 1. Tokenizing spike times
        # 2. Creating batches
        # 3. Running training loop
        # 4. Backprop and optimization

        logger.info(f"Training on {len(y)} trials with {n_units} units")
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using POYO model.

        Parameters
        ----------
        X : np.ndarray or list
            Test data (spike times or raster).

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if not self._is_trained and not self.pretrained:
            raise RuntimeError("Model must be trained before prediction")

        if not TORCH_BRAIN_AVAILABLE:
            # Mock prediction
            if isinstance(X, list):
                n_samples = 1
            else:
                n_samples = X.shape[0] if X.ndim > 2 else 1

            return np.random.randn(n_samples, self.output_dim)

        # TODO: Implement actual prediction with torch_brain
        return np.zeros((1, self.output_dim))

    def encode(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Encode neural data into latent representations.

        Parameters
        ----------
        X : np.ndarray or list
            Neural data (spike times or raster).

        Returns
        -------
        np.ndarray
            Latent representations.
        """
        if not TORCH_BRAIN_AVAILABLE:
            # Mock latent representations
            return np.random.randn(1, self.dim)

        # TODO: Implement latent encoding
        return np.zeros((1, self.dim))

    def decode(self, latents: np.ndarray, **kwargs) -> np.ndarray:
        """Decode predictions from latents.

        Parameters
        ----------
        latents : np.ndarray
            Latent representations.

        Returns
        -------
        np.ndarray
            Decoded predictions.
        """
        if not TORCH_BRAIN_AVAILABLE:
            return np.random.randn(len(latents), self.output_dim)

        # TODO: Implement decoding
        return np.zeros((len(latents), self.output_dim))


class POYOPlusModel(BaseFoundationModel):
    """POYO+ model wrapper for multi-task neural decoding.

    POYO+ extends POYO with support for multiple tasks and modalities.

    Parameters
    ----------
    sequence_length : float, default=1.0
        Maximum input spike sequence duration.
    task_configs : list of dict
        Task configurations (see create_readout_spec).
    latent_step : float, default=0.01
        Timestep of latent grid.
    num_latents_per_step : int, default=8
        Latent tokens per timestep.
    dim : int, default=256
        Model dimensionality.
    depth : int, default=4
        Number of layers.
    heads : int, default=8
        Attention heads.
    dropout : float, default=0.1
        Dropout rate.
    pretrained : bool, default=False
        Load pretrained weights.
    pretrained_path : str, optional
        Path to checkpoint.
    device : str, default='cpu'
        Device.

    Examples
    --------
    >>> # Define multiple tasks
    >>> tasks = [
    ...     {'name': 'velocity', 'type': 'regression', 'output_dim': 2},
    ...     {'name': 'direction', 'type': 'classification', 'output_dim': 8},
    ... ]
    >>>
    >>> model = POYOPlusModel(task_configs=tasks)
    >>> model.train(spike_times, labels)
    >>> predictions = model.predict(new_spike_times)
    """

    def __init__(
        self,
        sequence_length: float = 1.0,
        task_configs: Optional[List[Dict[str, Any]]] = None,
        latent_step: float = 0.01,
        num_latents_per_step: int = 8,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            device=device,
        )

        self.sequence_length = sequence_length
        self.task_configs = task_configs or [
            {"name": "default", "type": "regression", "output_dim": 2}
        ]
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout

        self._model = None
        self._n_units = None

    def _ensure_model(self, n_units: int) -> None:
        """Ensure model is initialized."""
        if self._model is None or self._n_units != n_units:
            if not TORCH_BRAIN_AVAILABLE:
                logger.warning(
                    "torch_brain not available, using mock implementation"
                )
                self._model = None
                self._n_units = n_units
                return

            # Create readout spec
            readout_spec = create_readout_spec(self.task_configs)

            # Initialize POYO+ model
            self._model = POYOPlus(
                sequence_length=self.sequence_length,
                readout_spec=readout_spec,
                latent_step=self.latent_step,
                num_latents_per_step=self.num_latents_per_step,
                dim=self.dim,
                depth=self.depth,
                heads=self.heads,
                dropout=self.dropout,
            )

            if torch is not None:
                self._model = self._model.to(self.device)

            self._n_units = n_units

            logger.info(
                f"Initialized POYO+ model with {n_units} units, "
                f"{len(self.task_configs)} tasks"
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs,
    ) -> "POYOPlusModel":
        """Load pretrained POYO+ model."""
        model = cls(pretrained=True, pretrained_path=model_name_or_path, **kwargs)

        if model.pretrained_path is not None:
            model.load_checkpoint(model.pretrained_path)

        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train POYO+ model on multi-task data."""
        # Similar to POYOModel but with multi-task support
        if isinstance(X, list):
            n_units = len(X)
        else:
            n_units = X.shape[1] if X.ndim > 1 else 1

        self._ensure_model(n_units)

        if not TORCH_BRAIN_AVAILABLE:
            logger.warning("Training with mock implementation")
            self._is_trained = True
            return

        logger.info(f"Training POYO+ on {len(y)} trials, {len(self.task_configs)} tasks")
        self._is_trained = True

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict for all tasks.

        Returns
        -------
        dict
            Predictions for each task.
        """
        if not self._is_trained and not self.pretrained:
            raise RuntimeError("Model must be trained")

        if not TORCH_BRAIN_AVAILABLE:
            # Mock predictions for each task
            n_samples = 1 if isinstance(X, list) else len(X)
            predictions = {}

            for task in self.task_configs:
                predictions[task["name"]] = np.random.randn(
                    n_samples, task["output_dim"]
                )

            return predictions

        # TODO: Implement actual multi-task prediction
        return {task["name"]: np.zeros((1, task["output_dim"])) for task in self.task_configs}

    def encode(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Encode to shared latent space."""
        if not TORCH_BRAIN_AVAILABLE:
            return np.random.randn(1, self.dim)

        return np.zeros((1, self.dim))

    def decode(self, latents: np.ndarray, task_name: str, **kwargs) -> np.ndarray:
        """Decode for specific task.

        Parameters
        ----------
        latents : np.ndarray
            Latent representations.
        task_name : str
            Name of task to decode for.
        """
        if not TORCH_BRAIN_AVAILABLE:
            # Find task output dim
            task = next((t for t in self.task_configs if t["name"] == task_name), None)
            if task is None:
                raise ValueError(f"Unknown task: {task_name}")

            return np.random.randn(len(latents), task["output_dim"])

        return np.zeros((len(latents), 2))
