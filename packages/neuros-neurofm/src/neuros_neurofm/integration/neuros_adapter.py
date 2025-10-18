"""
neurOS Integration for NeuroFM-X.

Provides adapter to use NeuroFM-X models within the neurOS real-time pipeline.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class NeuroFMXNeurOSAdapter:
    """Adapter for using NeuroFMX in neurOS pipelines.

    This class wraps a NeuroFMX model to be compatible with the
    neurOS pipeline interface.

    Parameters
    ----------
    model : nn.Module
        NeuroFMX model instance.
    tokenizer_type : str, optional
        Type of tokenizer ("binned", "spike", "lfp").
        Default: "binned".
    device : str, optional
        Device to run model on ("cpu", "cuda", "mps").
        Default: "cpu".
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer_type: str = "binned",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer_type = tokenizer_type
        self.device = device

        # Get tokenizer from model if available
        if hasattr(model, "tokenizer"):
            self.tokenizer = model.tokenizer
        else:
            # Create default tokenizer
            from neuros_neurofm.tokenizers import BinnedTokenizer
            self.tokenizer = BinnedTokenizer(
                n_units=96,  # Default, should be configured
                d_model=256,
            ).to(device)

    def predict(self, neural_data: np.ndarray) -> np.ndarray:
        """Predict behavioral variables from neural data.

        Compatible with neurOS model interface.

        Parameters
        ----------
        neural_data : np.ndarray
            Neural data, shape (seq_length, n_units) for binned data.

        Returns
        -------
        np.ndarray
            Predicted behavioral variables, shape (behavior_dim,).
        """
        # Convert to torch tensor
        if len(neural_data.shape) == 2:
            # Add batch dimension
            neural_data = neural_data[np.newaxis, :]

        data_tensor = torch.tensor(
            neural_data,
            dtype=torch.float32,
            device=self.device,
        )

        # Tokenize
        with torch.no_grad():
            tokens, mask = self.tokenizer(data_tensor)

            # Forward through model
            if hasattr(self.model, "decode_behavior"):
                prediction = self.model.decode_behavior(tokens, mask)
            else:
                # Fall back to manual forward
                latents = self.model.encode(tokens, mask)
                pooled = latents.mean(dim=1)
                prediction = self.model.heads(pooled, task="decoder")

        # Convert to numpy
        prediction_np = prediction.cpu().numpy()

        # Remove batch dimension if single sample
        if prediction_np.shape[0] == 1:
            prediction_np = prediction_np[0]

        return prediction_np

    def __call__(self, neural_data: np.ndarray) -> np.ndarray:
        """Call predict (alias for neurOS compatibility)."""
        return self.predict(neural_data)

    def get_latents(self, neural_data: np.ndarray) -> np.ndarray:
        """Extract latent representations.

        Parameters
        ----------
        neural_data : np.ndarray
            Neural data.

        Returns
        -------
        np.ndarray
            Latent features, shape (n_latents, latent_dim).
        """
        if len(neural_data.shape) == 2:
            neural_data = neural_data[np.newaxis, :]

        data_tensor = torch.tensor(
            neural_data,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            tokens, mask = self.tokenizer(data_tensor)
            latents = self.model.encode(tokens, mask)

        latents_np = latents.cpu().numpy()

        if latents_np.shape[0] == 1:
            latents_np = latents_np[0]

        return latents_np


def create_neuros_model(
    model_path: str,
    tokenizer_type: str = "binned",
    device: str = "cpu",
) -> NeuroFMXNeurOSAdapter:
    """Create neurOS-compatible model from checkpoint.

    Parameters
    ----------
    model_path : str
        Path to NeuroFMX checkpoint.
    tokenizer_type : str, optional
        Tokenizer type.
    device : str, optional
        Device.

    Returns
    -------
    NeuroFMXNeurOSAdapter
        neurOS-compatible model.
    """
    from neuros_neurofm.models.neurofmx_complete import NeuroFMXComplete

    # Load model
    model = NeuroFMXComplete.from_pretrained(model_path)

    # Create adapter
    adapter = NeuroFMXNeurOSAdapter(
        model=model,
        tokenizer_type=tokenizer_type,
        device=device,
    )

    return adapter
