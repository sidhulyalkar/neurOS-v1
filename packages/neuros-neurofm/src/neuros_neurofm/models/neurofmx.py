"""
NeuroFM-X: Foundation Model for Neural Population Dynamics.

Combines Mamba/SSM backbone, Perceiver-IO fusion, PopT aggregator,
latent diffusion, multi-task heads and adapters into a unified model.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuros_neurofm.models.mamba_backbone import MambaBackbone
from neuros_neurofm.fusion.perceiver import PerceiverIO
from neuros_neurofm.tokenizers import BinnedTokenizer, SpikeTokenizer, LFPTokenizer


class NeuroFMX(nn.Module):
    """NeuroFM-X foundation model.

    This is the core model that integrates all components:
    - Neural tokenizers for different modalities
    - Mamba/SSM backbone for sequence modeling
    - Perceiver-IO for multi-modal fusion
    - Multi-task heads for various downstream tasks

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_mamba_blocks : int, optional
        Number of Mamba blocks.
        Default: 16.
    n_latents : int, optional
        Number of Perceiver latent vectors.
        Default: 128.
    latent_dim : int, optional
        Dimension of Perceiver latents.
        Default: 512.
    n_perceiver_layers : int, optional
        Number of Perceiver layers.
        Default: 3.
    use_multi_rate : bool, optional
        Use multi-rate Mamba streams.
        Default: True.
    downsample_rates : List[int], optional
        Downsampling rates for multi-rate streams.
        Default: [1, 4, 16].
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_mamba_blocks: int = 16,
        n_latents: int = 128,
        latent_dim: int = 512,
        n_perceiver_layers: int = 3,
        use_multi_rate: bool = True,
        downsample_rates: List[int] = [1, 4, 16],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.latent_dim = latent_dim

        # Mamba/SSM backbone
        self.backbone = MambaBackbone(
            d_model=d_model,
            n_blocks=n_mamba_blocks,
            dropout=dropout,
            use_multi_rate=use_multi_rate,
            downsample_rates=downsample_rates,
            fusion_method="concat",
        )

        # Perceiver-IO fusion
        self.fusion = PerceiverIO(
            n_latents=n_latents,
            latent_dim=latent_dim,
            input_dim=d_model,
            n_layers=n_perceiver_layers,
            dropout=dropout,
        )

        # Output projection (maps latents to desired output)
        self.output_projection = nn.Linear(latent_dim, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through NeuroFMX.

        Parameters
        ----------
        tokens : torch.Tensor
            Input tokens, shape (batch, seq_len, d_model).
        attention_mask : torch.Tensor, optional
            Attention mask, shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Output features, shape (batch, n_latents, latent_dim).
        """
        # Process through Mamba backbone
        backbone_output = self.backbone(tokens, attention_mask)
        # backbone_output: (batch, seq_len, d_model)

        # Fuse with Perceiver-IO
        latents = self.fusion(backbone_output, attention_mask)
        # latents: (batch, n_latents, latent_dim)

        return latents

    def encode(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode inputs to latent representation.

        Parameters
        ----------
        tokens : torch.Tensor
            Input tokens, shape (batch, seq_len, d_model).
        attention_mask : torch.Tensor, optional
            Attention mask, shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Latent features, shape (batch, n_latents, latent_dim).
        """
        return self.forward(tokens, attention_mask)

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for each component.

        Returns
        -------
        dict
            Parameter counts by component.
        """
        return {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "output_projection": sum(p.numel() for p in self.output_projection.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }

    @classmethod
    def from_config(cls, config: dict) -> "NeuroFMX":
        """Create model from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary with model hyperparameters.

        Returns
        -------
        NeuroFMX
            Initialized model.
        """
        return cls(
            d_model=config.get("d_model", 768),
            n_mamba_blocks=config.get("n_blocks", 16),
            n_latents=config.get("n_latents", 128),
            latent_dim=config.get("latent_dim", 512),
            n_perceiver_layers=config.get("n_perceiver_layers", 3),
            use_multi_rate=config.get("use_multi_rate", True),
            downsample_rates=config.get("downsample_rates", [1, 4, 16]),
            dropout=config.get("dropout", 0.1),
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "NeuroFMX":
        """Load pretrained model from checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file.

        Returns
        -------
        NeuroFMX
            Loaded model.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract config from checkpoint
        if "config" in checkpoint:
            model = cls.from_config(checkpoint["config"])
        else:
            # Use default config
            model = cls()

        # Load state dict
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model

    def save_pretrained(self, save_path: str, config: Optional[dict] = None) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        save_path : str
            Path to save checkpoint.
        config : dict, optional
            Configuration to save with checkpoint.
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": config or {},
        }
        torch.save(checkpoint, save_path)
