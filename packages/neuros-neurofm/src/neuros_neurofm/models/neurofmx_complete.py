"""
NeuroFM-X: Complete Foundation Model for Neural Population Dynamics.

This is the full NeuroFMX model integrating all Phase 1-8 components:
- Tokenizers (spike, binned, LFP)
- Mamba/SSM backbone
- Perceiver-IO fusion
- PopT aggregator
- Multi-task heads
- Adapters (Unit-ID, LoRA, session stitching)
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuros_neurofm.models.mamba_backbone import MambaBackbone
from neuros_neurofm.models.popt import PopT, PopTWithLatents
from neuros_neurofm.fusion.perceiver import PerceiverIO
from neuros_neurofm.models.heads import MultiTaskHeads
from neuros_neurofm.adapters import UnitIDAdapter, SessionStitcher


class NeuroFMXComplete(nn.Module):
    """Complete NeuroFM-X foundation model with all components.

    Architecture flow:
    1. Tokenization (modality-specific)
    2. Mamba/SSM backbone (temporal modeling)
    3. Perceiver-IO fusion (multi-modal integration)
    4. PopT aggregation (population-level features)
    5. Multi-task heads (task-specific outputs)
    6. Optional adapters (transfer learning)

    Parameters
    ----------
    d_model : int
        Model dimension for backbone.
        Default: 768.
    n_mamba_blocks : int, optional
        Number of Mamba blocks.
        Default: 16.
    n_latents : int, optional
        Number of Perceiver latent vectors.
        Default: 128.
    latent_dim : int, optional
        Dimension of latents.
        Default: 512.
    n_perceiver_layers : int, optional
        Number of Perceiver layers.
        Default: 3.
    n_popt_layers : int, optional
        Number of PopT layers.
        Default: 3.
    use_popt : bool, optional
        Use PopT for population aggregation.
        Default: True.
    use_multi_rate : bool, optional
        Use multi-rate Mamba streams.
        Default: True.
    downsample_rates : List[int], optional
        Multi-rate downsampling factors.
        Default: [1, 4, 16].
    enable_decoder : bool, optional
        Enable behavioral decoder head.
        Default: True.
    enable_encoder : bool, optional
        Enable neural encoder head.
        Default: True.
    enable_contrastive : bool, optional
        Enable contrastive head.
        Default: True.
    decoder_output_dim : int, optional
        Decoder output dimension (behavioral variables).
    encoder_output_dim : int, optional
        Encoder output dimension (neural dimension).
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
        n_popt_layers: int = 3,
        use_popt: bool = True,
        use_multi_rate: bool = True,
        downsample_rates: List[int] = [1, 4, 16],
        enable_decoder: bool = True,
        enable_encoder: bool = True,
        enable_contrastive: bool = True,
        decoder_output_dim: Optional[int] = None,
        encoder_output_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.use_popt = use_popt

        # 1. Mamba/SSM backbone
        self.backbone = MambaBackbone(
            d_model=d_model,
            n_blocks=n_mamba_blocks,
            dropout=dropout,
            use_multi_rate=use_multi_rate,
            downsample_rates=downsample_rates,
            fusion_method="concat",
        )

        # 2. Perceiver-IO fusion
        self.fusion = PerceiverIO(
            n_latents=n_latents,
            latent_dim=latent_dim,
            input_dim=d_model,
            n_layers=n_perceiver_layers,
            dropout=dropout,
        )

        # 3. PopT aggregator (optional)
        if use_popt:
            self.popt = PopTWithLatents(
                d_model=latent_dim,
                latent_dim=latent_dim,
                n_latents=n_latents,
                n_popt_layers=n_popt_layers,
                dropout=dropout,
            )
        else:
            self.popt = None

        # 4. Multi-task heads
        head_input_dim = latent_dim
        self.heads = MultiTaskHeads(
            input_dim=head_input_dim,
            decoder_output_dim=decoder_output_dim,
            encoder_output_dim=encoder_output_dim,
            enable_decoder=enable_decoder,
            enable_encoder=enable_encoder,
            enable_contrastive=enable_contrastive,
            dropout=dropout,
        )

        # 5. Adapters (optional, added via methods)
        self.unit_id_adapter = None
        self.session_stitcher = None

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        unit_indices: Optional[torch.Tensor] = None,
        session_id: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through complete NeuroFMX.

        Parameters
        ----------
        tokens : torch.Tensor
            Input tokens, shape (batch, seq_len, d_model).
        attention_mask : torch.Tensor, optional
            Attention mask, shape (batch, seq_len).
        unit_indices : torch.Tensor, optional
            Unit indices for Unit-ID adapter, shape (batch, n_units).
        session_id : torch.Tensor, optional
            Session ID for session stitching, shape (batch,).
        task : str, optional
            Specific task to run ('decoder', 'encoder', 'contrastive').
            If None, returns raw latents.

        Returns
        -------
        torch.Tensor or dict
            If task specified: task-specific output
            If task=None: latent features (batch, n_latents, latent_dim)
        """
        # 1. Process through Mamba backbone
        backbone_output = self.backbone(tokens, attention_mask)
        # backbone_output: (batch, seq_len, d_model)

        # 2. Fuse with Perceiver-IO
        latents = self.fusion(backbone_output, attention_mask)
        # latents: (batch, n_latents, latent_dim)

        # 3. Apply PopT aggregation if enabled
        if self.popt is not None:
            # Treat latents as population features
            latents = self.popt(latents, unit_indices=unit_indices)
            # latents: (batch, n_latents, latent_dim)

        # 4. Apply adapters if present
        if self.unit_id_adapter is not None and unit_indices is not None:
            latents = self.unit_id_adapter(latents, unit_indices)

        if self.session_stitcher is not None and session_id is not None:
            latents = self.session_stitcher(latents, session_id)

        # 5. Apply task-specific head if requested
        if task is not None:
            # Pool latents for task heads (mean pooling)
            pooled_latents = latents.mean(dim=1)  # (batch, latent_dim)
            output = self.heads(pooled_latents, task=task)
            return output
        else:
            # Return raw latents
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
            Input tokens.
        attention_mask : torch.Tensor, optional
            Attention mask.

        Returns
        -------
        torch.Tensor
            Latent features, shape (batch, n_latents, latent_dim).
        """
        return self.forward(tokens, attention_mask, task=None)

    def decode_behavior(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode behavioral variables from neural activity.

        Parameters
        ----------
        tokens : torch.Tensor
            Input tokens.
        attention_mask : torch.Tensor, optional
            Attention mask.

        Returns
        -------
        torch.Tensor
            Predicted behavioral variables.
        """
        return self.forward(tokens, attention_mask, task='decoder')

    def add_unit_id_adapter(
        self,
        n_units: int,
        bottleneck_dim: int = 128,
        freeze_backbone: bool = True,
    ) -> None:
        """Add Unit-ID adapter for transfer learning.

        Parameters
        ----------
        n_units : int
            Number of units in target population.
        bottleneck_dim : int, optional
            Bottleneck dimension.
        freeze_backbone : bool, optional
            Whether to freeze backbone weights.
        """
        self.unit_id_adapter = UnitIDAdapter(
            backbone_dim=self.latent_dim,
            n_units=n_units,
            bottleneck_dim=bottleneck_dim,
            freeze_backbone=freeze_backbone,
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.fusion.parameters():
                param.requires_grad = False
            if self.popt is not None:
                for param in self.popt.parameters():
                    param.requires_grad = False

    def add_session_stitcher(
        self,
        n_sessions: int,
        use_affine: bool = True,
    ) -> None:
        """Add session/region stitching adapter.

        Parameters
        ----------
        n_sessions : int
            Number of sessions/regions.
        use_affine : bool, optional
            Use affine transformation.
        """
        self.session_stitcher = SessionStitcher(
            d_model=self.latent_dim,
            n_sessions=n_sessions,
            use_affine=use_affine,
        )

    def get_num_params(self, trainable_only: bool = False) -> Dict[str, int]:
        """Get parameter counts.

        Parameters
        ----------
        trainable_only : bool, optional
            Only count trainable parameters.

        Returns
        -------
        dict
            Parameter counts by component.
        """
        def count_params(module):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            else:
                return sum(p.numel() for p in module.parameters())

        counts = {
            "backbone": count_params(self.backbone),
            "fusion": count_params(self.fusion),
            "heads": count_params(self.heads),
        }

        if self.popt is not None:
            counts["popt"] = count_params(self.popt)

        if self.unit_id_adapter is not None:
            counts["unit_id_adapter"] = count_params(self.unit_id_adapter)

        if self.session_stitcher is not None:
            counts["session_stitcher"] = count_params(self.session_stitcher)

        counts["total"] = sum(counts.values())

        return counts

    @classmethod
    def from_config(cls, config: dict) -> "NeuroFMXComplete":
        """Create model from configuration dict."""
        return cls(
            d_model=config.get("d_model", 768),
            n_mamba_blocks=config.get("n_blocks", 16),
            n_latents=config.get("n_latents", 128),
            latent_dim=config.get("latent_dim", 512),
            n_perceiver_layers=config.get("n_perceiver_layers", 3),
            n_popt_layers=config.get("n_popt_layers", 3),
            use_popt=config.get("use_popt", True),
            use_multi_rate=config.get("use_multi_rate", True),
            downsample_rates=config.get("downsample_rates", [1, 4, 16]),
            enable_decoder=config.get("enable_decoder", True),
            enable_encoder=config.get("enable_encoder", True),
            enable_contrastive=config.get("enable_contrastive", True),
            decoder_output_dim=config.get("decoder_output_dim"),
            encoder_output_dim=config.get("encoder_output_dim"),
            dropout=config.get("dropout", 0.1),
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "NeuroFMXComplete":
        """Load from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "config" in checkpoint:
            model = cls.from_config(checkpoint["config"])
        else:
            model = cls()

        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model

    def save_pretrained(self, save_path: str, config: Optional[dict] = None) -> None:
        """Save checkpoint."""
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": config or {},
        }
        torch.save(checkpoint, save_path)
