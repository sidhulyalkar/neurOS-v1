"""
ENGRAM Backbone for ENGRAM-FMx.

Stacks multiple ENGRAM blocks to form the complete backbone.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from neuros_neurofm.backbones.engram_fmx.config import ENGRAMFMxConfig
from neuros_neurofm.backbones.engram_fmx.block import ENGRAMBlock


@dataclass
class ENGRAMBackboneOutput:
    """Output from the ENGRAM backbone.

    Attributes
    ----------
    sequence_output : torch.Tensor
        Final sequence representation, shape [B, T, D].
    latent_output : torch.Tensor
        Final latent workspace, shape [B, K, D].
    memory_state : Optional[Any]
        Final recurrent states from all layers.
    diagnostics : dict
        Aggregated diagnostics from all layers.
    """

    sequence_output: torch.Tensor
    latent_output: torch.Tensor
    memory_state: Optional[Any]
    diagnostics: dict


class ENGRAMBackbone(nn.Module):
    """ENGRAM-FMx backbone: stacks ENGRAM blocks.

    Parameters
    ----------
    config : ENGRAMFMxConfig
        Configuration for the backbone.
    """

    def __init__(self, config: ENGRAMFMxConfig):
        super().__init__()
        self.config = config

        # Input projection (if input_dim != hidden_dim)
        if config.input_dim != config.hidden_dim:
            self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        else:
            self.input_proj = None

        # Learned initial latents [K, D]
        self.initial_latents = nn.Parameter(
            torch.randn(config.num_latents, config.hidden_dim) * 0.02
        )

        # Stack of ENGRAM blocks
        self.blocks = nn.ModuleList([
            ENGRAMBlock(config)
            for _ in range(config.num_layers)
        ])

        # Final layer norms
        self.final_sequence_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.final_latent_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        # Output projection (if output_dim != hidden_dim)
        if config.output_dim != config.hidden_dim:
            self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        else:
            self.output_proj = None

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        initial_memory_states: Optional[List[Any]] = None,
        return_diagnostics: bool = True,
    ) -> ENGRAMBackboneOutput:
        """Forward pass through ENGRAM backbone.

        Parameters
        ----------
        tokens : torch.Tensor
            Input tokens, shape [B, T, D_in].
        attention_mask : torch.Tensor, optional
            Attention mask [B, T]. True = valid token, False = padding.
        initial_memory_states : List[Any], optional
            Initial recurrent states for each layer.
        return_diagnostics : bool
            Whether to collect and return diagnostics. Default: True.

        Returns
        -------
        ENGRAMBackboneOutput
            Backbone outputs including sequence, latents, state, and diagnostics.
        """
        B, T, D_in = tokens.shape

        # Input projection
        if self.input_proj is not None:
            tokens = self.input_proj(tokens)

        # Initialize latents: [K, D] -> [B, K, D]
        latents = self.initial_latents.unsqueeze(0).expand(B, -1, -1)

        # Track diagnostics and states
        all_diagnostics: Dict[str, Any] = {}
        memory_states: List[Any] = []

        # Process through blocks
        for layer_idx, block in enumerate(self.blocks):
            # Get initial state for this layer if provided
            layer_state = None
            if initial_memory_states is not None and layer_idx < len(initial_memory_states):
                layer_state = initial_memory_states[layer_idx]

            # Forward through block
            block_output = block(
                tokens=tokens,
                latents=latents,
                memory_state=layer_state,
                attention_mask=attention_mask,
            )

            # Update tokens and latents
            tokens = block_output.sequence_output
            latents = block_output.latent_output
            memory_states.append(block_output.memory_state)

            # Collect diagnostics with layer prefix
            if return_diagnostics:
                for key, value in block_output.diagnostics.items():
                    all_diagnostics[f"layer{layer_idx}_{key}"] = value

        # Final normalization
        tokens = self.final_sequence_norm(tokens)
        latents = self.final_latent_norm(latents)

        # Output projection
        if self.output_proj is not None:
            tokens = self.output_proj(tokens)

        # Add summary diagnostics
        if return_diagnostics:
            all_diagnostics["num_layers"] = len(self.blocks)
            all_diagnostics["sequence_output_norm"] = tokens.norm(dim=-1).mean().item()
            all_diagnostics["latent_output_norm"] = latents.norm(dim=-1).mean().item()

        return ENGRAMBackboneOutput(
            sequence_output=tokens,
            latent_output=latents,
            memory_state=memory_states if memory_states else None,
            diagnostics=all_diagnostics if return_diagnostics else {},
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters.

        Parameters
        ----------
        non_embedding : bool
            If True, exclude embedding parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.initial_latents.numel()
        return n_params

    @classmethod
    def from_config(cls, config: ENGRAMFMxConfig) -> "ENGRAMBackbone":
        """Create backbone from config.

        Parameters
        ----------
        config : ENGRAMFMxConfig
            Configuration.

        Returns
        -------
        ENGRAMBackbone
            Instantiated backbone.
        """
        return cls(config)

    @classmethod
    def tiny(cls) -> "ENGRAMBackbone":
        """Create a tiny backbone for testing."""
        return cls(ENGRAMFMxConfig.tiny())

    @classmethod
    def small(cls) -> "ENGRAMBackbone":
        """Create a small backbone for local GPU training."""
        return cls(ENGRAMFMxConfig.small())
