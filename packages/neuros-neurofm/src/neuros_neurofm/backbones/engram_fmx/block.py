"""
ENGRAM Block for ENGRAM-FMx.

Composes all modules into a single ENGRAM block that processes
both sequence tokens and latent workspace.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from neuros_neurofm.backbones.engram_fmx.config import ENGRAMFMxConfig
from neuros_neurofm.backbones.engram_fmx.modules.local_processing import LocalProcessingBlock
from neuros_neurofm.backbones.engram_fmx.modules.selective_ssm import SelectiveSSMBlock
from neuros_neurofm.backbones.engram_fmx.modules.latent_workspace import LatentWorkspace
from neuros_neurofm.backbones.engram_fmx.modules.attractor_memory import AttractorMemory
from neuros_neurofm.backbones.engram_fmx.modules.operator_dynamics import SpectralOperatorDynamics
from neuros_neurofm.backbones.engram_fmx.modules.sparse_anchor_attention import SparseAnchorAttention
from neuros_neurofm.backbones.engram_fmx.modules.gated_fusion import GatedFusion


@dataclass
class ENGRAMBlockOutput:
    """Output from a single ENGRAM block.

    Attributes
    ----------
    sequence_output : torch.Tensor
        Updated sequence tokens, shape [B, T, D].
    latent_output : torch.Tensor
        Updated latent workspace, shape [B, K, D].
    memory_state : Optional[Any]
        Recurrent state from SSM (for streaming inference).
    diagnostics : dict
        Diagnostic information from all modules.
    """

    sequence_output: torch.Tensor
    latent_output: torch.Tensor
    memory_state: Optional[Any]
    diagnostics: dict


class ENGRAMBlock(nn.Module):
    """Single ENGRAM block composing all modules.

    Block flow:
        tokens -> local processing -> selective SSM -> sequence_out
                                          |
                                          v
        latents -> latent workspace (cross-attn to sequence)
                          |
                          v
                   attractor memory
                          |
                          v
                   operator dynamics
                          |
                          v
                   sparse anchor attention (to sequence)
                          |
                          v
                   gated fusion -> latent_out

    Parameters
    ----------
    config : ENGRAMFMxConfig
        Configuration for the block.
    """

    def __init__(self, config: ENGRAMFMxConfig):
        super().__init__()
        self.config = config

        # Sequence processing modules
        if config.use_local_processing:
            self.local_processing = LocalProcessingBlock(
                hidden_dim=config.hidden_dim,
                conv_width=config.local_conv_width,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
        else:
            self.local_processing = None

        if config.use_ssm:
            self.selective_ssm = SelectiveSSMBlock(
                hidden_dim=config.hidden_dim,
                state_dim=config.ssm_state_dim,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
        else:
            self.selective_ssm = None

        # Latent processing modules
        if config.use_latent_workspace:
            self.latent_workspace = LatentWorkspace(
                hidden_dim=config.hidden_dim,
                num_latents=config.num_latents,
                num_heads=config.num_heads,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
        else:
            self.latent_workspace = None

        if config.use_attractor_memory:
            self.attractor_memory = AttractorMemory(
                hidden_dim=config.hidden_dim,
                memory_slots=config.memory_slots,
                beta=config.memory_beta,
                alpha=config.memory_residual_alpha,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
        else:
            self.attractor_memory = None

        if config.use_operator_dynamics:
            self.operator_dynamics = SpectralOperatorDynamics(
                hidden_dim=config.hidden_dim,
                num_latents=config.num_latents,
                operator_modes=config.operator_modes,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
        else:
            self.operator_dynamics = None

        if config.use_sparse_anchor_attention:
            self.sparse_anchor_attention = SparseAnchorAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                sparse_top_k=config.sparse_top_k,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
        else:
            self.sparse_anchor_attention = None

        # Gated fusion (only if we have multiple latent streams)
        if config.use_controller:
            # Determine which streams are active
            stream_names = []
            if config.use_latent_workspace:
                stream_names.append("workspace")
            if config.use_attractor_memory:
                stream_names.append("memory")
            if config.use_operator_dynamics:
                stream_names.append("operator")
            if config.use_sparse_anchor_attention:
                stream_names.append("sparse")

            if len(stream_names) > 1:
                self.gated_fusion = GatedFusion(
                    hidden_dim=config.hidden_dim,
                    stream_names=stream_names,
                    dropout=config.dropout,
                    layer_norm_eps=config.layer_norm_eps,
                )
            else:
                self.gated_fusion = None
        else:
            self.gated_fusion = None

        # Sequence output projection from latents
        self.sequence_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.sequence_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
        memory_state: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ENGRAMBlockOutput:
        """Forward pass through ENGRAM block.

        Parameters
        ----------
        tokens : torch.Tensor
            Input sequence tokens, shape [B, T, D].
        latents : torch.Tensor, optional
            Input latent workspace, shape [B, K, D].
            If None, initializes from learned latents.
        memory_state : Any, optional
            Recurrent state from previous block/step.
        attention_mask : torch.Tensor, optional
            Token attention mask [B, T]. True = valid.

        Returns
        -------
        ENGRAMBlockOutput
            Block outputs including sequence, latents, state, and diagnostics.
        """
        diagnostics: Dict[str, Any] = {}

        # === Sequence Processing ===

        # Local processing
        if self.local_processing is not None:
            tokens, local_diag = self.local_processing(tokens)
            diagnostics.update({f"local_{k}": v for k, v in local_diag.items()})

        # Selective SSM
        ssm_state = None
        if self.selective_ssm is not None:
            tokens, ssm_state, ssm_diag = self.selective_ssm(tokens, memory_state)
            diagnostics.update({f"ssm_{k}": v for k, v in ssm_diag.items()})

        # === Latent Processing ===

        # Track intermediate latent states for fusion
        latent_streams: Dict[str, torch.Tensor] = {}
        current_latents = latents

        # Latent workspace compression
        if self.latent_workspace is not None:
            current_latents, workspace_diag = self.latent_workspace(
                tokens=tokens,
                latents=latents,
                attention_mask=attention_mask,
            )
            latent_streams["workspace"] = current_latents
            diagnostics.update({f"workspace_{k}": v for k, v in workspace_diag.items()})

        # Attractor memory retrieval
        if self.attractor_memory is not None and current_latents is not None:
            memory_out, memory_diag = self.attractor_memory(current_latents)
            latent_streams["memory"] = memory_out
            current_latents = memory_out
            diagnostics.update({f"memory_{k}": v for k, v in memory_diag.items()})

        # Operator dynamics
        if self.operator_dynamics is not None and current_latents is not None:
            operator_out, operator_diag = self.operator_dynamics(current_latents)
            latent_streams["operator"] = operator_out
            current_latents = operator_out
            diagnostics.update({f"operator_{k}": v for k, v in operator_diag.items()})

        # Sparse anchor attention
        if self.sparse_anchor_attention is not None and current_latents is not None:
            sparse_out, sparse_diag = self.sparse_anchor_attention(
                latents=current_latents,
                tokens=tokens,
                attention_mask=attention_mask,
            )
            latent_streams["sparse"] = sparse_out
            current_latents = sparse_out
            diagnostics.update({f"sparse_{k}": v for k, v in sparse_diag.items()})

        # Gated fusion of latent streams
        if self.gated_fusion is not None and len(latent_streams) > 1:
            # Use first available latent as residual
            residual_latent = latents if latents is not None else current_latents
            fused_latents, fusion_diag = self.gated_fusion(latent_streams, residual_latent)
            current_latents = fused_latents
            diagnostics.update({f"fusion_{k}": v for k, v in fusion_diag.items()})

        # === Output ===

        # Project latents back to sequence for residual update
        if current_latents is not None:
            # Broadcast latent info to sequence via cross-attention-like projection
            # Simple approach: add mean latent representation
            latent_broadcast = current_latents.mean(dim=1, keepdim=True)  # [B, 1, D]
            sequence_update = self.sequence_proj(latent_broadcast)
            tokens = tokens + self.sequence_norm(sequence_update)

        return ENGRAMBlockOutput(
            sequence_output=tokens,
            latent_output=current_latents,
            memory_state=ssm_state,
            diagnostics=diagnostics,
        )
