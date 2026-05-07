"""
Latent Workspace for ENGRAM-FMx.

Compresses sequence states into learned latent slots using
Perceiver-style cross-attention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LatentWorkspace(nn.Module):
    """Latent workspace with cross-attention compression.

    Compresses sequence tokens [B, T, D] into latent slots [B, K, D]
    using cross-attention, where K << T.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension (D).
    num_latents : int
        Number of latent slots (K). Default: 64.
    num_heads : int
        Number of attention heads. Default: 4.
    dropout : float
        Dropout probability. Default: 0.1.
    layer_norm_eps : float
        LayerNorm epsilon. Default: 1e-5.
    use_latent_self_attention : bool
        Apply self-attention among latents after cross-attention. Default: True.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_latents: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_latent_self_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.use_latent_self_attention = use_latent_self_attention

        # Note: Initial latents are provided by the backbone (ENGRAMBackbone.initial_latents)
        # This allows sharing latents across blocks and cleaner gradient flow.

        # Layer norms
        self.norm_latents = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm_tokens = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Cross-attention: latents attend to tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Optional latent self-attention
        if use_latent_self_attention:
            self.norm_self = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        latents: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass: compress tokens into latent workspace.

        Parameters
        ----------
        tokens : torch.Tensor
            Sequence tokens, shape [B, T, D].
        latents : torch.Tensor
            Latent slots [B, K, D]. Provided by backbone.initial_latents.
        attention_mask : torch.Tensor, optional
            Attention mask for tokens [B, T]. True = valid, False = masked.

        Returns
        -------
        Tuple[torch.Tensor, dict]
            - Latent output [B, K, D]
            - Diagnostics dict
        """
        B, T, D = tokens.shape
        z = latents

        # Normalize
        z_norm = self.norm_latents(z)
        tokens_norm = self.norm_tokens(tokens)

        # Convert attention mask to key_padding_mask format if provided
        # key_padding_mask: True = ignore, False = attend (opposite of our convention)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # Invert: True->False, False->True

        # Cross-attention: latents (query) attend to tokens (key, value)
        z_cross, cross_attn_weights = self.cross_attention(
            query=z_norm,
            key=tokens_norm,
            value=tokens_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        z = z + z_cross

        # Optional self-attention among latents
        if self.use_latent_self_attention:
            z_norm = self.norm_self(z)
            z_self, _ = self.self_attention(
                query=z_norm,
                key=z_norm,
                value=z_norm,
            )
            z = z + z_self

        # Feed-forward
        z = z + self.ffn(z)

        # Compute diagnostics
        # Cross-attention entropy: how spread out is the attention?
        cross_attn_entropy = -(
            cross_attn_weights * (cross_attn_weights + 1e-10).log()
        ).sum(dim=-1).mean().item()

        diagnostics = {
            "latent_cross_attn_entropy": cross_attn_entropy,
            "latent_output_norm": z.norm(dim=-1).mean().item(),
        }

        return z, diagnostics
