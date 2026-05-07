"""
Sparse Anchor Attention for ENGRAM-FMx.

Implements exact sparse grounding to selected sequence tokens,
preserving Transformer's strength at binding while avoiding full attention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SparseAnchorAttention(nn.Module):
    """Sparse anchor attention for grounding latents to selected tokens.

    Uses a router to select top-k tokens, then applies attention
    only from latents to the selected anchors.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension (D).
    num_heads : int
        Number of attention heads. Default: 4.
    sparse_top_k : int
        Number of tokens to select. Default: 128.
    dropout : float
        Dropout probability. Default: 0.1.
    layer_norm_eps : float
        LayerNorm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        sparse_top_k: int = 128,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.sparse_top_k = sparse_top_k

        # Router: compute importance scores for each token
        self.router_proj = nn.Linear(hidden_dim, 1)

        # Layer norms
        self.norm_latents = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm_tokens = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Attention from latents to selected tokens
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass: sparse attention from latents to selected tokens.

        Parameters
        ----------
        latents : torch.Tensor
            Latent states, shape [B, K, D].
        tokens : torch.Tensor
            Sequence tokens, shape [B, T, D].
        attention_mask : torch.Tensor, optional
            Token attention mask [B, T]. True = valid, False = masked.

        Returns
        -------
        Tuple[torch.Tensor, dict]
            - Context tensor [B, K, D]
            - Diagnostics dict with selected indices, router scores
        """
        B, K, D = latents.shape
        _, T, _ = tokens.shape

        # Compute router scores for each token
        # Use pooled latent as context for routing
        pooled_latent = latents.mean(dim=1, keepdim=True)  # [B, 1, D]

        # Router scores: dot product of pooled latent with each token
        tokens_norm = self.norm_tokens(tokens)
        router_scores = torch.einsum("bkd,btd->bt", pooled_latent, tokens_norm)  # [B, T]

        # Apply attention mask if provided (set masked positions to -inf)
        if attention_mask is not None:
            router_scores = router_scores.masked_fill(~attention_mask, float("-inf"))

        # Select top-k tokens
        actual_k = min(self.sparse_top_k, T)
        top_scores, top_indices = router_scores.topk(actual_k, dim=-1)  # [B, k]

        # Gather selected tokens
        # top_indices: [B, k] -> expand to [B, k, D] for gathering
        indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(tokens, dim=1, index=indices_expanded)  # [B, k, D]

        # Normalize
        latents_norm = self.norm_latents(latents)
        selected_norm = self.norm_tokens(selected_tokens)

        # Attention: latents attend to selected tokens
        context, attn_weights = self.attention(
            query=latents_norm,
            key=selected_norm,
            value=selected_norm,
            need_weights=True,
            average_attn_weights=True,
        )

        # Output projection and residual
        context = self.output_proj(context)
        context = self.dropout(context)
        output = latents + context

        # Compute diagnostics
        # Attention entropy
        attn_entropy = -(
            attn_weights * (attn_weights + 1e-10).log()
        ).sum(dim=-1).mean().item()

        # Router score statistics
        valid_scores = router_scores[router_scores > float("-inf")]
        if valid_scores.numel() > 0:
            router_mean = valid_scores.mean().item()
            router_std = valid_scores.std().item() if valid_scores.numel() > 1 else 0.0
        else:
            router_mean = 0.0
            router_std = 0.0

        diagnostics = {
            "sparse_selected_indices": top_indices.detach(),  # [B, k]
            "sparse_router_scores_mean": router_mean,
            "sparse_router_scores_std": router_std,
            "sparse_attn_entropy": attn_entropy,
            "sparse_num_selected": actual_k,
        }

        return output, diagnostics
