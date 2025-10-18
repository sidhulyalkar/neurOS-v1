"""
Perceiver-IO for multi-modal fusion in NeuroFM-X.

Implements the Perceiver-IO architecture that uses cross-attention to
map high-dimensional multi-modal inputs to a fixed set of latent vectors,
achieving O(L*M) complexity instead of O(L²) for standard Transformers.

Reference:
    Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
    ICLR 2022
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Cross-attention module for Perceiver.

    Parameters
    ----------
    query_dim : int
        Dimension of query vectors (latents).
    key_value_dim : int
        Dimension of key/value vectors (inputs).
    n_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability.
        Default: 0.0.
    """

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads

        assert query_dim % n_heads == 0, "query_dim must be divisible by n_heads"

        # Query projection (from latents)
        self.q_proj = nn.Linear(query_dim, query_dim)

        # Key and value projections (from inputs)
        self.k_proj = nn.Linear(key_value_dim, query_dim)
        self.v_proj = nn.Linear(key_value_dim, query_dim)

        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor (latents), shape (batch, n_latents, query_dim).
        key_value : torch.Tensor
            Key/value tensor (inputs), shape (batch, seq_len, key_value_dim).
        attention_mask : torch.Tensor, optional
            Attention mask for key/value, shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, n_latents, query_dim).
        """
        batch_size, n_latents, _ = query.shape
        _, seq_len, _ = key_value.shape

        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        # Reshape for multi-head attention
        # (batch, n_latents, n_heads, head_dim) -> (batch, n_heads, n_latents, head_dim)
        Q = Q.view(batch_size, n_latents, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: (batch, n_heads, n_latents, seq_len)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for heads and latents
            # (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # output: (batch, n_heads, n_latents, head_dim)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, n_latents, self.query_dim
        )

        # Final projection
        output = self.out_proj(output)

        return output


class SelfAttention(nn.Module):
    """Self-attention for processing latents.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability.
        Default: 0.0.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # QKV projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, n_latents, d_model).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, n_latents, d_model).
        """
        batch_size, n_latents, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, n_latents, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, n_latents, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # output: (batch, n_heads, n_latents, head_dim)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, n_latents, self.d_model
        )

        # Final projection
        output = self.out_proj(output)

        return output


class PerceiverBlock(nn.Module):
    """Single Perceiver block with cross-attention and self-attention.

    Parameters
    ----------
    latent_dim : int
        Dimension of latent vectors.
    input_dim : int
        Dimension of input vectors.
    n_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Cross-attention (latents attend to inputs)
        self.cross_attn_norm = nn.LayerNorm(latent_dim)
        self.cross_attn = CrossAttention(
            query_dim=latent_dim,
            key_value_dim=input_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Self-attention (latents attend to each other)
        self.self_attn_norm = nn.LayerNorm(latent_dim)
        self.self_attn = SelfAttention(
            d_model=latent_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        latents : torch.Tensor
            Latent vectors, shape (batch, n_latents, latent_dim).
        inputs : torch.Tensor
            Input vectors, shape (batch, seq_len, input_dim).
        attention_mask : torch.Tensor, optional
            Mask for inputs, shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Updated latents, shape (batch, n_latents, latent_dim).
        """
        # Cross-attention
        latents = latents + self.cross_attn(
            self.cross_attn_norm(latents),
            inputs,
            attention_mask,
        )

        # Self-attention
        latents = latents + self.self_attn(
            self.self_attn_norm(latents)
        )

        # Feed-forward
        latents = latents + self.ffn(
            self.ffn_norm(latents)
        )

        return latents


class PerceiverIO(nn.Module):
    """Perceiver-IO for multi-modal fusion.

    Uses cross-attention to map multi-modal inputs to a fixed set of
    latent vectors, then processes latents with self-attention.

    Parameters
    ----------
    n_latents : int
        Number of latent vectors.
    latent_dim : int
        Dimension of latent vectors.
    input_dim : int
        Dimension of input vectors (after modality-specific encoding).
    n_layers : int, optional
        Number of Perceiver blocks.
        Default: 3.
    n_heads : int, optional
        Number of attention heads.
        Default: 8.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        n_latents: int,
        latent_dim: int,
        input_dim: int,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.n_layers = n_layers

        # Learnable latent array
        self.latents = nn.Parameter(
            torch.randn(n_latents, latent_dim) * 0.02
        )

        # Perceiver blocks
        self.blocks = nn.ModuleList([
            PerceiverBlock(
                latent_dim=latent_dim,
                input_dim=input_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input features from backbone, shape (batch, seq_len, input_dim).
        attention_mask : torch.Tensor, optional
            Mask for inputs, shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Latent representations, shape (batch, n_latents, latent_dim).
        """
        batch_size = inputs.shape[0]

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Process through Perceiver blocks
        for block in self.blocks:
            latents = block(latents, inputs, attention_mask)

        # Final normalization
        latents = self.final_norm(latents)

        return latents
