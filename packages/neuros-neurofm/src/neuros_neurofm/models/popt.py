"""
PopT (Population Transformer) for NeuroFM-X.

Implements permutation-invariant neural population aggregation using
set-based attention mechanisms. PopT enables the model to handle
variable numbers of neurons across sessions/recordings.

Reference:
    Approximate concept from population-level neural modeling literature.
    Uses permutation-invariant aggregation similar to Set Transformers.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetAttention(nn.Module):
    """Set-based attention for permutation-invariant aggregation.

    Uses multi-head attention where queries are seed vectors (learned or pooled)
    and keys/values are the neural population.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int, optional
        Number of attention heads.
        Default: 8.
    n_seeds : int, optional
        Number of seed vectors for aggregation.
        Default: 1 (equivalent to global average pooling with attention).
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_seeds: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_seeds = n_seeds

        # Learnable seed vectors (queries for aggregation)
        self.seeds = nn.Parameter(torch.randn(n_seeds, d_model) * 0.02)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate population using set attention.

        Parameters
        ----------
        x : torch.Tensor
            Population features, shape (batch, n_units, d_model).
        key_padding_mask : torch.Tensor, optional
            Mask for invalid units (True = masked), shape (batch, n_units).

        Returns
        -------
        torch.Tensor
            Aggregated features, shape (batch, n_seeds, d_model).
        """
        batch_size = x.shape[0]

        # Expand seeds for batch
        queries = self.seeds.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply set attention (queries attend to population)
        attn_output, _ = self.multihead_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )

        # Residual connection and normalization
        output = self.norm(queries + attn_output)

        return output


class PopTLayer(nn.Module):
    """Single PopT layer with self-attention and feed-forward.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int, optional
        Number of attention heads.
        Default: 8.
    dim_feedforward : int, optional
        Dimension of feed-forward network.
        Default: 2048.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention for population
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through PopT layer.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, n_units, d_model).
        key_padding_mask : torch.Tensor, optional
            Mask for invalid units, shape (batch, n_units).

        Returns
        -------
        torch.Tensor
            Output features, shape (batch, n_units, d_model).
        """
        # Self-attention with residual
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm1(x + attn_output)

        # Feed-forward with residual
        x = self.norm2(x + self.ffn(x))

        return x


class PopT(nn.Module):
    """Population Transformer (PopT) for neural population aggregation.

    PopT processes neural population data in a permutation-invariant manner,
    enabling the model to handle variable numbers of neurons across sessions.

    Architecture:
    1. PopT layers (self-attention on population)
    2. Set attention (aggregate to fixed representation)
    3. Optional pooling (mean/max/attention-weighted)

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_layers : int, optional
        Number of PopT layers.
        Default: 3.
    n_heads : int, optional
        Number of attention heads.
        Default: 8.
    dim_feedforward : int, optional
        Dimension of feed-forward network.
        Default: 2048.
    n_output_seeds : int, optional
        Number of output vectors after aggregation.
        If 1, outputs single vector per batch.
        Default: 1.
    max_units : int, optional
        Maximum number of units to handle (for positional encoding).
        Default: 1000.
    use_unit_embeddings : bool, optional
        Add learnable embeddings for each unit position.
        Default: False.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 3,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        n_output_seeds: int = 1,
        max_units: int = 1000,
        use_unit_embeddings: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_output_seeds = n_output_seeds
        self.use_unit_embeddings = use_unit_embeddings

        # Optional unit position embeddings
        if use_unit_embeddings:
            self.unit_embeddings = nn.Embedding(max_units, d_model)
        else:
            self.unit_embeddings = None

        # PopT layers (permutation-equivariant)
        self.layers = nn.ModuleList([
            PopTLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Set attention for final aggregation (permutation-invariant)
        self.set_attention = SetAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_seeds=n_output_seeds,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        unit_indices: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process neural population through PopT.

        Parameters
        ----------
        x : torch.Tensor
            Population features, shape (batch, n_units, d_model).
        unit_indices : torch.Tensor, optional
            Unit indices for embeddings, shape (batch, n_units).
            Only used if use_unit_embeddings=True.
        padding_mask : torch.Tensor, optional
            Mask for invalid units (True = masked), shape (batch, n_units).

        Returns
        -------
        torch.Tensor
            Aggregated population features.
            If n_output_seeds=1: shape (batch, d_model)
            If n_output_seeds>1: shape (batch, n_output_seeds, d_model)
        """
        # Add unit embeddings if enabled
        if self.use_unit_embeddings:
            if unit_indices is None:
                # Default to sequential indices
                batch_size, n_units, _ = x.shape
                unit_indices = torch.arange(
                    n_units, device=x.device
                ).unsqueeze(0).expand(batch_size, -1)

            unit_embs = self.unit_embeddings(unit_indices)
            x = x + unit_embs

        # Process through PopT layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)

        # Aggregate using set attention
        aggregated = self.set_attention(x, key_padding_mask=padding_mask)
        # aggregated: (batch, n_output_seeds, d_model)

        # If single output, squeeze seed dimension
        if self.n_output_seeds == 1:
            aggregated = aggregated.squeeze(1)
            # aggregated: (batch, d_model)

        return aggregated


class PopTWithLatents(nn.Module):
    """PopT variant that outputs to Perceiver-like latent space.

    This combines PopT's population processing with explicit latent
    projection, useful when PopT output needs to match Perceiver latents.

    Parameters
    ----------
    d_model : int
        Model dimension for PopT processing.
    latent_dim : int
        Output latent dimension.
    n_latents : int, optional
        Number of output latent vectors.
        Default: 128.
    n_popt_layers : int, optional
        Number of PopT layers.
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
        d_model: int,
        latent_dim: int,
        n_latents: int = 128,
        n_popt_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # PopT for population processing
        self.popt = PopT(
            d_model=d_model,
            n_layers=n_popt_layers,
            n_heads=n_heads,
            n_output_seeds=n_latents,  # Output multiple seeds
            dropout=dropout,
        )

        # Project to latent dimension if different
        if d_model != latent_dim:
            self.projection = nn.Linear(d_model, latent_dim)
        else:
            self.projection = nn.Identity()

        # Layer norm
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        unit_indices: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process population to latent space.

        Parameters
        ----------
        x : torch.Tensor
            Population features, shape (batch, n_units, d_model).
        unit_indices : torch.Tensor, optional
            Unit indices, shape (batch, n_units).
        padding_mask : torch.Tensor, optional
            Padding mask, shape (batch, n_units).

        Returns
        -------
        torch.Tensor
            Latent features, shape (batch, n_latents, latent_dim).
        """
        # Process through PopT
        latents = self.popt(x, unit_indices, padding_mask)
        # latents: (batch, n_latents, d_model)

        # Project to latent dimension
        latents = self.projection(latents)
        latents = self.norm(latents)

        return latents
