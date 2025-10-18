"""
Mamba/SSM backbone for NeuroFM-X.

Implements the Selective State-Space Model (Mamba) architecture for
efficient sequence modeling with linear complexity O(L).

This module wraps the mamba-ssm library and adds multi-rate streams
for hierarchical temporal modeling.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    # Create a placeholder for type hints
    class Mamba(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError(
                "mamba-ssm is not installed. "
                "Install with: pip install mamba-ssm"
            )


class MambaBlock(nn.Module):
    """Single Mamba block with pre-normalization.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_state : int, optional
        SSM state dimension.
        Default: 16.
    d_conv : int, optional
        Local convolution width.
        Default: 4.
    expand : int, optional
        Expansion factor for inner dimension.
        Default: 2.
    dt_rank : Union[int, str], optional
        Rank of Δ (discretization step size). If "auto", sets to ceil(d_model / 16).
        Default: "auto".
    dropout : float, optional
        Dropout probability.
        Default: 0.0.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dropout: float = 0.0,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for MambaBlock. "
                "Install with: pip install mamba-ssm"
            )

        self.d_model = d_model

        # Pre-normalization
        self.norm = nn.LayerNorm(d_model)

        # Mamba layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, seq_len, d_model).
        """
        # Pre-norm
        residual = x
        x = self.norm(x)

        # Mamba layer
        x = self.mamba(x)

        # Dropout and residual
        x = self.dropout(x)
        x = x + residual

        return x


class MambaBackbone(nn.Module):
    """Mamba/SSM backbone for sequence modeling.

    Stacks multiple Mamba blocks to create a deep SSM model with
    linear complexity in sequence length.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_blocks : int
        Number of Mamba blocks to stack.
    d_state : int, optional
        SSM state dimension.
        Default: 16.
    d_conv : int, optional
        Local convolution width.
        Default: 4.
    expand : int, optional
        Expansion factor for inner dimension.
        Default: 2.
    dt_rank : Union[int, str], optional
        Rank of Δ. If "auto", sets to ceil(d_model / 16).
        Default: "auto".
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    use_multi_rate : bool, optional
        Use multi-rate streams for hierarchical modeling.
        Default: False.
    downsample_rates : List[int], optional
        Downsampling rates for multi-rate streams (e.g., [1, 4, 16]).
        Only used if use_multi_rate=True.
        Default: [1, 4, 16].
    fusion_method : str, optional
        Method for fusing multi-rate streams: "concat", "add", or "attention".
        Default: "concat".
    """

    def __init__(
        self,
        d_model: int,
        n_blocks: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dropout: float = 0.1,
        use_multi_rate: bool = False,
        downsample_rates: List[int] = [1, 4, 16],
        fusion_method: str = "concat",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.use_multi_rate = use_multi_rate
        self.downsample_rates = downsample_rates if use_multi_rate else [1]
        self.fusion_method = fusion_method

        if use_multi_rate:
            # Create separate Mamba stacks for each rate
            self.rate_backbones = nn.ModuleList()
            for rate in downsample_rates:
                blocks = nn.ModuleList([
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dt_rank=dt_rank,
                        dropout=dropout,
                    )
                    for _ in range(n_blocks)
                ])
                self.rate_backbones.append(blocks)

            # Fusion layer
            if fusion_method == "concat":
                # Project concatenated features back to d_model
                self.fusion_projection = nn.Linear(
                    d_model * len(downsample_rates),
                    d_model,
                )
            elif fusion_method == "attention":
                # Cross-attention for fusing streams
                self.fusion_attention = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=8,
                    dropout=dropout,
                    batch_first=True,
                )
            elif fusion_method == "add":
                # Simple addition, no additional parameters
                self.fusion_projection = None
            else:
                raise ValueError(
                    f"Unknown fusion method: {fusion_method}. "
                    "Choose from 'concat', 'add', or 'attention'."
                )
        else:
            # Single-rate backbone
            self.blocks = nn.ModuleList([
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ])

        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)

    def _downsample(self, x: torch.Tensor, rate: int) -> torch.Tensor:
        """Downsample sequence by averaging.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, seq_len, d_model).
        rate : int
            Downsampling rate.

        Returns
        -------
        torch.Tensor
            Downsampled tensor, shape (batch, seq_len // rate, d_model).
        """
        if rate == 1:
            return x

        batch_size, seq_len, d_model = x.shape
        # Truncate to multiple of rate
        truncated_len = (seq_len // rate) * rate
        x = x[:, :truncated_len, :]

        # Reshape and average
        x = x.view(batch_size, truncated_len // rate, rate, d_model)
        x = x.mean(dim=2)

        return x

    def _upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Upsample sequence to target length using interpolation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, seq_len, d_model).
        target_len : int
            Target sequence length.

        Returns
        -------
        torch.Tensor
            Upsampled tensor, shape (batch, target_len, d_model).
        """
        batch_size, seq_len, d_model = x.shape

        if seq_len == target_len:
            return x

        # Transpose to (batch, d_model, seq_len) for interpolation
        x = x.transpose(1, 2)
        x = torch.nn.functional.interpolate(
            x,
            size=target_len,
            mode='linear',
            align_corners=False,
        )
        # Transpose back
        x = x.transpose(1, 2)

        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Mamba backbone.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens, shape (batch, seq_len, d_model).
        attention_mask : torch.Tensor, optional
            Attention mask (not used by Mamba, included for API compatibility).

        Returns
        -------
        torch.Tensor
            Output features, shape (batch, seq_len, d_model).
        """
        if self.use_multi_rate:
            # Process each rate stream
            rate_outputs = []
            original_len = x.shape[1]

            for rate, blocks in zip(self.downsample_rates, self.rate_backbones):
                # Downsample input
                x_rate = self._downsample(x, rate)

                # Process through Mamba blocks
                for block in blocks:
                    x_rate = block(x_rate)

                # Upsample back to original length
                x_rate = self._upsample(x_rate, original_len)
                rate_outputs.append(x_rate)

            # Fuse multi-rate outputs
            if self.fusion_method == "concat":
                fused = torch.cat(rate_outputs, dim=-1)
                fused = self.fusion_projection(fused)
            elif self.fusion_method == "add":
                fused = torch.stack(rate_outputs, dim=0).sum(dim=0)
            elif self.fusion_method == "attention":
                # Use first stream as query, all streams as key/value
                query = rate_outputs[0]
                keys_values = torch.stack(rate_outputs, dim=1)
                # Reshape for attention: (batch * seq_len, n_rates, d_model)
                batch_size, seq_len, d_model = query.shape
                query_flat = query.view(batch_size * seq_len, 1, d_model)
                kv_flat = keys_values.view(batch_size * seq_len, len(self.downsample_rates), d_model)
                fused_flat, _ = self.fusion_attention(query_flat, kv_flat, kv_flat)
                fused = fused_flat.view(batch_size, seq_len, d_model)

            x = fused
        else:
            # Single-rate processing
            for block in self.blocks:
                x = block(x)

        # Final normalization
        x = self.final_norm(x)

        return x
