"""
Local Processing Block for ENGRAM-FMx.

Captures short-range temporal or token-level detail before
long-range state propagation using depthwise convolution and gated MLP.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LocalProcessingBlock(nn.Module):
    """Local processing block with depthwise convolution and gated MLP.

    Processes local temporal context before global SSM propagation.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    conv_width : int
        Kernel width for depthwise convolution. Default: 7.
    dropout : float
        Dropout probability. Default: 0.1.
    layer_norm_eps : float
        LayerNorm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_dim: int,
        conv_width: int = 7,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_width = conv_width

        # Pre-normalization
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Depthwise temporal convolution
        # groups=hidden_dim makes it depthwise (each channel independently)
        self.depthwise_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=conv_width,
            padding=conv_width // 2,
            groups=hidden_dim,
        )

        # Gated MLP: split into gate and value paths
        self.linear_gate = nn.Linear(hidden_dim, hidden_dim)
        self.linear_value = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

        # Activation and dropout
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass with residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape [B, T, D].

        Returns
        -------
        Tuple[torch.Tensor, dict]
            Output tensor [B, T, D] and diagnostics dict.
        """
        residual = x
        x = self.norm(x)

        # Depthwise convolution: [B, T, D] -> [B, D, T] -> conv -> [B, D, T] -> [B, T, D]
        x_conv = x.transpose(1, 2)  # [B, D, T]
        x_conv = self.depthwise_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, T, D]

        # Combine original and conv output
        x = x + x_conv

        # Gated MLP
        gate = torch.sigmoid(self.linear_gate(x))
        value = self.activation(self.linear_value(x))
        x = gate * value
        x = self.linear_out(x)

        # Dropout and residual
        x = self.dropout(x)
        x = x + residual

        # Diagnostics
        diagnostics = {
            "local_gate_mean": gate.mean().item(),
            "local_output_norm": x.norm(dim=-1).mean().item(),
        }

        return x, diagnostics
