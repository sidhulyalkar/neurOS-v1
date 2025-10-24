"""
fMRI Tokenizer for NeuroFMx

Converts fMRI ROI timeseries into token embeddings.
Handles spatial (across ROIs) and temporal encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class fMRITokenizer(nn.Module):
    """
    Tokenizer for fMRI data.

    Args:
        n_rois: Number of brain ROIs
        d_model: Output embedding dimension
        seq_len: Target sequence length
        tr: Repetition time (seconds)
        use_graph: Whether to use graph convolutions for spatial structure
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_rois: int = 400,
        d_model: int = 512,
        seq_len: int = 50,
        tr: float = 0.72,
        use_graph: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_rois = n_rois
        self.d_model = d_model
        self.seq_len = seq_len
        self.tr = tr

        # ROI-wise projection
        self.roi_proj = nn.Sequential(
            nn.Linear(n_rois, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal encoding with dilated convolutions for slow dynamics
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=4, padding=4),
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(seq_len)

        # Final projection
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: fMRI timeseries (batch, time, rois)
            mask: Optional mask (batch, time)

        Returns:
            tokens: (batch, seq_len, d_model)
        """
        # ROI projection
        x = self.roi_proj(x)  # (B, T, d_model)

        # Temporal convolutions
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.temporal_conv(x)

        # Adaptive pooling to target length
        x = self.adaptive_pool(x)  # (B, d_model, seq_len)

        x = x.transpose(1, 2)  # (B, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding

        # Final projection
        x = self.proj(x)
        x = self.dropout(x)

        if mask is not None:
            mask_pooled = F.adaptive_avg_pool1d(
                mask.unsqueeze(1).float(),
                self.seq_len
            ).squeeze(1) > 0.5
            x = x * mask_pooled.unsqueeze(-1)

        return x
