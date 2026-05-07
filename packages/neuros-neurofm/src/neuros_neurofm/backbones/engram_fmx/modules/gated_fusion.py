"""
Gated Fusion for ENGRAM-FMx.

Combines multiple module outputs using learned softmax gates.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class GatedFusion(nn.Module):
    """Gated fusion for combining multiple streams.

    Combines module outputs with learned softmax gates:
        gate_logits = W @ concat(streams)
        gates = softmax(gate_logits)
        output = sum(gates[i] * streams[i])

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension (D).
    stream_names : List[str]
        Names of streams to fuse.
    dropout : float
        Dropout probability. Default: 0.1.
    layer_norm_eps : float
        LayerNorm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_dim: int,
        stream_names: List[str],
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stream_names = stream_names
        self.num_streams = len(stream_names)

        # Gate network: takes concatenated or pooled streams
        # Use a small MLP to compute gate logits
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * self.num_streams, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_streams),
        )

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        streams: Dict[str, torch.Tensor],
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass: fuse streams with gating.

        Parameters
        ----------
        streams : Dict[str, torch.Tensor]
            Dictionary mapping stream names to tensors [B, K, D].
        residual : torch.Tensor
            Residual connection tensor [B, K, D].

        Returns
        -------
        Tuple[torch.Tensor, dict]
            - Fused output [B, K, D]
            - Diagnostics dict with gate values
        """
        # Collect streams in order
        stream_tensors = []
        for name in self.stream_names:
            if name in streams:
                stream_tensors.append(streams[name])
            else:
                # Missing stream: use zeros
                B, K, D = residual.shape
                stream_tensors.append(torch.zeros(B, K, D, device=residual.device, dtype=residual.dtype))

        # Stack streams: [num_streams, B, K, D]
        stacked = torch.stack(stream_tensors, dim=0)
        num_streams, B, K, D = stacked.shape

        # Compute gate logits from concatenated mean-pooled streams
        # Mean pool over K dimension: [num_streams, B, D]
        pooled = stacked.mean(dim=2)  # [num_streams, B, D]

        # Concatenate streams: [B, num_streams * D]
        pooled_concat = pooled.permute(1, 0, 2).reshape(B, -1)  # [B, num_streams * D]

        # Compute gates: [B, num_streams]
        gate_logits = self.gate_network(pooled_concat)
        gates = torch.softmax(gate_logits, dim=-1)  # [B, num_streams]

        # Expand gates for broadcasting: [B, 1, num_streams, 1]
        gates_expanded = gates.unsqueeze(1).unsqueeze(-1)  # [B, 1, num_streams, 1]

        # Weighted sum: [B, K, D]
        # stacked is [num_streams, B, K, D] -> transpose to [B, K, num_streams, D]
        stacked_transposed = stacked.permute(1, 2, 0, 3)  # [B, K, num_streams, D]

        # gates_expanded: [B, 1, num_streams, 1] broadcasts to [B, K, num_streams, D]
        fused = (gates_expanded * stacked_transposed).sum(dim=2)  # [B, K, D]

        # Output projection
        fused = self.output_norm(fused)
        fused = self.output_proj(fused)
        fused = self.dropout(fused)

        # Add residual
        output = residual + fused

        # Diagnostics: gate values per stream
        gate_values = {
            f"gate_{name}": gates[:, i].mean().item()
            for i, name in enumerate(self.stream_names)
        }

        diagnostics = {
            "fusion_gates": gates.detach(),  # [B, num_streams]
            **gate_values,
        }

        return output, diagnostics
