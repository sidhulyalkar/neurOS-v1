"""
Attractor Memory for ENGRAM-FMx.

Implements Hopfield-style energy-guided associative memory retrieval.
The memory can be interpreted as an energy landscape that shapes latent trajectories.
"""

import torch
import torch.nn as nn
from typing import Tuple


class AttractorMemory(nn.Module):
    """Hopfield-style attractor memory with energy-guided retrieval.

    Memory retrieval is computed as:
        scores = beta * queries @ memory_keys.T
        weights = softmax(scores)
        retrieved = weights @ memory_values
        output = queries + alpha * retrieved

    The corresponding energy function is:
        E(z; K) = -1/beta * log(sum_i exp(beta * z @ k_i))

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension (D).
    memory_slots : int
        Number of memory slots (M). Default: 256.
    beta : float
        Temperature for softmax (inverse temperature). Default: 8.0.
    alpha : float
        Residual mixing coefficient. Default: 0.5.
    dropout : float
        Dropout probability. Default: 0.1.
    layer_norm_eps : float
        LayerNorm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_slots: int = 256,
        beta: float = 8.0,
        alpha: float = 0.5,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.beta = beta
        self.alpha = alpha

        # Learnable memory bank: keys and values
        # Initialize with small values for stable training
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, hidden_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, hidden_dim) * 0.02)

        # Query projection
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass: retrieve from memory.

        Parameters
        ----------
        queries : torch.Tensor
            Query latents, shape [B, K, D].

        Returns
        -------
        Tuple[torch.Tensor, dict]
            - Output tensor [B, K, D] (queries + retrieved)
            - Diagnostics dict with memory weights, entropy, usage
        """
        B, K, D = queries.shape

        # Normalize and project queries
        q = self.norm(queries)
        q = self.query_proj(q)  # [B, K, D]

        # Normalize memory keys for stable dot products
        keys = self.memory_keys / (self.memory_keys.norm(dim=-1, keepdim=True) + 1e-8)
        q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute attention scores: [B, K, M]
        scores = self.beta * torch.einsum("bkd,md->bkm", q_norm, keys)

        # Softmax over memory slots
        weights = torch.softmax(scores, dim=-1)  # [B, K, M]

        # Retrieve values: [B, K, D]
        retrieved = torch.einsum("bkm,md->bkd", weights, self.memory_values)

        # Apply dropout to retrieved
        retrieved = self.dropout(retrieved)

        # Residual combination
        output = queries + self.alpha * retrieved

        # Compute diagnostics
        # Memory entropy per query
        entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1)  # [B, K]
        mean_entropy = entropy.mean().item()

        # Memory usage: average weight per slot across all queries
        usage = weights.mean(dim=(0, 1))  # [M]
        max_usage = usage.max().item()
        min_usage = usage.min().item()

        # Top memory indices
        top_indices = weights.mean(dim=(0, 1)).topk(min(5, self.memory_slots)).indices.tolist()

        diagnostics = {
            "memory_weights": weights.detach(),  # [B, K, M] for visualization
            "memory_entropy": mean_entropy,
            "memory_usage_max": max_usage,
            "memory_usage_min": min_usage,
            "memory_top_indices": top_indices,
        }

        return output, diagnostics

    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Hopfield energy for latent states.

        E(z; K) = -1/beta * log(sum_i exp(beta * z @ k_i))

        Parameters
        ----------
        z : torch.Tensor
            Latent states, shape [B, K, D].

        Returns
        -------
        torch.Tensor
            Energy values, shape [B, K].
        """
        # Normalize
        z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        keys = self.memory_keys / (self.memory_keys.norm(dim=-1, keepdim=True) + 1e-8)

        # Scores
        scores = self.beta * torch.einsum("bkd,md->bkm", z_norm, keys)

        # Log-sum-exp for energy
        energy = -1.0 / self.beta * torch.logsumexp(scores, dim=-1)

        return energy
