"""
Memory Metrics for ENGRAM-FMx.

Computes metrics about attractor memory behavior:
- Entropy of retrieval weights
- Memory slot utilization
- Retrieval pattern analysis
"""

import torch
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_memory_entropy(
    memory_weights: torch.Tensor,
    eps: float = 1e-10,
) -> Dict[str, float]:
    """Compute entropy statistics of memory retrieval weights.

    Parameters
    ----------
    memory_weights : torch.Tensor
        Memory attention weights [B, K, M] or [K, M].
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Dict[str, float]
        Dictionary with entropy statistics.
    """
    if memory_weights.dim() == 2:
        memory_weights = memory_weights.unsqueeze(0)

    # Compute entropy per query: H = -sum(p * log(p))
    log_weights = (memory_weights + eps).log()
    entropy = -(memory_weights * log_weights).sum(dim=-1)  # [B, K]

    # Max possible entropy (uniform distribution)
    M = memory_weights.shape[-1]
    max_entropy = np.log(M)

    # Normalized entropy (0 = peaked, 1 = uniform)
    normalized_entropy = entropy / max_entropy

    return {
        "mean_entropy": entropy.mean().item(),
        "std_entropy": entropy.std().item(),
        "min_entropy": entropy.min().item(),
        "max_entropy": entropy.max().item(),
        "mean_normalized_entropy": normalized_entropy.mean().item(),
        "max_possible_entropy": max_entropy,
    }


def compute_memory_usage(
    memory_weights: torch.Tensor,
    threshold: float = 0.01,
) -> Dict[str, float]:
    """Compute memory slot utilization statistics.

    Parameters
    ----------
    memory_weights : torch.Tensor
        Memory attention weights [B, K, M].
    threshold : float
        Minimum weight to consider a slot "used".

    Returns
    -------
    Dict[str, float]
        Dictionary with usage statistics.
    """
    if memory_weights.dim() == 2:
        memory_weights = memory_weights.unsqueeze(0)

    B, K, M = memory_weights.shape

    # Average weight per slot across all queries
    slot_usage = memory_weights.mean(dim=(0, 1))  # [M]

    # Count slots above threshold
    active_slots = (slot_usage > threshold).sum().item()
    active_ratio = active_slots / M

    # Usage distribution statistics
    usage_std = slot_usage.std().item()
    usage_max = slot_usage.max().item()
    usage_min = slot_usage.min().item()

    # Gini coefficient (inequality measure)
    sorted_usage = slot_usage.sort().values
    cumsum = sorted_usage.cumsum(0)
    gini = 1 - 2 * cumsum.sum().item() / (M * slot_usage.sum().item() + 1e-10)

    return {
        "active_slots": active_slots,
        "active_ratio": active_ratio,
        "total_slots": M,
        "usage_std": usage_std,
        "usage_max": usage_max,
        "usage_min": usage_min,
        "usage_gini": gini,  # 0 = equal usage, 1 = one slot dominates
    }


def analyze_memory_retrieval(
    memory_weights: torch.Tensor,
    memory_keys: Optional[torch.Tensor] = None,
    top_k: int = 5,
) -> Dict[str, any]:
    """Analyze memory retrieval patterns.

    Parameters
    ----------
    memory_weights : torch.Tensor
        Memory attention weights [B, K, M].
    memory_keys : torch.Tensor, optional
        Memory key embeddings [M, D] for similarity analysis.
    top_k : int
        Number of top slots to analyze.

    Returns
    -------
    Dict[str, any]
        Dictionary with retrieval analysis.
    """
    if memory_weights.dim() == 2:
        memory_weights = memory_weights.unsqueeze(0)

    B, K, M = memory_weights.shape

    # Average weights across batch and queries
    avg_weights = memory_weights.mean(dim=(0, 1))  # [M]

    # Top-k most used slots
    top_values, top_indices = avg_weights.topk(top_k)

    # Concentration: what fraction of weight goes to top-k slots
    top_k_concentration = top_values.sum().item()

    # Per-query analysis: how many slots does each query use significantly?
    slots_per_query = (memory_weights > 0.01).sum(dim=-1).float()  # [B, K]

    result = {
        "top_k_indices": top_indices.tolist(),
        "top_k_weights": top_values.tolist(),
        "top_k_concentration": top_k_concentration,
        "mean_slots_per_query": slots_per_query.mean().item(),
        "std_slots_per_query": slots_per_query.std().item(),
    }

    # If memory keys provided, analyze key similarity
    if memory_keys is not None:
        # Normalize keys
        keys_norm = memory_keys / (memory_keys.norm(dim=-1, keepdim=True) + 1e-8)

        # Similarity matrix
        similarity = keys_norm @ keys_norm.T  # [M, M]

        # Average similarity (excluding diagonal)
        mask = 1 - torch.eye(M, device=similarity.device)
        avg_similarity = (similarity * mask).sum() / mask.sum()

        # Similarity of top-k keys
        top_keys = keys_norm[top_indices]
        top_similarity = top_keys @ top_keys.T
        top_mask = 1 - torch.eye(top_k, device=top_similarity.device)
        top_avg_similarity = (top_similarity * top_mask).sum() / top_mask.sum()

        result["avg_key_similarity"] = avg_similarity.item()
        result["top_k_key_similarity"] = top_avg_similarity.item()

    return result


class MemoryTracker:
    """Track memory statistics over training."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history: List[Dict[str, float]] = []
        self.steps: List[int] = []

    def update(self, step: int, memory_weights: torch.Tensor):
        """Record memory statistics for a training step."""
        entropy_stats = compute_memory_entropy(memory_weights)
        usage_stats = compute_memory_usage(memory_weights)

        record = {
            "step": step,
            **entropy_stats,
            **usage_stats,
        }

        self.history.append(record)
        self.steps.append(step)

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.steps = self.steps[-self.max_history:]

    def get_history(self, key: str) -> Tuple[List[int], List[float]]:
        """Get history for a specific metric."""
        values = [h[key] for h in self.history if key in h]
        steps = self.steps[:len(values)]
        return steps, values

    def summary(self) -> Dict[str, float]:
        """Get summary statistics over history."""
        if not self.history:
            return {}

        keys = self.history[0].keys()
        summary = {}

        for key in keys:
            if key == "step":
                continue
            values = [h[key] for h in self.history if isinstance(h.get(key), (int, float))]
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_final"] = values[-1]

        return summary
