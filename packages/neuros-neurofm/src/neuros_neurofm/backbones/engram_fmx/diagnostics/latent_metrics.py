"""
Latent Metrics for ENGRAM-FMx.

Computes metrics about latent workspace behavior:
- PCA projections
- Trajectory analysis
- Similarity patterns
"""

import torch
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_latent_pca(
    latents: torch.Tensor,
    n_components: int = 3,
) -> Dict[str, torch.Tensor]:
    """Compute PCA projection of latent states.

    Parameters
    ----------
    latents : torch.Tensor
        Latent states [B, K, D] or [N, D].
    n_components : int
        Number of PCA components.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with PCA results.
    """
    # Flatten if needed
    if latents.dim() == 3:
        B, K, D = latents.shape
        latents_flat = latents.reshape(-1, D)  # [B*K, D]
    else:
        latents_flat = latents
        B, K = 1, latents.shape[0]
        D = latents.shape[1]

    # Center data
    mean = latents_flat.mean(dim=0, keepdim=True)
    centered = latents_flat - mean

    # SVD for PCA
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Project to top components
    components = Vh[:n_components]  # [n_components, D]
    projected = centered @ components.T  # [N, n_components]

    # Explained variance
    total_var = (S ** 2).sum()
    explained_var = (S[:n_components] ** 2) / total_var

    # Reshape projected back to [B, K, n_components] if needed
    if B > 1 or K > 1:
        projected = projected.reshape(B, K, n_components)

    return {
        "projected": projected,
        "components": components,
        "explained_variance_ratio": explained_var,
        "singular_values": S[:n_components],
        "mean": mean.squeeze(0),
    }


def compute_latent_similarity(
    latents: torch.Tensor,
    method: str = "cosine",
) -> Dict[str, torch.Tensor]:
    """Compute pairwise similarity between latent slots.

    Parameters
    ----------
    latents : torch.Tensor
        Latent states [B, K, D] or [K, D].
    method : str
        Similarity method: "cosine" or "euclidean".

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with similarity matrix and statistics.
    """
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)

    B, K, D = latents.shape

    if method == "cosine":
        # Normalize latents
        latents_norm = latents / (latents.norm(dim=-1, keepdim=True) + 1e-8)
        # Cosine similarity
        similarity = torch.bmm(latents_norm, latents_norm.transpose(1, 2))  # [B, K, K]
    elif method == "euclidean":
        # Euclidean distance -> similarity
        diff = latents.unsqueeze(2) - latents.unsqueeze(1)  # [B, K, K, D]
        distance = diff.norm(dim=-1)  # [B, K, K]
        similarity = 1 / (1 + distance)  # Convert to similarity
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute statistics (excluding diagonal)
    mask = 1 - torch.eye(K, device=similarity.device).unsqueeze(0)
    masked_sim = similarity * mask

    mean_similarity = masked_sim.sum(dim=(1, 2)) / mask.sum()
    max_similarity = (masked_sim - (1 - mask) * 1e10).max(dim=-1).values.max(dim=-1).values
    min_similarity = (masked_sim + (1 - mask) * 1e10).min(dim=-1).values.min(dim=-1).values

    return {
        "similarity_matrix": similarity,  # [B, K, K]
        "mean_similarity": mean_similarity,  # [B]
        "max_similarity": max_similarity,  # [B]
        "min_similarity": min_similarity,  # [B]
    }


def track_latent_trajectory(
    latents_over_time: List[torch.Tensor],
    reference_latents: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Track latent trajectory over time/layers.

    Parameters
    ----------
    latents_over_time : List[torch.Tensor]
        List of latent states [B, K, D] at each timestep/layer.
    reference_latents : torch.Tensor, optional
        Reference latents for computing drift.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with trajectory metrics.
    """
    if not latents_over_time:
        return {}

    T = len(latents_over_time)
    B, K, D = latents_over_time[0].shape

    # Stack into tensor [T, B, K, D]
    trajectory = torch.stack(latents_over_time, dim=0)

    # Compute step-to-step changes
    if T > 1:
        changes = trajectory[1:] - trajectory[:-1]  # [T-1, B, K, D]
        step_norms = changes.norm(dim=-1)  # [T-1, B, K]

        mean_step_size = step_norms.mean(dim=(1, 2))  # [T-1]
        max_step_size = step_norms.max(dim=-1).values.max(dim=-1).values  # [T-1]
    else:
        mean_step_size = torch.zeros(0)
        max_step_size = torch.zeros(0)

    # Compute drift from reference
    if reference_latents is not None:
        drift = trajectory - reference_latents.unsqueeze(0)  # [T, B, K, D]
        drift_norms = drift.norm(dim=-1).mean(dim=(1, 2))  # [T]
    else:
        drift_norms = trajectory.norm(dim=-1).mean(dim=(1, 2))  # [T]

    # Compute total path length
    if T > 1:
        path_length = step_norms.sum(dim=0).mean()  # Average over B, K
    else:
        path_length = torch.tensor(0.0)

    # Final displacement
    if T > 1:
        displacement = (trajectory[-1] - trajectory[0]).norm(dim=-1).mean()
    else:
        displacement = torch.tensor(0.0)

    return {
        "trajectory": trajectory,  # [T, B, K, D]
        "mean_step_size": mean_step_size,  # [T-1]
        "max_step_size": max_step_size,  # [T-1]
        "drift_norms": drift_norms,  # [T]
        "path_length": path_length.item(),
        "displacement": displacement.item(),
        "tortuosity": (path_length / (displacement + 1e-8)).item() if T > 1 else 1.0,
    }


class LatentTracker:
    """Track latent statistics over training."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history: List[Dict[str, float]] = []
        self.steps: List[int] = []
        self.pca_components: Optional[torch.Tensor] = None
        self.pca_mean: Optional[torch.Tensor] = None

    def update(self, step: int, latents: torch.Tensor):
        """Record latent statistics for a training step."""
        # Compute PCA
        pca_result = compute_latent_pca(latents, n_components=3)

        # Store first PCA for consistent projection
        if self.pca_components is None:
            self.pca_components = pca_result["components"].detach().clone()
            self.pca_mean = pca_result["mean"].detach().clone()

        # Compute similarity
        sim_result = compute_latent_similarity(latents)

        record = {
            "step": step,
            "explained_var_pc1": pca_result["explained_variance_ratio"][0].item(),
            "explained_var_pc2": pca_result["explained_variance_ratio"][1].item(),
            "explained_var_pc3": pca_result["explained_variance_ratio"][2].item(),
            "mean_similarity": sim_result["mean_similarity"].mean().item(),
            "latent_norm": latents.norm(dim=-1).mean().item(),
        }

        self.history.append(record)
        self.steps.append(step)

        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.steps = self.steps[-self.max_history:]

    def project_to_pca(self, latents: torch.Tensor) -> torch.Tensor:
        """Project latents using stored PCA components."""
        if self.pca_components is None:
            return compute_latent_pca(latents)["projected"]

        # Use stored components
        if latents.dim() == 3:
            B, K, D = latents.shape
            latents_flat = latents.reshape(-1, D)
        else:
            latents_flat = latents
            B, K = 1, latents.shape[0]

        centered = latents_flat - self.pca_mean.to(latents.device)
        projected = centered @ self.pca_components.T.to(latents.device)

        if B > 1 or K > 1:
            projected = projected.reshape(B, K, -1)

        return projected

    def get_history(self, key: str) -> Tuple[List[int], List[float]]:
        """Get history for a specific metric."""
        values = [h[key] for h in self.history if key in h]
        steps = self.steps[:len(values)]
        return steps, values
