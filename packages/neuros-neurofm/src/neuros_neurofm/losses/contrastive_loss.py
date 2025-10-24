"""
Contrastive Learning Losses for NeuroFMx

Implements:
- InfoNCE loss
- Tri-modal contrastive loss (neural + behavior + stimulus)
- Temporal contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss.

    Used for contrastive learning to pull positive pairs together
    and push negative pairs apart in embedding space.

    Args:
        temperature: Temperature parameter for softmax (lower = harder discrimination)
        reduction: How to reduce batch loss ('mean', 'sum', 'none')
    """

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: (batch, dim) anchor embeddings
            positive: (batch, dim) positive embeddings
            negatives: (batch, n_neg, dim) negative embeddings
                      If None, uses other samples in batch as negatives

        Returns:
            loss: Scalar loss value
        """
        batch_size = anchor.shape[0]

        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (batch,)

        # Negative similarities
        if negatives is None:
            # Use other samples in batch as negatives
            # Compute all pairwise similarities
            all_sims = torch.matmul(anchor, positive.T) / self.temperature  # (batch, batch)

            # Create labels (diagonal are positives)
            labels = torch.arange(batch_size, device=anchor.device)

            # Cross-entropy loss
            loss = F.cross_entropy(all_sims, labels, reduction=self.reduction)

        else:
            # Explicit negatives provided
            negatives = F.normalize(negatives, dim=-1)

            # Negative similarities: (batch, n_neg)
            neg_sims = torch.matmul(
                anchor.unsqueeze(1),  # (batch, 1, dim)
                negatives.transpose(1, 2)  # (batch, dim, n_neg)
            ).squeeze(1) / self.temperature  # (batch, n_neg)

            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)  # (batch, 1+n_neg)

            # Labels are always 0 (positive is first)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

            loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class TriModalContrastiveLoss(nn.Module):
    """
    Tri-modal contrastive loss for aligning neural activity, behavior, and stimulus.

    Learns a shared embedding space where:
    - Neural activity at time t aligns with behavior at time t
    - Neural activity during stimulus s aligns with stimulus embedding s
    - Behavior during stimulus s aligns with stimulus embedding s

    This creates a unified representation space across all three modalities.

    Args:
        temperature: Temperature for InfoNCE
        neural_weight: Weight for neural-behavior alignment
        stimulus_weight: Weight for stimulus-related alignments
        use_temporal: Whether to use temporal proximity for positives
        temporal_window: Window size for temporal positives (in timesteps)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        neural_weight: float = 1.0,
        stimulus_weight: float = 1.0,
        use_temporal: bool = True,
        temporal_window: int = 5
    ):
        super().__init__()

        self.temperature = temperature
        self.neural_weight = neural_weight
        self.stimulus_weight = stimulus_weight
        self.use_temporal = use_temporal
        self.temporal_window = temporal_window

        self.info_nce = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        neural_emb: torch.Tensor,
        behavior_emb: Optional[torch.Tensor] = None,
        stimulus_emb: Optional[torch.Tensor] = None,
        temporal_indices: Optional[torch.Tensor] = None,
        stimulus_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute tri-modal contrastive loss.

        Args:
            neural_emb: (batch, dim) neural embeddings
            behavior_emb: (batch, dim) behavior embeddings
            stimulus_emb: (batch, dim) stimulus embeddings
            temporal_indices: (batch,) timestep indices for temporal alignment
            stimulus_ids: (batch,) stimulus IDs for grouping

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        losses = {}
        total_loss = 0.0

        batch_size = neural_emb.shape[0]

        # 1. Neural-Behavior alignment
        if behavior_emb is not None:
            if self.use_temporal and temporal_indices is not None:
                # Use temporal proximity for positives
                nb_loss = self._temporal_contrastive(
                    neural_emb, behavior_emb, temporal_indices
                )
            else:
                # Standard pairwise alignment
                nb_loss = self.info_nce(neural_emb, behavior_emb)

            losses['neural_behavior'] = nb_loss
            total_loss += self.neural_weight * nb_loss

        # 2. Neural-Stimulus alignment
        if stimulus_emb is not None:
            if stimulus_ids is not None:
                # Group by stimulus ID
                ns_loss = self._stimulus_contrastive(
                    neural_emb, stimulus_emb, stimulus_ids
                )
            else:
                ns_loss = self.info_nce(neural_emb, stimulus_emb)

            losses['neural_stimulus'] = ns_loss
            total_loss += self.stimulus_weight * ns_loss

        # 3. Behavior-Stimulus alignment
        if behavior_emb is not None and stimulus_emb is not None:
            if stimulus_ids is not None:
                bs_loss = self._stimulus_contrastive(
                    behavior_emb, stimulus_emb, stimulus_ids
                )
            else:
                bs_loss = self.info_nce(behavior_emb, stimulus_emb)

            losses['behavior_stimulus'] = bs_loss
            total_loss += self.stimulus_weight * bs_loss

        return total_loss, losses

    def _temporal_contrastive(
        self,
        anchor: torch.Tensor,
        target: torch.Tensor,
        temporal_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss using temporal proximity.

        Samples within temporal_window are considered positives.
        """
        batch_size = anchor.shape[0]

        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        target = F.normalize(target, dim=-1)

        # Compute all pairwise similarities
        sim_matrix = torch.matmul(anchor, target.T) / self.temperature  # (batch, batch)

        # Create temporal proximity mask
        time_diff = torch.abs(
            temporal_indices.unsqueeze(1) - temporal_indices.unsqueeze(0)
        )  # (batch, batch)

        positive_mask = (time_diff <= self.temporal_window).float()

        # Exclude self (diagonal)
        positive_mask.fill_diagonal_(0)

        # Negative mask (everything not positive)
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)

        # Compute loss
        # For each anchor, positive log sum exp of positives / sum of all
        pos_exp = torch.exp(sim_matrix) * positive_mask
        neg_exp = torch.exp(sim_matrix) * negative_mask

        # Avoid log(0)
        pos_sum = pos_exp.sum(dim=1) + 1e-8
        all_sum = pos_exp.sum(dim=1) + neg_exp.sum(dim=1) + 1e-8

        loss = -torch.log(pos_sum / all_sum).mean()

        return loss

    def _stimulus_contrastive(
        self,
        anchor: torch.Tensor,
        target: torch.Tensor,
        stimulus_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss grouping by stimulus ID.

        All samples with same stimulus ID are positives.
        """
        batch_size = anchor.shape[0]

        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        target = F.normalize(target, dim=-1)

        # Compute similarities
        sim_matrix = torch.matmul(anchor, target.T) / self.temperature

        # Create stimulus mask
        stimulus_mask = (
            stimulus_ids.unsqueeze(1) == stimulus_ids.unsqueeze(0)
        ).float()

        # Exclude self
        stimulus_mask.fill_diagonal_(0)

        # Negative mask
        negative_mask = 1 - stimulus_mask
        negative_mask.fill_diagonal_(0)

        # Compute loss
        pos_exp = torch.exp(sim_matrix) * stimulus_mask
        neg_exp = torch.exp(sim_matrix) * negative_mask

        pos_sum = pos_exp.sum(dim=1) + 1e-8
        all_sum = pos_exp.sum(dim=1) + neg_exp.sum(dim=1) + 1e-8

        loss = -torch.log(pos_sum / all_sum).mean()

        return loss


class TemporalContrastiveLoss(nn.Module):
    """
    Contrastive loss for temporal sequences.

    Pulls together representations from nearby timepoints,
    pushes apart distant timepoints.

    Useful for learning smooth temporal dynamics.
    """

    def __init__(self, temperature: float = 0.07, positive_window: int = 5):
        super().__init__()
        self.temperature = temperature
        self.positive_window = positive_window

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, time, dim) sequence embeddings

        Returns:
            loss: Temporal contrastive loss
        """
        batch_size, seq_len, dim = embeddings.shape

        # Reshape to (batch*time, dim)
        emb_flat = embeddings.reshape(-1, dim)
        emb_flat = F.normalize(emb_flat, dim=-1)

        # Compute all pairwise similarities
        sim_matrix = torch.matmul(emb_flat, emb_flat.T) / self.temperature

        # Create temporal position matrix
        positions = torch.arange(seq_len, device=embeddings.device)
        positions = positions.unsqueeze(0).repeat(batch_size, 1)  # (batch, time)
        positions_flat = positions.reshape(-1)  # (batch*time,)

        # Temporal distance matrix
        time_diff = torch.abs(
            positions_flat.unsqueeze(1) - positions_flat.unsqueeze(0)
        )

        # Positive mask (within window, same batch)
        batch_ids = torch.arange(batch_size, device=embeddings.device)
        batch_ids = batch_ids.unsqueeze(1).repeat(1, seq_len).reshape(-1)

        same_batch = (batch_ids.unsqueeze(1) == batch_ids.unsqueeze(0)).float()
        positive_mask = ((time_diff <= self.positive_window) & (time_diff > 0)).float() * same_batch

        # Negative mask (different batch or distant in time)
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)

        # Loss computation
        pos_exp = torch.exp(sim_matrix) * positive_mask
        neg_exp = torch.exp(sim_matrix) * negative_mask

        pos_sum = pos_exp.sum(dim=1) + 1e-8
        all_sum = pos_exp.sum(dim=1) + neg_exp.sum(dim=1) + 1e-8

        loss = -torch.log(pos_sum / all_sum).mean()

        return loss


# Example usage
if __name__ == '__main__':
    # Test InfoNCE
    batch_size = 16
    dim = 128

    anchor = torch.randn(batch_size, dim)
    positive = torch.randn(batch_size, dim)

    info_nce = InfoNCELoss(temperature=0.07)
    loss = info_nce(anchor, positive)
    print(f"InfoNCE loss: {loss.item():.4f}")

    # Test Tri-modal
    neural = torch.randn(batch_size, dim)
    behavior = torch.randn(batch_size, dim)
    stimulus = torch.randn(batch_size, dim)

    temporal_idx = torch.arange(batch_size)
    stimulus_ids = torch.randint(0, 5, (batch_size,))

    tri_loss = TriModalContrastiveLoss()
    total, losses = tri_loss(neural, behavior, stimulus, temporal_idx, stimulus_ids)

    print(f"Tri-modal loss: {total.item():.4f}")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
