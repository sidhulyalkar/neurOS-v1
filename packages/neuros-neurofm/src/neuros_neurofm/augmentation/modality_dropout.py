"""
Modality Dropout and Augmentation for NeuroFMX
Implements robust cross-modal learning through strategic modality dropout and augmentation
"""

import random
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import numpy as np


class ModalityDropout(nn.Module):
    """
    Randomly drop entire modalities during training

    This forces the model to learn robust representations that don't
    over-rely on any single modality.

    Example:
        >>> dropout = ModalityDropout(dropout_prob=0.2, min_modalities=1)
        >>> modality_dict = {"eeg": tensor1, "spikes": tensor2, "video": tensor3}
        >>> filtered = dropout(modality_dict)  # May drop 1-2 modalities
    """

    def __init__(
        self,
        dropout_prob: float = 0.1,
        min_modalities: int = 1,
        protected_modalities: Optional[List[str]] = None,
    ):
        """
        Args:
            dropout_prob: Probability of dropping each modality
            min_modalities: Minimum number of modalities to keep
            protected_modalities: Modalities that are never dropped
        """
        super().__init__()
        self.dropout_prob = dropout_prob
        self.min_modalities = min_modalities
        self.protected_modalities = protected_modalities or []

    def forward(
        self,
        modality_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply modality dropout

        Args:
            modality_dict: Dictionary mapping modality names to tensors

        Returns:
            Filtered dictionary with some modalities dropped
        """
        if not self.training:
            return modality_dict

        # Separate protected and droppable modalities
        protected = {k: v for k, v in modality_dict.items() if k in self.protected_modalities}
        droppable = {k: v for k, v in modality_dict.items() if k not in self.protected_modalities}

        if not droppable:
            return modality_dict

        # Determine how many modalities to keep
        n_droppable = len(droppable)
        n_protected = len(protected)

        # Ensure we keep at least min_modalities total
        min_to_keep = max(1, self.min_modalities - n_protected)
        max_to_drop = max(0, n_droppable - min_to_keep)

        # Randomly select which modalities to keep
        keys = list(droppable.keys())
        n_to_drop = sum(random.random() < self.dropout_prob for _ in range(max_to_drop))
        n_to_drop = min(n_to_drop, max_to_drop)

        if n_to_drop > 0:
            keys_to_drop = random.sample(keys, n_to_drop)
            droppable = {k: v for k, v in droppable.items() if k not in keys_to_drop}

        # Combine protected and kept modalities
        return {**protected, **droppable}


class NeuralAugmentation(nn.Module):
    """
    SpecAugment-style augmentation for neural signals

    Applies time masking, channel masking, and noise injection
    to improve robustness of neural data representations.

    Example:
        >>> aug = NeuralAugmentation(
        ...     time_mask_param=20,
        ...     channel_mask_param=5,
        ...     noise_std=0.1
        ... )
        >>> x = torch.randn(32, 128, 64)  # (batch, time, channels)
        >>> x_aug = aug(x)
    """

    def __init__(
        self,
        time_mask_param: int = 20,
        channel_mask_param: int = 5,
        num_time_masks: int = 2,
        num_channel_masks: int = 2,
        noise_std: float = 0.05,
        apply_time_mask: bool = True,
        apply_channel_mask: bool = True,
        apply_noise: bool = True,
    ):
        """
        Args:
            time_mask_param: Maximum time steps to mask
            channel_mask_param: Maximum channels to mask
            num_time_masks: Number of time mask regions
            num_channel_masks: Number of channel mask regions
            noise_std: Standard deviation of Gaussian noise
            apply_time_mask: Enable time masking
            apply_channel_mask: Enable channel masking
            apply_noise: Enable noise injection
        """
        super().__init__()
        self.time_mask_param = time_mask_param
        self.channel_mask_param = channel_mask_param
        self.num_time_masks = num_time_masks
        self.num_channel_masks = num_channel_masks
        self.noise_std = noise_std
        self.apply_time_mask = apply_time_mask
        self.apply_channel_mask = apply_channel_mask
        self.apply_noise = apply_noise

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply augmentation

        Args:
            x: Input tensor of shape (B, T, C) or (B, C, T)
            mask: Optional attention mask (B, T)

        Returns:
            Augmented tensor
        """
        if not self.training:
            return x

        B, T, C = x.shape
        x = x.clone()

        # Time masking
        if self.apply_time_mask:
            for _ in range(self.num_time_masks):
                t = random.randint(0, min(self.time_mask_param, T))
                t0 = random.randint(0, T - t)
                x[:, t0:t0+t, :] = 0

        # Channel masking
        if self.apply_channel_mask:
            for _ in range(self.num_channel_masks):
                c = random.randint(0, min(self.channel_mask_param, C))
                c0 = random.randint(0, C - c)
                x[:, :, c0:c0+c] = 0

        # Gaussian noise
        if self.apply_noise and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x


class TimeWarping(nn.Module):
    """
    Time warping augmentation for temporal signals

    Stretches and compresses time randomly to improve temporal
    invariance.
    """

    def __init__(self, warp_factor: float = 0.1):
        """
        Args:
            warp_factor: Maximum warp factor (0.1 = ±10% time stretch)
        """
        super().__init__()
        self.warp_factor = warp_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping

        Args:
            x: Input tensor (B, T, C)

        Returns:
            Time-warped tensor
        """
        if not self.training or self.warp_factor == 0:
            return x

        B, T, C = x.shape

        # Generate random warp factor per sample
        warp = 1.0 + (torch.rand(B, device=x.device) * 2 - 1) * self.warp_factor

        # Interpolate to new length
        warped = []
        for i in range(B):
            new_T = int(T * warp[i])
            if new_T == T:
                warped.append(x[i])
            else:
                # Interpolate
                x_i = x[i].unsqueeze(0).transpose(1, 2)  # (1, C, T)
                x_warped = torch.nn.functional.interpolate(
                    x_i, size=new_T, mode='linear', align_corners=False
                )

                # Pad or crop to original length
                if new_T > T:
                    x_warped = x_warped[:, :, :T]
                else:
                    pad = T - new_T
                    x_warped = torch.nn.functional.pad(x_warped, (0, pad))

                warped.append(x_warped.transpose(1, 2).squeeze(0))

        return torch.stack(warped)


class MixUp(nn.Module):
    """
    MixUp augmentation for neural data

    Mixes pairs of samples with random weights to improve
    generalization.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter (larger = more mixing)
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, tuple]:
        """
        Apply mixup

        Args:
            x: Input tensor (B, ...)
            labels: Optional labels (B, ...)

        Returns:
            Mixed tensor (and mixed labels if provided)
        """
        if not self.training or self.alpha == 0:
            if labels is not None:
                return x, labels
            return x

        batch_size = x.size(0)

        # Sample mixing weight from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        # Random permutation
        index = torch.randperm(batch_size, device=x.device)

        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index]

        if labels is not None:
            mixed_labels = lam * labels + (1 - lam) * labels[index]
            return mixed_x, mixed_labels

        return mixed_x


class MultiModalAugmentation(nn.Module):
    """
    Combined augmentation pipeline for multi-modal neural data

    Applies modality dropout + per-modality neural augmentation

    Example:
        >>> aug = MultiModalAugmentation(
        ...     modality_dropout_prob=0.1,
        ...     time_mask_param=20,
        ...     channel_mask_param=5
        ... )
        >>> batch = {
        ...     "eeg": torch.randn(32, 128, 64),
        ...     "spikes": torch.randn(32, 128, 100),
        ...     "video": torch.randn(32, 128, 512)
        ... }
        >>> batch_aug = aug(batch)
    """

    def __init__(
        self,
        modality_dropout_prob: float = 0.1,
        min_modalities: int = 1,
        time_mask_param: int = 20,
        channel_mask_param: int = 5,
        noise_std: float = 0.05,
        apply_time_warp: bool = False,
        warp_factor: float = 0.1,
        apply_mixup: bool = False,
        mixup_alpha: float = 0.2,
    ):
        """
        Args:
            modality_dropout_prob: Probability of dropping each modality
            min_modalities: Minimum modalities to keep
            time_mask_param: Time masking parameter
            channel_mask_param: Channel masking parameter
            noise_std: Noise standard deviation
            apply_time_warp: Enable time warping
            warp_factor: Time warp factor
            apply_mixup: Enable mixup
            mixup_alpha: Mixup alpha parameter
        """
        super().__init__()

        self.modality_dropout = ModalityDropout(
            dropout_prob=modality_dropout_prob,
            min_modalities=min_modalities
        )

        self.neural_aug = NeuralAugmentation(
            time_mask_param=time_mask_param,
            channel_mask_param=channel_mask_param,
            noise_std=noise_std
        )

        self.time_warp = TimeWarping(warp_factor) if apply_time_warp else None
        self.mixup = MixUp(mixup_alpha) if apply_mixup else None

    def forward(
        self,
        modality_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply full augmentation pipeline

        Args:
            modality_dict: Dictionary of modality tensors

        Returns:
            Augmented dictionary
        """
        # Modality dropout
        modality_dict = self.modality_dropout(modality_dict)

        # Per-modality augmentation
        augmented = {}
        for modality, tensor in modality_dict.items():
            x = tensor

            # Neural augmentation (time/channel masking, noise)
            x = self.neural_aug(x)

            # Time warping
            if self.time_warp is not None:
                x = self.time_warp(x)

            # Mixup
            if self.mixup is not None:
                x = self.mixup(x)

            augmented[modality] = x

        return augmented


# Example usage
if __name__ == "__main__":
    print("Testing Multi-Modal Augmentation")
    print("=" * 80)

    # Create augmentation pipeline
    aug = MultiModalAugmentation(
        modality_dropout_prob=0.2,
        time_mask_param=20,
        channel_mask_param=5,
        noise_std=0.05,
    )

    # Create sample batch
    batch = {
        "eeg": torch.randn(4, 100, 32),
        "spikes": torch.randn(4, 100, 64),
        "lfp": torch.randn(4, 100, 16),
        "video": torch.randn(4, 100, 512),
    }

    print(f"\nOriginal batch:")
    for mod, tensor in batch.items():
        print(f"  {mod}: {tensor.shape}")

    # Apply augmentation
    aug.train()
    batch_aug = aug(batch)

    print(f"\nAugmented batch:")
    for mod, tensor in batch_aug.items():
        print(f"  {mod}: {tensor.shape}")

    print(f"\nModalities dropped: {set(batch.keys()) - set(batch_aug.keys())}")
