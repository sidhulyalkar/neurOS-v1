"""
Masked Modeling Loss Functions for NeuroFMX

Implements various masking strategies for self-supervised pre-training:
- Random masking (BERT-style)
- Block/temporal masking (SpanBERT-style)
- Adaptive masking based on neural activity
- Per-modality masking strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Literal
import numpy as np


class MaskedModelingLoss(nn.Module):
    """
    Masked token modeling loss for self-supervised pre-training.

    Supports multiple masking strategies optimized for neural data:
    - 'random': Random token masking (BERT-style)
    - 'block': Contiguous temporal block masking (SpanBERT-style)
    - 'adaptive': Variance-based adaptive masking

    Args:
        mask_ratio: Fraction of tokens to mask (0.15-0.75)
        masking_strategy: Masking strategy ('random', 'block', 'adaptive')
        block_size: Mean block size for block masking (in tokens)
        block_size_std: Std dev of block size for variability
        reconstruction_loss: Loss type ('mse' for continuous, 'ce' for discrete)
        normalize_targets: Whether to normalize reconstruction targets
        gradient_clip_val: Max gradient norm (for stability)
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        masking_strategy: Literal['random', 'block', 'adaptive'] = 'random',
        block_size: int = 10,
        block_size_std: float = 3.0,
        reconstruction_loss: Literal['mse', 'ce', 'smooth_l1'] = 'mse',
        normalize_targets: bool = False,
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()

        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy
        self.block_size = block_size
        self.block_size_std = block_size_std
        self.reconstruction_loss = reconstruction_loss
        self.normalize_targets = normalize_targets
        self.gradient_clip_val = gradient_clip_val

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute masked modeling loss.

        Args:
            predictions: Model predictions, shape (batch, seq_len, dim)
            targets: Ground truth targets, shape (batch, seq_len, dim)
            mask: Pre-computed mask (if None, will generate), shape (batch, seq_len)
            attention_mask: Valid token mask, shape (batch, seq_len)

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with loss components and statistics
        """
        batch_size, seq_len, dim = predictions.shape

        # Generate mask if not provided
        if mask is None:
            mask = self.generate_mask(
                batch_size, seq_len,
                targets=targets if self.masking_strategy == 'adaptive' else None,
                attention_mask=attention_mask,
                device=predictions.device
            )

        # Apply attention mask if provided (exclude padding)
        if attention_mask is not None:
            mask = mask & attention_mask.bool()

        # Normalize targets if requested
        if self.normalize_targets:
            targets = F.layer_norm(targets, (dim,))

        # Compute reconstruction loss only on masked tokens
        loss, metrics = self._compute_reconstruction_loss(
            predictions, targets, mask
        )

        # Gradient clipping (per-sample)
        if self.gradient_clip_val > 0 and self.training:
            loss = self._clip_gradients(loss)

        # Add mask statistics
        metrics['mask_ratio'] = mask.float().mean()
        metrics['num_masked'] = mask.sum()

        return loss, metrics

    def generate_mask(
        self,
        batch_size: int,
        seq_len: int,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate mask based on selected strategy.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            targets: Target values (for adaptive masking)
            attention_mask: Valid positions mask
            device: Device to create mask on

        Returns:
            mask: Boolean mask, shape (batch, seq_len), True = masked
        """
        if self.masking_strategy == 'random':
            mask = self._random_mask(batch_size, seq_len, device)

        elif self.masking_strategy == 'block':
            mask = self._block_mask(batch_size, seq_len, device)

        elif self.masking_strategy == 'adaptive':
            if targets is None:
                raise ValueError("Adaptive masking requires targets")
            mask = self._adaptive_mask(targets, attention_mask)

        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

        # Respect attention mask (don't mask padding)
        if attention_mask is not None:
            mask = mask & attention_mask.bool()

        return mask

    def _random_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: str
    ) -> torch.Tensor:
        """Random masking (BERT-style)."""
        mask = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio
        return mask

    def _block_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: str
    ) -> torch.Tensor:
        """
        Block/temporal masking (SpanBERT-style).

        Masks contiguous blocks of tokens, better for temporal coherence.
        """
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for b in range(batch_size):
            masked_count = 0
            target_masked = int(seq_len * self.mask_ratio)

            while masked_count < target_masked:
                # Sample block size from truncated normal
                block_len = max(1, int(np.random.normal(self.block_size, self.block_size_std)))
                block_len = min(block_len, seq_len - masked_count)

                # Sample start position
                start = np.random.randint(0, max(1, seq_len - block_len + 1))
                end = min(start + block_len, seq_len)

                # Mark block as masked
                mask[b, start:end] = True
                masked_count = mask[b].sum().item()

                # Prevent infinite loop
                if masked_count >= target_masked or end >= seq_len:
                    break

        return mask

    def _adaptive_mask(
        self,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Adaptive masking based on neural activity variance.

        Preferentially masks tokens with high variance (more informative).
        """
        batch_size, seq_len, dim = targets.shape

        # Compute token-level variance (across feature dimension)
        token_variance = torch.var(targets, dim=-1)  # (batch, seq_len)

        # Mask out padding if provided
        if attention_mask is not None:
            token_variance = token_variance.masked_fill(~attention_mask.bool(), -float('inf'))

        # Compute per-sample threshold for masking
        num_to_mask = int(seq_len * self.mask_ratio)

        # Get top-k most variable tokens per sample
        mask = torch.zeros_like(token_variance, dtype=torch.bool)
        for b in range(batch_size):
            if attention_mask is not None:
                valid_len = attention_mask[b].sum().item()
                num_to_mask_sample = int(valid_len * self.mask_ratio)
            else:
                num_to_mask_sample = num_to_mask

            if num_to_mask_sample > 0:
                _, top_indices = torch.topk(token_variance[b], num_to_mask_sample)
                mask[b, top_indices] = True

        return mask

    def _compute_reconstruction_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute reconstruction loss on masked tokens."""

        # Extract masked predictions and targets
        masked_predictions = predictions[mask]  # (num_masked, dim)
        masked_targets = targets[mask]  # (num_masked, dim)

        metrics = {}

        if masked_predictions.numel() == 0:
            # No masked tokens (edge case)
            return torch.tensor(0.0, device=predictions.device), metrics

        # Compute loss based on type
        if self.reconstruction_loss == 'mse':
            loss = F.mse_loss(masked_predictions, masked_targets)

        elif self.reconstruction_loss == 'smooth_l1':
            loss = F.smooth_l1_loss(masked_predictions, masked_targets)

        elif self.reconstruction_loss == 'ce':
            # For discrete targets (e.g., binned spike counts)
            # Assume targets are class indices
            loss = F.cross_entropy(
                masked_predictions.view(-1, masked_predictions.shape[-1]),
                masked_targets.long().view(-1)
            )
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss}")

        # Additional metrics
        metrics['reconstruction_loss'] = loss

        # Compute mean absolute error for monitoring
        with torch.no_grad():
            mae = (masked_predictions - masked_targets).abs().mean()
            metrics['mae'] = mae

        return loss, metrics

    def _clip_gradients(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply gradient clipping for stability."""
        # Note: This is a placeholder - actual gradient clipping should be done
        # at the optimizer level. We return loss as-is.
        return loss


class PerModalityMaskedLoss(nn.Module):
    """
    Masked modeling loss with different strategies per modality.

    Allows different masking ratios and strategies for different neural modalities
    (e.g., spikes vs LFP vs EEG vs calcium imaging).

    Args:
        modality_configs: Dict mapping modality names to config dicts
            Each config can have: mask_ratio, masking_strategy, reconstruction_loss
        default_config: Default config for modalities not specified
    """

    def __init__(
        self,
        modality_configs: Dict[str, Dict],
        default_config: Optional[Dict] = None
    ):
        super().__init__()

        if default_config is None:
            default_config = {
                'mask_ratio': 0.15,
                'masking_strategy': 'random',
                'reconstruction_loss': 'mse'
            }

        self.default_config = default_config
        self.modality_configs = modality_configs

        # Create loss modules for each modality
        self.modality_losses = nn.ModuleDict()
        for modality, config in modality_configs.items():
            # Merge with defaults
            full_config = {**default_config, **config}
            self.modality_losses[modality] = MaskedModelingLoss(**full_config)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        modality_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute per-modality masked losses.

        Args:
            predictions: Dict of predictions per modality
            targets: Dict of targets per modality
            masks: Dict of masks per modality (optional)
            attention_masks: Dict of attention masks per modality (optional)
            modality_weights: Dict of weights per modality (for combining)

        Returns:
            total_loss: Combined loss across modalities
            metrics: Dictionary with per-modality and total metrics
        """
        if modality_weights is None:
            modality_weights = {k: 1.0 for k in predictions.keys()}

        total_loss = 0.0
        all_metrics = {}

        for modality in predictions.keys():
            # Get modality-specific loss module
            if modality in self.modality_losses:
                loss_module = self.modality_losses[modality]
            else:
                # Use default config
                loss_module = MaskedModelingLoss(**self.default_config)

            # Get inputs
            pred = predictions[modality]
            target = targets[modality]
            mask = masks.get(modality) if masks else None
            attn_mask = attention_masks.get(modality) if attention_masks else None

            # Compute loss
            loss, metrics = loss_module(pred, target, mask, attn_mask)

            # Weight and accumulate
            weight = modality_weights.get(modality, 1.0)
            total_loss += weight * loss

            # Store metrics
            for key, value in metrics.items():
                all_metrics[f"{modality}_{key}"] = value
            all_metrics[f"{modality}_loss"] = loss
            all_metrics[f"{modality}_weighted_loss"] = weight * loss

        all_metrics['total_loss'] = total_loss

        return total_loss, all_metrics


# Example usage
if __name__ == '__main__':
    # Test basic masked modeling
    batch_size, seq_len, dim = 4, 100, 128

    predictions = torch.randn(batch_size, seq_len, dim)
    targets = torch.randn(batch_size, seq_len, dim)

    # Random masking
    loss_fn = MaskedModelingLoss(mask_ratio=0.15, masking_strategy='random')
    loss, metrics = loss_fn(predictions, targets)
    print(f"Random masking loss: {loss.item():.4f}")
    print(f"Mask ratio: {metrics['mask_ratio'].item():.4f}")

    # Block masking
    loss_fn_block = MaskedModelingLoss(
        mask_ratio=0.30,
        masking_strategy='block',
        block_size=10
    )
    loss, metrics = loss_fn_block(predictions, targets)
    print(f"\nBlock masking loss: {loss.item():.4f}")
    print(f"Mask ratio: {metrics['mask_ratio'].item():.4f}")

    # Adaptive masking
    loss_fn_adaptive = MaskedModelingLoss(
        mask_ratio=0.25,
        masking_strategy='adaptive'
    )
    loss, metrics = loss_fn_adaptive(predictions, targets)
    print(f"\nAdaptive masking loss: {loss.item():.4f}")
    print(f"Mask ratio: {metrics['mask_ratio'].item():.4f}")

    # Per-modality masking
    modality_configs = {
        'spikes': {
            'mask_ratio': 0.50,
            'masking_strategy': 'block',
            'block_size': 20,
            'reconstruction_loss': 'mse'
        },
        'lfp': {
            'mask_ratio': 0.15,
            'masking_strategy': 'random',
            'reconstruction_loss': 'mse'
        },
        'eeg': {
            'mask_ratio': 0.30,
            'masking_strategy': 'adaptive',
            'reconstruction_loss': 'smooth_l1'
        }
    }

    per_modality_loss = PerModalityMaskedLoss(modality_configs)

    # Create multi-modal data
    multi_pred = {
        'spikes': torch.randn(batch_size, seq_len, 64),
        'lfp': torch.randn(batch_size, seq_len, 128),
        'eeg': torch.randn(batch_size, seq_len, 32)
    }
    multi_target = {
        'spikes': torch.randn(batch_size, seq_len, 64),
        'lfp': torch.randn(batch_size, seq_len, 128),
        'eeg': torch.randn(batch_size, seq_len, 32)
    }

    loss, metrics = per_modality_loss(multi_pred, multi_target)
    print(f"\nPer-modality total loss: {loss.item():.4f}")
    for modality in ['spikes', 'lfp', 'eeg']:
        print(f"  {modality} loss: {metrics[f'{modality}_loss'].item():.4f}")
        print(f"  {modality} mask ratio: {metrics[f'{modality}_mask_ratio'].item():.4f}")
