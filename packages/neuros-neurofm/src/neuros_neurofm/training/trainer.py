"""
Training module for NeuroFM-X using PyTorch Lightning.

This module provides a Lightning trainer wrapper for NeuroFM-X
with multi-task learning, mixed precision, and distributed training.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    # Create placeholder
    class pl:
        class LightningModule:
            pass


class NeuroFMXTrainer:
    """Simple trainer wrapper for NeuroFMX.

    This is a placeholder that will be expanded with full
    PyTorch Lightning integration in future commits.

    Parameters
    ----------
    model : nn.Module
        NeuroFMX model to train.
    optimizer : torch.optim.Optimizer, optional
        Optimizer instance.
    lr : float, optional
        Learning rate.
        Default: 3e-4.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 3e-4,
    ):
        self.model = model
        self.lr = lr

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step.

        Parameters
        ----------
        batch : dict
            Batch of data with keys:
            - "tokens": Input tokens, shape (batch, seq_len, d_model)
            - "targets": Target values (task-specific)
            - "attention_mask": Optional mask

        Returns
        -------
        dict
            Loss and metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(
            batch["tokens"],
            attention_mask=batch.get("attention_mask"),
        )

        # Compute loss (placeholder - will add multi-task losses)
        loss = torch.tensor(0.0, device=outputs.device)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
        }

    def eval_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single evaluation step.

        Parameters
        ----------
        batch : dict
            Batch of data.

        Returns
        -------
        dict
            Metrics.
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                batch["tokens"],
                attention_mask=batch.get("attention_mask"),
            )

            # Compute metrics (placeholder)
            loss = torch.tensor(0.0, device=outputs.device)

        return {
            "loss": loss.item(),
        }
