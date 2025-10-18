"""
PyTorch Lightning module for NeuroFM-X training.

Provides a complete Lightning module with multi-task learning,
mixed precision, and distributed training support.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pl = None
    # Create dummy decorator
    def rank_zero_only(fn):
        return fn


class NeuroFMXLightningModule(nn.Module if not LIGHTNING_AVAILABLE else pl.LightningModule):
    """PyTorch Lightning module for NeuroFM-X.

    Handles multi-task training with configurable loss weights,
    mixed precision, and distributed training.

    Parameters
    ----------
    model : nn.Module
        NeuroFMX model instance.
    learning_rate : float, optional
        Learning rate.
        Default: 3e-4.
    weight_decay : float, optional
        Weight decay for AdamW.
        Default: 0.01.
    warmup_epochs : int, optional
        Number of warmup epochs.
        Default: 10.
    max_epochs : int, optional
        Maximum training epochs.
        Default: 100.
    loss_weights : Dict[str, float], optional
        Weights for each task loss.
        Default: {"decoder": 1.0, "encoder": 0.5, "contrastive": 0.3}.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        if not LIGHTNING_AVAILABLE:
            raise ImportError(
                "pytorch-lightning is required for NeuroFMXLightningModule. "
                "Install with: pip install pytorch-lightning"
            )

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Default loss weights
        if loss_weights is None:
            self.loss_weights = {
                "decoder": 1.0,
                "encoder": 0.5,
                "contrastive": 0.3,
            }
        else:
            self.loss_weights = loss_weights

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        batch : dict
            Batch with keys:
            - "tokens": Input tokens
            - "attention_mask": Optional attention mask
            - "behavior": Behavioral targets
            - "neural": Neural targets

        Returns
        -------
        dict
            Model outputs for each task.
        """
        tokens = batch["tokens"]
        attention_mask = batch.get("attention_mask")

        # Get latent representations
        latents = self.model.encode(tokens, attention_mask)

        # Pool latents for task heads
        pooled = latents.mean(dim=1)  # (batch, latent_dim)

        outputs = {}

        # Decoder (neural → behavior)
        if "decoder" in self.model.heads.heads:
            outputs["decoder"] = self.model.heads(pooled, task="decoder")

        # Encoder (behavior → neural)
        if "encoder" in self.model.heads.heads:
            outputs["encoder"] = self.model.heads(pooled, task="encoder")

        # Contrastive
        if "contrastive" in self.model.heads.heads:
            outputs["contrastive"] = self.model.heads(pooled, task="contrastive")

        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task losses.

        Parameters
        ----------
        outputs : dict
            Model outputs.
        batch : dict
            Batch data with targets.

        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of all losses.
        loss_dict : dict
            Individual losses for logging.
        """
        loss_dict = {}
        total_loss = 0.0

        # Decoder loss (MSE for continuous behavior)
        if "decoder" in outputs and "behavior" in batch:
            decoder_loss = F.mse_loss(outputs["decoder"], batch["behavior"])
            loss_dict["decoder_loss"] = decoder_loss
            total_loss += self.loss_weights.get("decoder", 1.0) * decoder_loss

        # Encoder loss (Poisson NLL for firing rates)
        if "encoder" in outputs and "neural" in batch:
            # Use Poisson negative log-likelihood for spike counts
            encoder_loss = F.poisson_nll_loss(
                outputs["encoder"],
                batch["neural"],
                log_input=False,
                reduction="mean",
            )
            loss_dict["encoder_loss"] = encoder_loss
            total_loss += self.loss_weights.get("encoder", 0.5) * encoder_loss

        # Contrastive loss
        if "contrastive" in outputs:
            # Simple contrastive loss (can be enhanced)
            z = outputs["contrastive"]
            # Create positive pairs by shuffling within batch
            batch_size = z.shape[0]
            if batch_size > 1:
                # Similarity matrix
                sim_matrix = torch.matmul(z, z.T) / 0.07  # temperature
                # Targets: positive pairs on diagonal
                targets = torch.arange(batch_size, device=z.device)
                contrastive_loss = F.cross_entropy(sim_matrix, targets)
                loss_dict["contrastive_loss"] = contrastive_loss
                total_loss += self.loss_weights.get("contrastive", 0.3) * contrastive_loss

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Parameters
        ----------
        batch : dict
            Training batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        outputs = self(batch)
        total_loss, loss_dict = self.compute_losses(outputs, batch)

        # Log losses
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step.

        Parameters
        ----------
        batch : dict
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict
            Validation outputs and losses.
        """
        outputs = self(batch)
        total_loss, loss_dict = self.compute_losses(outputs, batch)

        # Log losses
        for key, value in loss_dict.items():
            self.log(f"val/{key}", value, prog_bar=True, sync_dist=True)

        # Compute metrics
        metrics = self.compute_metrics(outputs, batch)
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=True, sync_dist=True)

        return {"loss": total_loss, **loss_dict, **metrics}

    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics.

        Parameters
        ----------
        outputs : dict
            Model outputs.
        batch : dict
            Batch data.

        Returns
        -------
        dict
            Metrics (R², accuracy, etc.).
        """
        metrics = {}

        # Decoder R² (coefficient of determination)
        if "decoder" in outputs and "behavior" in batch:
            y_pred = outputs["decoder"]
            y_true = batch["behavior"]
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            metrics["decoder_r2"] = r2

        # Encoder correlation (average over neurons)
        if "encoder" in outputs and "neural" in batch:
            y_pred = outputs["encoder"]
            y_true = batch["neural"]
            # Pearson correlation per neuron
            pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)
            true_centered = y_true - y_true.mean(dim=0, keepdim=True)
            correlation = (pred_centered * true_centered).sum(dim=0) / (
                torch.sqrt((pred_centered ** 2).sum(dim=0) * (true_centered ** 2).sum(dim=0)) + 1e-8
            )
            metrics["encoder_corr"] = correlation.mean()

        return metrics

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns
        -------
        dict
            Optimizer and scheduler configuration.
        """
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return epoch / self.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/learning_rate", current_lr)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving checkpoint."""
        # Add custom metadata
        checkpoint["neurofmx_version"] = "0.2.0"

    @rank_zero_only
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading checkpoint."""
        # Verify version compatibility
        version = checkpoint.get("neurofmx_version", "unknown")
        print(f"Loading NeuroFM-X checkpoint version: {version}")
