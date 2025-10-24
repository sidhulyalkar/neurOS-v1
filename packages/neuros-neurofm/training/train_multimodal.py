"""
Multimodal Multi-Task Training Script for NeuroFMx

Trains the MultiModalNeuroFMX model with:
- Multiple neural data modalities
- Multi-task learning (decode, encode, contrastive)
- Domain adversarial training for cross-species alignment
- Automatic loss balancing
- Distributed training support
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX
from neuros_neurofm.losses import (
    TriModalContrastiveLoss,
    DomainAdversarialLoss,
    MultiTaskLossManager
)


class MultiModalTrainer:
    """
    Trainer for multimodal multi-task learning.

    Features:
    - Dynamic loss balancing
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing
    - Logging (wandb/tensorboard)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        save_interval: int = 1000,
        checkpoint_dir: str = './checkpoints',
        use_wandb: bool = False,
        wandb_project: str = 'neurofmx',
        **config
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.get('learning_rate', 3e-4),
                weight_decay=config.get('weight_decay', 0.01),
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optimizer

        # Scheduler
        self.scheduler = scheduler

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # Loss functions
        self._setup_losses()

        # Logging
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(project=wandb_project, config=config)
            wandb.watch(model, log='all', log_freq=100)

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _setup_losses(self):
        """Initialize loss functions."""

        # Tri-modal contrastive loss
        self.contrastive_loss = TriModalContrastiveLoss(
            temperature=self.config.get('temperature', 0.07),
            neural_weight=self.config.get('neural_weight', 1.0),
            stimulus_weight=self.config.get('stimulus_weight', 1.0),
            use_temporal=self.config.get('use_temporal', True)
        )

        # Domain adversarial loss
        self.domain_loss = DomainAdversarialLoss()

        # Multi-task loss manager
        task_names = ['decoder', 'encoder', 'contrastive']
        if self.config.get('use_domain_adversarial', False):
            task_names.append('domain')

        self.loss_manager = MultiTaskLossManager(
            task_names=task_names,
            balancing_method=self.config.get('balancing_method', 'uncertainty')
        )

        # Reconstruction losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def train_epoch(self):
        """Train for one epoch."""

        self.model.train()
        self.epoch += 1

        epoch_losses = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            loss_dict = self.train_step(batch)

            epoch_losses.append(loss_dict)

            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_metrics(loss_dict)

            # Checkpointing
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'step_{self.global_step}.pt')

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'step': self.global_step
            })

        # Epoch summary
        avg_losses = self._average_losses(epoch_losses)
        print(f"\nEpoch {self.epoch} Summary:")
        for k, v in avg_losses.items():
            print(f"  {k}: {v:.4f}")

        # Validation
        if self.val_loader is not None:
            val_metrics = self.validate()
            print(f"Validation metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

        return avg_losses

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""

        # Move data to device
        modality_dict = {}
        for modality, data in batch['inputs'].items():
            modality_dict[modality] = data.to(self.device)

        behavior_target = batch.get('behavior', None)
        if behavior_target is not None:
            behavior_target = behavior_target.to(self.device)

        neural_target = batch.get('neural', None)
        if neural_target is not None:
            neural_target = neural_target.to(self.device)

        species_labels = batch.get('species', None)
        if species_labels is not None:
            species_labels = species_labels.to(self.device)

        stimulus_ids = batch.get('stimulus', None)
        if stimulus_ids is not None:
            stimulus_ids = stimulus_ids.to(self.device)

        # Forward pass with mixed precision
        with autocast(enabled=self.mixed_precision):
            outputs = self.model(
                modality_dict,
                task='multi-task',
                species_labels=species_labels if self.config.get('use_domain_adversarial') else None
            )

            # Compute task losses
            task_losses = {}

            # 1. Decoder loss (behavioral prediction)
            if 'decoder' in outputs and behavior_target is not None:
                decoder_pred = outputs['decoder']
                if len(behavior_target.shape) == 3:
                    # Sequence output, take mean over time
                    behavior_target = behavior_target.mean(dim=1)
                task_losses['decoder'] = self.mse_loss(decoder_pred, behavior_target)

            # 2. Encoder loss (neural reconstruction)
            if 'encoder' in outputs and neural_target is not None:
                encoder_pred = outputs['encoder']
                task_losses['encoder'] = self.mse_loss(encoder_pred, neural_target)

            # 3. Contrastive loss
            if 'contrastive' in outputs:
                neural_emb = outputs['contrastive']

                # Get behavior and stimulus embeddings if available
                behavior_emb = None
                stimulus_emb = None

                # For now, use simple contrastive with in-batch negatives
                # TODO: Implement proper tri-modal contrastive with behavior/stimulus
                batch_size = neural_emb.shape[0]
                if batch_size > 1:
                    # Simple InfoNCE with batch as negatives
                    contrastive_loss = self.contrastive_loss.info_nce(
                        neural_emb,
                        neural_emb  # Using same embeddings for simplicity
                    )
                    task_losses['contrastive'] = contrastive_loss

            # 4. Domain adversarial loss
            if 'domain_logits' in outputs and species_labels is not None:
                domain_pred = outputs['domain_logits']
                task_losses['domain'] = self.domain_loss(domain_pred, species_labels)

            # Combine losses with automatic balancing
            if len(task_losses) > 0:
                total_loss, loss_dict = self.loss_manager.compute_loss(task_losses)
            else:
                total_loss = torch.tensor(0.0, device=self.device)
                loss_dict = {'total': 0.0}

            # Scale loss for gradient accumulation
            total_loss = total_loss / self.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Optimizer step (after accumulation)
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

        self.global_step += 1

        return loss_dict

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""

        self.model.eval()

        val_losses = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Similar to train_step but without gradient computation
            modality_dict = {}
            for modality, data in batch['inputs'].items():
                modality_dict[modality] = data.to(self.device)

            outputs = self.model(modality_dict, task='multi-task')

            # Compute losses (simplified)
            losses = {}

            if 'decoder' in outputs and 'behavior' in batch:
                behavior_target = batch['behavior'].to(self.device)
                if len(behavior_target.shape) == 3:
                    behavior_target = behavior_target.mean(dim=1)
                losses['decoder'] = self.mse_loss(
                    outputs['decoder'], behavior_target
                ).item()

            if 'encoder' in outputs and 'neural' in batch:
                neural_target = batch['neural'].to(self.device)
                losses['encoder'] = self.mse_loss(
                    outputs['encoder'], neural_target
                ).item()

            val_losses.append(losses)

        # Average over validation set
        avg_val_losses = self._average_losses(val_losses)

        self.model.train()

        return avg_val_losses

    def _average_losses(self, losses_list: List[Dict]) -> Dict[str, float]:
        """Average losses over batches."""
        if len(losses_list) == 0:
            return {}

        avg_losses = {}
        all_keys = set()
        for loss_dict in losses_list:
            all_keys.update(loss_dict.keys())

        for key in all_keys:
            values = [d[key] for d in losses_list if key in d]
            if len(values) > 0:
                avg_losses[key] = np.mean(values)

        return avg_losses

    def _log_metrics(self, metrics: Dict):
        """Log metrics to wandb/console."""

        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)

        # Also log learning rate
        if self.scheduler is not None:
            lr = self.scheduler.get_last_lr()[0]
            if self.use_wandb:
                wandb.log({'learning_rate': lr}, step=self.global_step)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_manager_state_dict': self.loss_manager.state_dict(),
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'loss_manager_state_dict' in checkpoint:
            self.loss_manager.load_state_dict(checkpoint['loss_manager_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description='Train MultiModalNeuroFMX')

    # Model args
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_mamba_blocks', type=int, default=8)
    parser.add_argument('--n_latents', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=512)

    # Training args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--mixed_precision', action='store_true', default=True)

    # Loss args
    parser.add_argument('--balancing_method', type=str, default='uncertainty',
                       choices=['uncertainty', 'gradnorm', 'manual', 'equal'])
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--use_domain_adversarial', action='store_true')
    parser.add_argument('--n_domains', type=int, default=3)

    # Data args
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--modalities', nargs='+', default=['spike', 'eeg'])

    # Logging args
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='neurofmx')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Create model
    print("Creating MultiModalNeuroFMX model...")
    model = MultiModalNeuroFMX(
        d_model=args.d_model,
        n_mamba_blocks=args.n_mamba_blocks,
        n_latents=args.n_latents,
        latent_dim=args.latent_dim,
        use_domain_adversarial=args.use_domain_adversarial,
        n_domains=args.n_domains
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets and loaders
    # TODO: Implement proper multimodal dataset
    print("Loading data...")
    print("WARNING: Using dummy data loaders. Implement real data loading!")

    # Dummy loaders for now
    from torch.utils.data import TensorDataset

    # Create dummy data
    dummy_spike = torch.randn(1000, 100, 384)
    dummy_eeg = torch.randn(1000, 256, 64)
    dummy_behavior = torch.randn(1000, 10)

    train_dataset = TensorDataset(dummy_spike, dummy_eeg, dummy_behavior)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TensorDataset(
        torch.randn(200, 100, 384),
        torch.randn(200, 256, 64),
        torch.randn(200, 10)
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Convert to proper format
    def collate_fn(batch):
        spike, eeg, behavior = zip(*batch)
        return {
            'inputs': {
                'spike': torch.stack(spike),
                'eeg': torch.stack(eeg)
            },
            'behavior': torch.stack(behavior)
        }

    train_loader.collate_fn = collate_fn
    val_loader.collate_fn = collate_fn

    # Create trainer
    print("Initializing trainer...")
    trainer = MultiModalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        **vars(args)
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        trainer.train_epoch()

    print("Training complete!")


if __name__ == '__main__':
    main()
