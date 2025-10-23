"""
Unified Training Script for NeuroFM-X Foundation Model

Supports:
- Local training (RTX 3070 Ti)
- Cloud training (AWS A100, GCP TPU)
- Multi-modal datasets
- Distributed training
- Configuration-based setup

Usage:
    # Quick test (4 sessions, 2-3 hours)
    python training/train.py --config configs/quick_test.yaml

    # Full local training (20 sessions, 12-17 hours)
    python training/train.py --config configs/local_full.yaml

    # Cloud training (200+ sessions, multi-modal)
    python training/train.py --config configs/cloud_aws_a100.yaml

    # Resume from checkpoint
    python training/train.py --config configs/local_full.yaml --resume checkpoints/latest.pt
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neuros_neurofm.models.neurofmx_multitask import NeuroFMXMultiTask

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ConfigurableTrainer:
    """Trainer that works with YAML configurations."""

    def __init__(self, config: Dict, model, train_loader, val_loader, device='cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup mixed precision
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Setup logging
        self._setup_logging()

        # Checkpointing
        checkpoint_config = config.get('checkpointing', {})
        self.checkpoint_dir = Path(checkpoint_config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = checkpoint_config.get('save_interval', 5)

        print(f"\n{'='*80}")
        print(f"Trainer Initialized: {config['name']}")
        print(f"{'='*80}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Training batches: {len(train_loader):,}")
        print(f"Validation batches: {len(val_loader):,}")
        print(f"Effective batch size: {self._get_effective_batch_size()}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")

    def _create_optimizer(self):
        """Create optimizer from config."""
        training_config = self.config['training']
        opt_config = training_config.get('optimizer', {})

        opt_type = opt_config.get('type', 'adamw').lower()
        lr = training_config['learning_rate']
        weight_decay = training_config.get('weight_decay', 0.01)

        if opt_type == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999)),
                eps=opt_config.get('eps', 1e-8)
            )
        elif opt_type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

    def _create_scheduler(self):
        """Create learning rate scheduler from config."""
        training_config = self.config['training']
        lr_config = training_config.get('lr_scheduler', {})

        scheduler_type = lr_config.get('type', 'onecycle').lower()

        if scheduler_type == 'onecycle':
            steps_per_epoch = max(1, len(self.train_loader) // training_config.get('gradient_accumulation_steps', 1))
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=training_config['learning_rate'],
                epochs=training_config['max_epochs'],
                steps_per_epoch=steps_per_epoch,
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=lr_config.get('T_0', 10),
                T_mult=lr_config.get('T_mult', 2),
                eta_min=lr_config.get('min_lr', 1e-6)
            )
        else:
            # Constant LR
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

    def _setup_logging(self):
        """Setup TensorBoard and WandB logging."""
        log_config = self.config.get('logging', {})
        log_dir = Path(self.config.get('checkpointing', {}).get('log_dir', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        if log_config.get('use_tensorboard', True):
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            self.writer = None

        # WandB
        self.use_wandb = log_config.get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb_config = log_config.get('wandb', {})
            wandb.init(
                project=wandb_config.get('project', 'neurofm'),
                entity=wandb_config.get('entity'),
                name=self.config['name'],
                config=self.config,
                tags=wandb_config.get('tags', [])
            )
            wandb.watch(self.model, log='all', log_freq=100)

        self.log_interval = log_config.get('log_interval', 50)

    def _get_effective_batch_size(self):
        """Calculate effective batch size."""
        data_config = self.config['data']
        batch_size = data_config['batch_size']
        grad_accum = self.config['training'].get('gradient_accumulation_steps', 1)
        return batch_size * grad_accum

    def train(self):
        """Main training loop."""
        max_epochs = self.config['training']['max_epochs']

        print(f"\n{'='*80}")
        print(f"Starting Training: {self.config['name']}")
        print(f"{'='*80}\n")

        for epoch in range(max_epochs):
            self.epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Best Val:   {self.best_val_loss:.6f}")
            print(f"  LR:         {self.scheduler.get_last_lr()[0]:.2e}")

            # Check early stopping
            if self._check_early_stopping(val_loss):
                print(f"\n✓ Early stopping triggered at epoch {epoch + 1}")
                break

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % self.save_interval == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*80}\n")

        # Cleanup
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()

    def train_epoch(self):
        """Train for one epoch."""
        import torch.nn.functional as F
        from tqdm import tqdm
        import numpy as np

        if len(self.train_loader) == 0:
            return 0.0

        self.model.train()
        epoch_losses = []
        grad_accum = self.config['training'].get('gradient_accumulation_steps', 1)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            if batch_idx % grad_accum == 0:
                self.optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                model_output = self.model(
                    tokens_raw=batch['tokens_raw'].to(self.device),
                    unit_mask=batch['unit_mask'].to(self.device),
                    unit_indices=batch['unit_indices'].to(self.device),
                    task='multi-task'
                )

                # Compute loss
                loss = self._compute_loss(model_output, batch)
                loss = loss / grad_accum

            # Backward pass
            self.scaler.scale(loss).backward()

            # Optimizer step
            if (batch_idx + 1) % grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('gradient_clip_norm', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1

            epoch_losses.append(loss.item() * grad_accum)

            # Logging
            if self.global_step % self.log_interval == 0:
                if self.writer:
                    self.writer.add_scalar('train/loss', loss.item() * grad_accum, self.global_step)
                    self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

            pbar.set_postfix({'loss': f"{np.mean(epoch_losses[-100:]):.4f}"})

        return np.mean(epoch_losses)

    def validate(self):
        """Validation loop."""
        import torch.nn.functional as F
        from tqdm import tqdm
        import numpy as np

        if len(self.val_loader) == 0:
            return 0.0

        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    model_output = self.model(
                        tokens_raw=batch['tokens_raw'].to(self.device),
                        unit_mask=batch['unit_mask'].to(self.device),
                        unit_indices=batch['unit_indices'].to(self.device),
                        task='multi-task'
                    )

                    loss = self._compute_loss(model_output, batch)

                val_losses.append(loss.item())

        return np.mean(val_losses) if val_losses else 0.0

    def _compute_loss(self, model_output, batch):
        """Compute multi-task loss."""
        import torch.nn.functional as F

        total_loss = torch.tensor(0.0, device=self.device)
        task_config = self.config.get('tasks', {})
        loss_weights = task_config.get('loss_weights', {})

        # Reconstruction loss (encoder head)
        if 'encoder' in model_output:
            reconstructed = model_output['encoder']
            spikes = batch['tokens_raw'].to(self.device)
            unit_mask_expanded = (~batch['unit_mask'].to(self.device)).unsqueeze(1).expand(-1, spikes.shape[1], -1).float()

            rec_loss = F.mse_loss(
                reconstructed * unit_mask_expanded,
                spikes * unit_mask_expanded,
                reduction='sum'
            )
            rec_loss = rec_loss / (unit_mask_expanded.sum() + 1e-6)
            total_loss += rec_loss * loss_weights.get('encoder', 1.0)

        # Decoder loss
        if 'decoder' in model_output:
            predicted_behavior = model_output['decoder']
            target_behavior = batch['behavior_target'][:, -1, :].to(self.device)
            dec_loss = F.mse_loss(predicted_behavior, target_behavior)
            total_loss += dec_loss * loss_weights.get('decoder', 1.0)

        return total_loss

    def save_checkpoint(self, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (val_loss: {self.best_val_loss:.6f})")

    def _check_early_stopping(self, val_loss):
        """Check if early stopping criteria met."""
        es_config = self.config['training'].get('early_stopping', {})
        if not es_config.get('enabled', False):
            return False

        # Initialize early stopping tracker
        if not hasattr(self, '_es_patience_counter'):
            self._es_patience_counter = 0
            self._es_best_loss = float('inf')

        patience = es_config.get('patience', 5)
        min_delta = es_config.get('min_delta', 0.0001)

        # Check if loss improved
        if val_loss < (self._es_best_loss - min_delta):
            self._es_best_loss = val_loss
            self._es_patience_counter = 0
        else:
            self._es_patience_counter += 1

        # Trigger early stopping if patience exceeded
        if self._es_patience_counter >= patience:
            return True

        return False


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: Dict) -> nn.Module:
    """Create NeuroFMX model from configuration."""
    model_config = config['model']
    task_config = config['tasks']
    data_config = config['data']

    model = NeuroFMXMultiTask(
        d_model=model_config['d_model'],
        n_mamba_blocks=model_config['n_mamba_blocks'],
        n_latents=model_config['n_latents'],
        latent_dim=model_config['latent_dim'],
        n_perceiver_layers=model_config['n_perceiver_layers'],
        n_popt_layers=model_config['n_popt_layers'],
        use_popt=model_config['use_popt'],
        use_multi_rate=model_config['use_multi_rate'],
        downsample_rates=model_config['downsample_rates'],
        enable_decoder=task_config['enable_decoder'],
        enable_encoder=task_config['enable_encoder'],
        enable_contrastive=task_config['enable_contrastive'],
        enable_forecast=task_config['enable_forecast'],
        decoder_output_dim=task_config['decoder_output_dim'],
        encoder_output_dim=task_config['encoder_output_dim'],
        sequence_length=data_config['sequence_length'],
        dropout=model_config['dropout'],
        input_modality=model_config.get('input_modality', 'binned'),
    )

    return model


def create_dataloaders_from_config(config: Dict):
    """Create training and validation dataloaders."""
    # Import here to avoid circular dependencies
    from scripts.data_utils import create_dataloaders

    return create_dataloaders(config)


def main():
    parser = argparse.ArgumentParser(description="Train NeuroFM-X Foundation Model")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Create model
    print("Creating model...")
    model = create_model_from_config(config)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders_from_config(config)

    # Create trainer
    trainer = ConfigurableTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        trainer.best_val_loss = checkpoint['best_val_loss']

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
