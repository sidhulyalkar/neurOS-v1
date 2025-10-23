"""
Enhanced Training Script with Comprehensive Logging for NeuroFM-X

Features:
- TensorBoard integration
- WandB support (optional)
- Detailed metric tracking
- Gradient monitoring
- Real-time visualization
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Import from full_train_streaming
from full_train_streaming import (
    Config, load_allen_dataset, StreamingNeuropixelsDataset,
    collate_fn, Trainer as BaseTrainer
)
from neuros_neurofm.models.neurofmx_multitask import NeuroFMXMultiTask
from torch.utils.data import DataLoader

# Optional: WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available. Install with: pip install wandb")


class EnhancedTrainer(BaseTrainer):
    """Enhanced trainer with comprehensive logging."""

    def __init__(self, model, config, train_loader, val_loader, use_wandb=False):
        super().__init__(model, config, train_loader, val_loader)

        # Setup logging directories
        self.log_dir = config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

        # WandB initialization
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project="neurofmx-foundation",
                config={
                    'd_model': config.d_model,
                    'n_mamba_blocks': config.n_mamba_blocks,
                    'n_latents': config.n_latents,
                    'latent_dim': config.latent_dim,
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'max_epochs': config.max_epochs,
                    'max_units': config.max_units,
                },
                name=f"neurofmx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            wandb.watch(self.model, log='all', log_freq=100)

        # Metrics storage
        self.train_metrics_history = []
        self.val_metrics_history = []

        # Best metrics tracking
        self.best_metrics = {
            'val_loss': float('inf'),
            'train_loss': float('inf'),
            'rec_loss': float('inf'),
        }

        print(f"✓ Logging initialized:")
        print(f"   TensorBoard: {self.log_dir}")
        if self.use_wandb:
            print(f"   WandB: {wandb.run.get_url()}")

    def train_epoch(self):
        """Enhanced training epoch with detailed logging."""
        if len(self.train_loader) == 0:
            return float('nan')

        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'rec_loss': [],
            'dec_loss': [],
            'total_loss': []
        }

        # Gradient statistics
        grad_norms = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.max_epochs}", leave=False)

        for batch_idx, batch in enumerate(pbar):

            if batch_idx % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                model_output = self.model(
                    tokens_raw=batch['tokens_raw'].to(device),
                    unit_mask=batch['unit_mask'].to(device),
                    unit_indices=batch['unit_indices'].to(device),
                    task='multi-task'
                )

                loss, loss_dict = self.compute_loss(model_output, batch)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Unscale for gradient clipping
                self.scaler.unscale_(self.optimizer)

                # Compute gradient norm before clipping
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                grad_norms.append(total_norm.item())

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1

                # Log to TensorBoard (every N steps)
                if self.global_step % 10 == 0:
                    self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.global_step)
                    self.writer.add_scalar('train/gradient_norm', total_norm.item(), self.global_step)

                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)

                    if self.use_wandb:
                        wandb.log({
                            'train/learning_rate': self.scheduler.get_last_lr()[0],
                            'train/gradient_norm': total_norm.item(),
                            **{f'train/{k}': v for k, v in loss_dict.items()},
                            'global_step': self.global_step
                        })

            # Store metrics
            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
            for key, value in loss_dict.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{np.mean(epoch_losses[-100:]):.4f}",
                'grad': f"{grad_norms[-1]:.3f}" if grad_norms else "N/A",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                **{k: f"{v:.4f}" for k, v in loss_dict.items()}
            })

        # Epoch-level logging
        avg_train_loss = np.mean(epoch_losses)
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

        # Log epoch metrics
        self.writer.add_scalar('epoch/train_loss', avg_train_loss, self.epoch)
        self.writer.add_scalar('epoch/avg_gradient_norm', avg_grad_norm, self.epoch)

        for key, values in epoch_metrics.items():
            if values:
                avg_value = np.mean(values)
                self.writer.add_scalar(f'epoch/train_{key}', avg_value, self.epoch)

        # Store metrics
        self.train_metrics_history.append({
            'epoch': self.epoch,
            'loss': avg_train_loss,
            'grad_norm': avg_grad_norm,
            **{k: np.mean(v) for k, v in epoch_metrics.items() if v}
        })

        return avg_train_loss

    def validate(self):
        """Enhanced validation with detailed metrics."""
        if len(self.val_loader) == 0:
            return float('inf')

        self.model.eval()
        val_losses = []
        val_metrics = {
            'rec_loss': [],
            'dec_loss': []
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    model_output = self.model(
                        tokens_raw=batch['tokens_raw'].to(device),
                        unit_mask=batch['unit_mask'].to(device),
                        unit_indices=batch['unit_indices'].to(device),
                        task='multi-task'
                    )
                    loss, loss_dict = self.compute_loss(model_output, batch)

                val_losses.append(loss.item())
                for key, value in loss_dict.items():
                    if key in val_metrics:
                        val_metrics[key].append(value)

        avg_val_loss = np.mean(val_losses)

        # Log validation metrics
        self.writer.add_scalar('epoch/val_loss', avg_val_loss, self.epoch)

        for key, values in val_metrics.items():
            if values:
                avg_value = np.mean(values)
                self.writer.add_scalar(f'epoch/val_{key}', avg_value, self.epoch)

        if self.use_wandb:
            wandb.log({
                'epoch/val_loss': avg_val_loss,
                **{f'epoch/val_{k}': np.mean(v) for k, v in val_metrics.items() if v},
                'epoch': self.epoch
            })

        # Store metrics
        self.val_metrics_history.append({
            'epoch': self.epoch,
            'loss': avg_val_loss,
            **{k: np.mean(v) for k, v in val_metrics.items() if v}
        })

        return avg_val_loss

    def save_checkpoint(self, is_best=False):
        """Enhanced checkpoint saving with metrics."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
        }

        latest_path = self.config.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.config.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (val_loss: {self.best_val_loss:.4f})")

        # Save epoch checkpoint every 5 epochs
        if (self.epoch + 1) % 5 == 0:
            epoch_path = self.config.checkpoint_dir / f'epoch_{self.epoch + 1}.pt'
            torch.save(checkpoint, epoch_path)

    def train(self):
        """Enhanced training loop with comprehensive logging."""
        print(f"\n{'='*80}")
        print("Starting Enhanced Training (FULL NEUROFMX-MAMBA)")
        print(f"{'='*80}\n")

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch

            train_loss = self.train_epoch()

            if np.isnan(train_loss):
                print("\n⚠️  Training aborted due to NaN loss or insufficient data.")
                break

            val_loss = self.validate()

            # Log comparison
            self.writer.add_scalars('loss_comparison', {
                'train': train_loss,
                'val': val_loss
            }, epoch)

            # Update best metrics
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_metrics['val_loss'] = val_loss
                self.best_metrics['train_loss'] = train_loss

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            print(f"   Train Loss: {train_loss:.6f}")
            print(f"   Val Loss:   {val_loss:.6f}")
            print(f"   Best Val:   {self.best_val_loss:.6f}")
            print(f"   LR:         {self.scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)

        # Training complete
        print(f"\n{'='*80}")
        print("Enhanced Training Complete!")
        print(f"{'='*80}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Checkpoints saved to: {self.config.checkpoint_dir}")
        print(f"Logs saved to: {self.log_dir}")

        # Save final metrics
        self._save_final_metrics()

        # Close loggers
        self.writer.close()
        if self.use_wandb:
            wandb.finish()

    def _save_final_metrics(self):
        """Save training metrics to JSON."""
        metrics_file = self.config.log_dir / "training_metrics.json"

        final_metrics = {
            'best_metrics': self.best_metrics,
            'train_history': self.train_metrics_history,
            'val_history': self.val_metrics_history,
            'total_epochs': self.epoch + 1,
            'total_steps': self.global_step,
        }

        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)

        print(f"✓ Training metrics saved to: {metrics_file}")


def main():
    """Main training script with logging."""
    import argparse

    parser = argparse.ArgumentParser(description="Train NeuroFM-X with enhanced logging")
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable WandB logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    device = torch.device('cuda')
    config = Config()

    # Load data
    print("Loading dataset...")
    cache, session_ids = load_allen_dataset(config)

    import random
    random.shuffle(session_ids)
    n_train_sessions = int(len(session_ids) * config.train_split)
    train_session_ids = session_ids[:n_train_sessions]
    val_session_ids = session_ids[n_train_sessions:]

    train_dataset = StreamingNeuropixelsDataset(
        processed_dir=config.data_dir / "processed_sequences_full",
        session_ids=train_session_ids
    )

    val_dataset = StreamingNeuropixelsDataset(
        processed_dir=config.data_dir / "processed_sequences_full",
        session_ids=val_session_ids
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    # Create model
    print("Initializing model...")
    model = NeuroFMXMultiTask(
        d_model=config.d_model,
        n_mamba_blocks=config.n_mamba_blocks,
        n_latents=config.n_latents,
        latent_dim=config.latent_dim,
        n_perceiver_layers=config.n_perceiver_layers,
        n_popt_layers=config.n_popt_layers,
        use_popt=config.use_popt,
        use_multi_rate=config.use_multi_rate,
        downsample_rates=config.downsample_rates,
        enable_decoder=config.enable_decoder,
        enable_encoder=config.enable_encoder,
        enable_contrastive=config.enable_contrastive,
        enable_forecast=config.enable_forecast,
        decoder_output_dim=config.decoder_output_dim,
        encoder_output_dim=config.max_units,
        dropout=config.dropout,
    )

    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize enhanced trainer
    trainer = EnhancedTrainer(model, config, train_loader, val_loader, use_wandb=args.use_wandb)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
