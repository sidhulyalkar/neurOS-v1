"""
Training Script for NeuroFM-X on RTX 3070 Ti GPU - Synthetic Data Version
==========================================================================

This script trains NeuroFM-X using synthetic neural data that mimics
real Neuropixels recordings. Perfect for testing the architecture before
downloading real data.

Hardware: RTX 3070 Ti (8GB VRAM)
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Check CUDA availability
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Running on CPU (will be slow).")
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


# ==================== CONFIGURATION ====================

class Config:
    """Training configuration optimized for RTX 3070 Ti."""

    # Paths
    checkpoint_dir = Path("./checkpoints")
    log_dir = Path("./logs")

    # Model architecture (optimized for 8GB VRAM)
    d_model = 256
    n_transformer_blocks = 8
    n_latents = 64
    latent_dim = 256
    dropout = 0.1

    # Training
    batch_size = 8
    gradient_accumulation_steps = 4  # Effective batch size: 8 * 4 = 32
    learning_rate = 3e-4
    weight_decay = 0.01
    max_epochs = 20
    warmup_epochs = 2

    # Mixed precision
    use_amp = True  # Automatic Mixed Precision

    # Data
    n_train_samples = 5000
    n_val_samples = 1000
    n_units = 384  # Number of simulated neurons
    sequence_length = 100  # 1 second at 100 Hz
    bin_size_ms = 10.0

    # Logging
    log_interval = 50


# ==================== SYNTHETIC DATASET ====================

class SyntheticNeuropixelsDataset(Dataset):
    """
    Generates synthetic neural data that mimics real Neuropixels recordings.

    Features:
    - Realistic firing rates (1-10 Hz)
    - Temporal structure (smoothly varying rates)
    - Population coupling (correlated activity)
    - Behavior-related modulation
    """

    def __init__(
        self,
        n_samples: int,
        n_units: int = 384,
        sequence_length: int = 100,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_units = n_units
        self.sequence_length = sequence_length

        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"\nGenerating {n_samples} synthetic neural sequences...")
        self.data = self._generate_data()
        print(f"  ✓ Generated {len(self.data)} sequences")
        print(f"  ✓ Shape: ({sequence_length}, {n_units})")
        print(f"  ✓ Mean firing rate: {np.mean([d['spikes'].mean() for d in self.data]):.3f} spikes/bin")

    def _generate_data(self):
        """Generate realistic synthetic neural data."""
        data = []

        for i in tqdm(range(self.n_samples), desc="Generating data"):
            # Generate smooth temporal modulation
            t = np.linspace(0, 2 * np.pi, self.sequence_length)
            temporal_modulation = 0.5 + 0.5 * np.sin(t + np.random.rand() * 2 * np.pi)

            # Generate population structure
            # Some neurons are coupled
            n_populations = 8
            population_ids = np.random.randint(0, n_populations, self.n_units)

            spikes = np.zeros((self.sequence_length, self.n_units), dtype=np.float32)

            for pop_id in range(n_populations):
                pop_mask = population_ids == pop_id
                pop_size = pop_mask.sum()

                if pop_size == 0:
                    continue

                # Population-specific temporal pattern
                pop_modulation = 0.7 + 0.3 * np.sin(2 * t + pop_id)

                # Combined modulation
                combined_mod = temporal_modulation * pop_modulation

                # Base firing rates (log-normal distribution like real neurons)
                base_rates = np.exp(np.random.randn(pop_size) * 0.5 - 1.0) * 0.15

                # Generate Poisson spikes
                for t_idx in range(self.sequence_length):
                    rates = base_rates * combined_mod[t_idx]
                    spikes[t_idx, pop_mask] = np.random.poisson(rates)

            # Add some noise
            noise = np.random.poisson(0.01, size=spikes.shape)
            spikes += noise

            data.append({
                'spikes': spikes,
                'temporal_mod': temporal_modulation,
            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        spikes = torch.tensor(item['spikes'], dtype=torch.float32)

        # Square root stabilization (common for spike data)
        spikes = torch.sqrt(spikes + 1e-6)

        return {
            'spikes': spikes,  # (seq_len, n_units)
        }


def collate_fn(batch):
    """Collate function for batching."""
    spikes = torch.stack([item['spikes'] for item in batch])
    return {
        'spikes': spikes,  # (batch, seq_len, n_units)
        'mask': torch.ones_like(spikes),  # All ones for synthetic data
    }


# ==================== MODEL ====================

class NeuroFMX(nn.Module):
    """
    NeuroFM-X: Foundation Model for Neural Population Dynamics

    Simplified architecture for RTX 3070 Ti:
    - Transformer backbone (instead of Mamba for easier setup)
    - Cross-attention pooling to latent representation
    - Self-supervised reconstruction objective
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Input projection (per-neuron embedding)
        self.input_proj = nn.Linear(1, config.d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1000, config.d_model) * 0.02  # Max 1000 timesteps
        )

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=8,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_transformer_blocks,
        )

        # Learnable latent queries for cross-attention pooling
        self.latent_queries = nn.Parameter(
            torch.randn(config.n_latents, config.latent_dim) * 0.02
        )

        # Cross-attention for pooling to latents
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=8,
            batch_first=True,
        )

        self.value_proj = nn.Linear(config.d_model, config.latent_dim)
        self.query_norm = nn.LayerNorm(config.latent_dim)
        self.key_value_norm = nn.LayerNorm(config.latent_dim)

        # Reconstruction head
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, spikes, mask=None):
        """
        Args:
            spikes: (batch, seq_len, n_units)
            mask: (batch, seq_len, n_units) - optional

        Returns:
            latents: (batch, n_latents, latent_dim)
            reconstructed: (batch, seq_len, n_units)
        """
        batch_size, seq_len, n_units = spikes.shape

        # Process each neuron's time series through shared network
        # Reshape: (batch, seq_len, n_units) -> (batch * n_units, seq_len, 1)
        spikes_flat = spikes.permute(0, 2, 1).reshape(batch_size * n_units, seq_len, 1)

        # Project to d_model
        x = self.input_proj(spikes_flat)  # (batch * n_units, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Backbone processing
        x = self.backbone(x)  # (batch * n_units, seq_len, d_model)

        # Pool across time dimension
        x_pooled = x.mean(dim=1)  # (batch * n_units, d_model)

        # Reshape back to (batch, n_units, d_model)
        x_pooled = x_pooled.reshape(batch_size, n_units, -1)

        # Project to latent dimension for cross-attention
        kv = self.value_proj(x_pooled)  # (batch, n_units, latent_dim)
        kv = self.key_value_norm(kv)

        # Cross-attention pooling to fixed-size latent representation
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_norm(queries)  # (batch, n_latents, latent_dim)

        latents, _ = self.cross_attention(
            query=queries,
            key=kv,
            value=kv,
        )  # (batch, n_latents, latent_dim)

        # Decode from latent representation
        # Use mean pooling of latents for reconstruction
        latent_summary = latents.mean(dim=1, keepdim=True)  # (batch, 1, latent_dim)
        latent_summary = latent_summary.expand(-1, n_units, -1)  # (batch, n_units, latent_dim)

        # Decode to spike predictions
        reconstructed = self.decoder(latent_summary).squeeze(-1)  # (batch, n_units)
        reconstructed = reconstructed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, n_units)

        return latents, reconstructed


# ==================== TRAINER ====================

class Trainer:
    """Trainer for NeuroFM-X."""

    def __init__(self, model, config: Config, train_loader, val_loader):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                return (epoch + 1) / config.warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - config.warmup_epochs) / (config.max_epochs - config.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Create directories
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'='*80}")
        print("NeuroFM-X Trainer Initialized")
        print(f"{'='*80}")
        print(f"Model: NeuroFM-X")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")
        print(f"\nTraining Configuration:")
        print(f"  Device: {device}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Max epochs: {config.max_epochs}")
        print(f"  Mixed precision (AMP): {config.use_amp}")
        print(f"\nData:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"{'='*80}\n")

    def compute_loss(self, latents, reconstructed, spikes, mask):
        """Compute training loss."""
        # Reconstruction loss (MSE with mask)
        if mask is not None:
            diff = (reconstructed - spikes) ** 2
            mse_loss = (diff * mask).sum() / (mask.sum() + 1e-8)
        else:
            mse_loss = F.mse_loss(reconstructed, spikes)

        # Optional: add regularization on latents (encourage diversity)
        latent_reg = 0.0
        if latents is not None:
            # Encourage latent diversity (reduce redundancy)
            latent_cov = torch.einsum('bld,bmd->dlm', latents, latents) / latents.shape[0]
            identity = torch.eye(latents.shape[1], device=latents.device)
            latent_reg = 0.001 * ((latent_cov - identity.unsqueeze(0)) ** 2).mean()

        total_loss = mse_loss + latent_reg

        return total_loss, {
            'mse': mse_loss.item(),
            'latent_reg': latent_reg if isinstance(latent_reg, float) else latent_reg.item(),
        }

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'mse': [], 'latent_reg': []}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.max_epochs} [TRAIN]")

        for batch_idx, batch in enumerate(pbar):
            spikes = batch['spikes'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                latents, reconstructed = self.model(spikes, mask)
                loss, metrics = self.compute_loss(latents, reconstructed, spikes, mask)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                self.global_step += 1

            # Track metrics
            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
            for k, v in metrics.items():
                epoch_metrics[k].append(v)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{np.mean(epoch_losses[-100:]):.4f}",
                'mse': f"{np.mean(epoch_metrics['mse'][-100:]):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return np.mean(epoch_losses), {k: np.mean(v) for k, v in epoch_metrics.items()}

    def validate(self):
        """Validate model."""
        self.model.eval()
        val_losses = []
        val_metrics = {'mse': [], 'latent_reg': []}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1}/{self.config.max_epochs} [VAL]"):
                spikes = batch['spikes'].to(device)
                mask = batch['mask'].to(device)

                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    latents, reconstructed = self.model(spikes, mask)
                    loss, metrics = self.compute_loss(latents, reconstructed, spikes, mask)

                val_losses.append(loss.item())
                for k, v in metrics.items():
                    val_metrics[k].append(v)

        return np.mean(val_losses), {k: np.mean(v) for k, v in val_metrics.items()}

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'd_model': self.config.d_model,
                'n_transformer_blocks': self.config.n_transformer_blocks,
                'n_latents': self.config.n_latents,
                'latent_dim': self.config.latent_dim,
                'dropout': self.config.dropout,
            }
        }

        # Save latest
        latest_path = self.config.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.config.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"\n  ✓ Saved best checkpoint (val_loss: {self.best_val_loss:.6f})")

    def train(self):
        """Full training loop."""
        print(f"\n{'='*80}")
        print("Starting Training")
        print(f"{'='*80}\n")

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch

            # Train
            train_loss, train_metrics = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Log results
            print(f"\n{'─'*80}")
            print(f"Epoch {epoch + 1}/{self.config.max_epochs} Summary:")
            print(f"{'─'*80}")
            print(f"  Train Loss: {train_loss:.6f}  |  Train MSE: {train_metrics['mse']:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}  |  Val MSE:   {val_metrics['mse']:.6f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ★ New best validation loss!")

            self.save_checkpoint(is_best)

        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"{'='*80}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Checkpoints saved to: {self.config.checkpoint_dir.absolute()}")
        print(f"{'='*80}\n")


# ==================== MAIN ====================

def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("NeuroFM-X Training on RTX 3070 Ti - Synthetic Data")
    print("="*80)
    print("\nThis script trains NeuroFM-X on synthetic neural data that mimics")
    print("real Neuropixels recordings from visual cortex.\n")

    # Initialize config
    config = Config()

    # Create datasets
    print("Step 1: Creating synthetic datasets...")
    train_dataset = SyntheticNeuropixelsDataset(
        n_samples=config.n_train_samples,
        n_units=config.n_units,
        sequence_length=config.sequence_length,
        seed=42,
    )

    val_dataset = SyntheticNeuropixelsDataset(
        n_samples=config.n_val_samples,
        n_units=config.n_units,
        sequence_length=config.sequence_length,
        seed=43,
    )

    print(f"\n  ✓ Training samples: {len(train_dataset)}")
    print(f"  ✓ Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn,
    )

    # Create model
    print("\nStep 2: Creating NeuroFM-X model...")
    model = NeuroFMX(config)
    print(f"  ✓ Model created successfully")

    # Create trainer and train
    print("\nStep 3: Initializing trainer...")
    trainer = Trainer(model, config, train_loader, val_loader)

    print("\nStep 4: Training...")
    trainer.train()

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"\n📁 Checkpoints saved to: {config.checkpoint_dir.absolute()}")
    print(f"   - Best model: {(config.checkpoint_dir / 'best.pt').absolute()}")
    print(f"   - Latest model: {(config.checkpoint_dir / 'latest.pt').absolute()}")
    print("\n💡 To load the trained model:")
    print("   checkpoint = torch.load('checkpoints/best.pt')")
    print("   model = NeuroFMX(config)")
    print("   model.load_state_dict(checkpoint['model_state_dict'])")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
