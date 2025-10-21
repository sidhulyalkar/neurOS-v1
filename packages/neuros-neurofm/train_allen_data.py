"""
NeuroFM-X Training on Allen Brain Observatory Data
===================================================

This script trains NeuroFM-X on real Neuropixels recordings from the
Allen Brain Observatory Visual Coding dataset.

Prerequisites:
1. Run download_allen_data.py first to download the dataset
2. Ensure you have PyTorch installed with CUDA support
3. RTX 3070 Ti with 8GB VRAM (or adjust batch size for your GPU)

Hardware Requirements:
- GPU: RTX 3070 Ti (8GB VRAM) or better
- RAM: 16GB+ recommended
- Storage: ~50GB for dataset + checkpoints
"""

import os
import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==================== CONFIGURATION ====================

class Config:
    """Training configuration for RTX 3070 Ti."""

    # Paths
    data_dir = Path("./data/allen_neuropixels")
    checkpoint_dir = Path("./checkpoints_allen")
    log_dir = Path("./logs_allen")

    # Model architecture (optimized for 8GB VRAM)
    d_model = 256
    n_transformer_blocks = 8
    n_latents = 64
    latent_dim = 256
    dropout = 0.1

    # Training
    batch_size = 4  # Small for 8GB VRAM
    gradient_accumulation_steps = 8  # Effective batch size: 32
    learning_rate = 3e-4
    weight_decay = 0.01
    max_epochs = 50
    warmup_epochs = 5

    # Mixed precision
    use_amp = True

    # Data
    max_units = 384  # Limit for memory efficiency
    sequence_length = 100  # 1 second at 100 Hz
    bin_size_ms = 10.0
    train_split = 0.85
    val_split = 0.15

    # Logging
    log_interval = 50
    save_interval = 1

    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments."""
        config = cls()
        config.data_dir = Path(args.data_dir)
        config.batch_size = args.batch_size
        config.max_epochs = args.max_epochs
        config.learning_rate = args.learning_rate
        return config


# ==================== DATASET ====================

class AllenNeuropixelsDataset(Dataset):
    """
    Dataset for Allen Brain Observatory Neuropixels data.

    Loads cached sessions and processes them into training sequences.
    """

    def __init__(
        self,
        cache,
        session_ids,
        sequence_length=100,
        bin_size_ms=10.0,
        max_units=384,
        overlap=0.5,
    ):
        self.cache = cache
        self.session_ids = session_ids
        self.sequence_length = sequence_length
        self.bin_size_sec = bin_size_ms / 1000.0
        self.max_units = max_units
        self.overlap = overlap

        print("\n" + "="*80)
        print("Loading Allen Neuropixels Dataset")
        print("="*80)
        print(f"Sessions to process: {len(session_ids)}")
        print(f"Sequence length: {sequence_length} bins ({sequence_length * bin_size_ms / 1000:.1f}s)")
        print(f"Max units per session: {max_units}")
        print(f"Overlap: {overlap * 100:.0f}%")

        self.sequences = []
        self.session_info = []

        print("\nProcessing sessions...")
        for session_id in tqdm(session_ids, desc="Loading sessions"):
            try:
                session_seqs = self._process_session(session_id)
                self.sequences.extend(session_seqs)
            except Exception as e:
                print(f"  Warning: Failed to process session {session_id}: {e}")
                continue

        print(f"\n✓ Dataset created with {len(self.sequences)} training sequences")
        print("="*80)

    def _process_session(self, session_id):
        """Process a single session into training sequences."""
        # Load session data
        session = self.cache.get_session_data(session_id)

        # Get spike data
        units = session.units
        spike_times_dict = session.spike_times

        # Limit number of units
        unit_ids = units.index[:self.max_units].tolist()
        n_units = len(unit_ids)

        if n_units == 0:
            return []

        # Determine recording duration
        max_time = 0
        for unit_id in unit_ids:
            if unit_id in spike_times_dict:
                times = spike_times_dict[unit_id]
                if len(times) > 0:
                    max_time = max(max_time, times.max())

        if max_time == 0:
            return []

        # Create time bins
        n_bins = int(max_time / self.bin_size_sec)
        time_bins = np.arange(0, n_bins + 1) * self.bin_size_sec

        # Bin spikes
        binned_spikes = np.zeros((n_bins, n_units), dtype=np.float32)

        for i, unit_id in enumerate(unit_ids):
            if unit_id in spike_times_dict:
                times = spike_times_dict[unit_id]
                if len(times) > 0:
                    # Digitize spike times into bins
                    bin_indices = np.digitize(times, time_bins) - 1
                    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                    # Count spikes in each bin
                    for bin_idx in bin_indices:
                        binned_spikes[bin_idx, i] += 1

        # Create overlapping sequences
        sequences = []
        stride = int(self.sequence_length * (1 - self.overlap))

        for start_idx in range(0, len(binned_spikes) - self.sequence_length, stride):
            end_idx = start_idx + self.sequence_length
            seq_spikes = binned_spikes[start_idx:end_idx]

            # Skip sequences with very low activity
            if seq_spikes.sum() < 10:
                continue

            sequences.append({
                'spikes': seq_spikes,
                'session_id': session_id,
                'n_units': n_units,
                'time_range': (time_bins[start_idx], time_bins[end_idx]),
            })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        spikes = torch.tensor(seq['spikes'], dtype=torch.float32)

        # Square root normalization (variance stabilization for Poisson data)
        spikes = torch.sqrt(spikes + 1e-6)

        return {
            'spikes': spikes,  # (seq_len, n_units)
            'session_id': seq['session_id'],
        }


def collate_fn(batch):
    """Collate function to handle variable number of units."""
    # Find max units in batch
    max_units = max([item['spikes'].shape[1] for item in batch])
    batch_size = len(batch)
    seq_len = batch[0]['spikes'].shape[0]

    # Pad to max units
    padded_spikes = torch.zeros(batch_size, seq_len, max_units)
    masks = torch.zeros(batch_size, seq_len, max_units)

    for i, item in enumerate(batch):
        n_units = item['spikes'].shape[1]
        padded_spikes[i, :, :n_units] = item['spikes']
        masks[i, :, :n_units] = 1.0

    return {
        'spikes': padded_spikes,
        'mask': masks,
    }


# ==================== MODEL ====================

class NeuroFMX(nn.Module):
    """
    NeuroFM-X: Foundation Model for Neural Population Dynamics

    Transformer-based architecture optimized for RTX 3070 Ti.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(1, config.d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1000, config.d_model) * 0.02
        )

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=8,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_transformer_blocks,
        )

        # Latent pooling
        self.latent_queries = nn.Parameter(
            torch.randn(config.n_latents, config.latent_dim) * 0.02
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=8,
            batch_first=True,
        )

        self.value_proj = nn.Linear(config.d_model, config.latent_dim)
        self.query_norm = nn.LayerNorm(config.latent_dim)
        self.key_value_norm = nn.LayerNorm(config.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

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
        batch_size, seq_len, n_units = spikes.shape

        # Process each neuron's time series
        spikes_flat = spikes.permute(0, 2, 1).reshape(batch_size * n_units, seq_len, 1)
        x = self.input_proj(spikes_flat)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Backbone
        x = self.backbone(x)
        x_pooled = x.mean(dim=1)
        x_pooled = x_pooled.reshape(batch_size, n_units, -1)

        # Cross-attention to latents
        kv = self.key_value_norm(self.value_proj(x_pooled))
        queries = self.query_norm(
            self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        )
        latents, _ = self.cross_attention(queries, kv, kv)

        # Decode
        latent_summary = latents.mean(dim=1, keepdim=True).expand(-1, n_units, -1)
        reconstructed = self.decoder(latent_summary).squeeze(-1)
        reconstructed = reconstructed.unsqueeze(1).expand(-1, seq_len, -1)

        return latents, reconstructed


# ==================== TRAINER ====================

class Trainer:
    """Trainer for NeuroFM-X."""

    def __init__(self, model, config, train_loader, val_loader, device):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                return (epoch + 1) / config.warmup_epochs
            progress = (epoch - config.warmup_epochs) / (config.max_epochs - config.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print("NeuroFM-X Trainer")
        print(f"{'='*80}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {device}")
        print(f"Batch size: {config.batch_size}")
        print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"{'='*80}\n")

    def compute_loss(self, latents, reconstructed, spikes, mask):
        if mask is not None:
            diff = (reconstructed - spikes) ** 2
            mse_loss = (diff * mask).sum() / (mask.sum() + 1e-8)
        else:
            mse_loss = F.mse_loss(reconstructed, spikes)

        return mse_loss, {'mse': mse_loss.item()}

    def train_epoch(self):
        self.model.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.max_epochs} [TRAIN]")

        for batch_idx, batch in enumerate(pbar):
            spikes = batch['spikes'].to(self.device)
            mask = batch['mask'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                latents, reconstructed = self.model(spikes, mask)
                loss, _ = self.compute_loss(latents, reconstructed, spikes, mask)
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
            pbar.set_postfix({
                'loss': f"{np.mean(epoch_losses[-100:]):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return np.mean(epoch_losses)

    def validate(self):
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1}/{self.config.max_epochs} [VAL]"):
                spikes = batch['spikes'].to(self.device)
                mask = batch['mask'].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    latents, reconstructed = self.model(spikes, mask)
                    loss, _ = self.compute_loss(latents, reconstructed, spikes, mask)

                val_losses.append(loss.item())

        return np.mean(val_losses)

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
        }

        latest_path = self.config.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.config.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best checkpoint (val_loss: {self.best_val_loss:.6f})")

    def train(self):
        print(f"\n{'='*80}")
        print("Starting Training")
        print(f"{'='*80}\n")

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()

            print(f"\n{'─'*80}")
            print(f"Epoch {epoch + 1}/{self.config.max_epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ★ New best!")

            self.save_checkpoint(is_best)

        print(f"\n{'='*80}")
        print(f"Training Complete! Best val loss: {self.best_val_loss:.6f}")
        print(f"{'='*80}\n")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Train NeuroFM-X on Allen data')
    parser.add_argument('--data-dir', type=str, default='./data/allen_neuropixels',
                        help='Path to downloaded Allen data')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4 for RTX 3070 Ti)')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum epochs')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("NeuroFM-X Training on Allen Brain Observatory Data")
    print("="*80)

    # Check CUDA
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Training will be very slow on CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load config
    config = Config.from_args(args)

    # Check if data exists
    cache_dir = config.data_dir / "cache"
    if not cache_dir.exists():
        print(f"\nERROR: Data not found at {cache_dir}")
        print("Please run download_allen_data.py first to download the dataset.")
        sys.exit(1)

    # Load Allen SDK
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError:
        print("\nERROR: allensdk not installed. Install with: pip install allensdk")
        sys.exit(1)

    # Initialize cache
    print("\nLoading Allen SDK cache...")
    manifest_path = cache_dir / "manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

    # Get available sessions
    sessions = cache.get_session_table()
    session_ids = sessions.index.tolist()

    print(f"Found {len(session_ids)} cached sessions")

    # Create datasets
    print("\nCreating datasets...")
    full_dataset = AllenNeuropixelsDataset(
        cache=cache,
        session_ids=session_ids,
        sequence_length=config.sequence_length,
        bin_size_ms=config.bin_size_ms,
        max_units=config.max_units,
    )

    # Split
    n_train = int(len(full_dataset) * config.train_split)
    n_val = len(full_dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    # Create model
    print("\nCreating model...")
    model = NeuroFMX(config)

    # Train
    trainer = Trainer(model, config, train_loader, val_loader, device)
    trainer.train()

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"\nCheckpoints: {config.checkpoint_dir.absolute()}")
    print(f"Best model: {(config.checkpoint_dir / 'best.pt').absolute()}")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
