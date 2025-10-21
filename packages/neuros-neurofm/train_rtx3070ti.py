"""
Training Script for NeuroFM-X on RTX 3070 Ti GPU
================================================

This script:
1. Downloads Allen Brain Observatory Visual Coding Neuropixels dataset
2. Prepares the data for training
3. Trains NeuroFM-X foundation model optimized for RTX 3070 Ti (8GB VRAM)
4. Saves checkpoints and monitors training progress

Hardware: RTX 3070 Ti (8GB VRAM)
Optimizations:
- Reduced model size for 8GB VRAM
- Mixed precision training (FP16)
- Gradient accumulation
- Efficient batch sizes
"""

from logging import config
import os
import sys
from pathlib import Path
from typing import Dict, Optional
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
    print("WARNING: CUDA is not available. This script requires a GPU.")
    print("Please ensure you have CUDA installed and your GPU drivers are up to date.")
    sys.exit(1)

device = torch.device('cuda')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


# ==================== CONFIGURATION ====================

class Config:
    """Training configuration optimized for RTX 3070 Ti."""

    # Data paths
    data_dir = Path("./data/allen_neuropixels")
    checkpoint_dir = Path("./checkpoints")
    log_dir = Path("./logs")

    # Model architecture (optimized for 8GB VRAM)
    d_model = 256  # Reduced from 768 for memory efficiency
    n_mamba_blocks = 8  # Reduced from 16
    n_latents = 64  # Reduced from 128
    latent_dim = 256  # Reduced from 512
    n_perceiver_layers = 2  # Reduced from 3
    dropout = 0.1

    # Training
    batch_size = 4  # Small batch size for 8GB VRAM
    gradient_accumulation_steps = 8  # Effective batch size: 4 * 8 = 32
    learning_rate = 3e-4
    weight_decay = 0.01
    max_epochs = 50
    warmup_epochs = 5

    # Mixed precision
    use_amp = True  # Automatic Mixed Precision for RTX 3070 Ti

    # Data
    sequence_length = 100  # 1 second at 100 Hz
    bin_size_ms = 10.0  # 10ms bins
    train_split = 0.8

    # Logging
    log_interval = 100  # Log every N steps
    save_interval = 1  # Save checkpoint every N epochs

    # Allen dataset
    allen_cache_dir = data_dir / "cache"
    num_sessions_to_download = 5  # Start with 5 sessions, increase for more data

# =================== DATA LOADING ====================
def load_allen_dataset(config: Config):    
    """Load cached Allen Brain Observatory Visual Coding Neuropixels dataset."""

    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError:
        print("\nERROR: allensdk not installed. Installing now...")
        os.system("pip install allensdk")
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    # Initialize cache
    print(f"\nInitializing Allen SDK cache at: {config.allen_cache_dir}")
    cache = EcephysProjectCache.from_warehouse(manifest=str(config.allen_cache_dir / "manifest.json"))

     
    # Check if the cache directory exists
    if not config.allen_cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {config.allen_cache_dir}")

    # Get a list of all session directories in the cache
    session_dirs = [d for d in config.allen_cache_dir.iterdir() if d.is_dir() and str(d).find("session_") != -1]
    if not session_dirs:
        raise FileNotFoundError(f"No session directories found in: {config.allen_cache_dir}")

    # Extract session IDs from directory names
    session_ids = []
    for session_dir in session_dirs:
        try:
            session_id = int(session_dir.name.split('_')[1])
            session_ids.append(session_id)
        except (IndexError, ValueError):
            print(f"Warning: Could not extract session ID from directory name: {session_dir.name}")
            continue

    if not session_ids:
        raise ValueError("No valid session IDs found in the cache directory.")

    print(f"✓ Found {len(session_ids)} downloaded sessions in cache")

    return cache, session_ids

# ==================== DATA DOWNLOAD ====================

def download_allen_dataset(config: Config):
    """Download Allen Brain Observatory Visual Coding Neuropixels dataset."""
    print("\n" + "="*80)
    print("Downloading Allen Brain Observatory Dataset")
    print("="*80)

    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError:
        print("\nERROR: allensdk not installed. Installing now...")
        os.system("pip install allensdk")
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    # Create cache directory
    config.allen_cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cache
    print(f"\nInitializing Allen SDK cache at: {config.allen_cache_dir}")
    cache = EcephysProjectCache.from_warehouse(manifest=str(config.allen_cache_dir / "manifest.json"))

    # Get available sessions
    print("\nFetching session metadata...")
    sessions = cache.get_session_table()

    print(f"\nTotal sessions available: {len(sessions)}")
    print(f"Will download first {config.num_sessions_to_download} sessions")

    # Filter for good quality sessions
    # Focus on sessions with natural images stimulus
    filtered_sessions = sessions[
        sessions['session_type'].str.contains('functional_connectivity', na=False)
    ].head(config.num_sessions_to_download)

    if len(filtered_sessions) == 0:
        print("No sessions found with desired criteria. Using first N sessions...")
        filtered_sessions = sessions.head(config.num_sessions_to_download)

    session_ids = filtered_sessions.index.tolist()

    print(f"\nSelected sessions: {session_ids}")

    # Download sessions
    downloaded_sessions = []
    for i, session_id in enumerate(session_ids):
        print(f"\n[{i+1}/{len(session_ids)}] Downloading session {session_id}...")
        try:
            session = cache.get_session_data(session_id)
            downloaded_sessions.append(session)
            print(f"  ✓ Session {session_id} downloaded successfully")
            print(f"    - Units: {len(session.units)}")
            print(f"    - Duration: {session.stimulus_presentations.stop_time.max():.1f} seconds")
        except Exception as e:
            print(f"  ✗ Failed to download session {session_id}: {e}")
            continue

    print(f"\n✓ Successfully downloaded {len(downloaded_sessions)} sessions")
    print(f"✓ Data cached at: {config.allen_cache_dir}")

    return cache, session_ids


# ==================== DATASET ====================

class AllenNeuropixelsDataset(Dataset):
    """Dataset for Allen Neuropixels data."""

    def __init__(
        self,
        cache,
        session_ids,
        sequence_length: int = 100,
        bin_size_ms: float = 10.0,
        max_units: int = 384,  # Limit for memory efficiency
    ):
        self.cache = cache
        self.session_ids = session_ids
        self.sequence_length = sequence_length
        self.bin_size_sec = bin_size_ms / 1000.0
        self.max_units = max_units

        print("\nProcessing sessions into training sequences...")
        self.sequences = []

        for session_id in tqdm(session_ids, desc="Loading sessions"):
            try:
                session = cache.get_session_data(session_id)
                sequences = self._process_session(session)
                self.sequences.extend(sequences)
            except Exception as e:
                print(f"Error processing session {session_id}: {e}")
                continue

        print(f"✓ Created {len(self.sequences)} training sequences")

    def _process_session(self, session):
        """Process session into training sequences."""
        # Get spike times for all units
        units = session.units
        spike_times = session.spike_times

        # Limit number of units for memory efficiency
        if len(units) > self.max_units:
            unit_ids = units.index[:self.max_units].tolist()
        else:
            unit_ids = units.index.tolist()

        n_units = len(unit_ids)

        # Determine recording duration
        max_time = 0
        for unit_id in unit_ids:
            if unit_id in spike_times:
                times = spike_times[unit_id]
                if len(times) > 0:
                    max_time = max(max_time, times.max())

        if max_time == 0:
            return []

        # Create time bins
        n_bins = int(max_time / self.bin_size_sec)
        time_bins = np.linspace(0, max_time, n_bins + 1)

        # Bin spikes
        binned_spikes = np.zeros((n_bins, n_units), dtype=np.float32)

        for i, unit_id in enumerate(unit_ids):
            if unit_id in spike_times:
                times = spike_times[unit_id]
                if len(times) > 0:
                    bin_indices = np.digitize(times, time_bins) - 1
                    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                    for bin_idx in bin_indices:
                        binned_spikes[bin_idx, i] += 1

        # Create overlapping sequences
        sequences = []
        stride = self.sequence_length // 2  # 50% overlap

        for start_idx in range(0, len(binned_spikes) - self.sequence_length, stride):
            end_idx = start_idx + self.sequence_length
            seq_spikes = binned_spikes[start_idx:end_idx]

            # Skip sequences with very low activity
            if seq_spikes.sum() < 10:
                continue

            sequences.append({
                'spikes': seq_spikes,
                'n_units': n_units,
            })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        spikes = torch.tensor(seq['spikes'], dtype=torch.float32)

        # Normalize
        spikes = torch.sqrt(spikes + 1e-6)  # Square root transform for stabilization

        return {
            'spikes': spikes,  # (seq_len, n_units)
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

class SimpleNeuroFMX(nn.Module):
    """Simplified NeuroFM-X for RTX 3070 Ti.

    This is a lightweight version that captures the essence of NeuroFMX
    without requiring the full Mamba/SSM implementation.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(1, config.d_model)

        # Transformer backbone (simpler than Mamba for now)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=8,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_mamba_blocks,
        )

        # Pooling to latent representation
        self.latent_queries = nn.Parameter(torch.randn(config.n_latents, config.latent_dim))
        self.latent_attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=8,
            batch_first=True,
        )

        self.value_proj = nn.Linear(config.d_model, config.latent_dim)

        # Decoder head for self-supervised learning
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, spikes, mask=None):
        """
        Args:
            spikes: (batch, seq_len, n_units)
            mask: (batch, seq_len, n_units)

        Returns:
            latents: (batch, n_latents, latent_dim)
            reconstructed: (batch, seq_len, n_units)
        """
        batch_size, seq_len, n_units = spikes.shape

        # Reshape: (batch, seq_len, n_units) -> (batch * n_units, seq_len, 1)
        spikes_flat = spikes.permute(0, 2, 1).reshape(-1, seq_len, 1)

        # Project to d_model
        x = self.input_proj(spikes_flat)  # (batch * n_units, seq_len, d_model)

        # Create attention mask for transformer
        attn_mask = None
        if mask is not None:
            # Mask for padded units
            mask_flat = mask.permute(0, 2, 1).reshape(-1, seq_len)  # (batch * n_units, seq_len)
            attn_mask = mask_flat == 0

        # Backbone
        backbone_out = self.backbone(x, src_key_padding_mask=attn_mask)  # (batch * n_units, seq_len, d_model)

        # Pool across time
        if attn_mask is not None:
            # Masked mean pooling
            mask_expanded = (~attn_mask).unsqueeze(-1).float()
            pooled = (backbone_out * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-6)
        else:
            pooled = backbone_out.mean(1)  # (batch * n_units, d_model)

        # Reshape back
        pooled = pooled.reshape(batch_size, n_units, -1)  # (batch, n_units, d_model)

        # Project to latent dimension for attention
        values = self.value_proj(pooled)  # (batch, n_units, latent_dim)

        # Latent attention
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_latents, latent_dim)
        latents, _ = self.latent_attention(queries, values, values)  # (batch, n_latents, latent_dim)

        # Decode for reconstruction loss
        decoder_in = latents.mean(1, keepdim=True).expand(-1, n_units, -1)  # (batch, n_units, latent_dim)
        reconstructed = self.decoder(decoder_in).squeeze(-1)  # (batch, n_units)
        reconstructed = reconstructed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, n_units)

        return latents, reconstructed


# ==================== TRAINING ====================

class Trainer:
    """Trainer for NeuroFM-X on RTX 3070 Ti."""

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
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.max_epochs,
            steps_per_epoch=len(train_loader) // config.gradient_accumulation_steps,
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

        print(f"\n{'='*80}")
        print("Trainer initialized")
        print(f"{'='*80}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"Mixed precision: {config.use_amp}")
        print(f"{'='*80}\n")

    def compute_loss(self, latents, reconstructed, spikes, mask):
        """Compute training loss."""
        # Reconstruction loss (MSE)
        if mask is not None:
            # Apply mask
            mse_loss = F.mse_loss(reconstructed * mask, spikes * mask, reduction='sum')
            mse_loss = mse_loss / (mask.sum() + 1e-6)
        else:
            mse_loss = F.mse_loss(reconstructed, spikes)

        # Optional: Add contrastive loss or other auxiliary losses here
        total_loss = mse_loss

        return total_loss, {'mse': mse_loss.item()}

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.max_epochs}")

        for batch_idx, batch in enumerate(pbar):
            spikes = batch['spikes'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                latents, reconstructed = self.model(spikes, mask)
                loss, loss_dict = self.compute_loss(latents, reconstructed, spikes, mask)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{np.mean(epoch_losses[-100:]):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return np.mean(epoch_losses)

    def validate(self):
        """Validate model."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                spikes = batch['spikes'].to(device)
                mask = batch['mask'].to(device)

                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    latents, reconstructed = self.model(spikes, mask)
                    loss, _ = self.compute_loss(latents, reconstructed, spikes, mask)

                val_losses.append(loss.item())

        return np.mean(val_losses)

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
                'n_mamba_blocks': self.config.n_mamba_blocks,
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
            print(f"✓ Saved best checkpoint (val_loss: {self.best_val_loss:.4f})")

    def train(self):
        """Full training loop."""
        print(f"\n{'='*80}")
        print("Starting Training")
        print(f"{'='*80}\n")

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Log
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self.scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)

        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"{'='*80}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.config.checkpoint_dir}")


# ==================== MAIN ====================

def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("NeuroFM-X Training on RTX 3070 Ti")
    print("="*80)

    # Initialize config
    config = Config()

    # # Download dataset
    # print("\nStep 1: Downloading Allen Brain Observatory dataset...")
    # cache, session_ids = download_allen_dataset(config)

    # Load cached dataset
    print("\nStep 1: Loading cached Allen Brain Observatory dataset...")
    cache, session_ids = load_allen_dataset(config)


    # Create dataset
    print("\nStep 2: Creating datasets...")
    full_dataset = AllenNeuropixelsDataset(
        cache=cache,
        session_ids=session_ids,
        sequence_length=config.sequence_length,
        bin_size_ms=config.bin_size_ms,
    )

    # Split train/val
    n_train = int(len(full_dataset) * config.train_split)
    n_val = len(full_dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Create model
    print("\nStep 3: Creating model...")
    model = SimpleNeuroFMX(config)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    print("\nStep 4: Initializing trainer...")
    trainer = Trainer(model, config, train_loader, val_loader)

    # Train
    print("\nStep 5: Training...")
    trainer.train()

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"\nCheckpoints: {config.checkpoint_dir}")
    print(f"Best model: {config.checkpoint_dir / 'best.pt'}")
    print(f"Latest model: {config.checkpoint_dir / 'latest.pt'}")
    print("\nTo use the trained model:")
    print("  checkpoint = torch.load('checkpoints/best.pt')")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")


if __name__ == '__main__':
    main()
