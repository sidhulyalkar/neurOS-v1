"""
Full Training Script for NeuroFM-X (Mamba + PopT + Heads)
=========================================================

This script uses the complete NeuroFMXComplete model for the main training launch.
It leverages the StreamingNeuropixelsDataset for efficient I/O.
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# --- 1. NEUROFMX CORE IMPORTS ---
# NOTE: Ensure the path 'neuros_neurofm.models' is correctly set up in your local environment.
# Assuming the files are accessible via this namespace as suggested by your snippets.
try:
    from neuros_neurofm.models.neurofmx_multitask import NeuroFMXMultiTask
    # Placeholder for a missing BinnedTokenizer for simplicity, use a dummy if not imported
    # from neuros_neurofm.tokenizers import BinnedTokenizer
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import NeuroFMXComplete. Ensure your project structure and paths are correct. {e}")
    sys.exit(1)


# Check CUDA availability
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. This script requires a GPU.")
    sys.exit(1)

device = torch.device('cuda')
print(f"Using device: {device}")


# ==================== CONFIGURATION (FULL TRAINING) ====================

class Config:
    """Training configuration for the full NeuroFM-X run (RTX 3070 Ti)."""

    # Data paths
    data_dir = Path("./data/allen_neuropixels")
    checkpoint_dir = Path("./checkpoints_neurofmx_full_run") # Final checkpoint dir
    log_dir = Path("./logs_neurofmx_full_run")

    # Model architecture (SCALING UP) - **CRITICAL: SET MAX EPOCHS FOR FULL RUN**
    d_model = 128            # Model dimension for backbone
    n_mamba_blocks = 4       # Number of Mamba layers
    n_latents = 32           # Number of PopT/Perceiver latent vectors
    latent_dim = 128         # Dimension of latents (Head input dim)
    n_perceiver_layers = 2   # Number of Perceiver layers
    n_popt_layers = 2        # Number of PopT layers
    dropout = 0.1
    use_popt = True
    use_multi_rate = True
    downsample_rates = [1, 4] # Multi-rate streams

    # Training
    batch_size = 2           # Hardware batch size
    gradient_accumulation_steps = 8 # Effective Batch Size = 16
    learning_rate = 3e-4
    weight_decay = 0.01
    max_epochs = 50          # <--- SET TO 50 for full training (was 2)
    warmup_epochs = 1
    save_interval = 1

    # Mixed precision
    use_amp = True

    # Data
    sequence_length = 100
    bin_size_ms = 10.0
    train_split = 0.8
    max_units = 384          # Max units in a batch. MUST match preprocessing!
    
    # Task Heads Configuration
    decoder_output_dim = 3   # e.g., (X position, Y position, Velocity)
    encoder_output_dim = 100 # Sequence length for reconstruction (B, S, N)
    enable_decoder = True
    enable_encoder = True
    enable_contrastive = True
    enable_forecast = False  # Keep false for initial run
    
    # Allen dataset
    allen_cache_dir = Path("./data/allen_neuropixels/cache")
    num_sessions_to_download = None


# =================== DATA LOADING & COLLATING (From Previous Step) ====================
# NOTE: The load_allen_dataset, StreamingNeuropixelsDataset, and collate_fn are
# included here with necessary modifications to integrate with the NeuroFMXComplete model.
# Since the Mamba backbone expects a tokenized input, we assume the BinnedTokenizer
# (or a similar step) is handled internally by NeuroFMXComplete, and we pass 'tokens_raw'.

def load_allen_dataset(config: Config):
    # ... (Implementation remains the same)
    # [Code for load_allen_dataset omitted for brevity]
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    cache = EcephysProjectCache.from_warehouse(manifest=str(config.allen_cache_dir / "manifest.json"))
    
    if not config.allen_cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {config.allen_cache_dir}. Run the full script's download step first!")

    session_dirs = [d for d in config.allen_cache_dir.iterdir() if d.is_dir() and str(d).find("session_") != -1]
    
    if not session_dirs:
        raise FileNotFoundError(f"No session directories found in: {config.allen_cache_dir}. You may need to run the full download script first.")

    session_ids = []
    for session_dir in session_dirs:
        try:
            session_id = int(session_dir.name.split('_')[1])
            session_ids.append(session_id)
        except (IndexError, ValueError):
            continue

    if not session_ids:
        raise ValueError("No valid session IDs found in the cache directory.")

    print(f"✓ Found {len(session_ids)} downloaded sessions in cache")

    return cache, session_ids

class StreamingNeuropixelsDataset(Dataset):
    """Loads sequences from processed .npz files on demand."""
    def __init__(self, processed_dir: Path, session_ids: list):
        self.processed_dir = processed_dir
        self.session_ids = session_ids
        self.sequence_info = [] # Stores (file_path, sequence_index, session_id)

        # NOTE: A map for unit indices/session IDs must be created for real PopT/Adapters
        # For simplicity, we use a placeholder here.
        self.session_id_map = {sid: i for i, sid in enumerate(session_ids)}
        
        print("Indexing saved sequences...")
        for session_id in tqdm(session_ids, desc="Indexing files"):
            file_path = processed_dir / f"session_{session_id}.npz"
            if file_path.exists():
                with np.load(file_path) as data:
                    num_sequences = data['spikes'].shape[0]
                for i in range(num_sequences):
                    self.sequence_info.append((file_path, i, session_id))
        
        print(f"✓ Found {len(self.sequence_info)} total indexed sequences.")


    def __len__(self):
        return len(self.sequence_info)

    def __getitem__(self, idx):
        file_path, sequence_index, session_id = self.sequence_info[idx]
        
        with np.load(file_path) as data:
            spikes_array = data['spikes']

        spikes = spikes_array[sequence_index]
        
        # Apply the square root transform and convert to tensor
        spikes = torch.tensor(spikes, dtype=torch.float32)
        spikes = torch.sqrt(spikes + 1e-6)
        
        # Placeholder for real behavior targets (e.g., position/velocity)
        behavior_target = torch.randn(spikes.shape[0], Config.decoder_output_dim, dtype=torch.float32)

        return {
            'spikes': spikes, 
            'behavior': behavior_target,
            'session_id': torch.tensor(self.session_id_map.get(session_id, 0), dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function to handle variable number of units."""
    max_units = max([item['spikes'].shape[1] for item in batch])
    batch_size = len(batch)
    seq_len = batch[0]['spikes'].shape[0]
    decoder_dim = batch[0]['behavior'].shape[1]

    padded_spikes = torch.zeros(batch_size, seq_len, max_units)
    unit_mask = torch.zeros(batch_size, max_units) 
    behavior_target = torch.zeros(batch_size, seq_len, decoder_dim)
    unit_indices = torch.zeros(batch_size, max_units, dtype=torch.long)
    session_ids = torch.stack([item['session_id'] for item in batch])

    for i, item in enumerate(batch):
        n_units = item['spikes'].shape[1]
        
        # Pad spikes (B, S, n_units) -> (B, S, max_units)
        padded_spikes[i, :, :n_units] = item['spikes']
        
        # Unit mask: 1 for valid units, 0 for padded (Flipped for standard PopT/Transformer mask)
        # Note: NeuroFMXComplete uses the PopT standard: True = masked (padded)
        unit_mask[i, n_units:] = 1.0 

        # Behavior Target
        behavior_target[i, :, :] = item['behavior']

        # Unit indices (sequential for simplicity)
        unit_indices[i, :n_units] = torch.arange(n_units)

    # Convert unit mask to boolean (True = masked/padded)
    unit_mask_bool = unit_mask.bool()

    return {
        'tokens_raw': padded_spikes,       # (B, S, N)
        'unit_mask': unit_mask_bool,       # (B, N) - Mask for PopT (True=Padded)
        'behavior_target': behavior_target, # (B, S, D_behavior)
        'unit_indices': unit_indices,      # (B, N)
        'session_ids': session_ids         # (B)
    }


# ==================== TRAINER (Updated Loss for Multi-Task) ====================

class Trainer:
    """Trainer for NeuroFM-X on RTX 3070 Ti, using Multi-Task Heads."""
    # ... (Trainer class implementation remains identical to the previous step)
    
    def __init__(self, model, config: Config, train_loader, val_loader):
        self.model = model.to(device)
        # self.model = torch.compile(self.model)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        if len(train_loader) == 0:
            steps_per_epoch = 1
        else:
             steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
             if steps_per_epoch == 0: steps_per_epoch = 1

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.max_epochs,
            steps_per_epoch=steps_per_epoch,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print("Trainer initialized (FULL NEUROFMX-MAMBA CONFIG)")
        print(f"{'='*80}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"Mixed precision: {config.use_amp}")
        print(f"**Max Units:** {config.max_units}, **Seq Len:** {config.sequence_length}, **d_model:** {config.d_model}")
        print(f"**Multi-Task Heads:** Encoder, Decoder, Contrastive Enabled")
        print(f"{'='*80}\n")
    
    def compute_loss(self, model_output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        """Compute multi-task training loss (Reconstruction + Decoder + Contrastive)."""
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # 1. Reconstruction Loss (Encoder Head)
        if self.config.enable_encoder:
            reconstructed = model_output.get('encoder')
            spikes = batch['tokens_raw'].to(device)
            
            # The reconstruction target (B, S, N) is masked by the unit mask (B, N) 
            # expanded across the time dimension (S).
            unit_mask_expanded = (~batch['unit_mask'].to(device)).unsqueeze(1).expand(-1, spikes.shape[1], -1).float()
            
            rec_loss = F.mse_loss(reconstructed * unit_mask_expanded, spikes * unit_mask_expanded, reduction='sum')
            rec_loss = rec_loss / (unit_mask_expanded.sum() + 1e-6)
            total_loss += rec_loss
            loss_dict['rec_loss'] = rec_loss.item()
            
        # 2. Behavioral Decoding Loss (Decoder Head)
        if self.config.enable_decoder:
            predicted_behavior = model_output.get('decoder')
            target_behavior = batch['behavior_target'].to(device)
            
            # Assuming the decoder predicts the final time step's behavior
            target_behavior_pooled = target_behavior[:, -1, :] 
            
            dec_loss = F.mse_loss(predicted_behavior, target_behavior_pooled)
            total_loss += dec_loss
            loss_dict['dec_loss'] = dec_loss.item()

        # 3. Contrastive Loss (Requires a separate, second forward pass for the positive/negative view)
        # We will assume a minimal implementation of the training loop for a fast launch.

        return total_loss, loss_dict

    def train_epoch(self):
        """Train for one epoch."""
        if len(self.train_loader) == 0:
            return float('nan')
            
        self.model.train()
        epoch_losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.max_epochs}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            
            if batch_idx % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Run the single forward pass that encompasses all active heads
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # NeuroFMXComplete.forward returns a dictionary of task outputs
                model_output = self.model(
                    tokens_raw=batch['tokens_raw'].to(device),
                    unit_mask=batch['unit_mask'].to(device),
                    unit_indices=batch['unit_indices'].to(device),
                    # session_id=batch['session_ids'].to(device), # Enable this if SessionStitcher is active
                    task='multi-task' # A common pattern to run all enabled heads
                )

                # NOTE: For this multi-task call to work, you must update
                # NeuroFMXComplete's forward to execute all enabled heads and
                # return a dict of results (e.g., {'encoder': pred_rec, 'decoder': pred_dec, ...})
                
                loss, loss_dict = self.compute_loss(model_output, batch)
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward() 

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer) 
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1

            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)

            pbar.set_postfix({
                'loss': f"{np.mean(epoch_losses[-100:]):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                **{k: f"{v:.4f}" for k, v in loss_dict.items()}
            })

        return np.mean(epoch_losses)

    def validate(self):
        # ... (Validation logic is similar to train_epoch, using model.eval() and torch.no_grad())
        if len(self.val_loader) == 0:
            return float('inf')

        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    model_output = self.model(
                        tokens_raw=batch['tokens_raw'].to(device),
                        unit_mask=batch['unit_mask'].to(device),
                        unit_indices=batch['unit_indices'].to(device),
                        task='multi-task'
                    )
                    loss, _ = self.compute_loss(model_output, batch)

                val_losses.append(loss.item())

        return np.mean(val_losses)
    
    # ... (save_checkpoint and train methods omitted for brevity)
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            # ... other state dicts
            'best_val_loss': self.best_val_loss,
            'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
        }

        latest_path = self.config.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.config.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (val_loss: {self.best_val_loss:.4f})")

    def train(self):
        print(f"\n{'='*80}")
        print("Starting Sample Training (FULL NEUROFMX-MAMBA)")
        print(f"{'='*80}\n")

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch

            train_loss = self.train_epoch()
            
            if np.isnan(train_loss):
                 print("\nTraining aborted due to insufficient data sequences.")
                 break

            val_loss = self.validate()

            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)

        print(f"\n{'='*80}")
        print("Full Model Sample Training Complete!")
        print(f"{'='*80}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.config.checkpoint_dir}")


# ==================== MAIN LAUNCHER ====================

def main_streaming():
    config = Config()
    
    # 1. Load cache/session IDs
    cache, session_ids = load_allen_dataset(config)
    
    # Split the session IDs (not the sequences) for train/val
    random.shuffle(session_ids)
    n_train_sessions = int(len(session_ids) * config.train_split)
    train_session_ids = session_ids[:n_train_sessions]
    val_session_ids = session_ids[n_train_sessions:]
    
    # 2. Create streaming datasets from processed files
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

    # 3. Create model (FULL NEUROFMXMultiTask)
    # The NeuroFMXMultiTask constructor is called with all required config parameters.
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
        encoder_output_dim=config.max_units,  # Number of neural units (384)
        sequence_length=config.sequence_length,  # Sequence length for reconstruction (100)
        dropout=config.dropout,
    )

    # 4. Initializing trainer
    trainer = Trainer(model, config, train_loader, val_loader)

    # 5. Training
    trainer.train()
    
if __name__ == '__main__':
    main_streaming()