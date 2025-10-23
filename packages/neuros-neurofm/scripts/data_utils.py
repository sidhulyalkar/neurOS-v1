"""
Data utilities for NeuroFM-X training.

Handles:
- Dataset loading from multiple sources
- Multi-modal data processing
- Efficient streaming
- Data caching
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
from tqdm import tqdm


class StreamingNeuropixelsDataset(Dataset):
    """Streaming dataset for Neuropixels data."""

    def __init__(self, processed_dir: Path, session_ids: list, max_units: int = 384):
        self.processed_dir = processed_dir
        self.session_ids = session_ids
        self.max_units = max_units
        self.sequence_info = []
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
        spikes = torch.tensor(spikes, dtype=torch.float32)
        spikes = torch.sqrt(spikes + 1e-6)

        # Placeholder behavior target
        behavior_target = torch.randn(spikes.shape[0], 3, dtype=torch.float32)

        return {
            'spikes': spikes,
            'behavior': behavior_target,
            'session_id': torch.tensor(self.session_id_map.get(session_id, 0), dtype=torch.long)
        }


def collate_fn(batch, max_units=384):
    """Collate function to handle variable number of units."""
    actual_max_units = max([item['spikes'].shape[1] for item in batch])
    max_units_padded = min(max_units, actual_max_units)

    batch_size = len(batch)
    seq_len = batch[0]['spikes'].shape[0]
    decoder_dim = batch[0]['behavior'].shape[1]

    padded_spikes = torch.zeros(batch_size, seq_len, max_units)
    unit_mask = torch.zeros(batch_size, max_units)
    behavior_target = torch.zeros(batch_size, seq_len, decoder_dim)
    unit_indices = torch.zeros(batch_size, max_units, dtype=torch.long)
    session_ids = torch.stack([item['session_id'] for item in batch])

    for i, item in enumerate(batch):
        n_units = min(item['spikes'].shape[1], max_units)

        padded_spikes[i, :, :n_units] = item['spikes'][:, :n_units]
        unit_mask[i, n_units:] = 1.0
        behavior_target[i, :, :] = item['behavior']
        unit_indices[i, :n_units] = torch.arange(n_units)

    unit_mask_bool = unit_mask.bool()

    return {
        'tokens_raw': padded_spikes,
        'unit_mask': unit_mask_bool,
        'behavior_target': behavior_target,
        'unit_indices': unit_indices,
        'session_ids': session_ids
    }


def load_allen_dataset(config: Dict):
    """Load Allen dataset from cache."""
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    data_config = config['data']
    cache_dir = Path(data_config.get('cache_dir', './data/allen_neuropixels/cache'))

    cache = EcephysProjectCache.from_warehouse(manifest=str(cache_dir / "manifest.json"))

    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    session_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and "session_" in str(d)]

    session_ids = []
    for session_dir in session_dirs:
        try:
            session_id = int(session_dir.name.split('_')[1])
            session_ids.append(session_id)
        except (IndexError, ValueError):
            continue

    # Limit sessions if specified
    num_sessions = data_config.get('num_sessions')
    if num_sessions is not None:
        session_ids = session_ids[:num_sessions]

    print(f"✓ Found {len(session_ids)} sessions to use")

    return cache, session_ids


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders from config."""
    import random

    data_config = config['data']

    # Load dataset
    cache, session_ids = load_allen_dataset(config)

    # Split sessions
    random.shuffle(session_ids)
    n_train = int(len(session_ids) * data_config['train_split'])
    train_session_ids = session_ids[:n_train]
    val_session_ids = session_ids[n_train:]

    # Create datasets
    processed_dir = Path(data_config['processed_dir'])

    train_dataset = StreamingNeuropixelsDataset(
        processed_dir=processed_dir,
        session_ids=train_session_ids,
        max_units=data_config['max_units']
    )

    val_dataset = StreamingNeuropixelsDataset(
        processed_dir=processed_dir,
        session_ids=val_session_ids,
        max_units=data_config['max_units']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=lambda b: collate_fn(b, max_units=data_config['max_units'])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=lambda b: collate_fn(b, max_units=data_config['max_units'])
    )

    return train_loader, val_loader
