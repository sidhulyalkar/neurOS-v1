"""
Allen Brain Observatory Multimodal Dataset

Loads calcium imaging + astrocyte data for ablation experiments.

Supports:
- Calcium imaging (2-photon) from Allen NPZ files
- Astrocyte tokens from neuros-astro processing
- Temporal alignment between modalities
- Train/val/test splits
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json


class AllenMultiModalDataset(Dataset):
    """
    Dataset for Allen calcium + astrocyte multimodal experiments.

    Args:
        calcium_dir: Directory with Allen 2P session NPZ files
        astro_dir: Directory with neuros-astro processed outputs
        session_ids: List of session IDs to include (or 'all')
        seq_len: Sequence length for temporal windows
        modalities: Which modalities to include ['calcium', 'astro', 'both']
        transform: Optional data augmentation
        temporal_alignment: How to align calcium (30Hz) to astro (10Hz)
    """

    def __init__(
        self,
        calcium_dir: Union[str, Path],
        astro_dir: Union[str, Path],
        session_ids: Union[List[str], str] = 'all',
        seq_len: int = 100,
        modalities: str = 'both',  # 'calcium', 'astro', or 'both'
        transform=None,
        temporal_alignment: str = 'downsample',  # 'downsample' or 'interpolate'
        stride: int = 50,  # Sliding window stride
        min_astro_events: int = 5,  # Min events per window
    ):
        super().__init__()

        self.calcium_dir = Path(calcium_dir)
        self.astro_dir = Path(astro_dir)
        self.seq_len = seq_len
        self.modalities = modalities
        self.transform = transform
        self.temporal_alignment = temporal_alignment
        self.stride = stride
        self.min_astro_events = min_astro_events

        # Find sessions
        if session_ids == 'all':
            self.session_ids = self._find_all_sessions()
        else:
            self.session_ids = session_ids

        print(f"Found {len(self.session_ids)} sessions")

        # Load all data
        self.windows = []
        self._load_all_sessions()

        print(f"Created {len(self.windows)} temporal windows")

    def _find_all_sessions(self) -> List[str]:
        """Find all sessions that have both calcium and astro data."""
        calcium_sessions = set()
        astro_sessions = set()

        # Find calcium sessions
        for f in self.calcium_dir.glob('*.npz'):
            session_id = f.stem  # Remove .npz
            calcium_sessions.add(session_id)

        # Find astro sessions
        for d in self.astro_dir.iterdir():
            if d.is_dir() and (d / 'astro_tokens.npz').exists():
                session_id = d.name
                astro_sessions.add(session_id)

        # Intersection
        common_sessions = calcium_sessions & astro_sessions
        return sorted(list(common_sessions))

    def _load_all_sessions(self):
        """Load all sessions and create temporal windows."""

        for session_id in self.session_ids:
            print(f"  Loading {session_id}...")

            # Load calcium data
            calcium_file = self.calcium_dir / f"{session_id}.npz"
            if not calcium_file.exists():
                print(f"    Warning: Missing calcium file, skipping")
                continue

            calcium_data = np.load(calcium_file)

            # Get traces (neurons x timepoints)
            if 'dff_traces' in calcium_data:
                traces = calcium_data['dff_traces']
            elif 'traces' in calcium_data:
                traces = calcium_data['traces']
            else:
                print(f"    Warning: No traces found, skipping")
                continue

            n_neurons, n_timepoints = traces.shape
            sampling_rate_calcium = calcium_data.get('sampling_rate', 30.0)

            # Load astro tokens
            astro_file = self.astro_dir / session_id / 'astro_tokens.npz'
            if not astro_file.exists():
                print(f"    Warning: Missing astro file, skipping")
                continue

            astro_data = np.load(astro_file, allow_pickle=True)
            event_tokens = astro_data['event_tokens']
            timestamps = astro_data['timestamps']
            region_ids = astro_data.get('region_ids', astro_data.get('astrocyte_ids'))

            n_astrocytes = int(region_ids.max()) + 1 if len(region_ids) > 0 else 0

            # Create sliding windows
            total_duration = n_timepoints / sampling_rate_calcium
            window_duration = self.seq_len / 10.0  # At 10 Hz target

            n_windows = int((total_duration - window_duration) / (self.stride / 10.0)) + 1

            for i in range(n_windows):
                t_start = i * (self.stride / 10.0)
                t_end = t_start + window_duration

                # Extract calcium window
                idx_start = int(t_start * sampling_rate_calcium)
                idx_end = int(t_end * sampling_rate_calcium)

                if idx_end > n_timepoints:
                    break

                calcium_window = traces[:, idx_start:idx_end]  # (n_neurons, window_len)

                # Downsample calcium to 10 Hz
                if self.temporal_alignment == 'downsample':
                    # Simple decimation
                    downsample_factor = int(sampling_rate_calcium / 10.0)
                    calcium_window = calcium_window[:, ::downsample_factor]

                    # Ensure exactly seq_len
                    if calcium_window.shape[1] < self.seq_len:
                        pad_len = self.seq_len - calcium_window.shape[1]
                        calcium_window = np.pad(
                            calcium_window,
                            ((0, 0), (0, pad_len)),
                            mode='edge'
                        )
                    calcium_window = calcium_window[:, :self.seq_len]

                # Extract astro events in this window
                event_mask = (timestamps >= t_start) & (timestamps < t_end)
                window_events = event_tokens[event_mask]
                window_timestamps = timestamps[event_mask]
                window_region_ids = region_ids[event_mask]

                # Skip if too few events
                if len(window_events) < self.min_astro_events:
                    continue

                # Store window
                self.windows.append({
                    'session_id': session_id,
                    'window_idx': i,
                    't_start': t_start,
                    't_end': t_end,
                    'calcium': calcium_window.astype(np.float32),  # (n_neurons, seq_len)
                    'astro_events': window_events.astype(np.float32),
                    'astro_timestamps': window_timestamps.astype(np.float32),
                    'astro_region_ids': window_region_ids.astype(np.int64),
                    'n_neurons': n_neurons,
                    'n_astrocytes': n_astrocytes,
                })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single temporal window.

        Returns:
            Dict with:
                - calcium: (n_neurons, seq_len) if included
                - astro_events: (n_events, n_features) if included
                - astro_timestamps: (n_events,) if included
                - astro_region_ids: (n_events,) if included
                - metadata: session_id, window_idx, etc.
        """
        window = self.windows[idx]

        output = {
            'metadata': {
                'session_id': window['session_id'],
                'window_idx': window['window_idx'],
                't_start': window['t_start'],
                't_end': window['t_end'],
                'n_neurons': window['n_neurons'],
                'n_astrocytes': window['n_astrocytes'],
            }
        }

        # Add modalities based on config
        if self.modalities in ['calcium', 'both']:
            calcium = torch.from_numpy(window['calcium'])  # (n_neurons, seq_len)

            # Apply augmentation if specified
            if self.transform is not None:
                calcium = self.transform(calcium)

            output['calcium'] = calcium

        if self.modalities in ['astro', 'both']:
            output['astro_events'] = torch.from_numpy(window['astro_events'])
            output['astro_timestamps'] = torch.from_numpy(window['astro_timestamps'])
            output['astro_region_ids'] = torch.from_numpy(window['astro_region_ids'])
            output['n_astrocytes'] = window['n_astrocytes']

        return output


def collate_multimodal(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for multimodal data.

    Handles variable-length astro events by padding.
    """
    # Find max dimensions
    max_neurons = max(b['metadata']['n_neurons'] for b in batch)
    max_astrocytes = max(b['metadata']['n_astrocytes'] for b in batch)
    max_events = max(len(b['astro_events']) for b in batch if 'astro_events' in b)

    batch_size = len(batch)
    seq_len = batch[0]['calcium'].shape[1] if 'calcium' in batch[0] else 100

    output = {
        'metadata': [b['metadata'] for b in batch]
    }

    # Collate calcium (pad neurons)
    if 'calcium' in batch[0]:
        calcium_batch = torch.zeros(batch_size, max_neurons, seq_len)
        calcium_mask = torch.zeros(batch_size, max_neurons, dtype=torch.bool)

        for i, b in enumerate(batch):
            n_neurons = b['calcium'].shape[0]
            calcium_batch[i, :n_neurons] = b['calcium']
            calcium_mask[i, :n_neurons] = True

        output['calcium'] = calcium_batch
        output['calcium_mask'] = calcium_mask

    # Collate astro events (pad events)
    if 'astro_events' in batch[0]:
        n_features = batch[0]['astro_events'].shape[1]

        astro_events_batch = torch.zeros(batch_size, max_events, n_features)
        astro_timestamps_batch = torch.zeros(batch_size, max_events)
        astro_region_ids_batch = torch.zeros(batch_size, max_events, dtype=torch.long)
        astro_mask = torch.zeros(batch_size, max_events, dtype=torch.bool)

        for i, b in enumerate(batch):
            n_events = len(b['astro_events'])
            astro_events_batch[i, :n_events] = b['astro_events']
            astro_timestamps_batch[i, :n_events] = b['astro_timestamps']
            astro_region_ids_batch[i, :n_events] = b['astro_region_ids']
            astro_mask[i, :n_events] = True

        output['astro_events'] = astro_events_batch
        output['astro_timestamps'] = astro_timestamps_batch
        output['astro_region_ids'] = astro_region_ids_batch
        output['astro_mask'] = astro_mask
        output['n_astrocytes'] = max_astrocytes

    return output


# Example usage
if __name__ == '__main__':
    # Test dataset loading
    dataset = AllenMultiModalDataset(
        calcium_dir='/path/to/allen/calcium',
        astro_dir='/path/to/neuros-astro/allen_processed',
        session_ids='all',
        seq_len=100,
        modalities='both',
        stride=50,
        min_astro_events=5
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test sample
    sample = dataset[0]
    print("\nFirst sample:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {value}")

    # Test dataloader
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_multimodal,
        num_workers=0  # Set to 0 for debugging
    )

    batch = next(iter(loader))
    print("\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
