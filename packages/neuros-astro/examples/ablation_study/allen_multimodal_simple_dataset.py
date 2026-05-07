"""
Simple trial-based Allen multimodal dataset for ablation experiments.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Union


class AllenTrialDataset(Dataset):
    """
    Trial-based dataset for Allen calcium + astrocyte ablation.

    Args:
        calcium_dir: Directory with 2p_session_*.npz files
        astro_dir: Directory with session_*/astro_tokens.npz files
        session_ids: List of session IDs
        modalities: 'neural', 'astro', or 'both'
        split: 'train', 'val', or 'test'
        seed: Random seed
    """

    def __init__(
        self,
        calcium_dir: Union[str, Path],
        astro_dir: Union[str, Path],
        session_ids: List[str],
        modalities: str = 'both',
        split: str = 'train',
        seed: int = 42,
    ):
        self.calcium_dir = Path(calcium_dir)
        self.astro_dir = Path(astro_dir)
        self.session_ids = session_ids
        self.modalities = modalities
        self.split = split
        self.seed = seed

        self.samples = []
        self.max_neurons = 0  # Track global max
        self._load_data()

    def _load_data(self):
        """Load all trials from all sessions."""
        np.random.seed(self.seed)

        for session_id in self.session_ids:
            # Load calcium (trial-averaged)
            calcium_file = self.calcium_dir / f"2p_session_{session_id}.npz"
            if not calcium_file.exists():
                continue

            calcium_data = np.load(calcium_file)
            X = calcium_data['X']  # (trials, neurons)
            y = calcium_data['y_orientation']  # (trials,)
            n_trials, n_neurons = X.shape

            # Track global max neurons
            self.max_neurons = max(self.max_neurons, n_neurons)

            # Load astro (session summary)
            astro_file = self.astro_dir / f"session_{session_id}" / 'astro_tokens.npz'
            if not astro_file.exists():
                continue

            astro_data = np.load(astro_file, allow_pickle=True)
            tokens = astro_data['tokens']

            # Session-level astro summary
            if len(tokens) > 0:
                astro_mean = np.mean(tokens, axis=0)
                astro_std = np.std(tokens, axis=0)
                astro_summary = np.concatenate([astro_mean, astro_std])
            else:
                astro_summary = np.zeros(16, dtype=np.float32)

            # Split trials
            indices = np.arange(n_trials)
            np.random.shuffle(indices)

            n_train = int(0.8 * n_trials)
            n_val = int(0.1 * n_trials)

            if self.split == 'train':
                selected = indices[:n_train]
            elif self.split == 'val':
                selected = indices[n_train:n_train+n_val]
            else:
                selected = indices[n_train+n_val:]

            # Add samples
            # Map orientation degrees to class indices
            orientation_to_class = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4, 225: 5, 270: 6, 315: 7}

            for idx in selected:
                orientation_deg = int(y[idx])
                class_label = orientation_to_class.get(orientation_deg, 0)

                self.samples.append({
                    'neural': X[idx].astype(np.float32),
                    'astro': astro_summary.astype(np.float32),
                    'label': class_label,
                    'n_neurons': n_neurons,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        output = {
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'n_neurons': sample['n_neurons'],
        }

        if self.modalities in ['neural', 'both']:
            # Pad to global max neurons
            neural = torch.from_numpy(sample['neural'])
            if neural.shape[0] < self.max_neurons:
                padding = torch.zeros(self.max_neurons - neural.shape[0])
                neural = torch.cat([neural, padding])
            output['neural'] = neural

        if self.modalities in ['astro', 'both']:
            output['astro'] = torch.from_numpy(sample['astro'])

        return output


def collate_fn(batch):
    """Collate batches (already padded to global max)."""
    output = {
        'labels': torch.stack([b['label'] for b in batch]),
    }

    if 'neural' in batch[0]:
        output['neural'] = torch.stack([b['neural'] for b in batch])

    if 'astro' in batch[0]:
        output['astro'] = torch.stack([b['astro'] for b in batch])

    return output
