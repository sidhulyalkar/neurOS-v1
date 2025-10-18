"""
Synthetic dataset generator for NeuroFM-X testing and demos.

Creates realistic synthetic neural data with behavioral correlates.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticNeuralDataset(Dataset):
    """Synthetic neural population dataset.

    Generates synthetic spike data correlated with 2D position (e.g., motor cortex
    during reaching task). Uses Poisson spiking with tuning curves.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples.
        Default: 1000.
    n_units : int, optional
        Number of neurons.
        Default: 96.
    seq_length : int, optional
        Sequence length (time bins).
        Default: 100.
    bin_size_ms : float, optional
        Bin size in milliseconds.
        Default: 10.0.
    behavior_dim : int, optional
        Behavioral dimension (e.g., 2 for 2D position).
        Default: 2.
    noise_level : float, optional
        Noise level (0-1).
        Default: 0.1.
    seed : int, optional
        Random seed.
        Default: 42.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_units: int = 96,
        seq_length: int = 100,
        bin_size_ms: float = 10.0,
        behavior_dim: int = 2,
        noise_level: float = 0.1,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_units = n_units
        self.seq_length = seq_length
        self.bin_size_ms = bin_size_ms
        self.behavior_dim = behavior_dim
        self.noise_level = noise_level

        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate tuning curves (preferred directions for each neuron)
        self.preferred_directions = np.random.randn(n_units, behavior_dim)
        self.preferred_directions /= np.linalg.norm(
            self.preferred_directions, axis=1, keepdims=True
        )

        # Baseline firing rates (5-20 Hz)
        self.baseline_rates = np.random.uniform(5, 20, size=n_units)

        # Generate dataset
        self.data = self._generate_data()

    def _generate_data(self) -> list:
        """Generate synthetic dataset.

        Returns
        -------
        list
            List of (spikes, behavior) tuples.
        """
        data = []

        for _ in range(self.n_samples):
            # Generate smooth behavioral trajectory (e.g., reaching movement)
            t = np.linspace(0, 2 * np.pi, self.seq_length)
            behavior = np.stack([
                np.sin(t + np.random.randn() * 0.5),
                np.cos(1.5 * t + np.random.randn() * 0.5),
            ], axis=1)  # (seq_length, behavior_dim)

            # Add noise
            behavior += np.random.randn(*behavior.shape) * self.noise_level

            # Generate spike counts from tuning curves
            spikes = np.zeros((self.seq_length, self.n_units))

            for i in range(self.n_units):
                # Cosine tuning: rate = baseline + gain * cos(angle)
                alignment = behavior @ self.preferred_directions[i]
                modulation = 20 * alignment  # Max 20 Hz modulation
                rates = self.baseline_rates[i] + modulation
                rates = np.maximum(rates, 1.0)  # Minimum 1 Hz

                # Convert to spike counts (Poisson)
                expected_counts = rates * (self.bin_size_ms / 1000.0)
                spikes[:, i] = np.random.poisson(expected_counts)

            data.append((spikes, behavior))

        return data

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get item.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict
            Sample with keys:
            - "spikes": Spike counts, shape (seq_length, n_units)
            - "behavior": Behavioral variables, shape (seq_length, behavior_dim)
            - "behavior_target": Single target value (last time point)
        """
        spikes, behavior = self.data[idx]

        return {
            "spikes": torch.tensor(spikes, dtype=torch.float32),
            "behavior": torch.tensor(behavior, dtype=torch.float32),
            "behavior_target": torch.tensor(behavior[-1], dtype=torch.float32),
        }


def collate_neurofmx(batch: list) -> dict:
    """Collate function for NeuroFM-X datasets.

    Parameters
    ----------
    batch : list
        List of samples from dataset.

    Returns
    -------
    dict
        Batched data ready for NeuroFM-X.
    """
    # Stack all samples
    spikes = torch.stack([item["spikes"] for item in batch])
    behavior = torch.stack([item["behavior"] for item in batch])
    behavior_target = torch.stack([item["behavior_target"] for item in batch])

    # Create batch dict
    return {
        "spikes": spikes,  # (batch, seq_length, n_units)
        "behavior": behavior,  # (batch, seq_length, behavior_dim)
        "behavior_target": behavior_target,  # (batch, behavior_dim)
        "neural": spikes,  # For encoder task
    }


class MultiModalSyntheticDataset(Dataset):
    """Multi-modal synthetic dataset (spikes + LFP).

    Parameters
    ----------
    n_samples : int, optional
        Number of samples.
        Default: 1000.
    n_units : int, optional
        Number of neurons.
        Default: 96.
    n_lfp_channels : int, optional
        Number of LFP channels.
        Default: 64.
    seq_length : int, optional
        Sequence length.
        Default: 100.
    lfp_seq_length : int, optional
        LFP sequence length (higher sampling rate).
        Default: 1000.
    behavior_dim : int, optional
        Behavioral dimension.
        Default: 2.
    seed : int, optional
        Random seed.
        Default: 42.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_units: int = 96,
        n_lfp_channels: int = 64,
        seq_length: int = 100,
        lfp_seq_length: int = 1000,
        behavior_dim: int = 2,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_units = n_units
        self.n_lfp_channels = n_lfp_channels
        self.seq_length = seq_length
        self.lfp_seq_length = lfp_seq_length
        self.behavior_dim = behavior_dim

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate spike dataset
        self.spike_dataset = SyntheticNeuralDataset(
            n_samples=n_samples,
            n_units=n_units,
            seq_length=seq_length,
            behavior_dim=behavior_dim,
            seed=seed,
        )

        # Generate LFP data
        self.lfp_data = self._generate_lfp_data()

    def _generate_lfp_data(self) -> list:
        """Generate synthetic LFP data.

        Returns
        -------
        list
            List of LFP arrays.
        """
        lfp_data = []

        for _ in range(self.n_samples):
            # Generate multi-band LFP (alpha, beta, gamma)
            t = np.linspace(0, 1, self.lfp_seq_length)
            lfp = np.zeros((self.n_lfp_channels, self.lfp_seq_length))

            for ch in range(self.n_lfp_channels):
                # Alpha band (8-12 Hz)
                alpha = np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)

                # Beta band (12-30 Hz)
                beta = 0.5 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)

                # Gamma band (30-100 Hz)
                gamma = 0.3 * np.sin(2 * np.pi * 60 * t + np.random.rand() * 2 * np.pi)

                # Combine + noise
                lfp[ch] = alpha + beta + gamma + np.random.randn(self.lfp_seq_length) * 0.1

            lfp_data.append(lfp)

        return lfp_data

    def __len__(self) -> int:
        """Dataset length."""
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Get item.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict
            Multi-modal sample.
        """
        # Get spike data
        spike_item = self.spike_dataset[idx]

        # Get LFP data
        lfp = self.lfp_data[idx]

        return {
            **spike_item,
            "lfp": torch.tensor(lfp, dtype=torch.float32),
        }


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split.
    batch_size : int, optional
        Batch size.
        Default: 32.
    train_split : float, optional
        Fraction for training.
        Default: 0.8.
    num_workers : int, optional
        Number of workers.
        Default: 4.

    Returns
    -------
    train_loader : DataLoader
        Training dataloader.
    val_loader : DataLoader
        Validation dataloader.
    """
    # Split dataset
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_neurofmx,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_neurofmx,
        pin_memory=True,
    )

    return train_loader, val_loader
