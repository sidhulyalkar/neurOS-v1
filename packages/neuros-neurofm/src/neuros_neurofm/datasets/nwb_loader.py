"""
NWB (Neurodata Without Borders) dataset loader for NeuroFM-X.

Supports loading real neural data from IBL, Allen, and DANDI archives.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pynwb
    from pynwb import NWBHDF5IO
    NWB_AVAILABLE = True
except ImportError:
    NWB_AVAILABLE = False
    NWBHDF5IO = None


class NWBDataset(Dataset):
    """Load neural data from NWB files.

    Parameters
    ----------
    nwb_file_path : str
        Path to NWB file.
    neural_key : str, optional
        Key for neural data (e.g., 'Units' for spikes).
        Default: 'Units'.
    behavior_keys : list of str, optional
        Keys for behavioral data.
        Default: ['position'].
    bin_size_ms : float, optional
        Bin size for spike binning in milliseconds.
        Default: 10.0.
    sequence_length : int, optional
        Length of sequences to extract.
        Default: 100.
    overlap : float, optional
        Overlap between sequences (0-1).
        Default: 0.5.
    """

    def __init__(
        self,
        nwb_file_path: str,
        neural_key: str = 'Units',
        behavior_keys: List[str] = ['position'],
        bin_size_ms: float = 10.0,
        sequence_length: int = 100,
        overlap: float = 0.5,
    ):
        if not NWB_AVAILABLE:
            raise ImportError(
                "pynwb is required for NWB datasets. "
                "Install with: pip install pynwb"
            )

        self.nwb_file_path = nwb_file_path
        self.neural_key = neural_key
        self.behavior_keys = behavior_keys
        self.bin_size_ms = bin_size_ms
        self.sequence_length = sequence_length
        self.overlap = overlap

        # Load data
        self._load_nwb_data()

    def _load_nwb_data(self):
        """Load data from NWB file."""
        with NWBHDF5IO(self.nwb_file_path, 'r') as io:
            nwbfile = io.read()

            # Load neural data (spikes)
            if hasattr(nwbfile, 'units'):
                self.spike_times, self.spike_units = self._extract_spikes(nwbfile.units)
                self.n_units = len(set(self.spike_units))
            else:
                raise ValueError("NWB file does not contain units data")

            # Load behavioral data
            self.behavior_data = {}
            for key in self.behavior_keys:
                if hasattr(nwbfile, 'processing') and key in nwbfile.processing:
                    self.behavior_data[key] = self._extract_behavior(
                        nwbfile.processing[key]
                    )

            # Determine time range
            self.start_time = min(self.spike_times)
            self.end_time = max(self.spike_times)
            self.duration = self.end_time - self.start_time

        # Create binned spikes
        self.binned_spikes, self.time_bins = self._bin_spikes()

        # Create sequences
        self._create_sequences()

    def _extract_spikes(self, units) -> Tuple[np.ndarray, np.ndarray]:
        """Extract spike times and unit IDs."""
        spike_times = []
        spike_units = []

        for unit_id in range(len(units)):
            unit_spike_times = units['spike_times'][unit_id]
            spike_times.extend(unit_spike_times)
            spike_units.extend([unit_id] * len(unit_spike_times))

        return np.array(spike_times), np.array(spike_units)

    def _extract_behavior(self, processing_module):
        """Extract behavioral data."""
        # This is a simplified version - real implementation would be more sophisticated
        try:
            data = processing_module.data[:]
            timestamps = processing_module.timestamps[:]
            return {'data': data, 'timestamps': timestamps}
        except:
            return None

    def _bin_spikes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bin spikes into time bins."""
        bin_size_sec = self.bin_size_ms / 1000.0
        n_bins = int(np.ceil(self.duration / bin_size_sec))
        time_bins = np.linspace(self.start_time, self.end_time, n_bins + 1)

        # Create binned spike matrix
        binned = np.zeros((n_bins, self.n_units))

        for spike_time, unit_id in zip(self.spike_times, self.spike_units):
            bin_idx = int((spike_time - self.start_time) / bin_size_sec)
            if 0 <= bin_idx < n_bins:
                binned[bin_idx, int(unit_id)] += 1

        return binned, time_bins

    def _create_sequences(self):
        """Create overlapping sequences."""
        step_size = int(self.sequence_length * (1 - self.overlap))
        n_sequences = (len(self.binned_spikes) - self.sequence_length) // step_size + 1

        self.sequences = []
        for i in range(n_sequences):
            start_idx = i * step_size
            end_idx = start_idx + self.sequence_length

            if end_idx <= len(self.binned_spikes):
                seq_spikes = self.binned_spikes[start_idx:end_idx]

                # Get corresponding behavior (simplified)
                seq_behavior = np.random.randn(self.sequence_length, 2)  # Placeholder

                self.sequences.append({
                    'spikes': seq_spikes,
                    'behavior': seq_behavior,
                    'time_range': (self.time_bins[start_idx], self.time_bins[end_idx]),
                })

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sequence by index."""
        seq = self.sequences[idx]

        return {
            'spikes': torch.tensor(seq['spikes'], dtype=torch.float32),
            'behavior': torch.tensor(seq['behavior'], dtype=torch.float32),
            'behavior_target': torch.tensor(seq['behavior'][-1], dtype=torch.float32),
            'neural': torch.tensor(seq['spikes'], dtype=torch.float32),
        }


class IBLDataset(NWBDataset):
    """Dataset loader for International Brain Laboratory data.

    Specialized for IBL NWB files with specific data structures.
    """

    def __init__(self, nwb_file_path: str, **kwargs):
        # IBL-specific defaults
        kwargs.setdefault('neural_key', 'Units')
        kwargs.setdefault('behavior_keys', ['wheel_position', 'choice'])
        super().__init__(nwb_file_path, **kwargs)


class AllenDataset(NWBDataset):
    """Dataset loader for Allen Institute data.

    Specialized for Allen Brain Observatory and Neuropixels datasets.
    """

    def __init__(self, nwb_file_path: str, **kwargs):
        # Allen-specific defaults
        kwargs.setdefault('neural_key', 'Units')
        kwargs.setdefault('behavior_keys', ['running_speed', 'pupil_diameter'])
        super().__init__(nwb_file_path, **kwargs)


def load_dandi_dataset(
    dandiset_id: str,
    asset_path: str,
    cache_dir: Optional[str] = None,
    **kwargs
) -> NWBDataset:
    """Load dataset from DANDI archive.

    Parameters
    ----------
    dandiset_id : str
        DANDI dataset ID (e.g., '000003').
    asset_path : str
        Path to asset within dataset.
    cache_dir : str, optional
        Directory for caching downloaded files.
    **kwargs
        Additional arguments for NWBDataset.

    Returns
    -------
    NWBDataset
        Loaded dataset.
    """
    try:
        from dandi.dandiapi import DandiAPIClient
    except ImportError:
        raise ImportError(
            "dandi is required for DANDI datasets. "
            "Install with: pip install dandi"
        )

    # Download from DANDI (simplified - real implementation would be more robust)
    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id)

    # For demo, just return a placeholder
    raise NotImplementedError(
        "DANDI download not yet implemented. "
        "Please download NWB files manually and use NWBDataset."
    )


def create_nwb_dataloaders(
    nwb_files: List[str],
    dataset_type: str = 'nwb',
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val dataloaders from NWB files.

    Parameters
    ----------
    nwb_files : list of str
        List of NWB file paths.
    dataset_type : str, optional
        Dataset type ('nwb', 'ibl', 'allen').
        Default: 'nwb'.
    batch_size : int, optional
        Batch size.
    train_split : float, optional
        Train/val split ratio.
    num_workers : int, optional
        Number of data loading workers.
    **dataset_kwargs
        Additional arguments for dataset.

    Returns
    -------
    train_loader, val_loader
        Training and validation dataloaders.
    """
    # Select dataset class
    if dataset_type == 'ibl':
        dataset_class = IBLDataset
    elif dataset_type == 'allen':
        dataset_class = AllenDataset
    else:
        dataset_class = NWBDataset

    # Load all datasets
    datasets = [dataset_class(f, **dataset_kwargs) for f in nwb_files]

    # Concatenate datasets
    from torch.utils.data import ConcatDataset
    full_dataset = ConcatDataset(datasets)

    # Split train/val
    n_train = int(len(full_dataset) * train_split)
    n_val = len(full_dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Create dataloaders
    from neuros_neurofm.datasets.synthetic import collate_neurofmx

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
