"""
BCI dataset loaders for public EEG/BCI datasets.

Provides interfaces to BNCI Horizon 2020, PhysioNet, and other standard
BCI benchmark datasets.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_bnci_horizon(
    dataset_id: str = "001-2014",
    subject_id: int = 1,
    cache_dir: Optional[str] = None,
    download: bool = True,
) -> Dict[str, Any]:
    """Load BNCI Horizon 2020 dataset.

    BNCI Horizon provides standardized BCI datasets for motor imagery,
    P300, SSVEP, and other paradigms.

    Parameters
    ----------
    dataset_id : str, default='001-2014'
        Dataset identifier (e.g., '001-2014' for motor imagery).
    subject_id : int, default=1
        Subject number to load.
    cache_dir : str, optional
        Directory to cache downloaded data.
    download : bool, default=True
        Whether to download if not cached.

    Returns
    -------
    dict
        Dictionary containing:
        - 'X': EEG data, shape (n_trials, n_channels, n_timepoints)
        - 'y': Labels, shape (n_trials,)
        - 'fs': Sampling frequency in Hz
        - 'channels': Channel names
        - 'events': Event markers
        - 'metadata': Dataset metadata

    Examples
    --------
    >>> data = load_bnci_horizon(dataset_id='001-2014', subject_id=1)
    >>> print(f"Shape: {data['X'].shape}, Fs: {data['fs']} Hz")

    References
    ----------
    - BNCI Horizon 2020: http://bnci-horizon-2020.eu/database/data-sets

    Notes
    -----
    This function requires the mne and moabb packages:
        pip install mne moabb
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError:
        raise ImportError(
            "MOABB is required to load BNCI datasets. "
            "Install with: pip install moabb mne"
        )

    # Set up cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".neuros" / "datasets" / "bnci"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading BNCI {dataset_id}, subject {subject_id}")

    # Load dataset
    if dataset_id == "001-2014":
        dataset = BNCI2014_001()
        paradigm = MotorImagery(events=["left_hand", "right_hand", "feet", "tongue"])
    else:
        raise ValueError(f"Unsupported dataset ID: {dataset_id}")

    # Get data for specified subject
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    # Extract metadata
    fs = metadata.iloc[0]["sfreq"]
    channels = X.info["ch_names"]

    logger.info(f"Loaded {len(X)} trials with {len(channels)} channels at {fs} Hz")

    return {
        "X": X,
        "y": y,
        "fs": fs,
        "channels": channels,
        "metadata": metadata,
        "dataset_id": dataset_id,
        "subject_id": subject_id,
    }


def load_physionet_mi(
    subject_id: int = 1,
    runs: Optional[list] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load PhysioNet Motor Imagery dataset.

    Parameters
    ----------
    subject_id : int, default=1
        Subject number (1-109).
    runs : list of int, optional
        Specific runs to load. If None, loads all runs.
    cache_dir : str, optional
        Directory to cache downloaded data.

    Returns
    -------
    dict
        Dictionary with EEG data and metadata.

    Examples
    --------
    >>> data = load_physionet_mi(subject_id=1, runs=[3, 7, 11])
    >>> print(f"Motor imagery runs: {data['X'].shape}")

    References
    ----------
    - PhysioNet EEG Motor Movement/Imagery Dataset:
      https://physionet.org/content/eegmmidb/1.0.0/

    Notes
    -----
    Requires mne package: pip install mne
    """
    try:
        import mne
        from mne.datasets import eegbci
    except ImportError:
        raise ImportError(
            "MNE is required to load PhysioNet datasets. "
            "Install with: pip install mne"
        )

    # Default motor imagery runs (3, 7, 11 are MI runs)
    if runs is None:
        runs = [3, 7, 11]

    # Set up cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".neuros" / "datasets" / "physionet"
    else:
        cache_dir = Path(cache_dir)

    logger.info(f"Loading PhysioNet subject {subject_id}, runs {runs}")

    # Download and load files
    raw_fnames = eegbci.load_data(subject_id, runs, path=str(cache_dir))

    # Read raw data
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
    for fname in raw_fnames[1:]:
        raw_run = mne.io.read_raw_edf(fname, preload=True)
        raw.append(raw_run)

    # Extract events
    events, event_id = mne.events_from_annotations(raw)

    # Create epochs
    tmin, tmax = 0.0, 4.0  # 4-second epochs
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )

    # Get data
    X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]  # event labels
    fs = epochs.info['sfreq']
    channels = epochs.ch_names

    logger.info(f"Loaded {len(X)} epochs with {len(channels)} channels at {fs} Hz")

    return {
        "X": X,
        "y": y,
        "fs": fs,
        "channels": channels,
        "events": events,
        "event_id": event_id,
        "subject_id": subject_id,
        "runs": runs,
    }


def load_mock_bci_data(
    n_trials: int = 100,
    n_channels: int = 22,
    n_timepoints: int = 1000,
    n_classes: int = 2,
    fs: float = 250.0,
) -> Dict[str, Any]:
    """Generate mock BCI data for testing.

    Parameters
    ----------
    n_trials : int, default=100
        Number of trials.
    n_channels : int, default=22
        Number of EEG channels.
    n_timepoints : int, default=1000
        Number of time points per trial.
    n_classes : int, default=2
        Number of classes.
    fs : float, default=250.0
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Mock dataset with same structure as real BCI data.

    Examples
    --------
    >>> data = load_mock_bci_data(n_trials=50, n_classes=4)
    >>> print(f"Mock BCI data: {data['X'].shape}")
    """
    rng = np.random.default_rng(42)

    # Generate synthetic EEG-like data
    # Add some structure: different frequency content per class
    X = np.zeros((n_trials, n_channels, n_timepoints))

    for trial in range(n_trials):
        class_id = trial % n_classes

        for ch in range(n_channels):
            # Base noise
            signal = rng.normal(0, 1, n_timepoints)

            # Add class-specific oscillations
            t = np.arange(n_timepoints) / fs
            if class_id == 0:
                signal += 2 * np.sin(2 * np.pi * 10 * t)  # 10 Hz (alpha)
            elif class_id == 1:
                signal += 2 * np.sin(2 * np.pi * 20 * t)  # 20 Hz (beta)
            elif class_id == 2:
                signal += 2 * np.sin(2 * np.pi * 30 * t)  # 30 Hz (gamma)
            elif class_id == 3:
                signal += 2 * np.sin(2 * np.pi * 4 * t)   # 4 Hz (theta)

            # Add some spatial structure
            if ch < n_channels // 2:
                signal *= 1.2  # Stronger signal in first half of channels

            X[trial, ch, :] = signal

    # Generate labels
    y = np.array([trial % n_classes for trial in range(n_trials)])

    # Generate channel names
    channels = [f"C{i+1}" for i in range(n_channels)]

    logger.info(f"Generated mock BCI data: {n_trials} trials, {n_classes} classes")

    return {
        "X": X,
        "y": y,
        "fs": fs,
        "channels": channels,
        "n_classes": n_classes,
        "is_mock": True,
    }


# ==============================================================================
# Validation Extensions for SAE Feature Analysis
# ==============================================================================

try:
    from .base_dataset import BaseNeuralDataset, NeuralWindow
except ImportError:
    # Fallback if base_dataset not yet created
    BaseNeuralDataset = object
    NeuralWindow = None


class BCIMotorImageryValidator(BaseNeuralDataset):
    """
    BCI motor imagery dataset validator for cross-modal SAE feature validation.

    Tests whether SAE features generalize from spikes (Allen) to EEG (BCI).
    Uses mock data for quick validation without large downloads.

    Parameters
    ----------
    n_trials : int, default=200
        Number of motor imagery trials to generate.
    n_channels : int, default=22
        Number of EEG channels.
    sampling_rate : float, default=250.0
        EEG sampling rate in Hz.
    data_path : str, default='./bci_data'
        Path for data storage.
    cache_dir : str, default='./cache'
        Cache directory.

    Examples
    --------
    >>> validator = BCIMotorImageryValidator(n_trials=100)
    >>> windows = validator.get_neural_windows(window_length=2.0, stride=1.0)
    >>> labels = validator.get_task_labels()
    >>> print(f"Extracted {len(windows)} EEG windows")
    """

    def __init__(
        self,
        n_trials: int = 200,
        n_channels: int = 22,
        sampling_rate: float = 250.0,
        data_path: str = "./bci_data",
        cache_dir: str = "./cache"
    ):
        super().__init__(data_path, cache_dir)

        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.n_trials = n_trials

        # Class mapping: 0=left_hand, 1=right_hand
        self.class_names = ['left_hand', 'right_hand']
        self.n_classes = len(self.class_names)

        # Generate mock EEG data
        self.eeg_data = self._generate_motor_imagery_data()
        logger.info(f"Initialized BCI validator with {n_trials} trials")

    def _generate_motor_imagery_data(self) -> Dict[str, Any]:
        """Generate mock motor imagery EEG data with realistic features."""
        rng = np.random.default_rng(42)

        # Trial duration: 4 seconds
        trial_duration = 4.0
        n_samples = int(trial_duration * self.sampling_rate)

        trials = []
        for trial_idx in range(self.n_trials):
            class_id = trial_idx % self.n_classes

            # Generate EEG-like signal
            eeg_trial = np.zeros((n_samples, self.n_channels))
            t = np.arange(n_samples) / self.sampling_rate

            for ch in range(self.n_channels):
                # Base noise
                signal = rng.normal(0, 10, n_samples)

                # Add motor-related oscillations
                if class_id == 0:  # Left hand - stronger alpha/mu on right hemisphere
                    if ch >= self.n_channels // 2:
                        signal += 15 * np.sin(2 * np.pi * 10 * t)  # Mu rhythm
                else:  # Right hand - stronger on left hemisphere
                    if ch < self.n_channels // 2:
                        signal += 15 * np.sin(2 * np.pi * 10 * t)

                # Add beta band modulation
                signal += 8 * np.sin(2 * np.pi * 20 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))

                eeg_trial[:, ch] = signal

            trials.append({
                'data': eeg_trial,
                'label': class_id,
                'class_name': self.class_names[class_id],
                'trial_id': trial_idx
            })

        return trials

    def get_neural_windows(
        self,
        window_length: float = 2.0,
        stride: float = 1.0,
        bin_size: Optional[float] = None
    ) -> List[NeuralWindow]:
        """Extract EEG windows from motor imagery trials."""
        windows = []

        for trial in self.eeg_data:
            trial_data = trial['data']  # [samples, channels]
            trial_duration = len(trial_data) / self.sampling_rate

            window_samples = int(window_length * self.sampling_rate)
            stride_samples = int(stride * self.sampling_rate)

            for start_idx in range(0, len(trial_data) - window_samples + 1, stride_samples):
                end_idx = start_idx + window_samples
                window_data = trial_data[start_idx:end_idx]

                window = NeuralWindow(
                    data=window_data,
                    labels=np.array([trial['label']]),
                    metadata={
                        'trial_id': trial['trial_id'],
                        'class_name': trial['class_name'],
                        'motor_class': trial['label'],
                        'window_start_time': start_idx / self.sampling_rate,
                        'modality': 'eeg',
                        'n_channels': self.n_channels
                    },
                    window_id=f"trial_{trial['trial_id']}_t_{start_idx}"
                )
                windows.append(window)

        logger.info(f"Extracted {len(windows)} EEG windows")
        return windows

    def get_task_labels(self) -> Dict[str, np.ndarray]:
        """Return motor imagery class labels."""
        labels = np.array([trial['label'] for trial in self.eeg_data])
        class_names = [trial['class_name'] for trial in self.eeg_data]

        laterality = np.array([
            0 if 'left' in name else 1 for name in class_names
        ])

        return {
            'motor_class': labels,
            'class_names': class_names,
            'laterality': laterality
        }

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        temporal_split: bool = True
    ) -> Tuple[List[int], List[int], List[int]]:
        """Split trials into train/val/test sets."""
        n_trials = len(self.eeg_data)

        if temporal_split:
            train_end = int(n_trials * train_ratio)
            val_end = int(n_trials * (train_ratio + val_ratio))

            train_trials = list(range(train_end))
            val_trials = list(range(train_end, val_end))
            test_trials = list(range(val_end, n_trials))
        else:
            # Stratified random split
            indices = np.random.permutation(n_trials)
            train_end = int(n_trials * train_ratio)
            val_end = int(n_trials * (train_ratio + val_ratio))

            train_trials = indices[:train_end].tolist()
            val_trials = indices[train_end:val_end].tolist()
            test_trials = indices[val_end:].tolist()

        return train_trials, val_trials, test_trials

    def get_neural_properties(self) -> Dict[str, Any]:
        """Return BCI-specific neural properties."""
        return {
            'n_channels': self.n_channels,
            'brain_regions': ['motor_cortex', 'sensorimotor_cortex'],
            'sampling_rate': self.sampling_rate,
            'modality': 'eeg',
            'species': 'human',
            'recording_type': 'surface_eeg',
            'task_type': 'motor_imagery',
            'n_classes': self.n_classes
        }

    def compute_motor_selectivity(
        self,
        channel_responses: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute motor selectivity metrics for validation.

        Parameters
        ----------
        channel_responses : np.ndarray
            Response matrix [n_trials, n_channels].

        Returns
        -------
        Dict[str, np.ndarray]
            Selectivity statistics including preferred classes.
        """
        labels = np.array([trial['label'] for trial in self.eeg_data])
        unique_classes = np.unique(labels)

        n_channels = channel_responses.shape[1] if len(channel_responses.shape) > 1 else 1
        selectivity_matrix = np.zeros((len(unique_classes), n_channels))

        for i, class_id in enumerate(unique_classes):
            class_mask = labels == class_id
            if len(channel_responses.shape) > 1:
                selectivity_matrix[i] = np.mean(channel_responses[class_mask], axis=0)
            else:
                selectivity_matrix[i, 0] = np.mean(channel_responses[class_mask])

        max_response = np.max(selectivity_matrix, axis=0)
        min_response = np.min(selectivity_matrix, axis=0)
        selectivity_index = (max_response - min_response) / (max_response + min_response + 1e-8)

        preferred_classes = unique_classes[np.argmax(selectivity_matrix, axis=0)]

        return {
            'class_responses': selectivity_matrix,
            'preferred_classes': preferred_classes,
            'selectivity_index': selectivity_index,
            'class_names': [self.class_names[c] for c in unique_classes]
        }
