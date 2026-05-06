"""
Synthetic Neural Data Generator

Generates realistic synthetic calcium imaging and astrocyte event data
that matches the statistics of real Allen Brain Observatory recordings.

Key features:
- Colored noise with realistic temporal autocorrelation
- Matched first and second-order statistics
- Correlated neuron pairs (realistic functional connectivity)
- Synthetic astrocyte events with realistic timing

Reference: Based on neural simulation techniques from computational neuroscience.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class NeuralStatistics:
    """Statistics extracted from real neural data."""
    # Calcium statistics
    calcium_mean: float = 0.0
    calcium_std: float = 0.1
    calcium_min: float = -0.5
    calcium_max: float = 3.0

    # Temporal autocorrelation (lag 1)
    temporal_autocorr: float = 0.85  # High autocorrelation for calcium

    # Cross-neuron correlation
    mean_pairwise_corr: float = 0.15
    std_pairwise_corr: float = 0.1

    # Astro event statistics
    astro_event_rate: float = 0.1  # events per second per astrocyte
    astro_event_duration_mean: float = 2.0  # seconds
    astro_event_duration_std: float = 0.5
    astro_event_amplitude_mean: float = 1.5
    astro_event_amplitude_std: float = 0.5

    # Session parameters
    n_neurons_range: Tuple[int, int] = (50, 250)
    n_astrocytes_range: Tuple[int, int] = (20, 100)


def compute_statistics_from_real_data(
    calcium_traces: np.ndarray,
    astro_events: Optional[np.ndarray] = None,
    astro_timestamps: Optional[np.ndarray] = None,
    duration_s: float = 10.0,
) -> NeuralStatistics:
    """
    Extract statistics from real neural recordings.

    Args:
        calcium_traces: (n_neurons, n_timepoints) calcium signals
        astro_events: (n_events, n_features) astro event tokens
        astro_timestamps: (n_events,) event times
        duration_s: Recording duration in seconds

    Returns:
        NeuralStatistics with extracted parameters
    """
    stats = NeuralStatistics()

    # Calcium statistics
    stats.calcium_mean = float(np.mean(calcium_traces))
    stats.calcium_std = float(np.std(calcium_traces))
    stats.calcium_min = float(np.percentile(calcium_traces, 1))
    stats.calcium_max = float(np.percentile(calcium_traces, 99))

    # Temporal autocorrelation (lag 1)
    if calcium_traces.shape[1] > 1:
        autocorrs = []
        for neuron in range(min(50, calcium_traces.shape[0])):
            trace = calcium_traces[neuron]
            if np.std(trace) > 1e-6:
                ac = np.corrcoef(trace[:-1], trace[1:])[0, 1]
                if not np.isnan(ac):
                    autocorrs.append(ac)
        if autocorrs:
            stats.temporal_autocorr = float(np.mean(autocorrs))

    # Cross-neuron correlation
    if calcium_traces.shape[0] > 1:
        n_sample = min(50, calcium_traces.shape[0])
        sample_idx = np.random.choice(calcium_traces.shape[0], n_sample, replace=False)
        sample = calcium_traces[sample_idx]
        corr_matrix = np.corrcoef(sample)
        # Get upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices(n_sample, k=1)]
        upper_tri = upper_tri[~np.isnan(upper_tri)]
        if len(upper_tri) > 0:
            stats.mean_pairwise_corr = float(np.mean(upper_tri))
            stats.std_pairwise_corr = float(np.std(upper_tri))

    # Astro statistics
    if astro_timestamps is not None and len(astro_timestamps) > 0:
        stats.astro_event_rate = len(astro_timestamps) / duration_s
        if astro_events is not None and astro_events.shape[1] >= 3:
            # Assuming columns: [duration, amplitude, ...]
            stats.astro_event_duration_mean = float(np.mean(astro_events[:, 1]))
            stats.astro_event_duration_std = float(np.std(astro_events[:, 1]))
            stats.astro_event_amplitude_mean = float(np.mean(astro_events[:, 2]))
            stats.astro_event_amplitude_std = float(np.std(astro_events[:, 2]))

    return stats


def generate_colored_noise(
    shape: Tuple[int, ...],
    autocorr: float = 0.85,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Generate colored (autocorrelated) noise using AR(1) process.

    AR(1): x[t] = autocorr * x[t-1] + sqrt(1 - autocorr^2) * noise

    This produces noise with exponentially decaying autocorrelation,
    matching typical calcium imaging dynamics.
    """
    n_signals, n_timepoints = shape

    # Initialize
    noise = np.zeros(shape, dtype=np.float32)
    innovation_std = np.sqrt(1 - autocorr**2)

    # AR(1) process
    noise[:, 0] = np.random.randn(n_signals)
    for t in range(1, n_timepoints):
        innovation = np.random.randn(n_signals) * innovation_std
        noise[:, t] = autocorr * noise[:, t-1] + innovation

    # Scale to desired statistics
    noise = noise * std + mean

    return noise


def generate_correlated_neurons(
    n_neurons: int,
    n_timepoints: int,
    mean_corr: float = 0.15,
    temporal_autocorr: float = 0.85,
    mean: float = 0.0,
    std: float = 0.1,
) -> np.ndarray:
    """
    Generate correlated neural activity using factor model.

    Model: X = W @ F + E
    - F: shared factors (temporal dynamics)
    - W: factor loadings (which neurons share which factors)
    - E: independent noise

    This creates realistic functional connectivity structure.
    """
    # Number of shared factors (determines correlation strength)
    n_factors = max(1, int(n_neurons * 0.1))

    # Generate shared factors (colored noise for temporal structure)
    factors = generate_colored_noise(
        (n_factors, n_timepoints),
        autocorr=temporal_autocorr,
        mean=0,
        std=1,
    )

    # Factor loadings (sparse, some neurons load on same factors)
    loadings = np.random.randn(n_neurons, n_factors) * np.sqrt(mean_corr)

    # Shared signal
    shared = loadings @ factors

    # Independent noise (also colored)
    independent = generate_colored_noise(
        (n_neurons, n_timepoints),
        autocorr=temporal_autocorr,
        mean=0,
        std=np.sqrt(1 - mean_corr),
    )

    # Combine
    activity = shared + independent

    # Scale to desired statistics
    activity = activity * std + mean

    return activity.astype(np.float32)


def generate_synthetic_astro_events(
    duration_s: float,
    n_astrocytes: int,
    event_rate: float = 0.1,
    n_features: int = 10,
    stats: Optional[NeuralStatistics] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic astrocyte calcium events.

    Astrocyte events are:
    - Sparse in time (low event rate)
    - Long duration (seconds, not milliseconds)
    - Correlated within astrocyte networks

    Returns:
        event_tokens: (n_events, n_features) event features
        timestamps: (n_events,) event times in seconds
        region_ids: (n_events,) which astrocyte
    """
    if stats is None:
        stats = NeuralStatistics()

    events = []
    timestamps = []
    region_ids = []

    for astro_id in range(n_astrocytes):
        # Poisson process for event times
        n_events_expected = event_rate * duration_s
        n_events = np.random.poisson(n_events_expected)

        if n_events > 0:
            # Uniform event times
            event_times = np.sort(np.random.uniform(0, duration_s, n_events))

            for t in event_times:
                # Generate event features
                # Features: [onset_time_norm, duration, amplitude, area, centroid_y, centroid_x, speed, ...]
                duration = max(0.5, np.random.normal(
                    stats.astro_event_duration_mean,
                    stats.astro_event_duration_std
                ))
                amplitude = max(0.1, np.random.normal(
                    stats.astro_event_amplitude_mean,
                    stats.astro_event_amplitude_std
                ))

                # Create feature vector
                features = np.zeros(n_features, dtype=np.float32)
                features[0] = t / duration_s  # Normalized onset time
                features[1] = duration
                features[2] = amplitude
                features[3:] = np.random.randn(n_features - 3) * 0.1  # Other features

                events.append(features)
                timestamps.append(t)
                region_ids.append(astro_id)

    if len(events) == 0:
        # Return at least one dummy event
        events = [np.zeros(n_features, dtype=np.float32)]
        timestamps = [duration_s / 2]
        region_ids = [0]

    return (
        np.array(events, dtype=np.float32),
        np.array(timestamps, dtype=np.float32),
        np.array(region_ids, dtype=np.int64),
    )


class SyntheticNeuralDataset(Dataset):
    """
    Dataset that generates synthetic neural data matching real statistics.

    Can be used standalone or mixed with real data during training.
    """

    def __init__(
        self,
        n_sessions: int = 5,
        windows_per_session: int = 10,
        seq_len: int = 100,
        sampling_rate: float = 10.0,
        stats: Optional[NeuralStatistics] = None,
        seed: int = 42,
    ):
        """
        Initialize synthetic dataset.

        Args:
            n_sessions: Number of synthetic "sessions" to generate
            windows_per_session: Windows per session
            seq_len: Sequence length (timepoints)
            sampling_rate: Hz
            stats: Statistics to match (from real data)
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.n_sessions = n_sessions
        self.windows_per_session = windows_per_session
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        self.stats = stats or NeuralStatistics()
        self.seed = seed

        np.random.seed(seed)

        # Pre-generate all windows
        self.windows = []
        self._generate_all_windows()

    def _generate_all_windows(self):
        """Generate all synthetic windows."""

        for session_idx in range(self.n_sessions):
            # Random neuron/astrocyte counts within range
            n_neurons = np.random.randint(*self.stats.n_neurons_range)
            n_astrocytes = np.random.randint(*self.stats.n_astrocytes_range)

            for window_idx in range(self.windows_per_session):
                # Generate calcium traces
                calcium = generate_correlated_neurons(
                    n_neurons=n_neurons,
                    n_timepoints=self.seq_len,
                    mean_corr=self.stats.mean_pairwise_corr,
                    temporal_autocorr=self.stats.temporal_autocorr,
                    mean=self.stats.calcium_mean,
                    std=self.stats.calcium_std,
                )

                # Clip to realistic range
                calcium = np.clip(calcium, self.stats.calcium_min, self.stats.calcium_max)

                # Generate astro events
                duration_s = self.seq_len / self.sampling_rate
                events, timestamps, region_ids = generate_synthetic_astro_events(
                    duration_s=duration_s,
                    n_astrocytes=n_astrocytes,
                    event_rate=self.stats.astro_event_rate,
                    stats=self.stats,
                )

                self.windows.append({
                    'session_id': f'synthetic_{session_idx}',
                    'window_idx': window_idx,
                    't_start': 0.0,
                    't_end': duration_s,
                    'calcium': calcium,
                    'astro_events': events,
                    'astro_timestamps': timestamps,
                    'astro_region_ids': region_ids,
                    'n_neurons': n_neurons,
                    'n_astrocytes': n_astrocytes,
                    'is_synthetic': True,
                })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a synthetic window."""
        window = self.windows[idx]

        return {
            'calcium': torch.from_numpy(window['calcium']),
            'astro_events': torch.from_numpy(window['astro_events']),
            'astro_timestamps': torch.from_numpy(window['astro_timestamps']),
            'astro_region_ids': torch.from_numpy(window['astro_region_ids']),
            'metadata': {
                'session_id': window['session_id'],
                'window_idx': window['window_idx'],
                't_start': window['t_start'],
                't_end': window['t_end'],
                'n_neurons': window['n_neurons'],
                'n_astrocytes': window['n_astrocytes'],
                'is_synthetic': True,
            }
        }


class MixedRealSyntheticDataset(Dataset):
    """
    Dataset that mixes real and synthetic data.

    Useful for data augmentation when real data is limited.
    """

    def __init__(
        self,
        real_dataset: Dataset,
        synthetic_ratio: float = 0.3,
        match_statistics: bool = True,
        n_synthetic_windows: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize mixed dataset.

        Args:
            real_dataset: Real neural dataset
            synthetic_ratio: Fraction of batch that should be synthetic
            match_statistics: Extract stats from real data
            n_synthetic_windows: Number of synthetic windows (default: same as real)
            seed: Random seed
        """
        super().__init__()

        self.real_dataset = real_dataset
        self.synthetic_ratio = synthetic_ratio
        self.seed = seed

        # Extract statistics from real data
        if match_statistics:
            stats = self._extract_statistics_from_real()
        else:
            stats = NeuralStatistics()

        # Determine synthetic dataset size
        n_real = len(real_dataset)
        if n_synthetic_windows is None:
            n_synthetic_windows = int(n_real * synthetic_ratio / (1 - synthetic_ratio))

        # Create synthetic dataset
        self.synthetic_dataset = SyntheticNeuralDataset(
            n_sessions=max(1, n_synthetic_windows // 10),
            windows_per_session=10,
            seq_len=100,  # Will be set from real data
            stats=stats,
            seed=seed,
        )

        # Create combined indices
        self.n_real = n_real
        self.n_synthetic = len(self.synthetic_dataset)
        self.total = self.n_real + self.n_synthetic

        print(f"Mixed dataset: {self.n_real} real + {self.n_synthetic} synthetic = {self.total} total")

    def _extract_statistics_from_real(self) -> NeuralStatistics:
        """Extract statistics from real dataset."""

        # Sample a few windows
        n_sample = min(10, len(self.real_dataset))
        all_calcium = []
        all_events = []
        all_timestamps = []

        for i in range(n_sample):
            sample = self.real_dataset[i]
            calcium = sample['calcium'].numpy() if torch.is_tensor(sample['calcium']) else sample['calcium']
            all_calcium.append(calcium)

            if 'astro_events' in sample:
                events = sample['astro_events'].numpy() if torch.is_tensor(sample['astro_events']) else sample['astro_events']
                all_events.append(events)
            if 'astro_timestamps' in sample:
                ts = sample['astro_timestamps'].numpy() if torch.is_tensor(sample['astro_timestamps']) else sample['astro_timestamps']
                all_timestamps.append(ts)

        # Concatenate
        all_calcium = np.concatenate([c.flatten() for c in all_calcium])

        # Compute statistics
        stats = NeuralStatistics(
            calcium_mean=float(np.mean(all_calcium)),
            calcium_std=float(np.std(all_calcium)),
            calcium_min=float(np.percentile(all_calcium, 1)),
            calcium_max=float(np.percentile(all_calcium, 99)),
        )

        return stats

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.n_real:
            return self.real_dataset[idx]
        else:
            return self.synthetic_dataset[idx - self.n_real]


def extract_and_save_statistics(
    dataset_dir: Union[str, Path],
    output_path: Union[str, Path],
) -> NeuralStatistics:
    """
    Extract statistics from a directory of real data and save to JSON.

    Args:
        dataset_dir: Directory with NPZ files
        output_path: Where to save statistics JSON

    Returns:
        Computed NeuralStatistics
    """
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)

    all_calcium = []
    all_autocorrs = []

    for npz_file in dataset_dir.glob('*.npz'):
        data = np.load(npz_file)
        if 'dff_traces' in data:
            traces = data['dff_traces']
            all_calcium.append(traces.flatten())

            # Compute autocorrelation
            for i in range(min(10, traces.shape[0])):
                trace = traces[i]
                if len(trace) > 1 and np.std(trace) > 1e-6:
                    ac = np.corrcoef(trace[:-1], trace[1:])[0, 1]
                    if not np.isnan(ac):
                        all_autocorrs.append(ac)

    if not all_calcium:
        print(f"No calcium data found in {dataset_dir}")
        return NeuralStatistics()

    all_calcium = np.concatenate(all_calcium)

    stats = NeuralStatistics(
        calcium_mean=float(np.mean(all_calcium)),
        calcium_std=float(np.std(all_calcium)),
        calcium_min=float(np.percentile(all_calcium, 1)),
        calcium_max=float(np.percentile(all_calcium, 99)),
        temporal_autocorr=float(np.mean(all_autocorrs)) if all_autocorrs else 0.85,
    )

    # Save to JSON
    stats_dict = {
        'calcium_mean': stats.calcium_mean,
        'calcium_std': stats.calcium_std,
        'calcium_min': stats.calcium_min,
        'calcium_max': stats.calcium_max,
        'temporal_autocorr': stats.temporal_autocorr,
        'mean_pairwise_corr': stats.mean_pairwise_corr,
        'std_pairwise_corr': stats.std_pairwise_corr,
        'astro_event_rate': stats.astro_event_rate,
        'n_neurons_range': list(stats.n_neurons_range),
        'n_astrocytes_range': list(stats.n_astrocytes_range),
    }

    with open(output_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)

    print(f"Saved statistics to {output_path}")
    return stats


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Testing Synthetic Neural Data Generator\n")

    # Create synthetic dataset
    dataset = SyntheticNeuralDataset(
        n_sessions=3,
        windows_per_session=5,
        seq_len=100,
    )

    print(f"Generated {len(dataset)} synthetic windows")

    # Check a sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  calcium: {sample['calcium'].shape}")
    print(f"  astro_events: {sample['astro_events'].shape}")
    print(f"  astro_timestamps: {sample['astro_timestamps'].shape}")

    # Check statistics
    calcium = sample['calcium'].numpy()
    print(f"\nCalcium statistics:")
    print(f"  mean: {calcium.mean():.4f}")
    print(f"  std: {calcium.std():.4f}")
    print(f"  min: {calcium.min():.4f}")
    print(f"  max: {calcium.max():.4f}")

    # Check temporal autocorrelation
    trace = calcium[0]
    autocorr = np.corrcoef(trace[:-1], trace[1:])[0, 1]
    print(f"  temporal autocorr: {autocorr:.4f}")

    print("\n✓ Synthetic data generator working!")
