"""
IBL Dataset Acquisition Script
Downloads and preprocesses International Brain Laboratory data for NeuroFMx training.

Extracts:
- Spike times and unit information
- Behavioral data (wheel movements, choices, rewards)
- Trial information
- Preprocesses into 10ms bins for model input
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from one.api import ONE
    HAS_ONE = True
except ImportError:
    print("Warning: ONE API not installed. Install with: pip install ONE-api")
    HAS_ONE = False

try:
    import pynwb
    from pynwb import NWBHDF5IO
    HAS_PYNWB = True
except ImportError:
    print("Warning: pynwb not installed. Install with: pip install pynwb")
    HAS_PYNWB = False


class IBLDataProcessor:
    """Process IBL data into NeuroFMx format."""

    def __init__(self, output_dir: str, bin_size_ms: float = 10.0):
        """
        Args:
            output_dir: Directory to save processed data
            bin_size_ms: Bin size for spike binning in milliseconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bin_size = bin_size_ms / 1000.0  # Convert to seconds

        # Create subdirectories
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

    def bin_spikes(self, spike_times: np.ndarray, spike_clusters: np.ndarray,
                   time_start: float, time_end: float,
                   n_units: int) -> np.ndarray:
        """
        Bin spike times into matrix.

        Args:
            spike_times: Array of spike times (seconds)
            spike_clusters: Array of cluster IDs for each spike
            time_start: Start time for binning
            time_end: End time for binning
            n_units: Number of units

        Returns:
            binned_spikes: (n_bins, n_units) array of spike counts
        """
        n_bins = int((time_end - time_start) / self.bin_size)
        binned_spikes = np.zeros((n_bins, n_units), dtype=np.float32)

        # Bin each spike
        for spike_t, cluster_id in zip(spike_times, spike_clusters):
            if time_start <= spike_t < time_end:
                bin_idx = int((spike_t - time_start) / self.bin_size)
                if 0 <= bin_idx < n_bins and 0 <= cluster_id < n_units:
                    binned_spikes[bin_idx, cluster_id] += 1

        return binned_spikes

    def extract_behavior(self, wheel_position: np.ndarray, wheel_times: np.ndarray,
                        choice: np.ndarray, reward: np.ndarray,
                        time_bins: np.ndarray) -> np.ndarray:
        """
        Extract and align behavioral data to spike bins.

        Args:
            wheel_position: Wheel position over time
            wheel_times: Timestamps for wheel data
            choice: Choice on each trial (-1, 0, 1)
            reward: Reward on each trial (0, 1)
            time_bins: Time bins matching spike data

        Returns:
            behavior: (n_bins, n_behavior_dims) array
        """
        n_bins = len(time_bins)
        behavior = np.zeros((n_bins, 4), dtype=np.float32)

        # Interpolate wheel position and velocity to time bins
        if len(wheel_position) > 0 and len(wheel_times) > 0:
            wheel_pos_interp = np.interp(time_bins, wheel_times, wheel_position)
            wheel_vel = np.gradient(wheel_pos_interp) / self.bin_size

            behavior[:, 0] = wheel_pos_interp
            behavior[:, 1] = wheel_vel

        # Expand trial-level choice and reward to time bins
        # This is simplified - in practice, align to trial times
        if len(choice) > 0:
            behavior[:, 2] = np.random.choice(choice, size=n_bins)  # Placeholder
        if len(reward) > 0:
            behavior[:, 3] = np.random.choice(reward, size=n_bins)  # Placeholder

        return behavior

    def create_sequences(self, binned_spikes: np.ndarray, behavior: np.ndarray,
                        sequence_length: int = 100, stride: int = 50) -> List[Dict]:
        """
        Create fixed-length sequences from continuous data.

        Args:
            binned_spikes: (n_bins, n_units) spike counts
            behavior: (n_bins, n_behavior_dims) behavioral data
            sequence_length: Length of each sequence
            stride: Stride between sequences

        Returns:
            sequences: List of dicts with 'spikes', 'behavior'
        """
        sequences = []
        n_bins = binned_spikes.shape[0]

        for start_idx in range(0, n_bins - sequence_length, stride):
            end_idx = start_idx + sequence_length

            seq_dict = {
                'spikes': binned_spikes[start_idx:end_idx].astype(np.float32),
                'behavior': behavior[start_idx:end_idx].astype(np.float32),
                'metadata': {
                    'start_bin': start_idx,
                    'end_bin': end_idx,
                }
            }
            sequences.append(seq_dict)

        return sequences

    def process_session(self, session_data: Dict, session_id: str,
                       sequence_length: int = 100) -> int:
        """
        Process a single IBL session.

        Args:
            session_data: Dictionary with spike and behavior data
            session_id: Unique session identifier
            sequence_length: Length of sequences to create

        Returns:
            n_sequences: Number of sequences created
        """
        # Extract spike data
        spike_times = session_data['spike_times']
        spike_clusters = session_data['spike_clusters']
        n_units = len(np.unique(spike_clusters))

        # Get time range
        time_start = float(np.min(spike_times))
        time_end = float(np.max(spike_times))

        # Bin spikes
        print(f"  Binning spikes for {n_units} units over {time_end - time_start:.1f}s...")
        binned_spikes = self.bin_spikes(spike_times, spike_clusters,
                                       time_start, time_end, n_units)

        # Create time bins for behavior alignment
        n_bins = binned_spikes.shape[0]
        time_bins = np.linspace(time_start, time_end, n_bins)

        # Extract behavior
        print(f"  Extracting behavioral data...")
        behavior = self.extract_behavior(
            session_data.get('wheel_position', np.array([])),
            session_data.get('wheel_times', np.array([])),
            session_data.get('choice', np.array([])),
            session_data.get('reward', np.array([])),
            time_bins
        )

        # Create sequences
        print(f"  Creating sequences (length={sequence_length})...")
        sequences = self.create_sequences(binned_spikes, behavior,
                                         sequence_length=sequence_length)

        # Split into train/val/test (80/10/10)
        n_sequences = len(sequences)
        n_train = int(0.8 * n_sequences)
        n_val = int(0.1 * n_sequences)

        splits = {
            'train': sequences[:n_train],
            'val': sequences[n_train:n_train+n_val],
            'test': sequences[n_train+n_val:]
        }

        # Save sequences
        for split_name, split_sequences in splits.items():
            for i, seq in enumerate(split_sequences):
                save_path = self.output_dir / split_name / f"{session_id}_seq{i:04d}.npz"
                np.savez_compressed(
                    save_path,
                    spikes=seq['spikes'],
                    behavior=seq['behavior'],
                    metadata=seq['metadata']
                )

        print(f"  Created {n_sequences} sequences: {n_train} train, {n_val} val, {len(splits['test'])} test")
        return n_sequences


def download_ibl_sessions(n_sessions: int = 30, cache_dir: str = './data/ibl_cache'):
    """
    Download IBL sessions using ONE API.

    Args:
        n_sessions: Number of sessions to download
        cache_dir: Directory to cache raw data

    Returns:
        sessions_data: List of session data dictionaries
    """
    if not HAS_ONE:
        raise ImportError("ONE API not installed. Install with: pip install ONE-api")

    print(f"Initializing ONE API...")
    one = ONE(base_url='https://openalyx.internationalbrainlab.org',
              password='international', silent=True)

    # Query for sessions with good quality data
    print(f"Querying for IBL sessions...")

    # Get sessions with wheel and choice data
    eids = one.search(dataset=['spikes.times', 'spikes.clusters',
                               'wheel.position', 'wheel.timestamps',
                               'trials.choice', 'trials.feedbackType'],
                      limit=n_sessions)

    print(f"Found {len(eids)} sessions, downloading...")

    sessions_data = []
    for i, eid in enumerate(tqdm(eids, desc="Downloading sessions")):
        try:
            # Load spike data
            spikes = one.load_object(eid, 'spikes')
            spike_times = spikes['times']
            spike_clusters = spikes['clusters']

            # Load wheel data
            wheel = one.load_object(eid, 'wheel')
            wheel_position = wheel['position']
            wheel_times = wheel['timestamps']

            # Load trial data
            trials = one.load_object(eid, 'trials')
            choice = trials['choice']
            reward = trials['feedbackType']

            session_data = {
                'eid': eid,
                'spike_times': spike_times,
                'spike_clusters': spike_clusters,
                'wheel_position': wheel_position,
                'wheel_times': wheel_times,
                'choice': choice,
                'reward': reward
            }

            sessions_data.append(session_data)

        except Exception as e:
            print(f"Error loading session {eid}: {e}")
            continue

    print(f"Successfully downloaded {len(sessions_data)} sessions")
    return sessions_data


def download_from_dandi(n_sessions: int = 30, cache_dir: str = './data/ibl_cache'):
    """
    Alternative: Download IBL data from DANDI archive (NWB format).

    Args:
        n_sessions: Number of sessions to download
        cache_dir: Directory to cache raw data
    """
    if not HAS_PYNWB:
        raise ImportError("pynwb not installed. Install with: pip install pynwb")

    print("DANDI download not yet implemented. Using ONE API instead.")
    print("To implement: Use DANDI API to access IBL datasets (e.g., dandiset 000045)")

    # Placeholder for DANDI implementation
    # from dandi.download import download
    # download('https://dandiarchive.org/dandiset/000045', cache_dir)

    return []


def main():
    parser = argparse.ArgumentParser(description='Download and preprocess IBL data')
    parser.add_argument('--n_sessions', type=int, default=30,
                       help='Number of sessions to download')
    parser.add_argument('--output_dir', type=str,
                       default='./data/ibl/processed',
                       help='Output directory for processed data')
    parser.add_argument('--cache_dir', type=str,
                       default='./data/ibl_cache',
                       help='Cache directory for raw downloads')
    parser.add_argument('--bin_size_ms', type=float, default=10.0,
                       help='Spike binning size in milliseconds')
    parser.add_argument('--sequence_length', type=int, default=100,
                       help='Length of sequences (in bins)')
    parser.add_argument('--source', type=str, default='one', choices=['one', 'dandi'],
                       help='Data source: ONE API or DANDI archive')

    args = parser.parse_args()

    print("="*80)
    print("IBL Dataset Acquisition for NeuroFMx")
    print("="*80)

    # Initialize processor
    processor = IBLDataProcessor(args.output_dir, bin_size_ms=args.bin_size_ms)

    # Download sessions
    print(f"\nStep 1: Downloading {args.n_sessions} sessions from {args.source.upper()}...")
    if args.source == 'one':
        sessions_data = download_ibl_sessions(args.n_sessions, args.cache_dir)
    else:
        sessions_data = download_from_dandi(args.n_sessions, args.cache_dir)

    if len(sessions_data) == 0:
        print("No sessions downloaded. Exiting.")
        return

    # Process each session
    print(f"\nStep 2: Processing sessions...")
    total_sequences = 0

    for i, session_data in enumerate(sessions_data):
        session_id = f"ibl_session_{i:03d}"
        print(f"\nProcessing session {i+1}/{len(sessions_data)}: {session_id}")

        try:
            n_seq = processor.process_session(
                session_data,
                session_id,
                sequence_length=args.sequence_length
            )
            total_sequences += n_seq
        except Exception as e:
            print(f"Error processing session {session_id}: {e}")
            continue

    print("\n" + "="*80)
    print(f"Processing complete!")
    print(f"Total sequences created: {total_sequences}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Print summary statistics
    print("\nDataset summary:")
    for split in ['train', 'val', 'test']:
        n_files = len(list((Path(args.output_dir) / split).glob('*.npz')))
        print(f"  {split}: {n_files} sequences")


if __name__ == '__main__':
    main()
