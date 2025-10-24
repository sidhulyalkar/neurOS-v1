"""
EMG (Electromyography) Data Acquisition Script

Downloads and preprocesses muscle activity data for NeuroFMx training.

Data sources:
- Ninapro database (hand gestures, prosthetics)
- CapgMyo (high-density EMG)
- Phys IoNet EMG datasets
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
    import scipy.io as sio
    from scipy.signal import butter, filtfilt, hilbert
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy not installed. Install with: pip install scipy")
    HAS_SCIPY = False


class EMGProcessor:
    """Process EMG data for NeuroFMx."""

    def __init__(self, output_dir: str, sfreq: float = 1000.0):
        """
        Args:
            output_dir: Output directory
            sfreq: Sampling frequency (Hz) - EMG typically 1000-2000 Hz
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sfreq = sfreq

        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

    def preprocess_emg(self, emg_data: np.ndarray) -> np.ndarray:
        """
        Preprocess raw EMG data.

        Steps:
        1. Bandpass filter 20-450 Hz
        2. Notch filter at 60 Hz (line noise)
        3. Rectification (optional)
        4. Smoothing (optional)

        Args:
            emg_data: (n_samples, n_channels) raw EMG

        Returns:
            emg_processed: (n_samples, n_channels) processed EMG
        """
        n_samples, n_channels = emg_data.shape

        # Design bandpass filter (20-450 Hz)
        nyquist = self.sfreq / 2
        low_cut = 20.0 / nyquist
        high_cut = min(450.0, nyquist * 0.95) / nyquist

        b, a = butter(4, [low_cut, high_cut], btype='band')

        # Apply filter to each channel
        emg_filtered = np.zeros_like(emg_data)
        for ch in range(n_channels):
            emg_filtered[:, ch] = filtfilt(b, a, emg_data[:, ch])

        # Notch filter at 60 Hz
        notch_freq = 60.0 / nyquist
        if notch_freq < 0.95:
            b_notch, a_notch = butter(2, [notch_freq - 0.01, notch_freq + 0.01], btype='bandstop')
            for ch in range(n_channels):
                emg_filtered[:, ch] = filtfilt(b_notch, a_notch, emg_filtered[:, ch])

        return emg_filtered.astype(np.float32)

    def compute_envelope(self, emg_data: np.ndarray, window_ms: float = 50.0) -> np.ndarray:
        """
        Compute EMG envelope using Hilbert transform or moving average.

        Args:
            emg_data: (n_samples, n_channels) filtered EMG
            window_ms: Smoothing window in milliseconds

        Returns:
            envelope: (n_samples, n_channels) EMG envelope
        """
        # Rectify
        emg_rect = np.abs(emg_data)

        # Smooth with moving average
        window_samples = int(window_ms * self.sfreq / 1000.0)

        envelope = np.zeros_like(emg_rect)
        for ch in range(emg_rect.shape[1]):
            # Simple moving average
            kernel = np.ones(window_samples) / window_samples
            envelope[:, ch] = np.convolve(emg_rect[:, ch], kernel, mode='same')

        return envelope

    def create_sequences(
        self,
        emg_data: np.ndarray,
        labels: np.ndarray,
        kinematics: Optional[np.ndarray] = None,
        sequence_length: int = 200,
        stride: int = 100
    ) -> List[Dict]:
        """
        Create sequences from EMG data.

        Args:
            emg_data: (n_samples, n_channels) EMG data
            labels: (n_samples,) gesture/movement labels
            kinematics: (n_samples, n_dims) kinematic data (optional)
            sequence_length: Length of each sequence
            stride: Stride between sequences

        Returns:
            sequences: List of sequence dicts
        """
        sequences = []
        n_samples = emg_data.shape[0]

        for start_idx in range(0, n_samples - sequence_length, stride):
            end_idx = start_idx + sequence_length

            seq_dict = {
                'emg': emg_data[start_idx:end_idx].astype(np.float32),
                'labels': labels[start_idx:end_idx].astype(np.int32),
                'metadata': {
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
            }

            if kinematics is not None:
                seq_dict['kinematics'] = kinematics[start_idx:end_idx].astype(np.float32)

            sequences.append(seq_dict)

        return sequences

    def process_ninapro_subject(
        self,
        data_file: str,
        subject_id: str
    ) -> int:
        """
        Process Ninapro dataset subject.

        Ninapro format: .mat files with fields:
        - emg: (n_samples, n_channels)
        - restimulus: (n_samples,) gesture labels
        - rerepetition: (n_samples,) repetition numbers

        Args:
            data_file: Path to .mat file
            subject_id: Subject identifier

        Returns:
            n_sequences: Number of sequences created
        """
        print(f"  Loading Ninapro file: {data_file}")

        # Load .mat file
        try:
            data = sio.loadmat(data_file)
        except:
            print(f"  Error loading file")
            return 0

        # Extract EMG
        if 'emg' in data:
            emg_raw = data['emg']
        else:
            print(f"  No EMG data found in file")
            return 0

        # Extract labels
        if 'restimulus' in data:
            labels = data['restimulus'].flatten()
        elif 'stimulus' in data:
            labels = data['stimulus'].flatten()
        else:
            labels = np.zeros(emg_raw.shape[0], dtype=np.int32)

        # Preprocess
        print(f"  Preprocessing EMG ({emg_raw.shape[1]} channels)...")
        emg_processed = self.preprocess_emg(emg_raw)

        # Optionally compute envelope
        # emg_processed = self.compute_envelope(emg_processed)

        # Create sequences
        print(f"  Creating sequences...")
        sequences = self.create_sequences(
            emg_processed,
            labels,
            sequence_length=int(self.sfreq * 0.2)  # 200ms windows
        )

        # Split into train/val/test
        n_sequences = len(sequences)
        n_train = int(0.8 * n_sequences)
        n_val = int(0.1 * n_sequences)

        splits = {
            'train': sequences[:n_train],
            'val': sequences[n_train:n_train+n_val],
            'test': sequences[n_train+n_val:]
        }

        # Save sequences
        for split_name, split_seqs in splits.items():
            for i, seq in enumerate(split_seqs):
                save_path = self.output_dir / split_name / f"emg_{subject_id}_seq{i:04d}.npz"
                np.savez_compressed(
                    save_path,
                    emg=seq['emg'],
                    labels=seq['labels'],
                    metadata=seq['metadata']
                )

        print(f"  Created {n_sequences} sequences")
        return n_sequences


def download_ninapro_data(database: int = 1, cache_dir: str = './data/emg_cache'):
    """
    Download Ninapro database.

    Args:
        database: Database number (1-10)
        cache_dir: Cache directory

    Returns:
        file_paths: List of downloaded file paths
    """
    print(f"Ninapro Database {database}")
    print("Visit: http://ninapro.hevs.ch/")
    print("\nTo download:")
    print(f"1. Go to http://ninapro.hevs.ch/DB{database}")
    print("2. Register and download the dataset")
    print(f"3. Extract to {cache_dir}/ninapro_db{database}/")

    # Check for existing files
    cache_path = Path(cache_dir) / f"ninapro_db{database}"

    if not cache_path.exists():
        print(f"\nDirectory not found: {cache_path}")
        print("Please download data manually first.")
        return []

    # Find .mat files
    mat_files = list(cache_path.glob('**/*.mat'))

    if len(mat_files) > 0:
        print(f"\nFound {len(mat_files)} .mat files")
        return [str(f) for f in mat_files]

    print("\nNo .mat files found. Please download data.")
    return []


def download_capgmyo_data(cache_dir: str = './data/emg_cache'):
    """
    Download CapgMyo database (high-density EMG).

    Visit: http://zju-capg.org/myo/data/

    Returns:
        file_paths: List of data files
    """
    print("CapgMyo Database (High-Density EMG)")
    print("Visit: http://zju-capg.org/myo/data/")
    print("\nTo download:")
    print("1. Go to http://zju-capg.org/myo/data/")
    print("2. Download the dataset (DB-a, DB-b, or DB-c)")
    print(f"3. Extract to {cache_dir}/capgmyo/")

    cache_path = Path(cache_dir) / "capgmyo"

    if not cache_path.exists():
        print(f"\nDirectory not found: {cache_path}")
        return []

    data_files = list(cache_path.glob('**/*.mat'))

    if len(data_files) > 0:
        print(f"\nFound {len(data_files)} files")
        return [str(f) for f in data_files]

    return []


def main():
    parser = argparse.ArgumentParser(description='Download and preprocess EMG data')

    parser.add_argument('--output_dir', type=str,
                       default='./data/emg/processed',
                       help='Output directory for processed data')
    parser.add_argument('--cache_dir', type=str,
                       default='./data/emg_cache',
                       help='Cache directory for raw downloads')
    parser.add_argument('--sfreq', type=float, default=1000.0,
                       help='Sampling frequency (Hz)')
    parser.add_argument('--source', type=str, default='ninapro',
                       choices=['ninapro', 'capgmyo', 'local'],
                       help='Data source')
    parser.add_argument('--database', type=int, default=1,
                       help='Ninapro database number (1-10)')
    parser.add_argument('--local_dir', type=str, default=None,
                       help='Local directory with EMG files')

    args = parser.parse_args()

    if not HAS_SCIPY:
        print("Error: scipy is required for EMG processing")
        print("Install with: pip install scipy")
        return

    print("="*80)
    print("EMG Data Acquisition for NeuroFMx")
    print("="*80)

    # Initialize processor
    processor = EMGProcessor(args.output_dir, sfreq=args.sfreq)

    # Locate data files
    print(f"\nStep 1: Locating EMG data from {args.source}...")

    if args.source == 'ninapro':
        file_paths = download_ninapro_data(args.database, args.cache_dir)
    elif args.source == 'capgmyo':
        file_paths = download_capgmyo_data(args.cache_dir)
    elif args.source == 'local':
        if args.local_dir is None:
            print("Error: --local_dir required when using source=local")
            return
        local_path = Path(args.local_dir)
        file_paths = list(local_path.glob('**/*.mat'))
        file_paths = [str(f) for f in file_paths]
    else:
        file_paths = []

    if len(file_paths) == 0:
        print("\nNo EMG files found.")
        print("\nTo use this script:")
        print("1. Download EMG data from Ninapro or CapgMyo")
        print("2. Extract to cache directory")
        print("3. Run again with appropriate --source flag")
        return

    print(f"Found {len(file_paths)} EMG files")

    # Process each file
    print(f"\nStep 2: Processing files...")
    total_sequences = 0

    for i, file_path in enumerate(file_paths[:30]):  # Limit to 30 subjects
        subject_id = f"subj{i:03d}"
        print(f"\nProcessing file {i+1}/{min(len(file_paths), 30)}: {Path(file_path).name}")

        try:
            n_seq = processor.process_ninapro_subject(file_path, subject_id)
            total_sequences += n_seq
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print(f"Processing complete! Total sequences: {total_sequences}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Print summary
    print("\nDataset summary:")
    for split in ['train', 'val', 'test']:
        n_files = len(list((Path(args.output_dir) / split).glob('*.npz')))
        print(f"  {split}: {n_files} sequences")


if __name__ == '__main__':
    main()
