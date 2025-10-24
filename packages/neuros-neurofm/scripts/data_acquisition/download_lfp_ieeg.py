"""
LFP (Local Field Potential) and iEEG (Intracranial EEG) Data Acquisition Script

Downloads and preprocesses LFP and iEEG data for NeuroFMx training.

Data sources:
- Allen Neuropixels LFP
- DANDI UCLA seizure dataset (000004)
- DANDI epilepsy datasets
- MNI Open iEEG Atlas
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
    from scipy.signal import butter, filtfilt, spectrogram
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy not installed. Install with: pip install scipy")
    HAS_SCIPY = False

try:
    import mne
    HAS_MNE = True
except ImportError:
    print("Warning: MNE not installed. Install with: pip install mne")
    HAS_MNE = False


class LFPProcessor:
    """Process LFP/iEEG data for NeuroFMx."""

    def __init__(self, output_dir: str, sfreq: float = 1000.0):
        """
        Args:
            output_dir: Output directory
            sfreq: Target sampling frequency (Hz) - LFP typically 1000-2500 Hz
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sfreq = sfreq

        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

    def preprocess_lfp(self, lfp_data: np.ndarray, original_sfreq: float) -> np.ndarray:
        """
        Preprocess raw LFP data.

        Steps:
        1. Bandpass filter 1-300 Hz (LFP range)
        2. Notch filter at 60 Hz (line noise)
        3. Resample to target frequency if needed
        4. Z-score normalization per channel

        Args:
            lfp_data: (n_samples, n_channels) raw LFP
            original_sfreq: Original sampling frequency

        Returns:
            lfp_processed: (n_samples_new, n_channels) processed LFP
        """
        n_samples, n_channels = lfp_data.shape

        # Design bandpass filter (1-300 Hz)
        nyquist = original_sfreq / 2
        low_cut = 1.0 / nyquist
        high_cut = min(300.0, nyquist * 0.95) / nyquist

        b, a = butter(4, [low_cut, high_cut], btype='band')

        # Apply filter to each channel
        lfp_filtered = np.zeros_like(lfp_data)
        for ch in range(n_channels):
            lfp_filtered[:, ch] = filtfilt(b, a, lfp_data[:, ch])

        # Notch filter at 60 Hz and harmonics
        for notch_freq_hz in [60, 120, 180]:
            notch_freq = notch_freq_hz / nyquist
            if notch_freq < 0.95:
                b_notch, a_notch = butter(2, [notch_freq - 0.01, notch_freq + 0.01], btype='bandstop')
                for ch in range(n_channels):
                    lfp_filtered[:, ch] = filtfilt(b_notch, a_notch, lfp_filtered[:, ch])

        # Resample if needed
        if abs(original_sfreq - self.sfreq) > 1e-6:
            n_samples_new = int(n_samples * self.sfreq / original_sfreq)
            lfp_resampled = np.zeros((n_samples_new, n_channels), dtype=np.float32)

            for ch in range(n_channels):
                # Simple linear interpolation
                old_time = np.arange(n_samples) / original_sfreq
                new_time = np.arange(n_samples_new) / self.sfreq
                lfp_resampled[:, ch] = np.interp(new_time, old_time, lfp_filtered[:, ch])

            lfp_filtered = lfp_resampled

        # Z-score normalization per channel
        for ch in range(n_channels):
            mean = lfp_filtered[:, ch].mean()
            std = lfp_filtered[:, ch].std()
            if std > 1e-8:
                lfp_filtered[:, ch] = (lfp_filtered[:, ch] - mean) / std

        return lfp_filtered.astype(np.float32)

    def compute_spectral_features(
        self,
        lfp_data: np.ndarray,
        window_sec: float = 0.5,
        overlap: float = 0.25
    ) -> Dict[str, np.ndarray]:
        """
        Compute time-frequency features from LFP.

        Args:
            lfp_data: (n_samples, n_channels) LFP data
            window_sec: Window size in seconds
            overlap: Overlap fraction (0-1)

        Returns:
            features: Dict with 'power', 'freqs', 'times'
        """
        n_samples, n_channels = lfp_data.shape
        window_samples = int(window_sec * self.sfreq)
        overlap_samples = int(overlap * window_samples)

        # Compute spectrogram for first channel as example
        freqs, times, Sxx = spectrogram(
            lfp_data[:, 0],
            fs=self.sfreq,
            nperseg=window_samples,
            noverlap=overlap_samples
        )

        # Initialize output
        power = np.zeros((len(times), len(freqs), n_channels), dtype=np.float32)

        # Compute for all channels
        for ch in range(n_channels):
            _, _, Sxx_ch = spectrogram(
                lfp_data[:, ch],
                fs=self.sfreq,
                nperseg=window_samples,
                noverlap=overlap_samples
            )
            power[:, :, ch] = 10 * np.log10(Sxx_ch.T + 1e-10)  # Convert to dB

        return {
            'power': power,
            'freqs': freqs,
            'times': times
        }

    def extract_frequency_bands(self, lfp_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract power in standard frequency bands.

        Args:
            lfp_data: (n_samples, n_channels) LFP data

        Returns:
            band_power: Dict with power in each band
        """
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 80),
            'high_gamma': (80, 150)
        }

        n_samples, n_channels = lfp_data.shape
        band_power = {}

        for band_name, (low, high) in bands.items():
            # Bandpass filter
            nyquist = self.sfreq / 2
            low_norm = low / nyquist
            high_norm = min(high, nyquist * 0.95) / nyquist

            b, a = butter(4, [low_norm, high_norm], btype='band')

            filtered = np.zeros_like(lfp_data)
            for ch in range(n_channels):
                filtered[:, ch] = filtfilt(b, a, lfp_data[:, ch])

            # Compute power (Hilbert envelope)
            power = np.abs(filtered) ** 2
            band_power[band_name] = power.astype(np.float32)

        return band_power

    def create_sequences(
        self,
        lfp_data: np.ndarray,
        metadata: Optional[Dict] = None,
        sequence_length: int = 1000,
        stride: int = 500
    ) -> List[Dict]:
        """
        Create sequences from LFP data.

        Args:
            lfp_data: (n_samples, n_channels) LFP data
            metadata: Optional metadata dict
            sequence_length: Length of each sequence (samples)
            stride: Stride between sequences

        Returns:
            sequences: List of sequence dicts
        """
        sequences = []
        n_samples = lfp_data.shape[0]

        for start_idx in range(0, n_samples - sequence_length, stride):
            end_idx = start_idx + sequence_length

            seq_dict = {
                'lfp': lfp_data[start_idx:end_idx].astype(np.float32),
                'metadata': {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': start_idx / self.sfreq,
                    'end_time': end_idx / self.sfreq
                }
            }

            if metadata is not None:
                seq_dict['metadata'].update(metadata)

            sequences.append(seq_dict)

        return sequences

    def process_allen_neuropixels_lfp(
        self,
        lfp_file: str,
        session_id: str
    ) -> int:
        """
        Process Allen Neuropixels LFP data.

        Expected format: .npy file with shape (n_samples, n_channels)

        Args:
            lfp_file: Path to LFP .npy file
            session_id: Session identifier

        Returns:
            n_sequences: Number of sequences created
        """
        print(f"  Loading Allen LFP file: {lfp_file}")

        try:
            lfp_raw = np.load(lfp_file)
        except Exception as e:
            print(f"  Error loading file: {e}")
            return 0

        if lfp_raw.ndim == 1:
            lfp_raw = lfp_raw[:, np.newaxis]

        print(f"  Shape: {lfp_raw.shape}")

        # Allen LFP is typically sampled at 2500 Hz
        original_sfreq = 2500.0

        # Preprocess
        print(f"  Preprocessing LFP ({lfp_raw.shape[1]} channels)...")
        lfp_processed = self.preprocess_lfp(lfp_raw, original_sfreq)

        # Create sequences (1 second windows)
        print(f"  Creating sequences...")
        sequences = self.create_sequences(
            lfp_processed,
            metadata={'session_id': session_id, 'source': 'allen_neuropixels'},
            sequence_length=int(self.sfreq * 1.0),  # 1 second
            stride=int(self.sfreq * 0.5)  # 50% overlap
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
                save_path = self.output_dir / split_name / f"lfp_{session_id}_seq{i:04d}.npz"
                np.savez_compressed(
                    save_path,
                    lfp=seq['lfp'],
                    metadata=seq['metadata']
                )

        print(f"  Created {n_sequences} sequences")
        return n_sequences

    def process_ieeg_bids(
        self,
        ieeg_file: str,
        subject_id: str
    ) -> int:
        """
        Process iEEG data in BIDS format (from DANDI, OpenNeuro).

        Uses MNE to read BIDS-formatted iEEG data.

        Args:
            ieeg_file: Path to iEEG file (e.g., .edf, .vhdr)
            subject_id: Subject identifier

        Returns:
            n_sequences: Number of sequences created
        """
        if not HAS_MNE:
            print("  MNE not installed, skipping")
            return 0

        print(f"  Loading iEEG file: {ieeg_file}")

        try:
            # Read raw file
            raw = mne.io.read_raw(ieeg_file, preload=True, verbose=False)

            # Get data
            ieeg_data = raw.get_data().T  # (n_samples, n_channels)
            original_sfreq = raw.info['sfreq']

            print(f"  Shape: {ieeg_data.shape}, sfreq: {original_sfreq} Hz")

        except Exception as e:
            print(f"  Error loading file: {e}")
            return 0

        # Preprocess
        print(f"  Preprocessing iEEG ({ieeg_data.shape[1]} channels)...")
        ieeg_processed = self.preprocess_lfp(ieeg_data, original_sfreq)

        # Create sequences
        print(f"  Creating sequences...")
        sequences = self.create_sequences(
            ieeg_processed,
            metadata={'subject_id': subject_id, 'source': 'ieeg_bids'},
            sequence_length=int(self.sfreq * 1.0),
            stride=int(self.sfreq * 0.5)
        )

        # Split
        n_sequences = len(sequences)
        n_train = int(0.8 * n_sequences)
        n_val = int(0.1 * n_sequences)

        splits = {
            'train': sequences[:n_train],
            'val': sequences[n_train:n_train+n_val],
            'test': sequences[n_train+n_val:]
        }

        # Save
        for split_name, split_seqs in splits.items():
            for i, seq in enumerate(split_seqs):
                save_path = self.output_dir / split_name / f"ieeg_{subject_id}_seq{i:04d}.npz"
                np.savez_compressed(
                    save_path,
                    lfp=seq['lfp'],
                    metadata=seq['metadata']
                )

        print(f"  Created {n_sequences} sequences")
        return n_sequences


def download_allen_lfp(cache_dir: str = './data/lfp_cache'):
    """
    Download Allen Neuropixels LFP data.

    Returns:
        file_paths: List of LFP file paths
    """
    print("Allen Neuropixels LFP")
    print("Visit: https://allensdk.readthedocs.io/")
    print("\nTo download:")
    print("1. Install AllenSDK: pip install allensdk")
    print("2. Use AllenSDK to download Neuropixels sessions")
    print("3. Extract LFP data using AllenSDK API")
    print(f"4. Save to {cache_dir}/allen_lfp/")

    cache_path = Path(cache_dir) / "allen_lfp"

    if not cache_path.exists():
        print(f"\nDirectory not found: {cache_path}")
        return []

    # Find .npy files
    npy_files = list(cache_path.glob('**/*.npy'))

    if len(npy_files) > 0:
        print(f"\nFound {len(npy_files)} LFP files")
        return [str(f) for f in npy_files]

    return []


def download_dandi_ieeg(dataset_id: str = "000004", cache_dir: str = './data/lfp_cache'):
    """
    Download iEEG data from DANDI archive.

    Dataset 000004: UCLA epilepsy iEEG

    Returns:
        file_paths: List of iEEG file paths
    """
    print(f"DANDI iEEG Dataset {dataset_id}")
    print("Visit: https://dandiarchive.org/")
    print("\nTo download:")
    print("1. Install DANDI CLI: pip install dandi")
    print(f"2. Download dataset: dandi download https://dandiarchive.org/dandiset/{dataset_id}")
    print(f"3. Data will be saved to {cache_dir}/dandi_{dataset_id}/")
    print("\nOr use web interface to manually download.")

    cache_path = Path(cache_dir) / f"dandi_{dataset_id}"

    if not cache_path.exists():
        print(f"\nDirectory not found: {cache_path}")
        return []

    # Find iEEG files (.edf, .nwb, etc.)
    ieeg_files = []
    for ext in ['*.edf', '*.nwb', '*.vhdr']:
        ieeg_files.extend(list(cache_path.glob(f'**/{ext}')))

    if len(ieeg_files) > 0:
        print(f"\nFound {len(ieeg_files)} iEEG files")
        return [str(f) for f in ieeg_files]

    return []


def main():
    parser = argparse.ArgumentParser(description='Download and preprocess LFP/iEEG data')

    parser.add_argument('--output_dir', type=str,
                       default='./data/lfp/processed',
                       help='Output directory for processed data')
    parser.add_argument('--cache_dir', type=str,
                       default='./data/lfp_cache',
                       help='Cache directory for raw downloads')
    parser.add_argument('--sfreq', type=float, default=1000.0,
                       help='Target sampling frequency (Hz)')
    parser.add_argument('--source', type=str, default='allen',
                       choices=['allen', 'dandi', 'local'],
                       help='Data source')
    parser.add_argument('--dandi_dataset', type=str, default='000004',
                       help='DANDI dataset ID')
    parser.add_argument('--local_dir', type=str, default=None,
                       help='Local directory with LFP/iEEG files')

    args = parser.parse_args()

    if not HAS_SCIPY:
        print("Error: scipy is required for LFP processing")
        print("Install with: pip install scipy")
        return

    print("="*80)
    print("LFP/iEEG Data Acquisition for NeuroFMx")
    print("="*80)

    # Initialize processor
    processor = LFPProcessor(args.output_dir, sfreq=args.sfreq)

    # Locate data files
    print(f"\nStep 1: Locating LFP/iEEG data from {args.source}...")

    if args.source == 'allen':
        file_paths = download_allen_lfp(args.cache_dir)
        process_fn = processor.process_allen_neuropixels_lfp
    elif args.source == 'dandi':
        file_paths = download_dandi_ieeg(args.dandi_dataset, args.cache_dir)
        process_fn = processor.process_ieeg_bids
    elif args.source == 'local':
        if args.local_dir is None:
            print("Error: --local_dir required when using source=local")
            return
        local_path = Path(args.local_dir)
        file_paths = []
        for ext in ['*.npy', '*.edf', '*.nwb', '*.vhdr']:
            file_paths.extend(list(local_path.glob(f'**/{ext}')))
        file_paths = [str(f) for f in file_paths]
        # Determine processing function based on file type
        process_fn = None
    else:
        file_paths = []

    if len(file_paths) == 0:
        print("\nNo LFP/iEEG files found.")
        print("\nTo use this script:")
        print("1. Download LFP/iEEG data from Allen or DANDI")
        print("2. Extract to cache directory")
        print("3. Run again with appropriate --source flag")
        return

    print(f"Found {len(file_paths)} files")

    # Process each file
    print(f"\nStep 2: Processing files...")
    total_sequences = 0

    for i, file_path in enumerate(file_paths[:30]):  # Limit to 30 files
        file_name = Path(file_path).stem
        print(f"\nProcessing file {i+1}/{min(len(file_paths), 30)}: {Path(file_path).name}")

        try:
            if args.source == 'allen':
                n_seq = processor.process_allen_neuropixels_lfp(file_path, file_name)
            else:
                n_seq = processor.process_ieeg_bids(file_path, file_name)

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
