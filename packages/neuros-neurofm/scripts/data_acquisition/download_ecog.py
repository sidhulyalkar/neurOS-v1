"""
ECoG (Electrocorticography) Data Acquisition Script

Downloads and preprocesses intracranial EEG data for NeuroFMx training.

Data sources:
- Miller Lab ECoG (hand movement, speech)
- OpenNeuro datasets (ds003688, ds003490)
- DANDI archive ECoG datasets
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
    import mne
    HAS_MNE = True
except ImportError:
    print("Warning: MNE not installed. Install with: pip install mne")
    HAS_MNE = False

try:
    from dandi.download import download as dandi_download
    HAS_DANDI = True
except ImportError:
    print("Warning: DANDI client not installed. Install with: pip install dandi")
    HAS_DANDI = False


class ECoGProcessor:
    """Process ECoG data for NeuroFMx."""

    def __init__(self, output_dir: str, sfreq: float = 500.0):
        """
        Args:
            output_dir: Output directory
            sfreq: Target sampling frequency (Hz) - ECoG typically 500-2000 Hz
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sfreq = sfreq

        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

    def preprocess_raw(self, raw: object) -> object:
        """
        Preprocess raw ECoG data.

        Args:
            raw: MNE Raw object

        Returns:
            raw_preprocessed: Preprocessed Raw object
        """
        # Bandpass filter 1-200 Hz (ECoG has higher frequency content than scalp EEG)
        raw_filt = raw.copy().filter(l_freq=1.0, h_freq=200.0, fir_design='firwin')

        # Notch filter at 60 Hz (line noise)
        raw_filt.notch_filter([60, 120], fir_design='firwin')

        # Resample to target frequency
        if raw_filt.info['sfreq'] != self.sfreq:
            raw_filt.resample(self.sfreq)

        # Re-reference to average (or use bipolar referencing)
        raw_filt.set_eeg_reference('average', projection=True)
        raw_filt.apply_proj()

        return raw_filt

    def extract_epochs(
        self,
        raw: object,
        event_dict: Dict,
        tmin: float = -0.5,
        tmax: float = 2.0
    ) -> object:
        """
        Extract epochs around events.

        Args:
            raw: Preprocessed Raw object
            event_dict: Event dictionary
            tmin: Start time before event
            tmax: End time after event

        Returns:
            epochs: MNE Epochs object
        """
        events, event_id = mne.events_from_annotations(raw, event_id=event_dict)

        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=tmin, tmax=tmax,
            baseline=(tmin, 0),
            preload=True,
            reject=dict(eeg=200e-6),  # Reject trials with >200uV (less strict than scalp)
            verbose=False
        )

        return epochs

    def create_sequences(
        self,
        ecog_data: np.ndarray,
        behavior: np.ndarray,
        events: np.ndarray,
        sequence_length: int = 100,
        stride: int = 50
    ) -> List[Dict]:
        """
        Create sequences from continuous ECoG data.

        Args:
            ecog_data: (n_samples, n_channels) ECoG data
            behavior: (n_samples, n_behavior) behavioral data
            events: (n_samples,) event codes
            sequence_length: Length of each sequence
            stride: Stride between sequences

        Returns:
            sequences: List of sequence dicts
        """
        sequences = []
        n_samples = ecog_data.shape[0]

        for start_idx in range(0, n_samples - sequence_length, stride):
            end_idx = start_idx + sequence_length

            seq_dict = {
                'ecog': ecog_data[start_idx:end_idx].astype(np.float32),
                'behavior': behavior[start_idx:end_idx].astype(np.float32),
                'events': events[start_idx:end_idx].astype(np.int32),
                'metadata': {
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
            }
            sequences.append(seq_dict)

        return sequences

    def process_subject(
        self,
        raw_file: str,
        subject_id: str,
        event_dict: Optional[Dict] = None,
        behavior_data: Optional[np.ndarray] = None
    ) -> int:
        """
        Process a single subject's ECoG data.

        Args:
            raw_file: Path to raw ECoG file
            subject_id: Subject identifier
            event_dict: Event dictionary (if epoching)
            behavior_data: Behavioral data array

        Returns:
            n_sequences: Number of sequences created
        """
        print(f"  Loading raw file: {raw_file}")

        # Load raw data
        if raw_file.endswith('.fif'):
            raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
        elif raw_file.endswith('.edf'):
            raw = mne.io.read_raw_edf(raw_file, preload=True, verbose=False)
        else:
            print(f"  Unsupported file format: {raw_file}")
            return 0

        # Preprocess
        print(f"  Preprocessing...")
        raw_prep = self.preprocess_raw(raw)

        # Get data
        ecog_data = raw_prep.get_data().T  # (n_samples, n_channels)

        # Get or create behavior
        if behavior_data is None:
            # Create dummy behavior (zeros)
            behavior_data = np.zeros((ecog_data.shape[0], 2), dtype=np.float32)

        # Get events if available
        if event_dict is not None:
            try:
                epochs = self.extract_epochs(raw_prep, event_dict)
                # Use epochs instead of continuous data
                ecog_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
                n_epochs, n_channels, n_times = ecog_data.shape

                # Reshape to (n_epochs*n_times, n_channels)
                ecog_data = ecog_data.transpose(0, 2, 1).reshape(-1, n_channels)

                # Match behavior
                behavior_data = np.tile(behavior_data[:n_times], (n_epochs, 1))

                event_codes = np.repeat(epochs.events[:, -1], n_times)
            except:
                print(f"  Could not extract epochs, using continuous data")
                event_codes = np.zeros(ecog_data.shape[0], dtype=np.int32)
        else:
            event_codes = np.zeros(ecog_data.shape[0], dtype=np.int32)

        # Create sequences
        print(f"  Creating sequences...")
        sequences = self.create_sequences(
            ecog_data,
            behavior_data,
            event_codes,
            sequence_length=int(self.sfreq)  # 1 second sequences
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
                save_path = self.output_dir / split_name / f"ecog_{subject_id}_seq{i:04d}.npz"
                np.savez_compressed(
                    save_path,
                    ecog=seq['ecog'],
                    behavior=seq['behavior'],
                    events=seq['events'],
                    metadata=seq['metadata']
                )

        print(f"  Created {n_sequences} sequences")
        return n_sequences


def download_miller_lab_data(output_dir: str = './data/ecog_cache'):
    """
    Download Miller Lab ECoG data.

    Note: This typically requires manual download from the Miller Lab website
    or contacting the lab for access.

    Returns:
        file_paths: List of downloaded file paths
    """
    print("Miller Lab ECoG data typically requires manual download.")
    print("Visit: http://crunch.cs.washington.edu/research/brain-computer-interfaces")
    print("or contact the Miller Lab for access.")

    # Check if data already exists locally
    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(cache_dir.glob('**/*.fif')) + list(cache_dir.glob('**/*.edf'))

    if len(existing_files) > 0:
        print(f"Found {len(existing_files)} existing ECoG files")
        return [str(f) for f in existing_files]

    return []


def download_openneuro_ecog(output_dir: str = './data/ecog_cache'):
    """
    Download ECoG data from OpenNeuro.

    Available datasets:
    - ds003688: ECoG during spoken word production
    - ds003490: ECoG during finger movements
    """
    print("OpenNeuro ECoG datasets available:")
    print("  ds003688: Spoken word production")
    print("  ds003490: Finger movements")
    print("\nTo download, use AWS CLI or OpenNeuro CLI:")
    print("  aws s3 sync --no-sign-request s3://openneuro.org/ds003688 ./data/ecog_cache/ds003688")

    cache_dir = Path(output_dir)
    existing_files = list(cache_dir.glob('**/*.fif')) + list(cache_dir.glob('**/*.edf'))

    return [str(f) for f in existing_files]


def download_dandi_ecog(output_dir: str = './data/ecog_cache'):
    """
    Download ECoG data from DANDI archive.

    Note: Requires DANDI CLI installation.
    """
    if not HAS_DANDI:
        print("DANDI client not installed. Install with: pip install dandi")
        return []

    print("Searching for ECoG datasets on DANDI...")

    # Example: DANDI dandiset with ECoG data
    # dandisets = ['000055', '000056']  # Example IDs

    print("Note: Please search DANDI archive for specific ECoG datasets")
    print("Visit: https://dandiarchive.org/")

    cache_dir = Path(output_dir)
    existing_files = list(cache_dir.glob('**/*.nwb'))

    return [str(f) for f in existing_files]


def main():
    parser = argparse.ArgumentParser(description='Download and preprocess ECoG data')

    parser.add_argument('--output_dir', type=str,
                       default='./data/ecog/processed',
                       help='Output directory for processed data')
    parser.add_argument('--cache_dir', type=str,
                       default='./data/ecog_cache',
                       help='Cache directory for raw downloads')
    parser.add_argument('--sfreq', type=float, default=500.0,
                       help='Target sampling frequency (Hz)')
    parser.add_argument('--source', type=str, default='miller',
                       choices=['miller', 'openneuro', 'dandi', 'local'],
                       help='Data source')
    parser.add_argument('--local_dir', type=str, default=None,
                       help='Local directory with ECoG files (if source=local)')

    args = parser.parse_args()

    print("="*80)
    print("ECoG Data Acquisition for NeuroFMx")
    print("="*80)

    # Initialize processor
    processor = ECoGProcessor(args.output_dir, sfreq=args.sfreq)

    # Download/locate data files
    print(f"\nStep 1: Locating ECoG data from {args.source}...")

    if args.source == 'miller':
        file_paths = download_miller_lab_data(args.cache_dir)
    elif args.source == 'openneuro':
        file_paths = download_openneuro_ecog(args.cache_dir)
    elif args.source == 'dandi':
        file_paths = download_dandi_ecog(args.cache_dir)
    elif args.source == 'local':
        if args.local_dir is None:
            print("Error: --local_dir required when using source=local")
            return
        local_path = Path(args.local_dir)
        file_paths = list(local_path.glob('**/*.fif')) + list(local_path.glob('**/*.edf'))
        file_paths = [str(f) for f in file_paths]
    else:
        file_paths = []

    if len(file_paths) == 0:
        print("\nNo ECoG files found.")
        print("\nTo use this script:")
        print("1. Download ECoG data manually from one of the sources")
        print("2. Place files in the cache directory")
        print("3. Run again with --source=local --local_dir=<your_directory>")
        return

    print(f"Found {len(file_paths)} ECoG files")

    # Process each file
    print(f"\nStep 2: Processing files...")
    total_sequences = 0

    for i, file_path in enumerate(file_paths[:20]):  # Limit to 20 files
        subject_id = f"subj{i:03d}"
        print(f"\nProcessing file {i+1}/{min(len(file_paths), 20)}: {Path(file_path).name}")

        try:
            n_seq = processor.process_subject(file_path, subject_id)
            total_sequences += n_seq
        except Exception as e:
            print(f"  Error processing file: {e}")
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
