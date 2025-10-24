"""
Human EEG Data Acquisition Script

Downloads and preprocesses EEG data from PhysioNet and other sources.

Data sources:
- PhysioNet EEG Motor Movement/Imagery Dataset
- Temple University EEG Corpus (TUEG)

Extracts:
- Preprocessed EEG channels (10-20 system)
- Event markers and task labels
- Epochs aligned to experimental events
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
    from mne.datasets import eegbci
    HAS_MNE = True
except ImportError:
    print("Warning: MNE not installed. Install with: pip install mne")
    HAS_MNE = False


class EEGProcessor:
    """Process EEG data for NeuroFMx."""

    def __init__(self, output_dir: str, sfreq: float = 128.0):
        """
        Args:
            output_dir: Output directory
            sfreq: Target sampling frequency (Hz)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sfreq = sfreq

        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

    def preprocess_raw(self, raw: object) -> object:
        """
        Preprocess raw EEG data.

        Args:
            raw: MNE Raw object

        Returns:
            raw_preprocessed: Preprocessed Raw object
        """
        # Bandpass filter 0.5-50 Hz
        raw_filt = raw.copy().filter(l_freq=0.5, h_freq=50.0, fir_design='firwin')

        # Resample to target frequency
        if raw_filt.info['sfreq'] != self.sfreq:
            raw_filt.resample(self.sfreq)

        # Re-reference to average
        raw_filt.set_eeg_reference('average', projection=True)
        raw_filt.apply_proj()

        return raw_filt

    def extract_epochs(self, raw: object, event_dict: Dict,
                      tmin: float = -0.5, tmax: float = 2.0) -> object:
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
            reject=dict(eeg=100e-6),  # Reject trials with >100uV
            verbose=False
        )

        return epochs

    def epochs_to_sequences(self, epochs: object) -> List[Dict]:
        """
        Convert MNE Epochs to sequence format.

        Args:
            epochs: MNE Epochs object

        Returns:
            sequences: List of sequence dicts
        """
        sequences = []

        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        events = epochs.events[:, -1]  # Event codes

        for i in range(len(data)):
            seq_dict = {
                'eeg': data[i].T.astype(np.float32),  # (n_times, n_channels)
                'event_code': events[i],
                'metadata': {
                    'epoch_id': i,
                    'channels': epochs.ch_names
                }
            }
            sequences.append(seq_dict)

        return sequences

    def process_subject(self, subject_id: int, runs: List[int],
                       task_dict: Dict) -> int:
        """
        Process a single subject.

        Args:
            subject_id: Subject ID
            runs: List of run numbers
            task_dict: Task event dictionary

        Returns:
            n_sequences: Number of sequences created
        """
        print(f"  Loading runs for subject {subject_id}...")

        all_sequences = []

        for run in runs:
            try:
                # Load raw data
                raw_fname = eegbci.load_data(subject_id, runs=[run])[0]
                raw = mne.io.read_raw_edf(raw_fname, preload=True, verbose=False)

                # Preprocess
                raw_prep = self.preprocess_raw(raw)

                # Extract epochs
                epochs = self.extract_epochs(raw_prep, task_dict)

                # Convert to sequences
                sequences = self.epochs_to_sequences(epochs)
                all_sequences.extend(sequences)

            except Exception as e:
                print(f"    Error processing run {run}: {e}")
                continue

        # Split and save
        n_sequences = len(all_sequences)
        n_train = int(0.8 * n_sequences)
        n_val = int(0.1 * n_sequences)

        splits = {
            'train': all_sequences[:n_train],
            'val': all_sequences[n_train:n_train+n_val],
            'test': all_sequences[n_train+n_val:]
        }

        for split_name, split_seqs in splits.items():
            for i, seq in enumerate(split_seqs):
                save_path = self.output_dir / split_name / f"eeg_subj{subject_id:03d}_seq{i:04d}.npz"
                np.savez_compressed(
                    save_path,
                    eeg=seq['eeg'],
                    event_code=seq['event_code'],
                    metadata=seq['metadata']
                )

        print(f"  Created {n_sequences} sequences")
        return n_sequences


def download_physionet_eeg(n_subjects: int = 50):
    """
    Download PhysioNet EEG Motor Movement/Imagery dataset.

    Args:
        n_subjects: Number of subjects (max 109)

    Returns:
        subject_ids: List of subject IDs
    """
    if not HAS_MNE:
        raise ImportError("MNE not installed")

    print(f"Downloading PhysioNet EEG data for {n_subjects} subjects...")
    print("This uses the EEG Motor Movement/Imagery Dataset")

    subject_ids = list(range(1, min(n_subjects + 1, 110)))

    # Download will happen automatically when processing
    return subject_ids


def main():
    parser = argparse.ArgumentParser(description='Download and preprocess EEG data')
    parser.add_argument('--n_subjects', type=int, default=20,
                       help='Number of subjects to process')
    parser.add_argument('--output_dir', type=str,
                       default='./data/eeg/processed',
                       help='Output directory')
    parser.add_argument('--sfreq', type=float, default=128.0,
                       help='Target sampling frequency (Hz)')
    parser.add_argument('--runs', nargs='+', type=int,
                       default=[3, 7, 11],  # Motor imagery tasks
                       help='Run numbers to process')

    args = parser.parse_args()

    print("="*80)
    print("EEG Data Acquisition for NeuroFMx")
    print("="*80)

    # Event dictionary for motor imagery
    task_dict = {'T0': 1, 'T1': 2, 'T2': 3}  # Rest, left hand, right hand

    processor = EEGProcessor(args.output_dir, sfreq=args.sfreq)

    # Get subject list
    subject_ids = download_physionet_eeg(args.n_subjects)

    print(f"\nProcessing {len(subject_ids)} subjects...")
    total_sequences = 0

    for subj_id in tqdm(subject_ids, desc="Processing subjects"):
        try:
            n_seq = processor.process_subject(subj_id, args.runs, task_dict)
            total_sequences += n_seq
        except Exception as e:
            print(f"Error processing subject {subj_id}: {e}")
            continue

    print("\n" + "="*80)
    print(f"Processing complete! Total sequences: {total_sequences}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
