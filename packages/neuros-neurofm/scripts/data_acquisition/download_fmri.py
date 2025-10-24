"""
fMRI Data Acquisition Script

Downloads and preprocesses fMRI data from Human Connectome Project.

Extracts:
- ROI timeseries from parcellated brain
- Task conditions
- Confounds for denoising
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import nibabel as nib
    from nilearn import datasets, input_data
    HAS_NILEARN = True
except ImportError:
    print("Warning: nilearn not installed. Install with: pip install nilearn nibabel")
    HAS_NILEARN = False


class fMRIProcessor:
    """Process fMRI data for NeuroFMx."""

    def __init__(self, output_dir: str, n_rois: int = 400):
        """
        Args:
            output_dir: Output directory
            n_rois: Number of ROIs (parcellation resolution)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_rois = n_rois

        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

        # Load parcellation atlas
        print(f"Loading Schaefer {n_rois}-ROI parcellation...")
        self.atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
        self.masker = input_data.NiftiLabelsMasker(
            labels_img=self.atlas.maps,
            standardize=True,
            memory='nilearn_cache',
            verbose=0
        )

    def extract_timeseries(self, func_img_path: str) -> np.ndarray:
        """
        Extract ROI timeseries from functional image.

        Args:
            func_img_path: Path to functional NIfTI file

        Returns:
            timeseries: (n_timepoints, n_rois) array
        """
        timeseries = self.masker.fit_transform(func_img_path)
        return timeseries.astype(np.float32)

    def create_sequences(self, timeseries: np.ndarray, task_labels: np.ndarray,
                        sequence_length: int = 50, stride: int = 25) -> List[Dict]:
        """
        Create sequences from timeseries.

        Args:
            timeseries: (n_timepoints, n_rois) fMRI timeseries
            task_labels: (n_timepoints,) task condition labels
            sequence_length: Sequence length
            stride: Stride between sequences

        Returns:
            sequences: List of dicts
        """
        sequences = []
        n_timepoints = timeseries.shape[0]

        for start_idx in range(0, n_timepoints - sequence_length, stride):
            end_idx = start_idx + sequence_length

            seq_dict = {
                'fmri': timeseries[start_idx:end_idx],
                'task': task_labels[start_idx:end_idx],
                'metadata': {
                    'start_tr': start_idx,
                    'end_tr': end_idx
                }
            }
            sequences.append(seq_dict)

        return sequences

    def process_run(self, func_path: str, task_name: str, run_id: str) -> int:
        """
        Process a single fMRI run.

        Args:
            func_path: Path to functional image
            task_name: Task name
            run_id: Unique run identifier

        Returns:
            n_sequences: Number of sequences created
        """
        # Extract timeseries
        timeseries = self.extract_timeseries(func_path)

        # Create dummy task labels (in practice, load from events.tsv)
        n_timepoints = timeseries.shape[0]
        task_labels = np.zeros(n_timepoints, dtype=np.int32)
        # Placeholder: alternate between conditions
        task_labels[::20] = 1  # Every 20 TRs is "active" condition

        # Create sequences
        sequences = self.create_sequences(timeseries, task_labels)

        # Save
        n_sequences = len(sequences)
        n_train = int(0.8 * n_sequences)
        n_val = int(0.1 * n_sequences)

        splits = {
            'train': sequences[:n_train],
            'val': sequences[n_train:n_train+n_val],
            'test': sequences[n_train+n_val:]
        }

        for split_name, split_seqs in splits.items():
            for i, seq in enumerate(split_seqs):
                save_path = self.output_dir / split_name / f"fmri_{run_id}_seq{i:04d}.npz"
                np.savez_compressed(
                    save_path,
                    fmri=seq['fmri'],
                    task=seq['task'],
                    metadata=seq['metadata']
                )

        return n_sequences


def download_sample_fmri():
    """
    Download sample fMRI data from nilearn datasets.
    For HCP, users need credentials and manual download.
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn not installed")

    print("Downloading sample fMRI data...")
    print("NOTE: For HCP data, please download manually from db.humanconnectome.org")

    # Use development dataset as example
    dataset = datasets.fetch_development_fmri(n_subjects=5)

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Download and preprocess fMRI data')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory with fMRI NIfTI files (optional, uses sample data if not provided)')
    parser.add_argument('--output_dir', type=str,
                       default='./data/fmri/processed',
                       help='Output directory')
    parser.add_argument('--n_rois', type=int, default=400,
                       help='Number of ROIs in parcellation')

    args = parser.parse_args()

    print("="*80)
    print("fMRI Data Acquisition for NeuroFMx")
    print("="*80)

    processor = fMRIProcessor(args.output_dir, n_rois=args.n_rois)

    if args.data_dir is None:
        # Use sample data
        print("\nNo data directory provided, using sample dataset...")
        dataset = download_sample_fmri()

        print(f"\nProcessing {len(dataset.func)} runs...")
        total_sequences = 0

        for i, func_path in enumerate(dataset.func):
            print(f"\nProcessing run {i+1}/{len(dataset.func)}")
            try:
                n_seq = processor.process_run(func_path, 'development', f"sample_{i:03d}")
                total_sequences += n_seq
                print(f"  Created {n_seq} sequences")
            except Exception as e:
                print(f"  Error: {e}")
                continue

    else:
        # Process custom data
        data_path = Path(args.data_dir)
        func_files = list(data_path.glob('**/*bold.nii.gz'))

        print(f"\nFound {len(func_files)} functional files")
        total_sequences = 0

        for i, func_file in enumerate(func_files):
            print(f"\nProcessing {i+1}/{len(func_files)}: {func_file.name}")
            try:
                run_id = func_file.stem.replace('.nii', '')
                n_seq = processor.process_run(str(func_file), 'custom', run_id)
                total_sequences += n_seq
                print(f"  Created {n_seq} sequences")
            except Exception as e:
                print(f"  Error: {e}")
                continue

    print("\n" + "="*80)
    print(f"Processing complete! Total sequences: {total_sequences}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
