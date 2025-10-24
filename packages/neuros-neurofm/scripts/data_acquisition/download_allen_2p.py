"""
Allen Brain Observatory 2-Photon Calcium Imaging Data Acquisition

Downloads and preprocesses calcium imaging data from Allen Institute.

Extracts:
- dF/F traces for all ROIs
- Stimulus information (natural images, gratings, etc.)
- Running speed and other behavioral variables
- Timestamps and metadata
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
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    HAS_ALLENSDK = True
except ImportError:
    print("Warning: AllenSDK not installed. Install with: pip install allensdk")
    HAS_ALLENSDK = False


class Allen2PProcessor:
    """Process Allen 2-Photon calcium imaging data."""

    def __init__(self, output_dir: str, target_fps: float = 10.0):
        """
        Args:
            output_dir: Directory to save processed data
            target_fps: Target frame rate for downsampling (Hz)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_fps = target_fps

        # Create subdirectories
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

    def downsample_traces(self, traces: np.ndarray, original_fps: float) -> np.ndarray:
        """
        Downsample calcium traces to target FPS.

        Args:
            traces: (n_frames, n_cells) calcium traces
            original_fps: Original sampling rate

        Returns:
            downsampled: (n_frames_new, n_cells) downsampled traces
        """
        if original_fps <= self.target_fps:
            return traces

        downsample_factor = int(original_fps / self.target_fps)

        # Average over bins
        n_frames, n_cells = traces.shape
        n_frames_new = n_frames // downsample_factor

        downsampled = np.zeros((n_frames_new, n_cells), dtype=np.float32)

        for i in range(n_frames_new):
            start_idx = i * downsample_factor
            end_idx = start_idx + downsample_factor
            downsampled[i] = traces[start_idx:end_idx].mean(axis=0)

        return downsampled

    def extract_stimulus_info(self, experiment: object) -> Dict:
        """
        Extract stimulus information from experiment.

        Args:
            experiment: Allen experiment object

        Returns:
            stimulus_info: Dict with stimulus data
        """
        stimulus_info = {
            'stimulus_type': [],
            'stimulus_id': [],
            'stimulus_times': []
        }

        try:
            # Get stimulus table
            stim_table = experiment.get_stimulus_table('natural_scenes')
            if stim_table is not None:
                stimulus_info['stimulus_type'].extend(['natural_scenes'] * len(stim_table))
                stimulus_info['stimulus_id'].extend(stim_table['frame'].values)
                stimulus_info['stimulus_times'].extend(stim_table['start'].values)
        except:
            pass

        try:
            # Try gratings
            stim_table = experiment.get_stimulus_table('static_gratings')
            if stim_table is not None:
                stimulus_info['stimulus_type'].extend(['static_gratings'] * len(stim_table))
                stimulus_info['stimulus_id'].extend(stim_table['orientation'].values)
                stimulus_info['stimulus_times'].extend(stim_table['start'].values)
        except:
            pass

        return stimulus_info

    def align_behavior(self, running_speed: np.ndarray, running_times: np.ndarray,
                      trace_times: np.ndarray) -> np.ndarray:
        """
        Align running speed to calcium trace timestamps.

        Args:
            running_speed: Running speed values
            running_times: Timestamps for running speed
            trace_times: Timestamps for calcium traces

        Returns:
            aligned_speed: Running speed aligned to traces
        """
        if len(running_speed) == 0 or len(running_times) == 0:
            return np.zeros(len(trace_times), dtype=np.float32)

        # Interpolate running speed to trace times
        aligned_speed = np.interp(trace_times, running_times, running_speed)
        return aligned_speed.astype(np.float32)

    def create_sequences(self, calcium: np.ndarray, behavior: np.ndarray,
                        stimulus_ids: np.ndarray,
                        sequence_length: int = 100, stride: int = 50) -> List[Dict]:
        """
        Create fixed-length sequences from continuous calcium data.

        Args:
            calcium: (n_frames, n_cells) calcium traces
            behavior: (n_frames, n_behavior_dims) behavioral data
            stimulus_ids: (n_frames,) stimulus identifiers
            sequence_length: Length of each sequence
            stride: Stride between sequences

        Returns:
            sequences: List of dicts
        """
        sequences = []
        n_frames = calcium.shape[0]

        for start_idx in range(0, n_frames - sequence_length, stride):
            end_idx = start_idx + sequence_length

            seq_dict = {
                'calcium': calcium[start_idx:end_idx].astype(np.float32),
                'behavior': behavior[start_idx:end_idx].astype(np.float32),
                'stimulus': stimulus_ids[start_idx:end_idx].astype(np.int32),
                'metadata': {
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                }
            }
            sequences.append(seq_dict)

        return sequences

    def process_experiment(self, experiment: object, experiment_id: int,
                          sequence_length: int = 100) -> int:
        """
        Process a single experiment.

        Args:
            experiment: Allen experiment object
            experiment_id: Experiment ID
            sequence_length: Length of sequences

        Returns:
            n_sequences: Number of sequences created
        """
        print(f"  Loading data for experiment {experiment_id}...")

        # Get dF/F traces
        try:
            dff_traces = experiment.get_dff_traces()[1]  # [1] is dF/F, [0] is timestamps
            timestamps = experiment.get_dff_traces()[0]
        except:
            print(f"  Error loading dF/F traces for experiment {experiment_id}")
            return 0

        n_frames_original, n_cells = dff_traces.shape
        print(f"  Loaded {n_cells} cells, {n_frames_original} frames")

        # Get metadata
        metadata = experiment.get_metadata()
        original_fps = metadata.get('imaging_plane_group', {}).get('imaging_rate', 30.0)

        # Downsample traces
        print(f"  Downsampling from {original_fps:.1f} Hz to {self.target_fps:.1f} Hz...")
        calcium_downsampled = self.downsample_traces(dff_traces.T, original_fps)  # Transpose to (frames, cells)

        # Downsample timestamps
        downsample_factor = int(original_fps / self.target_fps)
        timestamps_downsampled = timestamps[::downsample_factor][:calcium_downsampled.shape[0]]

        # Get running speed
        try:
            running_speed, running_times = experiment.get_running_speed()
            aligned_speed = self.align_behavior(running_speed, running_times,
                                              timestamps_downsampled)
        except:
            print(f"  Warning: Could not load running speed")
            aligned_speed = np.zeros(len(timestamps_downsampled), dtype=np.float32)

        # Create behavior array
        behavior = np.column_stack([aligned_speed, np.zeros_like(aligned_speed)])  # [speed, placeholder]

        # Get stimulus information
        stimulus_info = self.extract_stimulus_info(experiment)

        # Create stimulus ID array aligned to frames
        stimulus_ids = np.zeros(len(timestamps_downsampled), dtype=np.int32)
        if len(stimulus_info['stimulus_times']) > 0:
            for stim_time, stim_id in zip(stimulus_info['stimulus_times'],
                                         stimulus_info['stimulus_id']):
                # Find closest frame
                idx = np.argmin(np.abs(timestamps_downsampled - stim_time))
                if idx < len(stimulus_ids):
                    stimulus_ids[idx] = int(stim_id) if not np.isnan(stim_id) else 0

        # Create sequences
        print(f"  Creating sequences (length={sequence_length})...")
        sequences = self.create_sequences(calcium_downsampled, behavior,
                                         stimulus_ids,
                                         sequence_length=sequence_length)

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
        for split_name, split_sequences in splits.items():
            for i, seq in enumerate(split_sequences):
                save_path = self.output_dir / split_name / f"allen2p_exp{experiment_id}_seq{i:04d}.npz"
                np.savez_compressed(
                    save_path,
                    calcium=seq['calcium'],
                    behavior=seq['behavior'],
                    stimulus=seq['stimulus'],
                    metadata=seq['metadata']
                )

        print(f"  Created {n_sequences} sequences: {n_train} train, {n_val} val, {len(splits['test'])} test")
        return n_sequences


def download_allen_2p_data(n_experiments: int = 15,
                          visual_areas: List[str] = ['VISp', 'VISl', 'VISal'],
                          cache_dir: str = './data/allen_2p_cache'):
    """
    Download Allen 2-Photon experiments.

    Args:
        n_experiments: Number of experiments to download
        visual_areas: List of visual areas to include
        cache_dir: Cache directory

    Returns:
        experiments: List of experiment objects
    """
    if not HAS_ALLENSDK:
        raise ImportError("AllenSDK not installed. Install with: pip install allensdk")

    print(f"Initializing Brain Observatory Cache...")
    boc = BrainObservatoryCache(manifest_file=os.path.join(cache_dir, 'manifest.json'))

    # Get all experiment containers
    print(f"Querying for experiments in visual areas: {visual_areas}...")

    # Get experiments with specific stimuli and areas
    experiments_list = []

    for area in visual_areas:
        exp_containers = boc.get_experiment_containers(
            targeted_structures=[area],
            include_failed=False
        )

        print(f"  Found {len(exp_containers)} containers in {area}")

        # Get experiments from containers
        for container in exp_containers[:n_experiments // len(visual_areas)]:
            container_exps = boc.get_ophys_experiments(
                experiment_container_ids=[container['id']],
                stimuli=['natural_scenes']  # Filter for specific stimulus
            )

            if len(container_exps) > 0:
                experiments_list.append(container_exps[0])  # Take first session

            if len(experiments_list) >= n_experiments:
                break

        if len(experiments_list) >= n_experiments:
            break

    print(f"Selected {len(experiments_list)} experiments to download")

    # Download experiments
    experiments = []
    for exp_info in tqdm(experiments_list, desc="Loading experiments"):
        try:
            exp = boc.get_ophys_experiment_data(exp_info['id'])
            experiments.append((exp, exp_info['id']))
        except Exception as e:
            print(f"Error loading experiment {exp_info['id']}: {e}")
            continue

    print(f"Successfully loaded {len(experiments)} experiments")
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description='Download and preprocess Allen 2-Photon calcium imaging data'
    )
    parser.add_argument('--n_experiments', type=int, default=15,
                       help='Number of experiments to download')
    parser.add_argument('--output_dir', type=str,
                       default='./data/allen_2p/processed',
                       help='Output directory for processed data')
    parser.add_argument('--cache_dir', type=str,
                       default='./data/allen_2p_cache',
                       help='Cache directory for raw downloads')
    parser.add_argument('--target_fps', type=float, default=10.0,
                       help='Target sampling rate in Hz')
    parser.add_argument('--sequence_length', type=int, default=100,
                       help='Length of sequences (in frames)')
    parser.add_argument('--visual_areas', nargs='+',
                       default=['VISp', 'VISl', 'VISal'],
                       help='Visual areas to include')

    args = parser.parse_args()

    print("="*80)
    print("Allen Brain Observatory 2-Photon Data Acquisition for NeuroFMx")
    print("="*80)

    # Initialize processor
    processor = Allen2PProcessor(args.output_dir, target_fps=args.target_fps)

    # Download experiments
    print(f"\nStep 1: Downloading {args.n_experiments} experiments...")
    experiments = download_allen_2p_data(
        n_experiments=args.n_experiments,
        visual_areas=args.visual_areas,
        cache_dir=args.cache_dir
    )

    if len(experiments) == 0:
        print("No experiments downloaded. Exiting.")
        return

    # Process each experiment
    print(f"\nStep 2: Processing experiments...")
    total_sequences = 0

    for i, (experiment, exp_id) in enumerate(experiments):
        print(f"\nProcessing experiment {i+1}/{len(experiments)}: {exp_id}")

        try:
            n_seq = processor.process_experiment(
                experiment,
                exp_id,
                sequence_length=args.sequence_length
            )
            total_sequences += n_seq
        except Exception as e:
            print(f"Error processing experiment {exp_id}: {e}")
            import traceback
            traceback.print_exc()
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
