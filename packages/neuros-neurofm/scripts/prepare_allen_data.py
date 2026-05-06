#!/usr/bin/env python3
"""
Preprocessing script for Allen Visual Coding data.

Converts trial-aligned 2P calcium data to continuous traces
for use with AllenMultiModalDataset.

Usage:
    python prepare_allen_data.py

This will convert all trial-aligned NPZ files in:
    packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions/

To continuous traces in:
    packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions_continuous/
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def convert_trial_aligned_to_continuous(
    trial_responses: np.ndarray,
    trial_duration_s: float = 0.5,
    frame_rate_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert trial-aligned data to pseudo-continuous traces.

    Args:
        trial_responses: (n_trials, n_cells) trial-averaged responses
        trial_duration_s: Duration of each trial in seconds
        frame_rate_hz: Imaging frame rate

    Returns:
        traces: (n_cells, n_timepoints) continuous traces
        timestamps: (n_timepoints,) timestamps in seconds
    """
    n_trials, n_cells = trial_responses.shape
    frames_per_trial = int(trial_duration_s * frame_rate_hz)

    # Create continuous traces by replicating trial responses
    total_frames = n_trials * frames_per_trial
    traces = np.zeros((n_cells, total_frames), dtype=np.float32)

    for trial_idx in range(n_trials):
        start_frame = trial_idx * frames_per_trial
        end_frame = start_frame + frames_per_trial

        # Replicate trial response across trial duration
        # Add small jitter to avoid perfectly flat traces
        jitter = np.random.randn(n_cells, frames_per_trial) * 0.01
        trial_signal = trial_responses[trial_idx, :, np.newaxis] + jitter
        traces[:, start_frame:end_frame] = trial_signal

    # Create timestamps
    timestamps = np.arange(total_frames) / frame_rate_hz

    return traces, timestamps


def process_session(input_path: Path, output_path: Path,
                    trial_duration_s: float = 0.5,
                    frame_rate_hz: float = 30.0):
    """Process a single Allen 2P session."""

    # Load trial-aligned data
    data = np.load(input_path)

    # Extract trial responses
    # X shape: (n_trials, n_cells)
    trial_responses = data['X']
    cell_ids = data['cell_ids']

    print(f"  Input: {trial_responses.shape[0]} trials, {trial_responses.shape[1]} cells")

    # Convert to continuous
    traces, timestamps = convert_trial_aligned_to_continuous(
        trial_responses=trial_responses,
        trial_duration_s=trial_duration_s,
        frame_rate_hz=frame_rate_hz,
    )

    print(f"  Output: {traces.shape[1]} timepoints ({timestamps[-1]:.1f}s)")

    # Save continuous data
    np.savez_compressed(
        output_path,
        dff_traces=traces,  # (n_cells, n_timepoints) - use dff_traces key
        timestamps=timestamps,
        cell_ids=cell_ids,
        sampling_rate=frame_rate_hz,
        # Include original metadata
        original_shape=trial_responses.shape,
        y_orientation=data.get('y_orientation'),
        y_temporal_freq=data.get('y_temporal_freq'),
        trial_indices=data.get('trial_indices'),
    )

    print(f"  ✓ Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Allen data for multimodal training")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions',
        help='Directory with trial-aligned NPZ files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions_continuous',
        help='Output directory for continuous traces'
    )
    parser.add_argument(
        '--trial_duration',
        type=float,
        default=0.5,
        help='Trial duration in seconds'
    )
    parser.add_argument(
        '--frame_rate',
        type=float,
        default=30.0,
        help='Imaging frame rate (Hz)'
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("ALLEN DATA PREPROCESSING")
    print("="*70)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}\n")

    # Find all 2P session files
    session_files = sorted(input_dir.glob("2p_session_*.npz"))

    if len(session_files) == 0:
        print(f"❌ No session files found in {input_dir}")
        return

    print(f"Found {len(session_files)} sessions\n")

    # Process each session
    for session_file in tqdm(session_files, desc="Processing sessions"):
        # Extract session ID from filename
        # e.g., 2p_session_545446482.npz -> 545446482
        session_id = session_file.stem.replace('2p_session_', '')

        # Output filename (just session ID)
        output_file = output_dir / f"{session_id}.npz"

        print(f"\n📂 Session {session_id}")

        try:
            process_session(
                input_path=session_file,
                output_path=output_file,
                trial_duration_s=args.trial_duration,
                frame_rate_hz=args.frame_rate,
            )
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nContinuous traces saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Verify astro tokens exist in: allen_nwb_results/")
    print(f"2. Update config paths if needed")
    print(f"3. Run training: python scripts/train_allen_multimodal.py")


if __name__ == '__main__':
    main()
