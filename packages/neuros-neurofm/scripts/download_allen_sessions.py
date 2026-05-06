#!/usr/bin/env python3
"""
Allen Brain Observatory Session Downloader

Downloads additional 2-photon calcium imaging sessions from the Allen Brain Observatory.

Disk space estimates:
- Raw NWB files: ~1-5 GB each (not needed after processing)
- Processed NPZ: ~2-5 MB each
- For 50 sessions: ~250 MB processed data

Usage:
    # Download 10 more sessions
    python scripts/download_allen_sessions.py --n-sessions 10

    # Download specific visual areas
    python scripts/download_allen_sessions.py --area VISp --n-sessions 20

    # List available sessions
    python scripts/download_allen_sessions.py --list

    # Download and process in one step
    python scripts/download_allen_sessions.py --n-sessions 10 --process
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

# Check for allensdk
try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    ALLENSDK_AVAILABLE = True
except ImportError:
    ALLENSDK_AVAILABLE = False
    print("⚠️  allensdk not installed. Install with: pip install allensdk")


def list_available_sessions(
    area: Optional[str] = None,
    stimuli: Optional[List[str]] = None,
    max_display: int = 50,
) -> List[Dict]:
    """
    List available sessions from Allen Brain Observatory.

    Args:
        area: Visual area filter (VISp, VISl, VISal, etc.)
        stimuli: Stimulus types filter
        max_display: Max sessions to display

    Returns:
        List of experiment metadata
    """
    if not ALLENSDK_AVAILABLE:
        print("ERROR: allensdk required. Install with: pip install allensdk")
        return []

    # Initialize cache
    cache_dir = Path.home() / '.allen_cache'
    boc = BrainObservatoryCache(manifest_file=str(cache_dir / 'manifest.json'))

    # Get experiments
    kwargs = {}
    if area:
        kwargs['targeted_structures'] = [area]
    if stimuli:
        kwargs['stimuli'] = stimuli

    experiments = boc.get_ophys_experiments(**kwargs)

    print(f"\nFound {len(experiments)} sessions")
    print(f"{'='*70}")

    if area:
        print(f"Area filter: {area}")
    if stimuli:
        print(f"Stimuli filter: {stimuli}")

    print(f"\n{'Session ID':<15} {'Area':<8} {'Depth':<8} {'Cre Line':<25} {'Stimuli'}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*25} {'-'*30}")

    for exp in experiments[:max_display]:
        session_id = exp['id']
        area = exp.get('targeted_structure', 'N/A')
        depth = exp.get('imaging_depth', 'N/A')
        cre = exp.get('cre_line', 'N/A')[:24]
        stim = ', '.join(exp.get('session_type', ['N/A'])[:2])

        print(f"{session_id:<15} {area:<8} {depth:<8} {cre:<25} {stim}")

    if len(experiments) > max_display:
        print(f"\n... and {len(experiments) - max_display} more sessions")

    return experiments


def download_sessions(
    session_ids: Optional[List[int]] = None,
    n_sessions: int = 10,
    area: str = 'VISp',
    exclude_existing: bool = True,
    output_dir: Optional[Path] = None,
) -> List[int]:
    """
    Download Allen Brain Observatory sessions.

    Args:
        session_ids: Specific sessions to download
        n_sessions: Number of new sessions to download
        area: Visual area to filter by
        exclude_existing: Skip already downloaded sessions
        output_dir: Where to save processed data

    Returns:
        List of downloaded session IDs
    """
    if not ALLENSDK_AVAILABLE:
        print("ERROR: allensdk required. Install with: pip install allensdk")
        return []

    # Setup paths
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / 'packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find existing sessions
    existing_sessions = set()
    if exclude_existing:
        for f in output_dir.glob('2p_session_*.npz'):
            session_id = int(f.stem.replace('2p_session_', ''))
            existing_sessions.add(session_id)
        print(f"Found {len(existing_sessions)} existing sessions")

    # Initialize cache
    cache_dir = Path.home() / '.allen_cache'
    boc = BrainObservatoryCache(manifest_file=str(cache_dir / 'manifest.json'))

    # Get candidate sessions
    if session_ids is None:
        experiments = boc.get_ophys_experiments(
            targeted_structures=[area],
            stimuli=['drifting_gratings'],  # Common stimulus
        )

        # Filter out existing
        candidates = [e for e in experiments if e['id'] not in existing_sessions]

        # Sort by experiment ID (newer sessions often have better data)
        candidates.sort(key=lambda x: x['id'], reverse=True)

        # Take requested number
        session_ids = [e['id'] for e in candidates[:n_sessions]]

    print(f"\nDownloading {len(session_ids)} sessions...")
    print(f"Output directory: {output_dir}")

    downloaded = []

    for session_id in tqdm(session_ids, desc="Downloading"):
        try:
            # Get experiment container
            experiment = boc.get_ophys_experiment_data(session_id)

            # Extract dF/F traces
            timestamps, dff = experiment.get_dff_traces()

            # Get cell specimen IDs
            cell_ids = experiment.get_cell_specimen_ids()

            # Get stimulus epochs (for drifting gratings)
            stim_table = experiment.get_stimulus_table('drifting_gratings')

            # Compute trial-averaged responses
            # Group by orientation
            orientations = stim_table['orientation'].unique()
            orientations = orientations[~np.isnan(orientations)]

            trial_responses = []
            trial_orientations = []
            trial_temporal_freqs = []

            for _, row in stim_table.iterrows():
                # Convert to int for array slicing
                start_idx = int(row['start'])
                end_idx = int(row['end'])

                # Skip invalid indices
                if start_idx >= end_idx or start_idx < 0 or end_idx > dff.shape[1]:
                    continue

                # Extract response (mean during stimulus)
                response = dff[:, start_idx:end_idx].mean(axis=1)
                trial_responses.append(response)
                trial_orientations.append(row['orientation'])
                trial_temporal_freqs.append(row.get('temporal_frequency', 2.0))

            # Check we got valid trials
            if len(trial_responses) == 0:
                print(f"  ✗ Session {session_id}: No valid trials found")
                continue

            trial_responses = np.array(trial_responses)  # (n_trials, n_cells)
            trial_orientations = np.array(trial_orientations)
            trial_temporal_freqs = np.array(trial_temporal_freqs)

            # Save
            output_file = output_dir / f'2p_session_{session_id}.npz'
            np.savez_compressed(
                output_file,
                X=trial_responses,
                y_orientation=trial_orientations,
                y_temporal_freq=trial_temporal_freqs,
                cell_ids=cell_ids,
                trial_indices=np.arange(len(trial_responses)),
                timestamps=timestamps,
                dff_traces=dff,
                session_id=session_id,
            )

            downloaded.append(session_id)
            print(f"  ✓ Session {session_id}: {trial_responses.shape[0]} trials, {trial_responses.shape[1]} cells")

        except Exception as e:
            print(f"  ✗ Session {session_id}: {e}")
            continue

    print(f"\n✓ Downloaded {len(downloaded)} sessions")
    return downloaded


def process_downloaded_sessions(
    input_dir: Path,
    output_dir: Path,
    trial_duration_s: float = 0.5,
    frame_rate_hz: float = 30.0,
):
    """
    Convert downloaded trial-aligned data to continuous traces.

    This is the same as prepare_allen_data.py but for new downloads.
    """
    from prepare_allen_data import convert_trial_aligned_to_continuous

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for npz_file in tqdm(list(input_dir.glob('2p_session_*.npz')), desc="Processing"):
        session_id = npz_file.stem.replace('2p_session_', '')
        output_file = output_dir / f'{session_id}.npz'

        if output_file.exists():
            continue

        data = np.load(npz_file)
        trial_responses = data['X']

        traces, timestamps = convert_trial_aligned_to_continuous(
            trial_responses,
            trial_duration_s=trial_duration_s,
            frame_rate_hz=frame_rate_hz,
        )

        np.savez_compressed(
            output_file,
            dff_traces=traces,
            timestamps=timestamps,
            cell_ids=data['cell_ids'],
            sampling_rate=frame_rate_hz,
        )

    print(f"✓ Processed sessions saved to {output_dir}")


def estimate_disk_space(n_sessions: int) -> Dict[str, str]:
    """
    Estimate disk space required for downloading sessions.

    Returns:
        Dict with space estimates
    """
    # Estimates based on typical Allen data
    raw_nwb_per_session_gb = 2.5  # ~2.5 GB per raw NWB
    processed_npz_per_session_mb = 3.0  # ~3 MB per processed NPZ
    cache_overhead_gb = 1.0  # ~1 GB for manifest and metadata

    estimates = {
        'raw_nwb_total': f"{n_sessions * raw_nwb_per_session_gb:.1f} GB (temporary)",
        'processed_npz_total': f"{n_sessions * processed_npz_per_session_mb:.1f} MB (permanent)",
        'cache_overhead': f"{cache_overhead_gb:.1f} GB (one-time)",
        'peak_usage': f"{n_sessions * raw_nwb_per_session_gb + cache_overhead_gb:.1f} GB",
        'final_usage': f"{n_sessions * processed_npz_per_session_mb / 1024 + cache_overhead_gb:.2f} GB",
    }

    return estimates


def main():
    parser = argparse.ArgumentParser(description="Download Allen Brain Observatory sessions")

    parser.add_argument('--list', action='store_true', help='List available sessions')
    parser.add_argument('--n-sessions', type=int, default=10,
                        help='Number of sessions to download')
    parser.add_argument('--area', type=str, default='VISp',
                        help='Visual area (VISp, VISl, VISal, VISpm, VISam, VISrl)')
    parser.add_argument('--session-ids', type=int, nargs='+',
                        help='Specific session IDs to download')
    parser.add_argument('--process', action='store_true',
                        help='Also process downloaded sessions to continuous traces')
    parser.add_argument('--estimate', action='store_true',
                        help='Estimate disk space required')

    args = parser.parse_args()

    print("="*70)
    print("ALLEN BRAIN OBSERVATORY DOWNLOADER")
    print("="*70)

    if not ALLENSDK_AVAILABLE:
        print("\n⚠️  allensdk not installed!")
        print("Install with: pip install allensdk")
        print("\nThis will also install required dependencies.")
        return

    if args.estimate:
        print(f"\nDisk space estimate for {args.n_sessions} sessions:")
        estimates = estimate_disk_space(args.n_sessions)
        for key, value in estimates.items():
            print(f"  {key}: {value}")
        print("\nNote: Raw NWB files can be deleted after processing.")
        return

    if args.list:
        list_available_sessions(area=args.area, max_display=100)
        return

    # Download
    downloaded = download_sessions(
        session_ids=args.session_ids,
        n_sessions=args.n_sessions,
        area=args.area,
    )

    # Process if requested
    if args.process and downloaded:
        project_root = Path(__file__).parent.parent.parent.parent
        input_dir = project_root / 'packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions'
        output_dir = project_root / 'packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions_continuous'

        print("\nProcessing to continuous traces...")
        process_downloaded_sessions(input_dir, output_dir)

    # Summary
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Process the neuros-astro pipeline on new sessions:")
    print("   python -m neuros_astro process --input <new_nwb_files>")
    print("\n2. Update config with new session IDs")
    print("\n3. Re-run training with more data:")
    print("   python scripts/train_with_cv.py --config configs/allen_multimodal.yaml")


if __name__ == '__main__':
    main()
