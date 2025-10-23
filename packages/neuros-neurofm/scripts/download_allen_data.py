"""
Allen Brain Observatory Dataset Downloader
===========================================

This script downloads the Allen Brain Observatory Visual Coding - Neuropixels dataset
for training NeuroFM-X.

Dataset: Allen Visual Coding - Neuropixels
Source: Allen Institute for Brain Science
Data Type: Neuropixels recordings from visual cortex during natural image presentations

Output: Data will be cached in ./data/allen_neuropixels/
"""

import os
import sys
from pathlib import Path
import argparse
import shutil
import traceback

def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError:
        missing.append("allensdk")

    if missing:
        print("\n" + "="*80)
        print("ERROR: Missing required packages!")
        print("="*80)
        print("\nPlease install the following packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        print("\n" + "="*80)
        sys.exit(1)

def download_session(cache, session_id, args, cache_dir):
    """
    Downloads a single session. Returns session metadata dict or None if failed/skipped.
    """
    try:
        session = cache.get_session_data(session_id)

        # Get metadata
        n_units = len(session.units)
        duration = session.stimulus_presentations.stop_time.max()
        brain_areas = session.units['ecephys_structure_acronym'].unique()

        # Check brain area
        if args.brain_area != 'all' and args.brain_area not in brain_areas:
            print(f"    ⊗ Skipped (brain area {args.brain_area} not found)")
            return None

        print(f"    ✓ Success!")
        print(f"      - Units: {n_units}")
        print(f"      - Duration: {duration:.1f} seconds")
        print(f"      - Brain areas: {', '.join(brain_areas)}")

        return {
            'session_id': session_id,
            'n_units': n_units,
            'duration': duration,
            'brain_areas': list(brain_areas),
        }

    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")
        traceback.print_exc()
        return None

def remove_partial_session(cache_dir, session_id):
    """
    Deletes the cached directory for a given session_id if it exists.
    """
    partial_dir = cache_dir / f'session_{session_id}'
    if partial_dir.exists():
        print(f"    ⚠ Deleting partial session directory: {partial_dir}")
        shutil.rmtree(partial_dir)

def main():
    """
    Downloads the Allen Brain Observatory Visual Coding - Neuropixels dataset.

    This script downloads the specified number of sessions from the Allen Brain Observatory, filters them based on the specified criteria, and caches the results locally.

    The following arguments are available:

    - data-dir: Directory to store downloaded data
    - num-sessions: Number of sessions to download (default: 10, max: ~60)
    - stimulus-type: Type of stimulus to filter sessions (natural_images, drifting_gratings, or all)
    - brain-area: Brain area to filter (e.g., VISp, VISl, VISal, or all)

    The script will print out the progress of the download and the statistics of the dataset.

    The dataset info is saved to a file named "dataset_info.txt" in the specified data directory.

    The data is cached in a directory named "cache" in the specified data directory.

    Next steps:

    1. The data has been downloaded and cached locally.
    2. Run the training script: python train_allen_data.py
    3. The training script will automatically find and use this data.

    """
    parser = argparse.ArgumentParser(description='Download Allen Brain Observatory data')
    parser.add_argument('--data-dir', type=str, default='./data/allen_neuropixels',
                        help='Directory to store downloaded data')
    parser.add_argument('--num-sessions', type=int, default=10,
                        help='Number of sessions to download (default: 10, max: ~60)')
    parser.add_argument('--stimulus-type', type=str, default='natural_images',
                        choices=['natural_images', 'drifting_gratings', 'all'],
                        help='Type of stimulus to filter sessions')
    parser.add_argument('--brain-area', type=str, default='all',
                        help='Brain area to filter (e.g., VISp, VISl, VISal, or "all")')
    parser.add_argument('--cleanup', action='store_true', help='Remove failed/incomplete session folders')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Allen Brain Observatory Visual Coding - Neuropixels Dataset Downloader")
    print("="*80)

    # Check dependencies
    print("\n[1/5] Checking dependencies...")
    check_dependencies()
    print("  ✓ All dependencies installed")

    # Import here after dependency check
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    import pandas as pd

    # Create data directory
    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/5] Initializing Allen SDK cache...")
    print(f"  Cache directory: {cache_dir.absolute()}")

    # Initialize the cache
    manifest_path = cache_dir / "manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

    print("  ✓ Cache initialized")

    # Get all available sessions
    print(f"\n[3/5] Fetching session metadata...")
    sessions = cache.get_session_table()

    print(f"  ✓ Found {len(sessions)} total sessions in database")

    # Filter sessions based on criteria
    print(f"\n[4/5] Filtering sessions...")
    filtered_sessions = sessions.copy()

    # Filter by stimulus type
    if args.stimulus_type != 'all':
        if args.stimulus_type == 'natural_images':
            # Look for sessions with natural images
            filtered_sessions = filtered_sessions[
                filtered_sessions['session_type'].str.contains('functional_connectivity', na=False)
            ]
            print(f"  - Filtered for natural images stimulus")
        elif args.stimulus_type == 'drifting_gratings':
            filtered_sessions = filtered_sessions[
                filtered_sessions['session_type'].str.contains('brain_observatory', na=False)
            ]
            print(f"  - Filtered for drifting gratings stimulus")

    # Filter by brain area if specified
    if args.brain_area != 'all':
        # This requires loading more detailed info, so we'll just note it
        print(f"  - Note: Brain area filtering ({args.brain_area}) will be applied during download")

    # Limit to requested number of sessions
    if len(filtered_sessions) > args.num_sessions:
        # Prioritize sessions with more units
        filtered_sessions = filtered_sessions.head(args.num_sessions)

    session_ids = filtered_sessions.index.tolist()

    print(f"  ✓ Selected {len(session_ids)} sessions to download")

    # Download sessions
    print(f"\n[5/5] Downloading sessions...")
    print("  This may take a while depending on your internet connection.")
    print("  Each session is approximately 1-3 GB.")
    print("")

    downloaded_sessions = []
    failed_sessions = []
    
    # Iterate and download each session
    for i, session_id in enumerate(session_ids):
        print(f"  [{i+1}/{len(session_ids)}] Downloading session {session_id}...")

        # Path where this session would be cached
        session_path = cache_dir / f'session_{session_id}'
        if session_path.exists() and not args.cleanup:
            print(f"    → Skipping (already exists)")
            continue
        elif session_path.exists() and args.cleanup:
            remove_partial_session(cache_dir, session_id)

        result = download_session(cache, session_id, args, cache_dir)
        if result:
            downloaded_sessions.append(result)
        else:
            failed_sessions.append(session_id)

    # Summary
    print("\n" + "="*80)
    print("Download Complete!")
    print("="*80)
    print(f"\nSuccessfully downloaded: {len(downloaded_sessions)} sessions")
    if failed_sessions:
        print(f"Failed to download: {len(failed_sessions)} sessions")

    # Print statistics
    if downloaded_sessions:
        import pandas as pd
        df = pd.DataFrame(downloaded_sessions)

        print("\n" + "="*80)
        print("Dataset Statistics")
        print("="*80)
        print(f"  Total sessions: {len(df)}")
        print(f"  Total units: {df['n_units'].sum():,}")
        print(f"  Average units per session: {df['n_units'].mean():.1f}")
        print(f"  Total recording time: {df['duration'].sum()/60:.1f} minutes")
        print(f"  Average duration per session: {df['duration'].mean():.1f} seconds")

        # Get unique brain areas
        all_brain_areas = set()
        for areas in df['brain_areas']:
            all_brain_areas.update(areas)
        print(f"  Brain areas represented: {', '.join(sorted(all_brain_areas))}")

    # Save session info
    info_file = data_dir / "dataset_info.txt"
    with open(info_file, 'w') as f:
        f.write("Allen Brain Observatory Visual Coding - Neuropixels Dataset\n")
        f.write("="*80 + "\n\n")
        f.write(f"Downloaded: {len(downloaded_sessions)} sessions\n")
        f.write(f"Cache directory: {cache_dir.absolute()}\n\n")
        f.write("Session IDs:\n")
        for sess in downloaded_sessions:
            f.write(f"  - {sess['session_id']}: {sess['n_units']} units, "
                   f"{sess['duration']:.1f}s, areas: {', '.join(sess['brain_areas'])}\n")

    print(f"\n✓ Dataset info saved to: {info_file.absolute()}")
    print(f"✓ Data cached at: {cache_dir.absolute()}")

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. The data has been downloaded and cached locally")
    print("2. Run the training script: python train_allen_data.py")
    print("3. The training script will automatically find and use this data")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
