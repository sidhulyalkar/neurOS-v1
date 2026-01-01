#!/usr/bin/env python3
"""
Download Allen Visual Coding Data for SAE Validation
=====================================================

This script downloads Allen Institute Neuropixels data specifically for
validating SAE features with orientation selectivity analysis.

It downloads sessions with:
- Drifting gratings stimulus (for orientation tuning)
- V1 recordings (primary visual cortex)
- Sufficient high-quality units (>100 good units)

Author: neurOS Validation Framework
"""

import sys
import argparse
from pathlib import Path


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
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        print("\n" + "="*80)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download Allen data for SAE validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 3 sessions for quick testing
  python scripts/download_validation_data.py --num-sessions 3

  # Download 10 sessions for thorough validation
  python scripts/download_validation_data.py --num-sessions 10

  # Specify custom cache directory
  python scripts/download_validation_data.py --cache-dir ./my_allen_cache
        """
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./allen_validation_cache',
        help='Directory to cache Allen data (default: ./allen_validation_cache)'
    )
    parser.add_argument(
        '--num-sessions',
        type=int,
        default=5,
        help='Number of sessions to download (default: 5, recommend 3-10)'
    )
    parser.add_argument(
        '--min-units',
        type=int,
        default=100,
        help='Minimum number of good units required per session (default: 100)'
    )
    parser.add_argument(
        '--brain-areas',
        type=str,
        nargs='+',
        default=['VISp'],
        help='Brain areas to include (default: VISp for V1)'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Allen Visual Coding Data Downloader for SAE Validation")
    print("="*80)
    print("\nThis script downloads Allen Neuropixels data for validating SAE features")
    print("with orientation selectivity analysis.")
    print("")

    # Check dependencies
    print("[1/6] Checking dependencies...")
    check_dependencies()
    print("  ✓ All dependencies installed")

    # Import after dependency check
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    import numpy as np

    # Setup cache
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/6] Initializing Allen SDK cache...")
    print(f"  Cache directory: {cache_dir.absolute()}")

    manifest_path = cache_dir / "manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))
    print("  ✓ Cache initialized")

    # Get session table
    print(f"\n[3/6] Fetching available sessions...")
    sessions = cache.get_session_table()
    print(f"  ✓ Found {len(sessions)} total sessions")

    # Filter for brain observatory sessions (have drifting gratings)
    print(f"\n[4/6] Filtering for validation-suitable sessions...")
    brain_obs_sessions = sessions[
        sessions.session_type == 'brain_observatory_1.1'
    ]
    print(f"  ✓ Found {len(brain_obs_sessions)} brain observatory sessions")
    print(f"    (These have drifting gratings stimulus for orientation analysis)")

    # Find sessions with sufficient units in target brain areas
    print(f"\n[5/6] Searching for sessions with {args.min_units}+ units in {args.brain_areas}...")
    print("  This may take a few minutes as we check each session...")
    print("")

    suitable_sessions = []

    for idx, (session_id, session_row) in enumerate(brain_obs_sessions.iterrows()):
        if len(suitable_sessions) >= args.num_sessions:
            break

        try:
            print(f"  [{idx+1}/{len(brain_obs_sessions)}] Checking session {session_id}...", end=' ')

            # Load session data
            session = cache.get_session_data(session_id)

            # Filter to target brain areas
            units_in_areas = session.units[
                session.units.ecephys_structure_acronym.isin(args.brain_areas)
            ]

            # Count good units
            good_units = units_in_areas[units_in_areas.quality == 'good']

            if len(good_units) >= args.min_units:
                # Check for drifting gratings stimulus
                stim_presentations = session.stimulus_presentations
                has_drifting_gratings = 'drifting_gratings' in stim_presentations.stimulus_name.values

                if has_drifting_gratings:
                    n_gratings = len(stim_presentations[
                        stim_presentations.stimulus_name == 'drifting_gratings'
                    ])

                    suitable_sessions.append({
                        'session_id': session_id,
                        'n_units': len(good_units),
                        'brain_areas': args.brain_areas,
                        'n_drifting_gratings': n_gratings
                    })

                    print(f"✓ GOOD!")
                    print(f"      - Good units: {len(good_units)}")
                    print(f"      - Drifting grating trials: {n_gratings}")
                else:
                    print("✗ No drifting gratings")
            else:
                print(f"✗ Only {len(good_units)} good units")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("Download Complete!")
    print("="*80)

    if not suitable_sessions:
        print("\n⚠ WARNING: No suitable sessions found!")
        print("\nTry:")
        print("  - Reducing --min-units (e.g., --min-units 50)")
        print("  - Increasing --num-sessions")
        print("  - Using different --brain-areas (e.g., VISl, VISal)")
        sys.exit(1)

    print(f"\n✓ Found and cached {len(suitable_sessions)} suitable sessions")
    print("\nSession Summary:")

    total_units = 0
    total_trials = 0

    for i, sess in enumerate(suitable_sessions, 1):
        total_units += sess['n_units']
        total_trials += sess['n_drifting_gratings']
        print(f"  {i}. Session {sess['session_id']}")
        print(f"     - Good units: {sess['n_units']}")
        print(f"     - Drifting grating trials: {sess['n_drifting_gratings']}")

    print(f"\nTotal Statistics:")
    print(f"  - Total good units: {total_units:,}")
    print(f"  - Total orientation trials: {total_trials:,}")
    print(f"  - Average units/session: {total_units/len(suitable_sessions):.1f}")

    # Save session info
    info_file = cache_dir / "validation_sessions.txt"
    with open(info_file, 'w') as f:
        f.write("Allen Visual Coding Sessions for SAE Validation\n")
        f.write("="*80 + "\n\n")
        f.write(f"Downloaded: {len(suitable_sessions)} sessions\n")
        f.write(f"Brain areas: {', '.join(args.brain_areas)}\n")
        f.write(f"Min units: {args.min_units}\n")
        f.write(f"Cache directory: {cache_dir.absolute()}\n\n")
        f.write("Session IDs:\n")
        for sess in suitable_sessions:
            f.write(f"  - {sess['session_id']}: {sess['n_units']} units, "
                   f"{sess['n_drifting_gratings']} orientation trials\n")

    print(f"\n✓ Session info saved to: {info_file}")
    print(f"✓ Data cached at: {cache_dir.absolute()}")

    # Print next steps
    print("\n" + "="*80)
    print("[6/6] Next Steps - Run Validation")
    print("="*80)
    print("\n1. The Allen data is now cached and ready to use!")
    print("\n2. Run the validation example:")
    print("   python examples/sae_validation_example.py")
    print("\n3. Or use in your own code:")
    print("   from neuros.datasets import AllenVisualCodingValidator")
    print(f"   validator = AllenVisualCodingValidator(")
    print(f"       session_id={suitable_sessions[0]['session_id']},  # Use downloaded session")
    print(f"       cache_dir='{cache_dir.absolute()}',")
    print(f"       brain_areas={args.brain_areas}")
    print(f"   )")
    print("   windows = validator.get_neural_windows()")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
