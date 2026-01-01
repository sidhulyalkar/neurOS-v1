#!/usr/bin/env python3
"""
Multi-Session SAE Validation Across All Downloaded Allen Sessions

This script:
1. Loads ALL available Allen sessions from cache
2. Uses ALL units (not just "good quality") to avoid bias
3. Properly handles direction → orientation conversion
4. Runs validation on each session
5. Generates comprehensive report showing which sessions have strong tuning

Usage:
    python examples/multi_session_validation.py --allen-cache allen_validation_cache
    python examples/multi_session_validation.py --allen-cache allen_validation_cache --max-sessions 5
"""

import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_session_orientation_tuning(
    session_id: int,
    cache_dir: Path,
    use_all_units: bool = True
) -> Dict[str, Any]:
    """
    Analyze orientation tuning for a single Allen session.

    Parameters
    ----------
    session_id : int
        Allen session ID
    cache_dir : Path
        Path to Allen cache directory
    use_all_units : bool
        If True, use all units. If False, use only "good quality" units.

    Returns
    -------
    Dict with validation results including:
        - session_id
        - n_units
        - n_windows
        - max_correlation
        - n_significant (units with corr > 0.3)
        - mean_correlation
        - orientation_selective_fraction
        - success (bool)
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "neuros-foundation" / "src"))

        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
        from scipy.stats import pearsonr

        # Load session
        cache = EcephysProjectCache.from_warehouse(manifest=str(cache_dir / "manifest.json"))
        session = cache.get_session_data(session_id)

        # Get drifting gratings stimulus
        stim_table = session.stimulus_presentations

        # Debug: check stimulus names
        if 'stimulus_name' in stim_table.columns:
            unique_stimuli = stim_table['stimulus_name'].unique()
            logger.info(f"  Session {session_id}: Available stimuli: {list(unique_stimuli)}")
            drifting_gratings = stim_table[stim_table['stimulus_name'] == 'drifting_gratings']
        else:
            logger.warning(f"  Session {session_id}: No 'stimulus_name' column in stimulus_presentations")
            return {'session_id': session_id, 'success': False, 'error': 'no_stimulus_name_column'}

        if len(drifting_gratings) == 0:
            logger.warning(f"  Session {session_id}: No drifting gratings stimulus found")
            return {'session_id': session_id, 'success': False, 'error': 'no_drifting_gratings'}

        # Get units from VISp (V1)
        units = session.units[session.units.ecephys_structure_acronym == 'VISp']

        if use_all_units:
            # Use ALL units
            logger.info(f"  Session {session_id}: Using all {len(units)} VISp units")
        else:
            # Filter to good quality only
            if 'quality' in units.columns:
                units = units[units.quality == 'good']
                logger.info(f"  Session {session_id}: Using {len(units)} good quality VISp units")
            else:
                logger.info(f"  Session {session_id}: No quality column, using all {len(units)} units")

        if len(units) < 10:
            logger.warning(f"  Session {session_id}: Too few units ({len(units)})")
            return {'session_id': session_id, 'success': False, 'error': f'too_few_units_{len(units)}'}

        # Extract neural responses for each stimulus presentation
        responses = []
        directions = []
        orientations = []

        for idx, stim in drifting_gratings.iterrows():
            try:
                # IMPORTANT: Validate orientation FIRST before extracting spikes
                # This prevents mismatch if orientation is invalid
                if 'orientation' not in stim.index:
                    continue

                stim_direction = stim['orientation']

                # Skip null/NaN orientations
                if pd.isna(stim_direction) or stim_direction == 'null':
                    continue

                # Convert to float early - if this fails, we skip this stimulus
                try:
                    direction_float = float(stim_direction)
                except (ValueError, TypeError):
                    continue

                # Now extract timing info
                stim_start = stim['start_time']
                stim_stop = stim['stop_time']

                # Extract spikes during stimulus window for all units
                unit_rates = []
                for unit_id in units.index:
                    spikes = session.spike_times[unit_id]
                    window_spikes = spikes[(spikes >= stim_start) & (spikes < stim_stop)]
                    firing_rate = len(window_spikes) / (stim_stop - stim_start)
                    unit_rates.append(firing_rate)

                # Only append if we successfully got spikes AND valid orientation
                responses.append(unit_rates)
                directions.append(direction_float)

                # Convert direction (0-360) to orientation (0-180)
                # E.g., 0° and 180° both map to 0° (vertical orientation)
                orientation = direction_float % 180
                orientations.append(orientation)

            except Exception as e:
                # This should now be rare - only for unexpected errors
                logger.debug(f"  Session {session_id}: Unexpected error processing stimulus {idx}: {e}")
                continue

        if len(responses) == 0:
            logger.warning(f"  Session {session_id}: No valid stimulus presentations extracted")
            return {'session_id': session_id, 'success': False, 'error': 'no_valid_stimuli'}

        responses = np.array(responses)  # [n_stimuli, n_units]
        directions = np.array(directions)
        orientations = np.array(orientations)

        logger.info(f"  Session {session_id}: {len(responses)} stimulus presentations, {len(units)} units")
        unique_dirs = np.unique(directions)
        unique_oris = np.unique(orientations)
        logger.info(f"    Directions: {unique_dirs.tolist()}")
        logger.info(f"    Orientations (collapsed): {unique_oris.tolist()}")

        # Compute orientation tuning using circular correlation
        # For orientation, we use 2*theta because orientation period is 180°, not 360°
        orientation_sin = np.sin(np.deg2rad(orientations * 2))
        orientation_cos = np.cos(np.deg2rad(orientations * 2))

        correlations = []
        p_values = []

        for unit_idx in range(responses.shape[1]):
            unit_response = responses[:, unit_idx]

            # Compute circular-linear correlation
            # Correlate with both sin and cos, then take max
            corr_sin, p_sin = pearsonr(unit_response, orientation_sin)
            corr_cos, p_cos = pearsonr(unit_response, orientation_cos)

            # Use the stronger correlation
            if abs(corr_sin) > abs(corr_cos):
                correlations.append(abs(corr_sin))
                p_values.append(p_sin)
            else:
                correlations.append(abs(corr_cos))
                p_values.append(p_cos)

        correlations = np.array(correlations)
        p_values = np.array(p_values)

        # Statistics
        max_corr = np.max(correlations)
        mean_corr = np.mean(correlations)
        n_significant = np.sum(correlations > 0.3)
        fraction_selective = n_significant / len(correlations)

        # Also compute direction selectivity for comparison
        direction_sin = np.sin(np.deg2rad(directions))
        direction_cos = np.cos(np.deg2rad(directions))

        direction_correlations = []
        for unit_idx in range(responses.shape[1]):
            unit_response = responses[:, unit_idx]
            corr_sin, _ = pearsonr(unit_response, direction_sin)
            corr_cos, _ = pearsonr(unit_response, direction_cos)
            direction_correlations.append(max(abs(corr_sin), abs(corr_cos)))

        direction_correlations = np.array(direction_correlations)

        logger.info(f"  Session {session_id} Results:")
        logger.info(f"    Orientation tuning - Max: {max_corr:.3f}, Mean: {mean_corr:.3f}, Significant: {n_significant}/{len(correlations)} ({fraction_selective:.1%})")
        logger.info(f"    Direction tuning   - Max: {np.max(direction_correlations):.3f}, Mean: {np.mean(direction_correlations):.3f}")

        return {
            'session_id': session_id,
            'n_units': len(units),
            'n_stimuli': len(responses),
            'use_all_units': use_all_units,
            'orientation_tuning': {
                'max_correlation': float(max_corr),
                'mean_correlation': float(mean_corr),
                'n_significant': int(n_significant),
                'fraction_selective': float(fraction_selective),
                'correlations': correlations.tolist()
            },
            'direction_tuning': {
                'max_correlation': float(np.max(direction_correlations)),
                'mean_correlation': float(np.mean(direction_correlations)),
                'correlations': direction_correlations.tolist()
            },
            'stimulus_info': {
                'unique_directions': np.unique(directions).tolist(),
                'unique_orientations': np.unique(orientations).tolist()
            },
            'success': True
        }

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"  Session {session_id}: Failed - {error_msg}")

        # Print full traceback for debugging
        tb_str = traceback.format_exc()
        logger.error(f"Full traceback for session {session_id}:\n{tb_str}")

        return {'session_id': session_id, 'success': False, 'error': error_msg}


def main():
    parser = argparse.ArgumentParser(description='Multi-session Allen validation')
    parser.add_argument('--allen-cache', type=str, required=True, help='Path to Allen cache directory')
    parser.add_argument('--max-sessions', type=int, default=None, help='Maximum number of sessions to process')
    parser.add_argument('--use-all-units', action='store_true', default=True, help='Use all units (not just good quality)')
    parser.add_argument('--output', type=str, default='multi_session_results.json', help='Output JSON file')
    args = parser.parse_args()

    cache_dir = Path(args.allen_cache)

    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return

    logger.info("="*80)
    logger.info("Multi-Session Allen Orientation Tuning Validation")
    logger.info("="*80)
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Use all units: {args.use_all_units}")

    # Find all downloaded sessions
    session_dirs = sorted([d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith('session_')])
    session_ids = [int(d.name.replace('session_', '')) for d in session_dirs]

    logger.info(f"\nFound {len(session_ids)} downloaded sessions")

    if args.max_sessions:
        session_ids = session_ids[:args.max_sessions]
        logger.info(f"Processing first {args.max_sessions} sessions")

    # Process each session
    all_results = []

    for i, session_id in enumerate(session_ids, 1):
        logger.info(f"\n[{i}/{len(session_ids)}] Processing session {session_id}...")

        result = analyze_session_orientation_tuning(
            session_id=session_id,
            cache_dir=cache_dir,
            use_all_units=args.use_all_units
        )

        all_results.append(result)

    # Generate summary report
    logger.info("\n" + "="*80)
    logger.info("SUMMARY REPORT")
    logger.info("="*80)

    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]

    logger.info(f"\nProcessed: {len(all_results)} sessions")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(failed_results)}")

    if failed_results:
        logger.info("\nFailed sessions:")
        for r in failed_results:
            logger.info(f"  {r['session_id']}: {r.get('error', 'unknown error')}")

    if successful_results:
        # Sort by orientation selectivity
        successful_results.sort(
            key=lambda x: x['orientation_tuning']['fraction_selective'],
            reverse=True
        )

        logger.info("\n" + "-"*80)
        logger.info("TOP SESSIONS BY ORIENTATION SELECTIVITY")
        logger.info("-"*80)
        logger.info(f"{'Session ID':<12} {'Units':<8} {'Max Corr':<10} {'Mean Corr':<10} {'Selective':<12} {'% Selective':<12}")
        logger.info("-"*80)

        for r in successful_results[:10]:
            ot = r['orientation_tuning']
            logger.info(
                f"{r['session_id']:<12} {r['n_units']:<8} {ot['max_correlation']:<10.3f} "
                f"{ot['mean_correlation']:<10.3f} {ot['n_significant']:<12} "
                f"{ot['fraction_selective']*100:<11.1f}%"
            )

        # Overall statistics
        logger.info("\n" + "-"*80)
        logger.info("AGGREGATE STATISTICS")
        logger.info("-"*80)

        all_max_corrs = [r['orientation_tuning']['max_correlation'] for r in successful_results]
        all_mean_corrs = [r['orientation_tuning']['mean_correlation'] for r in successful_results]
        all_fractions = [r['orientation_tuning']['fraction_selective'] for r in successful_results]

        logger.info(f"Max correlation across sessions:")
        logger.info(f"  Best: {np.max(all_max_corrs):.3f}")
        logger.info(f"  Median: {np.median(all_max_corrs):.3f}")
        logger.info(f"  Worst: {np.min(all_max_corrs):.3f}")

        logger.info(f"\nMean correlation across sessions:")
        logger.info(f"  Best: {np.max(all_mean_corrs):.3f}")
        logger.info(f"  Median: {np.median(all_mean_corrs):.3f}")
        logger.info(f"  Worst: {np.min(all_mean_corrs):.3f}")

        logger.info(f"\nFraction of selective units (>0.3 corr):")
        logger.info(f"  Best: {np.max(all_fractions)*100:.1f}%")
        logger.info(f"  Median: {np.median(all_fractions)*100:.1f}%")
        logger.info(f"  Worst: {np.min(all_fractions)*100:.1f}%")

        # Sessions with strong tuning
        strong_sessions = [
            r for r in successful_results
            if r['orientation_tuning']['fraction_selective'] > 0.3
        ]

        logger.info(f"\n✓ Sessions with >30% selective units: {len(strong_sessions)}")

        if strong_sessions:
            logger.info("\nRECOMMENDED SESSIONS FOR VALIDATION:")
            for r in strong_sessions:
                ot = r['orientation_tuning']
                logger.info(
                    f"  Session {r['session_id']}: {ot['fraction_selective']*100:.1f}% selective "
                    f"({ot['n_significant']}/{r['n_units']} units), max_corr={ot['max_correlation']:.3f}"
                )

    # Save results to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            'parameters': {
                'cache_dir': str(cache_dir),
                'use_all_units': args.use_all_units,
                'n_sessions': len(session_ids)
            },
            'results': all_results,
            'summary': {
                'n_total': len(all_results),
                'n_successful': len(successful_results),
                'n_failed': len(failed_results)
            }
        }, f, indent=2)

    logger.info(f"\n✓ Results saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
