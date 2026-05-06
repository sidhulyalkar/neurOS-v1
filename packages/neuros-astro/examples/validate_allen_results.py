#!/usr/bin/env python
"""
Validate Allen Visual Coding processing results.

Performs comprehensive validation:
- File integrity checks
- Data quality assessment
- Sanity checks on metrics
- Publication readiness verification

Usage:
    python validate_allen_results.py
    python validate_allen_results.py --results-dir ./allen_nwb_results
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime


def validate_session_outputs(session_dir):
    """Validate outputs for a single session."""

    session_id = session_dir.name.replace("session_", "")
    issues = []
    warnings = []

    # Check required files
    required_files = [
        'events.parquet',
        'astro_tokens.npz',
        'neurofm_manifest.json',
        'summary.json',
    ]

    for filename in required_files:
        filepath = session_dir / filename
        if not filepath.exists():
            issues.append(f"Missing file: {filename}")
        elif filepath.stat().st_size == 0:
            issues.append(f"Empty file: {filename}")

    # Check figures
    figures_dir = session_dir / "figures"
    if not figures_dir.exists():
        issues.append("Missing figures directory")
    else:
        expected_figures = [
            'event_raster.png',
            'event_distributions.png',
            'network_graph.png',
        ]
        for fig_name in expected_figures:
            fig_path = figures_dir / fig_name
            if not fig_path.exists():
                warnings.append(f"Missing figure: {fig_name}")

    # Validate summary data
    summary_path = session_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        # Sanity checks
        if summary['n_events'] == 0:
            warnings.append("No events detected")

        if summary['duration_min'] < 10:
            warnings.append(f"Short recording: {summary['duration_min']:.1f} min")

        if summary['statistics']['event_rate_hz'] > 5.0:
            warnings.append(f"Very high event rate: {summary['statistics']['event_rate_hz']:.2f} Hz")

        if summary['statistics']['event_rate_hz'] < 0.01:
            warnings.append(f"Very low event rate: {summary['statistics']['event_rate_hz']:.2f} Hz")

        # Check network stability
        if summary['network_metrics']['stability'] < 0.5:
            warnings.append(f"Low network stability: {summary['network_metrics']['stability']:.3f}")

    return {
        'session_id': session_id,
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
    }


def validate_all_results(results_dir, output_path=None):
    """Validate all session results."""

    results_dir = Path(results_dir)

    print("=" * 80)
    print("ALLEN VISUAL CODING - RESULTS VALIDATION")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # 1. Check Overall Summary
    # -------------------------------------------------------------------------
    print("[1/4] Validating overall summary...")

    summary_path = results_dir / "overall_summary.json"
    if not summary_path.exists():
        print("  ✗ Missing overall_summary.json")
        return

    with open(summary_path) as f:
        overall_summary = json.load(f)

    n_sessions = overall_summary['n_sessions_processed']
    print(f"  ✓ Found {n_sessions} processed sessions")

    # -------------------------------------------------------------------------
    # 2. Validate Individual Sessions
    # -------------------------------------------------------------------------
    print(f"\n[2/4] Validating individual session outputs...")

    session_results = []
    for session_info in overall_summary['sessions']:
        session_id = session_info['session_id']
        session_dir = results_dir / f"session_{session_id}"

        if not session_dir.exists():
            session_results.append({
                'session_id': session_id,
                'valid': False,
                'issues': ['Session directory not found'],
                'warnings': [],
            })
            continue

        result = validate_session_outputs(session_dir)
        session_results.append(result)

    # Count valid sessions
    valid_sessions = sum(1 for r in session_results if r['valid'])
    total_issues = sum(len(r['issues']) for r in session_results)
    total_warnings = sum(len(r['warnings']) for r in session_results)

    print(f"  ✓ Valid sessions: {valid_sessions}/{n_sessions}")
    if total_issues > 0:
        print(f"  ⚠️  Total issues: {total_issues}")
    if total_warnings > 0:
        print(f"  ⚠️  Total warnings: {total_warnings}")

    # -------------------------------------------------------------------------
    # 3. Data Quality Checks
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Performing data quality checks...")

    # Extract metrics
    event_rates = [s['statistics']['event_rate_hz'] for s in overall_summary['sessions']]
    event_durations = [s['statistics']['duration_mean'] for s in overall_summary['sessions']]
    network_stabilities = [s['network_metrics']['stability'] for s in overall_summary['sessions']]

    quality_checks = {
        'event_rate_range': {
            'values': event_rates,
            'expected_min': 0.01,
            'expected_max': 2.0,
            'actual_min': np.min(event_rates),
            'actual_max': np.max(event_rates),
        },
        'event_duration': {
            'values': event_durations,
            'expected_min': 0.5,
            'expected_max': 10.0,
            'actual_min': np.min(event_durations),
            'actual_max': np.max(event_durations),
        },
        'network_stability': {
            'values': network_stabilities,
            'expected_min': 0.5,
            'expected_max': 1.0,
            'actual_min': np.min(network_stabilities),
            'actual_max': np.max(network_stabilities),
        },
    }

    all_passed = True
    for check_name, check in quality_checks.items():
        values = check['values']
        mean_val = np.mean(values)
        std_val = np.std(values)

        within_range = (check['actual_min'] >= check['expected_min'] and
                       check['actual_max'] <= check['expected_max'])

        if within_range:
            print(f"  ✓ {check_name}: {mean_val:.3f} ± {std_val:.3f} (PASS)")
        else:
            print(f"  ⚠️  {check_name}: {mean_val:.3f} ± {std_val:.3f} (OUT OF RANGE)")
            print(f"     Expected: [{check['expected_min']:.2f}, {check['expected_max']:.2f}]")
            print(f"     Actual: [{check['actual_min']:.2f}, {check['actual_max']:.2f}]")
            all_passed = False

    # -------------------------------------------------------------------------
    # 4. Publication Readiness
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Checking publication readiness...")

    pub_checks = {
        'sufficient_sessions': n_sessions >= 3,
        'sufficient_events': overall_summary['total_events'] >= 1000,
        'sufficient_recording_time': overall_summary['total_recording_time_min'] >= 180,
        'all_sessions_valid': valid_sessions == n_sessions,
        'no_critical_issues': total_issues == 0,
        'stable_networks': np.mean(network_stabilities) >= 0.6,
    }

    pub_ready = all(pub_checks.values())

    for check_name, passed in pub_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name.replace('_', ' ').title()}")

    # -------------------------------------------------------------------------
    # Generate Report
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Generating validation report...")

    report = {
        'validation_timestamp': datetime.now().isoformat(),
        'results_directory': str(results_dir),
        'n_sessions': int(n_sessions),
        'valid_sessions': int(valid_sessions),
        'total_issues': int(total_issues),
        'total_warnings': int(total_warnings),
        'session_validations': session_results,
        'quality_checks': {
            k: {
                'mean': float(np.mean(v['values'])),
                'std': float(np.std(v['values'])),
                'min': float(v['actual_min']),
                'max': float(v['actual_max']),
                'expected_range': [float(v['expected_min']), float(v['expected_max'])],
                'passed': bool(v['actual_min'] >= v['expected_min'] and
                          v['actual_max'] <= v['expected_max']),
            }
            for k, v in quality_checks.items()
        },
        'publication_readiness': {
            'ready': bool(pub_ready),
            'checks': {k: bool(v) for k, v in pub_checks.items()},
        },
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  ✓ Validation report saved: {output_path}")

        # Also save text summary
        text_path = output_path.with_suffix('.txt')
        with open(text_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ALLEN VISUAL CODING - VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {results_dir}\n\n")

            f.write("-" * 80 + "\n")
            f.write("SESSION VALIDATION\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Total Sessions: {n_sessions}\n")
            f.write(f"Valid Sessions: {valid_sessions}\n")
            f.write(f"Issues: {total_issues}\n")
            f.write(f"Warnings: {total_warnings}\n\n")

            if total_issues > 0 or total_warnings > 0:
                f.write("Session Details:\n")
                for result in session_results:
                    if result['issues'] or result['warnings']:
                        f.write(f"\n  Session {result['session_id']}:\n")
                        for issue in result['issues']:
                            f.write(f"    ✗ ISSUE: {issue}\n")
                        for warning in result['warnings']:
                            f.write(f"    ⚠️  WARNING: {warning}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("DATA QUALITY\n")
            f.write("-" * 80 + "\n\n")

            for check_name, check_data in report['quality_checks'].items():
                status = "PASS" if check_data['passed'] else "FAIL"
                f.write(f"{check_name}:\n")
                f.write(f"  Mean: {check_data['mean']:.3f} ± {check_data['std']:.3f}\n")
                f.write(f"  Range: [{check_data['min']:.3f}, {check_data['max']:.3f}]\n")
                f.write(f"  Expected: [{check_data['expected_range'][0]:.2f}, {check_data['expected_range'][1]:.2f}]\n")
                f.write(f"  Status: {status}\n\n")

            f.write("-" * 80 + "\n")
            f.write("PUBLICATION READINESS\n")
            f.write("-" * 80 + "\n\n")

            if pub_ready:
                f.write("✓ READY FOR PUBLICATION\n\n")
            else:
                f.write("⚠️  NOT YET READY\n\n")

            f.write("Checklist:\n")
            for check_name, passed in pub_checks.items():
                status = "✓" if passed else "✗"
                f.write(f"  {status} {check_name.replace('_', ' ').title()}\n")

            f.write("\n" + "=" * 80 + "\n")

            if pub_ready:
                f.write("DATA IS PUBLICATION-READY!\n")
            else:
                f.write("PLEASE ADDRESS ISSUES BEFORE PUBLICATION\n")

            f.write("=" * 80 + "\n")

        print(f"  ✓ Text summary saved: {text_path}")

    # -------------------------------------------------------------------------
    # Print Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    if pub_ready:
        print("\n✅ ALL CHECKS PASSED - DATA IS PUBLICATION-READY!")
    else:
        print("\n⚠️  SOME ISSUES FOUND - PLEASE REVIEW")

    print(f"\nSummary:")
    print(f"  Sessions: {valid_sessions}/{n_sessions} valid")
    print(f"  Events: {overall_summary['total_events']:,}")
    print(f"  Recording Time: {overall_summary['total_recording_time_min']:.1f} min")
    print(f"  Issues: {total_issues}")
    print(f"  Warnings: {total_warnings}")

    if not pub_ready:
        print("\nFailed checks:")
        for check_name, passed in pub_checks.items():
            if not passed:
                print(f"  ✗ {check_name.replace('_', ' ').title()}")

    print()

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate Allen Visual Coding processing results"
    )
    parser.add_argument('--results-dir', type=str, default='./allen_nwb_results',
                       help='Directory containing processing results')
    parser.add_argument('--output', type=str, default='./validation_report.json',
                       help='Output path for validation report')

    args = parser.parse_args()

    report = validate_all_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()
