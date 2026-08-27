"""Qualify a measured or synthetic display observation against an Arena epoch."""
from __future__ import annotations

import argparse
import json

from .display_qualification import (
    EVIDENCE_CLASSES,
    DisplayQualificationConfig,
    TransitionDetectionConfig,
    load_display_observation_csv,
    qualify_display_observation,
    save_display_qualification,
)
from .manifest import load_manifest
from .presentation import compile_presentation_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Arena v1/v2 world manifest")
    parser.add_argument("--observation", required=True, help="UTF-8 CSV with timestamp_s and luminance columns")
    parser.add_argument("--epoch", required=True, type=int, help="zero-based presentation epoch index")
    parser.add_argument("--output", default="display-qualification.json")
    parser.add_argument("--timestamp-column", default="timestamp_s")
    parser.add_argument("--luminance-column", default="luminance")
    parser.add_argument("--units", default="arbitrary", help="observation amplitude units, e.g. volts, lux, adc_count")
    parser.add_argument(
        "--evidence-class",
        choices=sorted(EVIDENCE_CLASSES),
        default="unverified_observation",
        help="explicit evidence provenance; CSV input is never assumed to be photodiode evidence",
    )
    parser.add_argument("--source", default=None, help="optional source/provenance label overriding the CSV path")
    parser.add_argument(
        "--epoch-zero-s",
        type=float,
        default=None,
        help=(
            "observation timestamp aligned to Arena presentation-command epoch t=0; "
            "do not substitute the first photodiode transition; omit when clocks are unaligned"
        ),
    )
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--hysteresis-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-contrast", type=float, default=1e-9)
    parser.add_argument("--minimum-transition-separation-s", type=float, default=0.0)
    parser.add_argument(
        "--transition-match-tolerance-s",
        type=float,
        default=None,
        help="explicit aligned transition-match tolerance; default derives conservatively from the planned half-period",
    )
    args = parser.parse_args()

    if args.evidence_class in {"measured_photodiode", "measured_other"} and args.units.strip().lower() == "arbitrary":
        parser.error("measured display evidence requires explicit --units rather than the default 'arbitrary'")

    manifest = load_manifest(args.manifest)
    plan = compile_presentation_plan(
        manifest.scenario,
        manifest.display,
        manifest.device.sampling_rate_hz,
    )
    if args.epoch < 0 or args.epoch >= len(plan.epochs):
        parser.error(f"--epoch must be in [0, {len(plan.epochs) - 1}]")
    observation = load_display_observation_csv(
        args.observation,
        timestamp_column=args.timestamp_column,
        luminance_column=args.luminance_column,
        units=args.units,
        evidence_class=args.evidence_class,
        source=args.source,
        metadata={
            "manifest": args.manifest,
            "presentation_epoch_index": str(args.epoch),
        },
    )
    config = DisplayQualificationConfig(
        detection=TransitionDetectionConfig(
            low_quantile=args.low_quantile,
            high_quantile=args.high_quantile,
            hysteresis_fraction=args.hysteresis_fraction,
            minimum_contrast=args.minimum_contrast,
            minimum_transition_separation_s=args.minimum_transition_separation_s,
        ),
        epoch_zero_s=args.epoch_zero_s,
        transition_match_tolerance_s=args.transition_match_tolerance_s,
    )
    result = qualify_display_observation(plan.epochs[args.epoch], observation, config)
    output = save_display_qualification(result, args.output)
    payload = result.to_dict()
    print(json.dumps({
        "report": str(output),
        "schema": payload["schema"],
        "evidence_class": payload["observation"]["evidence_class"],
        "epoch": payload["epoch"]["presentation_epoch_index"],
        "planned_display_trace_model": payload["epoch"]["planned_display_trace_model"],
        "target_frequency_hz": payload["target_metrics"]["target_frequency_hz"],
        "observed_frequency_hz": payload["target_metrics"]["observed_frequency_hz"],
        "aligned_timing": payload["aligned_comparison"] is not None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
