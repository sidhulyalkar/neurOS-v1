"""Command-line entry point for reproducible Synthetic BCI Arena runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .application import evaluate_application_trace, load_application_trace
from .benchmark import load_benchmark_pack, run_benchmark_pack
from .manifest import ArenaManifest, load_manifest, save_manifest
from .presets import get_preset, list_presets
from .runner import run_scenario


def _write_json(payload: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list_presets(), default=None)
    parser.add_argument("--manifest", default=None, help="portable neuros.synthetic_bci_arena.manifest.v1 JSON")
    parser.add_argument(
        "--benchmark-pack",
        default=None,
        help="portable neuros.synthetic_bci_arena.benchmark_pack.v1 JSON",
    )
    parser.add_argument(
        "--application-trace",
        default=None,
        help="optional engine-neutral neuros.synthetic_bci_arena.application_trace.v1 JSON for a single-world run",
    )
    parser.add_argument("--application-silence-grace-s", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="arena-report.json")
    parser.add_argument("--npz", default=None, help="optional derived synthetic arrays for offline inspection")
    parser.add_argument("--write-manifest", default=None, help="write the resolved world manifest before running")
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args()
    if args.list_presets:
        print("\n".join(list_presets()))
        return
    if args.application_silence_grace_s < 0:
        parser.error("--application-silence-grace-s must be non-negative")

    selected = sum(value is not None for value in (args.manifest, args.preset, args.benchmark_pack))
    if selected > 1:
        parser.error("choose only one of --manifest, --preset, or --benchmark-pack")

    if args.benchmark_pack:
        if args.npz or args.write_manifest or args.application_trace:
            parser.error("--npz, --write-manifest, and --application-trace apply to single-world runs, not benchmark packs")
        pack = load_benchmark_pack(args.benchmark_pack)
        result = run_benchmark_pack(pack)
        output = _write_json(result.to_dict(), args.output)
        print(json.dumps({
            "report": str(output),
            "schema": "neuros.synthetic_bci_arena.benchmark_result.v1",
            "pack": pack.name,
            "version": pack.version,
            "passed": result.passed,
            "cases": {case.name: case.passed for case in result.cases},
        }, indent=2, sort_keys=True))
        if not result.passed:
            raise SystemExit(2)
        return

    if args.manifest:
        manifest = load_manifest(args.manifest)
    else:
        scenario, participant, device, display, transport = get_preset(args.preset or "dual-target-smoke", args.seed)
        manifest = ArenaManifest(scenario, participant, device, display, transport)
    if args.write_manifest:
        save_manifest(manifest, args.write_manifest)
    run = run_scenario(
        manifest.scenario,
        manifest.participant,
        manifest.device,
        manifest.display,
        manifest.transport,
        manifest.world_model,
    )
    report = dict(run.report)
    if args.application_trace:
        trace = load_application_trace(args.application_trace)
        report["application_trace"] = {
            "application": trace.application,
            "version": trace.version,
            "metadata": dict(trace.metadata),
            "metrics": evaluate_application_trace(
                run,
                trace,
                silence_grace_s=args.application_silence_grace_s,
            ),
            "evidence_boundary": (
                "Application metrics are scored against this Arena world's causal truth; acceptance thresholds remain application-specific."
            ),
        }
    output = _write_json(report, args.output)
    if args.npz:
        np.savez_compressed(
            args.npz,
            data_uv=run.device_output.data_uv,
            timestamps_s=run.device_output.timestamps_s,
            ground_truth_timestamps_s=run.device_output.ground_truth_timestamps_s,
            ground_truth_target_hz=run.ground_truth_target_hz,
            stage_index=run.stage_index,
        )
    print(json.dumps({
        "report": str(output),
        "schema": run.report["schema"],
        "world_model": run.world_model.name,
        "world_model_evidence_level": run.report["world_model_evidence"]["evidence_level"],
        "application_trace": (None if not args.application_trace else report["application_trace"]),
        "metrics": run.report["metrics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
