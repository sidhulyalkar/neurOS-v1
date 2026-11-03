"""Offline, synthetic-only showcase for the read-only MechInt Evidence Studio.

Three test organisms illustrate positive diagnostics, a misleading generic
degradation, and an incomplete intervention grid. No pretrained model runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuros_mechint.benchmarks.paired_selectivity import (
    PairedSelectivityDesign,
    PairedSelectivityObservation,
    evaluate_paired_selectivity,
)
from neuros_mechint.core.manifest import stable_hash
from neuros_mechint.visualization.evidence_studio import write_paired_selectivity_html


def synthetic_case(
    *,
    candidate_effect: float,
    generic_damage: float = 0.0,
    incomplete: bool = False,
) -> dict:
    """Construct planted known-answer fixtures, never empirical observations."""
    random_controls = tuple(
        (f"random-{index:02}", (f"r{index:02}", f"s{index:02}"))
        for index in range(24)
    )
    pair_ids = tuple(
        (f"validation-{index:02}", stable_hash({"synthetic_trial": index, "seed": 37}))
        for index in range(8)
    )
    design = PairedSelectivityDesign(
        design_id="showcase-synthetic-v1",
        model_revision="synthetic:evidence-studio",
        discovery_commitment=stable_hash({"demo": True, "not_preregistered": True}),
        candidate_id="planted-candidate",
        candidate_targets=("planted-a", "planted-b"),
        magnitude_id="magnitude",
        magnitude_targets=("nuisance-a", "nuisance-b"),
        random_controls=random_controls,
        validation_pair_hashes=pair_ids,
        bootstrap_samples=200,
    )
    observations = []
    ids = (design.candidate_id, design.magnitude_id, *(name for name, _ in random_controls))
    for index, (pair_id, digest) in enumerate(pair_ids):
        base = 4.4 + index * 0.1
        for circuit in ids:
            selected_effect = candidate_effect if circuit == design.candidate_id else 0.0
            # Candidate and baseline get the same intact outputs for each trial.
            observations.append(
                PairedSelectivityObservation(
                    pair_id=pair_id,
                    pair_content_hash=digest,
                    circuit_id=circuit,
                    intact_repeated=base,
                    intact_control=0.0,
                    intervened_repeated=base - selected_effect - generic_damage,
                    intervened_control=-generic_damage,
                )
            )
    if incomplete:
        observations.pop()
    return evaluate_paired_selectivity(design, tuple(observations))


def main(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cases = (
        ("01-planted-path", synthetic_case(candidate_effect=3.2),
         "Planted selective route | synthetic demonstration"),
        ("02-generic-degradation", synthetic_case(candidate_effect=0, generic_damage=7),
         "Generic degradation trap | synthetic demonstration"),
        ("03-incomplete-experiment", synthetic_case(candidate_effect=3.2, incomplete=True),
         "Missing intervention case | synthetic demonstration"),
    )
    summary: dict[str, dict] = {}
    for name, result, title in cases:
        (out / f"{name}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_paired_selectivity_html(result, out / f"{name}.html", title=title)
        summary[name] = {
            "status": result["status"],
            "supports_selective_effect": result["supports_selective_effect"],
            "source": f"{name}.json",
            "report": f"{name}.html",
        }
    (out / "manifest.json").write_text(
        json.dumps(
            {"kind": "synthetic demonstration only", "cases": summary}, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("mechint-evidence-studio-demo"))
    arguments = parser.parse_args()
    print(json.dumps(main(arguments.out), indent=2, sort_keys=True))
