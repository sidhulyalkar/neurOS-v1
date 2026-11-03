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
from neuros_mechint.evidence_studio import write_paired_selectivity_html


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
    gallery_cases = (
        ("01-planted-path", "Planted selective route",
         "A known-answer intervention selectively changes the repeated-context response."),
        ("02-generic-degradation", "Generic degradation trap",
         "A large raw disruption is not selective because both matched conditions worsen."),
        ("03-incomplete-experiment", "Missing intervention evidence",
         "One absent planned observation prevents a valid effect estimate."),
    )
    cards = "\n".join(
        '<article class="case"><p class="counter">EXPERIMENT '
        + str(index).zfill(2)
        + '</p><h2>' + title + '</h2><p>' + description + '</p>'
        + '<p class="status">' + summary[name]["status"] + '</p>'
        + '<div class="links"><a href="' + name + '.html">View evidence</a>'
        + '<a href="' + name + '.json">Source JSON</a></div></article>'
        for index, (name, title, description) in enumerate(gallery_cases, start=1)
    )
    index_html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        'style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        '<title>MechInt Evidence Studio | Synthetic Showcase</title>'
        '<style>:root{color-scheme:dark}*{box-sizing:border-box}'
        'body{background:#09101b;color:#e8f1f7;margin:0;'
        'font:15px/1.55 system-ui,-apple-system,sans-serif}'
        'main{max-width:1100px;margin:auto;padding:48px 24px 80px}'
        '.kicker{color:#6ed5e9;letter-spacing:.14em;font-size:12px;font-weight:700}'
        'h1{font-size:clamp(30px,5vw,55px);line-height:1.14;letter-spacing:-.04em}'
        '.intro{max-width:700px;color:#a7b8c9;font-size:17px}'
        '.cases{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;'
        'margin-top:35px}.case{border:1px solid #304052;border-radius:15px;'
        'padding:24px;background:#121d2b}.case h2{font-size:21px;line-height:1.2}'
        '.case p{color:#a7b8c9}.case .counter{font-size:11px;letter-spacing:.12em;'
        'color:#6ed5e9}.case .status{font-size:12px;overflow-wrap:anywhere;'
        'font-family:ui-monospace,monospace}.links{display:flex;gap:16px;flex-wrap:wrap}'
        'a{color:#6ed5e9}a:focus-visible{outline:3px solid #6ed5e9;outline-offset:3px}'
        'footer{margin-top:30px;color:#a7b8c9;font-size:12px}'
        '@media(max-width:760px){main{padding:28px 16px}.cases{grid-template-columns:1fr}}'
        '</style></head><body><main>'
        '<div class="kicker">NEUROS / MECHINT / EVIDENCE STUDIO</div>'
        '<h1>Three outcomes.<br>One honest evidence standard.</h1>'
        '<p class="intro">Synthetic known-answer examples only. Explore a planted '
        'positive diagnostic, a rejected false-positive trap, and a nonestimable '
        'study. None are evidence of a real-model mechanism.</p>'
        '<section class="cases" aria-label="Synthetic evidence cases">'
        + cards
        + '</section><footer>Fully offline. Source results accompany each '
        'HTML explanation. No model weights were executed.</footer></main></body></html>'
    )
    (out / "index.html").write_text(index_html, encoding="utf-8")
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
