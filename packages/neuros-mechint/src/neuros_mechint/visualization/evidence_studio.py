"""Dependency-light, read-only HTML evidence viewer for paired selectivity.

Visualizes an existing integrity-checked result; does not run models, select
candidates, infer calibrated p-values, or promote scientific claims.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from neuros_mechint.benchmarks.paired_selectivity import SELECTIVITY_SCHEMA
from neuros_mechint.core.manifest import stable_hash

_STATUS = {
    "SUPPORTED_DIAGNOSTIC_ONLY": (
        "Selectivity diagnostic supported", "supported",
        "A paired effect passed the predeclared diagnostic thresholds. "
        "This is not a discovered or independently reproduced mechanism.",
    ),
    "REJECTED_NO_SELECTIVE_EFFECT": (
        "No selective effect", "rejected",
        "The proposed intervention did not establish a positive, consistent selective effect.",
    ),
    "REJECTED_BEHAVIORAL_COMPETENCE": (
        "Behavioral gate rejected", "rejected",
        "The intact model did not meet the declared repetition-advantage gate.",
    ),
    "REJECTED_CONTROLS": (
        "Control comparison rejected", "rejected",
        "The candidate did not outperform all declared comparison criteria.",
    ),
    "INSUFFICIENT_COVERAGE": (
        "Insufficient validation coverage", "incomplete",
        "The planned number of validation trials or random controls was not reached.",
    ),
    "NONESTIMABLE_MISSING_OBSERVATIONS": (
        "Nonestimable: missing observations", "incomplete",
        "An expected intervention result is absent. No effect size is imputed.",
    ),
    "SENSITIVITY_ONLY_NOT_PRIMARY": (
        "Sensitivity analysis only", "incomplete",
        "The novel-suffix comparison changes several nuisance variables "
        "and cannot replace the primary reordered-control study.",
    ),
}

_CSS = """
:root {
  color-scheme: dark;
  --bg:#09101b; --panel:#121d2b; --panel2:#172636;
  --ink:#e8f1f7; --muted:#a7b8c9; --border:#304052;
  --cyan:#6ed5e9; --amber:#e6b777; --soft:#91a7d5;
}
* {box-sizing:border-box}
html {scroll-behavior:smooth}
body {margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a {color:var(--cyan)}
a:focus-visible,summary:focus-visible {outline:3px solid var(--cyan);outline-offset:4px}
.skip {position:absolute;top:-100px;left:12px;background:var(--panel);padding:10px;z-index:10}
.skip:focus {top:8px}
.shell {max-width:1120px;padding:28px 24px 72px;margin:auto}
.brand {display:flex;align-items:center;gap:10px;font-weight:750;letter-spacing:.03em}
.brand small {font-weight:400;color:var(--muted)}
.kicker {text-transform:uppercase;letter-spacing:.14em;font-size:11px;color:var(--cyan);font-weight:750}
h1 {font-size:clamp(27px,4.7vw,51px);line-height:1.15;letter-spacing:-.04em;max-width:820px;margin:14px 0}
h2 {font-size:21px;letter-spacing:-.025em;margin:0 0 8px}
h3 {font-size:15px;margin:0 0 8px}
p {margin:8px 0 14px}
.lead {color:var(--muted);max-width:760px;font-size:17px}
.tag {display:inline-block;border:1px solid var(--border);color:var(--muted);
  border-radius:999px;padding:3px 10px;font-size:12px}
.top {padding:18px 0 25px;border-bottom:1px solid var(--border)}
.nav {display:flex;gap:20px;flex-wrap:wrap;margin-top:20px;font-size:13px}
.nav a {text-decoration:none;color:var(--muted)}
.nav a:hover {color:var(--ink)}
.section {margin-top:26px;padding:26px;border-radius:16px;background:var(--panel);
  border:1px solid var(--border)}
.section-sub {color:var(--muted);font-size:13px;margin-top:0}
.hero {display:grid;grid-template-columns:minmax(0,1.6fr) minmax(210px,.8fr);gap:22px}
.verdict {border-left:3px solid var(--soft);padding:7px 0 7px 19px}
.verdict.rejected {border-color:var(--amber)}
.verdict.supported {border-color:var(--cyan)}
.verdict h2 {font-size:26px}
.metric-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.metric {background:var(--panel2);border:1px solid var(--border);
  border-radius:10px;padding:15px;min-width:0}
.metric strong {display:block;font-size:24px;font-variant-numeric:tabular-nums}
.metric span {font-size:12px;color:var(--muted)}
.stat-hint {font-size:12px;color:var(--muted)}
.grid {display:grid;grid-template-columns:1fr 1fr;gap:18px}
figure {margin:0}
svg {display:block;width:100%;height:auto;max-width:900px}
svg text {font-family:system-ui,-apple-system,sans-serif;fill:var(--muted);font-size:13px}
svg .lab {fill:var(--ink);font-weight:650}
table {width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td {border-bottom:1px solid var(--border);padding:9px 10px;text-align:left;font-size:13px}
th {color:var(--muted);font-weight:650}
td.num,th.num {text-align:right}
.scroll {overflow-x:auto}
.pill {font-size:12px;display:inline-block;padding:2px 9px;
  border-radius:999px;border:1px solid var(--border)}
.pill.good {color:var(--cyan)}
.pill.bad {color:var(--amber)}
.muted {color:var(--muted)}
.mono {font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}
.prov {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.prov-item {border-bottom:1px solid var(--border);padding-bottom:9px}
.prov-item dt {font-size:12px;color:var(--muted)}
.prov-item dd {font-size:13px;margin:5px 0}
.note {border-left:2px solid var(--amber);padding-left:15px;color:var(--muted)}
details {border-top:1px solid var(--border);padding:14px 0}
summary {cursor:pointer;font-weight:650}
footer {color:var(--muted);font-size:12px;margin-top:32px}
@media (max-width:700px) {
  .shell {padding:16px 14px 50px}.section {padding:18px}.hero,.grid {grid-template-columns:1fr}
  .metric-grid {grid-template-columns:1fr}.prov {grid-template-columns:1fr}
  .nav {gap:12px} h1 {letter-spacing:-.025em}
}
@media print {
  :root {color-scheme:light;--bg:#fff;--panel:#fff;--panel2:#f6f7fa;
    --ink:#161e2d;--muted:#39465b;--border:#cdd4df;--cyan:#006c81;--amber:#83500a}
  body {background:#fff}.section {break-inside:avoid} .nav,.skip {display:none}
}
@media (prefers-reduced-motion:reduce) {html {scroll-behavior:auto}}
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _interval(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-number interval")
    low, high = (_numeric(x, label) for x in value)
    if low > high:
        raise ValueError(f"{label} bounds are reversed")
    return low, high


def _expect_count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _valid_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on schema/identity/status, invented metrics and untrusted flags."""
    if not isinstance(payload, Mapping):
        raise ValueError("paired selectivity result must be a JSON object")
    result = dict(payload)
    digest = result.pop("result_hash", None)
    if not isinstance(digest, str) or stable_hash(result) != digest:
        raise ValueError("paired selectivity result_hash mismatch")
    if result.get("schema_version") != SELECTIVITY_SCHEMA:
        raise ValueError("unsupported selectivity result schema")
    status = result.get("status")
    if status not in _STATUS:
        raise ValueError("unrecognized selectivity status")
    if type(result.get("supports_selective_effect")) is not bool:
        raise ValueError("supports_selective_effect must be boolean")
    if result["supports_selective_effect"] != (status == "SUPPORTED_DIAGNOSTIC_ONLY"):
        raise ValueError("contradictory status and selective effect")
    if result.get("validation_only") is not True:
        raise ValueError("this viewer requires validation-only results")
    # These are hard-coded false in the v1 evaluator. Treat unexpected claims as
    # a schema mismatch, not evidence of independently verified science.
    for flag in (
        "real_model_evidence", "preregistration_independently_verified",
        "target_universe_execution_verified",
        "magnitude_matched_empirically_verified",
        "random_control_exchangeability_verified",
        "random_control_rank_is_calibrated_p_value",
    ):
        if result.get(flag) is not False:
            raise ValueError(f"unexpected authority flag: {flag}")
    for identity in ("design_fingerprint", "discovery_commitment"):
        value = result.get(identity)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"missing {identity}")
    if result.get("control_condition") not in {"reordered", "novel"}:
        raise ValueError("unknown control condition")
    planned = _expect_count(result.get("n_pairs_planned"), "n_pairs_planned")
    count = _expect_count(result.get("n_random_controls_planned"), "n_random_controls_planned")
    if planned < 1 or count < 1:
        raise ValueError("empty study design")
    if status == "NONESTIMABLE_MISSING_OBSERVATIONS":
        missing = result.get("missing")
        observed = _expect_count(result.get("n_observations_received"), "n_observations_received")
        if not isinstance(missing, list) or not missing:
            raise ValueError("nonestimable results require missing rows")
        if observed + len(missing) != planned * (count + 2):
            raise ValueError("missing observation count inconsistent with frozen design")
        for row in missing:
            if not isinstance(row, dict) or not all(
                isinstance(row.get(key), str) for key in ("pair_id", "circuit_id")
            ):
                raise ValueError("malformed missing observation")
        if "pair_evidence" in result:
            raise ValueError("nonestimable result cannot include completed effects")
    else:
        n = _expect_count(result.get("n_pairs"), "n_pairs")
        c = _expect_count(result.get("n_random_controls"), "n_random_controls")
        pairs = result.get("pair_evidence")
        controls = result.get("random_control_mean_effects")
        if n != planned or c != count or not isinstance(pairs, list) or len(pairs) != n:
            raise ValueError("held-out pair counts do not match frozen design")
        if not isinstance(controls, list) or len(controls) != c:
            raise ValueError("random-control counts do not match frozen design")
        ids: set[str] = set()
        candidates: list[float] = []
        magnitudes: list[float] = []
        for row in pairs:
            if not isinstance(row, dict) or not isinstance(row.get("pair_id"), str):
                raise ValueError("malformed validation pair")
            if row["pair_id"] in ids:
                raise ValueError("duplicate validation pair")
            ids.add(row["pair_id"])
            candidates.append(_numeric(row.get("candidate_effect"), "candidate_effect"))
            magnitudes.append(_numeric(row.get("magnitude_effect"), "magnitude_effect"))
        names: set[str] = set()
        for row in controls:
            if not isinstance(row, dict) or not isinstance(row.get("circuit_id"), str):
                raise ValueError("malformed random control")
            if row["circuit_id"] in names:
                raise ValueError("duplicate random control")
            names.add(row["circuit_id"])
            _numeric(row.get("mean_effect"), "random control mean")
        cmean = _numeric(
            result.get("mean_candidate_difference_in_differences"), "candidate mean"
        )
        mmean = _numeric(
            result.get("mean_magnitude_difference_in_differences"), "magnitude mean"
        )
        if not math.isclose(cmean, mean(candidates), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("candidate summary disagrees with trial effects")
        if not math.isclose(mmean, mean(magnitudes), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("magnitude summary disagrees with trial effects")
        _numeric(result.get("mean_intact_repetition_advantage"), "intact advantage")
        for key in (
            "ci_candidate_difference_in_differences",
            "ci_candidate_advantage_vs_magnitude",
            "ci_intact_repetition_advantage",
        ):
            _interval(result.get(key), key)
        rank = _numeric(
            result.get("random_control_plus_one_rank_fraction"), "descriptive rank"
        )
        expected = (
            1 + sum(row["mean_effect"] >= cmean for row in controls)
        ) / (count + 1)
        if not math.isclose(rank, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("random-control rank disagrees with control means")
        if not 0 < rank <= 1:
            raise ValueError("invalid control rank")
    return {**result, "result_hash": digest}


def _fmt(value: float, signed: bool = True) -> str:
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _stat(label: str, value: str, hint: str) -> str:
    return (
        f'<div class="metric"><span>{_escape(label)}</span>'
        f'<strong>{_escape(value)}</strong><div class="stat-hint">{_escape(hint)}</div></div>'
    )


def _effect_chart(result: Mapping[str, Any]) -> str:
    candidate = result["mean_candidate_difference_in_differences"]
    magnitude = result["mean_magnitude_difference_in_differences"]
    low, high = result["ci_candidate_difference_in_differences"]
    randoms = [row["mean_effect"] for row in result["random_control_mean_effects"]]
    values = [0.0, candidate, magnitude, low, high, *randoms]
    min_x, max_x = min(values), max(values)
    pad = max((max_x - min_x) * .12, .4)
    min_x -= pad
    max_x += pad

    def x(value: float) -> float:
        return 140 + (value - min_x) / (max_x - min_x) * 660

    zero = x(0)
    ticks = [
        f'<line x1="{zero:.1f}" x2="{zero:.1f}" y1="30" y2="238" '
        'stroke="#66798c" stroke-dasharray="4 5"/>'
    ]
    for value in (min_x, 0.0, max_x):
        ticks.append(
            f'<text x="{x(value):.1f}" y="257" text-anchor="middle">{_fmt(value)}</text>'
        )
    dots = []
    for index, value in enumerate(randoms):
        y = 183 + ((index % 7) - 3) * 5
        dots.append(
            f'<circle cx="{x(value):.1f}" cy="{y}" r="4.5" '
            'fill="#91a7d5" opacity=".72"/>'
        )
    return (
        '<svg viewBox="0 0 860 275" role="img" '
        'aria-labelledby="plot-title plot-description" xmlns="http://www.w3.org/2000/svg">'
        '<title id="plot-title">Matched intervention effect comparison</title>'
        '<desc id="plot-description">Candidate and magnitude-baseline mean effects, '
        'candidate bootstrap interval, and individual random-control mean effects. '
        'Horizontal reference is zero selective effect.</desc>'
        '<text x="10" y="71" class="lab">Candidate</text>'
        '<text x="10" y="133" class="lab">Magnitude</text>'
        '<text x="10" y="184" class="lab">Random controls</text>'
        f'{"".join(ticks)}'
        f'<line x1="{x(low):.1f}" x2="{x(high):.1f}" y1="67" y2="67" '
        'stroke="#6ed5e9" stroke-width="5" stroke-linecap="round"/>'
        f'<circle cx="{x(candidate):.1f}" cy="67" r="8" fill="#6ed5e9"/>'
        f'<circle cx="{x(magnitude):.1f}" cy="129" r="8" fill="#e6b777"/>'
        f'{"".join(dots)}'
        '<text x="470" y="272" text-anchor="middle">Selective effect (score difference)</text>'
        '</svg>'
    )


def _rows(result: Mapping[str, Any]) -> str:
    pairs = result["pair_evidence"]
    rows = []
    for row in pairs[:300]:
        cand, mag = row["candidate_effect"], row["magnitude_effect"]
        rows.append(
            f'<tr><td class="mono">{_escape(row["pair_id"])}</td>'
            f'<td class="num">{_fmt(cand)}</td>'
            f'<td class="num">{_fmt(mag)}</td>'
            f'<td class="num">{_fmt(cand - mag)}</td></tr>'
        )
    more = (
        f'<p class="muted">Showing first 300 of {len(pairs)} held-out pairs. '
        'Full source remains the authoritative evidence.</p>'
        if len(pairs) > 300 else ""
    )
    return (
        '<details><summary>Inspect held-out trial effects '
        f'({len(pairs)} paired trials)</summary>'
        '<div class="scroll"><table><thead><tr><th>Trial ID</th>'
        '<th class="num">Candidate</th><th class="num">Magnitude</th>'
        '<th class="num">Candidate minus magnitude</th></tr></thead><tbody>'
        f'{"".join(rows)}</tbody></table></div>{more}</details>'
    )


def _missing(result: Mapping[str, Any]) -> str:
    rows = [
        f'<tr><td class="mono">{_escape(row["pair_id"])}</td>'
        f'<td class="mono">{_escape(row["circuit_id"])}</td></tr>'
        for row in result["missing"][:300]
    ]
    return (
        f'<p class="note">{len(result["missing"])} required observations are missing; '
        'no effect size or uncertainty interval is displayed.</p>'
        '<details><summary>Show missing intervention cases</summary>'
        '<div class="scroll"><table><thead><tr><th>Trial ID</th>'
        '<th>Circuit</th></tr></thead><tbody>'
        f'{"".join(rows)}</tbody></table></div>'
        + (f'<p>Only the first 300 of {len(result["missing"])} missing cases are shown.</p>'
           if len(result["missing"]) > 300 else "")
        + '</details>'
    )


def render_paired_selectivity_html(
    report: Mapping[str, Any], *, title: str = "Causal selectivity | MechInt"
) -> str:
    """Render a portable, CSP-restricted report, never changing scientific state."""
    result = _valid_result(report)
    status = result["status"]
    heading, kind, meaning = _STATUS[status]
    complete = status != "NONESTIMABLE_MISSING_OBSERVATIONS"
    stats = (
        _stat("Held-out pairs", str(result["n_pairs"]), "Planned validation pairs")
        + _stat(
            "Candidate effect", _fmt(result["mean_candidate_difference_in_differences"]),
            "Paired difference-in-differences; not a causal guarantee",
        )
        + _stat(
            "Descriptive control rank",
            f'{result["random_control_plus_one_rank_fraction"]:.3f}',
            "Not a calibrated p-value",
        )
        if complete else (
            _stat("Planned pairs", str(result["n_pairs_planned"]), "No valid effect estimate")
            + _stat(
                "Observations received", str(result["n_observations_received"]),
                "Incomplete intervention grid",
            )
            + _stat("Missing cases", str(len(result["missing"])), "Not imputed as zero")
        )
    )
    if complete:
        ci = result["ci_candidate_difference_in_differences"]
        magnitude_ci = result["ci_candidate_advantage_vs_magnitude"]
        plot = (
            '<section class="section" id="effects" aria-labelledby="effects-h">'
            '<h2 id="effects-h">Intervention effects</h2>'
            '<p class="section-sub">Candidate versus a same-cardinality magnitude baseline '
            'and declared random controls. These are diagnostic statistics, not a full circuit proof.</p>'
            '<figure>'
            + _effect_chart(result)
            + f'<figcaption class="muted">Candidate bootstrap interval '
            f'[{_fmt(ci[0])}, {_fmt(ci[1])}]. Candidate advantage over magnitude '
            f'interval [{_fmt(magnitude_ci[0])}, {_fmt(magnitude_ci[1])}]. '
            'Bootstrap units are paired validation trials.</figcaption></figure>'
            '<p class="note">The control rank is a descriptive plus-one position among '
            'declared circuits. Exchangeability and a calibrated significance test '
            'were not established.</p></section>'
        )
        trials = _rows(result)
    else:
        plot = (
            '<section class="section" id="effects"><h2>Effects unavailable</h2>'
            '<p>No chart is drawn because the result is nonestimable.</p>'
            + _missing(result) + '</section>'
        )
        trials = '<p>Per-trial effect summaries are unavailable for incomplete evidence.</p>'
    prov_rows = (
        ("Scientific design fingerprint", result["design_fingerprint"]),
        ("Discovery commitment", result["discovery_commitment"]),
        ("Result integrity hash", result["result_hash"]),
        ("Control condition", result["control_condition"]),
        ("Evidence type", "Synthetic diagnostic; no real-model evidence established"),
        ("Independent preregistration", "Not independently verified"),
        ("Independent replication", "Not established by this artifact"),
        ("Target-universe execution", "Not independently verified"),
        ("Magnitude matching", "Not independently verified"),
    )
    provenance = "".join(
        f'<div class="prov-item"><dt>{_escape(label)}</dt>'
        f'<dd class="mono">{_escape(value)}</dd></div>'
        for label, value in prov_rows
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        'style-src \'unsafe-inline\'; img-src data:; base-uri \'none\'; form-action \'none\'">'
        f'<title>{_escape(title)}</title><style>{_CSS}</style></head><body>'
        '<a class="skip" href="#main">Skip to evidence</a>'
        '<div class="shell"><div class="brand">neurOS <small>/ mechint / evidence studio</small></div>'
        '<header class="top"><div class="kicker">Experimental · read-only scientific report</div>'
        f'<h1>{_escape(title)}</h1>'
        '<p class="lead">A compact, auditable view of paired intervention evidence. '
        'Negative and incomplete findings remain visible, and statistical authority '
        'is never inferred from visual polish.</p>'
        '<span class="tag">Synthetic diagnostic · not independently reproduced</span>'
        '<nav class="nav" aria-label="Report sections"><a href="#main">Verdict</a>'
        '<a href="#effects">Effects</a><a href="#trials">Trials</a>'
        '<a href="#provenance">Provenance</a></nav></header>'
        '<main id="main"><section class="section hero" aria-labelledby="verdict-h">'
        f'<div class="verdict {kind}"><div class="kicker">Scientific status</div>'
        f'<h2 id="verdict-h">{_escape(heading)}</h2><p>{_escape(meaning)}</p>'
        f'<p class="muted mono">{_escape(status)}</p></div>'
        f'<div><div class="metric-grid" style="grid-template-columns:1fr">{stats}</div></div>'
        '</section>'
        f'{plot}'
        '<section class="section" id="trials"><h2>Held-out trial evidence</h2>'
        '<p class="section-sub">Per-trial summaries are retained rather than '
        'replaced by a single attractive aggregate.</p>'
        f'{trials}</section>'
        '<section class="section" id="provenance"><h2>Provenance and claim boundary</h2>'
        '<p class="section-sub">Content hashes check self-consistency, not provenance '
        'authenticity or the truth of the experiment.</p>'
        f'<dl class="prov">{provenance}</dl>'
        '<details open><summary>Why this is not a mechanistic discovery claim</summary>'
        '<p>Neither a positive difference-in-differences nor the descriptor '
        '"supported" identifies a unique, naturally used circuit. '
        'The current design does not independently establish intervention-manifold '
        'validity, correct candidate selection, pretrained-model execution, '
        'multiple-testing control, cross-context generalization, or reproduction.</p>'
        '<p>A reordered suffix tests order-sensitive repetition, not all induction. '
        'Novel-suffix results are sensitivity analyses only.</p></details></section>'
        '</main><footer>MechInt Evidence Studio · static, self-contained report · '
        'all displayed effects derive from the supplied hash-checked diagnostic.</footer>'
        '</div></body></html>'
    )


def write_paired_selectivity_html(
    report: Mapping[str, Any], path: str | Path, *,
    title: str = "Causal selectivity | MechInt",
) -> Path:
    """Write one self-contained artifact without changing the underlying result."""
    destination = Path(path)
    content = render_paired_selectivity_html(report, title=title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an existing paired-selectivity result.")
    parser.add_argument("--input", type=Path, required=True, help="Evaluator result JSON")
    parser.add_argument("--out", type=Path, required=True, help="Standalone HTML output")
    parser.add_argument("--title", default="Causal selectivity | MechInt")
    args = parser.parse_args()
    if args.input.stat().st_size > 2_000_000:
        parser.error("result file exceeds the 2 MB viewer limit")
    payload = json.loads(
        args.input.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
        object_pairs_hook=_unique_json_keys,
    )
    destination = write_paired_selectivity_html(payload, args.out, title=args.title)
    print(destination)


if __name__ == "__main__":
    main()
