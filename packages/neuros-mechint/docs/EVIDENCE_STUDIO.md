# MechInt Evidence Studio (experimental)

MechInt Evidence Studio converts an existing **paired-selectivity result** into
a self-contained HTML research report. It is a **read-only display layer**.
It does not discover circuits, execute a model, certify experiments, infer new
metrics, or promote mechanistic explanations.

The first release intentionally avoids a server, CDNs, runtime JavaScript,
Plotly, and a separate scientific state model.

## Generate the synthetic three-case showcase

From the repository root after installing the MechInt package:

~~~bash
pip install -e 'packages/neuros-mechint[dev]'
python packages/neuros-mechint/examples/26_evidence_studio_showcase.py \
  --out ./mechint-evidence-demo
~~~

Open these files in any browser:

- `mechint-evidence-demo/01-planted-path.html`: planted selective route;
  a **synthetic diagnostic**, not a real-model discovery.
- `mechint-evidence-demo/02-generic-degradation.html`: intervention damages both
  conditions equally; the selective causal statistic is zero.
- `mechint-evidence-demo/03-incomplete-experiment.html`: one missing circuit
  observation makes the study nonestimable. The report **does not draw a
  phantom effect chart**.

Each report has a companion JSON file containing the evaluator output and its
self-checking hash, plus `manifest.json` for the demonstration collection.

## Render an existing result

~~~bash
python -m neuros_mechint.visualization.evidence_studio \
  --input ./run/paired-selectivity.json \
  --out ./run/evidence.html \
  --title 'Held-out selectivity diagnostic'
~~~

The input must use schema `neuros-mechint.paired-selectivity.v1`. Its
`result_hash` must match the repository-standard `stable_hash` of the
full result dictionary **excluding** `result_hash`.

The renderer also fails closed on:

- unsupported schema or contradictory verdict flags;
- real-model/preregistration/exchangeability/execution claims that the v1
  evaluator does not itself independently verify;
- missing, duplicated or inconsistent per-trial/control summaries;
- nonfinite plotted values and invalid numeric intervals;
- inconsistent random-control rank fractions;
- incomplete results that pretend to have complete effect metrics.

A content hash checks accidental or untrusted mutation. It **does not prove
authenticity**, honest preregistration, or that the analysis was ever run on
real model weights. Those remain separate evidence requirements.

## Visual information architecture

1. **Verdict / scope**: human-readable status plus the original machine code;
   all results clearly marked experimental and not independently reproduced.
2. **Paired causal effect**: candidate mean and bootstrap interval, zero
   reference, magnitude comparator, and all random-control mean effects.
3. **Held-out trials**: accessible table with candidate, magnitude, and
   candidate-minus-magnitude per pair.
4. **Provenance and limitations**: scientific fingerprint, discovery
   commitment, result hash, control condition, and explicit missing authorities.

Incomplete reports show planned/received counts and missing pair/circuit IDs.
They do **not** substitute zero for unavailable effects.

**Critical statistical boundary:** the plus-one random-control rank is
descriptive. It is **not** presented as a calibrated p-value. A successful
diagnostic is **not** uniquely causal, not proof of an on-manifold intervention,
and not an externally reproduced real-model result.

For the reordered-control study, a negative result does **not** exclude all
single-token induction mechanisms: both prompt conditions contain a previous
C-to-D transition. Novel-suffix comparisons change multiple variables.

## Visual and accessibility design

The report uses self-contained responsive CSS and inline SVG. It has a skip
link, keyboard focus states, section navigation, screen-reader chart
title/description, table alternatives, print layout and reduced-motion
behavior. User-controlled trial labels and titles are escaped. A restrictive
CSP blocks remote assets and scripts. Data is not sent to any server.

No time-of-execution or fabricated performance score appears in a display.
The same diagnostic input and title produce identical HTML bytes.

## Relationship to existing package APIs

- `benchmarks.paired_selectivity` remains the evaluator and authority for the
  scientific calculation.
- `visualization.evidence_studio` only displays its output.
- `benchmarks.evidence_pack` remains the primary held-out
  necessity/sufficiency framework, with its own artifact schema.
- The old `reporting.py` and interactive visualization modules are untouched;
  this research surface has no additional optional visualization dependency.

## Future visualization phases

- **Evidence-pack gallery:** display discovery/validation breakdowns and
  intervention baseline sensitivity from independently verified evidence packs.
- **Counterfactual prediction panel:** predicted versus observed effects and
  calibration under distribution shift, once issue #219 has a valid schema.
- **Circuit inspection view:** target interactions, coverage, and alternate
  pathways, but only from an actual graph artifact, never an invented layout.
- **Cross-model comparison:** stability and hierarchical replication with
  distinct model/subject/session units.
- **External showcase:** pinned real-checkpoint experiment, independently
  reproduced, with source/license and limitations attached.

Avoid decorative simulated neuron networks, invented circuits, or a
leaderboard that ranks incompatible scientific studies.

## Qualification

~~~bash
pytest -q packages/neuros-mechint/tests/test_mechint_evidence_studio.py
ruff check packages/neuros-mechint/src/neuros_mechint/visualization/evidence_studio.py \
  packages/neuros-mechint/tests/test_mechint_evidence_studio.py \
  packages/neuros-mechint/examples/26_evidence_studio_showcase.py
~~~

This feature is experimental until these checks and repository workflows pass.
