# Experiments

This directory contains exploratory research code, notebooks, papers, and evaluations that are valuable to preserve but are **not automatically stable neurOS APIs or supported examples**.

Experiments may:

- depend on large or unusual datasets,
- require optional GPU/cloud dependencies,
- use rapidly evolving research methods,
- contain negative results or incomplete hypotheses,
- change without a deprecation period.

## Promotion path

Research should move into a stable package or `examples/` only after it has:

1. a clear contract with the neurOS or ORION architecture;
2. deterministic/reproducible configuration where feasible;
3. tests appropriate to the claim being made;
4. leakage-controlled scientific evaluation when relevant;
5. documented compute/data/artifact provenance;
6. a maintained owner/surface and a reason to provide compatibility guarantees.

ORION research should prefer the shared tokenizer/encoder/decoder interfaces and benchmark harness instead of inventing parallel evaluation machinery.

## Current areas

- `vision/dinov3/`: DINOv3 neuroscience/cell-tracking/atlas experiments and associated research papers.

Additional research areas should receive their own subdirectory rather than being added to the repository root or `notebooks/`.
