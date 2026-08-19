# Historical package notebooks and examples

This directory contains the original `neuros-mechint` research examples. They are retained for project and scientific provenance, but presence here does **not** imply Stable API status or current execution coverage.

Maintained teaching material lives at:

```text
tutorials/mechint/
```

The current maintained progression ends with:

```text
08_held_out_evidence_packs.ipynb
09_factorial_architecture_tokenizer.ipynb
```

Real research artifacts belong under:

```text
experiments/mechint/evidence_packs/
experiments/mechint/factorial_studies/
```

Before relying on a historical example:

1. check the method card and maturity status;
2. verify imports against the current API;
3. identify the strongest evidence tier actually produced;
4. add controls and held-out validation appropriate to the claim;
5. for comparative architecture/tokenizer claims, run the v0.7 estimability audit rather than comparing unmatched results manually.

New exploratory work should normally live under `experiments/`; promote it into `tutorials/` only when it is maintained against current APIs and its scientific claim boundary is explicit.
