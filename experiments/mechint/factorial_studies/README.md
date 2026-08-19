# Factorial mechanism study artifacts

This directory is the repository home for v0.7 architecture × tokenizer studies.

Each observed factorial cell should reference a completed v0.6 evidence-pack artifact. Comparative artifacts belong here only after the full study design, missing cells, semantic partition IDs, matched covariates, and preregistered contrasts are recorded.

Recommended layout:

```text
factorial_studies/
  <study-id>/
    DESIGN.md
    study.json
    cells/
      <cell-id>.evidence.json
    NOTES.md
```

## Before running a primary contrast

Record:

- architecture and tokenizer levels;
- model/tokenizer/dataset revisions;
- training seed and checkpoint;
- checkpoint maturity definition;
- semantic discovery and validation partition IDs;
- task metric and tolerance;
- intervention target universe;
- v0.6 discovery and faithfulness policies;
- token budget;
- temporal resolution;
- downstream capacity;
- training compute;
- any additional matched covariates;
- every intended cell, including unavailable cells and reasons;
- preregistered primary contrasts and replication groups.

## Results policy

Keep:

- positive interactions;
- null interactions;
- negative evidence-pack cells;
- non-estimable contrasts;
- missing cells;
- failed replication groups.

Do not delete a factorial run because a desired tokenizer effect disappeared after task-performance matching or because the estimability audit found a confound. Those are scientifically useful outcomes.

## First recommended real grid

Begin with a tractable design rather than every tokenizer family:

```text
2 architectures
x 2 tokenizers
x 2 sessions
```

A useful candidate is:

```text
Transformer vs SSM
x event tokens vs relative-ISI tokens
```

If compute permits, duplicate the grid over at least two independently trained seeds before interpreting architecture-level effects strongly.

After this design is operationally clean, extend tokenizer levels to binned counts, burst tokens, synchrony/population packets, VQ motifs, and assembly tokens.

The goal is to locate **mechanistic invariances and interactions**, not manufacture a single tokenizer leaderboard.
