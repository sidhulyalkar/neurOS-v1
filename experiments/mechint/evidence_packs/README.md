# Mechanistic evidence-pack experiments

This directory is the repository home for **real execution outputs and run notes** produced by `neuros-mechint` evidence-pack studies.

The maintained scientific API and teaching material live elsewhere:

```text
packages/neuros-mechint/       library + schemas + tests
tutorials/mechint/             maintained CPU teaching track
experiments/mechint/evidence_packs/   real experiment manifests/artifacts/run notes
```

## Rules

1. Do not place model weights, private datasets, raw neural recordings, secrets, or raw prompt corpora here.
2. Evidence-pack JSON artifacts intentionally contain hashes/metadata rather than raw model inputs.
3. Resolve mutable model/tokenizer/SAE/transcoder aliases to immutable revisions before treating an artifact as publication-ready.
4. Retain negative results. `promotion.passed == false` is not a reason to delete a run.
5. Never overwrite an existing scientific artifact. New execution provenance should produce a new run record.
6. Verify copied artifacts with:

```bash
neuros-mechint verify-evidence-artifact path/to/evidence.json --json
```

## Suggested layout

```text
experiments/mechint/evidence_packs/
  <recipe-or-study-id>/
    README.md
    artifacts/
      <study-fingerprint-prefix>_<run-hash-prefix>.json
    notes/
      <date>_<run-hash-prefix>.md
```

A study README should record information that is intentionally external to the machine artifact, such as:

- scientific question;
- preregistered primary contrast;
- source dataset access instructions;
- why the discovery/validation split is defensible;
- model/tokenizer/SAE/transcoder source URLs;
- exact resolved revisions;
- hardware used;
- deviations from the maintained recipe;
- interpretation of negative or invalid cases.

## Starting points

List maintained external-model recipes with:

```bash
neuros-mechint evidence-recipes
```

The v0.6 recipes cover TransformerLens, NNsight, SAELens, and circuit-tracer. A recipe is a starting configuration, not measured evidence.
