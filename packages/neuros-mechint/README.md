# neuros-mechint v1.0

**Reproducible causal experiments for testing how computation is implemented, learned, shared, and replicated across artificial and neural-data models.**

`neuros-mechint` is the mechanistic-evidence layer of neurOS. It can operate on ordinary PyTorch modules, ORION/NeuroFM representations, and optional TransformerLens, NNsight, SAELens, or circuit-tracer surfaces while keeping causal claims narrower than the tools used to discover them.

## The v1 evidence ladder

```text
known-answer falsification fixtures
→ causal interventions + matched controls
→ quantitative circuit faithfulness
→ frozen discovery / held-out validation
→ self-checking evidence packs
→ matched architecture × tokenizer contrasts
→ held-out causal feature correspondence
→ claim-aware hierarchical replication
→ dose response + explicit manifold assumptions
→ independent execution reproduction
→ explicit empirical evidence closure
```

The central v1 rule is simple:

> **Software readiness is not empirical neuroscience evidence.**

The package can certify its schema, migration, falsification, artifact-integrity, tutorial, and reproduction contracts in CI. It cannot certify a real neural mechanism until real evidence artifacts exist.

## Frozen schemas and migrations

`ExperimentManifest` schema v3 separates:

- `scientific_fingerprint`: deterministic identity of the frozen scientific design;
- `run_hash`: execution-specific identity including host/time/environment provenance.

Historical schema-v2 manifests can be upgraded with:

```python
from neuros_mechint import migrate_manifest_payload
migrated = migrate_manifest_payload(old_manifest)
```

The frozen artifact catalog covers:

```text
evidence_pack
factorial
correspondence
replication
dose_response
```

Inspect it with:

```bash
neuros-mechint schemas --json
```

v0.6-v0.9 artifact result schemas remain valid. v1 migration attaches canonical envelope contract metadata without rewriting the scientific result or changing its integrity hash.

## Independent reproduction

A reproduction should not require byte-identical floating-point output across hardware. It should require:

1. the same scientific fingerprint;
2. a distinct run hash and execution identity;
3. the same preregistered qualitative decision;
4. specified metrics within preregistered absolute/relative tolerances.

Use `ReproductionSpec`, `ReproductionSnapshot`, and `assess_independent_reproduction(...)` to make that comparison executable.

## Dose-response artifacts

v1 promotes dose-response studies to self-checking artifacts:

```python
write_dose_response_artifact(result, "dose-response.json")
```

Verify later with:

```bash
neuros-mechint verify-dose-response-artifact dose-response.json --json
```

Manifold assumptions remain explicit. “Empirical donor,” “conditional resample,” or “generative” is provenance, not automatic proof that an intervention is valid or biologically natural.

## Nine synthetic scientific gates

```bash
neuros-mechint ground-truth --json
neuros-mechint shared-computation-ground-truth --json
neuros-mechint mechanism-emergence-ground-truth --json
neuros-mechint circuit-faithfulness-ground-truth --json
neuros-mechint evidence-pack-generalization-ground-truth --json
neuros-mechint factorial-ground-truth --json
neuros-mechint correspondence-ground-truth --json
neuros-mechint replication-ground-truth --json
neuros-mechint v1-ground-truth --json
```

The v1 gate specifically checks manifest migration, legacy artifact migration, duplicate-run rejection, decision-flip rejection, schema completeness, and the anti-overclaim rule that real empirical evidence must remain pending until artifacts exist.

## Executed teaching evidence

Routine tests still validate every maintained notebook as notebook JSON. Release/evidence CI goes further and **executes** the CPU-safe lessons for known causal localization, circuit faithfulness, held-out rejection, hierarchical replication, and independent reproduction.

```bash
pip install -e "packages/neuros-mechint[dev,notebooks]"
python packages/neuros-mechint/scripts/execute_cpu_tutorials.py
```

## Real v1 evidence program

The recommended first empirical grid is deliberately small:

```text
Transformer × SSM
      crossed with
Event × Relative-ISI
      crossed with
>= 3 independent training seeds
```

For each cell, freeze neural-data revisions, semantic discovery/validation partitions, task metric, capacity/compute budget, tokenizer budget and temporal resolution. Produce v0.6 evidence packs, v0.7 matched contrasts, v0.8 held-out causal correspondences, and v0.9/v1 hierarchical replication artifacts. Then add dose-response and a stronger manifold-aware control.

Track what has actually been completed with:

```bash
neuros-mechint release-status --json
```

The repository intentionally reports the real neural evidence requirements as **pending** until those artifacts exist. See `docs/V1_EMPIRICAL_EVIDENCE_STATUS.md` and `experiments/mechint/v1_evidence/README.md`.

## Quick start

```bash
python scripts/bootstrap.py --profile kernel --test-tools
pip install -e "packages/neuros-mechint[dev]"
neuros-mechint v1-ground-truth --json
```

With ORION:

```bash
python scripts/bootstrap.py --profile orion --test-tools
pip install -e "packages/neuros-mechint[dev,orion]"
```

## Scientific claim boundary

Passing v1 software gates means the framework can express, serialize, migrate, falsify, compare, and independently reproduce the declared experiment contracts. It does **not** establish biological homology, universal neural meaning, architecture-general mechanisms, cross-dataset transfer, or a superior neural tokenization without the corresponding real held-out and independently replicated intervention evidence.
