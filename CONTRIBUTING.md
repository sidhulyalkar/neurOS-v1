# Contributing to neurOS and ORION

Contributions should strengthen the stable runtime/representation boundaries rather than add new cross-package coupling. The repository contains both product-oriented infrastructure and active research, so **where a change lives is part of its API contract**.

## Development setup

Use a workspace profile rather than the historical root installation path:

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python -m venv .venv
source .venv/bin/activate
python scripts/bootstrap.py --profile bci --test-tools
```

Other profiles:

```bash
python scripts/bootstrap.py --profile kernel
python scripts/bootstrap.py --profile orion
python scripts/bootstrap.py --profile research
python scripts/bootstrap.py --profile all
```

Run the relevant tests before opening a PR. For broad kernel changes:

```bash
pytest tests/
python scripts/check_repo_hygiene.py
```

## Architecture rule

Dependency direction should remain:

```text
neural contracts
      <- runtime / config / recording / quality
      <- drivers / conventional models / SDK
      <- ORION and research packages
```

A kernel package must not import a research implementation. New hardware, transforms, tokenizers, encoders, decoders, sinks, and monitors should normally enter through plugin interfaces.

## Where changes belong

### `packages/neuros-core`

Only functionality that must be shared across nearly all BCI systems:

- neural data contracts,
- runtime graph/execution semantics,
- clock/synchronization primitives,
- configuration contracts,
- persistent replay semantics,
- generic scientific/runtime quality infrastructure,
- plugin discovery abstractions.

Avoid hardware SDKs, task-specific models, cloud vendors, large deep-learning frameworks, or research algorithms here.

### `packages/neuros-drivers`

Maintained source integrations and simulated/data sources. Optional device dependencies belong in extras or external plugin distributions.

### `packages/neuros-models`

Conventional task-specific decoder implementations and adapters. Decoders should return `DecoderOutput`; unavailable confidence/uncertainty stays unavailable.

### `packages/orion`

Neural tokenization, representation, adaptation, and neural-intelligence interfaces/implementations that have a common evaluation path. New ORION methods require leakage-controlled comparative evidence.

### Research packages and `experiments/`

Exploratory work belongs here first. Research code can move quickly and does not automatically receive compatibility guarantees.

### `examples/`

Only executable examples intended as supported user surfaces. **Promotion to `examples/` requires a CI smoke test.**

### `tutorials/`

Maintained educational material. Tutorials should use current APIs and should be reviewed when those APIs change.

### `notebooks/`

Transitional BCI notebooks that have not yet been promoted to supported examples/tutorials. Do not add broad exploratory research here; use `experiments/`.

### `docs/archive/`

Historical material only. Session summaries, migration reports, completion notes, and superseded plans belong here. Archived documents are retained for provenance and are not sources of current API/performance truth.

## Plugin interfaces

Entry-point groups include:

```text
neuros.sources
neuros.transforms
neuros.tokenizers
neuros.encoders
neuros.decoders
neuros.sinks
neuros.monitors
```

A plugin should:

1. implement the structural contract expected by its group;
2. avoid importing optional dependencies until the plugin is actually constructed when practical;
3. expose deterministic configuration options;
4. fail with a useful typed error when prerequisites are absent;
5. include a contract test;
6. include real-hardware qualification evidence separately if hardware support is claimed.

## Runtime changes

Changes to `RuntimeGraph`, `RuntimeExecutor`, queue policies, clocks, fusion, or lifecycle are foundational. They should include tests for the relevant combinations of:

- finite and live sources,
- cancellation/draining,
- failure propagation,
- backpressure/overflow,
- deterministic replay,
- multimodal ordering/timing,
- latency telemetry.

Do not hide dropped data or runtime failures.

## Recording changes

Recording changes must preserve or explicitly migrate:

- sequence identity,
- device/host/synchronized timestamps,
- clock domain,
- quality flags,
- stream descriptors,
- frame metadata/provenance,
- integrity verification.

NWB/Zarr interoperability should not weaken the canonical lossless replay contract.

## Scientific and ORION changes

Scientific comparisons must separate fit/train/adaptation/evaluation data. A new tokenizer or representation should include appropriate controls and report more than one convenient metric.

At minimum consider:

- compression/token budget,
- entropy/codebook utilization,
- behavior/motif/task decoding,
- neural prediction where applicable,
- cross-session transfer,
- jitter/unit/channel dropout robustness,
- runtime and memory,
- reproducibility seed/config/data/artifact manifests.

Do not describe synthetic, software-only, or public-dataset evidence as hardware or clinical validation.

## Tests and evidence tiers

Use the weakest accurate evidence label:

1. unit,
2. contract,
3. integration,
4. replay,
5. scientific synthetic,
6. real dataset,
7. hardware qualification,
8. closed-loop qualification,
9. clinical evidence.

A passing unit test is not evidence for a device latency claim. A hardware smoke test is not clinical validation.

## Pull requests

Keep PRs reviewable and intentional:

- one architectural responsibility per PR;
- no unrelated formatting churn;
- update tests and current docs in the same PR;
- use draft PRs for foundational changes until CI is green;
- for stacked PRs, state the base PR and merge order in the body;
- do not commit generated coverage, build outputs, local recordings, benchmark reports, or model artifacts unless the artifact is itself a reviewed research fixture.

For a foundational PR, describe:

- what contract changes,
- why the existing abstraction was insufficient,
- compatibility/migration behavior,
- scientific/runtime risks,
- tests and evidence used,
- known limitations and the next layer.

## Repository hygiene

The repository deliberately keeps the root quiet. Do not reintroduce:

- root `SESSION_SUMMARY*.md` files,
- root cleanup/completion reports,
- root `setup.py`,
- tracked `.coverage`,
- migration-only scripts in active `scripts/`,
- DINO/research notebooks under `notebooks/`.

Run:

```bash
python scripts/check_repo_hygiene.py
```

before submitting structural changes.

## Security, privacy, and safety

Do not commit raw participant data, secrets, credentials, or personally identifying neural/behavioral recordings. Prefer pseudonymous session/subject identifiers and explicit data-governance documentation.

Closed-loop or stimulation-related contributions must make action boundaries, quality/confidence gating, failure states, and stop semantics explicit. Software architecture should support safe research practice without making unsupported medical claims.
