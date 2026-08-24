# Contributing to neurOS and ORION

Contributions should strengthen stable runtime, representation, and evidence boundaries rather than add new cross-package coupling. The repository contains product-oriented infrastructure and active research, so **where a change lives is part of its API contract**.

Before making broad changes, read:

- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for current package maturity;
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for dependency direction and ownership;
- [`ROADMAP.md`](ROADMAP.md) for active qualification priorities.

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
      <- drivers / task models / SDK
      <- ORION / foundation / transfer / mechanism integrations
      <- experiments and studies
```

A kernel package must not import a research implementation. New hardware, transforms, tokenizers, encoders, decoders, sinks, and monitors should normally enter through plugin interfaces. Model/research ecosystems should use narrow adapters rather than making the runtime aware of their internals.

## Where changes belong

### `packages/neuros-core`

Only functionality that must be shared across nearly all BCI systems:

- neural data contracts;
- runtime graph/execution semantics;
- clock/synchronization primitives;
- configuration contracts;
- persistent replay semantics;
- generic scientific/runtime quality infrastructure;
- plugin discovery abstractions.

Avoid hardware SDKs, task-specific models, cloud vendors, large deep-learning frameworks, or research algorithms here.

### `packages/neuros-drivers`

Maintained source integrations and simulated/data sources. Optional device dependencies belong in extras or external plugin distributions.

A driver PR claiming hardware support should separate software contract tests from device qualification. Qualification evidence must identify the exact device, firmware, transport, OS, plugin revision, and protocol used.

### `packages/neuros-models`

Task-specific decoder implementations and model-side analysis contracts.

A new or changed decoder should:

1. execute the algorithm named by its public class/card;
2. fail clearly when a required backend is unavailable rather than silently substituting another architecture;
3. define its input axes/shape and preprocessing assumptions;
4. return honest `DecoderOutput` capabilities;
5. expose `encode(...)` only when the representation is well-defined;
6. provide an `InterpretabilityManifest` if it is promoted as mechanistically inspectable;
7. include regression tests for architecture identity and declared analysis paths.

Attention maps, saliency, or sparse features must not be presented as causal mechanisms without intervention evidence.

### `packages/neuros-foundation`

External neural foundation-model discovery, adapters, representation probes, and fair benchmark protocols.

A foundation integration should:

- separate catalog metadata from locally runnable adapter status;
- record upstream provenance/version/revision when available;
- fail closed when a checkpoint, package, or capability is unavailable;
- never use a different model as a silent stand-in;
- make subject/session/site/device evaluation splits explicit;
- distinguish upstream reported results from neurOS-reproduced evidence.

Legacy wrappers may remain for compatibility, but new scientific comparisons should use verified registry/adaptor paths.

### `packages/neuros-sourceweigher`

Reliability-aware source/domain selection and fusion.

New strategies should report more than normalized weights. Include appropriate stability, perturbation, effective-sample-size, held-out utility, or shift diagnostics. Distinguish domain similarity, predictive utility, signal quality, timing reliability, and calibrated uncertainty rather than collapsing them into an ambiguous trust score.

Keep the numerical core dependency-light. HTTP/service deployment remains optional.

### `packages/neuros-mechint`

Causal mechanism experiments and reproducible evidence contracts.

Changes should preserve the package distinction between:

- software contract readiness;
- candidate discovery;
- causal intervention evidence;
- held-out validation;
- cross-deployment-unit stability;
- empirical neuroscience/biological claims.

A new adapter establishes an execution boundary, not truth about a discovered mechanism. Immutable model/data/config/artifact identity and held-out evidence are preferred over notebook-only conclusions.

### `packages/orion`

Neural tokenization, representation, adaptation, and neural-intelligence contracts/implementations that have a common evaluation path. New ORION methods require leakage-controlled comparative evidence.

Fit, representation training, adaptation, and final evaluation partitions should remain explicit. New tokenizers or representations should be compared against simpler controls under matched downstream capacity and compute where practical.

### `packages/neuros-neurofm` and research packages

Experimental neural foundation-model and related research belongs here before promotion. Research code may move quickly and does not automatically receive compatibility guarantees.

A NeuroFM implementation moves behind a promoted ORION interface only after it satisfies the relevant contract, artifact, leakage-control, robustness, and comparative-evidence requirements.

### `packages/neuros-ui` and `packages/neuros-cloud`

Optional UI/API/distributed integration surfaces. These packages should consume the same config/runtime/event contracts as local execution rather than creating alternate orchestration semantics.

Provider- or UI-specific claims require their own tests. Package version numbers alone do not imply kernel-level qualification.

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

External plugins should not require edits to `neuros-core` unless they reveal a genuinely missing general contract.

## Runtime changes

Changes to `RuntimeGraph`, `RuntimeExecutor`, queue policies, clocks, fusion, or lifecycle are foundational. They should include tests for relevant combinations of:

- finite and live sources;
- cancellation/draining;
- failure propagation;
- backpressure/overflow;
- deterministic replay;
- multimodal ordering/timing;
- latency telemetry.

Do not hide dropped data or runtime failures.

## Recording changes

Recording changes must preserve or explicitly migrate:

- sequence identity;
- device/host/synchronized timestamps;
- clock domain;
- quality flags;
- stream descriptors;
- frame metadata/provenance;
- integrity verification.

NWB/Zarr interoperability should not weaken the canonical lossless replay contract.

## Scientific and ORION changes

Scientific comparisons must separate fit/train/adaptation/discovery/evaluation data where relevant. A new tokenizer, representation, transfer method, or mechanistic claim should include appropriate controls and report more than one convenient metric.

At minimum consider:

- compression/token budget;
- entropy/codebook utilization;
- behavior/motif/task decoding;
- neural prediction where applicable;
- subject/session/site/device transfer;
- few-shot calibration cost;
- jitter/unit/channel dropout robustness;
- artifact/montage/session-drift sensitivity;
- runtime and memory;
- representation geometry/domain leakage;
- mechanism intervention/faithfulness where claimed;
- reproducibility seed/config/data/artifact manifests.

Do not describe synthetic, software-only, or public-dataset evidence as hardware or clinical validation.

## Tests and evidence tiers

Use the weakest accurate evidence label:

1. unit;
2. contract;
3. integration;
4. replay;
5. scientific synthetic;
6. real dataset;
7. hardware qualification;
8. closed-loop qualification;
9. clinical evidence.

A passing unit test is not evidence for a device latency claim. A hardware smoke test is not clinical validation. A model-level causal intervention is not automatically a biological mechanism.

## Pull requests

The repository includes `.github/pull_request_template.md` to keep architectural and evidence claims reviewable.

Keep PRs intentional:

- one primary architectural responsibility per PR;
- no unrelated formatting churn;
- update tests and current docs in the same PR;
- use draft PRs for foundational changes until CI is green;
- for stacked PRs, state the base PR and merge order in the body;
- use expected-head guards for consequential merges where possible;
- do not commit generated coverage, build outputs, local recordings, benchmark reports, or model artifacts unless the artifact is itself a reviewed research fixture.

For a foundational PR, describe:

- what contract changes;
- why the existing abstraction was insufficient;
- compatibility/migration behavior;
- scientific/runtime risks;
- tests and evidence used;
- the strongest accurate evidence tier;
- known limitations and the next layer.

## Repository hygiene

The repository deliberately keeps the root quiet. Do not reintroduce:

- root `SESSION_SUMMARY*.md` files;
- root cleanup/completion reports;
- root `setup.py`;
- tracked `.coverage`;
- migration-only scripts in active `scripts/`;
- DINO/research notebooks under `notebooks/`.

Run:

```bash
python scripts/check_repo_hygiene.py
```

before submitting structural changes.

## Security, privacy, and safety

Do not commit raw participant data, secrets, credentials, or personally identifying neural/behavioral recordings. Prefer pseudonymous session/subject identifiers and explicit data-governance documentation.

Closed-loop or stimulation-related contributions must make action boundaries, quality/confidence gating, failure states, and stop semantics explicit. Software architecture should support safe research practice without making unsupported medical claims.
