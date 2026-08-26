# Developer Preview Journey

The neurOS developer preview is a **reproducible installed-product journey**, not a claim that every research package is production-ready.

The governing question is:

> Can a new developer install the public execution path from built wheels, inspect what is present, execute and replay a neural runtime, seal/reproduce evidence, inspect a decoder contract, stress the system in Arena, and extend it with an out-of-tree plugin without importing implementation code from the repository checkout?

The repository answers that question with the `Developer Preview Journey` CI workflow and `scripts/run_developer_preview_journey.py`.

## What the qualification environment contains

The canonical journey installs only built wheel artifacts for:

```text
neuros-core
neuros-drivers
neuros-models
neuros
neuros-arena
neuros-example-plugin   # deliberately outside the workspace
```

The orchestrator fails if any of those distributions is installed editable or resolves inside the repository checkout.

This is intentionally different from normal contributor setup. Contributors may use editable installs for development; developer-preview qualification may not.

## End-to-end path

The current journey executes:

```text
wheel install + pip check
        |
        v
neuros doctor / plugins / devices / compatibility
        |
        v
validate + run mock RuntimeGraph
        |
        v
record -> verify -> replay
        |
        v
qualify -> reproduce
        |
        +------> inspect eeg-conformer decoder card
        |
        +------> run dual-target Arena smoke world
        |
        +------> discover external wheel entry points
                        |
                        v
                  validate + run external YAML
```

Every command is a public console entry point. The Python orchestrator does not import internal runtime construction helpers to bypass those boundaries.

## Machine-readable evidence

A successful run writes:

```text
developer-preview/
  journey-report.json
  session/
  qualification/
  arena-report.json
```

`journey-report.json` uses schema:

```text
neuros.developer_preview_journey.v1
```

It records:

- active Python/platform identity;
- installed distribution names, versions, locations, and direct-install metadata;
- whether any package was editable;
- every command and argument vector;
- command elapsed time;
- captured stdout/stderr;
- parsed JSON where the public command exposes it;
- session, qualification, and Arena artifact locations;
- failure type/message when the path aborts.

This is onboarding evidence, not hardware or scientific validation.

## Run it locally

The script expects the required distributions to already be installed in the active environment. Until coordinated package publication is enabled, the easiest local developer setup remains the repository bootstrap described in [First 10 Minutes](first-10-minutes.md).

To reproduce the *wheel-qualified* path exactly, build/install the six artifacts above into a fresh environment and then run:

```bash
python scripts/run_developer_preview_journey.py \
  --repo-root . \
  --output /tmp/neuros-developer-preview \
  --duration 0.1
```

For ordinary exploratory development, editable installs are fine. The script rejects them because its purpose is specifically to detect package metadata, entry-point, missing-file, and install-boundary defects that editable monorepo tests can hide.

## What this proves

A green journey establishes that one exact neurOS revision can:

- be assembled from ordinary wheel distributions;
- report a healthy public SDK installation;
- discover built-in and external plugins;
- resolve versioned configuration into the native runtime;
- execute, record, integrity-check, and replay a session;
- create and independently reproduce a software qualification bundle;
- expose a decoder capability card without requiring model training;
- execute the deterministic BCI Arena;
- run an independently packaged source/transform through the same YAML/runtime path.

## What it does not prove

The journey does **not** establish:

- public PyPI availability;
- compatibility with every optional research dependency;
- physical device qualification;
- physiological realism of Arena;
- decoder accuracy on human data;
- ORION superiority;
- closed-loop safety;
- medical or clinical validity.

Those belong to stronger, separately governed evidence layers.

## Why the external plugin is part of the journey

A platform is not truly extensible if its own examples only work because they live inside the monorepo.

`neuros-example-plugin` is therefore built as an independent wheel and kept outside the root workspace. The developer journey verifies that its `example_sine` source and `example_gain` transform are discovered through package entry points and can drive an ordinary versioned neurOS YAML graph.

See [External Plugin Authoring](../PLUGIN_AUTHORING.md) for the extension contract.

## Relation to release qualification

The developer journey and release-candidate workflow answer different questions:

- **Release Candidate Artifacts** asks whether every maintained workspace distribution builds, validates, hashes, and installs consistently.
- **Developer Preview Journey** asks whether the minimal public developer path is coherent after installation.

Both should be green before presenting a revision as a developer-preview candidate. A wheel can be perfectly packaged while the user journey is broken; conversely, a source checkout can demonstrate a journey while hiding a broken wheel. neurOS tests both boundaries.
