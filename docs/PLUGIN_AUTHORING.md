# External Plugin Authoring

neurOS is designed to be extended **without editing the kernel**. A hardware vendor, lab, research group, or application team should be able to publish an ordinary Python distribution that contributes one or more entry points and exchanges only stable neurOS contracts.

The maintained reference package is [`examples/plugins/neuros-example-plugin`](../examples/plugins/neuros-example-plugin/README.md). It is intentionally excluded from the root workspace so CI proves that discovery works across a real distribution boundary.

## Extension groups

Current entry-point groups are:

```text
neuros.sources
neuros.transforms
neuros.tokenizers
neuros.encoders
neuros.decoders
neuros.sinks
neuros.monitors
neuros.world_models
```

Use the narrowest group that expresses the component's responsibility. A world-model plugin, for example, should not also take over Arena's display, device, transport, or application semantics merely because those layers are convenient to access.

## Minimal package metadata

A source/transform plugin can depend on `neuros-core` rather than the complete SDK:

```toml
[project]
name = "my-neuros-plugin"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "neuros-core>=2.0.0,<3.0.0",
]

[project.entry-points."neuros.sources"]
my_device = "my_plugin.source:MySource"

[project.entry-points."neuros.transforms"]
my_transform = "my_plugin.transforms:MyTransform"
```

The bounded `neuros-core` dependency is version negotiation. If your package has only been qualified against the 2.x contracts, do not publish an unbounded dependency that silently accepts an incompatible future major version.

## Source contract

A source is structural. Subclassing a neurOS base driver is not required.

```python
from collections.abc import AsyncIterator
from neuros.contracts import SignalFrame, StreamDescriptor

class MySource:
    @property
    def descriptor(self) -> StreamDescriptor:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def frames(self) -> AsyncIterator[SignalFrame]:
        ...
```

A high-quality source makes the following explicit:

- stable `stream_id`;
- modality and sample rate;
- channel names, channel types, and units;
- actual device/manufacturer identity where known;
- clock domain and which timestamps are genuinely available;
- sample-array geometry in metadata when multidimensional;
- quality flags for loss, clipping, saturation, disconnection, or suspected artifacts;
- acquisition-specific metadata/provenance needed for replay or qualification.

Do not synthesize a device timestamp or synchronized clock merely because downstream code can accept one.

## Transform contract

A transform exposes one method:

```python
class MyTransform:
    def transform(self, item):
        ...
```

When transforming a `SignalFrame`, preserve identity/timing/quality fields unless the operation has a documented reason to change them. Add provenance to metadata rather than replacing upstream metadata.

For explicit one-to-many emission, use the kernel's `TransformEmission` contract rather than returning an ambiguous list/tuple.

## Configuration path

Entry-point discovery and config resolution are the same path used by the CLI. Plugin constructor options come from the versioned YAML `options` mapping:

```yaml
streams:
  - id: eeg
    source:
      plugin: my_device
      options:
        serial_number: ABC123
        sample_rate: 250
    transforms:
      - plugin: my_transform
        options:
          parameter: 1.0
```

Constructor validation should fail early and clearly. neurOS wraps common plugin-construction failures with the plugin kind/name so a bad configuration is diagnosable before a long session begins.

## Dependency boundaries

Prefer this order:

1. `neuros-core` for contracts/runtime-facing plugins;
2. a specific neurOS package only when you actually require its higher-level API;
3. the full `neuros` SDK only when the plugin genuinely needs SDK composition.

Keep vendor SDKs, large ML frameworks, and optional scientific stacks in plugin-specific dependencies or extras. `neuros-core` should never need a new concrete hardware/model dependency merely because one external plugin uses it.

## Compatibility contract

For each release of an external plugin:

- declare bounded neurOS distribution ranges in package metadata;
- run `pip check` after installing the release wheel;
- test the oldest and newest Python versions you claim;
- test every neurOS major/minor range you claim when contract behavior differs;
- record the plugin distribution name/version in bug reports and qualification artifacts;
- fail closed when required optional dependencies or capabilities are missing;
- never silently substitute a different device/model/algorithm.

Python package resolution is the first line of version negotiation. Runtime discovery additionally records the distribution and version that supplied an entry point.

## Software contract vs qualification

A source plugin can have excellent software tests and still have no hardware evidence.

Keep the ladders separate:

```text
wheel/import/discovery
        -> source contract
        -> config/runtime integration
        -> replay/fault regression
        -> named hardware qualification
        -> human closed-loop evidence
```

A hardware qualification should identify at minimum the manufacturer/model, firmware, transport, host OS, plugin version, neurOS version, acquisition config, duration, packet/sample loss, clock behavior, reconnect behavior, and latency methodology.

Likewise, a world-model plugin can satisfy its software contract without establishing physiological truth. Synthetic and learned models need declared evidence boundaries and independent real-data/metamorphic tests appropriate to the claim.

## Clean-wheel reference gate

The repository's `External Plugin Contract` workflow deliberately:

1. builds the relevant neurOS distributions as wheels;
2. builds the example plugin as a separate wheel;
3. creates a fresh virtual environment;
4. installs the exact wheel set;
5. runs `pip check`;
6. verifies entry-point distribution/version metadata;
7. instantiates the source and transform through `PluginRegistry`;
8. runs the plugin's protocol tests;
9. validates and executes the example YAML through the installed `neuros` CLI.

This is stronger than an editable-install test because it catches missing package data, broken entry points, invalid dependency declarations, and accidental reliance on the repository checkout.

## Recommended external repository layout

```text
my-neuros-plugin/
  pyproject.toml
  README.md
  src/
    my_neuros_plugin/
      __init__.py
      source.py
  tests/
    test_contract.py
  examples/
    pipeline.yaml
  .github/workflows/
    ci.yml
```

Keep generated recordings, participant data, vendor binaries, model checkpoints, and qualification outputs out of source control unless they are deliberately small reviewed fixtures.

## When a kernel change is justified

Do not request a `neuros-core` change just because your plugin implementation is awkward. A kernel change is justified when multiple independent integrations expose a genuinely missing general contract, for example:

- a timestamp/clock semantic neurOS cannot express;
- a necessary loss/quality state absent from the canonical data contract;
- a reusable lifecycle or emission semantic needed by multiple plugin families;
- a compatibility capability that cannot be represented through package metadata or current descriptors.

That keeps the kernel conservative while allowing the ecosystem at the edges to move quickly.
