# neurOS example external plugin

This directory is a **real Python distribution kept outside the neurOS workspace**. It exists to prove that a lab or company can extend neurOS without editing `neuros-core`, subclassing an internal driver, or becoming a monorepo package.

It contributes two entry points:

```text
neuros.sources:    example_sine
neuros.transforms: example_gain
```

The package depends only on the narrow stable contract surface it needs:

```toml
dependencies = [
  "neuros-core>=2.0.0,<3.0.0",
  "numpy>=1.24.0",
]
```

That version range is the plugin's compatibility declaration. Standard Python dependency resolution and `pip check` are the first compatibility gate. Do not depend on the full `neuros` SDK unless your plugin genuinely needs SDK-level functionality.

## Source contract

`SineSource` implements the public structural `neuros.contracts.Source` protocol directly:

- `descriptor` returns an explicit `StreamDescriptor`;
- `start()` and `stop()` own lifecycle;
- `frames()` yields canonical `SignalFrame` chunks;
- channel names/types/units and sample rate are explicit;
- array geometry is declared as `sample x channel`;
- deterministic samples are separated from host-monotonic receive timing;
- no fictional device or synchronized clock is invented.

A real hardware plugin should replace only the acquisition-specific internals. It should keep the same contract discipline and publish hardware qualification separately from software contract tests.

## Transform contract

`GainTransform` implements the public structural `Transform` protocol. For a `SignalFrame`, it uses dataclass replacement to preserve stream identity, sequence identity, timing, quality flags, and existing metadata while adding transform provenance.

This is intentionally boring mathematics. The purpose of the example is to teach a correct extension boundary, not to smuggle a scientific algorithm into a plugin tutorial.

## Local development

From this directory:

```bash
python -m pip install -e ".[test]"
pytest -q
```

When working from the neurOS repository, install a compatible `neuros-core` first or let your package manager resolve it.

## Prove entry-point discovery

After installation:

```python
from neuros.plugins import PluginKind, PluginRegistry

registry = PluginRegistry()
registry.discover()

source = registry.create(
    PluginKind.SOURCE,
    "example_sine",
    sampling_rate=250.0,
    channels=4,
)
transform = registry.create(PluginKind.TRANSFORM, "example_gain", gain=0.5)
```

The descriptors returned by `PluginRegistry` include the plugin distribution name and version, so debugging does not require guessing which installed wheel supplied an entry point.

## Prove config-first execution

The repository CI installs this package **from its built wheel** together with exact neurOS workspace wheels, then executes:

```bash
neuros plugins --json
neuros validate examples/plugins/neuros-example-plugin/example.yaml --json
neuros run examples/plugins/neuros-example-plugin/example.yaml --duration 0.08 --json
```

The YAML uses the external `example_sine` source and `example_gain` transform while using neurOS' existing threshold decoder. That demonstrates the full boundary:

```text
external wheel
    -> Python entry point
    -> PluginRegistry
    -> versioned YAML
    -> RuntimeGraph
    -> canonical SignalFrame
```

## Compatibility and failures

For a maintained plugin:

1. declare a bounded compatible `neuros-core` range in `Requires-Dist`;
2. keep optional hardware/model dependencies in plugin extras where possible;
3. validate constructor options and fail before starting acquisition;
4. run `pip check` in release CI;
5. test the oldest and newest supported Python versions;
6. test against the neurOS contract versions you claim to support;
7. publish hardware or scientific qualification as separate evidence.

If a future neurOS major release changes an incompatible contract, the dependency range should make installation fail clearly rather than letting a subtly incompatible plugin enter a live neural pipeline.

## Copying this template

Copy the directory into a separate repository, then change:

- project/distribution name;
- Python import package;
- entry-point names;
- compatible neurOS version range;
- source/transform implementations;
- tests and domain-specific qualification.

Do **not** add the copied plugin to the neurOS workspace simply to make discovery work. If the out-of-tree package cannot be installed and discovered independently, the extension contract is not doing its job.
