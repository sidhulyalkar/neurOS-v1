# First 10 Minutes

The fastest way to understand neurOS is to make it produce an evidence artifact.

The starter path deliberately uses a deterministic mock neural stream and a training-free decoder. That keeps installation, runtime, recording, replay, and provenance failures visible instead of hiding them behind a model-training job.

## 1. Install the current developer preview

Until coordinated public package publishing is enabled and qualified, the repository checkout is the authoritative installation path:

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python scripts/bootstrap.py --profile bci --test-tools
```

A public release should eventually reduce this to a normal package install. Do not treat the monorepo bootstrap as the final onboarding design.

## 2. Create a clean project

```bash
neuros init my-neuros-project
cd my-neuros-project
```

The generated project contains only:

```text
my-neuros-project/
  neuros.yaml
  README.md
  .gitignore
```

It is intentionally not a framework generator. The point is to create the smallest configuration that exercises the maintained runtime and evidence boundary.

`neuros init` will not overwrite its managed starter files unless `--force` is supplied, and even then it preserves unrelated user files.

## 3. Inspect the environment

```bash
neuros doctor
neuros plugins
neuros devices
neuros compatibility
```

`compatibility` is an evidence registry, not a marketing banner. An integration is allowed to be experimental or planned; the command should tell you what has actually been exercised.

## 4. Validate and execute the runtime

```bash
neuros validate neuros.yaml
neuros run neuros.yaml --duration 2
```

This uses the same config resolution, plugin registry, bounded runtime graph, and decoder-output path used by maintained live and replay execution.

## 5. Seal a reproducible software qualification

```bash
neuros qualify neuros.yaml \
  --output evidence/qualification \
  --duration 1

neuros reproduce evidence/qualification
```

The qualification bundle binds the exact software path that ran, its configuration and environment identity, runtime evidence, recording integrity, decoder output evidence, file hashes, and a root artifact identity.

This establishes a **software/runtime evidence boundary only**. The mock source does not become real EEG. A software qualification is not hardware validation, decoder efficacy, closed-loop safety, or clinical evidence.

## 6. Record and replay explicitly

When exact session replay is the thing you care about:

```bash
neuros record neuros.yaml \
  --output sessions/example \
  --session-id example \
  --duration 2

neuros inspect sessions/example --verify
neuros replay sessions/example --config neuros.yaml
```

The canonical archive preserves sequence and timing identity, stream descriptors, quality metadata, configuration/provenance, runtime information, and per-frame integrity. NWB and Zarr remain interoperability exports rather than replacements for exact neurOS replay semantics.

## 7. Choose the lane that matches your actual problem

### I already have a model

Do not rewrite it in neurOS. Bring it through the external model/plugin and Neural System Qualification boundary so it can be evaluated under the same observation roles, preprocessing authority, calibration budget, score semantics, and failure preservation as competing methods.

Start with [Neural System Qualification](../NEURAL_SYSTEM_QUALIFICATION_V1.md) and [Plugin Authoring](../PLUGIN_AUTHORING.md).

### I already have MNE/MOABB/Braindecode code

Keep using those projects for the work they already do well. neurOS should add execution, replay, provenance, qualification, and claim authority around them rather than replace their preprocessing, dataset, or model ecosystems.

See [Ecosystem Compatibility](../COMPATIBILITY.md).

### I want to compare neural representations or reduce calibration

That is where ORION belongs. ORION is the intelligence plane for tokenization, representations, transfer, personalization, and governed adaptation. It should compete under the same external qualification authority rather than receive privileged benchmark access.

See [ORION Adaptation Authority](../ADAPTATION_AUTHORITY.md).

### I want to use physical EEG hardware

Start with a maintained BrainFlow, LSL, or device boundary. Do not describe a path as hardware-qualified until a named device, firmware, transport, host, timing, packet-loss, and replay configuration has physical evidence bound to a verified qualification artifact.

See [Hardware Qualification](../HARDWARE_QUALIFICATION.md).

## 8. Understand the evidence ladder

```text
software contract
      -> integration
      -> deterministic replay / scientific synthetic
      -> real dataset
      -> physical hardware
      -> closed loop
      -> clinical evidence
```

A result does not silently jump tiers because it looks convincing. See [Scientific Claims](../SCIENTIFIC_CLAIMS.md).

## If something fails

Include the following in an issue:

```bash
neuros doctor --json
neuros compatibility --json
python --version
```

Also include the exact Git commit, operating system, command, and smallest reproducible `neuros.yaml`. Never attach credentials or identifiable participant data to a public issue.

For the stricter installed-wheel qualification path, see [Developer Preview Journey](developer-preview.md).
