# Qualification Bundles

`neurOS` qualification is evidence generation, not a badge generator.

Version 1 deliberately qualifies one narrow property:

> A declared neurOS runtime configuration was executed, its exact `SignalFrame` inputs were recorded with integrity hashes, and the sealed recording reproduced the same canonical decoder-output digest through deterministic replay.

That is an **integration-level runtime/replay claim**. It is not automatically a real-dataset, hardware, closed-loop, or clinical claim.

## Create a bundle

```bash
neuros qualify configs/examples/mock_bci.yaml \
  --output qualification/mock \
  --session-id mock-reference \
  --duration 1.0
```

The command stages the bundle outside the final path, runs the live configuration, records the canonical input frames, verifies every recorded frame hash, replays the recording through the bundled configuration, compares the canonical decoder-output digest, seals every artifact with SHA-256, verifies the completed bundle, and only then publishes the final directory.

An existing destination is never replaced unless `--overwrite` is explicit.

## Reproduce a bundle

```bash
neuros reproduce qualification/mock
```

Reproduction happens in two stages:

1. every sealed file hash, artifact size, artifact-set membership, bundle digest, and embedded session-frame hash is verified;
2. only after integrity verification succeeds is the recorded session replayed through the bundled config and compared with the sealed decoder-output digest.

A modified file therefore fails before its contents can silently influence the reproduction result.

## Bundle layout

```text
qualification/
├── manifest.json
├── artifact_hashes.json
├── config.yaml
├── config.json
├── environment.json
├── compatibility.json
├── devices.json
├── clocks.json
├── model.json
├── runtime.json
├── decoder_outputs.json
└── session/
    ├── manifest.json
    ├── config.json
    └── streams/...
```

### `manifest.json`

The top-level authority for the qualification claim. It records:

- qualification schema version;
- bundle identity and creation time;
- Git identity when available;
- exact config-file SHA-256 and semantic config hash;
- embedded archive identity;
- record/replay integrity status;
- canonical decoder-output digest;
- the explicit claim boundary.

### `artifact_hashes.json`

Contains the byte size and SHA-256 of every other file in the bundle and a canonical digest over the complete artifact index. Unexpected files are considered a mutation of the sealed evidence bundle and make verification fail.

### `environment.json`

Captures Python/platform identity and installed versions of neurOS plus relevant neuroscience/runtime ecosystems when present.

### `compatibility.json`

Records compatibility-registry entries for recognized external runtime integrations actually selected by the config, such as BrainFlow, LSL, or Braindecode. It does not promote their evidence tier.

### `devices.json` and `clocks.json`

Extract exact stream-descriptor evidence from the recorded session. These files are useful inputs to future hardware qualification, but their presence does not imply hardware qualification.

### `model.json`

Records the decoder config and any `ModelArtifactManifest` entries bound to the recorded session. If no promoted model artifact is bound, the bundle says so explicitly and does **not** claim learned-weight identity.

### `runtime.json`

Contains both the record and replay runtime snapshots, including node failure counts, queue acceptance/drop statistics, and node latency summaries.

### `decoder_outputs.json`

Stores only the canonical output count/digest, not a second unbounded copy of every prediction. Version 1 uses SHA-256 over canonical JSONL serialization.

## Claim boundary

A successful v1 bundle sets:

```text
runtime_record_replay_qualified = true
real_dataset_qualified          = false
hardware_qualified              = false
closed_loop_qualified           = false
clinical_qualified              = false
```

This distinction is foundational. A software replay passing on a laptop does not demonstrate packet-loss behavior, device clock uncertainty, firmware correctness, wireless reliability, source-to-decision latency on named hardware, intervention safety, or clinical validity.

## Hardware qualification is the next evidence tier

A future named hardware profile should extend the same evidence object with measured fields such as:

- physical device and board ID;
- firmware and acquisition-library versions;
- operating system and transport;
- channel names/types/units and actual sampling rate;
- source timestamp and clock domain;
- measured clock offset/drift/uncertainty;
- packet/sample loss and neurOS queue loss;
- reconnect/recovery behavior;
- sustained run duration;
- source-to-decision p50/p95/p99 latency;
- exact recording/replay integrity.

Only a bundle containing those measurements under a named configuration should be eligible for `hardware_qualified = true`.

## Why this matters

The goal is to make neurOS results transportable as evidence. A collaborator should be able to receive a bundle and answer:

- What configuration ran?
- What exact neural frames entered the computational path?
- Which software and ecosystem versions were installed?
- Which device/timing metadata were actually observed?
- Did queues drop data or nodes fail?
- Was a promoted learned-weight artifact bound?
- Can the computational path reproduce the same outputs from the recorded neural stream?
- What stronger claims are explicitly unsupported?

That evidence contract is more valuable than a generic "pipeline completed successfully" flag and becomes the substrate for real-device, ORION, closed-loop, and eventually regulated qualification layers.
