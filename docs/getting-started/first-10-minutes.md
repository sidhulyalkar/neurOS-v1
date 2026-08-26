# First 10 Minutes

This path is designed to answer one question quickly: **does the neurOS execution and evidence stack work correctly on this machine before I connect real hardware or train a model?**

It deliberately uses the mock source and a training-free decoder so installation, runtime, recording, replay, and provenance failures cannot hide behind model fitting.

## 1. Create an isolated environment

```bash
git clone https://github.com/sidhulyalkar/neurOS-v1.git
cd neurOS-v1
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python scripts/bootstrap.py --profile bci --test-tools
```

The repository is currently the authoritative development install until coordinated public package publishing is enabled and qualified.

## 2. Inspect what is actually installed

```bash
neuros doctor --json
neuros plugins --json
neuros devices --json
neuros compatibility --json
```

`compatibility` is an evidence registry, not a package-detection banner. A planned integration has no qualification tier. An experimental integration states exactly which upstream surface has executable evidence.

Useful focused checks include:

```bash
neuros compatibility mne --json
neuros compatibility braindecode --json
neuros compatibility snap --json
neuros compatibility ngclearn --json
```

## 3. Validate and run a complete graph

```bash
neuros validate configs/examples/mock_bci.yaml --json
neuros run configs/examples/mock_bci.yaml --duration 2 --json
```

This exercises the same `SignalFrame -> RuntimeGraph -> DecoderOutput` path used by maintained live/replay execution. Runtime edges have bounded queues and explicit overload policy rather than unmeasured buffering.

## 4. Prove that input can be replayed

```bash
SESSION=/tmp/neuros-first-session

neuros record configs/examples/mock_bci.yaml \
  --output "${SESSION}" \
  --session-id first-10-minutes \
  --duration 2 \
  --json

neuros inspect "${SESSION}" --verify --json

neuros replay "${SESSION}" \
  --config configs/examples/mock_bci.yaml \
  --json
```

The canonical archive preserves stream/sequence identity, timing domains, quality metadata, configuration/provenance, runtime information, and frame integrity hashes. NWB/Zarr are interoperability exports rather than replacements for exact neurOS replay semantics.

## 5. Produce a reproducible qualification bundle

```bash
QUAL=/tmp/neuros-qualification

neuros qualify configs/examples/mock_bci.yaml \
  --output "${QUAL}" \
  --duration 1.0

neuros reproduce "${QUAL}"
```

The resulting bundle seals configuration, environment, compatibility, runtime, device/clock/model metadata, decoder output evidence, session data, file hashes, and a root artifact identity.

This proves a **runtime record/replay software qualification boundary**. It does not turn a mock source into physical hardware evidence.

## 6. Know the claim boundary

The public evidence ladder is:

```text
software contract
      -> integration
      -> replay / scientific synthetic
      -> real dataset
      -> hardware
      -> closed loop
      -> clinical
```

See [Scientific Claims](../SCIENTIFIC_CLAIMS.md) before interpreting a benchmark or compatibility label. The important discipline is simple: stronger language requires stronger evidence.

## 7. Choose the next path

### Existing MNE data

Install the optional bridge and convert an existing `Raw` object without hidden resampling or channel reordering:

```bash
pip install -e "packages/neuros[interop-mne]"
```

See [Ecosystem Compatibility](../COMPATIBILITY.md).

### Model benchmarking

Use the neural-window/Braindecode and longitudinal EEG evidence paths when the question is decoder or representation performance. Keep subject/session/device evaluation authority explicit rather than randomizing trials across the deployment unit you intend to generalize to.

### NeuroAI / representation research

See [NeuroAI Ecosystem](../NEUROAI_ECOSYSTEM.md) for SNAP-derived spectral evidence and the optional ngc-learn bridge. Their current evidence tiers are intentionally narrower than their broader research ecosystems.

### ORION

```bash
python scripts/bootstrap.py --profile orion --test-tools
python scripts/orion/run_tokenizer_benchmark.py \
  configs/orion/tokenization_smoke.yaml \
  --output /tmp/orion-tokenization
```

Synthetic tokenization evidence is a falsification surface, not proof of superior real-human transfer. ORION promotion requires leakage-controlled, deployment-unit-disjoint evidence.

### Physical hardware

Start from a maintained driver/BrainFlow/LSL boundary, but do not describe it as hardware-qualified until a named device + firmware + transport + host configuration has physical evidence bound to a verified qualification bundle.

## A useful issue report

If any step above fails, capture:

```bash
neuros doctor --json
neuros compatibility --json
python --version
```

and include the exact Git commit, operating system, failing command, and smallest reproducible configuration. Do not attach credentials or identifiable participant data to a public issue.

The full contributor/development path is documented in [Installation](installation.md) and the repository `SUPPORT.md`.
