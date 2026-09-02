# neurOS Validation Program

neurOS is not considered validated because one test suite is green. Different
claims require different evidence, and stronger evidence must not be inferred
from weaker evidence.

The validation program therefore treats software correctness, reproducibility,
scientific validity, hardware validity, and deployment validity as separate
promotion layers.

## Validation principle

A useful neurOS result should answer four questions independently:

1. **Did the software behave according to its contracts?**
2. **Can the exact computational result be reproduced from identified inputs and artifacts?**
3. **Does the study design support the scientific claim being made?**
4. **Does the evidence cover the real hardware, users, and deployment conditions named by the claim?**

Passing one layer never automatically promotes the next.

## Evidence ladder

| Layer | Primary question | Representative evidence | What it does **not** prove |
| --- | --- | --- | --- |
| Contract/unit | Are local invariants correct? | typed contract tests, deterministic identities, parser/config tests | installed-package usability, integration, neural efficacy |
| Property/adversarial | Do invariants survive generated edge cases and corruption? | Hypothesis-generated signals, invalid metadata, byte tampering | real-world performance |
| Clean-room product | Can a user install and use the default product from exact wheels? | wheel hashes, fresh venv, public CLI journey | non-default research packages, hardware, efficacy |
| Workspace integrity | Can every maintained distribution build without packaging collisions? | all workspace wheels, `twine check`, wheel ownership | that every package belongs in the default release |
| Runtime resilience | Does execution stop, recover, and fail closed under stress? | repeated lifecycle runs, queue/failure tests, cancellation tests | biological validity |
| Recording/replay | Are recorded observations integrity-bound and replayable? | payload hashes, descriptor fingerprints, semantic replay identity | device clock accuracy or signal quality |
| Qualification bundle | Does a sealed computational artifact reproduce exactly? | exact artifact set, bundle SHA-256, externally pinned root, replay digest | real-dataset, hardware, closed-loop, safety, clinical claims |
| Interoperability | Do integrations preserve neurOS contracts at ecosystem boundaries? | MNE/Braindecode/MOABB/BrainFlow/LSL/NWB/Zarr compatibility evidence | general scientific efficacy |
| Scientific qualification | Does the protocol support a specific neural-system claim? | subject-disjoint NSQ, calibration authority, untouched final assessment, failure preservation | hardware safety or clinical benefit |
| Real-data external floor | Does the method survive a frozen public neural dataset under the same referee? | promoted Kumar2024 authority and independently audited execution artifacts | broad population generalization |
| Hardware qualification | Does measured physical hardware meet explicit gates? | clock, loss, latency, reliability and device evidence | neural efficacy unless paired with a qualified study |
| Independent reproduction | Can another environment/team reproduce the evidence without repository-local assumptions? | release artifacts, hashes, documented commands, external rerun | universal validity |

## Whole Package Qualification

`.github/workflows/full-package-qualification.yml` is the umbrella software gate.
It intentionally validates both the default product and the larger development
workspace without conflating them.

### Default product matrix

The default public runtime is installed from exact wheels in fresh virtual
environments across:

- Ubuntu 24.04 x86-64;
- Windows 2025 x86-64;
- macOS 14 ARM64;
- Python 3.10, 3.11, and 3.12 on each operating system.

Every matrix job builds the release-policy-selected wheels and requires the
installed artifacts to resolve outside the repository checkout. The installed
wheel SHA-256 identities are checked against the wheels produced by that exact
source revision.

The clean-room path exercises:

- `neuros doctor`;
- compatibility and plugin discovery;
- `neuros init` and overwrite refusal;
- config validation and invalid-plugin rejection;
- runtime execution and lifecycle cleanup;
- recording, verified inspection, and replay;
- recording overwrite refusal;
- qualification and repeated reproduction;
- externally pinned qualification-root verification;
- incorrect external-root rejection;
- qualification overwrite refusal;
- bounded repeated runtime lifecycle/soak execution.

The runtime gate requires stopped execution, zero node failures, zero dropped
edges, and non-zero processed work for the maintained deterministic starter.
Timing fields are observed but are not incorrectly treated as deterministic
semantic output identity.

### Direct corruption attacks

The qualification does not merely ask whether valid artifacts work.

For a recorded session it copies the archive, flips one byte in an actual neural
`.npy` frame payload, and requires both:

```text
neuros inspect <archive> --verify
neuros replay <archive> --config <config>
```

to reject the corrupted archive.

For a sealed qualification bundle it mutates a checksum-bound artifact and
requires both normal and externally pinned `neuros reproduce` paths to reject the
bundle.

This directly tests the integrity boundary rather than mocking the verifier.

### Property-based contracts

`validation_tests/test_signal_contract_properties.py` generates finite neural
arrays and metadata combinations with Hypothesis. It checks:

- immutable, caller-detached `SignalFrame` sample ownership;
- exact archive data/dtype round-trip;
- descriptor fingerprint stability under mapping-order changes;
- rejection of non-finite/non-positive declared sampling rates;
- rejection of descriptor/frame channel-geometry contradictions;
- payload hash failure after direct byte mutation.

These tests intentionally live outside the ordinary unit-test directory because
property-test dependencies are validation tooling, not runtime dependencies.

### Workspace wheel audit

Every maintained workspace distribution is also built on the umbrella workflow.
The audit runs package metadata validation and checks that shared `neuros`
namespace portions do not own overlapping installed files.

A package passing this audit means it is buildable and packaging-compatible. It
does **not** mean the package is automatically part of the default neurOS
release or has earned a scientific claim.

## Required future validation

Whole Package Qualification is deliberately not the final validation state.
The strongest remaining gaps are:

### Exact installed-artifact provenance in ordinary qualification

The v1 runtime qualification bundle records package versions, but versions alone
are weaker than exact wheel/container identities. A later qualification schema
should bind the installed distribution artifact digest or a stronger environment
authority rather than assuming that equal version strings imply equal code.

### Runtime fault injection

Add deterministic tests for:

- source failure during startup and mid-stream;
- decoder/transform exceptions;
- cancellation during recording and qualification staging;
- queue saturation and each overflow policy;
- interrupted writes and orphan temporary files;
- repeated start/stop and task cleanup;
- resource growth over longer bounded runs.

Failures must be preserved as evidence, not converted into successful numerical
rows.

### Longer resource soak

Short CI lifecycle loops catch cleanup regressions but not slow leaks. Release
qualification should eventually include a longer scheduled soak measuring:

- resident-memory growth;
- open file descriptors/handles;
- task/thread counts;
- queue depth;
- frame/decoder throughput;
- latency distributions;
- data loss.

Thresholds should be established from measured stable baselines, not invented to
make CI green.

### External clean-room reproduction

At least one release candidate should be reproduced from downloadable artifacts
without using a local repository checkout for package code. The reproduction
should verify wheel/container digests, execute the public starter workflow, and
compare evidence roots.

### Real neural data

Software validation cannot establish neural decoding efficacy. Promoted neural
claims require the frozen NSQ authority, subject-disjoint evaluation, declared
calibration budgets, untouched final assessment, model-state identity, and
failure-preserving evidence.

The Kumar2024 promoted study is the current external-floor program. ORION must
remain outside favorable comparison until the external floor and execution
transport are fully qualified.

### Physical hardware

Synthetic drivers, replay, and Arena are useful systems tests but are not device
qualification. Physical BCI hardware claims require measured acquisition loss,
clock behavior, drift/uncertainty, end-to-end latency, reconnect/recovery,
long-duration stability, montage/electrode behavior where applicable, and an
explicit hardware evidence authority.

## Promotion rule

A release or scientific claim should name the strongest evidence layer actually
completed and preserve the lower layers that support it.

The target is not "100% tested." The target is a traceable chain in which every
important claim has an appropriate adversary and every stronger promotion
requires evidence that the weaker layer could not provide.
