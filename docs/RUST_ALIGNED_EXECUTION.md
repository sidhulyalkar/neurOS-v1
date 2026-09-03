# Exact multimodal execution contract

## Status

This document describes the execution half of the neurOS v1 exact-clock runtime contract. It depends on the promoted source/content provenance layer and the promoted exact alignment planner. It does not define interpolation, resampling, drift correction, hemodynamic alignment, or modality-specific preprocessing.

## Authority chain

Aligned execution keeps three pre-existing identities distinct:

1. `dataset_content_sha256` binds verified source bytes plus record interpretation.
2. `manifest_sha256` binds the exact serialized dataset manifest, including acquisition and clock metadata.
3. `plan_sha256` binds the exact temporal plan, including the first two identities, `sync_group`, clocks, and derived frame arithmetic.

The executor does not create a fourth synchronization authority. It consumes the exact plan object supplied by the caller and carries `plan_sha256` on every emitted aligned window.

## API

```python
from neuros import Dataset

study = Dataset.open("study")
plan = study.plan_aligned(
    sync_group="sub-01/run-01",
    modalities=["fmri", "behavior"],
    duration_ns=4_000_000_000,
    stride_ns=2_000_000_000,
)

for batch in study.stream_aligned(plan, prefetch=8):
    fmri = batch.fmri
    behavior = batch.behavior
```

`batch.fmri` and `batch.behavior` are separate zero-copy Arrow views over the selected mmap-backed source intervals. neurOS does not concatenate the arrays or impose a shared tensor shape.

## No replanning during execution

Execution validates the stored plan directly. It checks:

- plan schema and exact policy;
- dataset ID, dataset-content identity, and exact manifest identity;
- canonical acquisition identity;
- canonical stable entry order;
- one selected record per modality;
- record subject/modality/source digest/offset/dtype/shape identity;
- clock ID/start/period identity;
- stored start-frame equation;
- stored frames-per-window equation;
- stored frame-stride equation;
- selected-record common overlap;
- stored window count;
- final-window frame bounds.

It does not solve clock congruences again. The planner remains the synchronization authority; the executor proves that the supplied object is internally consistent with the currently opened dataset and then follows its stored arithmetic.

## Fresh source verification at execution start

The v0 mmap cache records whether a mapping matched a declared source SHA-256 when that mapping was verified. That state is intentionally described as `verified_at_open`; it is not a permanent filesystem freshness guarantee.

Before accepting an aligned stream, the executor therefore performs a separate fresh whole-file SHA-256 pass over the currently resolved physical source files. This pass does not trust the mmap verification cache. Shared canonical paths with the same declared digest are hashed once; conflicting declarations reject deterministically.

Only after all current source bytes match the plan's dataset-content authority does aligned execution begin.

This is a point-in-time execution-start integrity check. It does not make a mutable filesystem immutable, prevent an external writer from racing after verification, provide remote attestation, or establish acquisition provenance beyond the declared/local evidence.

## Bounded zero-copy execution

The executor uses a bounded crossbeam channel with caller-controlled `prefetch >= 1`. The producer derives each modality's frame start only from the stored plan and opens that interval through the existing `WindowHandle` path. Dropping the consumer disconnects the channel and terminates production naturally.

Each `AlignedWindow` carries:

```text
plan_sha256
dataset_content_sha256
manifest_sha256
sync_group
window_index
start_ns
end_ns
per-modality DataWindow provenance
```

The per-modality `DataWindow` retains the source mapping through Arrow ownership exactly as in the promoted single-modality runtime.

## Scientific boundary

Exact temporal coincidence is a systems/runtime property, not a scientific claim that two signals are biologically equivalent, causally related, properly preprocessed, or suitable for a particular model. Aligned execution does not strengthen ORION `LineageCompleteness.UNKNOWN`, does not infer hemodynamic lag, and does not authorize experiment/model promotion.

Future interpolation or resampling layers must be explicit versioned policies that consume qualified exact authority rather than silently changing this contract.

## Promotion gates

The execution tranche must qualify on one exact PR head with:

- `cargo fmt --all --check`;
- Clippy with `-D warnings`;
- native unit/adversarial tests;
- built native wheel;
- exact-plan execution over a hashed multimodal fixture;
- independent Python recomputation of emitted time/frame intervals;
- exact Arrow value checks for every modality;
- mutation-after-plan rejection even with a live previously verified mmap;
- invalid-prefetch and consumer-cancellation behavior;
- regression of the promoted v0 zero-copy/provenance path;
- the complete repository workflow matrix applicable to the changed surface.

Promotion still requires a guarded merge followed by fresh exact-main push qualification. No PR-head evidence is inherited by the squash commit.
