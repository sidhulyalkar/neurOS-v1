# Exact multimodal alignment v1

neurOS treats temporal correspondence as scientific authority, not as array plumbing.

The v1 contract introduced by issue #147 adds a **provenance-bound, exact-only planning layer** on top of the promoted Rust data plane and content-verification contract from #148. It deliberately stops before multimodal execution. A plan must first prove both which bytes are being interpreted and that every requested window boundary is exactly representable on every selected modality clock.

## Authority stack

Exact alignment keeps three identities separate.

### Dataset content identity

`dataset_content_sha256` is the verified `neuros.dataset_content.v1` identity promoted by #148. It binds stable record IDs, full-file source SHA-256 values, byte offsets, dtype, and shape. It intentionally excludes path, `sync_group`, sampling-rate annotations, and clock metadata.

This answers:

> Which verified bytes, under which record interpretation, did neurOS open?

### Manifest identity

`manifest_sha256` hashes the exact serialized `neuros.dataset.json` bytes. It therefore also binds acquisition grouping, clock metadata, record ordering, paths, sampling annotations, and every other manifest field.

This answers:

> Which exact manifest authorized this operation?

### Alignment-plan identity

`AlignmentPlan.sha256` is a domain-separated SHA-256 under:

```text
neuros.exact_alignment_plan.v1
```

It binds the verified dataset content identity, exact manifest identity, synchronization group, exact policy, clock mapping, duration/stride, common overlap, and stable per-record frame arithmetic.

This answers:

> Which exact temporal correspondence is execution later authorized to consume?

These hashes are related but are not interchangeable. Alignment metadata must never overwrite a dataset/source fingerprint in ORION or another research-authority layer.

## Acquisition identity

Records that participate in one acquisition declare an explicit optional `sync_group`:

```json
{
  "id": "sub-01-bold-run-01",
  "subject": "sub-01",
  "modality": "fmri",
  "sync_group": "sub-01/run-01",
  "path": "sub-01/bold.f32",
  "source_sha256": "<64 lowercase hex characters>",
  "dtype": "float32-le",
  "shape": [300, 8192],
  "clock": {
    "id": "scanner-clock",
    "start_ns": 0,
    "period_ns": 2000000000
  }
}
```

`sync_group` is independent from `clock.id`. The former identifies one acquisition; the latter identifies one record's sample-time grid. One subject can therefore have many runs, and one run can contain records measured by different clocks.

Legacy single-modality manifests remain valid because `sync_group` is optional. Exact multimodal planning requires an explicit group and at least two requested modalities.

## Content verification happens before planning succeeds

The public planner is dataset-bound:

```python
plan = dataset.plan_aligned(...)
```

Before returning a plan, the Rust runtime calls the complete dataset verification authority from #148. Every manifest record must declare `source_sha256`, and every mapped source must match that declaration. A partially hashed dataset cannot produce a v1 alignment plan.

Planning does not return source arrays. Verification may mmap/hash source files, but the result remains metadata-only until a later execution layer consumes the qualified plan.

A changed source byte therefore causes planning to fail with the source-hash mismatch before a temporal plan is returned.

## Clock semantics

`ClockSpec` is authoritative integer time:

```text
ClockSpec {
    id,
    start_ns,
    period_ns,
}
```

Frame `k` begins at:

```text
start_ns + k * period_ns
```

A record with `N` frames occupies the half-open interval:

```text
[start_ns, start_ns + N * period_ns)
```

The planner uses checked integer arithmetic. Floating-point timestamps and `sampling_hz` are not temporal alignment authority.

## Exact policy

The public API is:

```python
from neuros import Dataset

study = Dataset.open("study")
plan = study.plan_aligned(
    sync_group="sub-01/run-01",
    modalities=["fmri", "behavior"],
    duration_ns=4_000_000_000,
    stride_ns=2_000_000_000,
    policy="exact",
)
```

The planner:

1. verifies the complete dataset content identity;
2. resolves exactly one record for each requested modality inside the selected `sync_group`;
3. requires explicit clocks for every selected record;
4. rejects acquisition groups that cross subjects;
5. computes the common half-open recording-time overlap;
6. solves the generalized integer congruence problem for the selected clock phases;
7. requires duration and stride to preserve every selected clock boundary;
8. selects the first legal common boundary inside the overlap;
9. emits a compact stable-ordered plan rather than materializing every window.

Different clock periods can be exactly compatible even when their rates differ. Conversely, streams with superficially similar rates can be impossible to align exactly when their phases disagree.

## Compact plan descriptor

`ExactAlignmentPlan` binds:

```text
schema_version
policy = exact
dataset_id
dataset_content_sha256
manifest_sha256
sync_group
start_ns
overlap_end_ns
duration_ns
stride_ns
window_count
entries[]
```

Entries are stable-sorted by modality and then record ID. Each entry binds:

```text
record_id
subject
modality
source_sha256
offset_bytes
dtype
shape
clock_id
clock_start_ns
period_ns
start_frame
frames_per_window
frame_stride
```

The source digest plus offset/dtype/shape is the per-record byte/interpretation evidence inherited from #148. Clock fields and derived frame mapping are the temporal evidence added by #147.

Window `i` starts at:

```text
plan.start_ns + i * plan.stride_ns
```

and record `r` starts at:

```text
r.start_frame + i * r.frame_stride
```

This keeps planning memory independent of recording duration. A long EEG/fMRI study does not require a precomputed object for every aligned window.

## Stable ordering versus exact-manifest binding

Two requirements must not be conflated:

- Plan **entry ordering** is canonical and does not depend on manifest record order.
- The **full plan fingerprint** remains bound to the exact `manifest_sha256`.

Consequently, reordering otherwise identical manifest records can preserve `dataset_content_sha256` and produce identical stable entries while still changing `manifest_sha256` and therefore `plan.sha256`.

This is intentional. A plan must not float between distinct serialized manifests merely because their semantic byte identity is equivalent.

## Inspection before execution

`AlignmentPlan.window(i)` materializes only one window's metadata mapping:

```python
window_37 = plan.window(37)
```

It returns:

- plan, dataset-content, and manifest fingerprints;
- exact absolute time interval;
- stable per-modality record/frame intervals;
- source SHA-256 and record interpretation descriptors;
- exact clock identity.

It does not open or concatenate modality arrays. This surface exists for independent validation, logging, provenance, and debugging before aligned execution is enabled.

## Rejection is part of the contract

Exact planning rejects rather than guesses when:

- the dataset lacks a complete declared content identity;
- any declared source digest does not match its source bytes;
- fewer than two modalities are requested;
- a requested modality is duplicated or non-canonical;
- a synchronization group is absent or ambiguous;
- one requested modality is missing from the selected group;
- a selected record has no clock;
- selected records span multiple subjects;
- recording intervals do not overlap;
- clock phases have no exact common boundary;
- duration would end between samples for any selected clock;
- stride would move later windows off any selected clock;
- checked clock/frame arithmetic overflows;
- the common overlap is too short for one legal window.

These are scientific guardrails, not inconveniences to be repaired with silent nearest-neighbor matching.

## Explicit non-goals

Exact alignment v1 planning does **not** implement:

- linear, spline, sinc, or polyphase resampling;
- nearest-neighbor tolerance windows;
- drift estimation or clock-warp correction;
- hemodynamic response alignment;
- modality-specific preprocessing;
- multimodal source execution;
- NIfTI/DICOM/EEG/video decoding;
- GPU tensor export.

Those require separate versioned policies or adapters with their own qualification evidence.

## ORION and research-authority boundary

Verified local bytes remain the responsibility of the #148 data-plane identity and may feed conservative ORION `DatasetLineage`. Exact synchronization proves temporal correspondence for selected slices. It does not strengthen acquisition provenance, preprocessing ancestry, participant identity, licensing, pretraining-overlap closure, or model-performance claims.

The alignment-plan fingerprint should therefore be referenced as execution/experiment metadata, not substituted for `DatasetAuthority.source_fingerprint` or ORION `content_sha256`.

## Qualification

The dedicated Rust Runtime CI must require, on one exact source SHA:

- `cargo fmt --all --check`;
- Clippy with warnings denied;
- Rust unit/adversarial tests for exact clock arithmetic and provenance binding;
- native wheel build and install;
- regression of the promoted v0 zero-copy/provenance path;
- a synthetic hashed fMRI/behavior acquisition;
- automatic whole-dataset verification during planning;
- independent Python recomputation of every derived frame mapping;
- public `Dataset.plan_aligned()` / `AlignmentPlan.window()` parity with the native plan;
- unchanged provenance performance evidence from #148.

Only after the planning layer is qualified and promoted should `stream_aligned(plan, prefetch=...)` become an execution surface. Execution must consume the exact qualified plan rather than recompute synchronization ad hoc.
