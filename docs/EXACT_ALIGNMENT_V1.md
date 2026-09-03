# Exact multimodal alignment v1

neurOS treats temporal alignment as scientific authority, not as array plumbing.

The v1 contract introduced by issue #147 adds an **exact-only planning layer** on top of the promoted Rust data plane. It deliberately stops before multimodal execution. A plan must first prove that every requested window boundary can be represented exactly on every selected modality clock.

## Acquisition identity

Records that participate in one acquisition declare an explicit `sync_group` in `neuros.dataset.json`:

```json
{
  "id": "sub-01-bold-run-01",
  "subject": "sub-01",
  "modality": "fmri",
  "sync_group": "sub-01/run-01",
  "path": "sub-01/bold.f32",
  "dtype": "float32-le",
  "shape": [300, 8192],
  "clock": {
    "id": "scanner-clock",
    "start_ns": 0,
    "period_ns": 2000000000
  }
}
```

`sync_group` is independent from `clock.id`. The former says which records belong to the same acquisition; the latter identifies the time grid used by one record. One subject can therefore have many runs, and one run can contain modalities measured by different clocks.

Legacy single-modality v0 manifests remain valid because `sync_group` is optional. Exact multimodal planning requires it.

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

The planner uses checked integer arithmetic. Floating-point timestamps are not used in authoritative alignment.

## Exact policy

The current public API is:

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

1. resolves exactly one record for each requested modality inside the selected `sync_group`;
2. requires explicit clock metadata for every selected record;
3. rejects acquisition groups that cross subjects;
4. computes the common recording-time overlap;
5. solves the generalized integer congruence problem for the selected clock phases;
6. requires both window duration and stride to preserve every selected clock boundary;
7. selects the first legal common boundary inside the overlap;
8. emits a compact, stable-ordered plan rather than materializing every window.

The generalized congruence step is important. Different clock periods can be exactly compatible even when their rates differ, while two apparently similar streams can be impossible to align exactly because their phases disagree.

## Compact plan

An `AlignmentPlan` binds:

- dataset ID;
- manifest SHA-256;
- synchronization group;
- alignment policy and schema version;
- first exact common boundary;
- common overlap end;
- duration and stride;
- total window count;
- stable-ordered per-record clock and frame arithmetic.

Each record entry contains:

```text
record_id
subject
modality
clock_id
clock_start_ns
period_ns
start_frame
frames_per_window
frame_stride
```

Window `i` therefore starts at:

```text
start_ns + i * stride_ns
```

and record `r` starts at:

```text
r.start_frame + i * r.frame_stride
```

This keeps plan memory constant with recording duration. A long EEG/fMRI study does not require millions of precomputed slice objects.

## Plan identity

`AlignmentPlan.sha256` is a domain-separated SHA-256 over the serialized exact plan using the domain:

```text
neuros.exact_alignment_plan.v1
```

The identity therefore changes when the selected dataset manifest, acquisition, clock mapping, duration, stride, or derived frame mapping changes.

This plan identity is not a source-content identity. Source-byte verification and the ORION lineage bridge remain explicitly separated in issue #148.

## Inspection before execution

`AlignmentPlan.window(i)` materializes only one window's metadata mapping:

```python
window_37 = plan.window(37)
```

It returns exact absolute time bounds and the frame interval selected from each record, without opening source data. This is intended for independent validation, logging, provenance, and debugging before multimodal execution is enabled.

## Rejection is part of the contract

Exact planning rejects rather than guesses when:

- fewer than two modalities are requested;
- a requested modality is duplicated;
- a synchronization group is absent or ambiguous;
- one requested modality is missing from the group;
- a selected record has no clock;
- selected records span multiple subjects;
- recording intervals do not overlap;
- clock phases have no exact common boundary;
- duration would end between samples for any selected clock;
- stride would move subsequent windows off any selected clock;
- checked clock arithmetic overflows;
- the common overlap is too short for one legal window.

These failures are scientific guardrails. They should not be replaced with silent nearest-neighbor matching.

## Explicit non-goals

Exact alignment v1 does **not** implement:

- linear, spline, sinc, or polyphase resampling;
- nearest-neighbor tolerance windows;
- drift estimation or clock-warp correction;
- hemodynamic response alignment;
- modality-specific preprocessing;
- multimodal source execution;
- GPU tensor export.

Those require separate, versioned policies with their own qualification evidence.

## Qualification

The dedicated Rust Runtime CI requires:

- `cargo fmt --all --check`;
- clippy with warnings denied;
- native unit tests, including incompatible phase and ambiguity cases;
- native wheel build and install;
- regression of the v0 zero-copy Arrow path;
- a synthetic fMRI/behavior exact-clock plan;
- independent Python recomputation of each derived frame mapping.

Only after planning is qualified should `stream_aligned(plan, ...)` become an execution surface.
