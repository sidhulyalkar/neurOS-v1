# neurOS Rust data plane

## Purpose

The Rust data plane is a systems substrate beneath the Python scientific API. It is not a rewrite of neurOS and it is not a second model framework. Python continues to own experiment definition, statistical analysis, model construction, scientific policy, and notebooks. Rust owns operations whose correctness and performance benefit from explicit memory ownership, bounded concurrency, deterministic transport, and stable binary interfaces.

The intended long-term flow is:

```text
Python experiment / model API
          |
          v
validated stream request
          |
          v
neuros-runtime (Rust)
  source -> decode -> clock/sync -> transform -> window -> batch
          |
          v
Arrow C Data / DLPack
          |
          v
PyTorch / JAX / Polars / analytics
```

## v0 transport contract

The first implementation intentionally keeps the scientific surface narrow. A study directory contains `neuros.dataset.json` and one or more raw little-endian float32 payloads. The native runtime validates the manifest, rejects path traversal and malformed shapes, memory-maps source files read-only, plans deterministic windows, and hands windows to Python through a bounded prefetch queue.

Each exported Arrow array is constructed over the mmap using Arrow's externally owned `Buffer`. An `Arc` to the mapping is installed as the Arrow allocation owner. The Python Arrow capsule therefore extends the lifetime of the mapping and the values do not need to be copied at the Rust/Python boundary.

The public API remains intentionally small:

```python
from neuros import Dataset

study = Dataset.open("study/")
for window in study.stream(
    subjects=["sub-01", "sub-02"],
    modalities=["fmri"],
    window=32,
    stride=16,
    prefetch=8,
):
    fmri = window.fmri       # arro3 Arrow array, zero-copy from mmap
    shape = window.shape
    provenance = window.provenance
```

Multiple modalities are rejected by the high-level v0 stream. This is deliberate. fMRI volumes, behavioral events, EEG samples, eye tracking, and video frames usually live on different clocks. neurOS must never claim that samples are aligned merely because two arrays have matching indices. The next multimodal contract will require explicit clock identities, synchronization edges, interpolation/resampling policy, and tolerance bounds.

## Manifest and source identity

A manifest may optionally declare a strict lowercase full-file SHA-256 for each record source:

```json
{
  "schema_version": 1,
  "dataset_id": "example-study",
  "records": [
    {
      "id": "sub-01-bold",
      "subject": "sub-01",
      "modality": "fmri",
      "path": "sub-01/bold.f32",
      "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "dtype": "float32-le",
      "shape": [1200, 91282],
      "sampling_hz": 0.5,
      "clock": {
        "id": "scanner-clock",
        "start_ns": 0,
        "period_ns": 2000000000
      }
    }
  ]
}
```

The first dimension is the windowing axis. Remaining dimensions are a frame and stay described by `shape`; the Arrow values array is flat so the runtime does not impose NumPy, PyTorch, or JAX tensor semantics.

When `source_sha256` is present, neurOS canonicalizes and validates the source path, creates the read-only mmap that will back the Arrow view, hashes that exact mapped byte region, and rejects a mismatch before returning a scientific window. Verification state is cached separately from mmap existence, so a source first opened through an unhashed record can later be upgraded when another record requires verification.

`verified_at_open` means the mapped regular file matched its declared digest when the runtime verified that mapping. It does **not** mean the surrounding mutable filesystem has become immutable against later external writers. Strong immutable stores, Merkle structures, remote attestation, and cryptographic signatures remain separate future contracts.

## Three distinct identities

The runtime deliberately keeps three concepts separate.

### Manifest identity

`manifest_sha256` hashes the exact `neuros.dataset.json` bytes. It therefore binds path names, sampling metadata, clock declarations, record order in the serialized file, and every other manifest field.

### Dataset content identity

When every record declares a source hash, neurOS derives a domain-separated canonical `declared_dataset_content_sha256` under:

```text
neuros.dataset_content.v1
```

Records are stable-sorted by `record_id`. Each descriptor binds:

```text
record_id
source_sha256
record offset_bytes
dtype
shape
```

Path is intentionally excluded. Renaming an identical source does not change byte/interpretation identity. Sampling rate and clock metadata are also excluded because they remain part of manifest/semantic identity. Changing offset, dtype, or shape changes the dataset content identity even when the underlying source bytes are unchanged.

A manifest containing even one unhashed record has no complete dataset content identity. neurOS does not synthesize a partial digest and label it complete.

### Verified dataset content identity

`declared_dataset_content_sha256` proves only what the manifest claims. `Dataset.verify_content()` explicitly verifies every source required by that identity. Only after all declared sources match does `verified_content_sha256` become available.

```python
study = Dataset.open("study/")
assert study.declared_content_sha256 is not None
assert study.verified_content_sha256 is None

verified = study.verify_content()
assert verified == study.declared_content_sha256
assert study.verified_content_sha256 == verified
```

This explicit promotion step prevents lazy verification of one record from being mistaken for proof that an entire dataset was verified.

## Window provenance

Each `DataWindow.provenance` record carries enough identity to bind downstream evidence to what the runtime consumed:

```text
record_id
subject
modality
shape
sampling_hz
manifest_sha256
source_size_bytes
declared_source_sha256
verified_source_sha256
source_verification_state
declared_dataset_content_sha256
verified_dataset_content_sha256
record_byte_interval {start, end_exclusive}
window_frame_interval {start, end_exclusive}
```

The dataset-level verified hash appears on windows only after explicit whole-dataset verification has completed.

## ORION lineage bridge

A fully verified native dataset can be projected into ORION without inventing a second content vocabulary:

```python
study.verify_content()
lineage = study.to_orion_lineage(
    upstream_source="authorized-local-export",
    preprocessing_history=("producer preprocessing described externally",),
    sampling_assumptions={"sampling_rate_hz": 0.5},
)
```

The bridge maps the verified dataset content identity to ORION `DatasetLineage.content_sha256` and records native manifest/content identities in ORION metadata. It always emits `LineageCompleteness.UNKNOWN`.

That conservatism is intentional. Local byte verification proves which bytes/interpretation neurOS consumed. It does not establish original acquisition provenance, upstream preprocessing ancestry, participant/stimulus identity completeness, licensing, or pretraining-overlap closure. Stronger lineage claims require independent evidence and should be constructed through ORION's scientific authority directly.

## Ownership and concurrency invariants

The runtime follows several non-negotiable rules. Input mappings are read-only. Prefetch queues are bounded. Dropping a consumer cancels production naturally because the producer cannot send into a disconnected channel. Source paths must remain inside the dataset root. Window arithmetic uses checked integer operations. The GIL is released while Python waits for the next native window. Arrow owns a reference to every mapping it can expose. No multimodal temporal relationship is inferred without a declared synchronization policy.

Source hashing is performed over the exact mmap exposed downstream. The global mmap-cache lock is not held while hashing; verification has its own per-mapping lock. Two records referencing the same live mapping therefore share verification state without blocking unrelated sources behind a whole-file hash.

`tokio` is intentionally not in the v0 dependency graph. Local memory-mapped files and CPU decoding do not benefit from introducing an async executor by default, and an extra scheduler can fight PyTorch/JAX thread pools. Tokio should enter behind source adapters that actually need asynchronous network or live-stream I/O. Rayon is reserved for bounded CPU transforms and decoders once those adapters land.

## Adapter roadmap

The runtime kernel should stay format-agnostic. Adapters implement source discovery and decoding, then produce the same internal clocked array contract. The implementation order should be NIfTI/BIDS first, then EEG files and NWB, DICOM series, video, remote/object-store shards, and finally live acquisition transports. Uncompressed `.nii` can use direct mapping where layout permits it. `.nii.gz` should decode once into a content-addressed cache rather than repeatedly inflate on every epoch. DICOM requires a study/series index plus an explicit PHI boundary. EEG and video require clock semantics from the beginning.

## Derived-cache provenance roadmap

Source identity is only the first cryptographic layer. Derived artifacts should use immutable content-addressed keys containing source digest, adapter version, transform graph digest, parameters, and runtime version. Cache writes must be atomic and checksummed. Scientific claims should be able to report the exact source identity, manifest identity, transform lineage, and runtime version that produced a tensor.

## Zero-copy interoperability roadmap

Arrow is the canonical columnar interchange boundary. DLPack should be added for dense tensor consumers so a decoded contiguous buffer can move into PyTorch or JAX without a NumPy staging allocation. The runtime must distinguish zero-copy transport from zero-copy decoding: compressed NIfTI, JPEG/video codecs, and many DICOM transfer syntaxes necessarily allocate during decode even when the final Python handoff is zero-copy.

## Qualification gates

A runtime adapter should not be promoted because it is faster on one benchmark. Promotion requires correctness against trusted readers, property tests for window and clock arithmetic, malformed-input tests, source-mutation tests, content-identity invariance tests, cancellation tests, memory/lifetime tests, deterministic ordering tests, and throughput/RSS measurements. Format adapters should include tiny distributable golden fixtures. Large public neuroscience datasets belong in integration or benchmark workflows, not unit tests.

For source provenance specifically, qualification must distinguish cold verified open from warm iteration. Hash throughput is a separately reported systems cost; it is not hidden inside a claim of zero-copy transport.

This separation is the core product thesis: neurOS can become a scientific control plane over a high-performance, trustworthy neural data substrate rather than a collection of Python loaders that each reinvent caching, alignment, and provenance.
