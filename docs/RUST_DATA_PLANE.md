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

## v0 contract

The first implementation intentionally keeps the scientific surface narrow. A study directory contains `neuros.dataset.json` and one or more raw little-endian float32 payloads. The native runtime validates the manifest, rejects path traversal and malformed shapes, memory-maps source files read-only, plans deterministic windows, and hands windows to Python through a bounded prefetch queue.

Each exported Arrow array is constructed over the mmap using Arrow's externally owned `Buffer`. An `Arc` to the mapping is installed as the Arrow allocation owner. The Python Arrow capsule therefore extends the lifetime of the mapping and the values do not need to be copied at the Rust/Python boundary.

The v0 public API is:

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

## Manifest

A minimal manifest is:

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

## Ownership and concurrency invariants

The runtime follows several non-negotiable rules. Input mappings are read-only. Prefetch queues are bounded. Dropping a consumer cancels production naturally because the producer cannot send into a disconnected channel. Source paths must remain inside the dataset root. Window arithmetic uses checked integer operations. The GIL is released while Python waits for the next native window. Arrow owns a reference to every mapping it can expose. No multimodal temporal relationship is inferred without a declared synchronization policy.

`tokio` is intentionally not in the v0 dependency graph. Local memory-mapped files and CPU decoding do not benefit from introducing an async executor by default, and an extra scheduler can fight PyTorch/JAX thread pools. Tokio should enter behind source adapters that actually need asynchronous network or live-stream I/O. Rayon is reserved for bounded CPU transforms and decoders once those adapters land.

## Adapter roadmap

The runtime kernel should stay format-agnostic. Adapters implement source discovery and decoding, then produce the same internal clocked array contract. The implementation order should be NIfTI/BIDS first, then EEG files and NWB, DICOM series, video, remote/object-store shards, and finally live acquisition transports. Uncompressed `.nii` can use direct mapping where layout permits it. `.nii.gz` should decode once into a content-addressed cache rather than repeatedly inflate on every epoch. DICOM requires a study/series index plus an explicit PHI boundary. EEG and video require clock semantics from the beginning.

## Cache and provenance roadmap

The next provenance revision should add an optional source SHA-256 to each manifest record and verify it once when a mapping enters the process. Derived artifacts should use immutable content-addressed keys containing source digest, adapter version, transform graph digest, parameters, and runtime version. Cache writes must be atomic and checksummed. Scientific claims should be able to report the exact manifest digest and transform lineage that produced a tensor.

## Zero-copy interoperability roadmap

Arrow is the canonical columnar interchange boundary. DLPack should be added for dense tensor consumers so a decoded contiguous buffer can move into PyTorch or JAX without a NumPy staging allocation. The runtime must distinguish zero-copy transport from zero-copy decoding: compressed NIfTI, JPEG/video codecs, and many DICOM transfer syntaxes necessarily allocate during decode even when the final Python handoff is zero-copy.

## Qualification gates

A runtime adapter should not be promoted because it is faster on one benchmark. Promotion requires correctness against trusted readers, property tests for window and clock arithmetic, malformed-input tests, cancellation tests, memory/lifetime tests, deterministic ordering tests, and throughput/RSS measurements. Format adapters should include tiny distributable golden fixtures. Large public neuroscience datasets belong in integration or benchmark workflows, not unit tests.

This separation is the core product thesis: neurOS can become a scientific control plane over a high-performance, trustworthy neural data substrate rather than a collection of Python loaders that each reinvent caching, alignment, and provenance.
