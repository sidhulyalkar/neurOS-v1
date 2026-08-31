# DecoderOutput canonical contract

`DecoderOutput` is the neurOS inference-result boundary. It records what a decoder produced without inventing confidence, calibration, uncertainty, or biological meaning that the decoder did not supply.

This document describes the **representation and immutability** contract. It is not a model-performance or scientific-validity claim.

## Construction boundary

A constructed `DecoderOutput` must not retain caller-owned mutable state that can later change the canonical result.

The contract therefore canonicalizes fields at construction:

| Field | Canonical representation |
| --- | --- |
| numeric / boolean ndarray `prediction` | detached read-only NumPy array |
| string / object ndarray `prediction` | recursively validated immutable sequence |
| prediction list / tuple | immutable tuple |
| prediction mapping | immutable string-key mapping with recursively frozen values |
| `probabilities` | detached read-only numeric / boolean NumPy array when present |
| `logits` | detached read-only numeric / boolean NumPy array when present |
| `embedding` | detached read-only numeric / boolean NumPy array when present |
| `metadata` | recursively detached immutable deterministic provenance mapping |

Unordered sets, non-string prediction mapping keys, unsupported opaque prediction objects, and nonnumeric score / embedding arrays fail closed.

## Why numeric predictions stay arrays but string arrays become sequences

`prediction` intentionally has a broader semantic domain than score tensors. Classification labels may be integers, strings, scalars, arrays, or deterministic structured values.

For numeric and boolean ndarray predictions, neurOS preserves ndarray geometry while copying the backing storage and marking the canonical array read-only.

String and object ndarrays cannot use the qualified shared-memory ndarray byte lane. They are therefore recursively canonicalized through deterministic immutable sequences when their contents are supported. For example, a batch prediction equivalent to:

```python
np.array(["left", "right"])
```

has canonical prediction semantics equivalent to:

```python
("left", "right")
```

This preserves the ordered labels without pretending that string-label storage is a numeric tensor.

## Detachment is stronger than a read-only view

Calling `setflags(write=False)` on a view is not enough if the view still shares writable caller-owned storage. neurOS first copies canonical numeric arrays and only then marks the detached copy read-only.

Therefore this must not change the output:

```python
source = np.array([0.2, 0.8], dtype=np.float32)
output = DecoderOutput(prediction=1, probabilities=source)
source[:] = 0

assert output.probabilities.tolist() == [0.2, 0.8]
```

Direct writes through the canonical array are also rejected by NumPy.

## Metadata provenance

Metadata uses the same recursively frozen deterministic provenance semantics as canonical signal/window contracts:

- mappings require string keys and become immutable mappings;
- lists and tuples become immutable tuples;
- NumPy scalar values are detached to scalar values;
- NumPy metadata arrays are converted to immutable nested values;
- non-finite metadata floats/arrays, bytes, unordered sets, object-dtype metadata arrays, and unsupported opaque objects fail closed.

A caller retaining the original nested dict/list/array objects cannot mutate the canonical `DecoderOutput.metadata` afterward.

## `dataclasses.replace(...)`

`DecoderOutput` remains a frozen dataclass and can be evolved with `dataclasses.replace(...)`.

Replacement invokes the constructor again. Array fields are therefore copied and frozen again, and replacement metadata is recursively canonicalized. The replacement object does not silently acquire writable aliases to the original output's arrays.

This is important for compatibility layers that add runtime provenance such as adaptive-threshold metadata after inference.

## Shared-memory transport

The Phase C shared-memory codec reconstructs a fresh `DecoderOutput` on decode. Reconstruction passes through the same constructor contract, so received arrays are detached/read-only and metadata is recursively frozen again.

The transport does not weaken canonical immutability and does not require a separate mutable wire representation to become authoritative.

This remains shared-memory **transport**, not an end-to-end zero-copy callback contract. See [`RUNTIME_PROCESS_TRANSPORT.md`](RUNTIME_PROCESS_TRANSPORT.md).

## Generic pickle

Canonical provenance mappings use immutable mapping proxies. Generic pickle should not be treated as a universal serializer for canonical `DecoderOutput` objects. Process nodes using pickle remain valid when the actual process payload is pickle-representable, such as the standard numeric decoder input batch.

Use the qualified shared-memory transport when a canonical `DecoderOutput` itself must cross the process payload boundary.

## Scientific boundary

Canonical immutability proves that the represented result cannot be silently changed through retained Python aliases. It does **not** prove that:

- probabilities sum to one;
- probabilities are calibrated;
- logits have a particular probabilistic interpretation;
- embeddings encode a stable neural mechanism;
- uncertainty is statistically valid;
- a model generalizes across subjects, sessions, sites, or devices.

Those claims require model/evidence contracts and empirical qualification.

## Scalar authority follow-up

The immutability tranche deliberately preserves the pre-existing scalar validation for `confidence`, `uncertainty`, model identity, and `inference_time_ns` rather than changing two contract dimensions at once.

Issue #136 tracks stricter scalar authority, including finite uncertainty and exact non-negative inference timing. That follow-up is independent of the array/container immutability established here.

## Qualification

The maintained adversarial contract suite verifies:

- mutation of caller arrays after construction cannot change canonical output fields;
- direct writes through canonical numeric arrays fail;
- nested prediction and metadata sources cannot mutate canonical state;
- deterministic string-label batches remain supported;
- `dataclasses.replace(...)` recanonicalizes and redetaches storage;
- unsupported mutable/nondeterministic prediction values fail closed;
- shared-memory mailbox round trips preserve the immutable contract;
- the full runtime fault surface remains green across Ubuntu, macOS, and Windows on Python 3.10, 3.11, and 3.12.

The implementation and its adversarial tests are bound to Runtime Fault Qualification so changes to this canonical output boundary cannot bypass the cross-platform process/transport gate.
