# Neural windows

`SignalFrame` and `NeuralWindow` represent different stages of the neurOS data plane.

- **`SignalFrame`** is a timestamped streaming chunk. Its native multi-sample convention is `sample x channel` and must be declared through `metadata["axis_order"]` when two-dimensional.
- **`NeuralWindow`** is exactly one decoder-ready temporal window. Its geometry is always `channel x time`.
- **The runtime**, not the model adapter, adds the batch dimension. A `NeuralWindow(C, T)` therefore reaches a decoder as `(1, C, T)`.

This boundary exists so raw-window neural models do not depend on hidden NumPy conventions.

## Canonical path

```text
hardware / MNE / LSL / replay
              |
              v
         SignalFrame
       sample x channel
              |
              v
   SlidingWindowTransform
              |
              v
         NeuralWindow
        channel x time
              |
              v
        RuntimeExecutor
        adds batch axis
              |
              v
          Decoder
      1 x channel x time
              |
              v
       DecoderOutput
  + window provenance
```

## Window specification

Window geometry is sample-domain and deterministic:

```python
from neuros.contracts import WindowSpec

spec = WindowSpec(window_samples=500, stride_samples=125)
```

A seconds-based constructor is available when the sampling rate is known:

```python
spec = WindowSpec.from_seconds(
    sample_rate_hz=250.0,
    window_seconds=2.0,
    stride_seconds=0.5,
)
```

`stride_samples` may equal `window_samples` for non-overlapping windows. A stride larger than the window is rejected because neurOS does not silently create temporal holes.

## Windowing inside RuntimeGraph

```python
from neuros.contracts import WindowSpec
from neuros.processing import SlidingWindowTransform
from neuros.runtime import NodeKind, RuntimeEdge, RuntimeGraph, RuntimeNode

windowing = SlidingWindowTransform(
    WindowSpec(window_samples=500, stride_samples=125),
    descriptor=source.descriptor,
)

graph = RuntimeGraph()
graph.add_node(RuntimeNode("source", NodeKind.SOURCE, source))
graph.add_node(RuntimeNode("window", NodeKind.TRANSFORM, windowing))
graph.add_node(RuntimeNode("decoder", NodeKind.DECODER, decoder))
graph.connect(RuntimeEdge("source", "window"))
graph.connect(RuntimeEdge("window", "decoder"))
```

Chunked inputs may contain enough samples to create more than one window. The transform returns an explicit `TransformEmission`, and the native executor fans every emitted window downstream in deterministic order. Plain Python lists and tuples are **not** treated as fan-out because they may be legitimate data values.

## Configuration-first path

The same transform is registered as the `window` plugin:

```yaml
schema_version: 1
streams:
  - id: eeg
    source:
      plugin: lsl
      options: {}
    transforms:
      - plugin: window
        options:
          window_samples: 500
          stride_samples: 125
          discontinuity: error

decoder:
  plugin: your_raw_window_decoder
  options: {}
```

This is the intended path for EEGNet, CNN, Transformer, EEG-Conformer, and external raw-window decoder adapters. The legacy `Pipeline` remains a feature-vector compatibility facade and should not be used to route raw windows.

## Provenance carried into inference

Each `NeuralWindow` binds:

- stream ID;
- monotonic window ID;
- sample rate;
- channel names;
- half-open start/end timestamps;
- clock domain;
- aggregated signal-quality flags;
- the exact `SignalFrame.sequence_id` values used to construct it.

When a decoder returns a `DecoderOutput`, the runtime adds this information to `DecoderOutput.metadata`. Predictions can therefore be traced back to the exact live or replayed source chunks that produced them.

## Discontinuity semantics

The default behavior is fail-closed:

```python
SlidingWindowTransform(spec, discontinuity="error")
```

A sequence gap, dropped-sample flag, incompatible clock transition, channel geometry change, or inconsistent sample rate cannot be silently bridged.

For workloads where a discontinuity should begin a new independent segment, use:

```python
SlidingWindowTransform(spec, discontinuity="reset")
```

`reset` discards pending overlap and starts a new contiguous segment. It does **not** pad, interpolate, resample, or fabricate the missing samples. The transform records the discontinuity count in window metadata.

## Scientific boundary

Window creation is a representation boundary, not evidence of model validity. A decoder becomes trustworthy only when the resulting pipeline is exercised under the appropriate neurOS Evidence protocol, including subject/session separation, calibration accounting, artifact stress, device/montage transfer, and latency/closed-loop qualification where applicable.
