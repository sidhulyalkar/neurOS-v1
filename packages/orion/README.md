# ORION

ORION is the neural-intelligence layer that sits above the neurOS runtime.

neurOS owns acquisition, timing, processing, execution, recording/replay,
observability, and device/model integration. ORION owns neural tokenization,
representations, adaptive decoding, personalization, and future neural
foundation-model capabilities.

This package currently establishes the stable contracts between those layers.
It intentionally does **not** claim a finished foundation model. Existing
NeuroFM and neurotokenization research can migrate behind these interfaces as
experiments demonstrate scientific value.

```text
hardware -> neurOS SignalFrame -> ORION NeuroTokenizer
         -> NeuroTokenBatch -> NeuralEncoder
         -> RepresentationBatch -> AdaptiveDecoder
```
