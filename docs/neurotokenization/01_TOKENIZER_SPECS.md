# Tokenizer Specifications

## Common Interface

All spike tokenizers should implement a shared interface so benchmarking is fair and models can be swapped without rewriting pipelines.

```python
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol

@dataclass
class SpikeEvent:
    time: float
    unit_id: int
    session_id: Optional[str] = None
    region: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Token:
    token_id: int
    token_type: str
    time: Optional[float] = None
    duration: Optional[float] = None
    unit_id: Optional[int] = None
    value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class SpikeTokenizer(Protocol):
    name: str

    def fit(self, events: List[SpikeEvent], metadata: Optional[Dict[str, Any]] = None) -> None:
        ...

    def encode(self, events: List[SpikeEvent], metadata: Optional[Dict[str, Any]] = None) -> List[Token]:
        ...

    def decode(self, tokens: List[Token]) -> List[SpikeEvent]:
        ...

    def describe_token(self, token_id: int) -> Dict[str, Any]:
        ...
```

## Required Metadata

Each tokenizer should emit a manifest:

```json
{
  "name": "isi_midi",
  "version": "0.1.0",
  "fit_required": false,
  "vocab_size": 2048,
  "supports_decode": true,
  "supports_streaming": true,
  "time_resolution_ms": 1.0,
  "expected_inputs": ["spike_events"],
  "known_failure_modes": ["high firing rate vocabulary explosion"]
}
```

## 1. EventSpikeTokenizer

### Purpose

Baseline tokenizer inspired by POYO-style spike event representations.

### Encoding

Each spike becomes one token:

```text
SPIKE(unit_id, absolute_time)
```

The model receives:

- unit embedding
- timestamp encoding
- optional session / region / depth embedding
- optional waveform or unit-quality features

### Implementation Notes

- Use sorted spike events.
- Keep absolute time as a continuous feature rather than exploding it into vocabulary IDs.
- Support chunked windows for long sessions.
- Support optional unit metadata embeddings.

### Strengths

- Preserves precise timing.
- Streaming-friendly.
- Minimal assumptions.
- Strong baseline for sparse spike streams.

### Weaknesses

- Individual spikes may be too atomic.
- The model must learn bursts, synchrony, and assemblies from scratch.
- Unit identity does not transfer naturally across sessions.

## 2. BinnedCountTokenizer

### Purpose

Classical count-matrix baseline.

### Encoding

For each time bin:

```text
token_t = count_vector[num_units]
```

or, for sequence models:

```text
BIN_TOKEN(t, count_vector)
```

### Parameters

- `bin_ms`: 5, 10, 20, 50, 100
- `normalize`: none, zscore, sqrt, log1p
- `max_units`: optional cap for memory

### Strengths

- Simple and stable.
- Works with standard neural decoders.
- Easy reconstruction and held-out prediction.

### Weaknesses

- Bin size is a hidden scientific assumption.
- Fine timing can be smeared away.
- Sparse data becomes dense.
- Cross-session unit mismatch remains difficult.

## 3. ISIMidiTokenizer

### Purpose

Make relative temporal gaps first-class tokens.

### Encoding Options

#### Separate WAIT and SPIKE tokens

```text
WAIT_GLOBAL_4MS
SPIKE_UNIT_17
WAIT_GLOBAL_1MS
SPIKE_UNIT_22
```

#### Combined timing-context tokens

```text
SPIKE_UNIT_17_DTG_4_DTU_128_POP_HIGH
```

Where:

- `DTG`: delta time since previous global spike
- `DTU`: delta time since same unit last fired
- `POP`: recent population event density

### Recommended First Version

Use separate tokens:

1. Discretize global inter-event time with logarithmic bins.
2. Emit `WAIT_BIN_k` before each spike if the gap is nonzero.
3. Emit `SPIKE_UNIT_i`.
4. Store `dt_unit` and recent population rate as side features.

### Timing Bins

Use logarithmic bins in milliseconds:

```text
0-1, 1-2, 2-4, 4-8, 8-16, 16-32, 32-64, 64-128, 128-256, 256-512, 512+
```

### Strengths

- Captures rhythm, bursts, and sequence timing.
- More event-native than fixed bins.
- Useful for low-latency streaming.

### Weaknesses

- Vocabulary can grow if timing and unit identity are crossed too aggressively.
- Global event ordering becomes sensitive to simultaneous spikes.
- Requires careful handling of refractory violations and sorting noise.

## 4. BurstTokenizer

### Purpose

Convert local spike clusters into higher-level burst, pause, and rebound tokens.

### First-Pass Rule

For each unit independently:

```text
burst if at least 3 spikes occur with consecutive ISIs < 10 ms and total duration < 100 ms
pause if ISI > unit-specific percentile threshold
rebound if a spike or burst follows a long pause within a configurable window
```

### Token Types

```text
SINGLE_SPIKE(unit)
BURST_START(unit)
BURST(unit, count, duration, median_isi)
BURST_END(unit)
PAUSE(unit, duration_bin)
REBOUND(unit, latency_bin)
```

### Parameters

- `min_spikes`: default 3
- `max_isi_ms`: default 10
- `max_burst_duration_ms`: default 100
- `pause_percentile`: default 95
- `adaptive_per_unit`: true

### Upgrade Path

Add statistical burst detectors:

- Poisson Surprise
- Rank Surprise
- Robust Gaussian Surprise
- ISI-N thresholding

### Strengths

- More interpretable than raw spikes.
- Compresses repeated local spike events.
- Useful for bursting cell types and local motifs.

### Weaknesses

- Handcrafted thresholds may encode assumptions.
- Different cell types have different firing statistics.
- Bursts can be confounded by sorting artifacts.

## 5. SynchronyPacketTokenizer

### Purpose

Detect population-level coactivation packets.

### Encoding

Within a short window:

```text
SYNC_PACKET(time, active_units, size, duration, region_summary)
```

Simplified vocabulary:

```text
SYNC_SMALL
SYNC_MEDIUM
SYNC_LARGE
SYNC_REGION_M1
SYNC_REGION_STRIATUM
```

### Parameters

- `window_ms`: 1, 2, 5, 10
- `min_units`: default 3
- `max_jitter_ms`: default 2
- `region_aware`: optional

### Strengths

- Captures synchronous packets and cascade-like events.
- Good for population motifs.
- Interpretable if paired with active unit summaries.

### Weaknesses

- Can be biased by firing rate.
- Requires null-model controls.
- Sensitive to unit count and recording density.

## 6. VQMotifTokenizer

### Purpose

Learn neural subword tokens from local raster windows.

### Pipeline

```text
spike raster window -> local encoder -> latent vector -> vector quantization -> MOTIF_ID
```

### Recommended Architecture

- Input: `[num_units, time_bins]` raster windows.
- Encoder: small TCN, Conv1D, GRU, or compact Transformer.
- Codebook: VQ with 64, 128, 256, or 512 codes.
- Decoder: reconstruct raster or predict future raster.

### Objectives

Start with:

```text
reconstruction loss + VQ commitment loss + codebook diversity regularizer
```

Then add:

```text
future spike prediction + behavior/trial-state prediction + contrastive alignment
```

### Strengths

- Learns motifs instead of handcrafting them.
- Can discover reusable neural subwords.
- Can optimize for downstream meaning rather than pure reconstruction.

### Weaknesses

- More expensive.
- Can collapse codebook usage.
- Interpretability requires extra reporting.
- Risk of learning nuisance variables.

## 7. AssemblyTokenizer

### Purpose

Represent activity as activation of recurring cell assemblies.

### Candidate Methods

- PCA / ICA over binned spike matrices
- NMF for nonnegative assembly patterns
- seqNMF-style temporal factors
- graph/community detection over functional coupling
- CAD/SPADE/ASSET-inspired pattern extraction

### Token Types

```text
ASSEMBLY_ON(id)
ASSEMBLY_PEAK(id, strength_bin)
ASSEMBLY_OFF(id)
ASSEMBLY_TRANSITION(id_a, id_b)
```

### Strengths

- Often more interpretable than VQ motifs.
- Naturally supports population-level meaning.
- Good bridge to scientific reporting.

### Weaknesses

- Requires fitting per dataset or transfer alignment.
- Number of assemblies is a sensitive hyperparameter.
- May miss fine temporal sequence structure.

## Vocabulary Management

Avoid unconstrained token explosion. Use:

- typed tokens
- side features for continuous values
- separate token ID and metadata fields
- configurable unit hashing for large recordings
- optional per-session local vocabularies plus global token classes

## Streaming Requirements

Each tokenizer should declare whether it supports online mode. For streaming:

- no future context allowed
- bounded memory
- deterministic behavior
- latency measured per event or window

## Testing Requirements

For each tokenizer:

1. Deterministic encode/decode smoke test.
2. Handles empty input.
3. Handles unsorted input by sorting or raising clear error.
4. Handles simultaneous spikes.
5. Handles one unit and many units.
6. Handles long silence.
7. Produces valid manifest.
8. Produces stable token counts under seeded synthetic data.
