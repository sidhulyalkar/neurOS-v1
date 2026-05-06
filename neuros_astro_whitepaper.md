# neuros-astro Initial Whitepaper and Project Scope

## Working Title

**neuros-astro: A glial signal layer for neural foundation models**

## Recommended Repository Placement

```text
neurOS-v1/
  packages/
    neuros-astro/
```

Alternative tighter integration path:

```text
neurOS-v1/
  packages/
    neuros-neurofm/
      neuros_neurofm/
        astro/
```

The recommended initial implementation is a standalone package, `neuros-astro`, with clean adapters into `neuros-neurofm`. This keeps the astrocyte analysis layer modular, testable, publishable, and reusable outside the foundation-model stack.

---

## 1. Executive Summary

`neuros-astro` is a new package for extracting, storing, tokenizing, and modeling astrocyte-level calcium/network activity from optical physiology data. The package is designed to integrate with the broader `neurOS` and `neuroFMx` ecosystem as a new biological modality: a slow, spatially structured glial context signal that may improve prediction of neural dynamics, behavioral state, arousal, task context, and long-timescale latent drift.

Most existing large-scale neuroscience modeling pipelines treat the brain as primarily neuronal: spikes, LFP, calcium traces, video, behavior, and task variables. This is powerful, but incomplete. Astrocytes and glial networks may encode or regulate slow-timescale contextual variables that shape computation, plasticity, memory, arousal, decision-making, and state transitions. In foundation models for neuroscience, this creates a clear opportunity: represent astrocyte dynamics as an additional latent context field rather than ignoring them or forcing them into neuron-like trace formats.

The initial project focuses on three practical goals:

1. **Dataset triage:** identify whether existing NWB/DANDI/local calcium imaging datasets can be reanalyzed for astrocyte-like signals.
2. **Astrocyte event extraction:** detect candidate astrocytic calcium events from traces or movies using lightweight baseline methods, with adapters for specialized tools later.
3. **Model integration:** convert astrocyte events and functional astrocyte networks into tokenized features that can plug into `neuroFMx` as a new modality.

The longer-term vision is to build an astrocyte-aware foundation-model layer for neuroscience: one that asks whether glial dynamics improve prediction, generalization, interpretability, and biological realism in neural data models.

---

## 2. Core Hypotheses

### Scientific Hypothesis

Astrocytic calcium dynamics provide a slow-timescale, spatially structured regulatory state over neural computation. These dynamics may encode context, arousal, local metabolic state, neuromodulatory tone, and circuit-level coordination that are not fully recoverable from spikes, LFP, neuronal calcium, or behavior alone.

### Machine Learning Hypothesis

Adding astrocyte-derived event tokens and astrocyte network tokens to a multimodal neural foundation model may improve:

- future neural activity prediction
- behavioral state prediction
- task-context decoding
- arousal/running/pupil prediction
- latent-state stability across sessions
- cross-session and cross-animal generalization
- interpretability of slow-timescale neural state changes

### Infrastructure Hypothesis

Existing optical physiology datasets may contain underused astrocyte-relevant information, especially when raw movies, segmentation masks, fluorescence traces, or rich metadata are available. A reusable package can triage, reprocess, standardize, and tokenize these signals without requiring every dataset to have been originally designed for astrocyte analysis.

---

## 3. Why This Belongs in neurOS

The `neurOS` ecosystem already emphasizes multimodal neural data, tokenization, alignment, scalable storage, foundation models, and biologically grounded modeling. `neuros-astro` fits naturally because astrocyte signals are not merely another preprocessing output. They are a candidate missing modality.

Existing `neuroFMx` architecture already has:

```text
Neural data
  -> modality-specific tokenizers
  -> temporal backbone such as Mamba or Transformer
  -> Perceiver-style fusion
  -> population-level aggregation
  -> multitask heads
```

`neuros-astro` adds:

```text
Astrocyte movie/trace/event data
  -> astrocyte event extraction
  -> astrocyte network construction
  -> astrocyte tokenization
  -> multimodal temporal alignment
  -> neuroFMx fusion
```

The conceptual role of astrocytes in the model should be treated differently from fast neural signals. Astrocyte tokens may function as:

- slow context tokens
- graph/network state tokens
- local field modulation tokens
- arousal or metabolic context proxies
- long-timescale latent-state anchors
- cross-session drift explainers

This makes the project more than a preprocessing module. It becomes a glial layer for foundation-model neuroscience.

---

## 4. Target Users

### Primary User

The initial primary user is the developer/researcher building `neurOS`, `neuroFMx`, and related neural foundation-model infrastructure.

### Secondary Users

Potential future users include:

- computational neuroscientists studying astrocyte-neuron interactions
- systems neuroscientists with optical physiology datasets
- glial biologists who need reusable analysis tooling
- neuroAI researchers building multimodal biological foundation models
- labs standardizing astrocyte imaging workflows in NWB/Zarr-compatible formats

---

## 5. Design Principles

### Modular First

The package should stand alone. It should not require all of `neuroFMx` to function. It should expose clean exports that `neuroFMx` can consume.

### Lightweight Baselines Before Heavy Models

The MVP should use simple, testable methods:

- robust z-scoring
- connected components
- event merging
- coactivation networks
- Jaccard event overlap
- NumPy/Pandas/Zarr/Parquet exports

Specialized tools such as AQuA/AQuA2, astroCaST, ASTRA, or AstroNet-style algorithms can be added as adapters after the baseline pipeline works end-to-end.

### Metadata Matters

The first real bottleneck is not modeling. It is determining which datasets are suitable. The package should treat metadata as a first-class object.

The dataset triage layer should ask:

- Is raw imaging available?
- Are there processed traces?
- Are there segmentation masks?
- Is the imaging one-photon, two-photon, widefield, confocal, or miniscope?
- Does metadata mention astrocytes, glia, GFAP, Aldh1l1, S100B, SR101, sulforhodamine, GCaMP, or related terms?
- Are behavior, stimulus, pupil, running, LFP, spikes, or tracking data available?
- Is temporal alignment information available?

### Event-Centric Representation

Astrocyte calcium activity should be represented primarily as spatiotemporal events, not only as neuron-like ROI traces. Astrocyte signals can be diffuse, propagating, compartmental, slow, and spatially complex. The event should be a central abstraction.

### Foundation-Model Ready

Every major output should eventually support model ingestion:

- event tables
- graph summaries
- irregular event tokens
- binned state tokens
- session manifests
- neuroFMx-compatible exports

### Storage-Compatible

The package should support formats that are realistic for modern neuroscience workflows:

- NWB for neurophysiology standardization
- Zarr for scalable chunked arrays
- Parquet for event tables
- NPZ for simple token demos
- WebDataset for large-scale training shards

---

## 6. MVP Scope

### MVP Name

**MVP 1: Astrocyte Event and Network Tokenization Pipeline**

### MVP Goal

Build a package that can:

1. scan a dataset or metadata file
2. score astrocyte reanalysis potential
3. detect candidate astrocyte events from traces or movies
4. build simple astrocyte coactivation graphs
5. tokenize events into model-ready arrays
6. export the result for neuroFMx integration

### MVP Non-Goals

The first MVP should not initially:

- train a deep segmentation model
- replace specialized astrocyte analysis tools
- guarantee biological astrocyte identity from unlabeled data
- perform full NWB writing for all output types
- build a production DANDI-scale data crawler
- build a large-scale foundation model experiment immediately
- claim that every calcium imaging dataset contains usable astrocyte information

The first MVP is an end-to-end skeleton with scientifically sensible defaults.

---

## 7. Package Architecture

```text
packages/neuros-astro/
  pyproject.toml
  README.md
  neuros_astro/
    __init__.py
    io/
      __init__.py
      nwb_loader.py
      dandi_indexer.py
      tiff_loader.py
      suite2p_loader.py
      caiman_loader.py
      aqua_loader.py
      zarr_store.py
      synthetic.py
    metadata/
      __init__.py
      schema.py
      controlled_terms.py
      dataset_scoring.py
    preprocessing/
      __init__.py
      motion.py
      normalization.py
      neuropil.py
      denoise.py
      detrend.py
    segmentation/
      __init__.py
      candidate_regions.py
      astrocyte_morphology.py
      astra_adapter.py
      manual_roi.py
    events/
      __init__.py
      event_detection.py
      calcium_event_features.py
      event_qc.py
      aqua_adapter.py
    networks/
      __init__.py
      functional_connectivity.py
      propagation.py
      graph_features.py
      astro_net_adapter.py
    tokenization/
      __init__.py
      astro_tokenizer.py
      event_tokenizer.py
      graph_tokenizer.py
    fusion/
      __init__.py
      align_to_behavior.py
      align_to_lfp.py
      align_to_spikes.py
      align_to_stimulus.py
    export/
      __init__.py
      to_nwb.py
      to_webdataset.py
      to_parquet.py
      to_neurofm.py
    visualization/
      __init__.py
      movie_overlay.py
      event_raster.py
      network_plot.py
      dashboard.py
    cli/
      __init__.py
      main.py
  configs/
    dataset_triage.yaml
    local_test.yaml
    dandi_scan.yaml
    astro_event_detection.yaml
    neurofm_export.yaml
  examples/
    00_generate_synthetic_astro_data.py
    01_scan_dataset.py
    02_detect_events.py
    03_build_network.py
    04_export_to_neurofm.py
    05_visualize_synthetic_outputs.py
  tests/
    test_schema.py
    test_dataset_scoring.py
    test_event_detection.py
    test_synthetic.py
    test_tokenizer.py
    test_graph_features.py
    test_export_schema.py
```

---

## 8. Core Data Objects

### AstroSession

Represents a session or dataset unit.

```python
@dataclass
class AstroSession:
    session_id: str
    subject_id: str | None
    source_path: str
    source_type: Literal["nwb", "tiff", "suite2p", "caiman", "aqua", "zarr", "unknown"]
    frame_rate_hz: float | None
    imaging_modality: str | None
    indicator: str | None
    promoter: str | None
    brain_region: str | None
    has_raw_movie: bool = False
    has_masks: bool = False
    has_behavior: bool = False
    has_ephys: bool = False
    metadata_score: float = 0.0
```

### AstroRegion

Represents a candidate astrocyte ROI, domain, process segment, soma, or unknown region.

```python
@dataclass
class AstroRegion:
    region_id: str
    session_id: str
    mask_shape: tuple[int, int] | None
    centroid_yx: tuple[float, float] | None
    area_px: float | None
    perimeter_px: float | None
    eccentricity: float | None
    solidity: float | None
    branchiness: float | None
    confidence: float
    region_type: Literal["soma", "process", "domain", "unknown"]
```

### AstroEvent

Represents one spatiotemporal astrocytic calcium event.

```python
@dataclass
class AstroEvent:
    event_id: str
    session_id: str
    region_id: str | None
    onset_frame: int
    offset_frame: int
    peak_frame: int
    duration_s: float
    peak_dff: float
    area_px: float | None
    centroid_yx: tuple[float, float] | None
    propagation_speed: float | None
    direction_rad: float | None
    confidence: float
```

### AstroGraph

Represents a windowed astrocyte functional network.

```python
@dataclass
class AstroGraph:
    session_id: str
    window_start_s: float
    window_end_s: float
    nodes: list[str]
    edges: list[tuple[str, str]]
    edge_weights: list[float]
    edge_metric: Literal["correlation", "lagged_correlation", "event_coactivation", "mutual_information"]
```

### TokenizedAstroSequence

Represents model-ready astrocyte event tokens.

```python
@dataclass
class TokenizedAstroSequence:
    tokens: np.ndarray
    timestamps_s: np.ndarray
    region_ids: list[str | None]
    feature_names: list[str]
    metadata: dict[str, Any]
```

---

## 9. Dataset Triage System

### Purpose

Before processing or modeling, the package must determine whether a dataset is likely useful for astrocyte reanalysis.

### Inputs

The triage system should accept:

- NWB file path
- JSON metadata file
- DANDI dandiset metadata
- local directory path
- plain metadata dictionary
- arbitrary text metadata

### Outputs

```json
{
  "session_id": "sub-001_ses-001",
  "astro_reanalysis_score": 0.72,
  "has_raw_movie": true,
  "has_masks": true,
  "has_behavior": true,
  "has_ephys": false,
  "matched_astro_terms": ["GFAP", "astrocyte"],
  "matched_calcium_terms": ["GCaMP"],
  "matched_modality_terms": ["two-photon"],
  "warnings": [],
  "recommended_next_step": "run_candidate_region_detection"
}
```

### Controlled Terms

Astrocyte terms:

```python
ASTRO_TERMS = ["astrocyte", "astrocytic", "glia", "glial", "GFAP", "Aldh1l1", "S100B", "SR101", "sulforhodamine"]
```

Calcium terms:

```python
CALCIUM_TERMS = ["GCaMP", "jGCaMP", "Cal-520", "Oregon Green", "Fluo"]
```

Modality terms:

```python
MODALITY_TERMS = ["two-photon", "2p", "miniscope", "widefield", "confocal"]
```

### Scoring Logic

Start at 0.0.

- Add 0.25 if astrocyte or glial terms are found.
- Add 0.15 if calcium indicator terms are found.
- Add 0.15 if optical imaging modality terms are found.
- Add 0.20 if raw movie appears available.
- Add 0.10 if masks or ROIs appear available.
- Add 0.10 if behavior or stimulus timing appears available.
- Add 0.05 if electrophysiology, LFP, or spikes appear available.
- Clamp final score to `[0, 1]`.

### Recommended Next Step

```text
score < 0.20:
  reject_low_value

0.20 <= score < 0.40:
  inspect_metadata

0.40 <= score < 0.60 and no raw movie:
  load_processed_traces

0.40 <= score < 0.75 and raw movie:
  run_candidate_region_detection

score >= 0.75:
  run_event_detection
```

### CLI

```bash
neuros-astro scan PATH --out report.json
```

---

## 10. Event Detection Scope

### Trace-Based Event Detection

Trace-based detection should operate on one or more dF/F traces.

```python
def detect_events_from_trace(
    trace: np.ndarray,
    frame_rate_hz: float,
    session_id: str,
    region_id: str | None = None,
    z_threshold: float = 2.0,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
) -> list[AstroEvent]:
    ...
```

Algorithm:

1. Accept a 1D trace.
2. Compute robust z-score using median and MAD.
3. Detect contiguous supra-threshold segments.
4. Merge events separated by short gaps.
5. Remove events shorter than `min_duration_s`.
6. Compute onset, offset, peak, duration, peak dF/F, and confidence.
7. Return `AstroEvent` objects.

Astrocyte calcium events are often slower than neuronal somatic calcium events, so defaults should prioritize slower event dynamics rather than spike-like detection.

### Movie-Based Candidate Event Detection

Movie-based detection should operate on a small 3D array with shape `[time, y, x]`.

```python
def detect_candidate_events_from_movie(
    movie: np.ndarray,
    frame_rate_hz: float,
    session_id: str,
    z_threshold: float = 3.0,
    min_area_px: int = 10,
    min_duration_s: float = 0.5,
    max_events: int | None = None,
) -> list[AstroEvent]:
    ...
```

Algorithm:

1. Validate movie shape.
2. Compute baseline per pixel using median across time.
3. Compute robust noise using MAD across time.
4. Convert to z-movie.
5. Threshold z-movie.
6. Find connected components per frame.
7. Filter components by area.
8. Link components across adjacent frames using centroid distance or mask overlap.
9. Convert linked components to `AstroEvent` objects.
10. Stop at `max_events` if specified.

This is a lightweight baseline detector, not a replacement for specialized astrocyte analysis methods.

---

## 11. Astrocyte Network Construction

Astrocyte events can be converted into time-windowed functional graphs. Nodes represent regions. Edges represent event coactivation, lagged coactivation, correlation, or mutual information.

MVP method: simple event coactivation using Jaccard overlap.

```text
edge_weight = bins where both regions active / bins where either region active
```

```python
def build_event_coactivation_graph(
    events: list[AstroEvent],
    session_id: str,
    frame_rate_hz: float,
    window_size_s: float = 30.0,
    stride_s: float = 5.0,
    bin_size_s: float = 1.0,
    min_edge_weight: float = 0.1,
) -> list[AstroGraph]:
    ...
```

Graph summary features:

- number of nodes
- number of edges
- density
- mean edge weight
- max edge weight
- mean degree
- max degree
- connected components

Future network methods can include:

- lagged event propagation
- temporal correlation
- mutual information
- graph neural network embeddings
- community detection
- local propagation fields
- astrocyte-neuron bipartite graphs

---

## 12. Tokenization Strategy

### Event Tokens

Each astrocyte event becomes an irregular timestamped token.

Suggested features:

```text
onset_time_s
duration_s
peak_dff
area_px
centroid_y
centroid_x
propagation_speed
direction_sin
direction_cos
confidence
```

Token shape:

```text
[n_events, feature_dim]
```

Timestamps:

```text
[n_events]
```

### Binned Astrocyte State Tokens

Events can also be aggregated into regular time bins.

Suggested features per bin:

```text
event_count
mean_peak_dff
total_area_px
mean_confidence
active_region_count
mean_duration_s
network_density
mean_edge_weight
```

### Graph Tokens

Graph-level windows can become tokens representing slow network state.

Suggested features:

```text
window_start_s
window_end_s
n_nodes
n_edges
density
mean_edge_weight
max_edge_weight
degree_mean
degree_max
connected_components
```

### neuroFMx Integration

`neuroFMx` should be able to load astro tokens as a modality:

```yaml
modalities:
  neural:
    enabled: true
  behavior:
    enabled: true
  astro:
    enabled: true
    token_path: examples/data/astro_tokens.npz
    sampling: irregular
    timestamp_key: timestamps_s
```

---

## 13. Storage and Export

### Event Tables

Use Parquet for event tables.

Columns:

```text
event_id
session_id
region_id
onset_frame
offset_frame
peak_frame
duration_s
peak_dff
area_px
centroid_y
centroid_x
propagation_speed
direction_rad
confidence
```

### Token Exports

Use `.npz` for simple model-ready demos.

Stored fields:

```text
tokens
timestamps_s
region_ids
feature_names
metadata_json
```

### Manifests

Use JSON manifests to describe modality files.

```json
{
  "session_id": "demo-session",
  "modalities": {
    "astro_events": {
      "type": "event_tokens",
      "path": "astro_tokens.npz",
      "sampling": "irregular",
      "timestamp_key": "timestamps_s"
    }
  },
  "source_dataset": "synthetic",
  "metadata": {}
}
```

### Future Exports

Future export support should include:

- NWB processing modules
- Zarr chunked arrays
- WebDataset shards
- Hugging Face dataset cards
- DANDI-compatible derived-data metadata

---

## 14. Command-Line Interface

The package should expose a CLI called:

```bash
neuros-astro
```

Commands:

```bash
neuros-astro scan PATH --out report.json
neuros-astro detect-trace-events traces.npy --frame-rate 10 --session-id demo --out events.parquet
neuros-astro detect-movie-events movie.npy --frame-rate 10 --session-id demo --out events.parquet
neuros-astro build-network events.parquet --frame-rate 10 --session-id demo --out graphs.json
neuros-astro tokenize-events events.parquet --frame-rate 10 --session-id demo --out astro_tokens.npz
neuros-astro generate-synthetic --out-dir examples/data --frame-rate 10
```

---

## 15. Initial Experiment Design

### Experiment 1: Synthetic Pipeline Validation

Question: Can the pipeline recover known synthetic astrocyte events and convert them into stable tokens?

Data: synthetic traces and movies.

Metrics:

- event detection precision against ground truth
- event detection recall against ground truth
- timing error
- duration error
- token export validity
- graph construction sanity checks

Expected outcome: the pipeline should recover obvious synthetic events and produce model-ready outputs.

### Experiment 2: Astro Token Ablation with Mock Neural Data

Question: Can astrocyte event tokens be aligned with mock neural tokens and used in a config-driven neuroFMx experiment?

Conditions:

```text
A. neural only
B. neural + behavior
C. neural + astro events
D. neural + astro graph state
```

Metrics:

- future neural prediction loss
- behavior-state prediction accuracy
- masked reconstruction loss
- temporal alignment correctness

Expected outcome: this validates infrastructure, not biological discovery.

### Experiment 3: NWB/DANDI Candidate Scan

Question: Which public datasets appear suitable for astrocyte reanalysis?

Data: NWB files or DANDI metadata only.

Metrics:

- astro reanalysis score
- raw movie availability
- processed trace availability
- behavior/ephys/stimulus availability
- recommended next step

Expected outcome: a ranked list of candidate datasets for manual inspection.

### Experiment 4: Real Dataset Pilot

Question: Can one real optical physiology dataset be reanalyzed into astrocyte-relevant event/network tokens?

Data: one candidate dataset with raw movie or processed traces.

Metrics:

- number of candidate events
- spatial distribution of events
- event duration statistics
- event coactivation graph structure
- alignment with behavior/stimulus/neural events if available

Expected outcome: a first real example notebook or script demonstrating package value.

---

## 16. Research Questions for Future Work

### Biological Questions

1. Do astrocyte event networks predict slow changes in neural activity?
2. Are astrocyte network states coupled to behavior, arousal, or task context?
3. Do astrocyte events precede, follow, or co-occur with changes in LFP/spiking/calcium activity?
4. Are astrocyte network features stable across sessions or animals?
5. Do astrocyte-derived latent states explain neural drift?
6. Can astrocyte networks serve as memory/context variables in foundation models?
7. Do astrocyte features improve generalization across subjects, regions, or behavioral states?

### Modeling Questions

1. Should astrocyte signals be represented as irregular event tokens or binned context tokens?
2. Should astrocyte graph state be fused early or late with neural tokens?
3. Do slow state-space models handle astrocyte tokens better than attention-only models?
4. Can contrastive objectives align astrocyte states with neural/behavioral context?
5. Can astrocyte tokens improve masked neural reconstruction?
6. Can astrocyte tokens stabilize cross-session latent manifolds?
7. Are astrocyte states interpretable as modulatory variables?

### Infrastructure Questions

1. Which NWB/DANDI datasets contain enough metadata to identify astrocyte relevance?
2. Which datasets contain raw movies that can be reprocessed?
3. Can derived astrocyte event tables be represented cleanly in NWB?
4. What is the best shard format for astrocyte tokens at scale?
5. Can dataset triage be automated across public archives?

---

## 17. Risks and Mitigations

### Risk 1: Most public datasets are neuron-centered

Mitigation:

- Make dataset triage the first MVP.
- Do not assume astrocyte identity unless metadata supports it.
- Classify datasets by confidence.
- Use synthetic data for pipeline validation.

### Risk 2: Astrocyte event detection is biologically nuanced

Mitigation:

- Start with baseline detectors.
- Add adapters to specialized tools later.
- Preserve confidence scores and warnings.
- Do not overclaim biological identity.

### Risk 3: Raw movies are large

Mitigation:

- Default to metadata scans.
- Avoid loading large arrays unless explicitly requested.
- Use lazy loading and chunked formats later.
- Add Zarr support for scalable processing.

### Risk 4: neuroFMx integration could become tangled

Mitigation:

- Keep `neuros-astro` standalone.
- Export simple NPZ/Parquet/manifest formats.
- Add only a thin neuroFMx adapter.

### Risk 5: Weak initial model signal

Mitigation:

- Begin with infrastructure validation.
- Use carefully controlled ablations.
- Start with tasks where slow context should matter: arousal, running, pupil, drift, long-window prediction.

---

## 18. Definition of Done for Initial Release

The first usable release is complete when:

```bash
pip install -e packages/neuros-astro
neuros-astro --help
neuros-astro generate-synthetic --out-dir examples/data
neuros-astro detect-trace-events examples/data/synthetic_traces.npy --frame-rate 10 --session-id synthetic --out examples/data/events.parquet
neuros-astro build-network examples/data/events.parquet --frame-rate 10 --session-id synthetic --out examples/data/graphs.json
neuros-astro tokenize-events examples/data/events.parquet --frame-rate 10 --session-id synthetic --out examples/data/astro_tokens.npz
pytest packages/neuros-astro/tests
```

And the package can produce:

- a dataset triage report
- event table
- graph summary
- token file
- neuroFMx manifest
- basic plots

---

## 19. Suggested First Commit Series

```text
docs: add neuros-astro whitepaper and implementation checklist
feat(neuros-astro): create package scaffold and CLI shell
feat(neuros-astro): add core schemas and validation tests
feat(neuros-astro): implement metadata-based dataset triage
feat(neuros-astro): add synthetic astrocyte trace and movie generators
feat(neuros-astro): implement trace-based calcium event detection
feat(neuros-astro): implement movie-based candidate event detection
feat(neuros-astro): build event coactivation graphs
feat(neuros-astro): tokenize astrocyte events for model ingestion
feat(neuros-astro): export event tables, token arrays, and neuroFMx manifests
feat(neurofm): add astro modality loader
```

---

## 20. Short Project Description

`neuros-astro` extracts astrocyte calcium events and astrocyte functional network states from optical physiology data, then converts them into model-ready tokens for multimodal neural foundation models.

## 21. Longer Project Description

`neuros-astro` is a modular analysis and modeling package for astrocyte-aware neuroscience foundation models. It scans NWB, DANDI, and local calcium imaging datasets for astrocyte reanalysis potential; extracts candidate astrocyte calcium events from traces and movies; builds event coactivation graphs; tokenizes astrocyte events and network states; and exports standardized outputs for integration with `neuroFMx`. The package is designed to test whether astrocyte-derived slow context signals improve prediction, generalization, and interpretability in multimodal models of neural and behavioral dynamics.

---

## 22. README Opening Draft

```markdown
# neuros-astro

`neuros-astro` is a glial signal processing layer for neural foundation models.

The package extracts astrocyte calcium events and astrocyte functional network states from optical physiology data, then converts them into model-ready tokens for multimodal neural modeling. It is designed to integrate with the broader `neurOS` and `neuroFMx` ecosystem as an `astro` modality.

## Why astrocytes?

Most neural foundation-model pipelines focus on spikes, LFP, calcium traces, behavior, video, and task variables. Astrocytes may provide a slower, spatially structured context signal that helps explain neural state, arousal, plasticity, behavioral context, and cross-session drift.

## MVP pipeline

neuros-astro scan session.nwb --out scan.json
neuros-astro detect-trace-events traces.npy --frame-rate 10 --session-id demo --out events.parquet
neuros-astro build-network events.parquet --frame-rate 10 --session-id demo --out graphs.json
neuros-astro tokenize-events events.parquet --frame-rate 10 --session-id demo --out astro_tokens.npz

## Main outputs

- dataset triage reports
- astrocyte event tables
- astrocyte coactivation graphs
- astrocyte event tokens
- binned astrocyte state tokens
- neuroFMx manifests
```

---

## 23. Final North Star

The endgame is not merely astrocyte analysis. The endgame is to test whether glial dynamics are a missing context field for brain-scale foundation models.

In the short term, `neuros-astro` should become a clean, reliable package that turns calcium imaging data into astrocyte event tokens.

In the medium term, it should become a dataset discovery and reanalysis engine for public optical physiology data.

In the long term, it should help answer whether astrocyte networks carry computationally meaningful latent state that improves neural prediction, behavioral decoding, and biological interpretability.

That is the real project: not just adding another modality, but giving the model a slower biological clock to listen to.
