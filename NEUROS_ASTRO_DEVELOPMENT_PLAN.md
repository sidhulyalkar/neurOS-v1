# neuros-astro: Structured Development Plan

## Executive Summary

This document provides a phased development roadmap for implementing **neuros-astro**, a glial signal processing layer for neural foundation models. The project will integrate astrocyte calcium dynamics as a new "astro" modality into the existing neuroFMx architecture.

**Total Estimated Duration**: 3-4 weeks (with one developer)

---

## Development Philosophy

### Core Principles
1. **Modular First**: `neuros-astro` is a standalone package with clean adapters
2. **Lightweight Baselines**: Simple, testable methods before specialized algorithms
3. **Metadata Matters**: Dataset triage is a first-class concern
4. **Event-Centric**: Astrocyte signals represented as spatiotemporal events
5. **Foundation-Model Ready**: All outputs support model ingestion
6. **Storage-Compatible**: Support NWB, Zarr, Parquet, NPZ, WebDataset

### Development Rules
- Read both whitepaper and implementation plan before starting each phase
- Implement one objective at a time
- Keep tests passing after each change
- Prefer small composable functions over monoliths
- Keep `neuros-astro` standalone; add only thin adapters to `neuroFMx`
- Write helpful error messages for optional dependencies
- Avoid loading large arrays unless explicitly requested
- Never overclaim astrocyte identity from unlabeled data

---

## Phase 1: Foundation (Days 1-3)

### Objective
Create the package scaffold, core schemas, and dataset triage system.

### Milestones

#### Milestone 1.1: Package Scaffold (Day 1)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Create `packages/neuros-astro/` directory structure
- [ ] Add `pyproject.toml` with dependencies
- [ ] Create modular subpackage structure:
  - `cli/` - Typer-based command-line interface
  - `metadata/` - Schemas and dataset scoring
  - `io/` - Data loaders and synthetic generators
  - `events/` - Event detection algorithms
  - `networks/` - Graph construction
  - `tokenization/` - Model-ready token generation
  - `export/` - Format converters
  - `visualization/` - Plotting utilities
- [ ] Add `tests/` directory with basic import tests
- [ ] Create CLI entry point `neuros-astro`
- [ ] Write initial README with project overview

**Acceptance Criteria**:
```bash
pip install -e packages/neuros-astro
neuros-astro --help  # Shows available commands
pytest packages/neuros-astro/tests  # All tests pass
```

**Dependencies**:
```toml
# Core dependencies
numpy
scipy
pandas
pydantic
typer
rich
networkx

# Optional extras
[nwb]: pynwb, hdmf
[dandi]: dandi
[imaging]: tifffile, scikit-image
[viz]: matplotlib, seaborn

# Dev dependencies
pytest
ruff
mypy
```

---

#### Milestone 1.2: Core Schemas (Day 2)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement `AstroSession` dataclass/model
- [ ] Implement `AstroRegion` dataclass/model
- [ ] Implement `AstroEvent` dataclass/model
- [ ] Implement `AstroGraph` dataclass/model
- [ ] Implement `TokenizedAstroSequence` dataclass/model
- [ ] Add controlled vocabulary terms (ASTRO_TERMS, CALCIUM_TERMS, MODALITY_TERMS)
- [ ] Create validation logic:
  - Confidence values in [0, 1]
  - Frame ordering (offset >= onset, peak between onset and offset)
  - Edge/weight count matching in graphs
  - Window ordering (end > start)
- [ ] Write comprehensive schema tests

**Acceptance Criteria**:
- Invalid confidence values raise `ValidationError`
- Invalid event frame ordering raises `ValidationError`
- Graph edge/weight mismatch raises `ValidationError`
- All schemas serialize to/from JSON-compatible dicts
- 100% test coverage for validation logic

**Key Files**:
- `neuros_astro/metadata/schema.py`
- `neuros_astro/metadata/controlled_terms.py`
- `tests/test_schema.py`

---

#### Milestone 1.3: Dataset Triage System (Day 3)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement `DatasetTriageResult` schema
- [ ] Create `score_dataset_metadata()` function
- [ ] Implement scoring algorithm (per whitepaper Section 9):
  - Base score: 0.0
  - +0.25 for astrocyte/glial terms
  - +0.15 for calcium indicator terms
  - +0.15 for optical imaging modality
  - +0.20 for raw movie availability
  - +0.10 for masks/ROIs
  - +0.10 for behavior/stimulus
  - +0.05 for ephys/LFP/spikes
  - Clamp to [0, 1]
- [ ] Add recommendation logic:
  - score < 0.20: reject_low_value
  - 0.20-0.40: inspect_metadata
  - 0.40-0.60 (no movie): load_processed_traces
  - 0.40-0.75 (with movie): run_candidate_region_detection
  - score >= 0.75: run_event_detection
- [ ] Implement CLI command: `neuros-astro scan PATH --out report.json`
- [ ] Add Rich output formatting
- [ ] Write triage tests

**Acceptance Criteria**:
- High-value astro metadata (GFAP + GCaMP + 2p + raw movie + behavior) scores > 0.8
- Neuron-only metadata scores < 0.3
- CLI produces valid JSON report
- Rich console output is readable and informative

**Key Files**:
- `neuros_astro/metadata/dataset_scoring.py`
- `neuros_astro/cli/main.py`
- `tests/test_dataset_scoring.py`

---

## Phase 2: Synthetic Data & Event Detection (Days 4-7)

### Objective
Generate synthetic astrocyte data and implement baseline event detection from traces and movies.

### Milestones

#### Milestone 2.1: Synthetic Data Generation (Day 4)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement `generate_synthetic_astro_traces()`:
  - Output shape: `[n_regions, n_time]`
  - Slow calcium-like transients (rise time ~1-3s, decay ~3-10s)
  - Variable event amplitudes
  - Occasional cross-region coactivation
  - Return traces + ground-truth event metadata
  - Deterministic with seed parameter
- [ ] Implement `generate_synthetic_astro_movie()`:
  - Output shape: `[time, height, width]`
  - Gaussian blobs that appear, expand, decay, propagate
  - Spatially realistic event sizes (10-100 pixels)
  - Return movie + ground-truth event metadata
  - Deterministic with seed parameter
- [ ] Create example script: `examples/00_generate_synthetic_astro_data.py`
- [ ] Add CLI command: `neuros-astro generate-synthetic --out-dir DIR --frame-rate FPS`
- [ ] Write synthetic data tests

**Acceptance Criteria**:
- Generated traces have expected shape
- Generated movies have expected shape
- Events are visible above noise (SNR > 3)
- Same seed produces identical outputs
- Ground-truth events are accurate

**Key Files**:
- `neuros_astro/io/synthetic.py`
- `examples/00_generate_synthetic_astro_data.py`
- `tests/test_synthetic.py`

---

#### Milestone 2.2: Trace-Based Event Detection (Day 5)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement `robust_zscore()` using median and MAD
- [ ] Implement `contiguous_regions()` for threshold crossings
- [ ] Implement `merge_close_regions()` for event merging
- [ ] Implement `detect_events_from_trace()`:
  - Parameters: z_threshold=2.0, min_duration_s=1.0, merge_gap_s=0.5
  - Returns list of `AstroEvent` objects
  - Computes onset, offset, peak, duration, peak_dff, confidence
- [ ] Implement `detect_events_from_traces()` for multi-region batch processing
- [ ] Handle edge cases (NaNs, flat traces, all-negative traces)
- [ ] Create example script: `examples/02_detect_events.py`
- [ ] Add CLI command: `neuros-astro detect-trace-events TRACES.npy --frame-rate FPS --session-id ID --out events.parquet`
- [ ] Write comprehensive event detection tests

**Acceptance Criteria**:
- Flat trace produces no events
- Synthetic slow events are detected with >90% recall
- Close events (<0.5s gap) merge correctly
- Distant events remain separate
- Multi-trace detection preserves region IDs
- NaNs handled gracefully (no crashes)

**Key Files**:
- `neuros_astro/events/event_detection.py`
- `neuros_astro/events/calcium_event_features.py`
- `examples/02_detect_events.py`
- `tests/test_event_detection.py`

---

#### Milestone 2.3: Movie-Based Event Detection (Days 6-7)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement per-pixel robust z-scoring
- [ ] Implement 2D connected components detection
- [ ] Implement component feature extraction (centroid, area, perimeter)
- [ ] Implement temporal component linking:
  - Link components across adjacent frames using:
    - Centroid distance threshold
    - Mask Jaccard overlap
  - Build temporal event tracks
- [ ] Implement `detect_candidate_events_from_movie()`:
  - Parameters: z_threshold=3.0, min_area_px=10, min_duration_s=0.5
  - Returns list of `AstroEvent` objects
  - Optional max_events limit for memory safety
- [ ] Handle edge cases (noise-only movie, single-frame events, NaNs)
- [ ] Add CLI command: `neuros-astro detect-movie-events MOVIE.npy --frame-rate FPS --session-id ID --out events.parquet`
- [ ] Write movie event detection tests

**Acceptance Criteria**:
- Single expanding blob produces one event
- Two separate blobs produce two events
- Noise-only movie produces no events
- Short events (< min_duration_s) are filtered
- NaNs do not crash detector
- Component linking maintains spatial continuity

**Key Files**:
- `neuros_astro/events/event_detection.py` (extended)
- `neuros_astro/segmentation/candidate_regions.py`
- `tests/test_event_detection.py` (extended)

---

## Phase 3: Networks & Tokenization (Days 8-11)

### Objective
Build astrocyte functional networks and convert events into model-ready tokens.

### Milestones

#### Milestone 3.1: Functional Network Construction (Days 8-9)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement `events_to_binary_matrix()`:
  - Converts event list to binary activity matrix [time_bins, regions]
  - Configurable bin size (default: 1.0s)
- [ ] Implement `build_event_coactivation_graph()`:
  - Sliding window approach (window_size_s=30.0, stride_s=5.0)
  - Jaccard coactivation metric: `bins_both_active / bins_either_active`
  - Edge filtering: min_edge_weight threshold
  - Returns list of `AstroGraph` objects (one per window)
- [ ] Implement `compute_graph_summary_features()`:
  - Node count, edge count
  - Density, mean edge weight, max edge weight
  - Mean degree, max degree
  - Connected components count
- [ ] Create example script: `examples/03_build_network.py`
- [ ] Add CLI command: `neuros-astro build-network events.parquet --frame-rate FPS --session-id ID --out graphs.json`
- [ ] Write network construction tests

**Acceptance Criteria**:
- Empty event list produces valid empty graphs
- Single region produces zero edges
- Two coactive regions produce one edge with correct weight
- Non-overlapping events produce no edges
- Sliding windows produce expected count: `(total_duration - window_size) / stride + 1`
- Graph features are numerically stable

**Key Files**:
- `neuros_astro/networks/functional_connectivity.py`
- `neuros_astro/networks/graph_features.py`
- `examples/03_build_network.py`
- `tests/test_graph_features.py`

---

#### Milestone 3.2: Event Tokenization (Day 10)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement `AstroEventTokenizer`:
  - Features per event:
    - onset_time_s
    - duration_s
    - peak_dff
    - area_px (normalized by mean area)
    - centroid_y (normalized by image height)
    - centroid_x (normalized by image width)
    - propagation_speed (if available)
    - direction_sin (circular encoding)
    - direction_cos (circular encoding)
    - confidence
  - Output shape: `[n_events, feature_dim]`
  - Timestamps: `[n_events]` (onset times)
  - Returns `TokenizedAstroSequence` object
- [ ] Implement normalization metadata storage
- [ ] Handle edge cases (empty events, missing features, divide-by-zero)
- [ ] Write tokenizer tests

**Acceptance Criteria**:
- Empty event list returns valid empty token object
- Event features encoded correctly
- Direction encoded as sin/cos (not raw radians)
- Normalization avoids divide-by-zero
- Token arrays have correct shape and dtype
- Metadata preserved in TokenizedAstroSequence

**Key Files**:
- `neuros_astro/tokenization/event_tokenizer.py`
- `tests/test_tokenizer.py`

---

#### Milestone 3.3: Binned Tokenization (Day 11)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement `BinnedAstroTokenizer`:
  - Aggregates events into regular time bins
  - Features per bin:
    - event_count
    - mean_peak_dff
    - total_area_px
    - mean_confidence
    - active_region_count
    - mean_duration_s
    - network_density (if graph provided)
    - mean_edge_weight (if graph provided)
  - Output shape: `[n_bins, feature_dim]`
  - Timestamps: `[n_bins]` (bin centers)
  - Returns `TokenizedAstroSequence` object
- [ ] Add CLI command: `neuros-astro tokenize-events events.parquet --frame-rate FPS --session-id ID --out tokens.npz`
- [ ] Write binned tokenizer tests

**Acceptance Criteria**:
- Binned tokenizer produces expected number of bins
- Empty bins have sensible default values (zeros or NaNs)
- Bin aggregation is numerically correct
- Works with and without graph features

**Key Files**:
- `neuros_astro/tokenization/astro_tokenizer.py`
- `tests/test_tokenizer.py` (extended)

---

## Phase 4: Export & Integration (Days 12-15)

### Objective
Implement data export formats and integrate astro modality into neuroFMx.

### Milestones

#### Milestone 4.1: Export Formats (Days 12-13)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement Parquet event export:
  - `save_events_parquet(events, path)`
  - `load_events_parquet(path) -> list[AstroEvent]`
  - Columns: event_id, session_id, region_id, onset_frame, offset_frame, peak_frame, duration_s, peak_dff, area_px, centroid_y, centroid_x, propagation_speed, direction_rad, confidence
- [ ] Implement NPZ token export:
  - `save_tokenized_astro_sequence_npz(sequence, path)`
  - `load_tokenized_astro_sequence_npz(path) -> TokenizedAstroSequence`
  - Arrays: tokens, timestamps_s, region_ids, feature_names
  - Metadata: JSON-encoded dict
- [ ] Implement neuroFMx manifest builder:
  - `build_neurofm_manifest(session_id, modalities_dict, metadata) -> dict`
  - Output format matches neuroFMx expectations
- [ ] Create example script: `examples/04_export_to_neurofm.py`
- [ ] Write export/import roundtrip tests

**Acceptance Criteria**:
- NPZ token export/import roundtrips without data loss
- Event Parquet export/import preserves all fields
- Manifest contains all required keys for neuroFMx
- File formats are readable by standard tools (pandas, numpy)

**Key Files**:
- `neuros_astro/export/to_parquet.py`
- `neuros_astro/export/to_neurofm.py`
- `examples/04_export_to_neurofm.py`
- `tests/test_export_schema.py`

---

#### Milestone 4.2: neuroFMx Adapter (Days 14-15)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Explore existing neuroFMx architecture:
  - Identify modality registry/config system
  - Identify tokenizer interfaces
  - Identify dataset loader patterns
  - Identify temporal alignment utilities
- [ ] Implement `AstroModalityConfig`:
  - Fields: enabled, token_path, sampling (regular/irregular), timestamp_key
- [ ] Implement `AstroModalityLoader`:
  - Loads `astro_tokens.npz`
  - Preserves timestamps
  - Converts to neuroFMx token format
  - Handles irregular sampling
- [ ] Register `astro` modality in neuroFMx registry
- [ ] Create example config:
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
- [ ] Write integration tests:
  - Can load astro tokens
  - Can align with mock neural sequence
  - Batching doesn't break existing modalities
  - Can enable/disable via config

**Acceptance Criteria**:
- `astro` modality appears in neuroFMx modality list
- Astro tokens load without errors
- Temporal alignment works correctly
- Existing modalities still function
- Config validation catches errors

**Key Files**:
- `packages/neuros-neurofm/neuros_neurofm/modalities/astro.py` (or similar)
- `packages/neuros-neurofm/configs/examples/astro_ablation.yaml`
- `packages/neuros-neurofm/tests/test_astro_modality.py`

---

## Phase 5: NWB/DANDI Integration (Days 16-19)

### Objective
Add support for NWB file metadata scanning and DANDI dataset discovery.

### Milestones

#### Milestone 5.1: NWB Metadata Loader (Days 16-17)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Add `pynwb` as optional dependency `[nwb]`
- [ ] Implement `summarize_nwb(path)`:
  - Returns metadata dict WITHOUT loading large arrays
  - Fields: session_id, session_description, identifier, institution, lab
  - Subject metadata (if available)
  - Acquisition object names/types
  - Processing module names
  - Imaging plane names, indicators, wavelengths
  - Device names
  - Interval tables
  - Keywords, experiment description
- [ ] Implement `list_ophys_series(path)`:
  - Lists available optical physiology data series
  - Returns metadata only (not arrays)
- [ ] Implement `load_roi_response_series(path, series_name)`:
  - Loads processed traces for event detection
  - Optional series_name (uses first if None)
  - Returns: traces array, metadata dict
- [ ] Add graceful error handling for missing `pynwb`:
  - Raise `OptionalDependencyError` with install instructions
  - Message: `pip install neuros-astro[nwb]`
- [ ] Update CLI `scan` command to detect .nwb files and use `summarize_nwb()`
- [ ] Write NWB loader tests (with mocked NWB files if needed)

**Acceptance Criteria**:
- NWB metadata scan doesn't load large arrays
- Missing `pynwb` raises helpful error
- CLI scan works on .nwb files
- Trace loading works for standard NWB formats
- Tests cover edge cases (missing fields, empty files)

**Key Files**:
- `neuros_astro/io/nwb_loader.py`
- `neuros_astro/cli/main.py` (updated)
- `tests/test_nwb_loader.py`

---

#### Milestone 5.2: DANDI Metadata Scanner (Days 18-19)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Add `dandi` as optional dependency `[dandi]`
- [ ] Implement `summarize_dandiset(dandiset_id)`:
  - Fetches dandiset metadata only (no downloads)
  - Returns: dandiset_id, name, description, keywords, species
  - Measurement techniques, asset counts, NWB asset count
- [ ] Implement `score_dandiset_for_astro(dandiset_id)`:
  - Uses existing `score_dataset_metadata()` on dandiset metadata
  - Searches for astro/calcium/modality terms
  - Returns `DatasetTriageResult`
- [ ] Add CLI command: `neuros-astro scan-dandiset DANDISET_ID --out report.json`
- [ ] Add graceful error handling for missing `dandi`:
  - Raise `OptionalDependencyError` with install instructions
  - Message: `pip install neuros-astro[dandi]`
- [ ] Write DANDI scanner tests (with mocked DANDI API responses)

**Acceptance Criteria**:
- DANDI scan doesn't download data files
- Missing `dandi` raises helpful error
- Matched terms are reported correctly
- Output format matches standard triage result schema
- Tests use mocked API responses (don't hit live DANDI)

**Key Files**:
- `neuros_astro/io/dandi_indexer.py`
- `neuros_astro/cli/main.py` (updated)
- `tests/test_dandi_indexer.py`

---

## Phase 6: Visualization & Documentation (Days 20-21)

### Objective
Add visualization tools and polish documentation.

### Milestones

#### Milestone 6.1: Visualization Tools (Day 20)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Implement `plot_event_raster(events, frame_rate_hz, ax=None)`:
  - Raster plot of events over time
  - Color by region or confidence
  - Returns figure/axis objects
- [ ] Implement `plot_event_feature_histograms(events)`:
  - Subplots for: duration, peak_dff, area, confidence
  - Returns figure object
- [ ] Implement `plot_astro_graph(graph)`:
  - Network visualization using networkx
  - Node positions based on centroids (if available)
  - Edge weights as line width or opacity
  - Returns figure/axis objects
- [ ] Implement `overlay_events_on_mean_image(movie, events, max_events=50)`:
  - Shows mean movie frame
  - Overlays event spatial footprints
  - Returns figure/axis objects
- [ ] Create example script: `examples/05_visualize_synthetic_outputs.py`
- [ ] Write visualization tests (verify functions return figures without crashes)

**Acceptance Criteria**:
- All plotting functions return matplotlib figure/axis objects
- Functions work headlessly (no display required)
- Synthetic outputs can be visualized
- No forced styling (allow user customization)
- Tests verify basic functionality

**Key Files**:
- `neuros_astro/visualization/event_raster.py`
- `neuros_astro/visualization/network_plot.py`
- `neuros_astro/visualization/movie_overlay.py`
- `examples/05_visualize_synthetic_outputs.py`
- `tests/test_visualization.py`

---

#### Milestone 6.2: Documentation Polish (Day 21)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Update README.md with:
  - Clear project description
  - Installation instructions (base + optional extras)
  - Quick start guide
  - CLI command reference
  - Example workflows
  - Links to whitepaper and implementation plan
- [ ] Add docstrings to all public functions:
  - Google or NumPy style
  - Type hints for all parameters
  - Example usage where helpful
- [ ] Create `CONTRIBUTING.md`:
  - Development setup
  - Testing guidelines
  - Code style (Ruff, MyPy)
  - Commit message format
- [ ] Create `CHANGELOG.md`:
  - Document v0.1.0 initial release
- [ ] Add example notebooks (if applicable):
  - End-to-end synthetic pipeline
  - Real dataset analysis (if available)

**Acceptance Criteria**:
- README provides clear getting-started path
- All public APIs have docstrings with type hints
- Documentation builds without warnings
- Examples run without errors

---

## Phase 7: Testing & Validation (Days 22-25)

### Objective
Comprehensive testing, real-world validation, and initial experiments.

### Milestones

#### Milestone 7.1: Integration Testing (Day 22)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] End-to-end synthetic pipeline test:
  - Generate synthetic data
  - Detect events
  - Build networks
  - Tokenize
  - Export
  - Load in neuroFMx
- [ ] Performance benchmarks:
  - Event detection on 1000-frame movie
  - Network construction on 1000 events
  - Tokenization on 10,000 events
- [ ] Memory profiling:
  - Ensure large arrays aren't duplicated
  - Check for memory leaks in event linking
- [ ] CLI integration tests:
  - Test all commands in sequence
  - Verify file outputs
  - Check error handling

**Acceptance Criteria**:
- End-to-end pipeline completes without errors
- Performance is reasonable (< 1 min for typical synthetic data)
- No memory leaks detected
- All CLI commands work correctly

---

#### Milestone 7.2: Real Dataset Validation (Days 23-24)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Scan 5-10 DANDI datasets for astro potential
- [ ] Select one high-scoring dataset
- [ ] Download sample session
- [ ] Run full pipeline:
  - Metadata scan
  - Event detection (from traces or movie)
  - Network construction
  - Tokenization
  - Visualization
- [ ] Create validation report:
  - Dataset description
  - Triage score and justification
  - Event statistics (count, duration dist, spatial dist)
  - Network properties
  - Visualizations
  - Biological interpretation notes
  - Caveats and limitations

**Acceptance Criteria**:
- Pipeline runs on real data without crashes
- Event detection produces reasonable results
- Network features are biologically plausible
- Validation report is thorough and honest about limitations

---

#### Milestone 7.3: Initial neuroFMx Experiments (Day 25)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Create four experimental configs:
  1. `neural_only_baseline.yaml` - Neural data alone
  2. `neural_behavior_baseline.yaml` - Neural + behavior
  3. `neural_behavior_astro_events.yaml` - Neural + behavior + astro events
  4. `neural_behavior_astro_graph.yaml` - Neural + behavior + astro graph features
- [ ] Define prediction tasks:
  - Future neural activity prediction
  - Behavioral state decoding
  - Arousal/running/pupil prediction
  - Masked modality reconstruction
- [ ] Run ablation experiment (if compute available):
  - Train all four configs
  - Compare validation metrics
  - Analyze ablation results
- [ ] Document experiment:
  - Hypothesis
  - Dataset details
  - Config differences
  - Results summary
  - Interpretation and next steps

**Acceptance Criteria**:
- All configs are valid and run without errors
- Ablation results are reproducible
- Documentation explains what each condition tests
- Results inform future research directions

---

## Phase 8: Release Preparation (Days 26-28)

### Objective
Finalize package for initial release and publication.

### Milestones

#### Milestone 8.1: Package Polish (Day 26)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Code quality pass:
  - Run Ruff linter and fix issues
  - Run MyPy type checker and fix issues
  - Ensure 100% test coverage for core modules
  - Remove debug code and TODOs
- [ ] Performance optimization:
  - Profile bottlenecks
  - Optimize hot loops
  - Add progress bars for long operations
- [ ] Error handling review:
  - Ensure helpful error messages
  - Add validation for all user inputs
  - Handle edge cases gracefully
- [ ] License and metadata:
  - Choose license (suggest MIT or Apache 2.0)
  - Update pyproject.toml metadata
  - Add author information

**Acceptance Criteria**:
- Ruff and MyPy pass with zero errors
- Test coverage > 90%
- All TODOs resolved or filed as issues
- License file present

---

#### Milestone 8.2: Release Documentation (Day 27)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] API reference documentation:
  - Sphinx or MkDocs setup
  - Auto-generated API docs from docstrings
  - Hand-written guides for key workflows
- [ ] Tutorial notebooks:
  - Getting started with synthetic data
  - Analyzing real datasets
  - Integrating with neuroFMx
  - Interpreting results
- [ ] FAQ document:
  - Common errors and solutions
  - Performance tips
  - Dataset recommendations
- [ ] Release notes for v0.1.0:
  - Feature summary
  - Known limitations
  - Future roadmap

**Acceptance Criteria**:
- Documentation builds without errors
- All examples run successfully
- FAQ addresses likely user questions
- Release notes are comprehensive

---

#### Milestone 8.3: Initial Release (Day 28)
**Status**: ⬜ Not Started

**Deliverables**:
- [ ] Version tagging:
  - Tag v0.1.0 in git
  - Update version in pyproject.toml
- [ ] Package distribution:
  - Build wheel and sdist
  - Test installation in clean environment
  - Consider PyPI upload (optional for initial release)
- [ ] Announcement materials:
  - Blog post or documentation page
  - Example outputs and visualizations
  - Key findings from initial experiments
- [ ] Community setup:
  - GitHub issues template
  - Pull request template
  - Contributing guidelines

**Acceptance Criteria**:
- Package installs cleanly via pip
- All examples work in fresh environment
- Documentation is accessible
- Community contribution path is clear

---

## Definition of Done: Initial Release

The **neuros-astro** v0.1.0 release is complete when:

### Functional Requirements
✅ All CLI commands work:
```bash
neuros-astro --help
neuros-astro scan PATH --out report.json
neuros-astro generate-synthetic --out-dir examples/data
neuros-astro detect-trace-events traces.npy --frame-rate 10 --session-id demo --out events.parquet
neuros-astro detect-movie-events movie.npy --frame-rate 10 --session-id demo --out events.parquet
neuros-astro build-network events.parquet --frame-rate 10 --session-id demo --out graphs.json
neuros-astro tokenize-events events.parquet --frame-rate 10 --session-id demo --out astro_tokens.npz
```

✅ Package produces:
- Dataset triage reports (JSON)
- Event tables (Parquet)
- Graph summaries (JSON)
- Token files (NPZ)
- neuroFMx manifests (JSON)
- Visualizations (PNG/PDF)

✅ Integration with neuroFMx:
- Astro modality loads successfully
- Temporal alignment works
- Can run ablation experiments

### Quality Requirements
✅ All tests pass: `pytest packages/neuros-astro/tests`
✅ Code quality: Ruff and MyPy pass
✅ Documentation: README, API docs, examples complete
✅ Real-world validation: Pipeline runs on at least one real dataset

---

## Future Roadmap (Post v0.1.0)

### Short-Term (v0.2.0)
- [ ] Advanced event detection adapters (AQuA, astroCaST, ASTRA)
- [ ] Improved segmentation (astrocyte morphology features)
- [ ] Additional network metrics (lagged correlation, mutual information)
- [ ] Zarr storage backend for large datasets
- [ ] NWB writing support (derived data)

### Medium-Term (v0.3.0)
- [ ] Graph neural network embeddings for astro networks
- [ ] Astrocyte-neuron bipartite graphs
- [ ] Propagation field analysis
- [ ] WebDataset shards for large-scale training
- [ ] Multi-session alignment and drift analysis

### Long-Term (v1.0.0)
- [ ] Automated DANDI dataset crawler
- [ ] Deep learning segmentation models
- [ ] Cross-modal astro-neural analysis
- [ ] Foundation model pretraining with astro modality
- [ ] Publication-ready analysis pipelines

---

## Risk Mitigation

### Risk 1: Public datasets lack astrocyte labels
**Mitigation**:
- Triage system ranks datasets conservatively
- Use confidence scores for all detections
- Don't overclaim biological identity
- Validate on datasets with known astrocyte labeling

### Risk 2: Event detection is biologically complex
**Mitigation**:
- Start with simple baseline detectors
- Preserve ground-truth comparison when available
- Add adapters to specialized tools later
- Document assumptions and limitations

### Risk 3: Raw movies are very large
**Mitigation**:
- Default to metadata-only scans
- Use lazy loading (Zarr, NWB)
- Add memory limits and progress bars
- Support chunked processing

### Risk 4: Weak signal in foundation model
**Mitigation**:
- Start with carefully controlled synthetic experiments
- Focus on tasks where slow context matters (arousal, drift, state)
- Use appropriate baselines (neural-only, behavior-only)
- Interpret null results as scientifically valuable

---

## Success Metrics

### Technical Metrics
- [ ] Package installs without errors in clean environment
- [ ] 100% of core tests pass
- [ ] Code coverage > 90%
- [ ] CLI commands complete in reasonable time (< 5 min for typical data)

### Scientific Metrics
- [ ] Event detection achieves >85% precision/recall on synthetic data
- [ ] Network features are stable across parameter variations
- [ ] Tokenization preserves event information
- [ ] Astro modality loads successfully in neuroFMx

### Impact Metrics
- [ ] Pipeline runs on at least 3 real datasets successfully
- [ ] Validation report demonstrates biological plausibility
- [ ] At least one successful ablation experiment
- [ ] Documentation enables independent user to complete workflow

---

## Development Commands Quick Reference

```bash
# Setup
cd /mnt/c/Users/sidso/Documents/neurOS-v1
pip install -e packages/neuros-astro
pip install -e packages/neuros-astro[nwb,dandi,viz]

# Testing
pytest packages/neuros-astro/tests
pytest packages/neuros-astro/tests -v --cov=neuros_astro

# Code quality
ruff check packages/neuros-astro
ruff format packages/neuros-astro
mypy packages/neuros-astro/neuros_astro

# End-to-end synthetic pipeline
neuros-astro generate-synthetic --out-dir examples/data --frame-rate 10
neuros-astro detect-trace-events examples/data/synthetic_traces.npy \
  --frame-rate 10 --session-id synthetic --out examples/data/events.parquet
neuros-astro build-network examples/data/events.parquet \
  --frame-rate 10 --session-id synthetic --out examples/data/graphs.json
neuros-astro tokenize-events examples/data/events.parquet \
  --frame-rate 10 --session-id synthetic --out examples/data/astro_tokens.npz

# Documentation
cd packages/neuros-astro/docs
make html
```

---

## Contact and Collaboration

**Primary Developer**: [Your Name]
**Repository**: `neurOS-v1/packages/neuros-astro`
**Documentation**: See `docs/neuros_astro_whitepaper.md` for scientific background
**Issues**: File bugs and feature requests in GitHub Issues
**Discussions**: Use GitHub Discussions for questions and ideas

---

## Conclusion

This structured development plan provides a clear roadmap from package scaffold to initial release. The phased approach ensures:

1. **Incremental progress**: Each milestone builds on previous work
2. **Testability**: Tests accompany each feature
3. **Validation**: Real-world datasets validate the approach
4. **Integration**: Clean adapters into existing neuroFMx infrastructure
5. **Documentation**: Users can understand and extend the work

The ultimate goal is not just building a package, but answering a scientific question: **Do astrocyte dynamics provide a computationally meaningful context signal that improves neural foundation models?**

This plan gives you the infrastructure to answer that question rigorously.
