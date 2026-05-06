# neuros-astro Implementation Plan and Claude Code Prompt Pack

## Purpose

This document is designed to be placed in the repository so Claude Code can understand the entire `neuros-astro` project cohesively before modifying code. It pairs with:

```text
docs/neuros_astro_whitepaper.md
```

The whitepaper explains the scientific and architectural intent. This file gives concrete milestones, objectives, implementation requirements, acceptance criteria, CLI targets, and copy-paste Claude Code prompts.

---

## 1. Project Objective Summary

Build `neuros-astro`, a standalone Python package inside `neurOS-v1/packages/`, with adapters into `neuros-neurofm`.

The package should:

1. scan NWB/DANDI/local calcium imaging datasets for astrocyte reanalysis potential
2. score datasets using metadata and availability of raw movies, masks, traces, behavior, and ephys
3. generate synthetic astrocyte trace/movie data for testing
4. detect candidate astrocyte calcium events from traces
5. detect candidate spatiotemporal calcium events from movies
6. build astrocyte coactivation graphs
7. tokenize astrocyte events and graph summaries
8. export event tables, token arrays, and neuroFMx manifests
9. expose a composable CLI
10. integrate astro tokens into `neuroFMx` as a modality named `astro`

---

## 2. Development Objectives and Acceptance Criteria

## Objective 1: Create the Package Scaffold

### Goal

Create a clean Python package with tests, CLI, and modular subpackages.

### Deliverables

- `packages/neuros-astro/pyproject.toml`
- `packages/neuros-astro/README.md`
- `packages/neuros-astro/neuros_astro/`
- `packages/neuros-astro/tests/`
- Typer CLI entry point
- basic import tests

### Acceptance Criteria

```bash
pip install -e packages/neuros-astro
neuros-astro --help
pytest packages/neuros-astro/tests
```

All commands should run without errors.

---

## Objective 2: Implement Core Schemas

### Goal

Create typed objects for sessions, regions, events, graphs, and tokenized sequences.

### Deliverables

- `AstroSession`
- `AstroRegion`
- `AstroEvent`
- `AstroGraph`
- `TokenizedAstroSequence`
- validation tests

### Acceptance Criteria

- invalid confidence values fail validation
- invalid event frame ordering fails validation
- graph edge count and weight count must match
- schemas serialize to/from dictionaries or JSON-compatible structures

---

## Objective 3: Build Dataset Triage

### Goal

Score datasets for astrocyte reanalysis potential using metadata, paths, and optional NWB summaries.

### Deliverables

- controlled terms
- scoring function
- triage result schema
- CLI `scan` command
- JSON output

### Acceptance Criteria

- metadata containing GFAP, GCaMP, two-photon, raw movie, behavior gets high score
- neuron-only metadata gets lower score
- raw movie availability changes recommended next step
- CLI writes valid JSON report

---

## Objective 4: Generate Synthetic Astrocyte Data

### Goal

Create testable synthetic traces and movies with slow calcium events.

### Deliverables

- synthetic trace generator
- synthetic movie generator
- example script
- deterministic seeding

### Acceptance Criteria

- generated traces have shape `[n_regions, n_time]`
- generated movies have shape `[time, height, width]`
- events are visible above noise
- same seed gives identical output

---

## Objective 5: Implement Trace-Based Event Detection

### Goal

Detect candidate astrocyte calcium events from processed traces.

### Deliverables

- robust z-score
- contiguous region detection
- event merging
- event filtering
- multi-trace detection

### Acceptance Criteria

- flat traces produce no events
- synthetic slow events are detected
- close events merge correctly
- distant events remain separate
- NaNs are handled gracefully

---

## Objective 6: Implement Movie-Based Candidate Event Detection

### Goal

Detect candidate spatiotemporal events from small calcium movies.

### Deliverables

- per-pixel robust z-scoring
- connected components
- component linking across frames
- event conversion

### Acceptance Criteria

- synthetic expanding blob produces an event
- two separate blobs produce two events
- noise-only movie produces no events
- short events are filtered out
- NaNs do not crash detection

---

## Objective 7: Build Astrocyte Functional Networks

### Goal

Convert events into time-windowed coactivation graphs.

### Deliverables

- event-to-binary-matrix conversion
- windowed coactivation graph construction
- graph summary features

### Acceptance Criteria

- coactive regions create edges
- non-overlapping regions do not create edges
- single-region graphs have no edges
- sliding windows produce expected count
- graph summary features are stable

---

## Objective 8: Implement Tokenization

### Goal

Convert events and graph features into neuroFMx-ready token arrays.

### Deliverables

- `AstroEventTokenizer`
- `BinnedAstroTokenizer`
- graph tokenization utilities
- normalization metadata

### Acceptance Criteria

- empty event list returns valid empty token object
- event features are encoded correctly
- direction is encoded as sine/cosine
- normalization avoids divide-by-zero
- binned tokenizer gives expected number of bins

---

## Objective 9: Add Exports

### Goal

Save and load event tables, token arrays, and neuroFMx manifests.

### Deliverables

- Parquet event export
- NPZ token export
- NPZ token load
- manifest builder

### Acceptance Criteria

- token export roundtrips without data loss
- event dataframe contains expected columns
- manifest contains required modality keys
- outputs are easy for `neuroFMx` to consume

---

## Objective 10: Add neuroFMx Adapter

### Goal

Allow `neuroFMx` to load astrocyte tokens as a modality called `astro`.

### Deliverables

- astro modality config
- astro token loader
- timestamp preservation
- example config
- minimal integration tests

### Acceptance Criteria

- `astro_tokens.npz` loads into existing model input format
- astro tokens align with mock neural sequence
- batching does not break existing modalities
- `astro` can be enabled/disabled via config

---

## Objective 11: Add NWB Metadata Support

### Goal

Safely summarize NWB optical physiology metadata without loading large arrays.

### Deliverables

- optional `pynwb` dependency
- `summarize_nwb`
- `list_ophys_series`
- optional processed trace loader

### Acceptance Criteria

- missing `pynwb` gives helpful install error
- large arrays are not loaded during metadata scan
- acquisition, processing modules, devices, imaging planes, intervals, and subject metadata are summarized
- CLI scan uses NWB metadata when available

---

## Objective 12: Add DANDI Metadata Scanner

### Goal

Score DANDI dandisets for astrocyte reanalysis potential using metadata only.

### Deliverables

- optional `dandi` dependency
- dandiset metadata summary
- dandiset scoring command

### Acceptance Criteria

- missing `dandi` gives helpful install error
- no data is downloaded
- matched terms are reported
- DANDI scan produces same triage result schema

---

## Objective 13: Add Visualization Tools

### Goal

Make pipeline outputs inspectable.

### Deliverables

- event raster plot
- event feature histograms
- graph plot
- event overlay on mean image

### Acceptance Criteria

- functions return matplotlib figure/axis objects
- functions work headlessly
- synthetic outputs can be visualized
- no forced styling assumptions

---

## 3. Milestone Plan

## Milestone 0: Repository Foundation

Duration: 0.5 to 1 day

Deliverables:

- package scaffold
- pyproject
- CLI shell
- README
- basic tests

Exit condition:

```bash
pip install -e packages/neuros-astro
neuros-astro --help
pytest packages/neuros-astro/tests
```

## Milestone 1: Dataset Triage MVP

Duration: 1 day

Deliverables:

- schemas
- controlled terms
- scoring logic
- scan CLI
- JSON output

Exit condition:

```bash
neuros-astro scan examples/metadata/demo_astro.json --out scan.json
```

## Milestone 2: Synthetic Data and Trace Events

Duration: 1 day

Deliverables:

- synthetic trace generator
- trace event detector
- event Parquet export

Exit condition:

```bash
neuros-astro generate-synthetic --out-dir examples/data
neuros-astro detect-trace-events examples/data/synthetic_traces.npy --frame-rate 10 --session-id synthetic --out examples/data/events.parquet
```

## Milestone 3: Movie Event Baseline

Duration: 1 to 2 days

Deliverables:

- synthetic movie generator
- movie event detector
- connected component linking

Exit condition:

```bash
neuros-astro detect-movie-events examples/data/synthetic_movie.npy --frame-rate 10 --session-id synthetic --out examples/data/movie_events.parquet
```

## Milestone 4: Networks and Tokenization

Duration: 1 to 2 days

Deliverables:

- coactivation graphs
- graph summaries
- event tokenizer
- binned tokenizer
- NPZ export

Exit condition:

```bash
neuros-astro build-network examples/data/events.parquet --frame-rate 10 --session-id synthetic --out examples/data/graphs.json
neuros-astro tokenize-events examples/data/events.parquet --frame-rate 10 --session-id synthetic --out examples/data/astro_tokens.npz
```

## Milestone 5: neuroFMx Adapter

Duration: 1 to 2 days

Deliverables:

- astro modality config
- astro token loader
- example ablation config
- integration tests

Exit condition:

`neuroFMx` can load `astro_tokens.npz` as a modality called `astro`.

## Milestone 6: NWB/DANDI Integration

Duration: 2 to 4 days

Deliverables:

- safe NWB metadata summary
- optional processed trace extraction
- DANDI metadata scanner
- candidate dataset reports

Exit condition:

```bash
neuros-astro scan session.nwb --out session_scan.json
neuros-astro scan-dandiset DANDISET_ID --out dandiset_scan.json
```

---

## 4. Target Repository Files

```text
neurOS-v1/
  docs/
    neuros_astro_whitepaper.md
    neuros_astro_implementation_plan.md
  packages/
    neuros-astro/
      pyproject.toml
      README.md
      neuros_astro/
        __init__.py
        cli/
          __init__.py
          main.py
        metadata/
          __init__.py
          schema.py
          controlled_terms.py
          dataset_scoring.py
        io/
          __init__.py
          synthetic.py
          nwb_loader.py
          dandi_indexer.py
        events/
          __init__.py
          event_detection.py
          calcium_event_features.py
        segmentation/
          __init__.py
          candidate_regions.py
        networks/
          __init__.py
          functional_connectivity.py
          graph_features.py
        tokenization/
          __init__.py
          astro_tokenizer.py
          event_tokenizer.py
          graph_tokenizer.py
        export/
          __init__.py
          to_neurofm.py
          to_parquet.py
        visualization/
          __init__.py
          event_raster.py
          network_plot.py
          movie_overlay.py
      configs/
        dataset_triage.yaml
        astro_event_detection.yaml
        neurofm_export.yaml
      examples/
        00_generate_synthetic_astro_data.py
        01_scan_dataset.py
        02_detect_events.py
        03_build_network.py
        04_export_to_neurofm.py
      tests/
        test_imports.py
        test_cli.py
        test_schema.py
        test_dataset_scoring.py
        test_synthetic.py
        test_event_detection.py
        test_graph_features.py
        test_tokenizer.py
        test_export_schema.py
```

---

## 5. CLI Target Commands

```bash
neuros-astro --help

neuros-astro scan PATH --out report.json

neuros-astro generate-synthetic \
  --out-dir examples/data \
  --frame-rate 10

neuros-astro detect-trace-events examples/data/synthetic_traces.npy \
  --frame-rate 10 \
  --session-id synthetic \
  --out examples/data/events.parquet

neuros-astro detect-movie-events examples/data/synthetic_movie.npy \
  --frame-rate 10 \
  --session-id synthetic \
  --out examples/data/movie_events.parquet

neuros-astro build-network examples/data/events.parquet \
  --frame-rate 10 \
  --session-id synthetic \
  --out examples/data/graphs.json

neuros-astro tokenize-events examples/data/events.parquet \
  --frame-rate 10 \
  --session-id synthetic \
  --out examples/data/astro_tokens.npz
```

---

## 6. Claude Code Prompt A: Add Documentation to Repository

```text
You are working in my neurOS-v1 repository.

Create two documentation files:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

The whitepaper should explain the scientific motivation, core hypotheses, neurOS/neuroFMx integration, architecture, data objects, dataset triage design, event detection strategy, network construction, tokenization strategy, storage/export plan, CLI goals, experiment design, risks, and final north star.

The implementation plan should organize the project into milestones with checkboxes, expected files, commands to run, acceptance criteria, and Claude Code prompts for each development phase.

Do not implement package code yet. This commit is documentation only.

After editing, show:
1. files created
2. brief summary of each document
3. recommended first implementation step
```

---

## 7. Claude Code Prompt B: Create Package Scaffold

```text
You are working in my neurOS-v1 repository.

Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Now implement Milestone 0 only: repository foundation.

Create a standalone package:

    packages/neuros-astro/

Required structure:

    packages/neuros-astro/
      pyproject.toml
      README.md
      neuros_astro/
        __init__.py
        cli/
          __init__.py
          main.py
        metadata/
          __init__.py
          schema.py
          controlled_terms.py
          dataset_scoring.py
        io/
          __init__.py
        events/
          __init__.py
        networks/
          __init__.py
        tokenization/
          __init__.py
        export/
          __init__.py
        visualization/
          __init__.py
      tests/
        test_imports.py
        test_cli.py

Requirements:
- Python >=3.10
- Use pyproject.toml
- Add dependencies: numpy, scipy, pandas, pydantic, typer, rich, networkx
- Add optional extras: nwb, dandi, tiff, imaging
- Add dev dependencies: pytest, ruff, mypy
- Add CLI entry point: `neuros-astro`
- CLI should support `neuros-astro --help`
- Add minimal package docstring
- Add README with project purpose and MVP roadmap

Do not implement heavy algorithms yet.

Run tests if possible.

After editing, show:
1. tree of files created
2. install command
3. test command
4. CLI command
5. any issues or assumptions
```

---

## 8. Claude Code Prompt C: Implement Schemas

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 2: core schemas.

Files:

    packages/neuros-astro/neuros_astro/metadata/schema.py
    packages/neuros-astro/neuros_astro/metadata/controlled_terms.py
    packages/neuros-astro/tests/test_schema.py

Create models:
- AstroSession
- AstroRegion
- AstroEvent
- AstroGraph
- TokenizedAstroSequence

Use Pydantic if it keeps validation clean. Dataclasses are acceptable only if validation is still robust.

Validation requirements:
- confidence in [0, 1]
- metadata_score in [0, 1]
- offset_frame >= onset_frame
- peak_frame between onset_frame and offset_frame
- AstroGraph edge_weights length equals edges length
- window_end_s > window_start_s

Controlled terms:
- ASTRO_TERMS
- CALCIUM_TERMS
- MODALITY_TERMS

Add tests for valid and invalid objects.

Run pytest for the package and fix issues.
```

---

## 9. Claude Code Prompt D: Implement Dataset Triage

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 3: dataset triage.

Files:

    packages/neuros-astro/neuros_astro/metadata/dataset_scoring.py
    packages/neuros-astro/neuros_astro/cli/main.py
    packages/neuros-astro/tests/test_dataset_scoring.py

Create:

    DatasetTriageResult
    score_dataset_metadata(metadata)

The scoring logic should match the whitepaper exactly.

Add CLI command:

    neuros-astro scan PATH --out report.json

Behavior:
- If PATH is JSON, load it and score metadata.
- If PATH is plain text, score text.
- If PATH is any other file/path, score based on path string and simple sidecar hints.
- Save JSON when --out is passed.
- Print a Rich summary.

Tests:
- high-value astro metadata
- neuron-only metadata
- metadata with behavior/ephys
- CLI using a temporary JSON file

Keep the implementation robust and lightweight.
```

---

## 10. Claude Code Prompt E: Add Synthetic Data

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 4: synthetic astrocyte data.

Files:

    packages/neuros-astro/neuros_astro/io/synthetic.py
    packages/neuros-astro/examples/00_generate_synthetic_astro_data.py
    packages/neuros-astro/tests/test_synthetic.py

Implement:

    generate_synthetic_astro_traces
    generate_synthetic_astro_movie

Trace generator:
- output shape [n_regions, n_time]
- events should be slow calcium-like transients
- include variable rise and decay times
- include occasional coactivation across regions
- return traces and ground-truth event dictionaries
- deterministic with seed

Movie generator:
- output shape [time, height, width]
- Gaussian blobs appear, expand, decay, and optionally propagate
- return movie and ground-truth event dictionaries
- deterministic with seed

Add tests for shape, determinism, and event visibility.
```

---

## 11. Claude Code Prompt F: Implement Trace Event Detection

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 5: trace-based event detection.

Files:

    packages/neuros-astro/neuros_astro/events/event_detection.py
    packages/neuros-astro/neuros_astro/events/calcium_event_features.py
    packages/neuros-astro/tests/test_event_detection.py
    packages/neuros-astro/examples/02_detect_events.py

Implement:

    robust_zscore
    contiguous_regions
    merge_close_regions
    detect_events_from_trace
    detect_events_from_traces

Use conservative defaults suitable for slow astrocyte calcium events.

Tests:
- flat trace gives no events
- one synthetic slow event is detected
- two distant events remain separate
- two close events merge
- multi-trace detection preserves region IDs
- NaNs are handled gracefully

Run tests and fix issues.
```

---

## 12. Claude Code Prompt G: Implement Movie Event Detection

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 6: movie-based candidate event detection.

Files:

    packages/neuros-astro/neuros_astro/events/event_detection.py
    packages/neuros-astro/neuros_astro/segmentation/candidate_regions.py
    packages/neuros-astro/tests/test_event_detection.py

Implement:

    detect_candidate_events_from_movie
    connected_components_2d
    component_features

Algorithm:
- movie shape [T, Y, X]
- median baseline per pixel
- MAD noise per pixel
- z-movie thresholding
- connected components per frame
- filter by min_area_px
- link components across adjacent frames using centroid distance or mask overlap
- convert linked components to AstroEvent objects

Tests:
- one synthetic expanding blob gives one event
- two separate blobs give two events
- noise-only movie gives no events
- too-short event is filtered
- NaNs do not crash

Keep this as a baseline, not a heavy scientific algorithm.
```

---

## 13. Claude Code Prompt H: Implement Networks

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 7: astrocyte functional networks.

Files:

    packages/neuros-astro/neuros_astro/networks/functional_connectivity.py
    packages/neuros-astro/neuros_astro/networks/graph_features.py
    packages/neuros-astro/tests/test_graph_features.py
    packages/neuros-astro/examples/03_build_network.py

Implement:

    events_to_binary_matrix
    build_event_coactivation_graph
    compute_graph_summary_features

Use Jaccard event coactivation as the MVP edge weight.

Tests:
- empty event list
- single region gives no edges
- two coactive regions gives one edge
- non-overlapping events give no edge
- sliding windows produce expected number of graphs

Use networkx if available, with simple fallback for connected components.
```

---

## 14. Claude Code Prompt I: Implement Tokenization

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 8: tokenization.

Files:

    packages/neuros-astro/neuros_astro/tokenization/event_tokenizer.py
    packages/neuros-astro/neuros_astro/tokenization/astro_tokenizer.py
    packages/neuros-astro/tests/test_tokenizer.py
    packages/neuros-astro/examples/04_export_to_neurofm.py

Implement:

    TokenizedAstroSequence
    AstroEventTokenizer
    BinnedAstroTokenizer

AstroEventTokenizer features:
- onset_time_s
- duration_s
- peak_dff
- area_px
- centroid_y
- centroid_x
- propagation_speed
- direction_sin
- direction_cos
- confidence

BinnedAstroTokenizer features:
- event_count
- mean_peak_dff
- total_area_px
- mean_confidence
- active_region_count

Tests:
- empty event list
- single event
- direction sin/cos
- normalization stability
- expected bin counts

Use NumPy only. Do not add PyTorch dependency here.
```

---

## 15. Claude Code Prompt J: Implement Exports

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 9: exports.

Files:

    packages/neuros-astro/neuros_astro/export/to_neurofm.py
    packages/neuros-astro/neuros_astro/export/to_parquet.py
    packages/neuros-astro/tests/test_export_schema.py

Implement:

    save_tokenized_astro_sequence_npz
    load_tokenized_astro_sequence_npz
    events_to_dataframe
    dataframe_to_events
    save_events_parquet
    load_events_parquet
    build_neurofm_manifest

Tests:
- NPZ token roundtrip
- event dataframe columns
- dataframe to events roundtrip
- manifest required keys

Keep file formats simple and robust.
```

---

## 16. Claude Code Prompt K: Expand CLI Pipeline

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 9 and CLI pipeline commands.

Files:

    packages/neuros-astro/neuros_astro/cli/main.py
    packages/neuros-astro/tests/test_cli.py
    packages/neuros-astro/README.md

Add commands:

    neuros-astro scan PATH --out report.json
    neuros-astro generate-synthetic --out-dir examples/data --frame-rate 10
    neuros-astro detect-trace-events TRACES_NPY --frame-rate 10 --session-id demo --out events.parquet
    neuros-astro detect-movie-events MOVIE_NPY --frame-rate 10 --session-id demo --out events.parquet
    neuros-astro build-network EVENTS_PARQUET --frame-rate 10 --session-id demo --out graphs.json
    neuros-astro tokenize-events EVENTS_PARQUET --frame-rate 10 --session-id demo --out astro_tokens.npz

Use Rich output and helpful errors.

Add Typer CliRunner tests where possible.

Update README with a complete synthetic end-to-end demo.
```

---

## 17. Claude Code Prompt L: Add neuroFMx Adapter

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Now inspect the existing repository for `packages/neuros-neurofm`.

Find:
- tokenizer interfaces
- modality registries
- dataset loaders
- config system
- token sequence classes
- temporal alignment utilities

Implement the lightest clean integration for an `astro` modality.

Requirements:
1. Add an AstroModalityConfig or equivalent config object.
2. Add a loader for `astro_tokens.npz` generated by neuros-astro.
3. Preserve timestamps.
4. Convert NumPy token arrays to the token format expected by neuroFMx.
5. Add astro modality to registry if a registry exists.
6. Add an example config:

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

7. Add tests:
- can load astro tokens
- can align astro tokens with mock neural sequence
- can batch astro tokens without breaking existing modalities
- astro modality can be disabled

Do not refactor the whole package. Make minimal clean changes.
```

---

## 18. Claude Code Prompt M: Add NWB Metadata Loader

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 11: NWB metadata support.

Files:

    packages/neuros-astro/neuros_astro/io/nwb_loader.py
    packages/neuros-astro/tests/test_nwb_loader.py

Requirements:
1. Do not load huge arrays by default.
2. Use pynwb only if installed.
3. If pynwb is not installed, raise OptionalDependencyError with:
   `pip install neuros-astro[nwb]`

Implement:

    summarize_nwb(path: str | Path) -> dict[str, Any]
    list_ophys_series(path: str | Path) -> list[dict[str, Any]]
    load_roi_response_series(path: str | Path, series_name: str | None = None) -> tuple[np.ndarray, dict[str, Any]]

`summarize_nwb` should return high-level metadata only:
- session_id
- session_description
- identifier
- institution
- lab
- subject fields if available
- acquisition object names and types
- processing module names
- imaging plane names
- devices
- intervals
- keywords / experiment description if available

Tests can use mocks if creating real NWB files is too heavy.

Update CLI scan to use summarize_nwb for `.nwb` files.
```

---

## 19. Claude Code Prompt N: Add DANDI Metadata Scanner

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 12: DANDI metadata scanner.

Files:

    packages/neuros-astro/neuros_astro/io/dandi_indexer.py
    packages/neuros-astro/neuros_astro/cli/main.py
    packages/neuros-astro/tests/test_dandi_indexer.py

Goal:
Search or inspect DANDI dandisets for astrocyte-reanalysis candidates using metadata only.

Requirements:
1. Use the `dandi` Python client as an optional dependency.
2. If not installed, raise OptionalDependencyError with:
   `pip install neuros-astro[dandi]`
3. Avoid downloading data. Metadata only.

Implement:

    summarize_dandiset(dandiset_id: str) -> dict[str, Any]
    score_dandiset_for_astro(dandiset_id: str) -> DatasetTriageResult

CLI:

    neuros-astro scan-dandiset DANDISET_ID --out report.json

Return:
- dandiset_id
- name
- description
- keywords
- species if available
- measurement techniques if available
- asset count summary
- nwb asset count
- matched astro/calcium/modality terms

Add tests with mocked DANDI client responses.
```

---

## 20. Claude Code Prompt O: Add Visualization Tools

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Implement Objective 13: visualization tools.

Files:

    packages/neuros-astro/neuros_astro/visualization/event_raster.py
    packages/neuros-astro/neuros_astro/visualization/network_plot.py
    packages/neuros-astro/neuros_astro/visualization/movie_overlay.py
    packages/neuros-astro/examples/05_visualize_synthetic_outputs.py

Implement:

    plot_event_raster(events, frame_rate_hz, ax=None)
    plot_event_feature_histograms(events)
    plot_astro_graph(graph)
    overlay_events_on_mean_image(movie, events, max_events=50)

Requirements:
- Use matplotlib.
- Return figure/axis objects.
- Work headlessly.
- Do not force colors/styles.
- Add tests that functions return figures without crashing.
```

---

## 21. Claude Code Prompt P: Add First neuroFMx Experiment Configs

```text
Read:

    docs/neuros_astro_whitepaper.md
    docs/neuros_astro_implementation_plan.md

Create first neuroFMx experiment configs that compare neural-only vs neural+astro.

Inspect the existing training config system in packages/neuros-neurofm.

Create configs for:

1. neural_only_baseline.yaml
2. neural_behavior_baseline.yaml
3. neural_behavior_astro_events.yaml
4. neural_behavior_astro_graph.yaml

Each config should specify:
- modalities
- token paths
- temporal alignment window
- model backbone
- loss heads
- train/val split
- metrics
- output directory

Suggested prediction heads:
- future_neural_prediction
- behavior_state_prediction
- masked_modality_reconstruction
- contrastive_session_alignment

Add documentation:
- what each ablation tests
- expected interpretation
- how to run
- where outputs are saved

Do not assume real data exists. Use synthetic astro tokens and existing mock neural data if available.

Add tests or dry-run validation if the repo supports config validation.
```

---

## 22. Recommended First Commit Series

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
feat(neuros-astro): add NWB metadata scanner
feat(neuros-astro): add DANDI metadata scanner
feat(neuros-astro): add visualization tools
```

---

## 23. Initial Release Definition of Done

The first usable release is complete when all of the following commands work:

```bash
pip install -e packages/neuros-astro
neuros-astro --help
neuros-astro generate-synthetic --out-dir examples/data
neuros-astro detect-trace-events examples/data/synthetic_traces.npy --frame-rate 10 --session-id synthetic --out examples/data/events.parquet
neuros-astro build-network examples/data/events.parquet --frame-rate 10 --session-id synthetic --out examples/data/graphs.json
neuros-astro tokenize-events examples/data/events.parquet --frame-rate 10 --session-id synthetic --out examples/data/astro_tokens.npz
pytest packages/neuros-astro/tests
```

The package should produce:

- a dataset triage report
- event table
- graph summary
- token file
- neuroFMx manifest
- basic plots

---

## 24. Development Rule for Claude Code

When implementing this project, always:

1. read both docs first
2. implement one objective or milestone at a time
3. avoid giant refactors
4. keep tests passing
5. prefer small composable functions
6. keep `neuros-astro` standalone
7. add only thin adapters into `neuroFMx`
8. write helpful error messages for optional dependencies
9. avoid loading large NWB arrays unless explicitly requested
10. avoid overclaiming astrocyte identity from unlabeled data

This project should grow like a careful lab notebook, not a fireworks factory.
