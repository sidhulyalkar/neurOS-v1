# Implementation Sprints

## Sprint 0: Repo Orientation and Guardrails

### Goal

Map the current neurOS-v1 package layout and identify the correct locations for tokenizers, benchmarks, configs, scripts, and tests.

### Tasks

1. Inspect the repository tree.
2. Identify existing model interfaces.
3. Identify existing processor / feature extraction interfaces.
4. Identify existing benchmark CLI patterns.
5. Identify existing config conventions.
6. Identify test framework and CI structure.
7. Create a short implementation note before coding.

### Deliverables

- `docs/neurotokenization/REPO_INTEGRATION_NOTES.md`
- Initial empty package folders if they do not exist.
- No algorithm implementation yet.

### Acceptance Criteria

- The integration note names exact files/classes to extend.
- No duplicate framework is created if an existing neurOS abstraction already exists.

## Sprint 1: Shared Schemas and Synthetic Spike Generator

### Goal

Create the foundation needed to evaluate tokenizers without relying on external datasets.

### Tasks

1. Implement `SpikeEvent`, `Token`, `TokenizerManifest`, and `TokenizedSequence` schemas.
2. Implement `SpikeTokenizer` abstract base class.
3. Implement synthetic spike generator.
4. Generate sessions with ground-truth motif labels.
5. Add perturbation functions.
6. Add unit tests.

### Synthetic Generator Requirements

Generate:

- Poisson background
- single-unit bursts
- pauses and rebounds
- synchrony packets
- leader-follower chains
- assembly activations
- task-state-dependent firing-rate changes
- movement-onset volleys

Each generated session should include:

```text
spike_events
motif_labels
state_labels
unit_metadata
session_metadata
```

### Files

```text
neuros/tokenization/schemas.py
neuros/tokenization/base.py
neuros/tokenization_bench/synthetic.py
neuros/tokenization_bench/perturbations.py
tests/tokenization/test_schemas.py
tests/tokenization/test_synthetic_generator.py
```

### Acceptance Criteria

- Synthetic data generation is deterministic under a seed.
- Ground-truth labels are aligned to spike times.
- Unit tests pass for edge cases.
- A script can save synthetic data as `.npz` and `.jsonl`.

## Sprint 2: Baseline Tokenizers

### Goal

Implement the simplest comparison baselines.

### Tokenizers

1. `EventSpikeTokenizer`
2. `BinnedCountTokenizer`
3. `RateSummaryTokenizer`
4. `RandomControlTokenizer`

### Files

```text
neuros/tokenization/events.py
neuros/tokenization/binned.py
neuros/tokenization/controls.py
tests/tokenization/test_event_tokenizer.py
tests/tokenization/test_binned_tokenizer.py
tests/tokenization/test_control_tokenizers.py
```

### Acceptance Criteria

- Each tokenizer implements `fit`, `encode`, `decode` where possible, and `describe_token`.
- Each tokenizer emits a manifest.
- Event tokenizer preserves spike count after decode.
- Binned tokenizer reconstructs approximate count matrices.
- Control tokenizers are clearly marked as controls.

## Sprint 3: ISI/MIDI Tokenizer

### Goal

Make relative timing gaps into first-class tokens.

### Tasks

1. Implement logarithmic timing bins.
2. Implement global delta-time tokens.
3. Implement unit-local ISI side features.
4. Implement population-rate side features.
5. Add streaming encode mode.
6. Add tests for simultaneous spikes and long silence.

### Files

```text
neuros/tokenization/isi.py
tests/tokenization/test_isi_tokenizer.py
```

### Acceptance Criteria

- `WAIT_BIN_k` tokens appear before spike events when appropriate.
- Token sequence is deterministic after sorting.
- High-frequency bursts map to short-ISI bins.
- Long silences map to capped long-wait bins.
- The tokenizer can run online with bounded memory.

## Sprint 4: Burst and Synchrony Tokenizers

### Goal

Implement interpretable motif-level tokenizers.

### BurstTokenizer Tasks

1. Per-unit ISI threshold burst detection.
2. Adaptive per-unit thresholds.
3. Pause detection.
4. Rebound detection.
5. Token-level summaries.

### SynchronyPacketTokenizer Tasks

1. Small-window coactivation detection.
2. Size bins.
3. Optional region-aware summaries.
4. Null-control helper for firing-rate matched synchrony.

### Files

```text
neuros/tokenization/burst.py
neuros/tokenization/synchrony.py
tests/tokenization/test_burst_tokenizer.py
tests/tokenization/test_synchrony_tokenizer.py
```

### Acceptance Criteria

- Synthetic bursts are detected with high recall under low noise.
- Synthetic synchrony packets are detected with high recall under low noise.
- False positives are measured under Poisson-only controls.
- Token summaries are human-readable.

## Sprint 5: Benchmark Harness

### Goal

Evaluate all implemented tokenizers under a unified protocol.

### Tasks

1. Implement tokenizer registry.
2. Implement benchmark runner.
3. Implement shared train/validation/test splits.
4. Implement token-level metrics.
5. Implement synthetic motif recovery metrics.
6. Implement robustness perturbation sweeps.
7. Implement report writer.

### Files

```text
neuros/tokenization/registry.py
neuros/tokenization_bench/metrics.py
neuros/tokenization_bench/train_eval.py
neuros/tokenization_bench/reports.py
scripts/neurotokenization/run_tokenizer_benchmark.py
scripts/neurotokenization/compare_tokenizers.py
configs/neurotokenization/synthetic_smoke.yaml
configs/neurotokenization/tokenizer_grid.yaml
```

### Acceptance Criteria

- One command runs all tokenizers on synthetic smoke data.
- Outputs `metrics.json`, `comparison_table.csv`, and `tokenizer_cards.md`.
- Robustness curves are generated for jitter and unit dropout.
- Results are reproducible with a seed.

## Sprint 6: Tiny Foundation Model Baseline

### Goal

Test whether tokenization affects downstream sequence modeling.

### Tasks

1. Implement token dataset loader.
2. Implement small GRU baseline.
3. Implement tiny Transformer baseline.
4. Add masked-token objective.
5. Add next-token objective.
6. Add next-window spike prediction objective where applicable.
7. Compare model performance across tokenizers.

### Files

```text
neuros/tokenization_bench/models.py
scripts/neurotokenization/train_tiny_foundation_model.py
configs/neurotokenization/synthetic_medium.yaml
```

### Acceptance Criteria

- Training works on CPU for smoke tests.
- Training uses GPU if available.
- Each tokenizer can produce model-ready datasets.
- Report includes training time, GPU memory, and performance.

## Sprint 7: Learned VQ Motif Tokenizer

### Goal

Learn neural subword tokens from local raster windows.

### Tasks

1. Implement raster windowing.
2. Implement small encoder and decoder.
3. Implement VQ codebook.
4. Train with reconstruction and commitment losses.
5. Add codebook usage metrics.
6. Add token interpretability summaries.
7. Compare to handcrafted burst/synchrony tokens.

### Files

```text
neuros/tokenization/motif_vq.py
neuros/tokenization_bench/vq_train.py
tests/tokenization/test_vq_motif_tokenizer.py
```

### Acceptance Criteria

- Codebook does not collapse on synthetic medium benchmark.
- Learned tokens align with at least some synthetic motifs.
- Token cards show average raster patches per token.
- VQ tokenizer can be benchmarked through the same interface.

## Sprint 8: Assembly Tokenizer

### Goal

Represent spike data using population assembly activations.

### Tasks

1. Implement NMF-based assembly extraction.
2. Implement simple activation thresholding.
3. Emit assembly on/peak/off tokens.
4. Add assembly interpretability summaries.
5. Compare assembly tokens to synchrony and VQ motifs.

### Files

```text
neuros/tokenization/assembly.py
tests/tokenization/test_assembly_tokenizer.py
```

### Acceptance Criteria

- Synthetic assembly activations are recovered under low noise.
- Assembly summaries list top contributing units.
- Assembly tokens can be used in benchmark runner.

## Sprint 9: Real Dataset Adapter

### Goal

Run the benchmark on at least one public or local spike dataset.

### Candidate Sources

- NWB files already compatible with neurOS pipelines.
- DANDI datasets with spike times.
- IBL-style Neuropixels data if an adapter is available.

### Tasks

1. Implement `load_spike_events_from_nwb`.
2. Normalize unit metadata.
3. Load trial/behavior labels if present.
4. Run smoke benchmark.
5. Generate real-data tokenizer cards.

### Acceptance Criteria

- One real dataset can be tokenized by all non-learned tokenizers.
- At least one predictive or decoding task runs.
- Report clearly marks missing labels or metadata.

## Sprint 10: Scientific Report and Decision Gate

### Goal

Decide whether any tokenizer is promising enough for larger-scale cloud training.

### Required Report Sections

1. Dataset description.
2. Tokenizer descriptions.
3. Synthetic motif recovery.
4. Predictive modeling results.
5. Transfer results if available.
6. Robustness curves.
7. Interpretability analysis.
8. Compute cost.
9. Failure modes.
10. Recommendation for next scale-up.

### Decision Gate

Scale beyond local GPU only if at least one tokenizer shows:

- meaningful improvement over event/binned baseline; or
- equal performance with much better compression; or
- substantially better robustness; or
- better interpretability with acceptable performance tradeoff.

## Suggested Development Order for Claude/Codex

1. Sprint 0 repo map.
2. Sprint 1 schemas and synthetic generator.
3. Sprint 2 event and binned baselines.
4. Sprint 3 ISI tokenizer.
5. Sprint 4 burst/synchrony tokenizers.
6. Sprint 5 benchmark harness.
7. Stop and run first comparison.
8. Only then implement VQ and assembly.

Do not implement the learned tokenizer before the benchmark harness exists. That path summons notebook fog.
