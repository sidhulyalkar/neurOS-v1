# Neurotokenization Master Plan for neurOS-v1

## Purpose

This plan turns neurOS-v1 into a scientific testbed for evaluating spike tokenization strategies for neural foundation models. The goal is not to claim that any single tokenizer is universally best. The goal is to build a rigorous framework that can compare tokenization approaches across reconstruction, prediction, behavior decoding, cross-session transfer, robustness, compression, interpretability, and scientific usefulness.

neurOS-v1 already has several relevant extension points:

- modular drivers for real or simulated BCI hardware
- pluggable processing modules
- model training and inference interfaces
- benchmarking scripts
- CLI and dashboard surfaces
- Constellation-style multimodal ingestion and export to NWB/Zarr/WebDataset

The neurotokenization work should be implemented as a first-class research subsystem rather than a notebook-only prototype.

## Core Research Question

Can relative-time, burst, synchrony, assembly, and learned motif tokenizers produce better representations for spike-train foundation models than raw spike-event or fixed-bin count representations?

Subquestions:

1. Do relative-time / ISI tokens improve temporal motif discovery compared with absolute timestamp tokens?
2. Do burst and motif tokens improve compression and interpretability without harming downstream decoding?
3. Do learned vector-quantized motif tokens discover reusable neural subwords across sessions?
4. Do assembly tokens improve cross-animal or cross-session transfer?
5. Which tokenization approach is best under realistic perturbations: spike jitter, dropped units, sorting noise, missing metadata, and domain shift?
6. Can token quality be predicted from dataset structure, modality, firing statistics, task design, or metadata quality?

## Non-goals

This sprint should not attempt to train a full-scale universal neural foundation model. It should build the tokenizer evaluation engine and prove that the framework can produce scientifically meaningful comparisons on synthetic and small real datasets.

This sprint should not rely on expensive H100-scale training. The first version should run locally on a consumer NVIDIA GPU and optionally scale to cloud GPUs.

This sprint should not overfit to a single dataset. The design should support synthetic data first, then at least one real NWB-style spike dataset, then multiple datasets.

## Proposed Package Layout

Add the following package structure:

```text
neuros/
  tokenization/
    __init__.py
    base.py
    events.py
    isi.py
    burst.py
    synchrony.py
    motif_vq.py
    assembly.py
    schemas.py
    utils.py

  tokenization_bench/
    __init__.py
    synthetic.py
    datasets.py
    metrics.py
    perturbations.py
    train_eval.py
    reports.py

scripts/
  neurotokenization/
    generate_synthetic_spikes.py
    run_tokenizer_benchmark.py
    train_tiny_foundation_model.py
    compare_tokenizers.py
    make_tokenizer_report.py

configs/
  neurotokenization/
    synthetic_smoke.yaml
    synthetic_medium.yaml
    real_nwb_smoke.yaml
    tokenizer_grid.yaml
    compute_local.yaml
    compute_cloud.yaml

docs/
  neurotokenization/
    00_MASTER_PLAN.md
    01_TOKENIZER_SPECS.md
    02_EVALUATION_PROTOCOL.md
    03_IMPLEMENTATION_SPRINTS.md
    04_COMPUTE_AND_CLOUD_PLAN.md
    05_CLAUDE_CODEX_PROMPTS.md
```

## Tokenizers to Implement

Minimum viable set:

1. `EventSpikeTokenizer`
   - POYO-inspired baseline.
   - Emits individual spike event tokens with unit identity and time encoding.

2. `BinnedCountTokenizer`
   - Classical baseline.
   - Converts spike trains into neuron x time count matrices.

3. `ISIMidiTokenizer`
   - Relative-time tokenizer.
   - Emits wait/time-gap tokens plus spike tokens.
   - Supports global ISI, unit-local ISI, and population-rate context.

4. `BurstTokenizer`
   - Converts spike clusters into burst, single-spike, pause, and rebound events.
   - Starts with simple threshold rules, then supports statistical methods.

5. `SynchronyPacketTokenizer`
   - Emits population coactivation packets from small temporal windows.
   - Captures synchronous events and possible cascade structure.

6. `VQMotifTokenizer`
   - Learned neural subword tokenizer.
   - Encodes local raster windows into latent vectors and quantizes them into motif IDs.

7. `AssemblyTokenizer`
   - Detects recurring coactivation patterns or latent assemblies.
   - Emits assembly activation tokens.

## Model Baselines

Use small models first:

1. Markov / n-gram token model for sanity checks.
2. GRU or small TCN for fast sequence modeling.
3. Tiny Transformer with causal and masked objectives.
4. Optional state-space model backbone once tokenizers are stable.

Do not begin with a huge Transformer. The first goal is fair comparison, not leaderboard smoke.

## Dataset Stages

### Stage 0: Synthetic Unit Tests

Generate spike trains with known ground-truth structure:

- Poisson background
- single-unit bursts
- synchronous packets
- leader-follower chains
- assembly activations
- task-state-dependent firing rates
- movement-onset volleys
- noise, jitter, unit dropout, and sorting-like contamination

### Stage 1: Synthetic Benchmark

Train models on synthetic data and evaluate whether tokenizers recover known motifs and hidden states.

### Stage 2: Single Real Dataset Smoke Test

Load one small NWB or public spike dataset. Run tokenization and basic next-window prediction / reconstruction / held-out unit prediction.

### Stage 3: Multi-session Real Dataset

Compare transfer from session A to session B, then animal A to animal B if metadata permits.

### Stage 4: Multi-dataset Evaluation

Only after Stages 0-3 pass should this become a serious cross-dataset foundation-model benchmark.

## Core Metrics

Every tokenizer should be evaluated with:

- compression ratio
- reconstruction fidelity
- next-window spike prediction
- held-out unit prediction
- hidden motif recovery on synthetic data
- behavior / trial-state decoding when labels exist
- cross-session transfer
- few-shot adaptation
- robustness to jitter
- robustness to dropped units
- robustness to spike-sorting contamination
- token entropy and codebook usage
- interpretability reports
- wall-clock speed and GPU memory usage

## Deliverables

1. Tokenizer interface and schema.
2. Synthetic spike generator with ground-truth motifs.
3. Initial tokenizer implementations.
4. Shared benchmark harness.
5. Smoke tests and CI tests.
6. Tiny model training script.
7. Evaluation report generator.
8. Example results in Markdown and JSON.
9. Dashboard-compatible report artifacts.
10. Clear prompts for Claude and Codex implementation.

## Scientific Validation Standard

A tokenizer should only be considered promising if it satisfies at least one of the following under fair compute and model conditions:

1. Better downstream decoding at equal or lower token budget.
2. Better transfer across sessions or animals.
3. Better robustness to spike jitter or dropped units.
4. Better motif recovery on synthetic data with known ground truth.
5. Better interpretability without major performance loss.
6. Better sample efficiency in few-shot fine-tuning.

A tokenizer is not superior merely because it reconstructs spikes well.

## Recommended First Milestone

Build `EventSpikeTokenizer`, `BinnedCountTokenizer`, `ISIMidiTokenizer`, and `BurstTokenizer`, then compare them on synthetic data with known bursts and leader-follower motifs. This milestone should run on CPU or a single local GPU in under one hour.
