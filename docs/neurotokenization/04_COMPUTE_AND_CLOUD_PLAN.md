# Compute and Cloud Plan

## Recommendation

Start locally. Move to cheap cloud only after the synthetic benchmark and first real-data smoke test work end-to-end.

The early work is mostly algorithmic and benchmarking plumbing. It does not require large cloud GPUs. A local NVIDIA GPU is enough for:

- synthetic spike generation
- event, binned, ISI, burst, and synchrony tokenizers
- small GRU / tiny Transformer baselines
- VQ motif tokenizer smoke tests
- report generation

Cloud becomes useful when:

- training VQ motif tokenizers on many sessions
- running many hyperparameter sweeps
- training larger sequence models
- comparing multi-dataset transfer
- storing and streaming larger NWB/WebDataset shards

## Local Development Strategy

### Phase 1: CPU-Compatible Smoke Tests

Every tokenizer should run on CPU for small synthetic data.

Target workload:

```text
sessions: 4
units/session: 32
duration/session: 60 seconds
firing rates: 1-20 Hz
motifs: burst, synchrony, leader-follower, assembly
```

Expected output:

- `metrics.json`
- `comparison_table.csv`
- `tokenizer_cards.md`

### Phase 2: Local GPU Synthetic Medium

Target workload:

```text
sessions: 16-64
units/session: 64-256
duration/session: 5-30 minutes
models: GRU, tiny Transformer, small VQ motif tokenizer
```

This should fit on a consumer GPU if the model and batch sizes are modest.

Recommended local settings:

```yaml
precision: fp16_or_bf16_if_available
batch_size: auto_scale
num_workers: 2-8
context_length: 512-2048 tokens
model_dim: 128-512
layers: 2-6
codebook_size: 64-512
```

### Phase 3: First Real Dataset Local Run

Target one small NWB-style dataset or a trimmed subset.

Rules:

- start with one session
- cap duration to 10-30 minutes
- cap units to 128-256 if needed
- save tokenized output to disk
- do not train a large model yet

## When to Use Cloud

Use cloud only after these are true:

1. The benchmark runner is stable.
2. At least four tokenizers run end-to-end.
3. Reports are generated automatically.
4. Local synthetic results show at least one non-baseline tokenizer is promising.
5. The experiment config is reproducible.

## Cloud Instance Types

### Cheap Single-GPU Testing

Good for:

- tiny Transformer sweeps
- VQ tokenizer sweeps
- real-data smoke tests

Look for:

- RTX 3090
- RTX 4090
- L4
- A10
- A40

### Medium Experiments

Good for:

- larger VQ codebooks
- multi-session experiments
- medium sequence models

Look for:

- A100 40GB
- A100 80GB
- L40S

### Avoid at First

Avoid H100/H200 until the benchmark has produced clear evidence that scaling is worthwhile. They are excellent but unnecessary for proving the tokenization science.

## Providers to Consider

Check live pricing before renting. GPU prices change frequently.

Practical options:

- RunPod
- Lambda Cloud
- Vast.ai
- Paperspace / DigitalOcean Gradient
- AWS EC2 spot instances
- Google Cloud spot GPUs
- Azure spot GPUs

For this project, the most cost-effective path is usually:

1. Local GPU for development.
2. RunPod or Vast.ai for cheap single-GPU sweeps.
3. Lambda or cloud spot A100 for cleaner medium experiments.
4. Only use major cloud if storage, networking, or reproducibility matters more than price.

## Storage Plan

Tokenized data should be saved separately from raw data.

Suggested layout:

```text
artifacts/neurotokenization/
  raw_synthetic/
  tokenized/
    event/
    binned/
    isi_midi/
    burst/
    synchrony/
    vq_motif/
  reports/
  checkpoints/
```

For cloud:

- use object storage for datasets and tokenized artifacts
- keep checkpoints small
- save configs and metrics aggressively
- avoid repeatedly downloading large NWB files

## Cost Control Rules

1. Always run a 2-minute smoke test before a full run.
2. Always log estimated token count before training.
3. Save tokenized outputs so tokenization is not repeated.
4. Use mixed precision where possible.
5. Use early stopping for tokenizer/model sweeps.
6. Kill runs that show codebook collapse or invalid metrics early.
7. Start with fixed seeds and small grids.
8. Do not sweep model size and tokenizer design simultaneously.

## Suggested Experiment Ladder

### Ladder 1: Local Smoke

```text
CPU or local GPU
4 synthetic sessions
4 tokenizers
no learned VQ
no real data
```

### Ladder 2: Local Medium

```text
local GPU
16-64 synthetic sessions
event vs binned vs ISI vs burst vs synchrony
GRU and tiny Transformer
robustness sweeps
```

### Ladder 3: Local VQ

```text
local GPU
VQ motif tokenizer
small codebooks: 64, 128, 256
compare against handcrafted motifs
```

### Ladder 4: Cheap Cloud Sweep

```text
RTX 4090 / A10 / L4
multi-seed synthetic benchmark
hyperparameter sweep for ISI bins, burst thresholds, VQ codebook size
```

### Ladder 5: Real Dataset Pilot

```text
A100/L40S optional
one real dataset
all tokenizers
small sequence models
interpretability report
```

### Ladder 6: Serious Foundation Model Test

```text
A100 40GB/80GB or larger
multi-session/multi-dataset
larger transformer or state-space backbone
cross-session and few-shot transfer
```

## Local GPU Feasibility

You can absolutely do the first scientific validation locally if you keep the scope disciplined.

Local GPU is enough for:

- tokenizer implementation
- synthetic validation
- small real-data smoke tests
- tiny foundation model comparison
- VQ tokenizer prototypes

Cloud is only needed for scale and sweeps.

The first question should not be: can we train a giant model?

The first question should be: can we show that the tokenizer changes what structure the model can learn under controlled conditions?

## Minimum Hardware Guidance

### If local GPU has 8-12GB VRAM

Use:

- tiny models
- context length <= 512-1024
- small batch sizes
- 32-128 units
- short windows
- gradient accumulation

### If local GPU has 16-24GB VRAM

Use:

- context length 1024-4096 depending on token count
- model dim 256-512
- VQ codebook up to 512
- 64-256 units
- medium synthetic runs

### If local GPU has 32GB+ VRAM

You can run most pilot experiments locally.

## Reproducibility Requirements

Every run must log:

- git commit SHA
- config file
- tokenizer manifest
- random seed
- hardware type
- runtime
- GPU memory
- dataset summary
- token counts
- metrics

## Decision Rule for Scaling

Move to cloud only if one of these is true:

1. A tokenizer beats baseline by at least 5-10% on a meaningful metric across seeds.
2. A tokenizer matches performance with much better compression.
3. A tokenizer shows much better robustness to jitter/dropout.
4. Learned motifs align with known synthetic motifs and show promising real-data structure.
5. Local GPU memory prevents the next required experiment.

Otherwise, improve the tokenizer or benchmark locally first.
