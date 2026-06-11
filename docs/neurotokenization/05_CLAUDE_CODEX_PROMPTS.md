# Claude and Codex Implementation Prompts

## How to Use These Prompts

Use these prompts one sprint at a time. Do not ask an agent to implement the entire system in one pass. Each sprint should end with tests, a short report, and a clean git diff.

Recommended flow:

1. Give Claude the architecture/sprint prompt.
2. Ask Claude to inspect the repo and propose file-level changes.
3. Give Codex the implementation prompt for one sprint.
4. Run tests locally.
5. Ask Claude to review the diff scientifically and architecturally.
6. Iterate.

## Global Instruction for All Agents

```text
You are working in the sidhulyalkar/neurOS-v1 repository.

The goal is to add a neurotokenization subsystem for scientifically evaluating spike-train tokenization approaches for neural foundation models.

Do not create a disconnected prototype if existing neurOS abstractions can be extended. First inspect the repository structure, existing CLI, model interfaces, processor interfaces, benchmark patterns, config style, and tests.

Prioritize correctness, modularity, tests, and reproducibility over cleverness. Every tokenizer must share a common interface and be benchmarkable through the same runner.

Do not implement huge models. Use small baselines first. The goal is to validate tokenization scientifically, not to train a massive foundation model immediately.

After each sprint, produce:
1. files changed
2. tests added
3. commands to run
4. known limitations
5. next recommended step
```

## Prompt 1: Repo Integration Notes

```text
Inspect the neurOS-v1 repository and create `docs/neurotokenization/REPO_INTEGRATION_NOTES.md`.

Use the existing README and repository structure to identify exactly where the neurotokenization subsystem should integrate.

Please answer:
- Where are drivers implemented?
- Where are processors or feature extractors implemented?
- Where are models implemented?
- Where are benchmarks implemented?
- How does the CLI register commands?
- What test framework is used?
- What config pattern is used?
- Where should `neuros/tokenization/` and `neuros/tokenization_bench/` be added?
- Which existing abstractions should be reused?
- Which new abstractions are necessary?

Do not implement algorithms in this step. Only produce the integration note and optionally create empty package folders if appropriate.
```

## Prompt 2: Schemas and Synthetic Generator

```text
Implement Sprint 1 from `docs/neurotokenization/03_IMPLEMENTATION_SPRINTS.md`.

Add common schemas and a deterministic synthetic spike generator.

Required files:
- `neuros/tokenization/schemas.py`
- `neuros/tokenization/base.py`
- `neuros/tokenization_bench/synthetic.py`
- `neuros/tokenization_bench/perturbations.py`
- tests under `tests/tokenization/`

The synthetic generator must create spike events with ground-truth motif labels for:
- Poisson background
- single-unit bursts
- pauses and rebounds
- synchrony packets
- leader-follower chains
- assembly activations
- task-state-dependent firing-rate changes
- movement-onset volleys

All generation must be deterministic under a random seed.

Add tests for:
- determinism
- empty sessions
- one-unit sessions
- multi-unit sessions
- ground-truth label alignment
- perturbation functions

Do not implement learned models yet.
```

## Prompt 3: Baseline Tokenizers

```text
Implement Sprint 2 from `docs/neurotokenization/03_IMPLEMENTATION_SPRINTS.md`.

Add:
- `EventSpikeTokenizer`
- `BinnedCountTokenizer`
- `RateSummaryTokenizer`
- `RandomControlTokenizer`

All must implement the common tokenizer interface from `neuros/tokenization/base.py`.

Required behavior:
- deterministic encoding
- clear tokenizer manifest
- stable token IDs
- decode support where feasible
- robust handling of empty input, unsorted input, simultaneous spikes, and long silence

Add tests for every tokenizer.

Also add a simple tokenizer registry if useful, but do not build the full benchmark harness yet.
```

## Prompt 4: ISI/MIDI Tokenizer

```text
Implement Sprint 3 from `docs/neurotokenization/03_IMPLEMENTATION_SPRINTS.md`.

Add `ISIMidiTokenizer` in `neuros/tokenization/isi.py`.

The tokenizer should make relative timing explicit using WAIT tokens and SPIKE tokens.

Requirements:
- sorted spike event input
- configurable logarithmic time bins in milliseconds
- global delta-time tokenization
- unit-local ISI side features
- recent population-rate side features
- online/streaming mode with bounded state
- deterministic output
- clear manifest

Tests must cover:
- simple known sequence
- burst sequence with short ISIs
- long silence
- simultaneous spikes
- unsorted input
- one-unit and multi-unit cases
- reproducibility under same input

Add a short docstring explaining why this tokenizer differs from absolute-time event tokenization.
```

## Prompt 5: Burst and Synchrony Tokenizers

```text
Implement Sprint 4 from `docs/neurotokenization/03_IMPLEMENTATION_SPRINTS.md`.

Add:
- `BurstTokenizer` in `neuros/tokenization/burst.py`
- `SynchronyPacketTokenizer` in `neuros/tokenization/synchrony.py`

BurstTokenizer requirements:
- per-unit burst detection using configurable ISI thresholds
- adaptive per-unit threshold option
- pause detection
- rebound detection
- human-readable token summaries

SynchronyPacketTokenizer requirements:
- small-window coactivation detection
- configurable window size and min active units
- optional region-aware summaries if metadata exists
- firing-rate matched null-control helper if feasible

Tests:
- recover synthetic bursts
- recover synthetic synchrony packets
- low false positives on Poisson-only synthetic data
- handle edge cases

Do not implement the VQ learned tokenizer in this sprint.
```

## Prompt 6: Benchmark Harness

```text
Implement Sprint 5 from `docs/neurotokenization/03_IMPLEMENTATION_SPRINTS.md`.

Build the unified benchmark runner for tokenizers.

Required functionality:
- tokenizer registry
- config-driven benchmark execution
- shared synthetic train/validation/test generation
- metrics for compression, motif recovery, token entropy, robustness, and runtime
- perturbation sweeps for spike jitter and unit dropout
- report generation to JSON, CSV, and Markdown

Required files:
- `neuros/tokenization/registry.py`
- `neuros/tokenization_bench/metrics.py`
- `neuros/tokenization_bench/train_eval.py`
- `neuros/tokenization_bench/reports.py`
- `scripts/neurotokenization/run_tokenizer_benchmark.py`
- `scripts/neurotokenization/compare_tokenizers.py`
- `configs/neurotokenization/synthetic_smoke.yaml`
- `configs/neurotokenization/tokenizer_grid.yaml`

Acceptance criteria:
- one command runs a synthetic smoke benchmark
- output includes `metrics.json`, `comparison_table.csv`, and `tokenizer_cards.md`
- tests pass
- command examples are documented
```

## Prompt 7: Tiny Foundation Model Comparison

```text
Implement Sprint 6 from `docs/neurotokenization/03_IMPLEMENTATION_SPRINTS.md`.

Add small sequence model baselines to test whether tokenization changes downstream learning.

Models:
- GRU baseline
- tiny Transformer baseline

Tasks:
- next-token prediction
- masked-token prediction
- optional next-window spike prediction where supported

Requirements:
- CPU smoke test
- GPU acceleration if torch.cuda is available
- deterministic seed handling
- training-time and memory reporting
- same model budget across tokenizers where possible

Do not scale model size yet. Keep defaults small enough for a consumer GPU.
```

## Prompt 8: Learned VQ Motif Tokenizer

```text
Implement Sprint 7 from `docs/neurotokenization/03_IMPLEMENTATION_SPRINTS.md`.

Add `VQMotifTokenizer`.

Pipeline:
spike raster window -> local encoder -> latent vector -> vector quantization -> motif token ID

Requirements:
- raster windowing utility
- small encoder/decoder
- VQ codebook
- reconstruction loss
- commitment loss
- codebook usage metrics
- token interpretability summaries
- benchmark integration

Acceptance criteria:
- codebook does not collapse on synthetic medium benchmark
- learned tokens align with some synthetic motifs above random
- token cards show average raster patches per learned token
- tests cover shape handling, deterministic inference, and codebook usage metrics
```

## Prompt 9: Assembly Tokenizer

```text
Implement Sprint 8 from `docs/neurotokenization/03_IMPLEMENTATION_SPRINTS.md`.

Add `AssemblyTokenizer` using a simple, interpretable method first.

Recommended first version:
- bin spikes into count matrix
- fit NMF or ICA/PCA-based components
- threshold component activations
- emit ASSEMBLY_ON, ASSEMBLY_PEAK, and ASSEMBLY_OFF tokens

Requirements:
- deterministic fit under seed
- top-unit summaries per assembly
- activation timeline summaries
- integration with benchmark runner

Tests:
- recover synthetic assembly activations under low noise
- handle one assembly and multiple assemblies
- no crash on low-activity sessions
```

## Prompt 10: First Real Dataset Adapter

```text
Implement Sprint 9 from `docs/neurotokenization/03_IMPLEMENTATION_SPRINTS.md`.

Add an adapter for at least one real spike dataset format already compatible with neurOS-v1.

Start with NWB if possible.

Requirements:
- load spike times into `SpikeEvent` schemas
- load unit metadata if available
- load trial/behavior labels if available
- allow duration and unit caps for local testing
- run all non-learned tokenizers on one real dataset smoke test
- produce tokenizer cards even when labels are missing

Do not require users to download a huge dataset for tests. Use mocks or a tiny fixture for CI.
```

## Prompt 11: Scientific Review Prompt

```text
Review the current neurotokenization implementation scientifically and architecturally.

Evaluate:
- Is the tokenizer comparison fair?
- Are the synthetic motifs realistic enough?
- Are the baselines strong enough?
- Are there leakage risks?
- Are metrics measuring real scientific utility or just reconstruction?
- Are robustness perturbations meaningful?
- Are token interpretability reports useful?
- Is the compute plan realistic for local GPU testing?
- What must be fixed before running cloud experiments?

Return:
1. blocking issues
2. important non-blocking issues
3. recommended next implementation sprint
4. suggested experiment config
5. risks to scientific validity
```

## Prompt 12: First Experiment Run

```text
Run the first tokenizer comparison experiment after Sprints 1-5 are implemented.

Use synthetic smoke config.

Compare:
- EventSpikeTokenizer
- BinnedCountTokenizer
- ISIMidiTokenizer
- BurstTokenizer
- SynchronyPacketTokenizer

Report:
- compression ratio
- motif recovery F1
- token entropy
- runtime
- robustness to 1, 5, 10 ms jitter
- robustness to 10%, 25%, 50% unit dropout
- tokenizer cards

Then write a short interpretation:
- Which tokenizer won which metric?
- Which results are expected?
- Which results are surprising?
- Which tokenizer should be improved next?
- Is this ready for VQ motif tokenization?
```
