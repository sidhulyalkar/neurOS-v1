# Evaluation Protocol for Spike Tokenizers

## Principle

A tokenizer should be judged by scientific and modeling utility, not reconstruction alone. A perfect compressor of spike noise is not a good neural alphabet. The evaluation must measure whether the tokens preserve structure useful for prediction, transfer, interpretability, robustness, and downstream neuroscience.

## Fair Comparison Rule

Every tokenizer must be evaluated under matched conditions:

1. Same train/validation/test splits.
2. Same downstream model family where possible.
3. Same parameter budget where possible.
4. Same context length or equivalent temporal coverage.
5. Same compute budget or reported compute-normalized scores.
6. Same perturbation suite.
7. Same random seeds.
8. Same reporting format.

## Evaluation Stages

### Stage 0: Unit Tests

Goal: verify that each tokenizer is deterministic and stable.

Required tests:

- empty spike list
- one spike
- one unit with burst
- multiple units with simultaneous spikes
- long silence
- unsorted input
- repeated timestamps
- high firing-rate edge case
- decode smoke test if supported

### Stage 1: Synthetic Ground Truth

Goal: validate that tokenizers recover known motifs.

Synthetic motifs:

- Poisson background
- single-unit burst
- pause and rebound
- synchrony packet
- leader-follower chain
- assembly activation
- state-dependent firing-rate shift
- movement-onset volley
- noisy sorting contamination
- missing unit dropout

Metrics:

- motif classification accuracy
- motif F1 score
- event boundary error
- token purity by motif label
- normalized mutual information between token IDs and hidden motif IDs
- adjusted Rand index for learned motif clustering
- robustness under jitter and dropout

### Stage 2: Predictive Modeling

Goal: determine whether token streams improve modeling of neural dynamics.

Tasks:

1. Next-token prediction.
2. Next-window spike prediction.
3. Held-out unit prediction.
4. Masked token reconstruction.
5. Future population-rate prediction.

Metrics:

- cross entropy for discrete tokens
- bits per spike where applicable
- Poisson negative log-likelihood for spike counts
- MSE / correlation for continuous population rates
- AUROC / AUPRC for held-out spike events
- calibration error for event probabilities

### Stage 3: Behavior and Task Decoding

Goal: test whether tokens preserve meaningful neural-behavior structure.

Tasks depend on dataset labels:

- trial phase classification
- stimulus identity classification
- movement onset prediction
- reach direction / kinematic decoding
- choice decoding
- reward outcome prediction
- error / reversal state detection

Metrics:

- accuracy / balanced accuracy
- AUROC / AUPRC
- R2 for continuous behavior
- temporal error for event onset prediction
- sample efficiency with 1%, 5%, 10%, 50%, 100% labels

### Stage 4: Cross-Session and Cross-Animal Transfer

Goal: test whether tokens generalize beyond one recording.

Splits:

- train session A, test session B
- train animal A, test animal B
- train task A, fine-tune task B if available
- train with subset of units, test with dropped or replaced units

Metrics:

- zero-shot performance
- few-shot fine-tuning performance
- relative transfer gap
- alignment score between token distributions
- robustness to neuron identity remapping

### Stage 5: Robustness and Nuisance Controls

Perturbations:

1. Spike time jitter: 1, 2, 5, 10, 20 ms.
2. Random unit dropout: 5%, 10%, 25%, 50%.
3. Spike deletion: 5%, 10%, 25%.
4. False positive spike insertion: 5%, 10%, 25%.
5. Unit split/merge simulation.
6. Firing-rate scaling.
7. Session metadata removal.
8. Behavior-label shuffling.
9. Time reversal control.
10. Trial-label permutation.

Report each tokenizer's degradation curve.

## Tokenizer Metrics

### Compression

```text
compression_ratio = raw_event_count / token_count
bits_per_second = total_token_bits / session_duration
```

Also report:

- mean tokens per second
- median tokens per trial
- max token burst rate
- vocabulary size
- active vocabulary size

### Codebook Usage

For learned tokenizers:

- token entropy
- perplexity
- dead code percentage
- top-1 token frequency
- Gini coefficient of token usage
- codebook stability across seeds

### Interpretability

For each token type or learned token ID, report:

- most associated motif labels on synthetic data
- most associated behavior labels on real data
- average firing-rate pattern
- average raster patch
- active units / regions
- timing relative to trial events
- nearest tokens by embedding similarity
- cross-session recurrence

### Scientific Utility Score

Compute an aggregate score but keep component metrics visible.

Suggested first version:

```text
SUS = 0.20 * predictive_score
    + 0.20 * behavior_score
    + 0.15 * transfer_score
    + 0.15 * robustness_score
    + 0.10 * compression_score
    + 0.10 * interpretability_score
    + 0.10 * codebook_health_score
```

This score is only for navigation. Papers and reports should emphasize component-level results.

## Statistical Validation

Use:

- at least 5 random seeds for synthetic experiments
- bootstrap confidence intervals for real datasets
- paired comparisons across matched splits
- effect sizes, not only p-values
- multiple-comparison correction when many tokenizers are compared

Minimum reporting:

```text
mean ± 95% bootstrap confidence interval
paired delta vs baseline
relative improvement percentage
compute cost
memory cost
```

## Baselines

Required baselines:

1. Raw event tokenizer.
2. Fixed-bin count tokenizer.
3. Rate-only summary baseline.
4. Random token control.
5. Time-shuffled control.
6. Behavior-label-only upper leakage check where labels exist.

## Failure Modes to Detect

A tokenizer should be flagged if:

- it wins only by encoding session ID or trial timing
- learned codebook collapses to a few tokens
- performance vanishes when labels are shuffled
- it is too sensitive to small spike jitter
- token count explodes on high firing-rate data
- real-time latency is too high for streaming use
- interpretability reports show tokens correspond mostly to artifacts

## Report Artifact Format

Each benchmark run should produce:

```text
reports/neurotokenization/{run_id}/
  config.yaml
  metrics.json
  tokenizer_cards.md
  comparison_table.csv
  plots/
    compression_vs_accuracy.png
    robustness_curves.png
    codebook_usage.png
    token_timeline_examples.png
    motif_confusion_matrix.png
  model_checkpoints/
  token_samples.jsonl
```

## Minimum Publishable Result

A credible first result would show:

1. Synthetic motif recovery validates the benchmark.
2. ISI or burst/motif tokenizers outperform event/binned baselines on at least one timing-structure task.
3. Event or binned baselines still win on at least one task, showing the benchmark is not biased.
4. Learned motif tokens produce meaningful clusters and non-collapsed codebooks.
5. Robustness curves reveal when each tokenizer fails.

The paper/tool should argue for a science of tokenization, not a single magic tokenizer.
