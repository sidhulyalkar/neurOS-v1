# neuros-research

`neuros-research` is the provider-agnostic research authority layer for neurOS.

It is designed for autonomous or human-guided scientific search where many agents,
models, or collaborators may propose experiments, but none of them are allowed to
change the evaluation authority that judges their own work.

The package intentionally contains no LLM SDK and no neuroscience model dependency.
Its job is to make experiment identity, legal information access, evidence, lineage,
promotion decisions, and experiment history explicit and machine-verifiable.

## Core objects

- `DatasetAuthority`: binds a dataset/source fingerprint and access class.
- `EvaluationAuthority`: binds the split, metric set, optimization boundary, and
  forbidden feedback channels.
- `ResearchAgent`: identifies a human or machine proposer without granting authority.
- `Hypothesis`: records the falsifiable claim and deliberately changed variables.
- `ExperimentPacket`: immutable execution contract binding code, data, evaluation,
  hypothesis, seeds, compute budget, information regimes, and claim ceiling.
- `ExperimentEvidence`: failure-preserving result with metrics and adversarial checks.
- `PromotionPolicy`: vector-valued gates. There is deliberately no universal winner
  scalar.
- `EvidenceArbiter`: deterministic evaluator that can promote only evidence matching
  the exact packet/evaluation authority.
- `InsightCard`: compact, lineage-bound cross-pollination artifact that can be shared
  between otherwise independent research agents.
- `EvidenceLedger`: append-only SHA-256 hash chain for packet/evidence/decision history.

## Scientific boundary

This package does not determine whether an experiment is scientifically important.
It ensures that the evidence used to make that decision has a stable identity and
that autonomous search cannot silently rewrite its own referee.

For Algonauts-style work, competition leaderboards should be represented as external
evaluation channels, not as iterative optimization labels. Train/validation/OOD
authority belongs in the packet before a candidate is executed.
