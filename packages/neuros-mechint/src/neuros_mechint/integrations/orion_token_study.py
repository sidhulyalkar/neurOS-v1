"""End-to-end causal comparison of ORION neural tokenization schemes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from orion.contracts import NeuroTokenBatch

from neuros_mechint.benchmarks import (
    TokenizerComparisonReport,
    TokenizerEffectRecord,
    TokenizerMechanismContext,
    compare_tokenizer_mechanisms,
)
from neuros_mechint.core import EvidenceTier, stable_hash
from neuros_mechint.core.results import InputExperimentResult

from .orion import temporal_window_audit


@dataclass(frozen=True, slots=True)
class OrionTokenizerStudyContext:
    """One tokenizer output and downstream scorer in a matched study."""

    context: TokenizerMechanismContext
    token_batch: NeuroTokenBatch
    scorer: Callable[[NeuroTokenBatch], float]
    alignment_origin_ns: int | None = None
    alignment_label: str | None = None
    evidence_tier: EvidenceTier = EvidenceTier.UNIT

    def resolved_alignment_origin_ns(self) -> int:
        if self.alignment_origin_ns is not None:
            return int(self.alignment_origin_ns)
        timestamps = np.asarray(self.token_batch.timestamps_ns, dtype=np.int64)
        if len(timestamps) == 0:
            raise ValueError("cannot infer alignment origin from an empty token batch")
        return int(timestamps.min())


@dataclass(frozen=True, slots=True)
class OrionTokenizerAudit:
    context: TokenizerMechanismContext
    alignment_origin_ns: int
    alignment_label: str | None
    result: InputExperimentResult
    record: TokenizerEffectRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "alignment_origin_ns": self.alignment_origin_ns,
            "alignment_label": self.alignment_label,
            "audit": self.result.to_dict(),
            "record": self.record.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OrionTokenizerStudyResult:
    audits: tuple[OrionTokenizerAudit, ...]
    comparison: TokenizerComparisonReport
    window_ns: int
    stride_ns: int | None
    replacement_token_id: int
    top_k: int

    @property
    def context_manifest_hashes(self) -> dict[str, str]:
        return {
            audit.context.context_id: audit.result.manifest.content_hash
            for audit in self.audits
        }

    @property
    def study_fingerprint(self) -> str:
        payload = {
            "parameters": {
                "window_ns": self.window_ns,
                "stride_ns": self.stride_ns,
                "replacement_token_id": self.replacement_token_id,
                "top_k": self.top_k,
            },
            "records": [audit.record.to_dict() for audit in self.audits],
            "comparison": self.comparison.to_dict(),
        }
        return stable_hash(payload)

    @property
    def study_hash(self) -> str:
        """Compatibility alias for the deterministic scientific fingerprint."""

        return self.study_fingerprint

    @property
    def run_hash(self) -> str:
        return stable_hash(
            {
                "study_fingerprint": self.study_fingerprint,
                "context_manifest_hashes": self.context_manifest_hashes,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": {
                "window_ns": self.window_ns,
                "stride_ns": self.stride_ns,
                "replacement_token_id": self.replacement_token_id,
                "top_k": self.top_k,
            },
            "audits": [item.to_dict() for item in self.audits],
            "comparison": self.comparison.to_dict(),
            "study_fingerprint": self.study_fingerprint,
            "study_hash": self.study_hash,
            "run_hash": self.run_hash,
            "context_manifest_hashes": self.context_manifest_hashes,
        }


def _canonical_target(effect: Any, *, origin_ns: int) -> str:
    metadata = dict(getattr(effect, "metadata", {}))
    start_ns = metadata.get("start_ns")
    end_ns = metadata.get("end_ns")
    if start_ns is None or end_ns is None:
        return str(effect.target)
    return f"tokens_relative[{int(start_ns) - origin_ns}:{int(end_ns) - origin_ns}]"


def _canonical_map(items: Sequence[Any], *, origin_ns: int) -> dict[str, float]:
    effect_map = {}
    for item in items:
        target = _canonical_target(item, origin_ns=origin_ns)
        if target in effect_map:
            raise ValueError(f"duplicate canonical token target: {target}")
        effect_map[target] = float(item.effect)
    return effect_map


def _validate_alignment_labels(contexts: Sequence[OrionTokenizerStudyContext]) -> None:
    labels = {item.alignment_label for item in contexts if item.alignment_label is not None}
    if len(labels) > 1:
        raise ValueError(
            "tokenizer study contexts use incompatible alignment_label values: "
            f"{sorted(labels)}"
        )
    if labels and any(item.alignment_label is None for item in contexts):
        raise ValueError(
            "all tokenizer study contexts must specify alignment_label when any context does"
        )


def audit_orion_tokenizer_context(
    study_context: OrionTokenizerStudyContext,
    *,
    window_ns: int,
    stride_ns: int | None = None,
    replacement_token_id: int = 0,
    seed: int = 0,
    include_shuffle_controls: bool = True,
) -> OrionTokenizerAudit:
    context = study_context.context
    origin_ns = study_context.resolved_alignment_origin_ns()
    result = temporal_window_audit(
        study_context.token_batch,
        study_context.scorer,
        window_ns=window_ns,
        stride_ns=stride_ns,
        replacement_token_id=replacement_token_id,
        model_id=context.downstream_model_id,
        dataset_id=context.dataset_id,
        experiment_name=f"tokenizer-mechanism:{context.context_id}",
        seed=seed,
        evidence_tier=study_context.evidence_tier,
        include_shuffle_controls=include_shuffle_controls,
    )
    record = TokenizerEffectRecord(
        context=context,
        baseline_metric=result.baseline_metric,
        effect_map=_canonical_map(result.effects, origin_ns=origin_ns),
        control_map=_canonical_map(result.controls, origin_ns=origin_ns),
        metric_name=result.metric_name,
    )
    return OrionTokenizerAudit(
        context=context,
        alignment_origin_ns=origin_ns,
        alignment_label=study_context.alignment_label,
        result=result,
        record=record,
    )


def run_orion_tokenizer_study(
    contexts: Sequence[OrionTokenizerStudyContext],
    *,
    window_ns: int,
    stride_ns: int | None = None,
    replacement_token_id: int = 0,
    top_k: int = 5,
    seed: int = 0,
    include_shuffle_controls: bool = True,
    stable_spearman: float = 0.7,
    divergent_spearman: float = 0.3,
    min_shared_target_fraction: float = 0.75,
) -> OrionTokenizerStudyResult:
    """Compare tokenizers under aligned, matched temporal causal interventions."""

    contexts = tuple(contexts)
    if len(contexts) < 2:
        raise ValueError("at least two tokenizer study contexts are required")
    if window_ns <= 0:
        raise ValueError("window_ns must be positive")
    if stride_ns is not None and stride_ns <= 0:
        raise ValueError("stride_ns must be positive")

    _validate_alignment_labels(contexts)
    audits = tuple(
        audit_orion_tokenizer_context(
            context,
            window_ns=window_ns,
            stride_ns=stride_ns,
            replacement_token_id=replacement_token_id,
            seed=seed + index,
            include_shuffle_controls=include_shuffle_controls,
        )
        for index, context in enumerate(contexts)
    )
    comparison = compare_tokenizer_mechanisms(
        [audit.record for audit in audits],
        top_k=top_k,
        stable_spearman=stable_spearman,
        divergent_spearman=divergent_spearman,
        min_shared_target_fraction=min_shared_target_fraction,
    )
    return OrionTokenizerStudyResult(
        audits=audits,
        comparison=comparison,
        window_ns=window_ns,
        stride_ns=stride_ns,
        replacement_token_id=replacement_token_id,
        top_k=top_k,
    )
