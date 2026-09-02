"""Thin Algonauts-facing authority helpers.

This module deliberately depends only on generic neurOS research contracts. It does not
import competition code or measured brain data. Project-specific evaluators can map their
existing immutable manifests into these contracts without giving neurOS ownership of the
competition protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from .contracts import (
    DataAccess,
    DatasetAuthority,
    EvaluationAuthority,
    ExperimentPacket,
    Hypothesis,
    ResearchAgent,
)


@dataclass(frozen=True, slots=True)
class AlgonautsAuthoritySpec:
    """Serializable bridge from a project split/source authority into neurOS research."""

    dataset_id: str
    source_sha256: str
    source_revision: str
    split_sha256: str
    evaluator_id: str
    domains: tuple[str, ...] = (
        "validation",
        "id_test",
        "g2_ood",
        "geometry",
        "operational",
    )
    metrics: tuple[str, ...] = (
        "pearson",
        "rsa_spearman",
        "winner_frequency",
        "winner_entropy_stability",
        "runtime_seconds",
    )
    access: str = "authorized_restricted"

    def dataset_authority(self, *, metadata: Mapping[str, Any] | None = None) -> DatasetAuthority:
        return DatasetAuthority(
            dataset_id=self.dataset_id,
            source_fingerprint=self.source_sha256,
            access=cast(DataAccess, self.access),
            source_revision=self.source_revision,
            metadata={} if metadata is None else metadata,
        )

    def evaluation_authority(self, *, metadata: Mapping[str, Any] | None = None) -> EvaluationAuthority:
        return EvaluationAuthority(
            evaluator_id=self.evaluator_id,
            split_fingerprint=self.split_sha256,
            metric_names=self.metrics,
            evaluation_domains=self.domains,
            optimization_boundary="train_validation",
            forbidden_feedback=(
                "hidden_test_targets",
                "private_leaderboard",
                "g2_ood_for_model_selection",
                "g3_cross_game_for_model_selection",
                "g4_held_subject_for_model_selection",
            ),
            metadata={} if metadata is None else metadata,
        )

    def packet(
        self,
        *,
        experiment_id: str,
        agent: ResearchAgent,
        hypothesis: Hypothesis,
        code_revision: str,
        seeds: tuple[int, ...],
        representation_sha256: str | None = None,
        compute_budget: Mapping[str, Any] | None = None,
        dataset_metadata: Mapping[str, Any] | None = None,
        evaluation_metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentPacket:
        """Create a competition-facing packet with OOD truth excluded from selection."""

        return ExperimentPacket(
            experiment_id=experiment_id,
            dataset=self.dataset_authority(metadata=dataset_metadata),
            evaluation=self.evaluation_authority(metadata=evaluation_metadata),
            agent=agent,
            hypothesis=hypothesis,
            code_revision=code_revision,
            seeds=seeds,
            information_regimes=("external_pretrained", "train_only_inductive"),
            claim_ceiling="predictive_ood",
            representation_fingerprint=representation_sha256,
            compute_budget={} if compute_budget is None else compute_budget,
            metadata={
                "adapter": "neuros.research.algonauts.AlgonautsAuthoritySpec",
                "selection_rule": "train_validation_only",
            },
        )
