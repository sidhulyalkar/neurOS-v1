from __future__ import annotations

from dataclasses import replace

import pytest
from neuros.research import (
    DatasetAuthority,
    EvaluationAuthority,
    ExperimentPacket,
    Hypothesis,
    ResearchAgent,
    bind_runtime_alignment,
)


def digest(char: str) -> str:
    return char * 64


def runtime_metadata() -> dict[str, object]:
    return {
        "schema": "neuros.runtime_dataset_binding.v1",
        "dataset_id": "fixture",
        "manifest_sha256": digest("b"),
        "declared_dataset_content_sha256": digest("a"),
        "verified_dataset_content_sha256": digest("a"),
        "dataset_verification": "verified_whole_dataset",
        "source_verification_semantics": "verified_at_bridge",
        "lineage_completeness": "unknown",
        "claim_boundary": "local bytes are not complete scientific lineage",
    }


def packet(*, dataset: DatasetAuthority | None = None) -> ExperimentPacket:
    authority = dataset or DatasetAuthority(
        dataset_id="fixture",
        source_fingerprint=digest("a"),
        access="authorized_restricted",
        source_revision="release-v1",
        metadata={"neuros_runtime": runtime_metadata()},
    )
    return ExperimentPacket(
        experiment_id="runtime-aligned-fixture",
        dataset=authority,
        evaluation=EvaluationAuthority(
            evaluator_id="fixture-evaluator",
            split_fingerprint=digest("c"),
            metric_names=("balanced_accuracy",),
            evaluation_domains=("validation",),
        ),
        agent=ResearchAgent(
            agent_id="fixture-agent",
            kind="deterministic_program",
            provider="neuros",
            model="fixture",
        ),
        hypothesis=Hypothesis(
            hypothesis_id="fixture-hypothesis",
            statement="Exact temporal execution remains bound to the prospective packet.",
            changed_variables=("runtime.alignment_plan",),
        ),
        code_revision="30261595bba46baaf4134957c8e32e647651e9c7",
        seeds=(7,),
        information_regimes=("train_only_inductive",),
        claim_ceiling="predictive_id",
    )


def plan(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "plan_sha256": digest("e"),
        "dataset_id": "fixture",
        "dataset_content_sha256": digest("a"),
        "manifest_sha256": digest("b"),
        "sync_group": "sub-01/run-01",
        "policy": "exact",
        "modalities": ["behavior", "fmri"],
        "start_ns": 0,
        "overlap_end_ns": 20_000_000_000,
        "duration_ns": 4_000_000_000,
        "stride_ns": 2_000_000_000,
        "window_count": 9,
    }
    payload.update(updates)
    return payload


def dataset_with_runtime_metadata(**updates: object) -> DatasetAuthority:
    metadata = runtime_metadata()
    metadata.update(updates)
    return DatasetAuthority(
        dataset_id="fixture",
        source_fingerprint=digest("a"),
        access="authorized_restricted",
        source_revision="release-v1",
        metadata={"neuros_runtime": metadata},
    )


def test_exact_plan_binding_changes_packet_identity_not_dataset_identity() -> None:
    original = packet()
    bound = bind_runtime_alignment(original, plan())

    assert original.dataset.source_fingerprint == digest("a")
    assert bound.dataset.source_fingerprint == digest("a")
    assert original.fingerprint != bound.fingerprint
    assert "neuros_runtime_alignment" not in original.metadata

    authority = bound.metadata["neuros_runtime_alignment"]
    assert authority["plan_sha256"] == digest("e")
    assert authority["dataset_content_sha256"] == digest("a")
    assert authority["manifest_sha256"] == digest("b")
    assert authority["modalities"] == ("behavior", "fmri")
    assert bound.claim_ceiling == original.claim_ceiling


def test_different_exact_plans_remain_distinct_over_same_dataset() -> None:
    original = packet()
    first = bind_runtime_alignment(original, plan(plan_sha256=digest("e")))
    second = bind_runtime_alignment(original, plan(plan_sha256=digest("f")))

    assert first.dataset.source_fingerprint == second.dataset.source_fingerprint == digest("a")
    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"dataset_id": "other"}, "different dataset_id"),
        ({"dataset_content_sha256": digest("d")}, "dataset content identity"),
        ({"manifest_sha256": digest("d")}, "manifest identity"),
        ({"policy": "nearest"}, "exact alignment policy"),
        ({"window_count": 8}, "window_count is inconsistent"),
        ({"modalities": ["fmri"]}, "at least two modalities"),
        ({"modalities": ["fmri", "fmri"]}, "modalities must be unique"),
        ({"modalities": ["fmri", None]}, "alignment modality must be a string"),
        ({"dataset_id": None}, "alignment dataset_id must be a string"),
        ({"sync_group": " run-01"}, "surrounding whitespace"),
    ],
)
def test_tampered_plan_provenance_fails_closed(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bind_runtime_alignment(packet(), plan(**updates))


def test_caller_constructed_dataset_authority_cannot_claim_runtime_alignment() -> None:
    unbound_dataset = DatasetAuthority(
        dataset_id="fixture",
        source_fingerprint=digest("a"),
        access="authorized_restricted",
        source_revision="release-v1",
    )
    with pytest.raises(ValueError, match="neuros_runtime"):
        bind_runtime_alignment(packet(dataset=unbound_dataset), plan())


def test_runtime_dataset_binding_must_match_dataset_authority() -> None:
    forged_dataset = dataset_with_runtime_metadata(
        verified_dataset_content_sha256=digest("d")
    )
    with pytest.raises(ValueError, match="declared and verified"):
        bind_runtime_alignment(packet(dataset=forged_dataset), plan())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"source_verification_semantics": "verified_at_open"},
            "bridge-time runtime content verification",
        ),
        (
            {"declared_dataset_content_sha256": digest("d")},
            "declared and verified dataset content identities",
        ),
        (
            {"lineage_completeness": "complete"},
            "conservative unknown lineage completeness",
        ),
        ({"claim_boundary": ""}, "claim_boundary"),
    ],
)
def test_stale_or_overstated_runtime_dataset_binding_fails_closed(
    updates: dict[str, object], message: str
) -> None:
    forged_dataset = dataset_with_runtime_metadata(**updates)
    with pytest.raises(ValueError, match=message):
        bind_runtime_alignment(packet(dataset=forged_dataset), plan())


def test_reserved_alignment_namespace_cannot_be_rebound() -> None:
    original = packet()
    forged = replace(original, metadata={"neuros_runtime_alignment": {"spoofed": True}})
    with pytest.raises(ValueError, match="reserved"):
        bind_runtime_alignment(forged, plan())
