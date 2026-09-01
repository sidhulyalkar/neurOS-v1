"""Construct a leakage-bounded Algonauts-style research packet.

This example uses synthetic fingerprints only. It demonstrates how an external game-fMRI
project can bind its own manifest/split authority into neurOS without importing private
brain data or making neurOS own the competition protocol.
"""

from neuros.research import (
    DatasetAuthority,
    EvaluationAuthority,
    ExperimentPacket,
    Hypothesis,
    ResearchAgent,
)


def main() -> None:
    dataset = DatasetAuthority(
        dataset_id="cneuromod-mario-sub01",
        source_fingerprint="a" * 64,
        access="authorized_restricted",
        source_revision="example-manifest-revision",
        metadata={"game": "mario", "subject": "sub-01"},
    )
    evaluation = EvaluationAuthority(
        evaluator_id="algonaut-mario-g2",
        split_fingerprint="b" * 64,
        metric_names=(
            "pearson",
            "rsa_spearman",
            "winner_frequency",
            "runtime_seconds",
        ),
        evaluation_domains=("validation", "g2_ood", "geometry", "operational"),
        optimization_boundary="train_validation",
        forbidden_feedback=(
            "hidden_test_targets",
            "private_leaderboard",
            "g2_ood_for_model_selection",
            "g3_cross_game_for_model_selection",
            "g4_held_subject_for_model_selection",
        ),
        metadata={
            "temporal_alignment": {
                "model": "Nilearn-compatible zero-duration SPM regressor",
                "hrf_oversampling": 50,
                "sample_times": "TR midpoints",
            }
        },
    )
    agent = ResearchAgent(
        agent_id="representation-scout-01",
        kind="frontier_model",
        provider="example-provider",
        model="example-model",
        version="pinned-version",
        prompt_sha256="c" * 64,
        role="representation_scout",
    )
    hypothesis = Hypothesis(
        hypothesis_id="dense-temporal-v1",
        statement=(
            "Dense temporally structured video features improve unseen-level brain "
            "encoding while surviving temporal-shift nulls."
        ),
        changed_variables=("representation.pooling", "representation.frame_step"),
    )
    packet = ExperimentPacket(
        experiment_id="mario-g2-dense-temporal-001",
        dataset=dataset,
        evaluation=evaluation,
        agent=agent,
        hypothesis=hypothesis,
        code_revision="0123456789abcdef0123456789abcdef01234567",
        seeds=(7, 19, 41),
        information_regimes=("external_pretrained", "train_only_inductive"),
        claim_ceiling="predictive_ood",
        representation_fingerprint="d" * 64,
        compute_budget={"gpu_hours": 8.0, "max_cache_gb": 100},
    )

    print(packet.fingerprint)


if __name__ == "__main__":
    main()
