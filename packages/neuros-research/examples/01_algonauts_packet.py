"""Construct a leakage-bounded Algonauts-style research packet.

This example uses synthetic fingerprints only. It demonstrates how an external game-fMRI
project can bind its own manifest/split authority into neurOS without importing private
brain data or making neurOS own the competition protocol.
"""

from neuros.research import AlgonautsAuthoritySpec, Hypothesis, ResearchAgent


def main() -> None:
    authority = AlgonautsAuthoritySpec(
        dataset_id="cneuromod-mario-sub01",
        source_sha256="a" * 64,
        source_revision="example-manifest-revision",
        split_sha256="b" * 64,
        evaluator_id="algonaut-mario-g2-neural-geometry-v1",
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
    packet = authority.packet(
        experiment_id="mario-g2-dense-temporal-001",
        agent=agent,
        hypothesis=hypothesis,
        code_revision="0123456789abcdef0123456789abcdef01234567",
        seeds=(7, 19, 41),
        representation_sha256="d" * 64,
        compute_budget={"gpu_hours": 8.0, "max_cache_gb": 100},
        dataset_metadata={"game": "mario", "subject": "sub-01"},
        evaluation_metadata={
            "temporal_alignment": {
                "model": "Nilearn-0.10.4-compatible zero-duration SPM regressor",
                "hrf_oversampling": 50,
                "sample_times": "TR midpoints",
            }
        },
    )

    print(packet.fingerprint)


if __name__ == "__main__":
    main()
