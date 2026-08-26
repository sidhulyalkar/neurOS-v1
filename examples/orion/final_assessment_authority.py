"""Dependency-light ORION final-assessment authority example."""

from __future__ import annotations

import hashlib
import json

from orion import (
    ArtifactIdentity,
    FinalAssessmentAuthority,
    FinalAssessmentRecord,
    SelectedState,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def main() -> None:
    source_authority = "longitudinal-three-way-authority-v2"
    final = FinalAssessmentAuthority(
        authority_id="subject-07/session-03/final-v1",
        dataset_id="example-eeg",
        split_unit="session",
        assessment_indices=(20, 21, 22, 23),
        processed_data_sha256=_sha("processed-example-eeg"),
        n_samples=24,
        source_authority_fingerprint=source_authority,
        metric_names=("balanced_accuracy", "ece"),
        protocol_fingerprint="prospective-session-v3",
        seed=11,
        metadata={"policy": "score once after state selection"},
    )
    selected = SelectedState.frozen(
        selection_id="subject-07/session-03/frozen-baseline",
        source_authority_fingerprint=source_authority,
        artifact=ArtifactIdentity(
            artifact_id="decoder/frozen-baseline",
            artifact_type="decoder-state",
            sha256=_sha("frozen-baseline-state"),
        ),
        metadata={"calibration_examples": 0},
    )
    record = FinalAssessmentRecord.record(
        selected,
        authority=final,
        assessment_indices=final.assessment_indices,
        metrics={"balanced_accuracy": 0.68, "ece": 0.09},
    )
    print(
        json.dumps(
            {
                "authority": final.to_dict(),
                "selected_state": selected.to_dict(),
                "final_assessment": record.to_dict(),
                "claim_boundary": {
                    "example_only": True,
                    "real_neural_data": False,
                    "efficacy": False,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
