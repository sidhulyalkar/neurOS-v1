from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models import (
    GroupedEvaluationData,
    LadderRuntimeConfig,
    LongitudinalCaseAuthority,
    chronological_partition,
    make_nested_calibration_split,
    run_ladder_method,
)

pytest.importorskip("torch")
pytest.importorskip("neuros_sourceweigher")


def _case():
    rng = np.random.default_rng(808)
    X = []
    y = []
    metadata = []
    for session_index, session in enumerate(("0", "1", "2")):
        for label_index, label in enumerate(("left", "right")):
            for trial in range(6):
                signal = rng.normal(scale=0.4, size=(4, 96)).astype(np.float32)
                signal[label_index, 18:72] += (-1.0 if label_index == 0 else 1.0) * (
                    1.0 + 0.05 * session_index
                )
                X.append(signal)
                y.append(label)
                metadata.append(
                    {"subject": "1", "session": session, "run": f"r-{trial // 3}"}
                )
    data = GroupedEvaluationData.from_moabb_result(
        (np.asarray(X), np.asarray(y), metadata), dataset_id="identity-fixture"
    )
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="2",
        order=("0", "1", "2"),
    )
    split = make_nested_calibration_split(partition, evaluation_fraction=0.5, seed=22)
    authority = LongitudinalCaseAuthority.from_split(
        split,
        case_id="identity/subject-1/session-2",
        history_policy="prior",
        observed_group_order=("0", "1", "2"),
    )
    return data, authority


def test_task_decoder_rows_hash_actual_learned_state():
    data, authority = _case()
    result = run_ladder_method(
        data,
        authority,
        method="eegnet",
        budgets_per_class=(0, 1),
        model_seed=17,
        config=LadderRuntimeConfig(epochs=1, batch_size=8, device="cpu"),
    )
    hashes = [row["model_state_sha256"] for row in result.rows]
    assert len(hashes) == 2
    assert all(len(value) == 64 for value in hashes)
    assert all(set(value) <= set("0123456789abcdef") for value in hashes)


def test_weighted_and_unweighted_frozen_lanes_share_learned_state_and_embeddings():
    data, authority = _case()
    cache = {}
    config = LadderRuntimeConfig(epochs=1, batch_size=8, device="cpu")
    frozen = run_ladder_method(
        data,
        authority,
        method="frozen-eegnet",
        budgets_per_class=(0, 1),
        model_seed=17,
        config=config,
        prepared_cache=cache,
    )
    weighted = run_ladder_method(
        data,
        authority,
        method="sourceweigher-eegnet",
        budgets_per_class=(0, 1),
        model_seed=17,
        config=config,
        prepared_cache=cache,
    )

    frozen_state = frozen.encoder_state_manifest
    weighted_state = weighted.encoder_state_manifest
    assert len(frozen_state["model_state_sha256"]) == 64
    assert frozen_state["model_state_sha256"] == weighted_state["model_state_sha256"]
    assert frozen_state["representation_sha256"] == weighted_state["representation_sha256"]
    assert frozen_state["encoder_state_fingerprint"] == weighted_state["encoder_state_fingerprint"]
    assert {row["model_state_sha256"] for row in frozen.rows} == {
        frozen_state["model_state_sha256"]
    }
    assert {row["model_state_sha256"] for row in weighted.rows} == {
        frozen_state["model_state_sha256"]
    }
