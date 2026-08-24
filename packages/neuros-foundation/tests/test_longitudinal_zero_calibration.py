from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models import (
    GroupedEvaluationData,
    LongitudinalCaseAuthority,
    chronological_partition,
    make_nested_calibration_split,
)
from neuros.foundation_models.longitudinal_transfer import (
    FrozenTransferMethodSpec,
    prepare_frozen_encoder_case,
    run_frozen_transfer_case,
)

pytest.importorskip("torch")


def test_frozen_encoder_supports_empty_target_calibration_pool():
    rng = np.random.default_rng(44)
    X = []
    y = []
    metadata = []

    # Source history has enough trials to train; the target has exactly one
    # example per class, so the evidence contract reserves both for final eval
    # and exposes a legal maximum calibration budget of zero.
    for label_index, label in enumerate(("left", "right")):
        for trial in range(6):
            signal = rng.normal(size=(4, 64)).astype(np.float32)
            signal[label_index, 16:48] += -1.0 if label_index == 0 else 1.0
            X.append(signal)
            y.append(label)
            metadata.append({"subject": "1", "session": "0", "run": f"s-{trial}"})

    for label_index, label in enumerate(("left", "right")):
        signal = rng.normal(size=(4, 64)).astype(np.float32)
        signal[label_index, 16:48] += -1.0 if label_index == 0 else 1.0
        X.append(signal)
        y.append(label)
        metadata.append({"subject": "1", "session": "1", "run": "target"})

    data = GroupedEvaluationData.from_moabb_result(
        (np.asarray(X), np.asarray(y), metadata),
        dataset_id="tiny-zero-calibration",
    )
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="1",
        order=("0", "1"),
    )
    split = make_nested_calibration_split(partition, evaluation_fraction=0.5, seed=3)
    assert split.max_budget_per_class == 0

    authority = LongitudinalCaseAuthority.from_split(
        split,
        case_id="tiny/subject-1/session-1",
        history_policy="prior",
        observed_group_order=("0", "1"),
    )
    kwargs = {
        "n_epochs": 1,
        "batch_size": 8,
        "device": "cpu",
        "temporal_filters": 4,
        "depth_multiplier": 1,
        "separable_filters": 8,
        "temporal_kernel": 15,
        "separable_kernel": 7,
    }
    prepared = prepare_frozen_encoder_case(
        data,
        authority,
        encoder_id="eegnet",
        encoder_seed=9,
        encoder_kwargs=kwargs,
    )
    assert prepared.target_pool_indices.size == 0
    assert prepared.target_pool_embedding.shape == (0, prepared.source_embedding.shape[1])
    assert len(prepared.representation_sha256) == 64

    result = run_frozen_transfer_case(
        data,
        authority,
        spec=FrozenTransferMethodSpec(
            method_id="frozen-eegnet",
            strategy="frozen-logistic",
            encoder_id="eegnet",
            encoder_seed=9,
            encoder_kwargs=kwargs,
        ),
        budgets_per_class=(0,),
        prepared=prepared,
    )
    assert len(result.rows) == 1
    assert result.rows[0]["status"] == "ok"
    assert result.rows[0]["calibration_per_class"] == 0
