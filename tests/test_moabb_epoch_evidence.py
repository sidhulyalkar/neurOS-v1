from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models.moabb_epochs import (
    MOABBEpochDescriptor,
    collect_moabb_epochs,
)


class FakeEpochs:
    def __init__(self, *, n_trials: int = 8, channel_names=("C3", "C4")) -> None:
        self._data = np.arange(n_trials * 2 * 5, dtype=np.float32).reshape(
            n_trials, 2, 5
        )
        self.ch_names = list(channel_names)
        self.info = {"sfreq": 128.0}
        self.times = np.asarray([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=np.float64)
        self.event_id = {"left_hand": 1, "right_hand": 2}

    def get_data(self):
        return self._data.copy()

    def get_channel_types(self):
        return ["eeg", "eeg"]


class FakeParadigm:
    def __init__(self, epochs: FakeEpochs) -> None:
        self.epochs = epochs
        self.calls: list[dict[str, object]] = []

    def get_data(self, **kwargs):
        self.calls.append(dict(kwargs))
        n_trials = len(self.epochs.get_data())
        labels = np.asarray(["left_hand", "right_hand"] * (n_trials // 2), dtype=str)
        metadata = [
            {
                "subject": 1,
                "session": 0 if index < n_trials // 2 else 1,
                "run": index // 2,
            }
            for index in range(n_trials)
        ]
        return self.epochs, labels, metadata


class FakeDataset:
    code = "Fake"


def test_collect_moabb_epochs_preserves_processed_signal_contract():
    paradigm = FakeParadigm(FakeEpochs())
    dataset = FakeDataset()
    data, descriptor = collect_moabb_epochs(
        dataset,
        paradigm,
        subjects=[1],
        dataset_id="fixture-moabb",
    )

    assert len(paradigm.calls) == 1
    assert paradigm.calls[0]["dataset"] is dataset
    assert paradigm.calls[0]["subjects"] == [1]
    assert paradigm.calls[0]["return_epochs"] is True
    assert data.X.shape == (8, 2, 5)
    assert data.groups["session"].tolist() == ["0", "0", "0", "0", "1", "1", "1", "1"]
    assert descriptor.channel_names == ("C3", "C4")
    assert descriptor.channel_types == ("eeg", "eeg")
    assert descriptor.sampling_rate_hz == 128.0
    assert descriptor.n_times == 5
    assert descriptor.epoch_start_s == -0.5
    assert descriptor.epoch_end_s == 0.5
    assert descriptor.event_id == (("left_hand", 1), ("right_hand", 2))
    assert len(descriptor.signal_contract_sha256) == 64
    assert len(descriptor.sha256) == 64


def test_signal_contract_identity_excludes_subject_specific_trial_count():
    first = MOABBEpochDescriptor(
        channel_names=("C3", "C4"),
        channel_types=("eeg", "eeg"),
        sampling_rate_hz=128.0,
        n_times=5,
        epoch_start_s=-0.5,
        epoch_end_s=0.5,
        event_id=(("left_hand", 1), ("right_hand", 2)),
        n_trials=60,
    )
    second = MOABBEpochDescriptor(
        channel_names=("C3", "C4"),
        channel_types=("eeg", "eeg"),
        sampling_rate_hz=128.0,
        n_times=5,
        epoch_start_s=-0.5,
        epoch_end_s=0.5,
        event_id=(("right_hand", 2), ("left_hand", 1)),
        n_trials=80,
    )

    assert first.signal_contract_sha256 == second.signal_contract_sha256
    assert first.sha256 != second.sha256


def test_collect_moabb_epochs_rejects_channel_geometry_mismatch():
    paradigm = FakeParadigm(FakeEpochs(channel_names=("C3",)))
    with pytest.raises(ValueError, match="channel order length"):
        collect_moabb_epochs(
            FakeDataset(),
            paradigm,
            subjects=[1],
            dataset_id="fixture-moabb",
        )


def test_descriptor_rejects_duplicate_channel_identity():
    with pytest.raises(ValueError, match="duplicates"):
        MOABBEpochDescriptor(
            channel_names=("C3", "C3"),
            channel_types=("eeg", "eeg"),
            sampling_rate_hz=128.0,
            n_times=5,
            epoch_start_s=-0.5,
            epoch_end_s=0.5,
            event_id=(("left_hand", 1), ("right_hand", 2)),
            n_trials=8,
        )
