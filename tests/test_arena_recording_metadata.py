from __future__ import annotations

import numpy as np

from neuros.arena.baselines import save_eeg_baseline
from neuros.arena.recording import (
    ElectrodeCoordinate,
    RecordingMetadata,
    load_recording_metadata,
    recording_sidecar_path,
)


def test_recording_metadata_round_trip_preserves_bids_aligned_provenance(tmp_path):
    baseline = tmp_path / "subject-01-rest.npz"
    channels = ("PO7", "Oz", "PO8")
    metadata = RecordingMetadata(
        dataset="public-ssvep-example",
        subject="01",
        session="01",
        run="02",
        task="ssvep",
        acquisition="eeg",
        source_locator="bids://sub-01/ses-01/eeg/sub-01_ses-01_task-ssvep_run-02_eeg.edf",
        source_format="EDF via MNE",
        source_license="dataset-defined",
        reference="Cz",
        line_frequency_hz=60.0,
        channel_units={name: "uV" for name in channels},
        channel_types={name: "eeg" for name in channels},
        coordinate_system="CapTrak",
        coordinate_units="m",
        electrodes=(
            ElectrodeCoordinate("PO7", -0.03, -0.08, 0.07),
            ElectrodeCoordinate("Oz", 0.00, -0.10, 0.06),
            ElectrodeCoordinate("PO8", 0.03, -0.08, 0.07),
        ),
        preprocessing=("selected_occipital_channels",),
        notes=("de-identified public data",),
    )
    save_eeg_baseline(
        baseline,
        data_uv=np.zeros((3, 250), dtype=float),
        sampling_rate_hz=250.0,
        channel_names=channels,
        recording_metadata=metadata,
    )
    sidecar = recording_sidecar_path(baseline)
    assert sidecar.exists()
    loaded = load_recording_metadata(sidecar)
    assert loaded == metadata
    assert loaded.task == "ssvep"
    assert loaded.channel_units["Oz"] == "uV"
    assert loaded.electrodes[1].name == "Oz"


def test_recording_metadata_rejects_coordinates_for_absent_channel(tmp_path):
    baseline = tmp_path / "bad.npz"
    metadata = RecordingMetadata(
        coordinate_system="CapTrak",
        coordinate_units="m",
        electrodes=(ElectrodeCoordinate("Pz", 0.0, -0.05, 0.08),),
    )
    try:
        save_eeg_baseline(
            baseline,
            data_uv=np.zeros((1, 64), dtype=float),
            sampling_rate_hz=250.0,
            channel_names=("Oz",),
            recording_metadata=metadata,
        )
    except ValueError as exc:
        assert "not present" in str(exc)
    else:
        raise AssertionError("metadata referencing a missing channel should fail")
