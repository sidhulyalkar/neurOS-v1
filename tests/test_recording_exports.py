from pathlib import Path

import numpy as np
import pytest

from neuros.contracts import SignalFrame, StreamDescriptor
from neuros.recording import SessionArchiveWriter, export_nwb, export_zarr


async def _archive(root: Path) -> Path:
    writer = SessionArchiveWriter(root, session_id="interop")
    writer.register_stream(
        StreamDescriptor(
            stream_id="eeg",
            modality="eeg",
            sample_rate_hz=100.0,
            channel_names=("C3", "C4"),
            units=("uV", "uV"),
        )
    )
    for index in range(3):
        await writer.write(
            SignalFrame(
                stream_id="eeg",
                sequence_id=index,
                data=np.ones((4, 2), dtype=np.float32) * index,
                sample_rate_hz=100.0,
                host_receive_time_ns=1_000_000_000 + index * 10_000_000,
                synchronized_time_ns=1_000_000_000 + index * 10_000_000,
                metadata={"index": index, "axis_order": ("sample", "channel")},
            )
        )
    await writer.close()
    return root


@pytest.mark.asyncio
async def test_zarr_export_contains_frame_and_timing_arrays(tmp_path: Path):
    zarr = pytest.importorskip("zarr")
    archive = await _archive(tmp_path / "session")
    destination = export_zarr(archive, tmp_path / "session.zarr")
    root = zarr.open_group(str(destination), mode="r")
    assert root["eeg/data"].shape == (3, 4, 2)
    np.testing.assert_array_equal(root["eeg/sequence_id"][:], [0, 1, 2])
    assert "descriptor_json" in root["eeg"].attrs


@pytest.mark.asyncio
async def test_nwb_export_contains_acquisition_and_exact_metadata(tmp_path: Path):
    pynwb = pytest.importorskip("pynwb")
    archive = await _archive(tmp_path / "session")
    destination = export_nwb(archive, tmp_path / "session.nwb")
    with pynwb.NWBHDF5IO(str(destination), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert "eeg" in nwbfile.acquisition
        assert nwbfile.acquisition["eeg"].data.shape == (3, 4, 2)
        assert "neuros_exact_metadata" in nwbfile.scratch
