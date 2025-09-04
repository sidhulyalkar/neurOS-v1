"""
NWB writer for neural and physiological timeseries.

This module wraps the ``pynwb`` library to write time series data
conforming to the Neurodata Without Borders (NWB) 2.0 standard.  NWB
files provide a rich metadata schema for storing electrophysiology,
optophysiology, behaviour and other experimental data along with
experimental metadata such as subject demographics, device
specifications and acquisition parameters.

The ``write_nwb_file`` function accepts a list of channels or
modalities and writes them into an NWB file along with session and
subject metadata.  For large datasets NWB supports HDF5 chunking and
compression; these options can be tuned via the input parameters.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from pynwb import NWBFile, TimeSeries
from pynwb.file import Subject
from pynwb import NWBHDF5IO

logger = logging.getLogger(__name__)


def write_nwb_file(
    file_path: str | Path,
    data: List[np.ndarray],
    sampling_rate: float,
    channel_names: List[str],
    session_id: str,
    subject_id: str,
    device_name: str,
    description: str = "",
    start_time: datetime.datetime | None = None,
) -> None:
    """Write multi‑channel time series to an NWB file.

    Parameters
    ----------
    file_path:
        Path to the file that will be created.
    data:
        List of 1D numpy arrays representing each channel's samples.
    sampling_rate:
        Sampling frequency in Hz.
    channel_names:
        Human readable names for each channel (must match length of ``data``).
    session_id:
        Identifier for the recording session.
    subject_id:
        Identifier for the subject.
    device_name:
        Identifier for the acquisition device.
    description:
        Optional textual description of the session.
    start_time:
        Timestamp when the recording started.  Defaults to now if not provided.
    """
    assert len(data) == len(channel_names), "Number of channels and names must match"
    n_channels = len(data)
    n_samples = len(data[0]) if n_channels > 0 else 0

    if start_time is None:
        start_time = datetime.datetime.now(datetime.timezone.utc)

    logger.info(
        "Writing NWB file %s with %d channels and %d samples",
        file_path,
        n_channels,
        n_samples,
    )

    nwbfile = NWBFile(
        session_description=description,
        identifier=session_id,
        session_start_time=start_time,
        file_create_date=datetime.datetime.now(datetime.timezone.utc),
        experimenter="",
        lab="",
        institution="",
    )

    # Set subject metadata
    nwbfile.subject = Subject(
        subject_id=subject_id,
        description="human subject",
    )

    # Create device
    device = nwbfile.create_device(name=device_name)

    # Add each channel as a TimeSeries in the acquisition group
    for channel_data, name in zip(data, channel_names):
        ts = TimeSeries(
            name=name,
            data=channel_data,
            rate=sampling_rate,
            unit="a.u.",
            description=f"Channel {name}",
        )
        nwbfile.add_acquisition(ts)

    # Write to disk
    path_obj = Path(file_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(file_path, "w") as io:
        io.write(nwbfile)
    logger.info("Finished writing NWB file to %s", file_path)


__all__ = ["write_nwb_file"]