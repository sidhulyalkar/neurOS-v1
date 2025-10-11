"""
NWB (Neurodata Without Borders) file format support.

This module provides utilities for reading and writing neural data in the
NWB 2.0 format, the community standard for sharing neurophysiology data.

References:
    - NWB 2.0 Specification: https://nwb-schema.readthedocs.io/
    - PyNWB Documentation: https://pynwb.readthedocs.io/
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import pynwb (optional dependency)
try:
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.ecephys import ElectricalSeries, LFP, FilteredEphys
    from pynwb.behavior import BehavioralTimeSeries, Position, SpatialSeries
    from pynwb.file import Subject
    from pynwb.epoch import TimeIntervals

    NWB_AVAILABLE = True
except ImportError:
    NWB_AVAILABLE = False
    logger.warning("pynwb not available. Install with: pip install pynwb")


class NWBLoader:
    """
    Load neural data from NWB files.

    NWB (Neurodata Without Borders) is a data standard for neurophysiology,
    providing a common format for storing and sharing diverse neuroscience data.

    Parameters
    ----------
    file_path : str or Path
        Path to the NWB file to load.

    Attributes
    ----------
    nwbfile : NWBFile
        The loaded NWB file object.
    io : NWBHDF5IO
        The HDF5 I/O object for the NWB file.

    Examples
    --------
    >>> loader = NWBLoader("session_001.nwb")
    >>> spikes = loader.load_spikes()
    >>> lfp = loader.load_lfp()
    >>> behavior = loader.load_behavior()
    >>> loader.close()
    """

    def __init__(self, file_path: Union[str, Path]):
        if not NWB_AVAILABLE:
            raise ImportError(
                "pynwb is required for NWB support. "
                "Install it with: pip install pynwb"
            )

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"NWB file not found: {file_path}")

        logger.info(f"Loading NWB file: {self.file_path}")
        self.io = NWBHDF5IO(str(self.file_path), "r")
        self.nwbfile = self.io.read()

    def load_spikes(
        self,
        unit_ids: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load spike times from the NWB file.

        Parameters
        ----------
        unit_ids : list of int, optional
            List of unit IDs to load. If None, loads all units.

        Returns
        -------
        dict
            Dictionary mapping unit IDs to spike time arrays.
            Keys are string unit IDs, values are numpy arrays of spike times.
        """
        if not hasattr(self.nwbfile, "units") or self.nwbfile.units is None:
            logger.warning("No units table found in NWB file")
            return {}

        units_table = self.nwbfile.units
        spike_dict = {}

        if unit_ids is None:
            unit_ids = range(len(units_table))

        for unit_id in unit_ids:
            try:
                spike_times = units_table["spike_times"][unit_id]
                spike_dict[f"unit_{unit_id}"] = np.array(spike_times)
            except (IndexError, KeyError) as e:
                logger.warning(f"Could not load spikes for unit {unit_id}: {e}")

        logger.info(f"Loaded spikes for {len(spike_dict)} units")
        return spike_dict

    def load_lfp(
        self,
        electrode_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Load LFP (Local Field Potential) data from the NWB file.

        Parameters
        ----------
        electrode_ids : list of int, optional
            List of electrode IDs to load. If None, loads all electrodes.

        Returns
        -------
        dict
            Dictionary containing:
            - 'data': ndarray of shape (n_timepoints, n_electrodes)
            - 'timestamps': ndarray of timestamps
            - 'rate': sampling rate in Hz
            - 'electrode_ids': list of electrode IDs
        """
        # Try to find LFP data in acquisition or processing modules
        lfp_data = None

        if hasattr(self.nwbfile, "processing"):
            for module_name in self.nwbfile.processing:
                module = self.nwbfile.processing[module_name]
                if "LFP" in module.data_interfaces:
                    lfp = module.data_interfaces["LFP"]
                    lfp_data = lfp.electrical_series
                    break

        if lfp_data is None and hasattr(self.nwbfile, "acquisition"):
            for name, obj in self.nwbfile.acquisition.items():
                if isinstance(obj, ElectricalSeries) and "lfp" in name.lower():
                    lfp_data = {name: obj}
                    break

        if lfp_data is None:
            logger.warning("No LFP data found in NWB file")
            return {}

        # Get the first electrical series
        series_name = list(lfp_data.keys())[0]
        series = lfp_data[series_name]

        # Extract data
        data = series.data[:]
        timestamps = series.timestamps[:] if series.timestamps is not None else None
        rate = series.rate if hasattr(series, "rate") else None

        # Handle electrode selection
        if electrode_ids is not None:
            data = data[:, electrode_ids]

        result = {
            "data": data,
            "timestamps": timestamps,
            "rate": rate,
            "electrode_ids": electrode_ids or list(range(data.shape[1])),
            "series_name": series_name,
        }

        logger.info(f"Loaded LFP data: shape {data.shape}, rate {rate} Hz")
        return result

    def load_behavior(self) -> Dict[str, Any]:
        """
        Load behavioral data from the NWB file.

        Returns
        -------
        dict
            Dictionary containing behavioral time series data.
            Keys depend on what's available in the file (position, velocity, etc.)
        """
        behavior_data = {}

        if not hasattr(self.nwbfile, "processing"):
            logger.warning("No processing modules found in NWB file")
            return behavior_data

        # Look for behavior modules
        for module_name in self.nwbfile.processing:
            module = self.nwbfile.processing[module_name]

            # Check for position data
            if "Position" in module.data_interfaces:
                position = module.data_interfaces["Position"]
                for spatial_series_name in position.spatial_series:
                    spatial_series = position.spatial_series[spatial_series_name]
                    behavior_data[f"position_{spatial_series_name}"] = {
                        "data": spatial_series.data[:],
                        "timestamps": spatial_series.timestamps[:],
                        "unit": spatial_series.unit if hasattr(spatial_series, "unit") else None,
                    }

            # Check for other behavioral time series
            for interface_name, interface in module.data_interfaces.items():
                if isinstance(interface, BehavioralTimeSeries):
                    for ts_name in interface.time_series:
                        ts = interface.time_series[ts_name]
                        behavior_data[f"{interface_name}_{ts_name}"] = {
                            "data": ts.data[:],
                            "timestamps": ts.timestamps[:],
                            "unit": ts.unit if hasattr(ts, "unit") else None,
                        }

        if behavior_data:
            logger.info(f"Loaded {len(behavior_data)} behavioral time series")
        else:
            logger.warning("No behavioral data found in NWB file")

        return behavior_data

    def load_trials(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Load trial information from the NWB file.

        Returns
        -------
        dict or None
            Dictionary containing trial data (start_time, stop_time, etc.)
            or None if no trials are defined.
        """
        if not hasattr(self.nwbfile, "trials") or self.nwbfile.trials is None:
            logger.warning("No trials table found in NWB file")
            return None

        trials_table = self.nwbfile.trials
        trials_data = {}

        for column_name in trials_table.colnames:
            trials_data[column_name] = trials_table[column_name][:]

        logger.info(f"Loaded {len(trials_table)} trials")
        return trials_data

    def get_session_metadata(self) -> Dict[str, Any]:
        """
        Get session metadata from the NWB file.

        Returns
        -------
        dict
            Dictionary containing session metadata (description, start time, etc.)
        """
        metadata = {
            "session_description": self.nwbfile.session_description,
            "session_start_time": self.nwbfile.session_start_time,
            "experimenter": self.nwbfile.experimenter,
            "lab": self.nwbfile.lab if hasattr(self.nwbfile, "lab") else None,
            "institution": self.nwbfile.institution if hasattr(self.nwbfile, "institution") else None,
        }

        if hasattr(self.nwbfile, "subject") and self.nwbfile.subject is not None:
            metadata["subject"] = {
                "subject_id": self.nwbfile.subject.subject_id,
                "species": self.nwbfile.subject.species if hasattr(self.nwbfile.subject, "species") else None,
                "age": self.nwbfile.subject.age if hasattr(self.nwbfile.subject, "age") else None,
            }

        return metadata

    def close(self):
        """Close the NWB file."""
        if self.io is not None:
            self.io.close()
            logger.info("Closed NWB file")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class NWBWriter:
    """
    Write neural data to NWB files.

    Parameters
    ----------
    file_path : str or Path
        Path where the NWB file will be saved.
    session_description : str
        Description of the recording session.
    session_start_time : datetime, optional
        Start time of the session. If None, uses current time.
    experimenter : str, optional
        Name of the experimenter.
    lab : str, optional
        Lab name.
    institution : str, optional
        Institution name.

    Examples
    --------
    >>> writer = NWBWriter(
    ...     "output.nwb",
    ...     session_description="Motor imagery task",
    ...     experimenter="John Doe"
    ... )
    >>> writer.add_spikes(spike_data, sampling_rate=30000.0)
    >>> writer.add_lfp(lfp_data, sampling_rate=1000.0)
    >>> writer.save()
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        session_description: str,
        session_start_time: Optional[datetime] = None,
        experimenter: Optional[str] = None,
        lab: Optional[str] = None,
        institution: Optional[str] = None,
    ):
        if not NWB_AVAILABLE:
            raise ImportError(
                "pynwb is required for NWB support. "
                "Install it with: pip install pynwb"
            )

        self.file_path = Path(file_path)

        if session_start_time is None:
            session_start_time = datetime.now()

        self.nwbfile = NWBFile(
            session_description=session_description,
            identifier=str(self.file_path.stem),
            session_start_time=session_start_time,
            experimenter=experimenter,
            lab=lab,
            institution=institution,
        )

        logger.info(f"Created NWB file structure for: {self.file_path}")

    def add_spikes(
        self,
        spike_times: Dict[str, np.ndarray],
        unit_ids: Optional[List[int]] = None,
        waveform_mean: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Add spike data to the NWB file.

        Parameters
        ----------
        spike_times : dict
            Dictionary mapping unit IDs to spike time arrays.
        unit_ids : list of int, optional
            List of unit IDs. If None, uses keys from spike_times.
        waveform_mean : dict, optional
            Dictionary mapping unit IDs to mean waveforms.
        """
        for unit_name, times in spike_times.items():
            self.nwbfile.add_unit(
                spike_times=times,
                id=unit_name if unit_ids is None else unit_ids.pop(0),
                waveform_mean=waveform_mean.get(unit_name) if waveform_mean else None,
            )

        logger.info(f"Added {len(spike_times)} units to NWB file")

    def add_lfp(
        self,
        lfp_data: np.ndarray,
        sampling_rate: float,
        electrode_ids: Optional[List[int]] = None,
        timestamps: Optional[np.ndarray] = None,
    ):
        """
        Add LFP data to the NWB file.

        Parameters
        ----------
        lfp_data : ndarray
            LFP data of shape (n_timepoints, n_electrodes).
        sampling_rate : float
            Sampling rate in Hz.
        electrode_ids : list of int, optional
            List of electrode IDs.
        timestamps : ndarray, optional
            Timestamps for each sample. If None, uses sampling rate.
        """
        # Create electrode table if it doesn't exist
        if self.nwbfile.electrodes is None:
            n_electrodes = lfp_data.shape[1] if lfp_data.ndim > 1 else 1
            for i in range(n_electrodes):
                self.nwbfile.add_electrode(
                    id=electrode_ids[i] if electrode_ids else i,
                    x=0.0, y=0.0, z=0.0,  # Placeholder locations
                    imp=np.nan,
                    location="unknown",
                    filtering="unknown",
                    group=self.nwbfile.create_electrode_group(
                        name=f"electrode_group_{i}",
                        description="Electrode group",
                        location="unknown",
                        device=self.nwbfile.create_device(name="device")
                    ),
                )

        # Create electrical series
        lfp_series = ElectricalSeries(
            name="LFP",
            data=lfp_data,
            electrodes=list(range(lfp_data.shape[1])) if electrode_ids is None else electrode_ids,
            starting_time=0.0 if timestamps is None else timestamps[0],
            rate=sampling_rate if timestamps is None else None,
            timestamps=timestamps,
        )

        # Add to processing module
        ecephys_module = self.nwbfile.create_processing_module(
            name="ecephys",
            description="Processed extracellular electrophysiology data",
        )
        ecephys_module.add(LFP(electrical_series=lfp_series))

        logger.info(f"Added LFP data to NWB file: shape {lfp_data.shape}, rate {sampling_rate} Hz")

    def add_behavior(
        self,
        behavior_data: Dict[str, np.ndarray],
        sampling_rate: float,
        timestamps: Optional[np.ndarray] = None,
    ):
        """
        Add behavioral data to the NWB file.

        Parameters
        ----------
        behavior_data : dict
            Dictionary of behavioral time series (e.g., {'position': array, 'velocity': array}).
        sampling_rate : float
            Sampling rate in Hz.
        timestamps : ndarray, optional
            Timestamps for each sample.
        """
        behavior_module = self.nwbfile.create_processing_module(
            name="behavior",
            description="Behavioral data",
        )

        for name, data in behavior_data.items():
            time_series = TimeSeries(
                name=name,
                data=data,
                unit="unknown",
                starting_time=0.0 if timestamps is None else timestamps[0],
                rate=sampling_rate if timestamps is None else None,
                timestamps=timestamps,
            )

            behavior_ts = BehavioralTimeSeries(name=f"BehavioralTimeSeries_{name}")
            behavior_ts.add_timeseries(time_series)
            behavior_module.add(behavior_ts)

        logger.info(f"Added {len(behavior_data)} behavioral time series to NWB file")

    def save(self):
        """Save the NWB file to disk."""
        with NWBHDF5IO(str(self.file_path), "w") as io:
            io.write(self.nwbfile)

        logger.info(f"Saved NWB file to: {self.file_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save()
