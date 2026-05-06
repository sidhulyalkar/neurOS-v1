"""
NWB (Neurodata Without Borders) file loader for neuros-astro.

Provides comprehensive loading of calcium imaging data from NWB files,
with support for metadata extraction, multi-session handling, and
compatibility with DANDI datasets.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import pynwb
    from pynwb import NWBHDF5IO
    HAS_PYNWB = True
except ImportError:
    HAS_PYNWB = False
    warnings.warn("pynwb not installed. Install with: pip install pynwb")


@dataclass
class NWBSessionInfo:
    """Metadata for an NWB session."""

    session_id: str
    file_path: str

    # Recording info
    session_description: str
    session_start_time: Any
    institution: Optional[str] = None
    lab: Optional[str] = None

    # Subject info
    subject_id: Optional[str] = None
    species: Optional[str] = None
    age: Optional[str] = None
    sex: Optional[str] = None
    genotype: Optional[str] = None

    # Imaging info
    imaging_plane_description: Optional[str] = None
    excitation_lambda: Optional[float] = None
    imaging_rate_hz: Optional[float] = None
    indicator: Optional[str] = None

    # Data availability
    has_raw_fluorescence: bool = False
    has_dff: bool = False
    has_neuropil: bool = False
    has_roi_masks: bool = False
    n_rois: int = 0
    n_timepoints: int = 0


def summarize_nwb(
    nwb_path: str | Path,
) -> NWBSessionInfo:
    """
    Extract metadata summary from NWB file without loading full data.

    Args:
        nwb_path: Path to NWB file

    Returns:
        NWBSessionInfo object with metadata

    Example:
        >>> info = summarize_nwb("session_12345.nwb")
        >>> print(f"Session: {info.session_id}")
        >>> print(f"ROIs: {info.n_rois}, Timepoints: {info.n_timepoints}")
        >>> print(f"Has dF/F: {info.has_dff}")
    """
    if not HAS_PYNWB:
        raise ImportError("pynwb is required. Install with: pip install pynwb")

    nwb_path = Path(nwb_path)
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    with NWBHDF5IO(str(nwb_path), 'r') as io:
        nwbfile = io.read()

        # Basic session info
        session_id = nwbfile.identifier
        session_description = nwbfile.session_description
        session_start_time = nwbfile.session_start_time
        institution = nwbfile.institution
        lab = nwbfile.lab

        # Subject info
        subject = nwbfile.subject
        if subject:
            subject_id = subject.subject_id
            species = subject.species
            age = subject.age
            sex = subject.sex
            genotype = subject.genotype if hasattr(subject, 'genotype') else None
        else:
            subject_id = None
            species = None
            age = None
            sex = None
            genotype = None

        # Check for ophys data
        has_raw_fluorescence = False
        has_dff = False
        has_neuropil = False
        has_roi_masks = False
        n_rois = 0
        n_timepoints = 0
        imaging_rate_hz = None
        imaging_plane_description = None
        excitation_lambda = None
        indicator = None

        # Get ophys modules
        ophys_modules = [m for m in nwbfile.processing.values()
                        if 'ophys' in m.name.lower()]

        if ophys_modules:
            ophys = ophys_modules[0]

            # Check for fluorescence containers
            if 'Fluorescence' in ophys.data_interfaces:
                fluor = ophys.data_interfaces['Fluorescence']
                roi_response_series = list(fluor.roi_response_series.values())

                if roi_response_series:
                    has_raw_fluorescence = True
                    first_series = roi_response_series[0]
                    n_rois = first_series.data.shape[1] if len(first_series.data.shape) > 1 else 1
                    n_timepoints = first_series.data.shape[0]
                    imaging_rate_hz = first_series.rate if hasattr(first_series, 'rate') else None

            # Check for dF/F
            if 'DfOverF' in ophys.data_interfaces:
                has_dff = True
                dff = ophys.data_interfaces['DfOverF']
                dff_series = list(dff.roi_response_series.values())
                if dff_series and not has_raw_fluorescence:
                    first_series = dff_series[0]
                    n_rois = first_series.data.shape[1] if len(first_series.data.shape) > 1 else 1
                    n_timepoints = first_series.data.shape[0]
                    imaging_rate_hz = first_series.rate if hasattr(first_series, 'rate') else None

            # Check for neuropil
            if 'Neuropil' in ophys.data_interfaces:
                has_neuropil = True

        # Check for plane segmentation (ROI masks)
        if hasattr(nwbfile, 'imaging_planes') and nwbfile.imaging_planes:
            imaging_plane = list(nwbfile.imaging_planes.values())[0]
            imaging_plane_description = imaging_plane.description
            excitation_lambda = imaging_plane.excitation_lambda
            indicator = imaging_plane.indicator

            # Check for ROI masks in modules
            for module in ophys_modules:
                if 'ImageSegmentation' in module.data_interfaces:
                    has_roi_masks = True
                    break

    return NWBSessionInfo(
        session_id=session_id,
        file_path=str(nwb_path),
        session_description=session_description,
        session_start_time=session_start_time,
        institution=institution,
        lab=lab,
        subject_id=subject_id,
        species=species,
        age=age,
        sex=sex,
        genotype=genotype,
        imaging_plane_description=imaging_plane_description,
        excitation_lambda=excitation_lambda,
        imaging_rate_hz=imaging_rate_hz,
        indicator=indicator,
        has_raw_fluorescence=has_raw_fluorescence,
        has_dff=has_dff,
        has_neuropil=has_neuropil,
        has_roi_masks=has_roi_masks,
        n_rois=n_rois,
        n_timepoints=n_timepoints,
    )


def load_nwb_fluorescence(
    nwb_path: str | Path,
    data_type: str = 'dff',
    roi_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, NWBSessionInfo]:
    """
    Load fluorescence data from NWB file.

    Args:
        nwb_path: Path to NWB file
        data_type: Type of data to load ('dff', 'raw', 'neuropil')
        roi_indices: Optional list of ROI indices to load (if None, loads all)

    Returns:
        Tuple of (fluorescence_traces, timestamps, session_info)
        - fluorescence_traces: [n_rois, n_timepoints] array
        - timestamps: [n_timepoints] array of timestamps in seconds
        - session_info: NWBSessionInfo object

    Example:
        >>> traces, times, info = load_nwb_fluorescence("session.nwb", data_type='dff')
        >>> print(f"Loaded {traces.shape[0]} ROIs, {traces.shape[1]} timepoints")
        >>> print(f"Duration: {times[-1] - times[0]:.1f}s")
    """
    if not HAS_PYNWB:
        raise ImportError("pynwb is required. Install with: pip install pynwb")

    nwb_path = Path(nwb_path)
    session_info = summarize_nwb(nwb_path)

    with NWBHDF5IO(str(nwb_path), 'r') as io:
        nwbfile = io.read()

        # Get ophys module
        ophys_modules = [m for m in nwbfile.processing.values()
                        if 'ophys' in m.name.lower()]

        if not ophys_modules:
            raise ValueError("No ophys module found in NWB file")

        ophys = ophys_modules[0]

        # Load requested data type
        if data_type == 'dff':
            if 'DfOverF' not in ophys.data_interfaces:
                raise ValueError("DfOverF data not found in NWB file")

            dff_container = ophys.data_interfaces['DfOverF']
            roi_response_series = list(dff_container.roi_response_series.values())[0]

        elif data_type == 'raw':
            if 'Fluorescence' not in ophys.data_interfaces:
                raise ValueError("Fluorescence data not found in NWB file")

            fluor_container = ophys.data_interfaces['Fluorescence']
            roi_response_series = list(fluor_container.roi_response_series.values())[0]

        elif data_type == 'neuropil':
            if 'Neuropil' not in ophys.data_interfaces:
                raise ValueError("Neuropil data not found in NWB file")

            neuropil_container = ophys.data_interfaces['Neuropil']
            roi_response_series = list(neuropil_container.roi_response_series.values())[0]

        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # Load data
        data = roi_response_series.data[:]  # Load full array
        timestamps = roi_response_series.timestamps[:]

        # Transpose to [n_rois, n_timepoints] if needed
        if len(data.shape) == 2:
            if data.shape[0] > data.shape[1]:
                # Likely [n_timepoints, n_rois] - transpose
                data = data.T

        # Select ROIs if specified
        if roi_indices is not None:
            data = data[roi_indices, :]

    return data, timestamps, session_info


def list_ophys_series(
    nwb_path: str | Path,
) -> List[Dict[str, Any]]:
    """
    List all optical physiology time series available in NWB file.

    Args:
        nwb_path: Path to NWB file

    Returns:
        List of dicts with series information

    Example:
        >>> series_list = list_ophys_series("session.nwb")
        >>> for s in series_list:
        ...     print(f"{s['name']}: {s['shape']}, {s['data_type']}")
    """
    if not HAS_PYNWB:
        raise ImportError("pynwb is required. Install with: pip install pynwb")

    nwb_path = Path(nwb_path)

    series_info = []

    with NWBHDF5IO(str(nwb_path), 'r') as io:
        nwbfile = io.read()

        # Get ophys modules
        ophys_modules = [m for m in nwbfile.processing.values()
                        if 'ophys' in m.name.lower()]

        for ophys in ophys_modules:
            # Check all data interfaces
            for interface_name, interface in ophys.data_interfaces.items():
                if hasattr(interface, 'roi_response_series'):
                    for series_name, series in interface.roi_response_series.items():
                        series_info.append({
                            'name': series_name,
                            'module': ophys.name,
                            'interface': interface_name,
                            'data_type': interface_name,
                            'shape': series.data.shape,
                            'rate_hz': series.rate if hasattr(series, 'rate') else None,
                            'description': series.description if hasattr(series, 'description') else '',
                        })

    return series_info


def extract_roi_masks(
    nwb_path: str | Path,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extract ROI spatial masks from NWB file.

    Args:
        nwb_path: Path to NWB file

    Returns:
        Tuple of (masks, metadata)
        - masks: List of 2D arrays, one per ROI
        - metadata: Dict with imaging plane information

    Example:
        >>> masks, meta = extract_roi_masks("session.nwb")
        >>> print(f"Extracted {len(masks)} ROI masks")
        >>> print(f"Image size: {meta['imaging_plane_size']}")
    """
    if not HAS_PYNWB:
        raise ImportError("pynwb is required. Install with: pip install pynwb")

    nwb_path = Path(nwb_path)

    masks = []
    metadata = {}

    with NWBHDF5IO(str(nwb_path), 'r') as io:
        nwbfile = io.read()

        # Get ophys module
        ophys_modules = [m for m in nwbfile.processing.values()
                        if 'ophys' in m.name.lower()]

        if not ophys_modules:
            raise ValueError("No ophys module found")

        ophys = ophys_modules[0]

        if 'ImageSegmentation' not in ophys.data_interfaces:
            raise ValueError("ImageSegmentation not found in NWB file")

        img_seg = ophys.data_interfaces['ImageSegmentation']
        plane_segmentation = list(img_seg.plane_segmentations.values())[0]

        # Get imaging plane metadata
        imaging_plane = plane_segmentation.imaging_plane

        metadata['imaging_plane_description'] = imaging_plane.description
        metadata['excitation_lambda'] = imaging_plane.excitation_lambda
        metadata['indicator'] = imaging_plane.indicator
        metadata['imaging_rate_hz'] = imaging_plane.imaging_rate

        # Extract masks
        if 'image_mask' in plane_segmentation.colnames:
            for roi_idx in range(len(plane_segmentation)):
                mask = plane_segmentation['image_mask'][roi_idx]
                masks.append(mask)

        elif 'pixel_mask' in plane_segmentation.colnames:
            # Sparse format - need to reconstruct
            # This is more complex, placeholder for now
            print("Warning: pixel_mask format not yet fully supported")

        metadata['n_rois'] = len(masks)
        if masks:
            metadata['imaging_plane_size'] = masks[0].shape

    return masks, metadata
