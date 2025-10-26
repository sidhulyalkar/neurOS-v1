"""
WebDataset shard writer for NeuroFMX multi-modal data.

Converts neural data from various formats (NWB, raw arrays, etc.) into
WebDataset tar shards for scalable distributed training.
"""

import io
import json
import pickle
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm


class WebDatasetWriter:
    """Write multi-modal neural data to WebDataset shards.

    Parameters
    ----------
    output_dir : str or Path
        Directory to write shards.
    shard_size : int, optional
        Number of samples per shard.
        Default: 1000.
    shard_name_pattern : str, optional
        Pattern for shard filenames (must include {shard_idx:06d}).
        Default: "shard_{shard_idx:06d}.tar".
    compression : str, optional
        Compression type ("none", "gz", "bz2", "xz").
        Default: "none".
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        shard_size: int = 1000,
        shard_name_pattern: str = "shard_{shard_idx:06d}.tar",
        compression: str = "none",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shard_size = shard_size
        self.shard_name_pattern = shard_name_pattern
        self.compression = compression

        # State
        self.current_shard_idx = 0
        self.current_sample_idx = 0
        self.samples_in_current_shard = 0
        self.current_tar = None
        self.total_samples_written = 0

        # Statistics
        self.shard_metadata = []

    def _get_tar_mode(self) -> str:
        """Get tarfile mode based on compression."""
        if self.compression == "none":
            return "w"
        elif self.compression == "gz":
            return "w:gz"
        elif self.compression == "bz2":
            return "w:bz2"
        elif self.compression == "xz":
            return "w:xz"
        else:
            raise ValueError(f"Unknown compression: {self.compression}")

    def _open_new_shard(self):
        """Open a new shard file."""
        if self.current_tar is not None:
            self._close_current_shard()

        shard_name = self.shard_name_pattern.format(shard_idx=self.current_shard_idx)
        shard_path = self.output_dir / shard_name

        mode = self._get_tar_mode()
        self.current_tar = tarfile.open(shard_path, mode)
        self.samples_in_current_shard = 0

        # Track metadata
        self.shard_metadata.append({
            "shard_idx": self.current_shard_idx,
            "shard_name": shard_name,
            "shard_path": str(shard_path),
            "samples": [],
        })

    def _close_current_shard(self):
        """Close current shard."""
        if self.current_tar is not None:
            self.current_tar.close()
            self.current_tar = None
            self.current_shard_idx += 1

    def _add_to_tar(self, name: str, data: bytes):
        """Add data to tar file.

        Parameters
        ----------
        name : str
            Name within tar archive.
        data : bytes
            Data to write.
        """
        tarinfo = tarfile.TarInfo(name=name)
        tarinfo.size = len(data)
        self.current_tar.addfile(tarinfo, io.BytesIO(data))

    def write_sample(
        self,
        sample: Dict[str, Any],
        sample_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Write a single multi-modal sample to shards.

        Parameters
        ----------
        sample : dict
            Dictionary of modality_name -> data.
            Data can be numpy arrays, torch tensors, or any picklable object.
        sample_id : str, optional
            Unique sample ID. If None, uses auto-incrementing index.
        metadata : dict, optional
            Additional metadata to store with sample.
        """
        # Open new shard if needed
        if self.current_tar is None or self.samples_in_current_shard >= self.shard_size:
            self._open_new_shard()

        # Generate sample ID
        if sample_id is None:
            sample_id = f"sample_{self.current_sample_idx:09d}"
        self.current_sample_idx += 1

        # Write each modality
        modalities_written = []
        for modality_name, data in sample.items():
            # Convert to appropriate format
            if isinstance(data, (np.ndarray, torch.Tensor)):
                # Pickle tensors/arrays for efficient storage
                data_bytes = pickle.dumps(data)
                extension = "pyd"
            elif isinstance(data, (dict, list)):
                # JSON for structured data
                data_bytes = json.dumps(data).encode("utf-8")
                extension = "json"
            else:
                # Pickle anything else
                data_bytes = pickle.dumps(data)
                extension = "pyd"

            # Add to tar
            file_name = f"{sample_id}.{modality_name}.{extension}"
            self._add_to_tar(file_name, data_bytes)
            modalities_written.append(modality_name)

        # Write metadata
        sample_metadata = {
            "sample_id": sample_id,
            "modalities": modalities_written,
            "shard_idx": self.current_shard_idx,
            "sample_idx_in_shard": self.samples_in_current_shard,
        }
        if metadata is not None:
            sample_metadata.update(metadata)

        metadata_bytes = json.dumps(sample_metadata, indent=2).encode("utf-8")
        self._add_to_tar(f"{sample_id}.metadata.json", metadata_bytes)

        # Update stats
        self.samples_in_current_shard += 1
        self.total_samples_written += 1
        self.shard_metadata[-1]["samples"].append(sample_metadata)

    def write_batch(
        self,
        samples: List[Dict[str, Any]],
        sample_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True,
    ):
        """Write a batch of samples.

        Parameters
        ----------
        samples : list of dict
            List of sample dictionaries.
        sample_ids : list of str, optional
            List of sample IDs.
        metadata : list of dict, optional
            List of metadata dictionaries.
        show_progress : bool, optional
            Show progress bar.
            Default: True.
        """
        if sample_ids is None:
            sample_ids = [None] * len(samples)
        if metadata is None:
            metadata = [None] * len(samples)

        iterator = zip(samples, sample_ids, metadata)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(samples),
                desc="Writing samples",
                unit="sample",
            )

        for sample, sample_id, meta in iterator:
            self.write_sample(sample, sample_id, meta)

    def finalize(self) -> Dict[str, Any]:
        """Finalize writing and return summary.

        Returns
        -------
        dict
            Summary statistics including:
            - total_samples: Total samples written
            - total_shards: Total shards created
            - shard_metadata: Detailed metadata for each shard
        """
        # Close current shard
        self._close_current_shard()

        # Create global metadata file
        global_metadata = {
            "total_samples": self.total_samples_written,
            "total_shards": self.current_shard_idx,
            "shard_size": self.shard_size,
            "compression": self.compression,
            "shards": self.shard_metadata,
        }

        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(global_metadata, f, indent=2)

        print(f"Wrote {self.total_samples_written} samples to {self.current_shard_idx} shards")
        print(f"Output directory: {self.output_dir}")
        print(f"Metadata: {metadata_path}")

        return global_metadata


class NWBToWebDatasetConverter:
    """Convert NWB files to WebDataset shards.

    Parameters
    ----------
    output_dir : str or Path
        Directory to write shards.
    shard_size : int, optional
        Number of samples per shard.
        Default: 1000.
    modalities : list of str, optional
        Modalities to extract from NWB.
        Default: ["spikes", "lfp", "behavior"].
    bin_size_ms : float, optional
        Bin size for spike binning in milliseconds.
        Default: 10.0.
    sequence_length : int, optional
        Length of sequences to extract.
        Default: 100.
    overlap : float, optional
        Overlap between sequences (0-1).
        Default: 0.5.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        shard_size: int = 1000,
        modalities: List[str] = ["spikes", "lfp", "behavior"],
        bin_size_ms: float = 10.0,
        sequence_length: int = 100,
        overlap: float = 0.5,
    ):
        self.writer = WebDatasetWriter(output_dir, shard_size)
        self.modalities = modalities
        self.bin_size_ms = bin_size_ms
        self.sequence_length = sequence_length
        self.overlap = overlap

        # Check for NWB availability
        try:
            import pynwb
            from pynwb import NWBHDF5IO
            self.NWBHDF5IO = NWBHDF5IO
            self.nwb_available = True
        except ImportError:
            self.nwb_available = False
            raise ImportError(
                "pynwb is required for NWB conversion. "
                "Install with: pip install pynwb"
            )

    def convert_nwb_file(
        self,
        nwb_file_path: Union[str, Path],
        show_progress: bool = True,
    ):
        """Convert a single NWB file to shards.

        Parameters
        ----------
        nwb_file_path : str or Path
            Path to NWB file.
        show_progress : bool, optional
            Show progress bar.
            Default: True.
        """
        # Load NWB file
        from neuros_neurofm.datasets.nwb_loader import NWBDataset

        dataset = NWBDataset(
            nwb_file_path=str(nwb_file_path),
            bin_size_ms=self.bin_size_ms,
            sequence_length=self.sequence_length,
            overlap=self.overlap,
        )

        # Convert each sample
        samples = []
        for i in range(len(dataset)):
            sample_dict = dataset[i]

            # Convert tensors to numpy for storage
            converted_sample = {}
            for key, value in sample_dict.items():
                if isinstance(value, torch.Tensor):
                    converted_sample[key] = value.numpy()
                else:
                    converted_sample[key] = value

            # Filter to requested modalities if specified
            if self.modalities:
                converted_sample = {
                    k: v for k, v in converted_sample.items()
                    if k in self.modalities or k == "metadata"
                }

            samples.append(converted_sample)

        # Write to shards
        metadata_list = [{
            "source_file": str(nwb_file_path),
            "sequence_idx": i,
        } for i in range(len(samples))]

        self.writer.write_batch(
            samples,
            metadata=metadata_list,
            show_progress=show_progress,
        )

    def convert_nwb_files(
        self,
        nwb_file_paths: List[Union[str, Path]],
        show_progress: bool = True,
    ):
        """Convert multiple NWB files to shards.

        Parameters
        ----------
        nwb_file_paths : list of str or Path
            List of NWB file paths.
        show_progress : bool, optional
            Show progress bar.
            Default: True.
        """
        for nwb_path in tqdm(
            nwb_file_paths,
            desc="Converting NWB files",
            disable=not show_progress,
        ):
            print(f"\nProcessing: {nwb_path}")
            self.convert_nwb_file(nwb_path, show_progress=False)

    def finalize(self) -> Dict[str, Any]:
        """Finalize conversion.

        Returns
        -------
        dict
            Summary statistics.
        """
        return self.writer.finalize()


def create_shards_from_arrays(
    output_dir: Union[str, Path],
    data_dict: Dict[str, np.ndarray],
    shard_size: int = 1000,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create shards from numpy arrays (convenience function).

    Parameters
    ----------
    output_dir : str or Path
        Directory to write shards.
    data_dict : dict
        Dictionary mapping modality names to arrays.
        First dimension should be samples.
    shard_size : int, optional
        Number of samples per shard.
        Default: 1000.
    metadata : dict, optional
        Global metadata to include.

    Returns
    -------
    dict
        Summary statistics.

    Examples
    --------
    >>> data = {
    ...     "spikes": np.random.randn(10000, 100, 96),
    ...     "behavior": np.random.randn(10000, 100, 2),
    ... }
    >>> create_shards_from_arrays("./shards", data)
    """
    writer = WebDatasetWriter(output_dir, shard_size)

    # Get number of samples
    n_samples = len(next(iter(data_dict.values())))

    # Verify all arrays have same first dimension
    for name, arr in data_dict.items():
        if len(arr) != n_samples:
            raise ValueError(
                f"All arrays must have same first dimension. "
                f"Expected {n_samples}, got {len(arr)} for {name}"
            )

    # Write samples
    samples = []
    for i in range(n_samples):
        sample = {name: arr[i] for name, arr in data_dict.items()}
        samples.append(sample)

    metadata_list = [{"global_metadata": metadata} for _ in range(n_samples)]
    writer.write_batch(samples, metadata=metadata_list)

    return writer.finalize()
