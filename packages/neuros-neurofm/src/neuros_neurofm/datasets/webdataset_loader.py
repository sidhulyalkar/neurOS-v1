"""
WebDataset loader for NeuroFMX with resumable iteration and multi-worker support.

Provides efficient streaming data loading from WebDataset tar shards with:
- Lazy loading from disk
- Resumable iteration with cursor tracking
- Multi-worker DataLoader compatibility
- Shuffling within and across shards
- Prefetching for performance
"""

import io
import json
import pickle
import random
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset


class WebDatasetLoader(IterableDataset):
    """Iterable dataset for loading WebDataset shards.

    Parameters
    ----------
    shard_dir : str or Path
        Directory containing shard files.
    shard_pattern : str, optional
        Glob pattern for shard files.
        Default: "shard_*.tar".
    shuffle : bool, optional
        Shuffle samples within shards.
        Default: True.
    shuffle_shards : bool, optional
        Shuffle order of shards.
        Default: True.
    buffer_size : int, optional
        Size of shuffle buffer (number of samples to buffer for shuffling).
        Default: 1000.
    max_workers : int, optional
        Maximum number of workers (for multi-worker DataLoader).
        If None, determined automatically.
        Default: None.
    seed : int, optional
        Random seed for reproducibility.
        Default: 42.
    decode : bool, optional
        Automatically decode pickled tensors.
        Default: True.
    """

    def __init__(
        self,
        shard_dir: Union[str, Path],
        shard_pattern: str = "shard_*.tar",
        shuffle: bool = True,
        shuffle_shards: bool = True,
        buffer_size: int = 1000,
        max_workers: Optional[int] = None,
        seed: int = 42,
        decode: bool = True,
    ):
        self.shard_dir = Path(shard_dir)
        self.shard_pattern = shard_pattern
        self.shuffle = shuffle
        self.shuffle_shards = shuffle_shards
        self.buffer_size = buffer_size
        self.max_workers = max_workers
        self.seed = seed
        self.decode = decode

        # Find all shards
        self.shard_paths = sorted(self.shard_dir.glob(shard_pattern))
        if len(self.shard_paths) == 0:
            raise ValueError(f"No shards found in {shard_dir} matching {shard_pattern}")

        # Load metadata if available
        self.metadata = self._load_metadata()

        # State for resumption
        self.current_shard_idx = 0
        self.current_sample_offset = 0

        # Worker info (set during iteration)
        self.worker_id = 0
        self.num_workers = 1

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load dataset metadata if available."""
        metadata_path = self.shard_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def get_state(self) -> Dict[str, int]:
        """Get current iteration state for resumption.

        Returns
        -------
        dict
            State dictionary with:
            - shard_idx: Current shard index
            - sample_offset: Current sample offset within shard
        """
        return {
            "shard_idx": self.current_shard_idx,
            "sample_offset": self.current_sample_offset,
        }

    def set_state(self, state: Dict[str, int]):
        """Set iteration state for resumption.

        Parameters
        ----------
        state : dict
            State dictionary from get_state().
        """
        self.current_shard_idx = state.get("shard_idx", 0)
        self.current_sample_offset = state.get("sample_offset", 0)

    def _get_worker_info(self):
        """Get worker information for multi-worker DataLoader."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        else:
            self.worker_id = 0
            self.num_workers = 1

    def _assign_shards_to_worker(self) -> List[Path]:
        """Assign shards to current worker."""
        # Divide shards among workers
        worker_shards = []
        for i, shard_path in enumerate(self.shard_paths):
            if i % self.num_workers == self.worker_id:
                worker_shards.append(shard_path)
        return worker_shards

    def _load_shard(self, shard_path: Path) -> List[Dict[str, Any]]:
        """Load all samples from a shard.

        Parameters
        ----------
        shard_path : Path
            Path to shard tar file.

        Returns
        -------
        list of dict
            List of samples from shard.
        """
        samples = {}  # sample_id -> sample_dict

        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                # Parse filename: sample_id.modality.extension
                parts = member.name.split(".")
                if len(parts) < 3:
                    continue

                sample_id = parts[0]
                modality = ".".join(parts[1:-1])
                extension = parts[-1]

                # Read data
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    continue

                data_bytes = file_obj.read()

                # Decode based on extension
                if extension == "pyd":
                    if self.decode:
                        data = pickle.loads(data_bytes)
                    else:
                        data = data_bytes
                elif extension == "json":
                    data = json.loads(data_bytes.decode("utf-8"))
                else:
                    data = data_bytes

                # Add to sample
                if sample_id not in samples:
                    samples[sample_id] = {}
                samples[sample_id][modality] = data

        # Convert to list and sort by sample_id for consistency
        sample_list = [samples[sid] for sid in sorted(samples.keys())]
        return sample_list

    def _shuffle_buffer(self, iterator: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Apply shuffle buffer to iterator.

        Parameters
        ----------
        iterator : iterator
            Input iterator.

        Yields
        ------
        dict
            Shuffled samples.
        """
        buffer = []
        rng = random.Random(self.seed + self.worker_id)

        for item in iterator:
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                # Shuffle and yield random item
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)

        # Yield remaining items in random order
        rng.shuffle(buffer)
        for item in buffer:
            yield item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples.

        Yields
        ------
        dict
            Sample dictionary with modalities as keys.
        """
        # Get worker info
        self._get_worker_info()

        # Get shards for this worker
        worker_shards = self._assign_shards_to_worker()

        # Optionally shuffle shard order
        if self.shuffle_shards:
            rng = random.Random(self.seed + self.worker_id)
            rng.shuffle(worker_shards)

        # Create iterator over all samples
        def sample_iterator():
            for shard_idx, shard_path in enumerate(worker_shards):
                # Skip to resume point if needed
                if shard_idx < self.current_shard_idx:
                    continue

                # Load shard
                samples = self._load_shard(shard_path)

                # Skip samples if resuming within shard
                start_idx = self.current_sample_offset if shard_idx == self.current_shard_idx else 0

                for sample_idx, sample in enumerate(samples[start_idx:], start=start_idx):
                    # Update state
                    self.current_shard_idx = shard_idx
                    self.current_sample_offset = sample_idx + 1

                    yield sample

                # Reset offset for next shard
                self.current_sample_offset = 0

        # Apply shuffling if requested
        iterator = sample_iterator()
        if self.shuffle:
            iterator = self._shuffle_buffer(iterator)

        return iterator

    def __len__(self) -> int:
        """Get total number of samples (approximate for multi-worker).

        Returns
        -------
        int
            Number of samples.
        """
        if self.metadata is not None:
            total = self.metadata.get("total_samples", 0)
            # Adjust for current worker
            return total // self.num_workers + (1 if self.worker_id < total % self.num_workers else 0)
        else:
            # Estimate from number of shards
            return len(self.shard_paths) * 1000  # Assume 1000 samples per shard


class ResumableWebDatasetLoader(WebDatasetLoader):
    """WebDataset loader with built-in checkpoint/resume support.

    Automatically saves and loads iteration state to enable resuming from
    interruptions during training.

    Parameters
    ----------
    shard_dir : str or Path
        Directory containing shard files.
    checkpoint_path : str or Path, optional
        Path to checkpoint file for saving/loading state.
        If None, checkpointing is disabled.
        Default: None.
    checkpoint_interval : int, optional
        Save checkpoint every N samples.
        Default: 1000.
    **kwargs
        Additional arguments for WebDatasetLoader.
    """

    def __init__(
        self,
        shard_dir: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 1000,
        **kwargs,
    ):
        super().__init__(shard_dir, **kwargs)

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_interval = checkpoint_interval
        self.samples_since_checkpoint = 0

        # Try to load checkpoint
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.load_checkpoint()

    def save_checkpoint(self):
        """Save current state to checkpoint file."""
        if self.checkpoint_path is None:
            return

        state = self.get_state()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.checkpoint_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self):
        """Load state from checkpoint file."""
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return

        with open(self.checkpoint_path, "r") as f:
            state = json.load(f)

        self.set_state(state)
        print(f"Resumed from checkpoint: shard {state['shard_idx']}, "
              f"offset {state['sample_offset']}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate with automatic checkpointing.

        Yields
        ------
        dict
            Sample dictionary.
        """
        for sample in super().__iter__():
            yield sample

            # Periodic checkpointing
            self.samples_since_checkpoint += 1
            if self.samples_since_checkpoint >= self.checkpoint_interval:
                self.save_checkpoint()
                self.samples_since_checkpoint = 0


def collate_webdataset(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for WebDataset samples.

    Parameters
    ----------
    batch : list of dict
        List of samples from WebDatasetLoader.

    Returns
    -------
    dict
        Batched tensors.
    """
    # Get all modalities
    modalities = set()
    for sample in batch:
        modalities.update(sample.keys())

    # Remove metadata from modalities to collate
    modalities.discard("metadata")

    # Collate each modality
    collated = {}
    for modality in modalities:
        # Extract data for this modality
        data_list = []
        for sample in batch:
            if modality in sample:
                data = sample[modality]
                # Convert to tensor if needed
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                elif not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                data_list.append(data)

        if data_list:
            # Stack if all same shape, otherwise return as list
            try:
                collated[modality] = torch.stack(data_list)
            except RuntimeError:
                # Different shapes - return as list
                collated[modality] = data_list

    # Collect metadata
    metadata_list = [sample.get("metadata", {}) for sample in batch]
    if any(metadata_list):
        collated["metadata"] = metadata_list

    return collated


def create_webdataset_dataloader(
    shard_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    buffer_size: int = 1000,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for WebDataset shards.

    Parameters
    ----------
    shard_dir : str or Path
        Directory containing shard files.
    batch_size : int, optional
        Batch size.
        Default: 32.
    num_workers : int, optional
        Number of data loading workers.
        Default: 4.
    shuffle : bool, optional
        Shuffle samples.
        Default: True.
    buffer_size : int, optional
        Shuffle buffer size.
        Default: 1000.
    prefetch_factor : int, optional
        Number of batches to prefetch per worker.
        Default: 2.
    pin_memory : bool, optional
        Pin memory for faster GPU transfer.
        Default: True.
    **dataset_kwargs
        Additional arguments for WebDatasetLoader.

    Returns
    -------
    DataLoader
        Configured DataLoader.
    """
    dataset = WebDatasetLoader(
        shard_dir=shard_dir,
        shuffle=shuffle,
        buffer_size=buffer_size,
        **dataset_kwargs,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_webdataset,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return loader


class ShardedDatasetInfo:
    """Utility class to inspect WebDataset shards.

    Parameters
    ----------
    shard_dir : str or Path
        Directory containing shard files.
    """

    def __init__(self, shard_dir: Union[str, Path]):
        self.shard_dir = Path(shard_dir)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load dataset metadata."""
        metadata_path = self.shard_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get dataset summary.

        Returns
        -------
        dict
            Summary including total samples, shards, modalities, etc.
        """
        if self.metadata is None:
            return {"error": "No metadata found"}

        summary = {
            "total_samples": self.metadata.get("total_samples", 0),
            "total_shards": self.metadata.get("total_shards", 0),
            "shard_size": self.metadata.get("shard_size", 0),
            "compression": self.metadata.get("compression", "none"),
        }

        # Get modalities from first sample
        if self.metadata.get("shards") and len(self.metadata["shards"]) > 0:
            first_shard = self.metadata["shards"][0]
            if first_shard.get("samples") and len(first_shard["samples"]) > 0:
                first_sample = first_shard["samples"][0]
                summary["modalities"] = first_sample.get("modalities", [])

        return summary

    def print_summary(self):
        """Print dataset summary."""
        summary = self.get_summary()

        print("=" * 60)
        print("WebDataset Summary")
        print("=" * 60)
        print(f"Directory: {self.shard_dir}")
        print(f"Total Samples: {summary.get('total_samples', 'N/A')}")
        print(f"Total Shards: {summary.get('total_shards', 'N/A')}")
        print(f"Shard Size: {summary.get('shard_size', 'N/A')}")
        print(f"Compression: {summary.get('compression', 'N/A')}")
        if "modalities" in summary:
            print(f"Modalities: {', '.join(summary['modalities'])}")
        print("=" * 60)
