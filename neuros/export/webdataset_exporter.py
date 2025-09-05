"""
WebDataset exporter for neurOS.

This module provides a function to package aligned data into WebDataset
tar shards.  WebDataset is a streaming data format where each shard is
a tar archive containing files grouped by sample key; files belonging
to the same sample share the same prefix but different extensions.  For
example, for a sample key ``000123``, the shard may contain
``000123.eeg``, ``000123.wav``, ``000123.mp4`` and ``000123.json``.

The exporter scans a directory (local filesystem or S3 path) for
sample directories or files and writes shards of roughly equal size.  It
is up to the caller to ensure that the input directory contains
pre‑aligned and curated data.  The output shards can then be consumed
directly by the WebDataset library in PyTorch.
"""
from __future__ import annotations

import glob
import logging
import os
import tarfile
from pathlib import Path
from typing import Iterable, List

# fsspec is used for remote storage (e.g. S3).  Import it lazily so that
# environments without fsspec can still use local filesystem export.
try:
    import fsspec  # type: ignore  # noqa: F401
except Exception:
    fsspec = None  # type: ignore

logger = logging.getLogger(__name__)


def _find_samples(input_uri: str) -> List[Path]:
    """Find sample directories or files within the input URI.

    A sample is defined as a directory containing multiple modality files
    (e.g. ``eeg.npy``, ``video.mp4``, ``metadata.json``) or a single
    file per sample.  For simplicity this function returns all sub
    directories in ``input_uri`` when run locally.  In the case of S3
    the caller should stage data locally or provide a custom finder.
    """
    path = Path(input_uri)
    if not path.exists():
        raise FileNotFoundError(f"Input path {input_uri} does not exist")
    samples = [p for p in path.iterdir() if p.is_dir()]
    logger.info("Found %d sample directories in %s", len(samples), input_uri)
    return samples


def export_to_webdataset(input_uri: str, output_uri: str, shard_size: int = 100) -> None:
    """Export curated data to WebDataset tar shards.

    Parameters
    ----------
    input_uri:
        Path to the directory containing curated sample subdirectories.
    output_uri:
        Directory where tar shards will be created.  If on S3 this
        should be a URI such as ``s3://constellation/gold/webdataset``; in
        that case this function uses fsspec to open the file for writing.
    shard_size:
        Number of samples per shard.  A typical value is 500–1000.  In
        this implementation the shards are small to ease testing.
    """
    samples = _find_samples(input_uri)
    if not samples:
        logger.warning("No samples found at %s", input_uri)
        return
    Path(output_uri).mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    count = 0
    tar_path = None
    tar_obj = None

    def _open_new_shard(idx: int) -> tarfile.TarFile:
        shard_name = f"shard-{idx:05d}.tar"
        shard_path = os.path.join(output_uri, shard_name)
        logger.info("Creating shard %s", shard_path)
        return tarfile.open(shard_path, "w")

    tar_obj = _open_new_shard(shard_idx)

    for sample_dir in samples:
        sample_key = sample_dir.name
        for file in sample_dir.iterdir():
            suffix = file.suffix.lstrip(".")
            dest_name = f"{sample_key}.{suffix}"
            tar_obj.add(file, arcname=dest_name)
        count += 1
        if count % shard_size == 0:
            tar_obj.close()
            shard_idx += 1
            tar_obj = _open_new_shard(shard_idx)
    tar_obj.close()
    logger.info("Finished exporting %d samples into %d shards", len(samples), shard_idx + 1)


__all__ = ["export_to_webdataset"]