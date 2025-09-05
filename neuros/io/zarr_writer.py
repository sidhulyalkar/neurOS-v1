"""
OME‑Zarr writer for imaging and video data.

This module wraps the ``ome_zarr`` and ``zarr`` libraries to write
multi‑dimensional arrays (e.g. video frames or volumetric imaging data)
into the Next Generation File Format (NGFF) for microscopy.  OME‑Zarr
files are composed of chunked, compressed arrays stored within a
directory hierarchy on disk or object storage, making them ideal for
cloud‑native workflows.

The ``write_ome_zarr`` function accepts a 5D numpy array (t, c, z, y, x)
and writes it to a specified directory with user‑defined chunk sizes
and compressor.  Additional metadata can be attached via the
``metadata`` argument.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Import ome_zarr and zarr libraries lazily.  If these packages are
# missing, set sentinel variables so that ``write_ome_zarr`` can
# gracefully raise an ImportError when called.  This avoids errors
# during module import when optional dependencies are absent.
try:
    import zarr  # type: ignore
    from numcodecs import Blosc  # type: ignore
    from ome_zarr.io import parse_url  # type: ignore
    from ome_zarr.writer import write_image  # type: ignore
    _OME_ZARR_AVAILABLE = True
except Exception as _exc:
    zarr = None  # type: ignore
    Blosc = None  # type: ignore
    parse_url = None  # type: ignore
    write_image = None  # type: ignore
    _OME_ZARR_AVAILABLE = False
    _OME_ZARR_IMPORT_ERROR = _exc

logger = logging.getLogger(__name__)


def write_ome_zarr(
    array: np.ndarray,
    store_path: str | Path,
    chunk_shape: Tuple[int, int, int, int, int] | None = None,
    compressor: Blosc | None = None,
    metadata: Dict[str, str] | None = None,
) -> None:
    """Write a multi‑dimensional array to OME‑Zarr.

    Parameters
    ----------
    array:
        A numpy array of shape (t, c, z, y, x) representing time,
        channel, depth, height and width.
    store_path:
        Path or URI pointing to the directory where the Zarr hierarchy
        should be written.  This can be a local filesystem path or an S3
        URI supported by fsspec.
    chunk_shape:
        Optional chunk shape to use.  If None a reasonable default
        derived from the array shape will be used.
    compressor:
        Compressor instance from ``numcodecs``, e.g. ``Blosc(cname='zstd',
        clevel=5, shuffle=Blosc.BITSHUFFLE)``.  If None, Zarr's default
        compressor is used.
    metadata:
        Additional key/value metadata to attach to the root of the
        Zarr store.
    """
    if array.ndim != 5:
        raise ValueError("Input array must have 5 dimensions (t, c, z, y, x)")

    if not _OME_ZARR_AVAILABLE or zarr is None or Blosc is None:
        raise ImportError(
            "ome_zarr and its dependencies are required to write Zarr files; install the 'ome-zarr' and 'numcodecs' packages"
        )
    path_obj = Path(store_path)
    logger.info("Writing OME‑Zarr to %s with shape %s", store_path, array.shape)

    if chunk_shape is None:
        # Default chunk: 1 timepoint, all channels, 1 z slice, 512x512
        t, c, z, y, x = array.shape
        chunk_shape = (1, c, 1, min(y, 512), min(x, 512))

    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    store = parse_url(str(store_path), mode="w").store
    # Remove any existing data at destination
    if isinstance(store, zarr.storage.DirectoryStore) and path_obj.exists():
        import shutil

        shutil.rmtree(path_obj)

    write_image(
        array,
        group=store,
        channel_names=[f"c{i}" for i in range(array.shape[1])],
        chunk_sizes=[chunk_shape],
        axes=["t", "c", "z", "y", "x"],
        compressor=compressor,
        metadata=metadata or {},
    )
    logger.info("Finished writing OME‑Zarr to %s", store_path)


__all__ = ["write_ome_zarr"]