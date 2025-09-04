"""
Petastorm exporter for neurOS.

Petastorm provides utilities for creating Parquet datasets with a
defined schema and reading them efficiently on distributed training
clusters.  This module includes a simple exporter that scans
pre‑aligned sample directories (one directory per sample) and
materialises a Parquet dataset representing sequences of features or
time windows.

For production use you should define a custom ``Unischema`` that
matches your model's input tensor layout and use
``petastorm.etl.DatasetProcessor`` to transform raw files into the
Parquet rows.  Here we instead synthesise a minimal dataset for
illustration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from petastorm.codecs import ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.etl.rowgroup_indexing import build_rowgroup_index
from petastorm.unischema import Unischema, UnischemaField
from petastorm import make_batch_reader
from pyarrow import Table, parquet as pq

logger = logging.getLogger(__name__)


def define_unischema(feature_dim: int) -> Unischema:
    """Define a simple Unischema with a single vector field."""
    return Unischema(
        "BrainFeature",
        [
            UnischemaField("id", np.int32, (), ScalarCodec(np.int32), False),
            UnischemaField("features", np.float32, (feature_dim,), ScalarCodec(np.float32), False),
        ],
    )


def export_to_petastorm(input_uri: str, output_uri: str, feature_dim: int = 128) -> None:
    """Export sample directories into a Petastorm Parquet dataset.

    Parameters
    ----------
    input_uri:
        Path containing sample subdirectories.  Files within each sample
        directory are ignored in this simple exporter; instead we
        synthesise a random feature vector for each sample.  In
        production use you should parse the files and generate
        meaningful features.
    output_uri:
        Directory where the Parquet dataset will be written.  Should
        either be a local path or an S3 URI with proper credentials.
    feature_dim:
        Length of the feature vector assigned to each sample.
    """
    input_path = Path(input_uri)
    samples = [p for p in input_path.iterdir() if p.is_dir()]
    logger.info("Found %d samples for Petastorm export", len(samples))
    if not samples:
        logger.warning("No samples found at %s", input_uri)
        return
    schema = define_unischema(feature_dim)

    def row_generator() -> Iterable[Dict[str, np.ndarray]]:
        for idx, sample in enumerate(samples):
            yield {
                "id": np.int32(idx),
                "features": np.random.rand(feature_dim).astype(np.float32),
            }

    # Materialise dataset using Petastorm's helper
    with materialize_dataset(
        spark=None,  # Use in-memory materialisation; suitable for small datasets
        dataset_url=output_uri,
        schema=schema,
        rowgroup_size_mb=64,
    ) as dataset_writer:
        for row in row_generator():
            dataset_writer.write_row(row)
    logger.info("Finished writing Petastorm dataset to %s", output_uri)


__all__ = ["export_to_petastorm", "define_unischema"]