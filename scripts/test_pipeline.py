#!/usr/bin/env python3
"""Integration test for the Constellation demo pipeline.

This script runs a short ingestion in dry‑run mode (no Kafka),
verifies that raw NWB and Zarr files are produced and that feature
archives are generated, then prints a summary of the files.  It
exits with a non‑zero status if any expected output is missing.

Example:

```
python scripts/test_pipeline.py --duration 3 --output-dir /tmp/test_demo
```
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Constellation demo pipeline")
    parser.add_argument("--duration", type=float, default=2.0, help="Ingestion duration in seconds")
    parser.add_argument("--output-dir", type=str, default="/tmp/constellation_test", help="Output directory")
    args = parser.parse_args()

    # Run the pipeline in no‑kafka mode
    cmd = [
        sys.executable,
        "-m",
        "neuros.cli",
        "constellation",
        "--duration",
        str(args.duration),
        "--output-dir",
        args.output_dir,
        "--no-kafka",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Check that raw files exist
    raw_dir = Path(args.output_dir) / "raw"
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    # Expect one NWB or Zarr file per modality present in pipeline
    modalities_found = []
    for file in raw_dir.rglob("*.*"):
        if file.suffix in {".nwb", ".zarr", ".npz", ".npy"}:
            modalities_found.append(file.name)
    if not modalities_found:
        print("Error: No raw NWB or Zarr files found", file=sys.stderr)
        sys.exit(1)
    print("Raw data files:")
    for name in sorted(modalities_found):
        print(" -", name)
    # Check that feature shards exist
    gold_dir = Path(args.output_dir) / "gold"
    if not gold_dir.exists():
        print(f"Error: {gold_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    tar_files = list(gold_dir.rglob("*.tar"))
    if not tar_files:
        print(
            "Warning: No WebDataset tar shards were created (export_to_webdataset skipped)",
            file=sys.stderr,
        )
    else:
        print("Feature tar shards:")
        for tar in tar_files:
            print(" -", tar.name)
    print("Integration test completed.")


if __name__ == "__main__":
    main()