#!/usr/bin/env python
"""
CLI tool for converting neural datasets to WebDataset shards.

Supports:
- NWB files (Neurodata Without Borders)
- Numpy arrays (pickled or npz format)
- Custom data formats via plugin system

Usage:
    # Convert single NWB file
    python convert_to_shards.py --input data.nwb --output ./shards --format nwb

    # Convert multiple NWB files
    python convert_to_shards.py --input data/*.nwb --output ./shards --format nwb

    # Convert with parallel processing
    python convert_to_shards.py --input data/*.nwb --output ./shards --workers 8

    # Convert numpy arrays
    python convert_to_shards.py --input data.npz --output ./shards --format npz
"""

import argparse
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


def convert_nwb_to_shards(
    input_paths: List[str],
    output_dir: str,
    shard_size: int = 1000,
    modalities: Optional[List[str]] = None,
    bin_size_ms: float = 10.0,
    sequence_length: int = 100,
    overlap: float = 0.5,
    compression: str = "none",
) -> Dict[str, int]:
    """Convert NWB files to WebDataset shards.

    Parameters
    ----------
    input_paths : list of str
        Paths to NWB files.
    output_dir : str
        Output directory for shards.
    shard_size : int
        Samples per shard.
    modalities : list of str, optional
        Modalities to extract.
    bin_size_ms : float
        Bin size for spike binning.
    sequence_length : int
        Sequence length.
    overlap : float
        Sequence overlap.
    compression : str
        Compression type.

    Returns
    -------
    dict
        Summary statistics.
    """
    from neuros_neurofm.datasets.webdataset_writer import NWBToWebDatasetConverter

    # Create converter
    converter = NWBToWebDatasetConverter(
        output_dir=output_dir,
        shard_size=shard_size,
        modalities=modalities or ["spikes", "lfp", "behavior"],
        bin_size_ms=bin_size_ms,
        sequence_length=sequence_length,
        overlap=overlap,
    )

    # Set compression
    converter.writer.compression = compression

    # Convert files
    print(f"Converting {len(input_paths)} NWB files to shards...")
    converter.convert_nwb_files(input_paths, show_progress=True)

    # Finalize
    return converter.finalize()


def convert_npz_to_shards(
    input_path: str,
    output_dir: str,
    shard_size: int = 1000,
    compression: str = "none",
) -> Dict[str, int]:
    """Convert npz file to WebDataset shards.

    Parameters
    ----------
    input_path : str
        Path to npz file.
    output_dir : str
        Output directory for shards.
    shard_size : int
        Samples per shard.
    compression : str
        Compression type.

    Returns
    -------
    dict
        Summary statistics.
    """
    from neuros_neurofm.datasets.webdataset_writer import create_shards_from_arrays

    # Load npz file
    print(f"Loading {input_path}...")
    data = np.load(input_path)

    # Convert to dict
    data_dict = {key: data[key] for key in data.keys()}

    print(f"Found modalities: {list(data_dict.keys())}")
    print(f"Number of samples: {len(next(iter(data_dict.values())))}")

    # Create shards
    return create_shards_from_arrays(
        output_dir=output_dir,
        data_dict=data_dict,
        shard_size=shard_size,
    )


def convert_arrays_to_shards(
    input_paths: Dict[str, str],
    output_dir: str,
    shard_size: int = 1000,
    compression: str = "none",
) -> Dict[str, int]:
    """Convert multiple numpy arrays to WebDataset shards.

    Parameters
    ----------
    input_paths : dict
        Dictionary mapping modality names to file paths.
    output_dir : str
        Output directory for shards.
    shard_size : int
        Samples per shard.
    compression : str
        Compression type.

    Returns
    -------
    dict
        Summary statistics.
    """
    from neuros_neurofm.datasets.webdataset_writer import create_shards_from_arrays

    # Load all arrays
    data_dict = {}
    for modality, path in input_paths.items():
        print(f"Loading {modality} from {path}...")
        if path.endswith(".npy"):
            data_dict[modality] = np.load(path)
        elif path.endswith(".npz"):
            data = np.load(path)
            # If npz has single array, use it; otherwise use all
            if len(data.keys()) == 1:
                data_dict[modality] = data[list(data.keys())[0]]
            else:
                for key in data.keys():
                    data_dict[f"{modality}_{key}"] = data[key]

    print(f"\nTotal modalities: {list(data_dict.keys())}")
    print(f"Number of samples: {len(next(iter(data_dict.values())))}")

    # Create shards
    return create_shards_from_arrays(
        output_dir=output_dir,
        data_dict=data_dict,
        shard_size=shard_size,
    )


def validate_shards(shard_dir: str, n_samples: int = 5):
    """Validate created shards by loading samples.

    Parameters
    ----------
    shard_dir : str
        Directory containing shards.
    n_samples : int
        Number of samples to load for validation.
    """
    from neuros_neurofm.datasets.webdataset_loader import (
        ShardedDatasetInfo,
        WebDatasetLoader,
    )

    print("\n" + "=" * 60)
    print("Validating shards...")
    print("=" * 60)

    # Print summary
    info = ShardedDatasetInfo(shard_dir)
    info.print_summary()

    # Load a few samples
    print(f"\nLoading {n_samples} samples for validation...")
    loader = WebDatasetLoader(shard_dir, shuffle=False)

    samples_loaded = 0
    for i, sample in enumerate(loader):
        if i >= n_samples:
            break

        print(f"\nSample {i}:")
        for modality, data in sample.items():
            if modality == "metadata":
                print(f"  - {modality}: {data}")
            elif hasattr(data, "shape"):
                print(f"  - {modality}: shape={data.shape}, dtype={data.dtype}")
            else:
                print(f"  - {modality}: {type(data)}")

        samples_loaded += 1

    print(f"\nSuccessfully loaded {samples_loaded} samples!")
    print("Validation complete.")


def parallel_convert_worker(args):
    """Worker function for parallel conversion."""
    input_path, output_dir, config = args

    try:
        if config["format"] == "nwb":
            from neuros_neurofm.datasets.webdataset_writer import NWBToWebDatasetConverter

            converter = NWBToWebDatasetConverter(
                output_dir=output_dir,
                shard_size=config["shard_size"],
                modalities=config.get("modalities"),
                bin_size_ms=config.get("bin_size_ms", 10.0),
                sequence_length=config.get("sequence_length", 100),
                overlap=config.get("overlap", 0.5),
            )
            converter.convert_nwb_file(input_path, show_progress=False)
            return converter.finalize()
        else:
            return {"error": f"Unsupported format for parallel processing: {config['format']}"}
    except Exception as e:
        return {"error": str(e), "file": input_path}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert neural datasets to WebDataset shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/output
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        nargs="+",
        help="Input file(s) or glob pattern (e.g., data/*.nwb)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for shards",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["nwb", "npz", "npy", "auto"],
        default="auto",
        help="Input format (auto-detect by default)",
    )

    # Shard configuration
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of samples per shard (default: 1000)",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "gz", "bz2", "xz"],
        default="none",
        help="Compression type (default: none)",
    )

    # NWB-specific options
    parser.add_argument(
        "--modalities",
        nargs="+",
        help="Modalities to extract from NWB (default: spikes, lfp, behavior)",
    )
    parser.add_argument(
        "--bin-size-ms",
        type=float,
        default=10.0,
        help="Bin size for spike binning in ms (default: 10.0)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Sequence length (default: 100)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sequence overlap 0-1 (default: 0.5)",
    )

    # Processing options
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate shards after conversion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )

    args = parser.parse_args()

    # Check output directory
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Error: Output directory {output_path} already exists.")
        print("Use --overwrite to overwrite.")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # Expand glob patterns
    input_paths = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_paths.extend(expanded)
        else:
            # If no glob match, treat as literal path
            input_paths.append(pattern)

    if not input_paths:
        print("Error: No input files found.")
        sys.exit(1)

    # Auto-detect format if needed
    if args.format == "auto":
        ext = Path(input_paths[0]).suffix.lower()
        if ext == ".nwb":
            args.format = "nwb"
        elif ext == ".npz":
            args.format = "npz"
        elif ext == ".npy":
            args.format = "npy"
        else:
            print(f"Error: Cannot auto-detect format for extension: {ext}")
            sys.exit(1)
        print(f"Auto-detected format: {args.format}")

    # Convert based on format
    try:
        if args.format == "nwb":
            if args.workers > 1:
                print(f"Using {args.workers} parallel workers...")
                config = {
                    "format": "nwb",
                    "shard_size": args.shard_size,
                    "modalities": args.modalities,
                    "bin_size_ms": args.bin_size_ms,
                    "sequence_length": args.sequence_length,
                    "overlap": args.overlap,
                }
                worker_args = [(path, args.output, config) for path in input_paths]

                with Pool(args.workers) as pool:
                    results = pool.map(parallel_convert_worker, worker_args)

                # Check for errors
                errors = [r for r in results if "error" in r]
                if errors:
                    print("\nErrors occurred:")
                    for error in errors:
                        print(f"  - {error.get('file', 'Unknown')}: {error['error']}")

                summary = results[-1] if results else {}
            else:
                summary = convert_nwb_to_shards(
                    input_paths=input_paths,
                    output_dir=args.output,
                    shard_size=args.shard_size,
                    modalities=args.modalities,
                    bin_size_ms=args.bin_size_ms,
                    sequence_length=args.sequence_length,
                    overlap=args.overlap,
                    compression=args.compression,
                )

        elif args.format == "npz":
            if len(input_paths) > 1:
                print("Warning: Multiple npz files specified, using first one only.")
            summary = convert_npz_to_shards(
                input_path=input_paths[0],
                output_dir=args.output,
                shard_size=args.shard_size,
                compression=args.compression,
            )

        elif args.format == "npy":
            # For npy, expect modality names as prefixes or use generic names
            modality_paths = {}
            for i, path in enumerate(input_paths):
                modality_name = Path(path).stem
                modality_paths[modality_name] = path

            summary = convert_arrays_to_shards(
                input_paths=modality_paths,
                output_dir=args.output,
                shard_size=args.shard_size,
                compression=args.compression,
            )

        # Validate if requested
        if args.validate:
            validate_shards(args.output)

        print("\nConversion complete!")

    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
