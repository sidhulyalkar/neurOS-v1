"""Export tokenized sequences for neuroFMx integration."""

import json
from pathlib import Path
from typing import Any
import numpy as np

from neuros_astro.metadata.schema import TokenizedAstroSequence


def save_tokenized_sequence_npz(sequence: TokenizedAstroSequence, path: str | Path) -> None:
    """
    Save tokenized astrocyte sequence to NPZ format.

    Args:
        sequence: TokenizedAstroSequence object
        path: Output file path
    """
    path = Path(path)

    # Convert to numpy arrays
    tokens = np.array(sequence.tokens, dtype=np.float32)
    timestamps_s = np.array(sequence.timestamps_s, dtype=np.float32)
    feature_names = np.array(sequence.feature_names, dtype=object)

    # Convert region_ids (handle None values)
    region_ids_arr = np.array(
        [rid if rid is not None else "" for rid in sequence.region_ids], dtype=object
    )

    # Serialize metadata to JSON string
    metadata_json = json.dumps(sequence.metadata)

    # Save to NPZ
    np.savez(
        path,
        tokens=tokens,
        timestamps_s=timestamps_s,
        feature_names=feature_names,
        region_ids=region_ids_arr,
        session_id=sequence.session_id,
        metadata_json=metadata_json,
    )


def load_tokenized_sequence_npz(path: str | Path) -> TokenizedAstroSequence:
    """
    Load tokenized astrocyte sequence from NPZ format.

    Args:
        path: Input file path

    Returns:
        TokenizedAstroSequence object
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Load NPZ
    data = np.load(path, allow_pickle=True)

    tokens = data["tokens"].tolist()
    timestamps_s = data["timestamps_s"].tolist()
    feature_names = data["feature_names"].tolist()
    region_ids_arr = data["region_ids"]
    session_id = str(data["session_id"])

    # Convert region_ids back (empty string -> None)
    region_ids = [str(rid) if rid != "" else None for rid in region_ids_arr]

    # Deserialize metadata
    metadata_json = str(data["metadata_json"])
    metadata = json.loads(metadata_json)

    return TokenizedAstroSequence(
        session_id=session_id,
        tokens=tokens,
        timestamps_s=timestamps_s,
        region_ids=region_ids,
        feature_names=feature_names,
        metadata=metadata,
    )


def build_neurofm_manifest(
    session_id: str,
    modalities: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a neuroFMx-compatible manifest for multimodal data.

    Args:
        session_id: Session identifier
        modalities: Dictionary of modality configurations
            Example:
            {
                "astro": {
                    "type": "event_tokens",
                    "path": "astro_tokens.npz",
                    "sampling": "irregular",
                    "timestamp_key": "timestamps_s"
                }
            }
        metadata: Additional metadata

    Returns:
        Manifest dictionary
    """
    manifest = {
        "session_id": session_id,
        "modalities": modalities,
        "metadata": metadata or {},
    }

    return manifest


def save_neurofm_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """
    Save neuroFMx manifest to JSON file.

    Args:
        manifest: Manifest dictionary
        path: Output file path
    """
    path = Path(path)

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_neurofm_manifest(path: str | Path) -> dict[str, Any]:
    """
    Load neuroFMx manifest from JSON file.

    Args:
        path: Input file path

    Returns:
        Manifest dictionary
    """
    path = Path(path)

    with open(path, "r") as f:
        manifest = json.load(f)

    return manifest
