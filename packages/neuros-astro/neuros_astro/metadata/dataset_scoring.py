"""Dataset triage and scoring for astrocyte reanalysis potential."""

import json
from pathlib import Path
from typing import Any

from neuros_astro.metadata.schema import DatasetTriageResult
from neuros_astro.metadata.controlled_terms import (
    ASTRO_TERMS,
    CALCIUM_TERMS,
    MODALITY_TERMS,
    BEHAVIOR_TERMS,
    EPHYS_TERMS,
    search_terms_in_text,
)


def score_dataset_metadata(
    metadata: dict[str, Any] | str,
    session_id: str | None = None,
) -> DatasetTriageResult:
    """
    Score a dataset for astrocyte reanalysis potential using metadata.

    Scoring algorithm (per whitepaper):
    - Start at 0.0
    - +0.25 if astrocyte or glial terms found
    - +0.15 if calcium indicator terms found
    - +0.15 if optical imaging modality terms found
    - +0.20 if raw movie appears available
    - +0.10 if masks or ROIs appear available
    - +0.10 if behavior or stimulus timing appears available
    - +0.05 if electrophysiology, LFP, or spikes appear available
    - Clamp final score to [0, 1]

    Args:
        metadata: Dictionary of metadata or plain text string
        session_id: Optional session identifier

    Returns:
        DatasetTriageResult with score and recommendations
    """
    # Convert metadata to searchable text
    if isinstance(metadata, str):
        searchable_text = metadata
        meta_dict = {}
    elif isinstance(metadata, dict):
        searchable_text = json.dumps(metadata, default=str).lower()
        meta_dict = metadata
    else:
        searchable_text = str(metadata)
        meta_dict = {}

    # Extract session_id if not provided
    if session_id is None:
        session_id = meta_dict.get("session_id", "unknown")

    # Search for controlled terms
    matched_astro = search_terms_in_text(searchable_text, ASTRO_TERMS)
    matched_calcium = search_terms_in_text(searchable_text, CALCIUM_TERMS)
    matched_modality = search_terms_in_text(searchable_text, MODALITY_TERMS)
    matched_behavior = search_terms_in_text(searchable_text, BEHAVIOR_TERMS)
    matched_ephys = search_terms_in_text(searchable_text, EPHYS_TERMS)

    # Check for data availability
    has_raw_movie = _check_raw_movie_availability(searchable_text, meta_dict)
    has_masks = _check_masks_availability(searchable_text, meta_dict)
    has_behavior = len(matched_behavior) > 0 or _check_behavior_availability(meta_dict)
    has_ephys = len(matched_ephys) > 0 or _check_ephys_availability(meta_dict)

    # Calculate score
    score = 0.0

    if matched_astro:
        score += 0.25

    if matched_calcium:
        score += 0.15

    if matched_modality:
        score += 0.15

    if has_raw_movie:
        score += 0.20

    if has_masks:
        score += 0.10

    if has_behavior:
        score += 0.10

    if has_ephys:
        score += 0.05

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    # Determine recommended next step
    recommended_next_step = _determine_next_step(
        score=score,
        has_raw_movie=has_raw_movie,
        has_masks=has_masks,
    )

    # Generate warnings
    warnings = []
    if score < 0.40:
        warnings.append("Low astrocyte reanalysis score - dataset may not be suitable")
    if not matched_astro:
        warnings.append("No explicit astrocyte terms found - astrocyte identity uncertain")
    if not matched_calcium and not matched_modality:
        warnings.append("No optical imaging indicators found")

    return DatasetTriageResult(
        session_id=session_id,
        astro_reanalysis_score=score,
        has_raw_movie=has_raw_movie,
        has_masks=has_masks,
        has_behavior=has_behavior,
        has_ephys=has_ephys,
        matched_astro_terms=matched_astro,
        matched_calcium_terms=matched_calcium,
        matched_modality_terms=matched_modality,
        warnings=warnings,
        recommended_next_step=recommended_next_step,
        metadata=meta_dict,
    )


def _check_raw_movie_availability(text: str, metadata: dict[str, Any]) -> bool:
    """Check if raw movie data appears available."""
    # Check explicit metadata flags
    if metadata.get("has_raw_movie"):
        return True

    # Search for common indicators
    indicators = [
        "raw movie",
        "raw imaging",
        "raw frames",
        "tiff",
        "tif",
        "imaging data",
        "movie",
    ]

    return any(indicator in text for indicator in indicators)


def _check_masks_availability(text: str, metadata: dict[str, Any]) -> bool:
    """Check if segmentation masks appear available."""
    # Check explicit metadata flags
    if metadata.get("has_masks"):
        return True

    # Search for common indicators
    indicators = [
        "mask",
        "roi",
        "segmentation",
        "cell segmentation",
        "suite2p",
        "caiman",
    ]

    return any(indicator in text for indicator in indicators)


def _check_behavior_availability(metadata: dict[str, Any]) -> bool:
    """Check if behavioral data appears available."""
    return metadata.get("has_behavior", False)


def _check_ephys_availability(metadata: dict[str, Any]) -> bool:
    """Check if electrophysiology data appears available."""
    return metadata.get("has_ephys", False)


def _determine_next_step(score: float, has_raw_movie: bool, has_masks: bool) -> str:
    """
    Determine recommended next step based on score and data availability.

    Recommendation logic:
    - score < 0.20: reject_low_value
    - 0.20 <= score < 0.40: inspect_metadata
    - 0.40 <= score < 0.60 (no movie): load_processed_traces
    - 0.40 <= score < 0.75 (with movie): run_candidate_region_detection
    - score >= 0.75: run_event_detection
    """
    if score < 0.20:
        return "reject_low_value"

    if score < 0.40:
        return "inspect_metadata"

    if score < 0.60:
        if has_raw_movie:
            return "run_candidate_region_detection"
        else:
            return "load_processed_traces"

    if score < 0.75:
        if has_raw_movie:
            return "run_candidate_region_detection"
        else:
            return "run_event_detection"

    # score >= 0.75
    return "run_event_detection"


def scan_metadata_file(file_path: str | Path) -> DatasetTriageResult:
    """
    Scan a metadata file and score it for astrocyte reanalysis.

    Supports:
    - JSON files
    - Plain text files
    - NWB files (if pynwb is installed)

    Args:
        file_path: Path to metadata file

    Returns:
        DatasetTriageResult
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract session_id from filename
    session_id = file_path.stem

    # Handle JSON files
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r") as f:
            metadata = json.load(f)
        return score_dataset_metadata(metadata, session_id=session_id)

    # Handle NWB files
    if file_path.suffix.lower() == ".nwb":
        try:
            from neuros_astro.io.nwb_loader import summarize_nwb

            metadata = summarize_nwb(str(file_path))
            return score_dataset_metadata(metadata, session_id=session_id)
        except ImportError:
            # If pynwb not installed, use filename only
            metadata = {"filename": file_path.name, "path": str(file_path)}
            return score_dataset_metadata(metadata, session_id=session_id)

    # Handle plain text files
    with open(file_path, "r") as f:
        text = f.read()

    return score_dataset_metadata(text, session_id=session_id)
