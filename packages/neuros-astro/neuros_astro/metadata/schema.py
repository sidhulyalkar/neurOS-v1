"""Core data schemas for neuros-astro."""

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


class AstroSession(BaseModel):
    """Represents a session or dataset unit."""

    session_id: str = Field(..., description="Unique session identifier")
    subject_id: str | None = Field(None, description="Subject/animal identifier")
    source_path: str = Field(..., description="Path to source data")
    source_type: Literal["nwb", "tiff", "suite2p", "caiman", "aqua", "zarr", "unknown"] = Field(
        "unknown", description="Type of source data"
    )
    frame_rate_hz: float | None = Field(None, gt=0, description="Imaging frame rate in Hz")
    imaging_modality: str | None = Field(None, description="Imaging modality (e.g., two-photon)")
    indicator: str | None = Field(None, description="Calcium indicator (e.g., GCaMP6f)")
    promoter: str | None = Field(None, description="Promoter/cell type marker (e.g., GFAP)")
    brain_region: str | None = Field(None, description="Brain region")
    has_raw_movie: bool = Field(False, description="Whether raw movie data is available")
    has_masks: bool = Field(False, description="Whether segmentation masks are available")
    has_behavior: bool = Field(False, description="Whether behavioral data is available")
    has_ephys: bool = Field(False, description="Whether electrophysiology data is available")
    metadata_score: float = Field(0.0, ge=0.0, le=1.0, description="Astro reanalysis score")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sub-001_ses-001",
                "source_path": "/data/session.nwb",
                "source_type": "nwb",
                "frame_rate_hz": 10.0,
                "imaging_modality": "two-photon",
                "indicator": "GCaMP6f",
                "promoter": "GFAP",
                "metadata_score": 0.85,
            }
        }


class AstroRegion(BaseModel):
    """Represents a candidate astrocyte ROI or spatial region."""

    region_id: str = Field(..., description="Unique region identifier")
    session_id: str = Field(..., description="Session identifier")
    mask_shape: tuple[int, int] | None = Field(None, description="Shape of mask (height, width)")
    centroid_yx: tuple[float, float] | None = Field(None, description="Centroid (y, x)")
    area_px: float | None = Field(None, gt=0, description="Area in pixels")
    perimeter_px: float | None = Field(None, gt=0, description="Perimeter in pixels")
    eccentricity: float | None = Field(None, ge=0, le=1, description="Eccentricity")
    solidity: float | None = Field(None, ge=0, le=1, description="Solidity")
    branchiness: float | None = Field(None, ge=0, description="Branchiness/complexity metric")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    region_type: Literal["soma", "process", "domain", "unknown"] = Field(
        "unknown", description="Type of astrocyte region"
    )


class AstroEvent(BaseModel):
    """Represents one spatiotemporal astrocytic calcium event."""

    event_id: str = Field(..., description="Unique event identifier")
    session_id: str = Field(..., description="Session identifier")
    region_id: str | None = Field(None, description="Associated region identifier")
    onset_frame: int = Field(..., ge=0, description="Event onset frame")
    offset_frame: int = Field(..., ge=0, description="Event offset frame")
    peak_frame: int = Field(..., ge=0, description="Event peak frame")
    duration_s: float = Field(..., gt=0, description="Event duration in seconds")
    peak_dff: float = Field(..., description="Peak dF/F amplitude")
    area_px: float | None = Field(None, gt=0, description="Spatial area in pixels")
    centroid_yx: tuple[float, float] | None = Field(None, description="Event centroid (y, x)")
    propagation_speed: float | None = Field(None, ge=0, description="Propagation speed (px/s)")
    direction_rad: float | None = Field(None, description="Propagation direction (radians)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")

    @model_validator(mode='after')
    def validate_frame_ordering(self) -> 'AstroEvent':
        """Validate that frame indices are properly ordered."""
        if self.offset_frame < self.onset_frame:
            raise ValueError("offset_frame must be >= onset_frame")
        if self.peak_frame < self.onset_frame or self.peak_frame > self.offset_frame:
            raise ValueError("peak_frame must be between onset_frame and offset_frame")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_001",
                "session_id": "ses_001",
                "region_id": "roi_05",
                "onset_frame": 100,
                "offset_frame": 150,
                "peak_frame": 120,
                "duration_s": 5.0,
                "peak_dff": 0.25,
                "confidence": 0.9,
            }
        }


class AstroGraph(BaseModel):
    """Represents a windowed astrocyte functional network."""

    session_id: str = Field(..., description="Session identifier")
    window_start_s: float = Field(..., ge=0, description="Window start time (seconds)")
    window_end_s: float = Field(..., gt=0, description="Window end time (seconds)")
    nodes: list[str] = Field(..., description="List of node IDs (region IDs)")
    edges: list[tuple[str, str]] = Field(..., description="List of edges as (source, target) tuples")
    edge_weights: list[float] = Field(..., description="Edge weights (same length as edges)")
    edge_metric: Literal["correlation", "lagged_correlation", "event_coactivation", "mutual_information"] = Field(
        "event_coactivation", description="Metric used for edge weights"
    )

    @model_validator(mode='after')
    def validate_window_and_edges(self) -> 'AstroGraph':
        """Validate window ordering and edge/weight consistency."""
        if self.window_end_s <= self.window_start_s:
            raise ValueError("window_end_s must be > window_start_s")
        if len(self.edges) != len(self.edge_weights):
            raise ValueError("edges and edge_weights must have same length")
        return self


class TokenizedAstroSequence(BaseModel):
    """Represents model-ready astrocyte event tokens."""

    session_id: str = Field(..., description="Session identifier")
    tokens: list[list[float]] = Field(..., description="Token array [n_tokens, feature_dim]")
    timestamps_s: list[float] = Field(..., description="Timestamp for each token")
    region_ids: list[str | None] = Field(..., description="Region ID for each token (if applicable)")
    feature_names: list[str] = Field(..., description="Names of features in token vectors")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode='after')
    def validate_token_dimensions(self) -> 'TokenizedAstroSequence':
        """Validate that token dimensions are consistent."""
        n_tokens = len(self.tokens)
        if len(self.timestamps_s) != n_tokens:
            raise ValueError("timestamps_s length must match number of tokens")
        if len(self.region_ids) != n_tokens:
            raise ValueError("region_ids length must match number of tokens")
        if n_tokens > 0 and len(self.tokens[0]) != len(self.feature_names):
            raise ValueError("Token feature dimension must match feature_names length")
        return self

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert to NumPy arrays for easy export."""
        return {
            "tokens": np.array(self.tokens, dtype=np.float32),
            "timestamps_s": np.array(self.timestamps_s, dtype=np.float32),
            "feature_names": np.array(self.feature_names, dtype=object),
        }


class DatasetTriageResult(BaseModel):
    """Result of dataset triage/scoring for astrocyte reanalysis."""

    session_id: str = Field(..., description="Session or dataset identifier")
    astro_reanalysis_score: float = Field(..., ge=0.0, le=1.0, description="Overall score [0, 1]")
    has_raw_movie: bool = Field(False, description="Raw movie data available")
    has_masks: bool = Field(False, description="Segmentation masks available")
    has_behavior: bool = Field(False, description="Behavioral data available")
    has_ephys: bool = Field(False, description="Electrophysiology data available")
    matched_astro_terms: list[str] = Field(default_factory=list, description="Matched astro terms")
    matched_calcium_terms: list[str] = Field(default_factory=list, description="Matched calcium terms")
    matched_modality_terms: list[str] = Field(default_factory=list, description="Matched modality terms")
    warnings: list[str] = Field(default_factory=list, description="Warnings or caveats")
    recommended_next_step: str = Field("inspect_metadata", description="Recommended next action")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "demo_session",
                "astro_reanalysis_score": 0.75,
                "has_raw_movie": True,
                "has_masks": True,
                "has_behavior": True,
                "matched_astro_terms": ["GFAP", "astrocyte"],
                "matched_calcium_terms": ["GCaMP6f"],
                "matched_modality_terms": ["two-photon"],
                "recommended_next_step": "run_event_detection",
            }
        }
