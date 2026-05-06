"""Binned tokenization for regular time series of astrocyte activity."""

import numpy as np
from neuros_astro.metadata.schema import AstroEvent, AstroGraph, TokenizedAstroSequence
from neuros_astro.networks.graph_features import compute_graph_summary_features


class BinnedAstroTokenizer:
    """
    Tokenizer for binned astrocyte activity (regular time series).

    Aggregates events and optionally graph features into regular time bins.

    Features per bin:
    - event_count: Number of events in bin
    - mean_peak_dff: Mean peak amplitude
    - total_area_px: Total spatial area
    - mean_confidence: Mean detection confidence
    - active_region_count: Number of active regions
    - mean_duration_s: Mean event duration
    - network_density: Graph density (if graphs provided)
    - mean_edge_weight: Mean edge weight (if graphs provided)
    """

    def __init__(self, bin_size_s: float = 1.0, normalize: bool = True):
        """
        Initialize binned tokenizer.

        Args:
            bin_size_s: Time bin size in seconds
            normalize: Whether to normalize features
        """
        self.bin_size_s = bin_size_s
        self.normalize = normalize
        self.norm_stats = {}

    def tokenize(
        self,
        events: list[AstroEvent],
        duration_s: float,
        session_id: str | None = None,
        graphs: list[AstroGraph] | None = None,
    ) -> TokenizedAstroSequence:
        """
        Convert events to binned token sequence.

        Args:
            events: List of AstroEvent objects
            duration_s: Total duration in seconds
            session_id: Session identifier
            graphs: Optional list of AstroGraph objects for network features

        Returns:
            TokenizedAstroSequence with regular time bins
        """
        if session_id is None and len(events) > 0:
            session_id = events[0].session_id
        elif session_id is None:
            session_id = "empty"

        # Create time bins
        n_bins = int(np.ceil(duration_s / self.bin_size_s))
        bin_edges = np.arange(n_bins + 1) * self.bin_size_s
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Initialize feature matrix
        feature_vectors = []

        for bin_idx in range(n_bins):
            bin_start = bin_edges[bin_idx]
            bin_end = bin_edges[bin_idx + 1]

            # Find events in this bin
            # Note: We need frame_rate_hz to convert frames to time
            # For now, assume events have been converted to time already
            bin_events = [
                e
                for e in events
                if bin_start <= e.onset_frame <= bin_end  # Placeholder: needs time conversion
            ]

            # Extract bin features
            features = self._extract_bin_features(bin_events, bin_start, bin_end, graphs)
            feature_vectors.append(features)

        # Convert to numpy array
        feature_matrix = np.array(feature_vectors, dtype=np.float32)

        # Normalize if requested
        if self.normalize:
            feature_matrix, norm_stats = self._normalize_features(feature_matrix)
            self.norm_stats = norm_stats
        else:
            norm_stats = {}

        # Convert to list for Pydantic
        tokens = feature_matrix.tolist()
        timestamps_s = bin_centers.tolist()

        return TokenizedAstroSequence(
            session_id=session_id,
            tokens=tokens,
            timestamps_s=timestamps_s,
            region_ids=[None] * n_bins,  # No specific region for binned features
            feature_names=self._get_feature_names(include_network=graphs is not None),
            metadata={"bin_size_s": self.bin_size_s, "norm_stats": norm_stats},
        )

    def _extract_bin_features(
        self,
        bin_events: list[AstroEvent],
        bin_start_s: float,
        bin_end_s: float,
        graphs: list[AstroGraph] | None,
    ) -> list[float]:
        """Extract features for a single time bin."""
        features = []

        # Event count
        n_events = len(bin_events)
        features.append(float(n_events))

        if n_events > 0:
            # Mean peak amplitude
            peaks = [e.peak_dff for e in bin_events]
            features.append(float(np.mean(peaks)))

            # Total spatial area
            areas = [e.area_px for e in bin_events if e.area_px is not None]
            if areas:
                features.append(float(np.sum(areas)))
            else:
                features.append(0.0)

            # Mean confidence
            confidences = [e.confidence for e in bin_events]
            features.append(float(np.mean(confidences)))

            # Active region count
            active_regions = set(e.region_id for e in bin_events if e.region_id is not None)
            features.append(float(len(active_regions)))

            # Mean duration
            durations = [e.duration_s for e in bin_events]
            features.append(float(np.mean(durations)))
        else:
            # No events in this bin
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Network features (if graphs provided)
        if graphs is not None:
            # Find graph that overlaps with this bin
            overlapping_graphs = [
                g
                for g in graphs
                if g.window_start_s <= bin_end_s and g.window_end_s >= bin_start_s
            ]

            if overlapping_graphs:
                # Use first overlapping graph
                graph = overlapping_graphs[0]
                graph_features = compute_graph_summary_features(graph)

                features.append(graph_features.get("density", 0.0))
                features.append(graph_features.get("mean_edge_weight", 0.0))
            else:
                features.extend([0.0, 0.0])

        return features

    def _get_feature_names(self, include_network: bool = False) -> list[str]:
        """Get feature names."""
        names = [
            "event_count",
            "mean_peak_dff",
            "total_area_px",
            "mean_confidence",
            "active_region_count",
            "mean_duration_s",
        ]

        if include_network:
            names.extend(["network_density", "mean_edge_weight"])

        return names

    def _normalize_features(
        self, feature_matrix: np.ndarray
    ) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        """Normalize features to zero mean, unit variance."""
        feature_names = self._get_feature_names()
        normalized = feature_matrix.copy()
        norm_stats = {}

        for i, name in enumerate(feature_names):
            values = feature_matrix[:, i]

            # Compute statistics
            mean = float(np.mean(values))
            std = float(np.std(values))

            # Avoid divide by zero
            if std < 1e-10:
                std = 1.0

            # Normalize
            normalized[:, i] = (values - mean) / std

            # Store stats
            norm_stats[name] = {"mean": mean, "std": std}

        return normalized, norm_stats
