"""Event-based tokenization for irregular astrocyte event sequences."""

import numpy as np
from neuros_astro.metadata.schema import AstroEvent, TokenizedAstroSequence


class AstroEventTokenizer:
    """
    Tokenizer for astrocyte events (irregular time series).

    Converts AstroEvent objects into model-ready token vectors with features:
    - onset_time_s: Event onset time
    - duration_s: Event duration
    - peak_dff: Peak amplitude
    - area_px_norm: Normalized spatial area
    - centroid_y_norm: Normalized Y position
    - centroid_x_norm: Normalized X position
    - propagation_speed_norm: Normalized propagation speed
    - direction_sin: Sine of propagation direction
    - direction_cos: Cosine of propagation direction
    - confidence: Detection confidence
    """

    def __init__(
        self,
        normalize: bool = True,
        image_height: int | None = None,
        image_width: int | None = None,
    ):
        """
        Initialize tokenizer.

        Args:
            normalize: Whether to normalize features
            image_height: Image height for centroid normalization
            image_width: Image width for centroid normalization
        """
        self.normalize = normalize
        self.image_height = image_height
        self.image_width = image_width

        # Normalization statistics (computed during tokenization)
        self.norm_stats = {}

    def tokenize(
        self, events: list[AstroEvent], session_id: str | None = None
    ) -> TokenizedAstroSequence:
        """
        Convert events to tokenized sequence.

        Args:
            events: List of AstroEvent objects
            session_id: Session identifier (uses first event's session_id if None)

        Returns:
            TokenizedAstroSequence object
        """
        if len(events) == 0:
            # Return empty sequence
            return TokenizedAstroSequence(
                session_id=session_id or "empty",
                tokens=[],
                timestamps_s=[],
                region_ids=[],
                feature_names=self._get_feature_names(),
                metadata={"norm_stats": {}},
            )

        if session_id is None:
            session_id = events[0].session_id

        # Extract features
        timestamps_s = []
        region_ids = []
        feature_vectors = []

        for event in events:
            # Timestamp: onset time in seconds
            onset_s = event.onset_frame / event.duration_s * (
                event.offset_frame - event.onset_frame + 1
            )
            # Better: infer from duration_s and offset
            # Approximate onset_s from duration and frame indices
            onset_s = event.onset_frame  # Will need actual frame rate for conversion

            # For now, use event onset as timestamp
            # Note: This requires frame_rate_hz, which we don't have in AstroEvent
            # We'll use duration_s as a proxy or require it separately
            timestamps_s.append(float(event.onset_frame))  # Placeholder: use frame index

            region_ids.append(event.region_id)

            # Extract feature vector
            features = self._extract_features(event)
            feature_vectors.append(features)

        # Convert to numpy array
        feature_matrix = np.array(feature_vectors, dtype=np.float32)

        # Normalize if requested
        if self.normalize:
            feature_matrix, norm_stats = self._normalize_features(feature_matrix)
            self.norm_stats = norm_stats
        else:
            norm_stats = {}

        # Convert to list of lists for Pydantic
        tokens = feature_matrix.tolist()

        return TokenizedAstroSequence(
            session_id=session_id,
            tokens=tokens,
            timestamps_s=timestamps_s,
            region_ids=region_ids,
            feature_names=self._get_feature_names(),
            metadata={"norm_stats": norm_stats},
        )

    def _extract_features(self, event: AstroEvent) -> list[float]:
        """Extract feature vector from single event."""
        features = []

        # Temporal features
        features.append(float(event.onset_frame))  # onset_time (frame index for now)
        features.append(event.duration_s)  # duration_s
        features.append(event.peak_dff)  # peak_dff

        # Spatial features
        if event.area_px is not None:
            features.append(event.area_px)
        else:
            features.append(0.0)

        if event.centroid_yx is not None:
            centroid_y, centroid_x = event.centroid_yx

            # Normalize by image dimensions if available
            if self.image_height is not None:
                centroid_y = centroid_y / self.image_height
            if self.image_width is not None:
                centroid_x = centroid_x / self.image_width

            features.append(centroid_y)
            features.append(centroid_x)
        else:
            features.append(0.0)
            features.append(0.0)

        # Propagation features
        if event.propagation_speed is not None:
            features.append(event.propagation_speed)
        else:
            features.append(0.0)

        # Direction (circular encoding)
        if event.direction_rad is not None:
            features.append(np.sin(event.direction_rad))
            features.append(np.cos(event.direction_rad))
        else:
            features.append(0.0)
            features.append(0.0)

        # Confidence
        features.append(event.confidence)

        return features

    def _get_feature_names(self) -> list[str]:
        """Get feature names in order."""
        return [
            "onset_time",  # Frame index or time
            "duration_s",
            "peak_dff",
            "area_px",
            "centroid_y",
            "centroid_x",
            "propagation_speed",
            "direction_sin",
            "direction_cos",
            "confidence",
        ]

    def _normalize_features(
        self, feature_matrix: np.ndarray
    ) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        """
        Normalize features to zero mean, unit variance.

        Handles special cases:
        - Circular features (sin/cos) are not normalized
        - Confidence is already in [0, 1]
        - Avoid divide-by-zero

        Args:
            feature_matrix: Array [n_events, n_features]

        Returns:
            normalized_matrix: Normalized feature matrix
            norm_stats: Dictionary of normalization statistics
        """
        feature_names = self._get_feature_names()
        normalized = feature_matrix.copy()
        norm_stats = {}

        # Features to normalize (skip circular encoding and confidence)
        skip_normalize = {"direction_sin", "direction_cos", "confidence"}

        for i, name in enumerate(feature_names):
            if name in skip_normalize:
                continue

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
