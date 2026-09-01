"""Native train-only PCA representation baseline."""
from __future__ import annotations

import numpy as np

from .contracts import EvaluationScope, FitRegime, RepresentationEmbedding, SequenceBatch


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class PCARepresentation:
    """SVD PCA fit only from declared training observations."""

    fit_regime = FitRegime.TRAIN_ONLY_INDUCTIVE
    evaluation_scope = EvaluationScope.BATCH_TRANSFORM

    def __init__(self, n_components: int = 2, *, method_id: str = "pca") -> None:
        self.n_components = _positive_int(n_components, name="n_components")
        if not isinstance(method_id, str) or not method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        self.method_id = method_id
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def _fit(self, train: SequenceBatch) -> None:
        samples = np.asarray(train.concatenate(), dtype=np.float64)
        max_components = min(samples.shape)
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds train rank bound {max_components}"
            )
        mean = np.mean(samples, axis=0)
        centered = samples - mean
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        components = np.array(vh[: self.n_components], copy=True)
        mean = np.array(mean, copy=True)
        singular_values = np.array(singular_values[: self.n_components], copy=True)
        mean.setflags(write=False)
        components.setflags(write=False)
        singular_values.setflags(write=False)
        self.mean_ = mean
        self.components_ = components
        self.singular_values_ = singular_values

    def embed(
        self,
        train: SequenceBatch,
        evaluation: SequenceBatch,
    ) -> RepresentationEmbedding:
        if train.feature_count != evaluation.feature_count:
            raise ValueError("train and evaluation feature dimensions must match")
        self._fit(train)
        assert self.mean_ is not None
        assert self.components_ is not None
        embedded = tuple(
            (np.asarray(sequence, dtype=np.float64) - self.mean_) @ self.components_.T
            for sequence in evaluation.sequences
        )
        return RepresentationEmbedding(
            method_id=self.method_id,
            sequences=embedded,
            sequence_ids=evaluation.sequence_ids,
            fit_regime=self.fit_regime,
            metadata={
                "n_components": self.n_components,
                "fit_sample_count": train.sample_count,
                "target_specific_fit_observations": 0,
                "coordinate_frame": "shared_train_fitted_axes",
                "implementation": "numpy_svd",
            },
        )
