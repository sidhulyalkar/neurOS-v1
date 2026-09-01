"""Optional upstream T-PHATE adapter.

No T-PHATE implementation is vendored here. The upstream project currently
ships a Yale non-commercial license, while neurOS is MIT licensed. Users must
review and satisfy upstream terms before separately installing ``tphate``.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from .contracts import (
    FitRegime,
    RepresentationEmbedding,
    RepresentationError,
    RepresentationUnavailableError,
    SequenceBatch,
)
from .pca import _positive_int

UPSTREAM_REPOSITORY = "https://github.com/KrishnaswamyLab/TPHATE"
UPSTREAM_LICENSE_NOTICE = (
    "TPHATE is an optional external dependency. Its upstream repository currently "
    "ships a Yale Non-Commercial License. neurOS does not vendor TPHATE; review "
    "the upstream license and install the package separately if your use is permitted."
)


class TPHATEUnavailableError(RepresentationUnavailableError):
    """The optional upstream T-PHATE package is not installed."""


class TPHATEEmbeddingError(RepresentationError):
    """Upstream T-PHATE failed for one preserved trajectory."""


class TPHATERepresentation:
    """Transductive T-PHATE embedding, fit independently per trajectory.

    T-PHATE's autocorrelation view treats row order as temporal adjacency. This
    adapter therefore never concatenates independent trajectories. Each
    evaluation sequence receives a fresh upstream estimator. Resulting MDS
    coordinate frames are sequence-local and must not be pooled across
    sequences without an explicit alignment protocol.
    """

    fit_regime = FitRegime.TRANSDUCTIVE_TARGET_OBSERVED

    def __init__(
        self,
        n_components: int = 2,
        *,
        knn: int = 5,
        decay: int | None = 40,
        t: int | str = "auto",
        gamma: float = 1.0,
        n_pca: int | None = 100,
        mds_solver: str = "sgd",
        knn_dist: str = "euclidean",
        mds_dist: str = "euclidean",
        mds: str = "metric",
        n_jobs: int = 1,
        random_state: int = 0,
        smooth_window: int = 1,
        method_id: str = "tphate",
    ) -> None:
        self.n_components = _positive_int(n_components, name="n_components")
        self.knn = _positive_int(knn, name="knn")
        self.decay = None if decay is None else _positive_int(decay, name="decay")
        if t != "auto":
            t = _positive_int(t, name="t")
        self.t = t
        if isinstance(gamma, bool) or not isinstance(
            gamma, (int, float, np.integer, np.floating)
        ):
            raise TypeError("gamma must be a finite real between -1 and 1")
        self.gamma = float(gamma)
        if not np.isfinite(self.gamma) or not -1 <= self.gamma <= 1:
            raise ValueError("gamma must be a finite real between -1 and 1")
        self.n_pca = None if n_pca is None else _positive_int(n_pca, name="n_pca")
        if mds_solver not in {"sgd", "smacof"}:
            raise ValueError("mds_solver must be 'sgd' or 'smacof'")
        if mds not in {"classic", "metric", "nonmetric"}:
            raise ValueError("mds must be 'classic', 'metric', or 'nonmetric'")
        if not isinstance(knn_dist, str) or not knn_dist:
            raise ValueError("knn_dist must be a nonblank string")
        if not isinstance(mds_dist, str) or not mds_dist:
            raise ValueError("mds_dist must be a nonblank string")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
            raise TypeError("n_jobs must be an integer")
        if int(n_jobs) == 0:
            raise ValueError("n_jobs cannot be zero")
        if isinstance(random_state, bool) or not isinstance(
            random_state, (int, np.integer)
        ):
            raise TypeError("random_state must be an integer")
        self.mds_solver = mds_solver
        self.knn_dist = knn_dist
        self.mds_dist = mds_dist
        self.mds = mds
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.smooth_window = _positive_int(smooth_window, name="smooth_window")
        if not isinstance(method_id, str) or not method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        self.method_id = method_id

    def _load_upstream(self) -> tuple[Any, str]:
        try:
            module = import_module("tphate")
        except (ImportError, ModuleNotFoundError) as exc:
            raise TPHATEUnavailableError(
                f"{UPSTREAM_LICENSE_NOTICE} Upstream: {UPSTREAM_REPOSITORY}"
            ) from exc
        estimator = getattr(module, "TPHATE", None)
        if estimator is None or not callable(estimator):
            raise TPHATEUnavailableError(
                "Installed 'tphate' module does not expose callable TPHATE; "
                "upstream API compatibility cannot be established."
            )
        version = str(getattr(module, "__version__", "unknown"))
        return estimator, version

    def _embed_one(self, estimator_type: Any, sequence_id: str, sequence: np.ndarray) -> np.ndarray:
        n_pca = self.n_pca
        if n_pca is not None and n_pca >= min(sequence.shape):
            n_pca = None
        estimator = estimator_type(
            n_components=self.n_components,
            knn=self.knn,
            decay=self.decay,
            n_landmark=None,
            t=self.t,
            gamma=self.gamma,
            n_pca=n_pca,
            mds_solver=self.mds_solver,
            knn_dist=self.knn_dist,
            mds_dist=self.mds_dist,
            mds=self.mds,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0,
            smooth_window=self.smooth_window,
        )
        try:
            embedding = np.asarray(
                estimator.fit_transform(np.array(sequence, copy=True)),
                dtype=np.float64,
            )
        except IndexError as exc:
            raise TPHATEEmbeddingError(
                f"T-PHATE failed for sequence {sequence_id!r}. The upstream "
                "autocorrelation dropoff calculation can fail when its smoothed "
                "ACF has no negative crossing; no fallback or fabricated temporal "
                "kernel was applied."
            ) from exc
        except Exception as exc:
            raise TPHATEEmbeddingError(
                f"T-PHATE failed for sequence {sequence_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        expected = (sequence.shape[0], self.n_components)
        if embedding.shape != expected:
            raise TPHATEEmbeddingError(
                f"T-PHATE returned shape {embedding.shape} for sequence "
                f"{sequence_id!r}; expected {expected}"
            )
        if not np.all(np.isfinite(embedding)):
            raise TPHATEEmbeddingError(
                f"T-PHATE returned non-finite coordinates for sequence {sequence_id!r}"
            )
        return embedding

    def embed(
        self,
        train: SequenceBatch,
        evaluation: SequenceBatch,
    ) -> RepresentationEmbedding:
        if train.feature_count != evaluation.feature_count:
            raise ValueError("train and evaluation feature dimensions must match")
        estimator_type, version = self._load_upstream()
        embedded = tuple(
            self._embed_one(estimator_type, sequence_id, sequence)
            for sequence_id, sequence in zip(
                evaluation.sequence_ids,
                evaluation.sequences,
                strict=True,
            )
        )
        return RepresentationEmbedding(
            method_id=self.method_id,
            sequences=embedded,
            sequence_ids=evaluation.sequence_ids,
            fit_regime=self.fit_regime,
            metadata={
                "upstream_package": "TPHATE",
                "upstream_version": version,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "license_boundary": "upstream_yale_noncommercial_review_required",
                "n_components": self.n_components,
                "target_specific_fit_observations": evaluation.sample_count,
                "fit_sequence_count": len(evaluation.sequences),
                "coordinate_frame": "per_sequence_unaligned_mds",
                "sequence_boundary_policy": "fresh_estimator_per_sequence",
                "landmarking": "disabled",
            },
        )
