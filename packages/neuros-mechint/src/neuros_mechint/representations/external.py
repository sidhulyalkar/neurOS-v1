"""Adapters for externally computed temporal self-supervised representations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from .contracts import FitRegime, RepresentationEmbedding, SequenceBatch, _freeze_metadata

_ALLOWED_LINEAGE_STATUS = {
    "disjoint_verified",
    "overlap_detected",
    "possible_overlap",
    "unknown_lineage",
    "not_audited",
}


class PrecomputedTemporalSSLRepresentation:
    """Use fixed external/pretrained embeddings without retraining in the benchmark.

    The adapter deliberately does not call an external model. It binds already
    computed per-sequence embeddings to exact sequence identities and carries
    model/pretraining metadata for a higher-level scientific authority layer to
    audit separately.
    """

    fit_regime = FitRegime.EXTERNAL_PRETRAINED

    def __init__(
        self,
        embeddings: Mapping[str, Any],
        *,
        model_id: str,
        model_version: str,
        method_id: str = "temporal_ssl",
        pretraining_datasets: Sequence[str] = (),
        pretraining_lineage_status: str = "not_audited",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonblank string")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must be a nonblank string")
        if not isinstance(method_id, str) or not method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        if pretraining_lineage_status not in _ALLOWED_LINEAGE_STATUS:
            raise ValueError("pretraining_lineage_status must be an explicit audit status")
        if not isinstance(embeddings, Mapping) or not embeddings:
            raise ValueError("embeddings must be a nonempty sequence-ID mapping")

        stored: dict[str, np.ndarray] = {}
        latent_dims: set[int] = set()
        for sequence_id, value in embeddings.items():
            if not isinstance(sequence_id, str) or not sequence_id.strip():
                raise ValueError("embedding keys must be nonblank sequence IDs")
            array = np.array(value, copy=True, subok=False)
            if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 1:
                raise ValueError(
                    f"external embedding {sequence_id!r} must be 2-D [time, latent]"
                )
            if array.dtype.kind not in "iuf":
                raise TypeError(
                    "external embeddings must contain real numeric, non-boolean values"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError("external embeddings must contain only finite values")
            array.setflags(write=False)
            stored[sequence_id] = array
            latent_dims.add(int(array.shape[1]))
        if len(latent_dims) != 1:
            raise ValueError("all external embeddings must share one latent dimension")

        datasets: list[str] = []
        for dataset in pretraining_datasets:
            if not isinstance(dataset, str) or not dataset.strip():
                raise ValueError("pretraining dataset IDs must be nonblank strings")
            datasets.append(dataset)
        if len(set(datasets)) != len(datasets):
            raise ValueError("pretraining dataset IDs must be unique")

        self._embeddings = MappingProxyType(stored)
        self.model_id = model_id
        self.model_version = model_version
        self.method_id = method_id
        self.pretraining_datasets = tuple(datasets)
        self.pretraining_lineage_status = pretraining_lineage_status
        self.metadata = _freeze_metadata(metadata)

    def embed_sequence(
        self,
        train: SequenceBatch,
        evaluation: SequenceBatch,
    ) -> RepresentationEmbedding:
        """Bind exactly one external sequence so missing siblings remain local failures."""
        if len(evaluation.sequences) != 1:
            raise ValueError("embed_sequence requires exactly one evaluation sequence")
        return self.embed(train, evaluation)

    def embed(
        self,
        train: SequenceBatch,
        evaluation: SequenceBatch,
    ) -> RepresentationEmbedding:
        del train
        selected: list[np.ndarray] = []
        for sequence_id, source in zip(
            evaluation.sequence_ids,
            evaluation.sequences,
            strict=True,
        ):
            if sequence_id not in self._embeddings:
                raise KeyError(f"external representation is missing sequence {sequence_id!r}")
            embedding = self._embeddings[sequence_id]
            if embedding.shape[0] != source.shape[0]:
                raise ValueError(
                    f"external embedding {sequence_id!r} has {embedding.shape[0]} "
                    f"timepoints; evaluation sequence has {source.shape[0]}"
                )
            selected.append(embedding)

        return RepresentationEmbedding(
            method_id=self.method_id,
            sequences=tuple(selected),
            sequence_ids=evaluation.sequence_ids,
            fit_regime=self.fit_regime,
            metadata={
                "model_id": self.model_id,
                "model_version": self.model_version,
                "pretraining_datasets": self.pretraining_datasets,
                "pretraining_lineage_status": self.pretraining_lineage_status,
                "target_specific_fit_observations": 0,
                "coordinate_frame": "shared_external_encoder",
                "external_metadata": dict(self.metadata),
            },
        )
