"""Temporal-order interventions for sequence-local representation methods."""
from __future__ import annotations

import hashlib
import numpy as np

from .contracts import EvaluationScope, FitRegime, RepresentationEmbedding, RepresentationMethod, SequenceBatch


class TemporalOrderInterventionRepresentation:
    """Destroy temporal order before a wrapped method, then restore row identity."""

    evaluation_scope = EvaluationScope.SEQUENCE_LOCAL

    def __init__(self, method: RepresentationMethod, *, seed: int = 0, block_size: int = 1, method_id: str | None = None) -> None:
        try:
            scope = EvaluationScope(method.evaluation_scope)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("wrapped representation must declare a valid evaluation_scope") from exc
        if scope is not EvaluationScope.SEQUENCE_LOCAL:
            raise ValueError("temporal-order intervention requires a sequence-local method")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
            raise TypeError("block_size must be an integer")
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        wrapped_id = getattr(method, "method_id", None)
        if not isinstance(wrapped_id, str) or not wrapped_id.strip():
            raise ValueError("wrapped representation must expose a nonblank method_id")
        try:
            fit_regime = FitRegime(method.fit_regime)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("wrapped representation must declare a valid fit_regime") from exc
        if method_id is None:
            method_id = f"{wrapped_id}__temporal_order_destroyed"
        if not isinstance(method_id, str) or not method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        if method_id == wrapped_id:
            raise ValueError("intervention method_id must differ from wrapped method_id")
        self.method = method
        self.seed = int(seed)
        self.block_size = block_size
        self.method_id = method_id
        self.fit_regime = fit_regime

    def _rng_for(self, sequence_id: str) -> np.random.Generator:
        digest = hashlib.sha256(
            f"neuros.temporal_order.v1\0{self.seed}\0{sequence_id}".encode("utf-8")
        ).digest()
        return np.random.default_rng(int.from_bytes(digest[:16], "little"))

    def _permutation(self, n_rows: int, sequence_id: str) -> np.ndarray:
        n_blocks = (n_rows + self.block_size - 1) // self.block_size
        if n_blocks < 2:
            raise ValueError("block_size must leave at least two temporal blocks to permute")
        order = self._rng_for(sequence_id).permutation(n_blocks)
        if np.array_equal(order, np.arange(n_blocks)):
            order = np.roll(order, 1)
        blocks = [
            np.arange(block * self.block_size, min((block + 1) * self.block_size, n_rows), dtype=np.int64)
            for block in order
        ]
        permutation = np.concatenate(blocks)
        if permutation.shape != (n_rows,) or np.unique(permutation).size != n_rows:
            raise RuntimeError("internal temporal permutation is not bijective")
        if np.array_equal(permutation, np.arange(n_rows)):
            raise RuntimeError("temporal intervention unexpectedly produced identity order")
        return permutation

    @staticmethod
    def _inverse(permutation: np.ndarray) -> np.ndarray:
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(permutation.size, dtype=permutation.dtype)
        return inverse

    def embed(self, train: SequenceBatch, evaluation: SequenceBatch) -> RepresentationEmbedding:
        if train.feature_count != evaluation.feature_count:
            raise ValueError("train and evaluation feature dimensions must match")
        permutations = tuple(
            self._permutation(sequence.shape[0], sequence_id)
            for sequence_id, sequence in zip(evaluation.sequence_ids, evaluation.sequences, strict=True)
        )
        intervened = SequenceBatch(
            sequences=tuple(sequence[permutation] for sequence, permutation in zip(evaluation.sequences, permutations, strict=True)),
            sequence_ids=evaluation.sequence_ids,
            metadata={
                **dict(evaluation.metadata),
                "temporal_order_intervention": "deterministic_block_permutation_v1",
                "temporal_order_seed": self.seed,
                "temporal_order_block_size": self.block_size,
            },
        )
        embedding = self.method.embed(train, intervened)
        if embedding.sequence_ids != evaluation.sequence_ids:
            raise ValueError("wrapped representation changed evaluation sequence identity")
        if len(embedding.sequences) != len(permutations):
            raise ValueError("wrapped representation changed evaluation sequence count")

        restored = []
        digests = {}
        for sequence_id, source, latent, permutation in zip(
            evaluation.sequence_ids, evaluation.sequences, embedding.sequences, permutations, strict=True
        ):
            if latent.shape[0] != source.shape[0]:
                raise ValueError("wrapped representation changed evaluation timepoint count")
            restored.append(np.asarray(latent)[self._inverse(permutation)])
            digests[sequence_id] = hashlib.sha256(
                permutation.astype("<i8", copy=False).tobytes()
            ).hexdigest()

        return RepresentationEmbedding(
            method_id=self.method_id,
            sequences=tuple(restored),
            sequence_ids=evaluation.sequence_ids,
            fit_regime=self.fit_regime,
            metadata={
                "intervention": "temporal_order_destroyed",
                "intervention_schema": "neuros.temporal_order_intervention.v1",
                "permutation_policy": "deterministic_block_permutation",
                "seed": self.seed,
                "block_size": self.block_size,
                "wrapped_method_id": self.method.method_id,
                "wrapped_fit_regime": self.fit_regime.value,
                "permutation_sha256_by_sequence": digests,
                "row_identity_policy": "inverse_permute_embedding_before_scoring",
                "wrapped_embedding_metadata": dict(embedding.metadata),
            },
        )
