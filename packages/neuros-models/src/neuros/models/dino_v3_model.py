"""Fail-closed compatibility shim for the historical ``DinoV3Model`` name."""

from __future__ import annotations

import warnings

import numpy as np

from neuros.models.base_model import BaseModel


class DinoV3Model(BaseModel):
    """Deprecated compatibility shim.

    Earlier neurOS releases called a torchvision ViT-B/16 ImageNet backbone
    ``DinoV3Model`` when true DINOv3 weights were unavailable. That behavior was
    scientifically misleading and has been removed. Use a verified upstream
    vision/foundation-model adapter and ``FoundationEmbeddingDecoder`` instead.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        super().__init__()
        warnings.warn(
            "DinoV3Model is deprecated: the historical implementation did not load DINOv3. "
            "Use neuros-foundation with a verified upstream adapter and FoundationEmbeddingDecoder.",
            DeprecationWarning,
            stacklevel=2,
        )

    @staticmethod
    def _unsupported() -> RuntimeError:
        return RuntimeError(
            "DinoV3Model no longer fabricates a DINOv3 backend. Configure a verified foundation-model adapter instead."
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        del X, y
        raise self._unsupported()

    def predict(self, X: np.ndarray) -> np.ndarray:
        del X
        raise self._unsupported()
