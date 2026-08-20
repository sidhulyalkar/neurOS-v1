"""Adapter from the neurOS decoder analysis contract to mech-int experiments."""

from __future__ import annotations

from typing import Any

from .pytorch import PyTorchAdapter


class NeurOSModelAdapter(PyTorchAdapter):
    """Use a ``neuros.models.BaseModel`` through its declared analysis surface.

    The adapter intentionally relies on duck typing so ``neuros-mechint`` does not
    depend on ``neuros-models``.  This preserves the package direction:
    ``neuros-models`` may optionally use mech-int, while mech-int can still adapt
    third-party models independently.
    """

    def __init__(self, decoder: Any) -> None:
        if not hasattr(decoder, "analysis_manifest") or not hasattr(decoder, "analysis_model"):
            raise TypeError(
                "decoder must implement analysis_manifest() and analysis_model()"
            )
        manifest = decoder.analysis_manifest()
        model = decoder.analysis_model()
        if model is None:
            raise TypeError(
                f"{type(decoder).__name__} does not expose a mechanistically inspectable backend"
            )
        super().__init__(model)
        self.decoder = decoder
        self.manifest = manifest
        missing = self.missing_manifest_paths()
        if missing:
            raise ValueError(
                "analysis manifest contains paths absent from the backend model: "
                + ", ".join(missing)
            )

    @property
    def recommended_paths(self) -> tuple[str, ...]:
        return tuple(getattr(self.manifest, "recommended_paths", ()))

    def missing_manifest_paths(self) -> tuple[str, ...]:
        available = set(self._modules())
        paths = tuple(getattr(self.manifest, "surface_paths", self.recommended_paths))
        return tuple(path for path in paths if path not in available)

    def capture_recommended(self, inputs: Any) -> dict[str, Any]:
        return self.capture_outputs(inputs, self.recommended_paths)

    def model_fingerprint_payload(self) -> Any:
        base = super().model_fingerprint_payload()
        manifest = self.manifest.to_dict() if hasattr(self.manifest, "to_dict") else repr(self.manifest)
        return {
            "decoder_type": type(self.decoder).__name__,
            "manifest": manifest,
            "backend": base,
        }
