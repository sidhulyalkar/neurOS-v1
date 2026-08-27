"""Execution policy for built-in Model Artifact v1 decoders.

Canonical artifact parsing is intentionally separate from permission to execute.
This module answers the second question: whether a verified envelope is allowed
to construct one of neurOS's compact built-in PyTorch decoder factories in the
current environment.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

from neuros.models.artifact_v1 import ModelArtifactManifest, ModelOutputContract

REQUIRED_RUNTIME_PACKAGES = frozenset(
    {"neuros-models", "neuros-core", "numpy", "torch", "safetensors"}
)
BUILTIN_FACTORY_IDS = frozenset(
    {
        "neuros.attention_fusion.v1",
        "neuros.cnn.v1",
        "neuros.eeg_conformer.v1",
        "neuros.eegnet.v1",
        "neuros.lstm.v1",
        "neuros.transformer.v1",
    }
)
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_WEIGHTS_BYTES = 512 * 1024 * 1024
MAX_PARAMETER_BUDGET = 50_000_000


def _positive_int(config: Mapping[str, Any], name: str, *, maximum: int) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"artifact model_config[{name!r}] must be a positive integer")
    if value > maximum:
        raise ValueError(
            f"artifact model_config[{name!r}]={value} exceeds the built-in v1 resource "
            f"budget of {maximum}"
        )
    return value


def _bounded_product(name: str, *values: int, maximum: int = MAX_PARAMETER_BUDGET) -> int:
    result = 1
    for value in values:
        result *= value
        if result > maximum:
            raise ValueError(
                f"artifact {name} requests approximately {result:,} scalar parameters/elements, "
                f"exceeding the built-in v1 resource budget of {maximum:,}"
            )
    return result


def preflight_bundle_size(path: str | Path) -> None:
    """Bound untrusted bundle size before parsing/hashing large content."""

    root = Path(path)
    if root.is_symlink():
        raise ValueError("model artifact root cannot be a symbolic link")
    manifest_path = root / "manifest.json"
    weights_path = root / "weights.safetensors"
    if manifest_path.exists() and not manifest_path.is_symlink():
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError(
                f"model artifact manifest exceeds {MAX_MANIFEST_BYTES} byte v1 safety limit"
            )
    if weights_path.exists() and not weights_path.is_symlink():
        if weights_path.stat().st_size > MAX_WEIGHTS_BYTES:
            raise ValueError(
                f"model artifact weights exceed {MAX_WEIGHTS_BYTES} byte built-in v1 safety limit"
            )


def validate_output_contract(output: ModelOutputContract) -> None:
    """Require semantics actually implemented by the built-in v1 factories."""

    if not isinstance(output, ModelOutputContract):
        raise TypeError("output_contract must be a ModelOutputContract")
    if output.task != "classification":
        raise ValueError("built-in Model Artifact v1 factories support classification only")
    if output.score_semantics != "class_logits":
        raise ValueError("built-in Model Artifact v1 factories emit class_logits")
    if output.probability_semantics != "uncalibrated_softmax":
        raise ValueError("built-in Model Artifact v1 factories emit uncalibrated_softmax only")
    if output.uncertainty_semantics != "none":
        raise ValueError(
            "built-in Model Artifact v1 factories do not emit a qualified uncertainty estimate"
        )


def validate_factory_resource_budget(manifest: ModelArtifactManifest) -> None:
    """Reject malicious constructor geometry before trusted factory allocation.

    Limits apply to compact built-in v1 decoders only. They are software safety
    limits, not scientific statements about future model families.
    """

    if manifest.factory_id not in BUILTIN_FACTORY_IDS:
        raise ValueError(
            f"artifact factory {manifest.factory_id!r} is not a built-in safe Model Artifact v1 factory"
        )

    config = dict(manifest.model_config)
    factory = manifest.factory_id
    _positive_int(config, "n_classes", maximum=4096)

    if factory == "neuros.eegnet.v1":
        channels = _positive_int(config, "n_channels", maximum=8192)
        temporal = _positive_int(config, "temporal_filters", maximum=2048)
        depth = _positive_int(config, "depth_multiplier", maximum=128)
        separable = _positive_int(config, "separable_filters", maximum=4096)
        temporal_kernel = _positive_int(config, "temporal_kernel", maximum=65535)
        separable_kernel = _positive_int(config, "separable_kernel", maximum=65535)
        _bounded_product("EEGNet temporal convolution", temporal, temporal_kernel)
        _bounded_product("EEGNet spatial projection", temporal, depth, channels)
        _bounded_product("EEGNet separable projection", temporal, depth, separable_kernel)
        _bounded_product("EEGNet feature mixing", temporal, depth, separable)
        return

    if factory == "neuros.cnn.v1":
        channels = _positive_int(config, "n_channels", maximum=8192)
        hidden = _positive_int(config, "hidden_channels", maximum=4096)
        blocks = _positive_int(config, "n_blocks", maximum=128)
        kernel = _positive_int(config, "kernel_size", maximum=65535)
        _bounded_product("CNN input projection", channels, hidden, kernel)
        _bounded_product(
            "CNN residual stack", hidden, hidden, kernel, blocks, maximum=100_000_000
        )
        return

    if factory == "neuros.lstm.v1":
        channels = _positive_int(config, "n_channels", maximum=8192)
        units = _positive_int(config, "lstm_units", maximum=4096)
        layers = _positive_int(config, "n_lstm_layers", maximum=64)
        directions = 2 if config.get("bidirectional") is True else 1
        _bounded_product(
            "LSTM recurrent state",
            4,
            units,
            units + channels,
            layers,
            directions,
            maximum=100_000_000,
        )
        return

    if factory == "neuros.transformer.v1":
        channels = _positive_int(config, "n_channels", maximum=8192)
        d_model = _positive_int(config, "d_model", maximum=2048)
        heads = _positive_int(config, "n_heads", maximum=128)
        layers = _positive_int(config, "n_layers", maximum=64)
        feedforward = _positive_int(config, "dim_feedforward", maximum=16384)
        max_timepoints = _positive_int(config, "max_timepoints", maximum=262144)
        if d_model % heads:
            raise ValueError("artifact Transformer d_model must remain divisible by n_heads")
        _bounded_product("Transformer input projection", channels, d_model)
        _bounded_product(
            "Transformer positional buffer", max_timepoints + 1, d_model, maximum=25_000_000
        )
        _bounded_product(
            "Transformer feed-forward stack",
            layers,
            d_model,
            feedforward,
            maximum=100_000_000,
        )
        return

    if factory == "neuros.eeg_conformer.v1":
        channels = _positive_int(config, "n_channels", maximum=8192)
        embedding = _positive_int(config, "embedding_dim", maximum=2048)
        heads = _positive_int(config, "n_heads", maximum=128)
        layers = _positive_int(config, "n_layers", maximum=64)
        multiplier = _positive_int(config, "feedforward_multiplier", maximum=64)
        temporal_kernel = _positive_int(config, "temporal_kernel", maximum=65535)
        if embedding % heads:
            raise ValueError("artifact EEG-Conformer embedding_dim must remain divisible by n_heads")
        _bounded_product("EEG-Conformer temporal stem", embedding, temporal_kernel)
        _bounded_product("EEG-Conformer spatial stem", embedding, channels)
        _bounded_product(
            "EEG-Conformer transformer stack",
            layers,
            embedding,
            embedding * multiplier,
            maximum=100_000_000,
        )
        return

    if factory == "neuros.attention_fusion.v1":
        dims = config.get("modality_dims")
        if not isinstance(dims, (list, tuple)) or not dims or len(dims) > 128:
            raise ValueError("artifact modality_dims must contain between 1 and 128 dimensions")
        normalized: list[int] = []
        for index, value in enumerate(dims):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > 65536:
                raise ValueError(
                    f"artifact modality_dims[{index}] must be an integer in [1, 65536]"
                )
            normalized.append(value)
        fusion = _positive_int(config, "fusion_dim", maximum=4096)
        total = sum(normalized)
        if total > 262144:
            raise ValueError("artifact modality_dims total exceeds built-in v1 resource budget")
        _bounded_product("attention-fusion projections", total, fusion)
        return

    raise AssertionError("built-in factory policy table is incomplete")


def validate_runtime_authority(manifest: ModelArtifactManifest) -> ModelArtifactManifest:
    """Validate identity semantics before any model constructor is invoked."""

    missing = REQUIRED_RUNTIME_PACKAGES - set(manifest.package_versions)
    if missing:
        raise ValueError(
            "Model Artifact v1 runtime authority is incomplete; missing exact package identities "
            f"for {sorted(missing)}"
        )
    if manifest.backend != "pytorch":
        raise ValueError("built-in Model Artifact v1 factories require backend='pytorch'")
    validate_output_contract(manifest.output_contract)
    validate_factory_resource_budget(manifest)
    return manifest


def validate_declared_environment(package_versions: Mapping[str, str]) -> None:
    """Prevent promotion metadata from lying about the running environment."""

    missing = REQUIRED_RUNTIME_PACKAGES - set(package_versions)
    if missing:
        raise ValueError(
            "custom package_versions cannot weaken Model Artifact v1 runtime authority; "
            f"missing {sorted(missing)}"
        )
    for distribution, declared in package_versions.items():
        if not isinstance(distribution, str) or not distribution.strip():
            raise ValueError("package_versions keys must be non-empty distribution names")
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError("package_versions values must be non-empty version strings")
        try:
            actual = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise ValueError(
                f"cannot promote package identity {distribution!r}: distribution is not installed"
            ) from exc
        if actual != declared:
            raise ValueError(
                f"declared package identity {distribution}=={declared} does not match "
                f"the promotion environment ({actual})"
            )


def validate_backend_runtime(manifest: ModelArtifactManifest) -> None:
    """Check framework-reported identity separately from package metadata."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load built-in Model Artifact v1 decoders") from exc
    actual = str(torch.__version__)
    if manifest.backend_version != actual:
        raise RuntimeError(
            f"artifact backend_version={manifest.backend_version!r} does not match "
            f"the active PyTorch runtime {actual!r}"
        )


__all__ = [
    "BUILTIN_FACTORY_IDS",
    "MAX_MANIFEST_BYTES",
    "MAX_WEIGHTS_BYTES",
    "REQUIRED_RUNTIME_PACKAGES",
    "preflight_bundle_size",
    "validate_backend_runtime",
    "validate_declared_environment",
    "validate_factory_resource_budget",
    "validate_output_contract",
    "validate_runtime_authority",
]
