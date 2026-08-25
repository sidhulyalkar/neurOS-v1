"""Faithful compact EEGNet-style decoder with stable mechanistic hook points."""

from __future__ import annotations

from typing import Any

import numpy as np

from neuros.models.analysis import AnalysisCapability, AnalysisSurface, InterpretabilityManifest
from neuros.models.torch_base import TorchDecoderModel


class EEGNetModel(TorchDecoderModel):
    """Compact temporal/spatial depthwise-separable EEG classifier.

    Inputs use the neurOS neural-window convention ``(batch, channels, time)``.
    The implementation follows the defining EEGNet ingredients: temporal
    filtering, depthwise spatial filtering, separable temporal convolution, and
    a compact linear readout. Adaptive pooling keeps the decoder usable across
    compatible window lengths while preserving named analysis surfaces.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 2,
        *,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel: int = 63,
        separable_kernel: int = 15,
        dropout: float = 0.25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int = 32,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if temporal_kernel % 2 == 0 or separable_kernel % 2 == 0:
            raise ValueError("temporal kernels must be odd so same-length padding is unambiguous")
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
        )
        self.n_channels = int(n_channels)
        self.temporal_filters = int(temporal_filters)
        self.depth_multiplier = int(depth_multiplier)
        self.separable_filters = int(separable_filters)
        self.temporal_kernel = int(temporal_kernel)
        self.separable_kernel = int(separable_kernel)
        self.dropout = float(dropout)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        arr = super()._validate_X(X)
        if arr.ndim != 3:
            raise ValueError("EEGNetModel expects X with shape (batch, channels, time)")
        if arr.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, received {arr.shape[1]}")
        return arr

    def _build_model(self) -> Any:
        _, nn = self._torch()
        n_channels = self.n_channels
        f1 = self.temporal_filters
        d = self.depth_multiplier
        f2 = self.separable_filters
        temporal_kernel = self.temporal_kernel
        separable_kernel = self.separable_kernel
        dropout = self.dropout
        n_classes = self.n_classes

        class EEGNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temporal = nn.Conv2d(
                    1, f1, kernel_size=(1, temporal_kernel),
                    padding=(0, temporal_kernel // 2), bias=False
                )
                self.temporal_bn = nn.BatchNorm2d(f1)
                self.spatial = nn.Conv2d(
                    f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False
                )
                self.spatial_bn = nn.BatchNorm2d(f1 * d)
                self.spatial_activation = nn.ELU()
                self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
                self.dropout1 = nn.Dropout(dropout)
                self.separable_depthwise = nn.Conv2d(
                    f1 * d,
                    f1 * d,
                    kernel_size=(1, separable_kernel),
                    padding=(0, separable_kernel // 2),
                    groups=f1 * d,
                    bias=False,
                )
                self.separable_pointwise = nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False)
                self.separable_bn = nn.BatchNorm2d(f2)
                self.separable_activation = nn.ELU()
                self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
                self.dropout2 = nn.Dropout(dropout)
                self.embedding_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(f2, n_classes)

            def forward_features(self, x: Any) -> Any:
                x = x.unsqueeze(1)
                x = self.temporal_bn(self.temporal(x))
                x = self.spatial_activation(self.spatial_bn(self.spatial(x)))
                x = self.dropout1(self.pool1(x))
                x = self.separable_depthwise(x)
                x = self.separable_pointwise(x)
                x = self.separable_activation(self.separable_bn(x))
                x = self.dropout2(self.pool2(x))
                x = self.embedding_pool(x)
                return x.flatten(1)

            def forward(self, x: Any) -> Any:
                return self.classifier(self.forward_features(x))

        return EEGNet()

    def analysis_manifest(self) -> InterpretabilityManifest:
        capture = (
            AnalysisCapability.ACTIVATION_CAPTURE,
            AnalysisCapability.ACTIVATION_REPLACEMENT,
            AnalysisCapability.GRADIENT_ATTRIBUTION,
        )
        return InterpretabilityManifest(
            model_type=type(self).__name__,
            architecture="EEGNet-style depthwise-separable temporal/spatial CNN",
            backend="pytorch",
            input_axes=("batch", "channel", "time"),
            output_semantics="class logits",
            capabilities=capture + (AnalysisCapability.REPRESENTATIONS,),
            surfaces=(
                AnalysisSurface(
                    "temporal", "temporal_filter_bank", ("batch", "filter", "channel", "time"),
                    "Learned temporal filters before cross-electrode mixing.", capture,
                    ("activation patching", "gradient attribution", "frequency response audit"),
                ),
                AnalysisSurface(
                    "spatial", "electrode_spatial_projection", ("batch", "filter", "virtual_channel", "time"),
                    "Depthwise projection across the complete electrode montage.", capture,
                    ("channel ablation", "activation patching", "topographic weight analysis"),
                ),
                AnalysisSurface(
                    "separable_pointwise", "feature_mixing", ("batch", "feature", "virtual_channel", "time"),
                    "Pointwise mixing after depthwise temporal refinement.", capture,
                    ("activation patching", "sparse feature decomposition"),
                ),
                AnalysisSurface(
                    "embedding_pool", "pooled_representation", ("batch", "feature", "one", "one"),
                    "Compact representation consumed by the classifier.", capture,
                    ("linear probes", "RSA/CKA", "concept attribution"),
                ),
                AnalysisSurface(
                    "classifier", "decision_readout", ("batch", "class"),
                    "Linear readout from the pooled EEG representation.", capture,
                    ("logit attribution", "necessity/sufficiency"),
                ),
            ),
            method_notes={
                "attention": "Not applicable; EEGNet is convolutional.",
                "causal": "Prefer interventions on held-out trials and report accuracy/logit effects, not saliency alone.",
            },
        )
