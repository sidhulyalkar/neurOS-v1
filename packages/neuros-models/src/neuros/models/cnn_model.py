"""Inspectable temporal convolutional decoder for neural windows."""

from __future__ import annotations

from typing import Any

import numpy as np

from neuros.models.analysis import AnalysisCapability, AnalysisSurface, InterpretabilityManifest
from neuros.models.torch_base import TorchDecoderModel


class CNNModel(TorchDecoderModel):
    """Residual dilated temporal CNN operating on ``(batch, channels, time)`` windows."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 2,
        *,
        hidden_channels: int = 64,
        n_blocks: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 32,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        if n_channels <= 0 or n_blocks <= 0:
            raise ValueError("n_channels and n_blocks must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
        )
        self.n_channels = int(n_channels)
        self.hidden_channels = int(hidden_channels)
        self.n_blocks = int(n_blocks)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        arr = super()._validate_X(X)
        if arr.ndim != 3 or arr.shape[1] != self.n_channels:
            raise ValueError(
                f"CNNModel expects (batch, {self.n_channels}, time), received {tuple(arr.shape)}"
            )
        return arr

    def _build_model(self) -> Any:
        _, nn = self._torch()
        in_channels = self.n_channels
        hidden = self.hidden_channels
        n_blocks = self.n_blocks
        kernel = self.kernel_size
        dropout = self.dropout
        n_classes = self.n_classes

        class ResidualBlock(nn.Module):
            def __init__(self, dilation: int) -> None:
                super().__init__()
                padding = dilation * (kernel // 2)
                self.conv1 = nn.Conv1d(hidden, hidden, kernel, padding=padding, dilation=dilation)
                self.norm1 = nn.BatchNorm1d(hidden)
                self.act1 = nn.GELU()
                self.drop1 = nn.Dropout(dropout)
                self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=padding, dilation=dilation)
                self.norm2 = nn.BatchNorm1d(hidden)
                self.act2 = nn.GELU()

            def forward(self, x: Any) -> Any:
                residual = x
                x = self.drop1(self.act1(self.norm1(self.conv1(x))))
                x = self.norm2(self.conv2(x))
                return self.act2(x + residual)

        class TemporalCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(in_channels, hidden, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(hidden),
                    nn.GELU(),
                )
                self.blocks = nn.ModuleList(
                    [ResidualBlock(2**index) for index in range(n_blocks)]
                )
                self.embedding_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Linear(hidden, n_classes)

            def forward_features(self, x: Any) -> Any:
                x = self.stem(x)
                for block in self.blocks:
                    x = block(x)
                return self.embedding_pool(x).squeeze(-1)

            def forward(self, x: Any) -> Any:
                return self.classifier(self.forward_features(x))

        return TemporalCNN()

    def analysis_manifest(self) -> InterpretabilityManifest:
        caps = (
            AnalysisCapability.ACTIVATION_CAPTURE,
            AnalysisCapability.ACTIVATION_REPLACEMENT,
            AnalysisCapability.GRADIENT_ATTRIBUTION,
            AnalysisCapability.REPRESENTATIONS,
        )
        surfaces = [
            AnalysisSurface(
                "stem", "local_temporal_features", ("batch", "feature", "time"),
                "Initial channel-to-feature temporal projection.",
                caps[:3], ("temporal attribution", "activation patching"),
            )
        ]
        for index in range(self.n_blocks):
            surfaces.append(
                AnalysisSurface(
                    f"blocks.{index}", f"dilated_residual_block_{index}",
                    ("batch", "feature", "time"),
                    f"Residual temporal block with dilation {2**index}.",
                    caps[:3], ("activation patching", "block ablation", "causal scrubbing"),
                )
            )
        surfaces.extend(
            [
                AnalysisSurface(
                    "embedding_pool", "pooled_representation", ("batch", "feature", "one"),
                    "Global pooled temporal representation.", caps[:3],
                    ("linear probes", "RSA/CKA", "sparse feature decomposition"),
                ),
                AnalysisSurface(
                    "classifier", "decision_readout", ("batch", "class"),
                    "Linear class readout.", caps[:3], ("logit attribution", "readout ablation"),
                ),
            ]
        )
        return InterpretabilityManifest(
            model_type=type(self).__name__,
            architecture="residual dilated temporal CNN",
            backend="pytorch",
            input_axes=("batch", "channel", "time"),
            output_semantics="class logits",
            capabilities=caps,
            surfaces=tuple(surfaces),
        )
