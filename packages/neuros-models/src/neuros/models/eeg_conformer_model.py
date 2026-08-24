"""Convolution-first EEG Conformer with explicit mechanistic surfaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from neuros.models.analysis import AnalysisCapability, AnalysisSurface, InterpretabilityManifest
from neuros.models.torch_base import TorchDecoderModel


class EEGConformerModel(TorchDecoderModel):
    """Compact EEG Conformer inspired by convolutional Transformer EEG decoders.

    The convolutional stem learns temporal filters and a full-montage spatial
    projection before pooling the signal into tokens. Transformer blocks then
    integrate longer-range context. The named components are intentionally
    stable so mechanistic experiments can compare filters, token states, MLP
    features, attention, and readout contributions across checkpoints.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 2,
        *,
        embedding_dim: int = 40,
        temporal_kernel: int = 25,
        pool_length: int = 25,
        pool_stride: int = 10,
        n_heads: int = 4,
        n_layers: int = 4,
        feedforward_multiplier: int = 4,
        dropout: float = 0.3,
        learning_rate: float = 3e-4,
        n_epochs: int = 20,
        batch_size: int = 32,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if n_channels <= 0 or n_layers <= 0:
            raise ValueError("n_channels and n_layers must be positive")
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
        )
        self.n_channels = int(n_channels)
        self.embedding_dim = int(embedding_dim)
        self.temporal_kernel = int(temporal_kernel)
        self.pool_length = int(pool_length)
        self.pool_stride = int(pool_stride)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.feedforward_multiplier = int(feedforward_multiplier)
        self.dropout = float(dropout)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        arr = super()._validate_X(X)
        if arr.ndim != 3 or arr.shape[1] != self.n_channels:
            raise ValueError(
                f"EEGConformerModel expects (batch, {self.n_channels}, time), received {tuple(arr.shape)}"
            )
        if arr.shape[2] < self.pool_length:
            raise ValueError("time dimension must be at least pool_length")
        return arr

    def _build_model(self) -> Any:
        _, nn = self._torch()
        n_channels = self.n_channels
        emb = self.embedding_dim
        temporal_kernel = self.temporal_kernel
        pool_length = self.pool_length
        pool_stride = self.pool_stride
        heads = self.n_heads
        layers = self.n_layers
        ff = emb * self.feedforward_multiplier
        dropout = self.dropout
        n_classes = self.n_classes

        class PatchEmbedding(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temporal = nn.Conv2d(
                    1, emb, kernel_size=(1, temporal_kernel),
                    padding=(0, temporal_kernel // 2), bias=False
                )
                self.spatial = nn.Conv2d(emb, emb, kernel_size=(n_channels, 1), bias=False)
                self.norm = nn.BatchNorm2d(emb)
                self.activation = nn.ELU()
                self.pool = nn.AvgPool2d(
                    kernel_size=(1, pool_length), stride=(1, pool_stride)
                )
                self.dropout = nn.Dropout(dropout)
                self.projection = nn.Conv2d(emb, emb, kernel_size=(1, 1))

            def forward(self, x: Any) -> Any:
                x = x.unsqueeze(1)
                x = self.temporal(x)
                x = self.spatial(x)
                x = self.dropout(self.pool(self.activation(self.norm(x))))
                x = self.projection(x)
                return x.squeeze(2).transpose(1, 2)

        class EEGConformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_embedding = PatchEmbedding()
                layer = nn.TransformerEncoderLayer(
                    d_model=emb,
                    nhead=heads,
                    dim_feedforward=ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
                self.embedding_norm = nn.LayerNorm(emb)
                self.classifier = nn.Linear(emb, n_classes)

            def forward_tokens(self, x: Any) -> Any:
                return self.encoder(self.patch_embedding(x))

            def forward_features(self, x: Any) -> Any:
                tokens = self.forward_tokens(x)
                return self.embedding_norm(tokens.mean(dim=1))

            def forward(self, x: Any) -> Any:
                return self.classifier(self.forward_features(x))

        return EEGConformer()

    def analysis_manifest(self) -> InterpretabilityManifest:
        capture = (
            AnalysisCapability.ACTIVATION_CAPTURE,
            AnalysisCapability.ACTIVATION_REPLACEMENT,
            AnalysisCapability.GRADIENT_ATTRIBUTION,
        )
        surfaces: list[AnalysisSurface] = [
            AnalysisSurface(
                "patch_embedding.temporal", "temporal_filter_bank",
                ("batch", "filter", "channel", "time"),
                "Convolutional temporal filter bank before montage mixing.", capture,
                ("frequency response audit", "activation patching", "temporal attribution"),
            ),
            AnalysisSurface(
                "patch_embedding.spatial", "montage_projection",
                ("batch", "embedding", "virtual_channel", "time"),
                "Full-electrode spatial projection that collapses the montage.", capture,
                ("electrode ablation", "topographic weight analysis", "activation patching"),
            ),
            AnalysisSurface(
                "patch_embedding.projection", "eeg_token_sequence",
                ("batch", "embedding", "one", "token"),
                "Pooled convolutional EEG patches before Transformer context integration.", capture,
                ("token probes", "RSA/CKA", "activation patching"),
            ),
        ]
        for index in range(self.n_layers):
            surfaces.extend(
                [
                    AnalysisSurface(
                        f"encoder.layers.{index}.self_attn", f"layer_{index}_attention",
                        ("batch", "token", "embedding"),
                        "Self-attention context mixing over EEG tokens.",
                        (AnalysisCapability.GRADIENT_ATTRIBUTION,),
                        ("attention analysis", "selector-aware attention intervention"),
                        "MultiheadAttention has structured outputs; use specialized adapters for output replacement.",
                    ),
                    AnalysisSurface(
                        f"encoder.layers.{index}.linear1", f"layer_{index}_mlp_features",
                        ("batch", "token", "mlp"),
                        "Transformer feed-forward expansion; suitable for learned sparse feature dictionaries.",
                        capture,
                        ("SAE", "transcoder", "activation patching", "circuit tracing"),
                    ),
                ]
            )
        surfaces.extend(
            [
                AnalysisSurface(
                    "embedding_norm", "global_eeg_representation", ("batch", "embedding"),
                    "Mean-pooled contextual EEG representation.", capture,
                    ("linear probes", "RSA/CKA", "concept attribution"),
                ),
                AnalysisSurface(
                    "classifier", "decision_readout", ("batch", "class"),
                    "Linear output readout.", capture,
                    ("logit attribution", "necessity/sufficiency"),
                ),
            ]
        )
        return InterpretabilityManifest(
            model_type=type(self).__name__,
            architecture="EEG Conformer: convolutional temporal/spatial patch embedding + Transformer encoder",
            backend="pytorch",
            input_axes=("batch", "channel", "time"),
            output_semantics="class logits",
            capabilities=capture
            + (AnalysisCapability.REPRESENTATIONS, AnalysisCapability.ATTENTION),
            surfaces=tuple(surfaces),
            method_notes={
                "recommended_ladder": "Start with filter/token probes, nominate candidate features/circuits, then test necessity and sufficiency on held-out trials.",
                "sparse_features": "SAE/transcoder features are candidate computational variables, not causal proof until interventions validate them.",
            },
        )
