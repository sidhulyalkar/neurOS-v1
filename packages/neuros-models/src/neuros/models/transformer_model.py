"""Real temporal Transformer decoder for neural windows.

This replaces the historical sklearn-MLP placeholder that was previously
published under the ``TransformerModel`` name.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from neuros.models.analysis import AnalysisCapability, AnalysisSurface, InterpretabilityManifest
from neuros.models.torch_base import TorchDecoderModel


class TransformerModel(TorchDecoderModel):
    """Temporal Transformer encoder over channel vectors at each time step."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 2,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_timepoints: int = 4096,
        learning_rate: float = 3e-4,
        n_epochs: int = 20,
        batch_size: int = 32,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        if n_channels <= 0 or d_model <= 0 or n_layers <= 0:
            raise ValueError("n_channels, d_model, and n_layers must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
        )
        self.n_channels = int(n_channels)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.max_timepoints = int(max_timepoints)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        arr = super()._validate_X(X)
        if arr.ndim != 3 or arr.shape[1] != self.n_channels:
            raise ValueError(
                f"TransformerModel expects (batch, {self.n_channels}, time), received {tuple(arr.shape)}"
            )
        if arr.shape[2] > self.max_timepoints:
            raise ValueError(f"time dimension exceeds max_timepoints={self.max_timepoints}")
        return arr

    def _build_model(self) -> Any:
        torch, nn = self._torch()
        n_channels = self.n_channels
        d_model = self.d_model
        heads = self.n_heads
        layers = self.n_layers
        ff = self.dim_feedforward
        dropout = self.dropout
        max_time = self.max_timepoints
        n_classes = self.n_classes

        class SinusoidalPosition(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                position = torch.arange(max_time + 1, dtype=torch.float32).unsqueeze(1)
                div = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model)
                )
                pe = torch.zeros(max_time + 1, d_model)
                pe[:, 0::2] = torch.sin(position * div)
                pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
                self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

            def forward(self, x: Any) -> Any:
                return x + self.pe[:, : x.size(1)]

        class TemporalTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Linear(n_channels, d_model)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
                nn.init.normal_(self.cls_token, std=0.02)
                self.position = SinusoidalPosition()
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
                self.embedding_norm = nn.LayerNorm(d_model)
                self.classifier = nn.Linear(d_model, n_classes)

            def forward_tokens(self, x: Any) -> Any:
                x = x.transpose(1, 2)
                x = self.input_projection(x)
                cls = self.cls_token.expand(x.size(0), -1, -1)
                x = self.position(torch.cat([cls, x], dim=1))
                return self.encoder(x)

            def forward_features(self, x: Any) -> Any:
                tokens = self.forward_tokens(x)
                return self.embedding_norm(tokens[:, 0])

            def forward(self, x: Any) -> Any:
                return self.classifier(self.forward_features(x))

        return TemporalTransformer()

    def analysis_manifest(self) -> InterpretabilityManifest:
        capture = (
            AnalysisCapability.ACTIVATION_CAPTURE,
            AnalysisCapability.ACTIVATION_REPLACEMENT,
            AnalysisCapability.GRADIENT_ATTRIBUTION,
        )
        surfaces: list[AnalysisSurface] = [
            AnalysisSurface(
                "input_projection", "neural_token_embedding", ("batch", "time", "model"),
                "Projection of each time step's channel vector into model space.", capture,
                ("activation patching", "token-level probes", "temporal attribution"),
            )
        ]
        for index in range(self.n_layers):
            surfaces.extend(
                [
                    AnalysisSurface(
                        f"encoder.layers.{index}.self_attn",
                        f"layer_{index}_self_attention",
                        ("batch", "time", "model"),
                        "Context mixing performed by multi-head self-attention.",
                        (AnalysisCapability.GRADIENT_ATTRIBUTION,),
                        ("attention pattern analysis", "QK/OV analysis with specialized tooling"),
                        "nn.MultiheadAttention returns structured outputs; use selector-aware tools for direct replacement.",
                    ),
                    AnalysisSurface(
                        f"encoder.layers.{index}.linear1",
                        f"layer_{index}_mlp_expansion",
                        ("batch", "time", "mlp"),
                        "Transformer feed-forward expansion, a useful sparse-feature/transcoder target.", capture,
                        ("activation patching", "SAE", "transcoder", "circuit discovery"),
                    ),
                ]
            )
        surfaces.extend(
            [
                AnalysisSurface(
                    "embedding_norm", "pooled_cls_representation", ("batch", "model"),
                    "Final CLS representation before classification.", capture,
                    ("linear probes", "RSA/CKA", "activation patching"),
                ),
                AnalysisSurface(
                    "classifier", "decision_readout", ("batch", "class"),
                    "Linear decoder readout.", capture, ("logit attribution", "causal intervention"),
                ),
            ]
        )
        return InterpretabilityManifest(
            model_type=type(self).__name__,
            architecture="temporal Transformer encoder",
            backend="pytorch",
            input_axes=("batch", "channel", "time"),
            output_semantics="class logits",
            capabilities=capture
            + (
                AnalysisCapability.REPRESENTATIONS,
                AnalysisCapability.ATTENTION,
            ),
            surfaces=tuple(surfaces),
            method_notes={
                "circuit_tracing": "Treat attribution or sparse features as hypotheses, then validate with held-out interventions.",
                "attention": "Attention weights alone are not a causal explanation; pair them with value-path or activation interventions.",
            },
        )


TemporalTransformerModel = TransformerModel
