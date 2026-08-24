"""Inspectable LSTM decoder for neural time series."""

from __future__ import annotations

from typing import Any

import numpy as np

from neuros.models.analysis import AnalysisCapability, AnalysisSurface, InterpretabilityManifest
from neuros.models.torch_base import TorchDecoderModel


class LSTMModel(TorchDecoderModel):
    def __init__(
        self,
        n_channels: int,
        n_timepoints: int | None = None,
        n_classes: int = 2,
        *,
        lstm_units: int = 64,
        n_lstm_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 32,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        del n_timepoints  # retained as a compatibility argument; recurrent models support variable windows
        if n_channels <= 0 or n_lstm_layers <= 0:
            raise ValueError("n_channels and n_lstm_layers must be positive")
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
        )
        self.n_channels = int(n_channels)
        self.lstm_units = int(lstm_units)
        self.n_lstm_layers = int(n_lstm_layers)
        self.bidirectional = bool(bidirectional)
        self.dropout = float(dropout)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        arr = super()._validate_X(X)
        if arr.ndim != 3 or arr.shape[1] != self.n_channels:
            raise ValueError(
                f"LSTMModel expects (batch, {self.n_channels}, time), received {tuple(arr.shape)}"
            )
        return arr

    def _build_model(self) -> Any:
        _, nn = self._torch()
        n_channels = self.n_channels
        units = self.lstm_units
        layers = self.n_lstm_layers
        bidirectional = self.bidirectional
        dropout = self.dropout
        n_classes = self.n_classes
        directions = 2 if bidirectional else 1

        class LSTMNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_channels,
                    hidden_size=units,
                    num_layers=layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.embedding_norm = nn.LayerNorm(units * directions)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(units * directions, n_classes)

            def forward_features(self, x: Any) -> Any:
                sequence = x.transpose(1, 2)
                output, _state = self.lstm(sequence)
                return self.embedding_norm(output[:, -1, :])

            def forward(self, x: Any) -> Any:
                return self.classifier(self.dropout(self.forward_features(x)))

        return LSTMNet()

    def analysis_manifest(self) -> InterpretabilityManifest:
        caps = (
            AnalysisCapability.ACTIVATION_CAPTURE,
            AnalysisCapability.ACTIVATION_REPLACEMENT,
            AnalysisCapability.GRADIENT_ATTRIBUTION,
            AnalysisCapability.REPRESENTATIONS,
        )
        return InterpretabilityManifest(
            model_type=type(self).__name__,
            architecture="stacked LSTM neural sequence decoder",
            backend="pytorch",
            input_axes=("batch", "channel", "time"),
            output_semantics="class logits",
            capabilities=caps,
            surfaces=(
                AnalysisSurface(
                    "lstm", "recurrent_state_sequence", ("batch", "time", "hidden"),
                    "Recurrent sequence output. PyTorch returns a tuple, so generic output replacement requires an explicit selector in advanced tooling.",
                    (AnalysisCapability.GRADIENT_ATTRIBUTION,),
                    ("temporal gradient attribution", "hidden-state analysis"),
                    "Use a selector-aware adapter for direct tuple-output activation replacement.",
                ),
                AnalysisSurface(
                    "embedding_norm", "final_recurrent_representation", ("batch", "hidden"),
                    "Normalized last-step representation used by the readout.",
                    caps[:3], ("activation patching", "linear probes", "concept attribution"),
                ),
                AnalysisSurface(
                    "classifier", "decision_readout", ("batch", "class"),
                    "Linear class readout.", caps[:3], ("logit attribution", "readout intervention"),
                ),
            ),
            limitations=(
                "The raw nn.LSTM module returns structured tuple outputs; use embedding_norm for generic tensor-output replacement or a selector-aware adapter.",
            ),
        )
