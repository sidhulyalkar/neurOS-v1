"""Curated decoder catalog for honest model discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DecoderCard:
    id: str
    class_name: str
    family: str
    backend: str
    input_contract: str
    status: str
    mechint: str
    best_for: str
    caveats: str = ""
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_CATALOG: tuple[DecoderCard, ...] = (
    DecoderCard(
        "eegnet", "EEGNetModel", "compact EEG CNN", "PyTorch",
        "(batch, channels, time)", "reference", "high",
        "Low-parameter EEG classification and interpretable temporal/spatial filters.",
        "Canonical compact baseline, not claimed as universally state of the art.",
        "Lawhern et al., J Neural Eng 2018",
    ),
    DecoderCard(
        "eeg-conformer", "EEGConformerModel", "convolution + Transformer", "PyTorch",
        "(batch, channels, time)", "modern", "very-high",
        "EEG decoding where local filter-bank inductive bias and long-range context both matter.",
        "Attention maps require causal validation; do not equate attention with explanation.",
        "Song et al., IEEE TNSRE 2023",
    ),
    DecoderCard(
        "temporal-transformer", "TransformerModel", "Transformer encoder", "PyTorch",
        "(batch, channels, time)", "modern", "very-high",
        "A clean attention-based research baseline with stable MLP and readout hook points.",
        "Needs enough data and careful temporal token resolution; no convolutional EEG inductive bias.",
    ),
    DecoderCard(
        "temporal-cnn", "CNNModel", "residual dilated temporal CNN", "PyTorch",
        "(batch, channels, time)", "reference", "high",
        "Efficient neural time-series decoding and temporal receptive-field studies.",
    ),
    DecoderCard(
        "lstm", "LSTMModel", "recurrent sequence model", "PyTorch",
        "(batch, channels, time)", "reference", "medium",
        "Sequential neural dynamics and recurrent-state analyses.",
        "Raw nn.LSTM outputs are structured tuples; generic interventions should target embedding_norm.",
    ),
    DecoderCard(
        "attention-fusion", "AttentionFusionModel", "multimodal learned gating", "PyTorch",
        "(batch, concatenated modality features)", "modern", "very-high",
        "Multimodal BCI fusion, modality routing, and causal modality-dependence studies.",
        "Gate weights are not causal evidence until validated with modality interventions.",
    ),
    DecoderCard(
        "classical", "SimpleClassifier/SVM/RandomForest/GBDT", "classical ML", "scikit-learn",
        "(batch, features)", "reference", "feature-level",
        "Strong low-data baselines and sanity checks.",
        "Use feature-level importance/permutation analyses rather than pretending these expose deep circuits.",
    ),
)


def list_decoder_cards(*, mechint_ready: bool = False) -> tuple[DecoderCard, ...]:
    if not mechint_ready:
        return MODEL_CATALOG
    return tuple(card for card in MODEL_CATALOG if card.mechint in {"high", "very-high", "medium"})


def get_decoder_card(model_id: str) -> DecoderCard:
    for card in MODEL_CATALOG:
        if card.id == model_id:
            return card
    raise KeyError(f"Unknown decoder id: {model_id}")
