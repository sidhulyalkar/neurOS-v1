"""Trainable modality-attention fusion with causal analysis surfaces."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from neuros.models.analysis import AnalysisCapability, AnalysisSurface, InterpretabilityManifest
from neuros.models.torch_base import TorchDecoderModel


class AttentionFusionModel(TorchDecoderModel):
    """Fuse concatenated modality features using sample-dependent learned gates."""

    def __init__(
        self,
        modality_dims: Sequence[int],
        n_classes: int = 2,
        *,
        fusion_dim: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 32,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        dims = tuple(int(value) for value in modality_dims)
        if not dims or any(value <= 0 for value in dims):
            raise ValueError("modality_dims must contain positive dimensions")
        super().__init__(
            n_classes=n_classes,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
        )
        self.modality_dims = dims
        self.total_dim = sum(dims)
        self.fusion_dim = int(fusion_dim)
        self.dropout = float(dropout)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        arr = super()._validate_X(X)
        if arr.ndim != 2 or arr.shape[1] != self.total_dim:
            raise ValueError(
                f"AttentionFusionModel expects (batch, {self.total_dim}), received {tuple(arr.shape)}"
            )
        return arr

    def _build_model(self) -> Any:
        torch, nn = self._torch()
        dims = self.modality_dims
        fusion_dim = self.fusion_dim
        dropout = self.dropout
        n_classes = self.n_classes

        class FusionNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projections = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(dim, fusion_dim),
                            nn.LayerNorm(fusion_dim),
                            nn.GELU(),
                        )
                        for dim in dims
                    ]
                )
                self.gate = nn.Sequential(
                    nn.Linear(fusion_dim, max(8, fusion_dim // 2)),
                    nn.GELU(),
                    nn.Linear(max(8, fusion_dim // 2), 1),
                )
                self.embedding_norm = nn.LayerNorm(fusion_dim)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(fusion_dim, n_classes)

            def projected(self, x: Any) -> list[Any]:
                chunks = torch.split(x, dims, dim=1)
                return [projection(chunk) for projection, chunk in zip(self.projections, chunks)]

            def attention(self, x: Any) -> Any:
                projected = self.projected(x)
                stacked = torch.stack(projected, dim=1)
                scores = self.gate(stacked).squeeze(-1)
                return torch.softmax(scores, dim=1)

            def forward_features(self, x: Any) -> Any:
                projected = self.projected(x)
                stacked = torch.stack(projected, dim=1)
                scores = self.gate(stacked).squeeze(-1)
                weights = torch.softmax(scores, dim=1)
                fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
                return self.embedding_norm(fused)

            def forward(self, x: Any) -> Any:
                return self.classifier(self.dropout(self.forward_features(x)))

        return FusionNet()

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        self._require_trained()
        torch, _ = self._torch()
        model = self._ensure_model()
        model.eval()
        with torch.no_grad():
            weights = model.attention(self._tensor(X))
        return weights.detach().cpu().numpy()

    def analysis_manifest(self) -> InterpretabilityManifest:
        capture = (
            AnalysisCapability.ACTIVATION_CAPTURE,
            AnalysisCapability.ACTIVATION_REPLACEMENT,
            AnalysisCapability.GRADIENT_ATTRIBUTION,
        )
        surfaces = [
            AnalysisSurface(
                f"projections.{index}", f"modality_{index}_representation",
                ("batch", "fusion_feature"),
                f"Projected representation for modality {index}.", capture,
                ("modality patching", "cross-modal representation comparison"),
            )
            for index in range(len(self.modality_dims))
        ]
        surfaces.extend(
            [
                AnalysisSurface(
                    "gate", "sample_dependent_modality_gate", ("batch", "modality", "score"),
                    "Shared scorer that assigns a relevance logit to each modality representation.",
                    capture + (AnalysisCapability.MODALITY_GATING,),
                    ("gate ablation", "counterfactual modality suppression", "reliability correlation"),
                ),
                AnalysisSurface(
                    "embedding_norm", "fused_representation", ("batch", "fusion_feature"),
                    "Weighted fused representation before the readout.", capture,
                    ("activation patching", "linear probes", "RSA/CKA"),
                ),
                AnalysisSurface(
                    "classifier", "decision_readout", ("batch", "class"),
                    "Linear class readout.", capture, ("logit attribution", "necessity/sufficiency"),
                ),
            ]
        )
        return InterpretabilityManifest(
            model_type=type(self).__name__,
            architecture="sample-dependent modality attention/gating fusion",
            backend="pytorch",
            input_axes=("batch", "concatenated_modality_feature"),
            output_semantics="class logits",
            capabilities=capture
            + (
                AnalysisCapability.REPRESENTATIONS,
                AnalysisCapability.MODALITY_GATING,
            ),
            surfaces=tuple(surfaces),
            method_notes={
                "gating": "Gate values are a routing signal, not by themselves a causal explanation. Validate by suppressing/patching modalities.",
            },
        )
