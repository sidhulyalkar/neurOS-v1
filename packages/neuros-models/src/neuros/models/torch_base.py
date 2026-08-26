"""Shared PyTorch training/runtime behavior for inspectable neurOS decoders."""

from __future__ import annotations

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neuros.contracts import DecoderOutput
from neuros.models.base_model import BaseModel


class TorchDecoderModel(BaseModel):
    """Consistent training, inference, embeddings, and analysis access for PyTorch decoders."""

    def __init__(
        self,
        *,
        n_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int = 32,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2 for classification")
        if n_epochs <= 0 or batch_size <= 0:
            raise ValueError("n_epochs and batch_size must be positive")
        self.n_classes = int(n_classes)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.device_spec = str(device)
        self.random_state = int(random_state)
        self.model: Any | None = None
        self.device: Any | None = None
        self.training_history: list[dict[str, float]] = []

    @staticmethod
    def _torch() -> tuple[Any, Any]:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for this model. Install with `pip install neuros-models[pytorch]`."
            ) from exc
        return torch, nn

    def _resolve_device(self, torch: Any) -> Any:
        if self.device_spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_spec)

    @abstractmethod
    def _build_model(self) -> Any:
        ...

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.isfinite(arr).all():
            raise ValueError("X contains NaN or infinite values")
        return arr

    def _ensure_model(self) -> Any:
        torch, _ = self._torch()
        if self.model is None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
            self.device = self._resolve_device(torch)
            self.model = self._build_model().to(self.device)
        return self.model

    def _tensor(self, X: np.ndarray) -> Any:
        torch, _ = self._torch()
        arr = self._validate_X(X)
        if self.device is None:
            self._ensure_model()
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def analysis_model(self) -> Any:
        return self._ensure_model()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        torch, nn = self._torch()
        X_arr = self._validate_X(X)
        y_arr = np.asarray(y, dtype=np.int64)
        if y_arr.ndim != 1 or len(y_arr) != len(X_arr):
            raise ValueError("y must be one-dimensional and aligned with X")
        if y_arr.size and (y_arr.min() < 0 or y_arr.max() >= self.n_classes):
            raise ValueError(f"labels must be in [0, {self.n_classes - 1}]")

        model = self._ensure_model()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_state)
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(X_arr, dtype=torch.float32),
            torch.as_tensor(y_arr, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.training_history = []
        for _epoch in range(self.n_epochs):
            model.train()
            total_loss = 0.0
            total = 0
            correct = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(batch_x)
                correct += int((logits.argmax(dim=1) == batch_y).sum().detach().cpu())
                total += len(batch_x)
            self.training_history.append(
                {
                    "loss": total_loss / max(total, 1),
                    "accuracy": correct / max(total, 1),
                }
            )
        self.is_trained = True

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

    def snapshot_state(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Capture exact parameters, buffers, RNG state, and training provenance.

        The returned object is a data-only ``TorchDecoderStateSnapshot``. Its
        ``parameter_state_sha256`` preserves the existing longitudinal learned-
        state hash semantics; ``learning_state_sha256`` additionally binds RNG
        state for exact stochastic fine-tuning rollback.
        """

        from neuros.models.artifacts import snapshot_torch_decoder_state

        return snapshot_torch_decoder_state(self, metadata=metadata)

    def restore_state(self, snapshot: Any) -> None:
        """Restore a compatible snapshot after validating the complete state geometry."""

        from neuros.models.artifacts import restore_torch_decoder_state

        restore_torch_decoder_state(self, snapshot)

    def export_artifact(
        self,
        output: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write a non-pickle JSON + NumPy decoder state artifact directory."""

        from neuros.models.artifacts import write_torch_decoder_artifact

        return write_torch_decoder_artifact(
            self,
            output,
            metadata=metadata,
            overwrite=overwrite,
        )

    def infer(self, X: np.ndarray) -> DecoderOutput:
        """Run one representation pass and return logits/probabilities/embedding together."""

        self._require_trained()
        torch, _ = self._torch()
        model = self._ensure_model()
        model.eval()
        started_ns = time.perf_counter_ns()
        with torch.no_grad():
            tensor = self._tensor(X)
            features = model.forward_features(tensor)
            if not hasattr(model, "classifier"):
                raise RuntimeError("Inspectable torch decoders must expose a classifier module")
            logits = model.classifier(features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
        elapsed_ns = time.perf_counter_ns() - started_ns

        preds_np = predictions.detach().cpu().numpy()
        probs_np = probabilities.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        features_np = features.detach().cpu().numpy()
        manifest = self.analysis_manifest()
        prediction: Any = preds_np.reshape(-1)[0].item() if preds_np.size == 1 else preds_np
        return DecoderOutput(
            prediction=prediction,
            confidence=float(probs_np[0].max()) if len(probs_np) else None,
            probabilities=probs_np[0] if len(probs_np) else probs_np,
            logits=logits_np[0] if len(logits_np) else logits_np,
            embedding=features_np[0] if len(features_np) else features_np,
            model_id=type(self).__name__,
            model_version=self.model_version,
            inference_time_ns=elapsed_ns,
            metadata={
                "architecture": manifest.architecture,
                "backend": manifest.backend,
                "analysis_manifest": manifest.fingerprint(),
                "mechint_ready": manifest.mechint_ready,
            },
        )

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        self._require_trained()
        torch, _ = self._torch()
        model = self._ensure_model()
        model.eval()
        with torch.no_grad():
            logits = model(self._tensor(X))
        return logits.detach().cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch, _ = self._torch()
        logits = torch.as_tensor(self.predict_logits(X))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict_logits(X)).argmax(axis=1)

    def encode(self, X: np.ndarray) -> np.ndarray:
        self._require_trained()
        torch, _ = self._torch()
        model = self._ensure_model()
        if not hasattr(model, "forward_features"):
            raise RuntimeError(f"{type(model).__name__} does not expose forward_features")
        model.eval()
        with torch.no_grad():
            features = model.forward_features(self._tensor(X))
        if not isinstance(features, torch.Tensor):
            raise TypeError("forward_features must return a tensor")
        return features.detach().cpu().numpy()
