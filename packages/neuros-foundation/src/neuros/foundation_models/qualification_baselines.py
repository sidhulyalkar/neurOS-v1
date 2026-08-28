"""Reference external methods for Neural System Qualification v1.

These adapters exist to prove that the NSQ referee can qualify maintained
upstream implementations without asking researchers to adopt neurOS training
code. They deliberately do not copy MNE or Braindecode algorithms.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .qualification import ExternalDecoderMethodSpec, ExternalLearnedState


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            f"optional NSQ reference method requires installed distribution {distribution!r}"
        ) from exc


def _frozen_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    options = dict(value)
    try:
        payload = json.dumps(
            options,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("reference-method options must be finite JSON values") from exc
    del payload
    return MappingProxyType(options)


def _torch_state_sha256(module: Any) -> str:
    """Hash exact inference-relevant tensor/buffer state without pickle."""

    state = module.state_dict()
    digest = hashlib.sha256()
    digest.update(b"neuros.external-torch-state.v1\0")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class _MNECSPLDADecoder:
    def __init__(self, *, n_components: int) -> None:
        self.n_components = int(n_components)
        self._pipeline: Any | None = None
        self._classes: tuple[str, ...] = ()

    def _require_pipeline(self) -> Any:
        if self._pipeline is None:
            raise RuntimeError("MNE CSP+LDA decoder has not been fitted")
        return self._pipeline

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from mne.decoding import CSP
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
            from sklearn.pipeline import make_pipeline
        except ImportError as exc:  # pragma: no cover - optional integration lane
            raise ImportError("MNE CSP+LDA requires MNE and scikit-learn") from exc

        array = np.asarray(X)
        labels = np.asarray(y).astype(str)
        if array.ndim != 3:
            raise ValueError("MNE CSP+LDA expects X=(sample, channel, time)")
        if labels.ndim != 1 or len(labels) != len(array):
            raise ValueError("MNE CSP+LDA labels must align with X")
        if not np.isfinite(array).all():
            raise ValueError("MNE CSP+LDA refuses non-finite neural input")
        classes = tuple(sorted(np.unique(labels).tolist()))
        if len(classes) < 2:
            raise ValueError("MNE CSP+LDA requires at least two classes")

        # Keep the reference intentionally boring and aligned with the existing
        # neurOS longitudinal baseline: upstream MNE CSP followed by sklearn LDA.
        pipeline = make_pipeline(
            CSP(n_components=self.n_components, reg=None),
            LDA(),
        )
        pipeline.fit(array, labels)
        self._pipeline = pipeline
        self._classes = classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._require_pipeline().predict(np.asarray(X))).astype(str)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._require_pipeline().predict_proba(np.asarray(X)),
            dtype=np.float64,
        )

    def probability_class_labels(self) -> tuple[str, ...]:
        pipeline = self._require_pipeline()
        estimator = pipeline.steps[-1][1]
        return tuple(str(value) for value in estimator.classes_)

    def learned_state(self) -> ExternalLearnedState:
        # MNE/sklearn do not currently have a qualified tensor-only NSQ state
        # serializer. Preserve scientific comparison without manufacturing a
        # strong checkpoint identity through pickle/joblib.
        self._require_pipeline()
        return ExternalLearnedState(
            state_identity_kind="opaque_unverified",
            metadata={
                "reason": "mne_sklearn_state_serializer_not_qualified",
                "n_components": self.n_components,
            },
        )


@dataclass(frozen=True, slots=True)
class MNECSPLDAFactory:
    """Trusted-code factory for upstream MNE CSP + sklearn LDA."""

    n_components: int = 8
    source_reference: str = "MNE CSP + scikit-learn LinearDiscriminantAnalysis"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("MNECSPLDAFactory schema_version must be 1")
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, int):
            raise ValueError("n_components must be an integer without coercion")
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        mne_version = _package_version("mne")
        sklearn_version = _package_version("scikit-learn")
        return ExternalDecoderMethodSpec(
            method_id="mne-csp-lda",
            implementation="mne.decoding.CSP+sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            implementation_version=f"mne={mne_version};scikit-learn={sklearn_version}",
            input_axes=("sample", "channel", "time"),
            probability_semantics="uncalibrated_probability",
            target_adaptation_mode="none",
            source_reference=self.source_reference,
            metadata={
                "csp_n_components": self.n_components,
                "csp_reg": None,
                "lda": "default",
                "hidden_preprocessing": False,
            },
        )

    def create(self) -> _MNECSPLDADecoder:
        return _MNECSPLDADecoder(n_components=self.n_components)


class _UpstreamBraindecodeDecoder:
    def __init__(
        self,
        *,
        model_name: str,
        sample_rate_hz: float | None,
        model_options: Mapping[str, Any],
        learning_rate: float,
        weight_decay: float,
        n_epochs: int,
        batch_size: int,
        device: str,
        random_state: int,
    ) -> None:
        self.model_name = model_name
        self.sample_rate_hz = sample_rate_hz
        self.model_options = dict(model_options)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self._classifier: Any | None = None
        self._module: Any | None = None
        self._classes: tuple[str, ...] = ()

    def _require_classifier(self) -> Any:
        if self._classifier is None:
            raise RuntimeError("upstream Braindecode decoder has not been fitted")
        return self._classifier

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import torch
            from braindecode import EEGClassifier
            import braindecode.models as models
        except ImportError as exc:  # pragma: no cover - optional integration lane
            raise ImportError("upstream Braindecode NSQ reference requires braindecode") from exc

        array = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y).astype(str)
        if array.ndim != 3:
            raise ValueError("Braindecode reference expects X=(sample, channel, time)")
        if labels.ndim != 1 or len(labels) != len(array):
            raise ValueError("Braindecode labels must align with X")
        if not np.isfinite(array).all():
            raise ValueError("Braindecode reference refuses non-finite neural input")
        classes = tuple(sorted(np.unique(labels).tolist()))
        if len(classes) < 2:
            raise ValueError("Braindecode reference requires at least two classes")
        mapping = {label: index for index, label in enumerate(classes)}
        encoded = np.asarray([mapping[value] for value in labels], dtype=np.int64)

        model_type = getattr(models, self.model_name, None)
        if model_type is None:
            # A pinned Braindecode installation that does not expose a requested
            # upstream architecture is a capability absence, not a failed model
            # run. NSQ maps ImportError to an explicit `unavailable` result row
            # so missing architectures remain visible without being confused
            # with numerical/training failures.
            raise ImportError(
                f"installed Braindecode does not expose model {self.model_name!r}"
            )
        kwargs: dict[str, Any] = {
            "n_chans": int(array.shape[1]),
            "n_outputs": len(classes),
            "n_times": int(array.shape[2]),
            **self.model_options,
        }
        if self.sample_rate_hz is not None:
            signature = inspect.signature(model_type)
            if "sfreq" in signature.parameters:
                kwargs["sfreq"] = self.sample_rate_hz

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        module = model_type(**kwargs)
        classifier = EEGClassifier(
            module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.learning_rate,
            optimizer__weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.n_epochs,
            train_split=None,
            device=self.device,
            classes=np.arange(len(classes)),
            verbose=0,
        )
        classifier.fit(array, encoded)
        self._module = module
        self._classifier = classifier
        self._classes = classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        encoded = np.asarray(
            self._require_classifier().predict(np.asarray(X, dtype=np.float32)),
            dtype=np.int64,
        )
        return np.asarray([self._classes[index] for index in encoded], dtype=str)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._require_classifier().predict_proba(np.asarray(X, dtype=np.float32)),
            dtype=np.float64,
        )

    def probability_class_labels(self) -> tuple[str, ...]:
        self._require_classifier()
        return self._classes

    def learned_state(self) -> ExternalLearnedState:
        classifier = self._require_classifier()
        module = getattr(classifier, "module_", None) or self._module
        if module is None:
            raise RuntimeError("Braindecode classifier has no fitted module state")
        return ExternalLearnedState(
            state_identity_kind="tensor_sha256",
            state_sha256=_torch_state_sha256(module),
            metadata={
                "state_scope": "upstream_model_state_dict_including_registered_buffers",
                "optimizer_state_included": False,
            },
        )


@dataclass(frozen=True, slots=True)
class UpstreamBraindecodeFactory:
    """Direct upstream Braindecode model + EEGClassifier NSQ factory."""

    model_name: str = "EEGNet"
    sample_rate_hz: float | None = None
    model_options: Mapping[str, Any] = field(default_factory=dict)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    n_epochs: int = 1
    batch_size: int = 32
    device: str = "cpu"
    random_state: int = 0
    source_reference: str = "Braindecode upstream model + EEGClassifier"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("UpstreamBraindecodeFactory schema_version must be 1")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.sample_rate_hz is not None:
            rate = float(self.sample_rate_hz)
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("sample_rate_hz must be finite and positive")
            object.__setattr__(self, "sample_rate_hz", rate)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.n_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("n_epochs and batch_size must be positive")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer without coercion")
        object.__setattr__(self, "model_name", self.model_name.strip())
        object.__setattr__(self, "model_options", _frozen_json_mapping(self.model_options))

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        braindecode_version = _package_version("braindecode")
        torch_version = _package_version("torch")
        return ExternalDecoderMethodSpec(
            method_id=f"braindecode-{self.model_name.lower()}",
            implementation=f"braindecode.models.{self.model_name}+braindecode.EEGClassifier",
            implementation_version=(
                f"braindecode={braindecode_version};torch={torch_version}"
            ),
            input_axes=("sample", "channel", "time"),
            probability_semantics="uncalibrated_softmax",
            target_adaptation_mode="none",
            source_reference=self.source_reference,
            metadata={
                "model_name": self.model_name,
                "sample_rate_hz": self.sample_rate_hz,
                "model_options": dict(self.model_options),
                "criterion": "torch.nn.CrossEntropyLoss",
                "optimizer": "torch.optim.AdamW",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "device": self.device,
                "random_state": self.random_state,
                "train_split": None,
                "hidden_preprocessing": False,
                "neuros_model_wrapper_used": False,
            },
        )

    def create(self) -> _UpstreamBraindecodeDecoder:
        return _UpstreamBraindecodeDecoder(
            model_name=self.model_name,
            sample_rate_hz=self.sample_rate_hz,
            model_options=self.model_options,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            device=self.device,
            random_state=self.random_state,
        )


__all__ = [
    "MNECSPLDAFactory",
    "UpstreamBraindecodeFactory",
]
