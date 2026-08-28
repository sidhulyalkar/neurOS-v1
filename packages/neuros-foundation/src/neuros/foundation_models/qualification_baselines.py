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


class _RiemannianTangentLogRegDecoder:
    def __init__(
        self,
        *,
        covariance_estimator: str,
        tangent_metric: str,
        logistic_c: float,
        max_iter: int,
    ) -> None:
        self.covariance_estimator = covariance_estimator
        self.tangent_metric = tangent_metric
        self.logistic_c = logistic_c
        self.max_iter = max_iter
        self._pipeline: Any | None = None

    def _require_pipeline(self) -> Any:
        if self._pipeline is None:
            raise RuntimeError("pyRiemann RG+LR decoder has not been fitted")
        return self._pipeline

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
        except ImportError as exc:  # pragma: no cover - optional integration lane
            raise ImportError(
                "pyRiemann RG+LR requires pyriemann and scikit-learn"
            ) from exc

        array = np.asarray(X)
        labels = np.asarray(y).astype(str)
        if array.ndim != 3:
            raise ValueError("pyRiemann RG+LR expects X=(sample, channel, time)")
        if labels.ndim != 1 or len(labels) != len(array):
            raise ValueError("pyRiemann RG+LR labels must align with X")
        if not np.isfinite(array).all():
            raise ValueError("pyRiemann RG+LR refuses non-finite neural input")
        if len(np.unique(labels)) < 2:
            raise ValueError("pyRiemann RG+LR requires at least two classes")

        # Standard MOABB-style source-only Riemannian baseline. tsupdate=False
        # makes the tangent reference a fitted training state rather than a
        # transductive function of final-assessment batch composition.
        pipeline = make_pipeline(
            Covariances(estimator=self.covariance_estimator),
            TangentSpace(metric=self.tangent_metric, tsupdate=False),
            LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                C=self.logistic_c,
                max_iter=self.max_iter,
            ),
        )
        pipeline.fit(array, labels)
        self._pipeline = pipeline

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._require_pipeline().predict(np.asarray(X))).astype(str)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._require_pipeline().predict_proba(np.asarray(X)),
            dtype=np.float64,
        )

    def probability_class_labels(self) -> tuple[str, ...]:
        estimator = self._require_pipeline().steps[-1][1]
        return tuple(str(value) for value in estimator.classes_)

    def learned_state(self) -> ExternalLearnedState:
        self._require_pipeline()
        return ExternalLearnedState(
            state_identity_kind="opaque_unverified",
            metadata={
                "reason": "pyriemann_sklearn_state_serializer_not_qualified",
                "covariance_estimator": self.covariance_estimator,
                "tangent_metric": self.tangent_metric,
                "tangent_space_update": False,
                "logistic_c": self.logistic_c,
                "max_iter": self.max_iter,
            },
        )


@dataclass(frozen=True, slots=True)
class RiemannianTangentLogRegFactory:
    """Upstream pyRiemann covariance/tangent-space + sklearn LR baseline."""

    covariance_estimator: str = "scm"
    tangent_metric: str = "riemann"
    logistic_c: float = 1.0
    max_iter: int = 1000
    source_reference: str = (
        "pyRiemann Covariances + TangentSpace + scikit-learn LogisticRegression"
    )
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("RiemannianTangentLogRegFactory schema_version must be 1")
        if not isinstance(self.covariance_estimator, str) or not self.covariance_estimator.strip():
            raise ValueError("covariance_estimator must be non-empty")
        if not isinstance(self.tangent_metric, str) or not self.tangent_metric.strip():
            raise ValueError("tangent_metric must be non-empty")
        c_value = float(self.logistic_c)
        if not np.isfinite(c_value) or c_value <= 0:
            raise ValueError("logistic_c must be finite and positive")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int):
            raise ValueError("max_iter must be an integer without coercion")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        object.__setattr__(self, "covariance_estimator", self.covariance_estimator.strip())
        object.__setattr__(self, "tangent_metric", self.tangent_metric.strip())
        object.__setattr__(self, "logistic_c", c_value)

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        pyriemann_version = _package_version("pyriemann")
        sklearn_version = _package_version("scikit-learn")
        return ExternalDecoderMethodSpec(
            method_id="pyriemann-rg-lr",
            implementation=(
                "pyriemann.estimation.Covariances+"
                "pyriemann.tangentspace.TangentSpace+"
                "sklearn.linear_model.LogisticRegression"
            ),
            implementation_version=(
                f"pyriemann={pyriemann_version};scikit-learn={sklearn_version}"
            ),
            input_axes=("sample", "channel", "time"),
            probability_semantics="uncalibrated_probability",
            target_adaptation_mode="none",
            source_reference=self.source_reference,
            metadata={
                "covariance_estimator": self.covariance_estimator,
                "tangent_metric": self.tangent_metric,
                "tangent_space_update": False,
                "logistic_solver": "lbfgs",
                "logistic_penalty": "l2",
                "logistic_c": self.logistic_c,
                "logistic_max_iter": self.max_iter,
                "hidden_preprocessing": False,
                "transductive_evaluation_batch_update": False,
            },
        )

    def create(self) -> _RiemannianTangentLogRegDecoder:
        return _RiemannianTangentLogRegDecoder(
            covariance_estimator=self.covariance_estimator,
            tangent_metric=self.tangent_metric,
            logistic_c=self.logistic_c,
            max_iter=self.max_iter,
        )


class _RecordingValidSplit:
    """Direct skorch ValidSplit wrapper that preserves the exact relative membership."""

    def __init__(self, *, fraction: float, random_state: int) -> None:
        self.fraction = float(fraction)
        self.random_state = int(random_state)
        self.train_relative_indices: tuple[int, ...] = ()
        self.validation_relative_indices: tuple[int, ...] = ()

    def __call__(self, dataset: Any, y: Any = None, groups: Any = None) -> tuple[Any, Any]:
        try:
            from skorch.dataset import ValidSplit
        except ImportError as exc:  # pragma: no cover - braindecode pulls skorch
            raise ImportError("Braindecode validation authority requires skorch") from exc

        splitter = ValidSplit(
            self.fraction,
            stratified=False,
            random_state=self.random_state,
        )
        train_dataset, validation_dataset = splitter(dataset, y=y, groups=groups)
        train_indices = getattr(train_dataset, "indices", None)
        validation_indices = getattr(validation_dataset, "indices", None)
        if train_indices is None or validation_indices is None:
            raise RuntimeError(
                "skorch ValidSplit did not expose relative train/validation indices"
            )
        train = tuple(int(value) for value in np.asarray(train_indices).tolist())
        validation = tuple(
            int(value) for value in np.asarray(validation_indices).tolist()
        )
        if not train or not validation:
            raise RuntimeError("Braindecode validation split must keep non-empty partitions")
        if set(train).intersection(validation):
            raise RuntimeError("Braindecode train and validation partitions overlap")
        expected = set(range(len(dataset)))
        if set(train).union(validation) != expected:
            raise RuntimeError(
                "Braindecode train/validation membership does not partition the fit set"
            )
        self.train_relative_indices = train
        self.validation_relative_indices = validation
        return train_dataset, validation_dataset


def _relative_indices_sha256(role: str, values: tuple[int, ...]) -> str:
    payload = json.dumps(
        {
            "schema": "neuros.external_fit_relative_indices.v1",
            "role": role,
            "indices": list(values),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _UpstreamBraindecodeDecoder:
    def __init__(
        self,
        *,
        model_name: str,
        sample_rate_hz: float | None,
        model_options: Mapping[str, Any],
        optimizer_name: str,
        learning_rate: float,
        weight_decay: float,
        n_epochs: int,
        batch_size: int,
        device: str,
        random_state: int,
        validation_fraction: float | None,
        validation_seed: int | None,
        early_stopping_patience: int | None,
        early_stopping_threshold: float,
        restore_best: bool,
    ) -> None:
        self.model_name = model_name
        self.sample_rate_hz = sample_rate_hz
        self.model_options = dict(model_options)
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.validation_seed = validation_seed
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.restore_best = restore_best
        self._classifier: Any | None = None
        self._module: Any | None = None
        self._classes: tuple[str, ...] = ()
        self._training_metadata: dict[str, Any] = {}

    def _require_classifier(self) -> Any:
        if self._classifier is None:
            raise RuntimeError("upstream Braindecode decoder has not been fitted")
        return self._classifier

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import random
            import torch
            from braindecode import EEGClassifier
            import braindecode.models as models
            from skorch.callbacks import EarlyStopping
            from skorch.utils import noop
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

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        optimizer_type = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }[self.optimizer_name]

        train_split: Any = None
        callbacks: list[Any] = []
        recording_split: _RecordingValidSplit | None = None
        if self.validation_fraction is not None:
            assert self.validation_seed is not None
            assert self.early_stopping_patience is not None
            recording_split = _RecordingValidSplit(
                fraction=self.validation_fraction,
                random_state=self.validation_seed,
            )
            train_split = recording_split
            callbacks.append(
                (
                    "nsq_early_stopping",
                    EarlyStopping(
                        monitor="valid_loss",
                        patience=self.early_stopping_patience,
                        threshold=self.early_stopping_threshold,
                        threshold_mode="rel",
                        lower_is_better=True,
                        sink=noop,
                        load_best=self.restore_best,
                    ),
                )
            )

        module = model_type(**kwargs)
        classifier = EEGClassifier(
            module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=optimizer_type,
            optimizer__lr=self.learning_rate,
            optimizer__weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.n_epochs,
            train_split=train_split,
            callbacks=callbacks,
            device=self.device,
            classes=np.arange(len(classes)),
            verbose=0,
        )
        classifier.fit(array, encoded)

        metadata: dict[str, Any] = {
            "fit_samples": int(len(array)),
            "model_seed": self.random_state,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "epochs_completed": int(len(classifier.history)),
        }
        if recording_split is not None:
            train_relative = recording_split.train_relative_indices
            validation_relative = recording_split.validation_relative_indices
            history = list(classifier.history)
            valid_losses = [float(row["valid_loss"]) for row in history]
            best_offset = int(np.argmin(np.asarray(valid_losses, dtype=np.float64)))
            metadata.update(
                {
                    "validation_policy": "skorch.ValidSplit",
                    "validation_fraction": self.validation_fraction,
                    "validation_stratified": False,
                    "validation_seed": self.validation_seed,
                    "validation_relative_indices": list(validation_relative),
                    "validation_relative_indices_sha256": _relative_indices_sha256(
                        "validation", validation_relative
                    ),
                    "train_relative_indices_sha256": _relative_indices_sha256(
                        "train", train_relative
                    ),
                    "training_samples_after_validation_split": len(train_relative),
                    "validation_samples": len(validation_relative),
                    "state_selection_monitor": "valid_loss",
                    "state_selection_rule": "minimum_observed_validation_loss",
                    "restore_best": self.restore_best,
                    "best_observed_epoch": int(history[best_offset]["epoch"]),
                    "best_observed_valid_loss": valid_losses[best_offset],
                    "stopped_epoch": int(history[-1]["epoch"]),
                    "early_stopping_patience": self.early_stopping_patience,
                    "early_stopping_threshold": self.early_stopping_threshold,
                    "final_assessment_used_for_state_selection": False,
                }
            )

        self._module = module
        self._classifier = classifier
        self._classes = classes
        self._training_metadata = metadata

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
                **self._training_metadata,
            },
        )


@dataclass(frozen=True, slots=True)
class UpstreamBraindecodeFactory:
    """Direct upstream Braindecode model + EEGClassifier NSQ factory.

    The generic adapter remains useful for integration probes, but when validation
    and early stopping are configured all selection happens strictly inside the
    NSQ-authorized fit set. Final-assessment observations are never exposed to
    this object by the NSQ referee.
    """

    model_name: str = "EEGNet"
    sample_rate_hz: float | None = None
    model_options: Mapping[str, Any] = field(default_factory=dict)
    optimizer_name: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    n_epochs: int = 1
    batch_size: int = 32
    device: str = "cpu"
    random_state: int = 0
    validation_fraction: float | None = None
    validation_seed: int | None = None
    early_stopping_patience: int | None = None
    early_stopping_threshold: float = 0.0
    restore_best: bool = False
    source_reference: str = "Braindecode upstream model + EEGClassifier"
    schema_version: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            raise ValueError("UpstreamBraindecodeFactory schema_version must be 2")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.optimizer_name not in {"Adam", "AdamW"}:
            raise ValueError("optimizer_name must be 'Adam' or 'AdamW'")
        if self.sample_rate_hz is not None:
            rate = float(self.sample_rate_hz)
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("sample_rate_hz must be finite and positive")
            object.__setattr__(self, "sample_rate_hz", rate)
        learning_rate = float(self.learning_rate)
        weight_decay = float(self.weight_decay)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.n_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("n_epochs and batch_size must be positive")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer without coercion")
        if self.validation_fraction is None:
            if self.validation_seed is not None or self.early_stopping_patience is not None:
                raise ValueError(
                    "validation_seed/patience require validation_fraction"
                )
            if self.restore_best:
                raise ValueError("restore_best requires validation_fraction")
        else:
            fraction = float(self.validation_fraction)
            if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
                raise ValueError("validation_fraction must lie strictly between zero and one")
            if isinstance(self.validation_seed, bool) or not isinstance(self.validation_seed, int):
                raise ValueError("validation_seed must be an integer when validation is enabled")
            if (
                isinstance(self.early_stopping_patience, bool)
                or not isinstance(self.early_stopping_patience, int)
                or self.early_stopping_patience <= 0
            ):
                raise ValueError("early_stopping_patience must be a positive integer")
            object.__setattr__(self, "validation_fraction", fraction)
        threshold = float(self.early_stopping_threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("early_stopping_threshold must be finite and non-negative")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "early_stopping_threshold", threshold)
        object.__setattr__(self, "model_name", self.model_name.strip())
        object.__setattr__(self, "model_options", _frozen_json_mapping(self.model_options))

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        braindecode_version = _package_version("braindecode")
        torch_version = _package_version("torch")
        skorch_version = _package_version("skorch")
        sklearn_version = _package_version("scikit-learn")
        validation = None
        if self.validation_fraction is not None:
            validation = {
                "implementation": "skorch.dataset.ValidSplit",
                "fraction": self.validation_fraction,
                "stratified": False,
                "seed": self.validation_seed,
            }
        return ExternalDecoderMethodSpec(
            method_id=f"braindecode-{self.model_name.lower()}",
            implementation=f"braindecode.models.{self.model_name}+braindecode.EEGClassifier",
            implementation_version=(
                f"braindecode={braindecode_version};torch={torch_version};"
                f"skorch={skorch_version};scikit-learn={sklearn_version}"
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
                "optimizer": f"torch.optim.{self.optimizer_name}",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "n_epochs_ceiling": self.n_epochs,
                "batch_size": self.batch_size,
                "device": self.device,
                "model_seed": self.random_state,
                "train_split": validation,
                "state_selection": (
                    None
                    if validation is None
                    else {
                        "monitor": "valid_loss",
                        "patience": self.early_stopping_patience,
                        "threshold": self.early_stopping_threshold,
                        "threshold_mode": "rel",
                        "restore_best": self.restore_best,
                    }
                ),
                "hidden_preprocessing": False,
                "final_assessment_used_for_state_selection": False,
                "neuros_model_wrapper_used": False,
            },
        )

    def create(self) -> _UpstreamBraindecodeDecoder:
        return _UpstreamBraindecodeDecoder(
            model_name=self.model_name,
            sample_rate_hz=self.sample_rate_hz,
            model_options=self.model_options,
            optimizer_name=self.optimizer_name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            device=self.device,
            random_state=self.random_state,
            validation_fraction=self.validation_fraction,
            validation_seed=self.validation_seed,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_threshold=self.early_stopping_threshold,
            restore_best=self.restore_best,
        )


__all__ = [
    "MNECSPLDAFactory",
    "RiemannianTangentLogRegFactory",
    "UpstreamBraindecodeFactory",
]
