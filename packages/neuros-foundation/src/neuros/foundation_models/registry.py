"""Registry and fail-closed execution adapters for neural foundation models."""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC
from collections.abc import Iterable, Mapping
from typing import Any

from .catalog import DEFAULT_MODEL_CARDS
from .schema import (
    AccessLevel,
    AdapterAvailability,
    FoundationModelCard,
    IntegrationLevel,
    ModelTask,
    NeuralModality,
)


class FoundationModelError(RuntimeError):
    """Base error for foundation-model interoperability failures."""


class AdapterUnavailableError(FoundationModelError):
    """Raised when a model is known to the catalog but cannot execute locally."""


class UnsupportedCapabilityError(FoundationModelError):
    """Raised when an adapter does not implement a requested operation."""


class FoundationAdapter(ABC):
    """Small execution surface around an upstream model implementation.

    Adapters intentionally do not force every neural model into ``encode`` and
    ``predict``. A reconstruction model such as ZUNA and an autoregressive
    spiking decoder are different tools. ``capabilities`` makes that difference
    explicit while the registry provides a common discovery/loading layer.
    """

    model_id: str
    package: str | None = None
    capabilities: tuple[str, ...] = ()

    def availability(self) -> AdapterAvailability:
        if self.package is None:
            return AdapterAvailability(
                model_id=self.model_id,
                available=True,
                reason="adapter has no external package requirement",
                capabilities=self.capabilities,
            )
        available = importlib.util.find_spec(self.package) is not None
        return AdapterAvailability(
            model_id=self.model_id,
            available=available,
            reason=(
                f"Python package '{self.package}' is installed"
                if available
                else f"Python package '{self.package}' is not installed"
            ),
            package=self.package,
            capabilities=self.capabilities,
        )

    def require_available(self) -> None:
        status = self.availability()
        if not status.available:
            raise AdapterUnavailableError(
                f"{self.model_id} is not runnable: {status.reason}. "
                "Use registry.get_card(...) for metadata without installing the model."
            )


class ZunaAdapter(FoundationAdapter):
    """Thin adapter over Zyphra's public ``zuna`` package."""

    model_id = "zuna-1.1"
    package = "zuna"
    capabilities = ("reconstruct_fif", "denoise", "impute", "upsample")

    def reconstruct_fif(self, **kwargs: Any) -> Any:
        self.require_available()
        module = importlib.import_module("zuna")
        fn = getattr(module, "reconstruct_fif", None)
        if fn is None:
            raise UnsupportedCapabilityError(
                "Installed zuna package does not expose reconstruct_fif; "
                "check the upstream ZUNA version."
            )
        return fn(**kwargs)


class NeuroFMXAdapter(FoundationAdapter):
    """Bridge to the neurOS-native ``neuros-neurofm`` research package."""

    model_id = "neuros-neurofmx"
    package = "neuros_neurofm"
    capabilities = ("load", "raw_model", "encode_if_supported")

    def __init__(self) -> None:
        self._model: Any | None = None

    def load(self, *, multitask: bool = False, **kwargs: Any) -> Any:
        self.require_available()
        module = importlib.import_module("neuros_neurofm")
        class_name = "NeuroFMXMultiTask" if multitask else "NeuroFMXComplete"
        model_class = getattr(module, class_name, None)
        if model_class is None:
            raise AdapterUnavailableError(
                f"neuros_neurofm does not export {class_name}; install a compatible monorepo version."
            )
        self._model = model_class(**kwargs)
        return self._model

    @property
    def raw_model(self) -> Any:
        if self._model is None:
            raise AdapterUnavailableError("Call NeuroFMXAdapter.load(...) before accessing raw_model")
        return self._model

    def encode(self, data: Any, **kwargs: Any) -> Any:
        model = self.raw_model
        for name in ("encode", "forward_features", "embed"):
            method = getattr(model, name, None)
            if callable(method):
                return method(data, **kwargs)
        raise UnsupportedCapabilityError(
            "This NeuroFM-X build does not expose a stable encode/forward_features/embed method. "
            "Use adapter.raw_model with the neuros-neurofm API for this experimental model version."
        )


class CallableAdapter(FoundationAdapter):
    """Simple extension hook for local or third-party model integrations."""

    def __init__(
        self,
        model_id: str,
        *,
        package: str | None = None,
        capabilities: Iterable[str] = (),
        operations: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.package = package
        self.capabilities = tuple(capabilities)
        self._operations = dict(operations or {})

    def call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        self.require_available()
        fn = self._operations.get(operation)
        if fn is None:
            raise UnsupportedCapabilityError(
                f"Adapter '{self.model_id}' does not implement operation '{operation}'"
            )
        return fn(*args, **kwargs)


class ModelRegistry:
    """Searchable catalog plus explicit execution adapters."""

    def __init__(self, cards: Iterable[FoundationModelCard] = DEFAULT_MODEL_CARDS) -> None:
        card_list = tuple(cards)
        self._cards = {card.id: card for card in card_list}
        if len(self._cards) != len(card_list):
            raise ValueError("Foundation model card IDs must be unique")
        self._adapters: dict[str, FoundationAdapter] = {}

    def register_card(self, card: FoundationModelCard, *, replace: bool = False) -> None:
        if card.id in self._cards and not replace:
            raise ValueError(f"Model card already registered: {card.id}")
        self._cards[card.id] = card

    def register_adapter(self, adapter: FoundationAdapter, *, replace: bool = False) -> None:
        if adapter.model_id not in self._cards:
            raise KeyError(
                f"Register a FoundationModelCard for '{adapter.model_id}' before its adapter"
            )
        if adapter.model_id in self._adapters and not replace:
            raise ValueError(f"Adapter already registered: {adapter.model_id}")
        self._adapters[adapter.model_id] = adapter

    def get_card(self, model_id: str) -> FoundationModelCard:
        try:
            return self._cards[model_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._cards))
            raise KeyError(f"Unknown foundation model '{model_id}'. Known models: {known}") from exc

    def cards(self) -> tuple[FoundationModelCard, ...]:
        return tuple(sorted(self._cards.values(), key=lambda card: (-card.year, card.name.lower())))

    def filter(
        self,
        *,
        modality: NeuralModality | str | None = None,
        task: ModelTask | str | None = None,
        access: AccessLevel | str | None = None,
        integration: IntegrationLevel | str | None = None,
        min_year: int | None = None,
        tags: Iterable[str] = (),
        runnable: bool | None = None,
    ) -> tuple[FoundationModelCard, ...]:
        modality_value = NeuralModality(modality) if modality is not None else None
        task_value = ModelTask(task) if task is not None else None
        access_value = AccessLevel(access) if access is not None else None
        integration_value = IntegrationLevel(integration) if integration is not None else None
        tags = tuple(tags)

        selected: list[FoundationModelCard] = []
        for card in self.cards():
            if modality_value is not None and not card.supports_modality(modality_value):
                continue
            if task_value is not None and not card.supports_task(task_value):
                continue
            if access_value is not None and card.access != access_value:
                continue
            if integration_value is not None and card.integration != integration_value:
                continue
            if min_year is not None and card.year < min_year:
                continue
            if tags and not card.has_tags(tags):
                continue
            if runnable is not None and self.availability(card.id).available != runnable:
                continue
            selected.append(card)
        return tuple(selected)

    def availability(self, model_id: str) -> AdapterAvailability:
        self.get_card(model_id)
        adapter = self._adapters.get(model_id)
        if adapter is None:
            return AdapterAvailability(
                model_id=model_id,
                available=False,
                reason="catalog entry has no neurOS execution adapter yet",
            )
        return adapter.availability()

    def adapter(self, model_id: str) -> FoundationAdapter:
        card = self.get_card(model_id)
        adapter = self._adapters.get(model_id)
        if adapter is None:
            raise AdapterUnavailableError(
                f"'{model_id}' is cataloged as {card.integration.value} but has no execution adapter. "
                "This is intentional: neurOS never substitutes mock/random outputs for an unavailable upstream model."
            )
        adapter.require_available()
        return adapter

    def compare(
        self,
        model_ids: Iterable[str],
        *,
        fields: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        field_names = tuple(
            fields
            or (
                "id",
                "name",
                "organization",
                "year",
                "modalities",
                "tasks",
                "architecture",
                "pretraining_objective",
                "input_geometry",
                "pretraining_scale",
                "access",
                "integration",
                "license",
            )
        )
        rows: list[dict[str, Any]] = []
        for model_id in model_ids:
            data = self.get_card(model_id).to_dict()
            row = {name: data.get(name) for name in field_names}
            row["runnable"] = self.availability(model_id).available
            rows.append(row)
        return rows


def build_default_registry() -> ModelRegistry:
    registry = ModelRegistry(DEFAULT_MODEL_CARDS)
    registry.register_adapter(ZunaAdapter())
    registry.register_adapter(NeuroFMXAdapter())
    return registry


DEFAULT_REGISTRY = build_default_registry()
