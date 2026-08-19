"""NNsight adapter using native trace-time output access and assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .base import ModelAdapter
from .pytorch import ModelCall


@dataclass(frozen=True, slots=True)
class NNsightTarget:
    """Address one NNsight module output, optionally selecting a tuple element.

    String form uses ``module.path::index`` for tuple outputs, for example
    ``transformer.h.0::0``. Tensor-valued module outputs use ``module.path``.
    """

    label: str
    module_path: str
    output_index: int | None = None

    @classmethod
    def parse(cls, value: str) -> NNsightTarget:
        label = str(value)
        if "::" not in label:
            return cls(label=label, module_path=label)
        module_path, raw_index = label.rsplit("::", 1)
        if not module_path:
            raise ValueError(f"invalid NNsight target {label!r}")
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"invalid NNsight output selector in {label!r}") from exc
        if index < 0:
            raise ValueError("NNsight output selectors must be non-negative")
        return cls(label=label, module_path=module_path, output_index=index)


def _extract_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    logits = getattr(value, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _extract_tensor(item)
            except TypeError:
                continue
    raise TypeError(
        "NNsight adapter expected a saved tensor, an object with tensor .logits, "
        "or a tuple/list containing a tensor"
    )


def _resolve_envoy(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if not part:
            raise ValueError(f"invalid empty segment in NNsight path {path!r}")
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _selected_output(node: Any, target: NNsightTarget) -> Any:
    value = node.output
    if target.output_index is not None:
        value = value[target.output_index]
    return value


class NNsightAdapter(ModelAdapter):
    """Normalize NNsight tracing and activation replacement to ``ModelAdapter``.

    This module does not import NNsight. It accepts any current NNsight wrapper
    exposing ``trace(...)`` plus module Envoys with ``.output`` and ``.save()``.
    Paths are ordered by the wrapped PyTorch module graph when ``_module`` is
    available, reducing accidental out-of-order trace access.
    """

    def __init__(
        self,
        model: Any,
        *,
        output_path: str | None = None,
        trace_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(getattr(model, "trace", None)):
            raise TypeError("NNsight model must expose callable trace()")
        self.model = model
        self.output_path = output_path
        self.trace_kwargs = dict(trace_kwargs or {})

    def _trace(self, inputs: Any) -> Any:
        if isinstance(inputs, ModelCall):
            overlap = set(inputs.kwargs) & set(self.trace_kwargs)
            if overlap:
                raise ValueError(f"ModelCall and trace_kwargs collide on keys: {sorted(overlap)}")
            kwargs = {**dict(inputs.kwargs), **self.trace_kwargs}
            return self.model.trace(*inputs.args, **kwargs)
        return self.model.trace(inputs, **self.trace_kwargs)

    def _order_targets(self, targets: Sequence[NNsightTarget]) -> tuple[NNsightTarget, ...]:
        underlying = getattr(self.model, "_module", None)
        named_modules = getattr(underlying, "named_modules", None)
        if not callable(named_modules):
            return tuple(targets)
        order = {name: index for index, (name, _) in enumerate(named_modules())}
        original = {target.label: index for index, target in enumerate(targets)}
        return tuple(
            sorted(
                targets,
                key=lambda target: (
                    order.get(target.module_path, len(order) + original[target.label]),
                    original[target.label],
                ),
            )
        )

    def _save_model_output(self) -> Any:
        if self.output_path is None:
            return self.model.output.save()
        target = NNsightTarget.parse(self.output_path)
        node = _resolve_envoy(self.model, target.module_path)
        return _selected_output(node, target).save()

    def forward(self, inputs: Any) -> torch.Tensor:
        with torch.no_grad(), self._trace(inputs):
            saved_output = self._save_model_output()
        return _extract_tensor(saved_output)

    def capture_outputs(self, inputs: Any, paths: Sequence[str]) -> dict[str, torch.Tensor]:
        targets = self._order_targets(tuple(NNsightTarget.parse(path) for path in paths))
        saved = {}
        with torch.no_grad(), self._trace(inputs):
            for target in targets:
                node = _resolve_envoy(self.model, target.module_path)
                saved[target.label] = _selected_output(node, target).save()
        captured = {}
        for target in targets:
            value = _extract_tensor(saved[target.label])
            captured[target.label] = value.detach().clone()
        return captured

    def forward_with_replacements(
        self,
        inputs: Any,
        replacements: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        targets = self._order_targets(
            tuple(NNsightTarget.parse(path) for path in replacements)
        )
        with torch.no_grad(), self._trace(inputs):
            for target in targets:
                replacement = replacements[target.label]
                if not isinstance(replacement, torch.Tensor):
                    raise TypeError(f"replacement for {target.label!r} must be a tensor")
                node = _resolve_envoy(self.model, target.module_path)
                selected = _selected_output(node, target)
                selected[:] = replacement
            saved_output = self._save_model_output()
        return _extract_tensor(saved_output)

    def model_fingerprint_payload(self) -> Any | None:
        underlying = getattr(self.model, "_module", None)
        state_dict = getattr(underlying, "state_dict", None)
        if not callable(state_dict):
            return None
        return {
            "module": type(underlying).__module__,
            "qualname": type(underlying).__qualname__,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in sorted(state_dict().items())
                if isinstance(tensor, torch.Tensor)
            },
        }
