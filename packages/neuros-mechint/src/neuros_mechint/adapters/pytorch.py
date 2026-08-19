"""Default adapter for ordinary ``torch.nn.Module`` models."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from .base import ModelAdapter


def _require_tensor_output(path: str, output: Any) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"component {path!r} returned {type(output).__name__}; "
            "the PyTorchAdapter currently supports tensor module outputs only"
        )
    return output


@dataclass(frozen=True, slots=True)
class ModelCall:
    """Explicit positional/keyword arguments for a model forward call.

    Ordinary tensor inputs continue to be passed as ``model(inputs)``. Use this
    wrapper when a model needs additional arguments such as an attention mask.
    """

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))


class PyTorchAdapter(ModelAdapter):
    """Trace and replace module outputs using temporary forward hooks."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def _modules(self) -> dict[str, nn.Module]:
        return dict(self.model.named_modules())

    def _validate_paths(self, paths: Sequence[str]) -> None:
        modules = self._modules()
        missing = [path for path in paths if path not in modules]
        if missing:
            raise KeyError(f"unknown module path(s): {missing}")

    def forward(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, ModelCall):
            output = self.model(*inputs.args, **dict(inputs.kwargs))
        else:
            output = self.model(inputs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("PyTorchAdapter expects the model's top-level output to be a tensor")
        return output

    @contextmanager
    def _hooks(self, registrations: Sequence[tuple[str, Any]]) -> Iterator[None]:
        handles = []
        modules = self._modules()
        try:
            for path, hook in registrations:
                handles.append(modules[path].register_forward_hook(hook))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def capture_outputs(self, inputs: Any, paths: Sequence[str]) -> dict[str, torch.Tensor]:
        paths = tuple(paths)
        self._validate_paths(paths)
        cache: dict[str, torch.Tensor] = {}
        registrations = []

        for path in paths:

            def _capture(
                module: nn.Module,
                args: tuple[Any, ...],
                output: Any,
                *,
                _path: str = path,
            ) -> None:
                del module, args
                cache[_path] = _require_tensor_output(_path, output).detach().clone()

            registrations.append((path, _capture))

        with self._hooks(registrations), torch.no_grad():
            self.forward(inputs)
        return cache

    def forward_with_replacements(
        self,
        inputs: Any,
        replacements: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        paths = tuple(replacements)
        self._validate_paths(paths)
        registrations = []

        for path, replacement in replacements.items():

            def _replace(
                module: nn.Module,
                args: tuple[Any, ...],
                output: Any,
                *,
                _path: str = path,
                _replacement: torch.Tensor = replacement,
            ) -> torch.Tensor:
                del module, args
                original = _require_tensor_output(_path, output)
                value = _replacement.to(device=original.device, dtype=original.dtype)
                if value.shape != original.shape:
                    raise ValueError(
                        f"replacement for {_path!r} has shape {tuple(value.shape)}, "
                        f"expected {tuple(original.shape)}"
                    )
                return value

            registrations.append((path, _replace))

        with self._hooks(registrations), torch.no_grad():
            return self.forward(inputs)

    def model_fingerprint_payload(self) -> Any:
        return {
            "module": type(self.model).__module__,
            "qualname": type(self.model).__qualname__,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in sorted(self.model.state_dict().items())
            },
        }
