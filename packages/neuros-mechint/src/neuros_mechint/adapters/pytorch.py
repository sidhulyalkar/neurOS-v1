"""Default adapter for ordinary ``torch.nn.Module`` models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from .base import ModelAdapter
from .targets import ModuleOutputTarget, _replace_leaf, _tensor_leaf


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
    """Trace and replace module outputs using temporary forward hooks.

    Legacy tensor module paths still work; named targets select structured
    tensor leaves or a specified invocation of a reused module.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        targets: Mapping[str, ModuleOutputTarget] | None = None,
    ) -> None:
        self.model = model
        declared = dict(targets or {})
        modules = self._modules()
        for alias, target in declared.items():
            if not isinstance(alias, str) or not alias:
                raise ValueError("target aliases must be nonempty strings")
            if alias in modules:
                raise ValueError(f"target alias {alias!r} collides with a module name")
            if not isinstance(target, ModuleOutputTarget):
                raise TypeError(f"target {alias!r} must be a ModuleOutputTarget")
            if target.module_path not in modules:
                raise KeyError(f"target {alias!r} references unknown module {target.module_path!r}")
        self.targets: Mapping[str, ModuleOutputTarget] = MappingProxyType(declared)

    def _modules(self) -> dict[str, nn.Module]:
        return dict(self.model.named_modules())

    def _resolve(self, names: Sequence[str]) -> dict[str, ModuleOutputTarget]:
        modules = self._modules()
        resolved: dict[str, ModuleOutputTarget] = {}
        for name in names:
            if name in self.targets:
                resolved[name] = self.targets[name]
            elif name in modules:
                resolved[name] = ModuleOutputTarget(name)
            else:
                raise KeyError(f"unknown module path or named target: {name!r}")
        return resolved

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

    @staticmethod
    def _validate_call_counts(
        targets: Mapping[str, ModuleOutputTarget],
        counts: Mapping[str, int],
    ) -> None:
        for name, target in targets.items():
            n = counts.get(target.module_path, 0)
            expected = target.expected_calls if target.expected_calls is not None else 1
            if target.occurrence is None and n != 1:
                raise ValueError(
                    f"target {name!r}: module called {n} times; "
                    "declare occurrence and expected_calls for repeated modules"
                )
            if target.occurrence is not None and n != expected:
                raise ValueError(f"target {name!r}: module called {n} times; expected {expected}")

    def capture_outputs(self, inputs: Any, paths: Sequence[str]) -> dict[str, torch.Tensor]:
        resolved = self._resolve(tuple(paths))
        grouped: dict[str, list[tuple[str, ModuleOutputTarget]]] = defaultdict(list)
        for name, target in resolved.items():
            grouped[target.module_path].append((name, target))
        counts: dict[str, int] = defaultdict(int)
        cache: dict[str, torch.Tensor] = {}
        registrations = []

        for path, requested in grouped.items():
            def _capture(
                module: nn.Module,
                args: tuple[Any, ...],
                output: Any,
                *,
                _path: str = path,
                _requested: list[tuple[str, ModuleOutputTarget]] = requested,
            ) -> None:
                del module, args
                occurrence = counts[_path]
                counts[_path] += 1
                for name, target in _requested:
                    if target.occurrence is None or target.occurrence == occurrence:
                        cache[name] = _tensor_leaf(output, target.selector, name).detach().clone()
            registrations.append((path, _capture))

        with self._hooks(registrations), torch.no_grad():
            self.forward(inputs)
        self._validate_call_counts(resolved, counts)
        return cache

    def forward_with_replacements(
        self,
        inputs: Any,
        replacements: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        resolved = self._resolve(tuple(replacements))
        grouped: dict[str, list[tuple[str, ModuleOutputTarget, torch.Tensor]]] = defaultdict(list)
        for name, target in resolved.items():
            grouped[target.module_path].append((name, target, replacements[name]))

        for requested in grouped.values():
            for i, (name_a, target_a, _) in enumerate(requested):
                for name_b, target_b, _ in requested[i + 1 :]:
                    if (target_a.occurrence or 0) != (target_b.occurrence or 0):
                        continue
                    a, b = target_a.selector, target_b.selector
                    if a[: len(b)] == b or b[: len(a)] == a:
                        raise ValueError(
                            f"overlapping selectors for targets {name_a!r} and {name_b!r}"
                        )

        counts: dict[str, int] = defaultdict(int)
        registrations = []
        for path, requested in grouped.items():
            def _replace(
                module: nn.Module,
                args: tuple[Any, ...],
                output: Any,
                *,
                _path: str = path,
                _requested: list[tuple[str, ModuleOutputTarget, torch.Tensor]] = requested,
            ) -> Any:
                del module, args
                occurrence = counts[_path]
                counts[_path] += 1
                for name, target, replacement in _requested:
                    if target.occurrence is None or target.occurrence == occurrence:
                        output = _replace_leaf(output, target.selector, replacement, name)
                return output
            registrations.append((path, _replace))

        with self._hooks(registrations), torch.no_grad():
            result = self.forward(inputs)
        self._validate_call_counts(resolved, counts)
        return result

    def model_fingerprint_payload(self) -> Any:
        payload: dict[str, Any] = {
            "module": type(self.model).__module__,
            "qualname": type(self.model).__qualname__,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in sorted(self.model.state_dict().items())
            },
        }
        if self.targets:
            # Bind alias semantics to the effective execution surface's identity.
            payload["named_intervention_targets"] = {
                name: target.to_dict() for name, target in sorted(self.targets.items())
            }
        return payload
