"""Explicit, auditable tensor-leaf selectors for PyTorch module outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ModuleOutputTarget:
    """Select a tensor leaf from one module output.

    selector walks builtin tuples/lists by index and dictionaries by key.
    An unspecified occurrence requires exactly one module call. To select
    a repeated invocation, declare both occurrence and expected_calls.
    """

    module_path: str
    selector: tuple[int | str, ...] = ()
    occurrence: int | None = None
    expected_calls: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.module_path, str) or not self.module_path:
            raise ValueError("module_path must be a nonempty string")
        if isinstance(self.selector, (str, bytes)):
            raise TypeError("selector must be a tuple of keys/indices")
        selector = tuple(self.selector)
        if any(
            isinstance(part, bool)
            or not isinstance(part, (int, str))
            or (isinstance(part, int) and part < 0)
            or (isinstance(part, str) and not part)
            for part in selector
        ):
            raise ValueError("selector requires nonnegative indices or nonempty dictionary keys")
        object.__setattr__(self, "selector", selector)
        if self.occurrence is not None:
            if (
                isinstance(self.occurrence, bool)
                or not isinstance(self.occurrence, int)
                or self.occurrence < 0
            ):
                raise ValueError("occurrence must be a nonnegative integer")
            if self.expected_calls is None:
                raise ValueError("occurrence selection requires expected_calls")
        if self.expected_calls is not None:
            if (
                isinstance(self.expected_calls, bool)
                or not isinstance(self.expected_calls, int)
                or self.expected_calls < 1
            ):
                raise ValueError("expected_calls must be a positive integer")
            if self.occurrence is None and self.expected_calls != 1:
                raise ValueError("multiple calls require an explicit occurrence")
            if self.occurrence is not None and self.occurrence >= self.expected_calls:
                raise ValueError("occurrence must be smaller than expected_calls")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "selector": list(self.selector),
            "occurrence": self.occurrence,
            "expected_calls": self.expected_calls,
        }


def _require_tensor_output(name: str, output: Any) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"component {name!r} returned {type(output).__name__}; "
            "configure a ModuleOutputTarget selector for structured outputs"
        )
    return output


def _tensor_leaf(output: Any, selector: tuple[int | str, ...], name: str) -> torch.Tensor:
    for part in selector:
        if isinstance(part, int) and type(output) in (tuple, list):
            if part >= len(output):
                raise KeyError(f"target {name!r} selector index {part} is out of bounds")
            output = output[part]
        elif isinstance(part, str) and type(output) is dict:
            if part not in output:
                raise KeyError(f"target {name!r} selector key {part!r} is absent")
            output = output[part]
        else:
            raise TypeError(
                f"target {name!r} selector {part!r} does not match the output type"
            )
    return _require_tensor_output(name, output)


def _replace_leaf(
    output: Any,
    selector: tuple[int | str, ...],
    replacement: torch.Tensor,
    name: str,
) -> Any:
    if not selector:
        original = _require_tensor_output(name, output)
        if not isinstance(replacement, torch.Tensor):
            raise TypeError(f"replacement for {name!r} must be a tensor")
        if replacement.shape != original.shape:
            raise ValueError(
                f"replacement for {name!r} has shape {tuple(replacement.shape)}, "
                f"expected {tuple(original.shape)}"
            )
        return replacement.to(device=original.device, dtype=original.dtype)
    part, *remaining = selector
    tail = tuple(remaining)
    if isinstance(part, int) and type(output) in (tuple, list):
        if part >= len(output):
            raise KeyError(f"target {name!r} selector index {part} is out of bounds")
        updated = list(output)
        updated[part] = _replace_leaf(updated[part], tail, replacement, name)
        return tuple(updated) if type(output) is tuple else updated
    if isinstance(part, str) and type(output) is dict:
        if part not in output:
            raise KeyError(f"target {name!r} selector key {part!r} is absent")
        updated = dict(output)
        updated[part] = _replace_leaf(updated[part], tail, replacement, name)
        return updated
    raise TypeError(f"target {name!r} selector {part!r} does not match the output type")
