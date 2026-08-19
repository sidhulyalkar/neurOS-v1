"""TransformerLens/TransformerBridge adapter using their native hook APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .base import ModelAdapter
from .pytorch import ModelCall


def _extract_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return _extract_output_tensor(item)
            except TypeError:
                continue
    raise TypeError(
        "TransformerLens adapter expected a tensor output, an object with tensor .logits, "
        "or a tuple/list containing a tensor"
    )


def _call_parts(inputs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(inputs, ModelCall):
        return inputs.args, dict(inputs.kwargs)
    return (inputs,), {}


class TransformerLensAdapter(ModelAdapter):
    """Normalize TransformerLens 3 hook/cache semantics to ``ModelAdapter``.

    The adapter is deliberately duck-typed and therefore works with both
    ``HookedRootModule``-style models and TransformerLens 3 ``TransformerBridge``
    objects exposing ``run_with_cache`` and ``run_with_hooks``. Importing
    ``neuros_mechint`` never imports TransformerLens.
    """

    def __init__(self, model: Any) -> None:
        for method in ("run_with_cache", "run_with_hooks"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"TransformerLens model must expose callable {method}()")
        self.model = model

    def forward(self, inputs: Any) -> torch.Tensor:
        args, kwargs = _call_parts(inputs)
        if "fwd_hooks" in kwargs:
            raise ValueError("ModelCall kwargs must not override TransformerLens fwd_hooks")
        with torch.no_grad():
            output = self.model.run_with_hooks(*args, fwd_hooks=[], **kwargs)
        return _extract_output_tensor(output)

    def capture_outputs(self, inputs: Any, paths: Sequence[str]) -> dict[str, torch.Tensor]:
        paths = tuple(dict.fromkeys(str(path) for path in paths))
        if not paths:
            return {}
        args, kwargs = _call_parts(inputs)
        if "names_filter" in kwargs:
            raise ValueError("ModelCall kwargs must not override TransformerLens names_filter")
        with torch.no_grad():
            _, cache = self.model.run_with_cache(*args, names_filter=list(paths), **kwargs)
        captured = {}
        for path in paths:
            try:
                value = cache[path]
            except (KeyError, TypeError) as exc:
                raise KeyError(f"TransformerLens did not cache hook point {path!r}") from exc
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"cached hook point {path!r} is not tensor-valued")
            captured[path] = value.detach().clone()
        return captured

    def forward_with_replacements(
        self,
        inputs: Any,
        replacements: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        args, kwargs = _call_parts(inputs)
        if "fwd_hooks" in kwargs:
            raise ValueError("ModelCall kwargs must not override TransformerLens fwd_hooks")
        hooks = []
        for path, replacement in replacements.items():

            def _replace(
                activation: torch.Tensor,
                hook: Any,
                *,
                _path: str = str(path),
                _replacement: torch.Tensor = replacement,
            ) -> torch.Tensor:
                del hook
                value = _replacement.to(device=activation.device, dtype=activation.dtype)
                if value.shape != activation.shape:
                    raise ValueError(
                        f"replacement for {_path!r} has shape {tuple(value.shape)}, "
                        f"expected {tuple(activation.shape)}"
                    )
                return value

            hooks.append((str(path), _replace))
        with torch.no_grad():
            output = self.model.run_with_hooks(*args, fwd_hooks=hooks, **kwargs)
        return _extract_output_tensor(output)

    def model_fingerprint_payload(self) -> Any | None:
        state_dict = getattr(self.model, "state_dict", None)
        if not callable(state_dict):
            return None
        return {
            "module": type(self.model).__module__,
            "qualname": type(self.model).__qualname__,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in sorted(state_dict().items())
                if isinstance(tensor, torch.Tensor)
            },
        }
