"""Activation patching and receiver-mediated module path patching.

The previous implementation exposed module-output replacement as "path
patching" while not isolating sender/receiver paths. This module separates the
two operations:

* :class:`ModuleActivationPatcher` performs ordinary clean-to-corrupted module
  output patching.
* :class:`PathPatcher` estimates the effect mediated from a sender module
  through a receiver module by caching the receiver under a sender-patched run
  and transplanting only that receiver state into the corrupted run.

This remains module-level path patching. Attention-head, SAE-feature, and
edge-level paths belong in framework-specific adapters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from neuros_mechint.adapters import PyTorchAdapter


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("path-patching metric must return a scalar")
        return float(value.detach().cpu().item())
    return float(value)


@dataclass(frozen=True)
class PatchEffect:
    """Effect of patching a single module output."""

    layer_name: str
    direct_effect: float
    recovered_fraction: float | None
    patched_metric: float
    component: str = "module_output"
    total_effect: float = 0.0
    indirect_effect: float = 0.0


@dataclass(frozen=True)
class PathEffect:
    """Receiver-mediated effect attributed to one sender/receiver pair."""

    sender: str
    receiver: str
    mediated_effect: float
    recovered_fraction: float | None
    mediated_metric: float

    @property
    def layer_name(self) -> str:
        return f"{self.sender}->{self.receiver}"

    @property
    def direct_effect(self) -> float:
        return self.mediated_effect


@dataclass
class PathPatchingResult:
    effects: list[PathEffect] = field(default_factory=list)
    clean_output: torch.Tensor | None = None
    corrupted_output: torch.Tensor | None = None
    target_metric: float = 0.0
    baseline_metric: float = 0.0
    layer_names: list[str] = field(default_factory=list)
    metric_name: str = "custom_metric"
    method: str = "module_path_patching"

    def get_top_paths(self, k: int = 10, by: str = "mediated_effect") -> list[PathEffect]:
        if by not in {"mediated_effect", "direct_effect"}:
            raise ValueError("by must be 'mediated_effect' or 'direct_effect'")
        return sorted(self.effects, key=lambda item: abs(item.mediated_effect), reverse=True)[:k]

    def get_layer_importance(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for effect in self.effects:
            scores[effect.sender] = scores.get(effect.sender, 0.0) + abs(effect.mediated_effect)
        return scores


@dataclass
class ActivationPatchingResult:
    effects: list[PatchEffect] = field(default_factory=list)
    target_metric: float = 0.0
    baseline_metric: float = 0.0
    method: str = "module_activation_patching"

    def get_top_components(self, k: int = 10) -> list[PatchEffect]:
        return sorted(self.effects, key=lambda item: abs(item.direct_effect), reverse=True)[:k]


class ModuleActivationPatcher:
    """Perform ordinary clean-to-corrupted module-output activation patching."""

    def __init__(
        self,
        model: nn.Module,
        metric_fn: Callable[[torch.Tensor], torch.Tensor | float],
        layers_to_patch: Sequence[str] | None = None,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.metric_fn = metric_fn
        self.adapter = PyTorchAdapter(model)
        self.device = device
        self.verbose = verbose
        self.layers_to_patch = list(layers_to_patch or self._default_layers())

    def _default_layers(self) -> Iterable[str]:
        for name, module in self.model.named_modules():
            if name and len(list(module.children())) == 0:
                yield name

    def patch_all(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
    ) -> ActivationPatchingResult:
        clean_output = self.adapter.forward(clean_input)
        corrupted_output = self.adapter.forward(corrupted_input)
        clean_metric = _as_float(self.metric_fn(clean_output))
        corrupted_metric = _as_float(self.metric_fn(corrupted_output))
        total = clean_metric - corrupted_metric
        clean_cache = self.adapter.capture_outputs(clean_input, self.layers_to_patch)

        effects: list[PatchEffect] = []
        for layer_name in self.layers_to_patch:
            patched_output = self.adapter.forward_with_replacements(
                corrupted_input,
                {layer_name: clean_cache[layer_name]},
            )
            patched_metric = _as_float(self.metric_fn(patched_output))
            effect = patched_metric - corrupted_metric
            recovered = effect / total if abs(total) > 1e-12 else None
            effects.append(
                PatchEffect(
                    layer_name=layer_name,
                    direct_effect=effect,
                    total_effect=effect,
                    indirect_effect=0.0,
                    recovered_fraction=recovered,
                    patched_metric=patched_metric,
                )
            )

        return ActivationPatchingResult(
            effects=effects,
            target_metric=clean_metric,
            baseline_metric=corrupted_metric,
        )


class PathPatcher:
    """Estimate sender effects mediated through receiver module outputs.

    For each sender/receiver pair:

    1. Run the corrupted example while patching the sender with its clean value.
    2. Cache the receiver output produced by that sender-patched run.
    3. Run the original corrupted example again and patch only the receiver with
       the value from step 2.

    The resulting metric change measures the portion of the sender intervention
    that is transmitted through the selected receiver, without allowing the
    sender's other downstream paths to remain patched in the final evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        metric_fn: Callable[[torch.Tensor], torch.Tensor | float],
        layers_to_patch: Sequence[str] | None = None,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.metric_fn = metric_fn
        self.adapter = PyTorchAdapter(model)
        self.device = device
        self.verbose = verbose
        self.layers_to_patch = list(layers_to_patch or self._default_layers())

    def _default_layers(self) -> Iterable[str]:
        for name, module in self.model.named_modules():
            if name and len(list(module.children())) == 0:
                yield name

    def _ordered_pairs(
        self,
        senders: Sequence[str],
        receivers: Sequence[str],
    ) -> Iterable[tuple[str, str]]:
        order = {name: index for index, name in enumerate(self.layers_to_patch)}
        for sender in senders:
            for receiver in receivers:
                if sender == receiver:
                    continue
                if sender in order and receiver in order and order[sender] >= order[receiver]:
                    continue
                yield sender, receiver

    def patch_all_paths(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        components: Sequence[str] | None = None,
        batch_size: int | None = None,
        *,
        senders: Sequence[str] | None = None,
        receivers: Sequence[str] | None = None,
        max_paths: int | None = None,
    ) -> PathPatchingResult:
        # ``components`` and ``batch_size`` are retained only for source
        # compatibility with the legacy API. Module-level patching does not
        # pretend that a string label selects residual/attention/MLP internals.
        del components, batch_size

        senders = list(senders or self.layers_to_patch[:-1])
        receivers = list(receivers or self.layers_to_patch[1:])
        all_paths = list(self._ordered_pairs(senders, receivers))
        if max_paths is not None:
            all_paths = all_paths[:max_paths]

        required = sorted(set(senders) | set(receivers))
        clean_cache = self.adapter.capture_outputs(clean_input, required)
        clean_output = self.adapter.forward(clean_input)
        corrupted_output = self.adapter.forward(corrupted_input)
        clean_metric = _as_float(self.metric_fn(clean_output))
        corrupted_metric = _as_float(self.metric_fn(corrupted_output))
        total = clean_metric - corrupted_metric

        effects: list[PathEffect] = []
        for sender, receiver in all_paths:
            # Capture the receiver state that results when only the sender is
            # patched into the corrupted run.
            receiver_cache: dict[str, torch.Tensor] = {}
            modules = dict(self.model.named_modules())
            if receiver not in modules:
                raise KeyError(f"unknown receiver module: {receiver}")

            def _receiver_capture(
                module,
                args,
                output,
                *,
                _receiver: str = receiver,
                _cache: dict[str, torch.Tensor] = receiver_cache,
            ):
                del module, args
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"receiver {_receiver!r} must return a tensor")
                _cache[_receiver] = output.detach().clone()

            handle = modules[receiver].register_forward_hook(_receiver_capture)
            try:
                self.adapter.forward_with_replacements(
                    corrupted_input,
                    {sender: clean_cache[sender]},
                )
            finally:
                handle.remove()

            if receiver not in receiver_cache:
                raise RuntimeError(f"receiver {receiver!r} was not executed")

            mediated_output = self.adapter.forward_with_replacements(
                corrupted_input,
                {receiver: receiver_cache[receiver]},
            )
            mediated_metric = _as_float(self.metric_fn(mediated_output))
            effect = mediated_metric - corrupted_metric
            recovered = effect / total if abs(total) > 1e-12 else None
            effects.append(
                PathEffect(
                    sender=sender,
                    receiver=receiver,
                    mediated_effect=effect,
                    recovered_fraction=recovered,
                    mediated_metric=mediated_metric,
                )
            )

        return PathPatchingResult(
            effects=effects,
            clean_output=clean_output,
            corrupted_output=corrupted_output,
            target_metric=clean_metric,
            baseline_metric=corrupted_metric,
            layer_names=self.layers_to_patch,
            metric_name=getattr(self.metric_fn, "__name__", "custom_metric"),
        )


ActivationPatcher = ModuleActivationPatcher

__all__ = [
    "ActivationPatcher",
    "ActivationPatchingResult",
    "ModuleActivationPatcher",
    "PatchEffect",
    "PathEffect",
    "PathPatcher",
    "PathPatchingResult",
]
