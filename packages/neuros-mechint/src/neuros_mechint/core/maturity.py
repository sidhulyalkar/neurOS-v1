"""Scientific maturity metadata for interpretability methods."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class MethodMaturity(str, Enum):
    """Maturity of a method implementation, separate from result evidence."""

    STABLE = "stable"
    INTEGRATED = "integrated"
    RESEARCH = "research"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


@dataclass(frozen=True, slots=True)
class MethodCard:
    """Compact scientific contract for an analysis method."""

    name: str
    maturity: MethodMaturity
    establishes: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    required_controls: tuple[str, ...] = field(default_factory=tuple)
    references: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "maturity": self.maturity.value,
            "establishes": list(self.establishes),
            "limitations": list(self.limitations),
            "required_controls": list(self.required_controls),
            "references": list(self.references),
        }


_REGISTRY: dict[str, MethodCard] = {}


def register_method_card(card: MethodCard, *, replace: bool = False) -> MethodCard:
    if card.name in _REGISTRY and not replace:
        raise ValueError(f"method card already registered: {card.name}")
    _REGISTRY[card.name] = card
    return card


def get_method_card(name: str) -> MethodCard:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown method card: {name}") from exc


def list_method_cards() -> Iterable[MethodCard]:
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


register_method_card(
    MethodCard(
        name="module_activation_patching",
        maturity=MethodMaturity.STABLE,
        establishes=("causal effect of replacing a module output",),
        limitations=("does not isolate a unique computational path",),
        required_controls=("matched corrupted examples", "shuffled donor control"),
    )
)
register_method_card(
    MethodCard(
        name="module_output_ablation",
        maturity=MethodMaturity.STABLE,
        establishes=("necessity of a module output for a chosen metric",),
        limitations=("zero and mean ablations may be off distribution",),
        required_controls=("matched random components", "multiple ablation baselines"),
    )
)
register_method_card(
    MethodCard(
        name="input_causal_audit",
        maturity=MethodMaturity.STABLE,
        establishes=("causal sensitivity of a metric to a specified input edit",),
        limitations=(
            "input edits can be off distribution",
            "does not localize internal computation",
        ),
        required_controls=("matched edit controls", "held-out evaluation when fit is involved"),
    )
)
register_method_card(
    MethodCard(
        name="causal_effect_map_stability",
        maturity=MethodMaturity.STABLE,
        establishes=(
            "statistical agreement of named causal effects across aligned contexts",
            "explicit overlap of comparable intervention targets",
        ),
        limitations=(
            "agreement is not proof that the underlying biological mechanism is identical",
            "pairwise comparisons can confound multiple changing context axes",
        ),
        required_controls=(
            "aligned intervention definitions",
            "explicit shared-target coverage",
            "held-out contexts",
        ),
    )
)
register_method_card(
    MethodCard(
        name="orion_token_causal_audit",
        maturity=MethodMaturity.INTEGRATED,
        establishes=("causal sensitivity to ORION token windows, types, or side features",),
        limitations=(
            "effects depend on the downstream scorer",
            "mask-token choice can induce distribution shift",
        ),
        required_controls=("matched random windows", "shuffle controls", "multiple mask baselines"),
    )
)
register_method_card(
    MethodCard(
        name="orion_representation_causal_audit",
        maturity=MethodMaturity.INTEGRATED,
        establishes=("causal sensitivity to ORION latent time windows or feature dimensions",),
        limitations=(
            "zero and mean replacement can be off the learned representation manifold",
            "does not identify the upstream token mechanism that produced a latent",
        ),
        required_controls=(
            "within-window shuffle controls",
            "feature temporal-permutation controls",
            "multiple ablation baselines",
            "cross-session replication",
        ),
    )
)
register_method_card(
    MethodCard(
        name="orion_shared_representation_study",
        maturity=MethodMaturity.INTEGRATED,
        establishes=(
            "comparative causal-effect maps over event-aligned ORION representations",
            "quantified stability across architecture/session/dataset/subject/checkpoint contexts",
        ),
        limitations=(
            "event alignment must be semantically comparable across contexts",
            "raw latent feature indices are not assumed to have cross-model semantic identity",
        ),
        required_controls=(
            "matched event-relative intervention windows",
            "shuffle or in-distribution replacement controls",
            "matched task metrics and performance",
            "held-out sessions or datasets",
        ),
    )
)
register_method_card(
    MethodCard(
        name="shared_computation_hypothesis_engine",
        maturity=MethodMaturity.RESEARCH,
        establishes=(
            "transparent prioritization of falsifiable hypotheses from causal-map comparisons",
        ),
        limitations=(
            "threshold-triggered hypotheses are candidates, never automatic discoveries",
            "causal-map stability alone cannot identify a biological mechanism",
        ),
        required_controls=(
            "known-ground-truth synthetic benchmark",
            "explicit threshold policy",
            "alternative interventions",
            "held-out validation",
        ),
    )
)
register_method_card(
    MethodCard(
        name="module_path_patching",
        maturity=MethodMaturity.RESEARCH,
        establishes=("receiver-mediated effect of a sender module",),
        limitations=(
            "module-level paths are coarser than attention-head or feature-level paths",
            "results depend on clean/corrupted pair design",
        ),
        required_controls=("shuffled donors", "random sender/receiver pairs", "held-out pairs"),
    )
)
register_method_card(
    MethodCard(
        name="acdc_inspired_module_pruning",
        maturity=MethodMaturity.EXPERIMENTAL,
        establishes=("module-level necessity ranking under a chosen ablation",),
        limitations=(
            "not a faithful implementation of canonical edge-level ACDC",
            "module pruning can merge multiple computational pathways",
        ),
        required_controls=("random equal-sparsity subnetworks", "held-out circuit evaluation"),
    )
)
