"""Discover and compare models without downloading any checkpoints."""

from neuros.foundation_models import DEFAULT_REGISTRY

print("Modern EEG models")
for card in DEFAULT_REGISTRY.filter(modality="eeg", min_year=2025):
    status = DEFAULT_REGISTRY.availability(card.id)
    print(
        f"{card.id:16s} {card.name:24s} access={card.access.value:12s} "
        f"integration={card.integration.value:8s} runnable={status.available}"
    )

print("\nGeometry strategies")
for row in DEFAULT_REGISTRY.compare(
    ["zuna-1.1", "reve-base", "neuript", "luna-eeg"],
    fields=("id", "input_geometry", "pretraining_objective", "pretraining_scale"),
):
    print(row)
