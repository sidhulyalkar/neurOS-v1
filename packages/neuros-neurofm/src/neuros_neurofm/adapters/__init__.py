"""
Transfer learning adapters for NeuroFM-X.

Provides efficient fine-tuning methods for adapting the pre-trained model
to new neural populations, sessions, or tasks.
"""

from neuros_neurofm.adapters.unit_id import (
    UnitIDAdapter,
    SessionStitcher,
    LoRAAdapter,
    LoRALinear,
    apply_lora_to_model,
)

__all__ = [
    "UnitIDAdapter",
    "SessionStitcher",
    "LoRAAdapter",
    "LoRALinear",
    "apply_lora_to_model",
]
