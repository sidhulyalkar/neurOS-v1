"""Portable JSON manifests for sharing exact synthetic BCI worlds."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .specs import ArenaScenario, DeviceProfile, DisplayProfile, ParticipantProfile, TransportProfile, WorldModelProfile

SCHEMA_V1 = "neuros.synthetic_bci_arena.manifest.v1"
SCHEMA = "neuros.synthetic_bci_arena.manifest.v2"
SUPPORTED_SCHEMAS = frozenset({SCHEMA_V1, SCHEMA})


@dataclass(frozen=True)
class ArenaManifest:
    scenario: ArenaScenario
    participant: ParticipantProfile
    device: DeviceProfile
    display: DisplayProfile
    transport: TransportProfile
    world_model: WorldModelProfile = field(default_factory=WorldModelProfile)

    def validate(self) -> None:
        self.scenario.validate()
        self.participant.validate()
        self.device.validate()
        self.display.validate()
        self.transport.validate()
        self.world_model.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "scenario": self.scenario.to_dict(),
            "participant": asdict(self.participant),
            "device": asdict(self.device),
            "display": asdict(self.display),
            "transport": asdict(self.transport),
            "world_model": asdict(self.world_model),
        }


def _device(raw: dict[str, Any]) -> DeviceProfile:
    values = dict(raw)
    if "channel_names" in values:
        values["channel_names"] = tuple(values["channel_names"])
    return DeviceProfile(**values)


def _transport(raw: dict[str, Any]) -> TransportProfile:
    values = dict(raw)
    if "silence_windows" in values:
        values["silence_windows"] = tuple(tuple(float(v) for v in window) for window in values["silence_windows"])
    return TransportProfile(**values)


def manifest_from_dict(raw: dict[str, Any]) -> ArenaManifest:
    schema = raw.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"expected manifest schema in {sorted(SUPPORTED_SCHEMAS)!r}")
    manifest = ArenaManifest(
        scenario=ArenaScenario.from_dict(dict(raw["scenario"])),
        participant=ParticipantProfile(**dict(raw["participant"])),
        device=_device(dict(raw["device"])),
        display=DisplayProfile(**dict(raw["display"])),
        transport=_transport(dict(raw["transport"])),
        world_model=WorldModelProfile(**dict(raw.get("world_model", {}))),
    )
    manifest.validate()
    return manifest


def load_manifest(path: str | Path) -> ArenaManifest:
    return manifest_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_manifest(manifest: ArenaManifest, path: str | Path) -> None:
    manifest.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
