"""BIDS-aligned provenance metadata for recorded EEG entering Arena.

Arena's compact NPZ baseline format is intentionally not advertised as BIDS.
This sidecar preserves the subset of acquisition/task/channel provenance needed
to trace a derived Arena baseline back to a canonical BIDS/MNE/public-dataset
source without inventing another primary neuroscience interchange standard.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

RECORDING_SCHEMA = "neuros.synthetic_bci_arena.recording_metadata.v1"


@dataclass(frozen=True)
class ElectrodeCoordinate:
    name: str
    x: float
    y: float
    z: float

    def validate(self) -> None:
        if not self.name:
            raise ValueError("electrode name is required")
        for value in (self.x, self.y, self.z):
            if not isinstance(value, (int, float)):
                raise ValueError("electrode coordinates must be numeric")


@dataclass(frozen=True)
class RecordingMetadata:
    """Minimal traceable metadata for an EEG recording or derived window.

    Field names intentionally mirror common EEG-BIDS concepts where practical,
    but this object is an Arena provenance record, not a BIDS validator/writer.
    ``source_locator`` may be a BIDS-relative path, DOI, dataset identifier, or
    other non-secret reproducibility locator. Do not put participant names or
    other direct identifiers here.
    """

    dataset: str = ""
    subject: str = ""
    session: str = ""
    run: str = ""
    task: str = ""
    acquisition: str = ""
    source_locator: str = ""
    source_format: str = ""
    source_license: str = ""
    reference: str = ""
    line_frequency_hz: float | None = None
    channel_units: Mapping[str, str] = field(default_factory=dict)
    channel_types: Mapping[str, str] = field(default_factory=dict)
    coordinate_system: str = ""
    coordinate_units: str = ""
    electrodes: tuple[ElectrodeCoordinate, ...] = ()
    preprocessing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def validate(self, channel_names: Sequence[str] | None = None) -> None:
        if self.line_frequency_hz is not None and self.line_frequency_hz <= 0:
            raise ValueError("line_frequency_hz must be positive when present")
        for mapping_name, mapping in (("channel_units", self.channel_units), ("channel_types", self.channel_types)):
            for key, value in mapping.items():
                if not str(key) or not str(value):
                    raise ValueError(f"{mapping_name} keys and values must be non-empty")
        electrode_names = [electrode.name for electrode in self.electrodes]
        if len(electrode_names) != len(set(electrode_names)):
            raise ValueError("electrode names must be unique")
        for electrode in self.electrodes:
            electrode.validate()
        if self.electrodes and (not self.coordinate_system or not self.coordinate_units):
            raise ValueError("electrode coordinates require coordinate_system and coordinate_units")
        if channel_names is not None:
            allowed = set(str(name) for name in channel_names)
            extras = (set(self.channel_units) | set(self.channel_types) | set(electrode_names)) - allowed
            if extras:
                raise ValueError(f"recording metadata references channels not present in data: {sorted(extras)}")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": RECORDING_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RecordingMetadata":
        if raw.get("schema") != RECORDING_SCHEMA:
            raise ValueError(f"expected recording metadata schema {RECORDING_SCHEMA!r}")
        values = dict(raw)
        values.pop("schema", None)
        values["channel_units"] = dict(values.get("channel_units", {}))
        values["channel_types"] = dict(values.get("channel_types", {}))
        values["electrodes"] = tuple(ElectrodeCoordinate(**item) for item in values.get("electrodes", []))
        values["preprocessing"] = tuple(values.get("preprocessing", []))
        values["notes"] = tuple(values.get("notes", []))
        metadata = cls(**values)
        metadata.validate()
        return metadata


def recording_sidecar_path(baseline_path: str | Path) -> Path:
    path = Path(baseline_path)
    return path.with_suffix(path.suffix + ".json")


def save_recording_metadata(metadata: RecordingMetadata, path: str | Path, *, channel_names: Sequence[str] | None = None) -> Path:
    metadata.validate(channel_names)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output


def load_recording_metadata(path: str | Path) -> RecordingMetadata:
    return RecordingMetadata.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
