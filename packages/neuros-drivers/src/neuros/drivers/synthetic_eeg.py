"""Protocol-grade synthetic EEG generator for acquisition and BCI stress tests.

The generator is deliberately not a physiological digital twin. It creates a
controlled signal with useful nuisance structure so downstream systems can be
tested against weak SSVEPs, endogenous alpha, contact loss, movement and EMG
without requiring physical hardware for every iteration.

Determinism is a contract: for a fixed seed, controls, and sample-indexed event
schedule, the generated sample sequence must not depend on how callers partition
the same duration across ``render()`` calls. Artifact events are independently
seeded and rendered in canonical order, so adding or reordering an unrelated
event cannot steal random draws from another event.

The source-level ``saturation`` and ``dropout`` artifact names are retained for
backward compatibility as synthetic stress conveniences. They are not claims
about physical Unicorn amplifier saturation or transport loss; device clipping
and transport semantics belong to the Unicorn/device layer.

Every declared oscillatory component must also be representable at the configured
sample rate. The generator fails closed instead of silently aliasing alpha, SSVEP
fundamental/harmonic, or fixed artifact carrier frequencies into a different
synthetic cause.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal, Sequence

import numpy as np

ArtifactKind = Literal["blink", "jaw", "controller", "motion", "saturation", "dropout"]
SUPPORTED_ARTIFACTS = frozenset(
    {"blink", "jaw", "controller", "motion", "saturation", "dropout"}
)
_ARTIFACT_MAX_FREQUENCY_HZ: dict[ArtifactKind, float] = {
    "jaw": 71.0,
    "controller": 46.0,
    "motion": 2.2,
}


@dataclass(frozen=True)
class SyntheticEEGConfig:
    sampling_rate_hz: float = 250.0
    channel_names: tuple[str, ...] = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
    colored_noise_uv: float = 4.5
    white_noise_uv: float = 1.25
    alpha_frequency_hz: float = 9.4
    alpha_amplitude_uv: float = 2.8
    ssvep_amplitude_uv: float = 8.0
    first_harmonic_ratio: float = 0.34
    seed: int = 7

    def validate(self) -> None:
        numeric = {
            "sampling_rate_hz": self.sampling_rate_hz,
            "colored_noise_uv": self.colored_noise_uv,
            "white_noise_uv": self.white_noise_uv,
            "alpha_frequency_hz": self.alpha_frequency_hz,
            "alpha_amplitude_uv": self.alpha_amplitude_uv,
            "ssvep_amplitude_uv": self.ssvep_amplitude_uv,
            "first_harmonic_ratio": self.first_harmonic_ratio,
        }
        for name, value in numeric.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if len(self.channel_names) != 8:
            raise ValueError("the default synthetic EEG profile expects exactly 8 channels")
        if any(not isinstance(name, str) or not name for name in self.channel_names):
            raise ValueError("channel_names must contain eight non-empty strings")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")
        if self.colored_noise_uv < 0 or self.white_noise_uv < 0:
            raise ValueError("noise amplitudes must be non-negative")
        if self.alpha_frequency_hz <= 0 or self.alpha_amplitude_uv < 0:
            raise ValueError("alpha parameters must be positive/non-negative")
        nyquist_hz = 0.5 * float(self.sampling_rate_hz)
        if self.alpha_frequency_hz >= nyquist_hz:
            raise ValueError(
                "alpha_frequency_hz must be strictly below the configured Nyquist frequency "
                f"({nyquist_hz:g} Hz)"
            )
        if self.ssvep_amplitude_uv < 0:
            raise ValueError("ssvep_amplitude_uv must be non-negative")
        if not 0 <= self.first_harmonic_ratio <= 2:
            raise ValueError("first_harmonic_ratio must be in [0, 2]")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be a non-negative integer")
        if int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class ArtifactEvent:
    """One sample-indexed synthetic nuisance event.

    ``end_sample`` is exclusive. ``channel_indices=None`` means the artifact's
    built-in spatial pattern applies to all channels. For source-level dropout
    and saturation compatibility events, the scheduler resolves the historical
    default to Oz explicitly so provenance records the affected channel.
    """

    event_id: str
    kind: ArtifactKind
    start_sample: int
    end_sample: int
    severity: float
    channel_indices: tuple[int, ...] | None
    seed: int
    evidence_class: Literal["synthetic_assumption"] = "synthetic_assumption"

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "severity": self.severity,
            "channel_indices": (
                None if self.channel_indices is None else list(self.channel_indices)
            ),
            "seed": self.seed,
            "evidence_class": self.evidence_class,
        }


@dataclass(frozen=True)
class SyntheticEEGBlock:
    data_uv: np.ndarray
    timestamps_s: np.ndarray
    target_frequency_hz: float | None
    attention_gain: float
    artifact: str | None
    artifact_events: tuple[ArtifactEvent, ...] = ()

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.artifact_events)


class SyntheticEEGGenerator:
    """Stateful eight-channel EEG source with controllable SSVEP and artifacts."""

    posterior_weights = np.asarray([0.05, 0.05, 0.08, 0.05, 0.45, 0.85, 1.0, 0.85])
    central_weights = np.asarray([0.18, 0.95, 1.0, 0.95, 0.25, 0.10, 0.08, 0.10])
    frontal_weights = np.asarray([1.0, 0.30, 0.55, 0.30, 0.22, 0.12, 0.10, 0.12])

    def __init__(self, config: SyntheticEEGConfig | None = None) -> None:
        self.config = config or SyntheticEEGConfig()
        self.config.validate()
        seed_sequence = np.random.SeedSequence(self.config.seed)
        # Keep four children so the v2 phase/background/white streams retain
        # their identities even though v3 replaces the shared artifact stream
        # with event-local deterministic seeds.
        phase_seed, colored_seed, white_seed, _legacy_artifact_seed = seed_sequence.spawn(4)
        self._phase_rng = np.random.default_rng(phase_seed)
        self._colored_rng = np.random.default_rng(colored_seed)
        self._white_rng = np.random.default_rng(white_seed)
        self.sample_index = 0
        self.target_frequency_hz: float | None = None
        self.attention_gain = 0.0
        self.channel_gain = np.ones(8, dtype=float)
        # Start the AR components in their stationary N(0,1) marginal rather
        # than introducing a seed-dependent warm-up transient from zero.
        self._colored_state = self._colored_rng.normal(size=(8, 4))
        self._phase = self._phase_rng.uniform(0, 2 * np.pi, 8)
        self._alpha_phase = self._phase_rng.uniform(0, 2 * np.pi, 8)
        self._artifact_events: list[ArtifactEvent] = []
        self._artifact_noise_cache: dict[str, np.ndarray] = {}
        self._legacy_artifact_serial = 0

    @staticmethod
    def _require_integer(value: object, *, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        return int(value)

    @property
    def nyquist_hz(self) -> float:
        return 0.5 * float(self.config.sampling_rate_hz)

    def _require_representable_frequency(self, frequency_hz: float, *, component: str) -> None:
        if frequency_hz >= self.nyquist_hz:
            raise ValueError(
                f"{component} frequency {frequency_hz:g} Hz must be strictly below the "
                f"configured Nyquist frequency ({self.nyquist_hz:g} Hz); refusing to "
                "silently alias the declared synthetic component"
            )

    def _channel_index(self, channel: str | int) -> int:
        if isinstance(channel, str):
            try:
                return self.config.channel_names.index(channel)
            except ValueError as exc:
                raise ValueError(f"unknown channel: {channel!r}") from exc
        index = self._require_integer(channel, name="channel index")
        if not 0 <= index < 8:
            raise IndexError("channel index out of range")
        return index

    def set_attention(self, frequency_hz: float | None, gain: float = 1.0) -> None:
        if frequency_hz is None:
            self.target_frequency_hz = None
            self.attention_gain = 0.0
            return
        frequency = float(frequency_hz)
        gain_value = float(gain)
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency_hz must be positive and finite")
        if not np.isfinite(gain_value):
            raise ValueError("gain must be finite")
        self._require_representable_frequency(frequency, component="SSVEP fundamental")
        if self.config.first_harmonic_ratio > 0:
            self._require_representable_frequency(
                2.0 * frequency,
                component="SSVEP first harmonic",
            )
        self.target_frequency_hz = frequency
        self.attention_gain = float(np.clip(gain_value, 0.0, 1.5))

    def set_channel_gain(self, channel: str | int, gain: float) -> None:
        index = self._channel_index(channel)
        gain_value = float(gain)
        if not np.isfinite(gain_value):
            raise ValueError("gain must be finite")
        self.channel_gain[index] = max(0.0, gain_value)

    def _normalize_channels(
        self,
        channels: str | int | Sequence[str | int] | None,
        *,
        kind: ArtifactKind,
    ) -> tuple[int, ...] | None:
        if channels is None:
            # Preserve the historical Oz-only support of these compatibility
            # stressors, but make it explicit in the event provenance.
            return (6,) if kind in {"saturation", "dropout"} else None
        values: Sequence[str | int]
        if isinstance(channels, (str, int, np.integer)) and not isinstance(channels, (bool, np.bool_)):
            values = (channels,)
        else:
            values = channels
        indices = [self._channel_index(channel) for channel in values]
        if not indices:
            raise ValueError("channels must contain at least one channel")
        return tuple(sorted(set(indices)))

    def _derive_event_seed(
        self,
        *,
        event_id: str,
        kind: ArtifactKind,
        start_sample: int,
        end_sample: int,
        severity: float,
        channel_indices: tuple[int, ...] | None,
    ) -> int:
        payload = (
            f"{int(self.config.seed)}|{event_id}|{kind}|{start_sample}|{end_sample}|"
            f"{severity:.17g}|{channel_indices}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)

    @property
    def scheduled_artifacts(self) -> tuple[ArtifactEvent, ...]:
        return tuple(
            sorted(self._artifact_events, key=lambda event: (event.start_sample, event.event_id))
        )

    def schedule_artifact(
        self,
        kind: ArtifactKind,
        *,
        event_id: str,
        duration_seconds: float = 0.35,
        severity: float = 1.0,
        start_sample: int | None = None,
        delay_seconds: float = 0.0,
        channels: str | int | Sequence[str | int] | None = None,
        seed: int | None = None,
    ) -> ArtifactEvent:
        """Schedule an independently reproducible nuisance event.

        The event start is expressed on the generator's sample clock. Callers may
        instead supply ``delay_seconds`` relative to the current sample. Event
        identity is explicit so stochastic waveforms do not depend on insertion
        order. A duplicate currently scheduled ``event_id`` is rejected.
        """

        if kind not in SUPPORTED_ARTIFACTS:
            raise ValueError(f"unsupported artifact kind: {kind}")
        max_frequency_hz = _ARTIFACT_MAX_FREQUENCY_HZ.get(kind)
        if max_frequency_hz is not None:
            self._require_representable_frequency(
                max_frequency_hz,
                component=f"{kind} artifact carrier",
            )
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if not np.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive and finite")
        if not np.isfinite(severity) or severity < 0:
            raise ValueError("severity must be non-negative and finite")
        if not np.isfinite(delay_seconds) or delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative and finite")
        if start_sample is not None and delay_seconds != 0:
            raise ValueError("specify either start_sample or delay_seconds, not both")
        if any(event.event_id == event_id for event in self._artifact_events):
            raise ValueError(f"artifact event_id already scheduled: {event_id}")

        fs = self.config.sampling_rate_hz
        if start_sample is None:
            start = self.sample_index + int(round(delay_seconds * fs))
        else:
            start = self._require_integer(start_sample, name="start_sample")
        if start < self.sample_index:
            raise ValueError("cannot schedule an artifact in the already-rendered past")
        duration_samples = max(1, int(round(duration_seconds * fs)))
        end = start + duration_samples
        channel_indices = self._normalize_channels(channels, kind=kind)
        if seed is not None:
            event_seed = self._require_integer(seed, name="seed")
            if event_seed < 0:
                raise ValueError("seed must be non-negative")
        else:
            event_seed = self._derive_event_seed(
                event_id=event_id,
                kind=kind,
                start_sample=start,
                end_sample=end,
                severity=float(severity),
                channel_indices=channel_indices,
            )

        event = ArtifactEvent(
            event_id=event_id,
            kind=kind,
            start_sample=start,
            end_sample=end,
            severity=float(severity),
            channel_indices=channel_indices,
            seed=event_seed,
        )
        self._artifact_events.append(event)
        if event.kind == "controller":
            self._artifact_noise_cache[event.event_id] = np.random.default_rng(event.seed).normal(
                size=event.duration_samples
            )
        return event

    def cancel_artifact(self, event_id: str) -> bool:
        before = len(self._artifact_events)
        self._artifact_events = [
            event for event in self._artifact_events if event.event_id != event_id
        ]
        removed = len(self._artifact_events) != before
        if removed:
            self._artifact_noise_cache.pop(event_id, None)
        return removed

    def inject_artifact(
        self,
        kind: ArtifactKind,
        duration_seconds: float = 0.35,
        severity: float = 1.0,
    ) -> None:
        """Backward-compatible single-slot artifact injection.

        Historically a new injection replaced any currently active artifact.
        Keep that behavior here. New multi-artifact scenarios should use
        ``schedule_artifact`` with stable explicit event IDs.
        """

        self._artifact_events.clear()
        self._artifact_noise_cache.clear()
        self._legacy_artifact_serial += 1
        self.schedule_artifact(
            kind,
            event_id=f"legacy-{self._legacy_artifact_serial}",
            duration_seconds=duration_seconds,
            severity=severity,
            start_sample=self.sample_index,
        )

    def _colored_noise(self, samples: int) -> np.ndarray:
        alphas = np.asarray([0.70, 0.90, 0.975, 0.995])
        weights = np.asarray([0.55, 0.42, 0.30, 0.20])
        # Each AR component has stationary unit variance, so this fixed scale is
        # the theoretical marginal SD of their weighted sum. Unlike per-block
        # normalization it does not make the signal depend on render boundaries.
        normalizer = float(np.sqrt(np.sum(weights**2)))
        out = np.empty((8, samples), dtype=float)
        innovation_scale = np.sqrt(1.0 - alphas**2)
        for index in range(samples):
            innovation = self._colored_rng.normal(size=(8, 4))
            self._colored_state = alphas * self._colored_state + innovation_scale * innovation
            out[:, index] = (self._colored_state * weights).sum(axis=1)
        return out * (self.config.colored_noise_uv / normalizer)

    def _white_noise(self, samples: int) -> np.ndarray:
        # Draw time-major so one render of N samples and multiple renders whose
        # lengths sum to N assign the same random values to channel/time pairs.
        return self._white_rng.normal(
            0.0,
            self.config.white_noise_uv,
            size=(samples, 8),
        ).T

    def _event_weights(
        self,
        event: ArtifactEvent,
        weights: np.ndarray,
    ) -> np.ndarray:
        if event.channel_indices is None:
            return weights
        selected = np.zeros(8, dtype=float)
        selected[list(event.channel_indices)] = weights[list(event.channel_indices)]
        return selected

    def _render_event_additive(
        self,
        event: ArtifactEvent,
        *,
        block_start: int,
        block_end: int,
        time_s: np.ndarray,
    ) -> np.ndarray:
        output = np.zeros((8, block_end - block_start), dtype=float)
        overlap_start = max(block_start, event.start_sample)
        overlap_end = min(block_end, event.end_sample)
        if overlap_start >= overlap_end or event.kind == "dropout":
            return output

        block_slice = slice(overlap_start - block_start, overlap_end - block_start)
        event_offset_start = overlap_start - event.start_sample
        event_offset_end = overlap_end - event.start_sample
        event_offsets = np.arange(event_offset_start, event_offset_end)
        phase = event_offsets / max(1, event.duration_samples - 1)
        t = time_s[block_slice]
        severity = event.severity

        if event.kind == "blink":
            pulse = np.sin(np.pi * np.clip(phase, 0, 1)) ** 2
            weights = self._event_weights(event, self.frontal_weights)
            output[:, block_slice] += weights[:, None] * (120 * severity * pulse)
        elif event.kind == "jaw":
            high_frequency = (
                np.sin(2 * np.pi * 38 * t)
                + 0.8 * np.sin(2 * np.pi * 53 * t + 0.7)
                + 0.55 * np.sin(2 * np.pi * 71 * t + 1.1)
            )
            weights = self._event_weights(
                event,
                0.55 * self.central_weights + 0.35,
            )
            output[:, block_slice] += weights[:, None] * (
                48 * severity * high_frequency
            )
        elif event.kind == "controller":
            # Event-local random samples are cached once from the event seed.
            # Slicing is therefore random-access stable across render boundaries.
            random_component = self._artifact_noise_cache[event.event_id][
                event_offset_start:event_offset_end
            ]
            controller_emg = (
                np.sin(2 * np.pi * 31 * t)
                + 0.55 * np.sin(2 * np.pi * 46 * t + 0.4)
                + 0.30 * random_component
            )
            weights = self._event_weights(event, self.central_weights)
            output[:, block_slice] += weights[:, None] * (
                24 * severity * controller_emg
            )
        elif event.kind == "motion":
            drift = np.sin(np.pi * np.clip(phase, 0, 1)) * np.sign(
                np.sin(2 * np.pi * 2.2 * t)
            )
            weights = self._event_weights(
                event,
                0.60 + 0.40 * self.frontal_weights,
            )
            output[:, block_slice] += weights[:, None] * (55 * severity * drift)
        elif event.kind == "saturation":
            # Compatibility source stressor only. True Unicorn clipping is
            # modeled later by UnicornHybridBlackSimulator._quantize_eeg().
            channels = event.channel_indices or (6,)
            output[list(channels), block_slice] += 480 * severity
        return output

    def _overlapping_events(
        self,
        block_start: int,
        block_end: int,
    ) -> tuple[ArtifactEvent, ...]:
        # Canonical event-id order makes floating-point summation independent of
        # scheduling/insertion order.
        return tuple(
            sorted(
                (
                    event
                    for event in self._artifact_events
                    if event.start_sample < block_end and event.end_sample > block_start
                ),
                key=lambda event: event.event_id,
            )
        )

    def _apply_artifacts(
        self,
        data: np.ndarray,
        *,
        block_start: int,
        block_end: int,
        time_s: np.ndarray,
    ) -> tuple[np.ndarray, tuple[ArtifactEvent, ...]]:
        events = self._overlapping_events(block_start, block_end)
        for event in events:
            data += self._render_event_additive(
                event,
                block_start=block_start,
                block_end=block_end,
                time_s=time_s,
            )

        data *= self.channel_gain[:, None]

        # Multiplicative/masking effects are applied after source/channel gain.
        # Multiple dropouts compose as the union of their exact sample/channel
        # supports.
        for event in events:
            if event.kind != "dropout":
                continue
            overlap_start = max(block_start, event.start_sample)
            overlap_end = min(block_end, event.end_sample)
            if overlap_start >= overlap_end:
                continue
            channels = event.channel_indices or (6,)
            data[
                list(channels),
                overlap_start - block_start : overlap_end - block_start,
            ] = 0.0

        # Retain only future or still-active events. Completed event objects are
        # carried by the returned block, not accumulated indefinitely in state.
        completed_ids = {
            event.event_id for event in self._artifact_events if event.end_sample <= block_end
        }
        self._artifact_events = [
            event for event in self._artifact_events if event.end_sample > block_end
        ]
        for event_id in completed_ids:
            self._artifact_noise_cache.pop(event_id, None)
        return data, events

    def render(self, samples: int) -> SyntheticEEGBlock:
        sample_count = self._require_integer(samples, name="samples")
        if sample_count <= 0:
            raise ValueError("samples must be positive")
        fs = self.config.sampling_rate_hz
        block_start = self.sample_index
        block_end = block_start + sample_count
        sample_index = block_start + np.arange(sample_count)
        time_s = sample_index / fs
        data = self._colored_noise(sample_count)
        data += self._white_noise(sample_count)
        alpha = np.sin(
            2 * np.pi * self.config.alpha_frequency_hz * time_s[None, :]
            + self._alpha_phase[:, None]
        )
        data += self.posterior_weights[:, None] * self.config.alpha_amplitude_uv * alpha
        if self.target_frequency_hz is not None and self.attention_gain > 0:
            frequency = self.target_frequency_hz
            fundamental = np.sin(
                2 * np.pi * frequency * time_s[None, :] + self._phase[:, None]
            )
            if self.config.first_harmonic_ratio > 0:
                harmonic = np.sin(
                    2 * np.pi * 2 * frequency * time_s[None, :]
                    + 0.5 * self._phase[:, None]
                )
                ssvep = fundamental + self.config.first_harmonic_ratio * harmonic
            else:
                ssvep = fundamental
            data += (
                self.posterior_weights[:, None]
                * self.config.ssvep_amplitude_uv
                * self.attention_gain
                * ssvep
            )

        data, active_events = self._apply_artifacts(
            data,
            block_start=block_start,
            block_end=block_end,
            time_s=time_s,
        )

        self.sample_index = block_end
        kinds = tuple(dict.fromkeys(event.kind for event in active_events))
        legacy_artifact = (
            None if not kinds else kinds[0] if len(kinds) == 1 else "multiple"
        )
        return SyntheticEEGBlock(
            data_uv=data.astype(np.float32),
            timestamps_s=time_s.astype(float),
            target_frequency_hz=self.target_frequency_hz,
            attention_gain=self.attention_gain,
            artifact=legacy_artifact,
            artifact_events=active_events,
        )
