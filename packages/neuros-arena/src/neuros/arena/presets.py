"""Built-in deterministic profiles and scenarios for common closed-loop BCI stress cases."""
from __future__ import annotations

from .specs import ArenaScenario, ArtifactEvent, DeviceProfile, DisplayProfile, ParticipantProfile, StageSpec, TransportProfile


_PRESETS = {"dual-target-smoke", "dual-target-torture", "alpha-collision", "weak-responder"}


def list_presets() -> tuple[str, ...]:
    return tuple(sorted(_PRESETS))


def get_preset(name: str, seed: int = 7) -> tuple[ArenaScenario, ParticipantProfile, DeviceProfile, DisplayProfile, TransportProfile]:
    if name not in _PRESETS:
        raise KeyError(f"unknown Arena preset {name!r}; choose from {', '.join(list_presets())}")
    participant = ParticipantProfile(seed=seed)
    device = DeviceProfile()
    display = DisplayProfile()
    transport = TransportProfile()
    stages = (
        StageSpec("rest", 5.0, None, 0.0),
        StageSpec("sight", 6.0, 10.0, 1.0),
        StageSpec("guard", 6.0, 12.0, 1.0),
        StageSpec("sight-return", 5.0, 10.0, 1.0),
    )
    if name == "dual-target-torture":
        stages = (
            StageSpec("rest", 5.0, None, 0.0),
            StageSpec("sight-controller", 6.0, 10.0, 1.0, (ArtifactEvent(2.0, "controller", 0.8, 1.0),)),
            StageSpec("guard-jaw", 6.0, 12.0, 1.0, (ArtifactEvent(2.5, "jaw", 0.7, 1.0),)),
            StageSpec("rest-2", 3.0, None, 0.0),
            StageSpec("sight-motion", 6.0, 10.0, 0.78, (ArtifactEvent(1.8, "motion", 0.8, 1.0),)),
            StageSpec("guard-return", 6.0, 12.0, 0.72),
        )
        participant = ParticipantProfile(seed=seed, gaze_duty_cycle=0.82, response_attenuation_per_minute=0.08)
        display = DisplayProfile(name="busy-120hz", refresh_hz=120.0, frame_jitter_ms=0.8, frame_drop_probability=0.008)
        transport = TransportProfile(name="crowded-demo-network", drop_probability=0.015, jitter_ms=8.0, reorder_probability=0.01, silence_windows=((18.0, 2.5),))
    elif name == "alpha-collision":
        participant = ParticipantProfile(name="alpha-collision", seed=seed, alpha_frequency_hz=10.0, alpha_amplitude_uv=8.0, ssvep_amplitude_uv=5.5)
    elif name == "weak-responder":
        participant = ParticipantProfile(name="weak-responder", seed=seed, ssvep_amplitude_uv=3.2, colored_noise_uv=6.0, white_noise_uv=2.0, gaze_duty_cycle=0.72, response_delay_s=0.28, switch_time_constant_s=0.65)
    scenario = ArenaScenario(name=name, stages=stages, seed=seed, metadata={"paradigm": "dual-target-ssvep"})
    return scenario, participant, device, display, transport
