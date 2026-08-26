"""Deterministic closed-loop arena runner."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from neuros.plugins import PluginKind, load_plugin

from .evidence import evidence_card_for_model
from .simulators import DeviceOutput, StimulusTrace, TransportPacket, apply_device, packetize, sample_stimulus, simulate_stimulus
from .specs import ArenaScenario, DeviceProfile, DisplayProfile, ParticipantProfile, TransportProfile, WorldModelProfile
from .world_input import WorldInputBlock
from .world_models import NeuralWorldModel


@dataclass(frozen=True)
class StageInterval:
    label: str
    start_s: float
    end_s: float
    target_frequency_hz: float | None


@dataclass(frozen=True)
class ArenaRun:
    scenario: ArenaScenario
    participant: ParticipantProfile
    device: DeviceProfile
    display: DisplayProfile
    transport: TransportProfile
    world_model: WorldModelProfile
    device_output: DeviceOutput
    packets: tuple[TransportPacket, ...]
    ground_truth_target_hz: np.ndarray
    stage_index: np.ndarray
    stages: tuple[StageInterval, ...]
    stimulus_traces: tuple[StimulusTrace, ...]
    report: dict


def _spectral_snr_db(data_uv: np.ndarray, fs: float, target_hz: float) -> float:
    if data_uv.shape[1] < max(32, int(fs)):
        return 0.0
    x = np.mean(data_uv, axis=0).astype(float)
    x -= np.mean(x)
    window = np.hanning(x.size)
    power = np.abs(np.fft.rfft(x * window)) ** 2
    freq = np.fft.rfftfreq(x.size, d=1.0 / fs)
    signal = np.abs(freq - target_hz) <= 0.30
    noise = (np.abs(freq - target_hz) >= 0.8) & (np.abs(freq - target_hz) <= 3.0)
    signal_power = float(np.mean(power[signal])) if np.any(signal) else 0.0
    noise_power = float(np.mean(power[noise])) if np.any(noise) else 1e-12
    return float(10.0 * np.log10(max(signal_power, 1e-12) / max(noise_power, 1e-12)))


def _create_world_model(
    profile: WorldModelProfile,
    participant: ParticipantProfile,
    sampling_rate_hz: float,
    seed: int,
) -> NeuralWorldModel:
    profile.validate()
    model = load_plugin(
        profile.name,
        kind=PluginKind.WORLD_MODEL,
        participant=participant,
        sampling_rate_hz=sampling_rate_hz,
        seed=seed,
        parameters=profile.parameters,
    )
    if not hasattr(model, "render") and not hasattr(model, "render_world"):
        raise TypeError(f"world model {profile.name!r} exposes neither render nor render_world")
    if not hasattr(model, "inject_artifact") or not hasattr(model, "channel_names"):
        raise TypeError(f"world model {profile.name!r} does not satisfy the Arena contract")
    return model


def _render_world_model(
    model: NeuralWorldModel,
    *,
    input_block: WorldInputBlock,
    target_frequency_hz: float | None,
    attention_gain: float,
):
    render_world = getattr(model, "render_world", None)
    if callable(render_world):
        input_block.validate()
        return render_world(input_block)
    return model.render(
        input_block.sample_times_s,
        input_block.visual_luminance,
        target_frequency_hz,
        attention_gain,
    )


def run_scenario(
    scenario: ArenaScenario,
    participant: ParticipantProfile,
    device: DeviceProfile,
    display: DisplayProfile,
    transport: TransportProfile,
    world_model: WorldModelProfile | None = None,
) -> ArenaRun:
    scenario.validate()
    participant.validate()
    device.validate()
    display.validate()
    transport.validate()
    model_profile = world_model or WorldModelProfile()
    model = _create_world_model(
        model_profile,
        participant,
        device.sampling_rate_hz,
        seed=participant.seed + scenario.seed,
    )
    evidence_card = evidence_card_for_model(model, model_profile.name)
    paradigm = scenario.metadata.get("paradigm", "ssvep")

    block_samples = max(1, device.chunk_samples)
    data_blocks: list[np.ndarray] = []
    timestamp_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    stage_blocks: list[np.ndarray] = []
    stage_intervals: list[StageInterval] = []
    stimulus_traces: list[StimulusTrace] = []
    latent_stage_end: list[dict[str, float | str]] = []
    global_start = 0.0

    for stage_i, stage in enumerate(scenario.stages):
        stage_start = global_start
        stage_end = stage_start + stage.duration_s
        stage_intervals.append(StageInterval(stage.label, stage_start, stage_end, stage.target_frequency_hz))
        trace = simulate_stimulus(
            stage.target_frequency_hz,
            stage.duration_s,
            display,
            seed=scenario.seed * 1009 + stage_i,
        )
        stimulus_traces.append(trace)
        samples_total = max(1, int(round(stage.duration_s * device.sampling_rate_hz)))
        artifact_cursor: set[int] = set()
        produced = 0
        last_latent: dict[str, float] = {}
        while produced < samples_total:
            count = min(block_samples, samples_total - produced)
            elapsed = produced / device.sampling_rate_hz
            local_times = elapsed + np.arange(count, dtype=float) / device.sampling_rate_hz
            global_times = stage_start + local_times
            for artifact_i, artifact in enumerate(stage.artifacts):
                if artifact_i not in artifact_cursor and elapsed <= artifact.at_s < elapsed + count / device.sampling_rate_hz:
                    model.inject_artifact(
                        artifact.kind,
                        duration_seconds=artifact.duration_s,
                        severity=artifact.severity * participant.artifact_gain,
                    )
                    artifact_cursor.add(artifact_i)
            if stage.target_frequency_hz is None:
                effective_gain = 0.0
            else:
                after_delay = max(0.0, elapsed - participant.response_delay_s)
                switch_gain = 0.0 if elapsed < participant.response_delay_s else 1.0 - np.exp(-after_delay / participant.switch_time_constant_s)
                global_elapsed_min = (stage_start + elapsed) / 60.0
                attenuation = max(0.0, 1.0 - participant.response_attenuation_per_minute * global_elapsed_min)
                effective_gain = stage.attention_gain * participant.gaze_duty_cycle * switch_gain * attenuation
            emitted_drive = sample_stimulus(trace, local_times, display)
            target = dict(stage.target)
            if stage.target_frequency_hz is not None:
                target.setdefault("frequency_hz", float(stage.target_frequency_hz))
            input_block = WorldInputBlock(
                sample_times_s=global_times,
                paradigm=paradigm,
                stage_label=stage.label,
                emitted_streams={"visual_luminance": emitted_drive},
                target=target,
                task_state=dict(stage.task_state),
                participant_state={"attention_gain": float(effective_gain)},
            )
            emission = _render_world_model(
                model,
                input_block=input_block,
                target_frequency_hz=stage.target_frequency_hz,
                attention_gain=float(effective_gain),
            )
            if emission.data_uv.shape != (len(model.channel_names), count):
                raise ValueError(
                    f"world model emitted {emission.data_uv.shape}; expected {(len(model.channel_names), count)}"
                )
            data_blocks.append(emission.data_uv)
            timestamp_blocks.append(global_times)
            last_latent = dict(emission.latent)
            truth_value = np.nan if stage.target_frequency_hz is None else float(stage.target_frequency_hz)
            truth_blocks.append(np.full(count, truth_value, dtype=float))
            stage_blocks.append(np.full(count, stage_i, dtype=np.int32))
            produced += count
        latent_stage_end.append({"stage": stage.label, **last_latent})
        global_start = stage_end

    raw_data = np.concatenate(data_blocks, axis=1)
    raw_timestamps = np.concatenate(timestamp_blocks)
    truth = np.concatenate(truth_blocks)
    stage_index = np.concatenate(stage_blocks)
    output = apply_device(raw_data, raw_timestamps, tuple(model.channel_names), device, seed=scenario.seed + 2003)
    packets, transport_metrics = packetize(
        output.data_uv,
        output.timestamps_s,
        device.chunk_samples,
        transport,
        seed=scenario.seed + 3001,
        ground_truth_timestamps_s=output.ground_truth_timestamps_s,
    )

    posterior = [index for index, name in enumerate(output.channel_names) if name in {"Pz", "PO7", "Oz", "PO8"}]
    if not posterior:
        posterior = list(range(output.data_uv.shape[0]))
    snr: dict[str, float] = {}
    for frequency in sorted({stage.target_frequency_hz for stage in scenario.stages if stage.target_frequency_hz is not None}):
        mask = np.isclose(truth, frequency, equal_nan=False)
        snr[f"{frequency:g}Hz"] = _spectral_snr_db(output.data_uv[posterior][:, mask], device.sampling_rate_hz, float(frequency))

    display_metrics = []
    for stage, trace in zip(scenario.stages, stimulus_traces, strict=True):
        display_metrics.append({
            "stage": stage.label,
            "target_frequency_hz": stage.target_frequency_hz,
            "observed_frequency_hz": trace.observed_frequency_hz,
            "frequency_error_hz": (0.0 if stage.target_frequency_hz is None else abs(trace.observed_frequency_hz - stage.target_frequency_hz)),
            "frame_drop_fraction": trace.frame_drop_fraction,
            "interval_jitter_ms": trace.interval_jitter_ms,
        })

    report = {
        "schema": "neuros.synthetic_bci_arena.v1",
        "scenario": scenario.to_dict(),
        "paradigm": paradigm,
        "participant": asdict(participant),
        "world_model": asdict(model_profile),
        "world_model_evidence": evidence_card.to_dict(),
        "device": asdict(device),
        "display": asdict(display),
        "transport": asdict(transport),
        "metrics": {
            "duration_s": float(output.ground_truth_timestamps_s[-1] - output.ground_truth_timestamps_s[0]) if output.ground_truth_timestamps_s.size > 1 else 0.0,
            "samples": int(output.timestamps_s.size),
            "device_lsb_uv": output.lsb_uv,
            "device_clipped_fraction": output.clipped_fraction,
            "target_snr_db": snr,
            "transport": transport_metrics,
            "display": display_metrics,
            "world_model": {
                "name": model_profile.name,
                "display_coupled": bool(any(item.get("stimulus_coupling", 0.0) > 0 for item in latent_stage_end)),
                "stage_end_latent": latent_stage_end,
            },
        },
        "evidence_boundary": "Synthetic conformance evidence only; not human physiological performance.",
    }
    return ArenaRun(
        scenario=scenario,
        participant=participant,
        device=device,
        display=display,
        transport=transport,
        world_model=model_profile,
        device_output=output,
        packets=tuple(packets),
        ground_truth_target_hz=truth,
        stage_index=stage_index,
        stages=tuple(stage_intervals),
        stimulus_traces=tuple(stimulus_traces),
        report=report,
    )
