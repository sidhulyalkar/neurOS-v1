"""Deterministic closed-loop arena runner."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from neuros.plugins import PluginKind, load_plugin

from .evidence import evidence_card_for_model
from .participant import compile_participant_state_trace
from .simulators import DeviceOutput, StimulusTrace, TransportPacket, apply_device, packetize, sample_stimulus, simulate_stimulus
from .specs import ArenaScenario, DeviceProfile, DisplayProfile, ParticipantProfile, TransportProfile, WorldModelProfile
from .world_input import WorldInputBlock
from .world_models import NeuralWorldModel


# Neural-world integration cadence is an Arena execution policy. It is
# deliberately independent of DeviceProfile.chunk_samples, which belongs to the
# acquisition/packetization layer and must not change the simulated neural source.
WORLD_RENDER_CHUNK_SAMPLES = 5


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


def _stage_sample_count(duration_s: float, sampling_rate_hz: float) -> int:
    return max(1, int(round(float(duration_s) * float(sampling_rate_hz))))


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


def _artifact_identity_payload(artifact) -> str:
    payload = {
        "at_s": float(artifact.at_s),
        "channels": None if artifact.channels is None else sorted(str(name) for name in artifact.channels),
        "duration_s": float(artifact.duration_s),
        "kind": str(artifact.kind),
        "seed": None if artifact.seed is None else int(artifact.seed),
        "severity": float(artifact.severity),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _compile_artifact_schedule(
    model: NeuralWorldModel,
    scenario: ArenaScenario,
    participant: ParticipantProfile,
    sampling_rate_hz: float,
) -> tuple[str, list[dict[str, object]]]:
    schedule = getattr(model, "schedule_artifact", None)
    if not callable(schedule):
        return "legacy_injection", []

    compiled: list[dict[str, object]] = []
    global_sample = 0
    fs = float(sampling_rate_hz)
    for stage_i, stage in enumerate(scenario.stages):
        stage_samples = _stage_sample_count(stage.duration_s, fs)
        entries = [(_artifact_identity_payload(artifact), artifact_i, artifact) for artifact_i, artifact in enumerate(stage.artifacts)]
        entries.sort(key=lambda item: ("" if item[2].event_id is None else str(item[2].event_id), item[0], item[1]))
        duplicate_ordinals: dict[str, int] = {}
        for identity_payload, artifact_i, artifact in entries:
            onset_sample = int(np.ceil(artifact.at_s * fs - 1e-12))
            if onset_sample >= stage_samples:
                raise ValueError(
                    f"artifact {artifact_i} in stage {stage.label!r} has no source sample "
                    f"at-or-after onset {artifact.at_s:g}s at {fs:g} Hz"
                )
            if artifact.event_id is not None:
                event_id = artifact.event_id
            else:
                digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:16]
                ordinal = duplicate_ordinals.get(identity_payload, 0)
                duplicate_ordinals[identity_payload] = ordinal + 1
                event_id = f"{scenario.name}/stage-{stage_i}/{digest}/{ordinal}"
            resolved = schedule(
                artifact.kind,
                event_id=event_id,
                start_sample=global_sample + onset_sample,
                duration_seconds=artifact.duration_s,
                severity=artifact.severity * participant.artifact_gain,
                channels=artifact.channels,
                seed=artifact.seed,
            )
            event_payload = (
                resolved.to_dict()
                if hasattr(resolved, "to_dict") and callable(resolved.to_dict)
                else {
                    "event_id": event_id,
                    "kind": artifact.kind,
                    "start_sample": global_sample + onset_sample,
                    "duration_seconds": artifact.duration_s,
                    "severity": artifact.severity * participant.artifact_gain,
                    "channels": None if artifact.channels is None else list(artifact.channels),
                    "seed": artifact.seed,
                }
            )
            compiled.append({
                "stage": stage.label,
                "stage_index": stage_i,
                "artifact_index": artifact_i,
                "requested_at_s": float(artifact.at_s),
                **event_payload,
            })
        global_sample += stage_samples
    return "sample_indexed", compiled


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
    artifact_execution, compiled_artifacts = _compile_artifact_schedule(
        model, scenario, participant, device.sampling_rate_hz
    )

    fs = float(device.sampling_rate_hz)
    participant_trace = compile_participant_state_trace(scenario, participant, fs)
    block_samples = WORLD_RENDER_CHUNK_SAMPLES
    data_blocks: list[np.ndarray] = []
    timestamp_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    stage_blocks: list[np.ndarray] = []
    stage_intervals: list[StageInterval] = []
    stage_timing: list[dict[str, float | int | str]] = []
    stimulus_traces: list[StimulusTrace] = []
    latent_stage_end: list[dict[str, float | str]] = []
    global_sample_start = 0

    for stage_i, stage in enumerate(scenario.stages):
        samples_total = _stage_sample_count(stage.duration_s, fs)
        stage_start = global_sample_start / fs
        stage_end = (global_sample_start + samples_total) / fs
        resolved_duration = samples_total / fs
        stage_intervals.append(StageInterval(stage.label, stage_start, stage_end, stage.target_frequency_hz))
        stage_timing.append({
            "stage": stage.label,
            "stage_index": stage_i,
            "start_sample": global_sample_start,
            "end_sample": global_sample_start + samples_total,
            "resolved_start_s": float(stage_start),
            "resolved_end_s": float(stage_end),
            "requested_duration_s": float(stage.duration_s),
            "resolved_duration_s": float(resolved_duration),
            "duration_error_ms": float((resolved_duration - stage.duration_s) * 1000.0),
        })
        trace = simulate_stimulus(
            stage.target_frequency_hz,
            stage.duration_s,
            display,
            seed=scenario.seed * 1009 + stage_i,
        )
        stimulus_traces.append(trace)
        artifact_cursor: set[int] = set()
        produced = 0
        last_latent: dict[str, float] = {}
        while produced < samples_total:
            count = min(block_samples, samples_total - produced)
            sample_offsets = produced + np.arange(count, dtype=np.int64)
            global_indices = global_sample_start + sample_offsets
            local_times = sample_offsets.astype(float) / fs
            global_times = global_indices.astype(float) / fs
            elapsed = produced / fs
            if artifact_execution == "legacy_injection":
                for artifact_i, artifact in enumerate(stage.artifacts):
                    if artifact_i not in artifact_cursor and elapsed <= artifact.at_s < elapsed + count / fs:
                        model.inject_artifact(
                            artifact.kind,
                            duration_seconds=artifact.duration_s,
                            severity=artifact.severity * participant.artifact_gain,
                        )
                        artifact_cursor.add(artifact_i)

            attention_stream = participant_trace.attention_gain[global_indices]
            requested_attention_stream = participant_trace.requested_attention_gain[global_indices]
            target_switch_stream = participant_trace.target_switch[global_indices].astype(float)
            # Legacy render(...) plugins retain the historical scalar surface. The
            # first source sample is the least surprising summary because the old
            # runner updated the value at block start.
            effective_gain = float(attention_stream[0]) if attention_stream.size else 0.0
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
                participant_state={"attention_gain": effective_gain},
                participant_streams={
                    "attention_gain": attention_stream,
                    "requested_attention_gain": requested_attention_stream,
                    "target_switch": target_switch_stream,
                },
            )
            emission = _render_world_model(
                model,
                input_block=input_block,
                target_frequency_hz=stage.target_frequency_hz,
                attention_gain=effective_gain,
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
        global_sample_start += samples_total

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
        "schema": "neuros.synthetic_bci_arena.v2",
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
            "stage_timing": stage_timing,
            "participant_state": participant_trace.to_summary(),
            "world_model": {
                "name": model_profile.name,
                "display_coupled": bool(any(item.get("stimulus_coupling", 0.0) > 0 for item in latent_stage_end)),
                "render_chunk_samples": block_samples,
                "artifact_execution": artifact_execution,
                "compiled_artifact_schedule": compiled_artifacts,
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
