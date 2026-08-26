#!/usr/bin/env python3
"""Publish a device-faithful synthetic Unicorn Hybrid Black stream over LSL.

This endpoint is a hardware-substitution tool, not a physical-hardware claim.
It can expose one of three schemas:

* ``eeg8_anatomical``: 8 EEG channels labeled by the standard cap montage;
* ``device17_api``: 17 acquired channels in the device/API order;
* ``recorder19``: Recorder/network-style 19-field view including DT/STATUS.

Local controls (stdin or UDP 127.0.0.1:19744):

    0 / 1 / 2              rest / attend 10 Hz / attend 12 Hz
    b / j / c / m / s      blink / jaw / controller / motion / saturation
    validation:0|1         validation-indicator state
    status:INTEGER         Recorder/status trigger value
    still                  accel=(0,0,1), gyro=(0,0,0)
    turn                   deterministic head-turn telemetry
    shake                  deterministic larger motion telemetry + EEG motion artifact
    silence:SECONDS        stop publishing while the synthetic device keeps time
    gain:VALUE             synthetic SSVEP response gain
    q                      quit

The stream metadata declares the synthetic provenance explicitly.  It does not
claim to reproduce undocumented Unicorn LSL XML exactly.
"""
from __future__ import annotations

import argparse
import queue
import socket
import threading
import time

import numpy as np

from neuros.drivers.unicorn_hybrid_black_sim import (
    UnicornHybridBlackSimulationConfig,
    UnicornHybridBlackSimulator,
    UnicornHybridBlackSpec,
)


def _stdin_reader(commands: queue.Queue[str]) -> None:
    while True:
        try:
            command = input().strip().lower()
        except EOFError:
            return
        if command:
            commands.put(command)
        if command == "q":
            return


def _udp_reader(commands: queue.Queue[str], host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.25)
    try:
        while True:
            try:
                payload, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            command = payload.decode("utf-8", errors="ignore").strip().lower()
            if command:
                commands.put(command)
            if command == "q":
                return
    finally:
        sock.close()


def _parse_float(command: str, prefix: str, default: float) -> float:
    if not command.startswith(prefix + ":"):
        return default
    try:
        return float(command.split(":", 1)[1])
    except ValueError:
        return default


def _channel_type(name: str) -> str:
    upper = name.upper()
    if name.startswith("EEG") or name in {"Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"}:
        return "EEG"
    if "ACCELEROMETER" in upper or upper.startswith("ACC "):
        return "Accelerometer"
    if "GYROSCOPE" in upper or upper.startswith("GYR "):
        return "Gyroscope"
    if "BAT" in upper:
        return "Battery"
    if "VALID" in upper:
        return "Validation"
    if "COUNT" in upper or upper == "CNT":
        return "Counter"
    if upper == "DT":
        return "Timing"
    if upper == "STATUS":
        return "Status"
    return "MISC"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema",
        choices=("eeg8_anatomical", "device17_api", "recorder19"),
        default="device17_api",
    )
    parser.add_argument("--name", default="UnicornMock")
    parser.add_argument("--source-id", default="neuros-unicorn-hybrid-black-sim")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--drop-probability", type=float, default=0.0)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--device-delay-jitter-ms", type=float, default=0.0)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=19744)
    parser.add_argument("--no-stdin", action="store_true")
    args = parser.parse_args()
    if args.block <= 0:
        raise SystemExit("--block must be positive")
    if not 0.0 <= args.drop_probability < 1.0:
        raise SystemExit("--drop-probability must be in [0,1)")
    if args.jitter_ms < 0 or args.device_delay_jitter_ms < 0:
        raise SystemExit("jitter values must be non-negative")
    if not 1 <= args.control_port <= 65535:
        raise SystemExit("--control-port must be a valid UDP port")

    try:
        from pylsl import StreamInfo, StreamOutlet, local_clock
    except (ImportError, OSError, RuntimeError) as exc:
        raise SystemExit("Install neuros-drivers[lsl] with a working liblsl runtime") from exc

    spec = UnicornHybridBlackSpec()
    sim = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(
            schema=args.schema,
            seed=args.seed,
            acquisition_delay_jitter_ms=args.device_delay_jitter_ms,
        )
    )
    preview = sim.render(1)
    # Re-create after metadata discovery so the published sample counter starts
    # exactly at the configured origin.
    sim = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(
            schema=args.schema,
            seed=args.seed,
            acquisition_delay_jitter_ms=args.device_delay_jitter_ms,
        )
    )

    info = StreamInfo(
        args.name,
        "EEG",
        len(preview.channel_names),
        spec.sampling_rate_hz,
        "float32",
        args.source_id,
    )
    channels = info.desc().append_child("channels")
    for label, unit in zip(preview.channel_names, preview.channel_units, strict=True):
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", unit)
        channel.append_child_value("type", _channel_type(label))

    provenance = info.desc().append_child("provenance")
    provenance.append_child_value("synthetic", "true")
    provenance.append_child_value("producer", "neurOS")
    provenance.append_child_value("emulated_manufacturer", "g.tec medical engineering")
    provenance.append_child_value("emulated_device", "Unicorn Hybrid Black")
    provenance.append_child_value("contract", "neuros.unicorn_hybrid_black_sim.lsl.v1")
    provenance.append_child_value("schema", args.schema)
    provenance.append_child_value("evidence_boundary", "synthetic device substitution; not physical hardware")

    hardware = info.desc().append_child("emulated_hardware")
    hardware.append_child_value("sampling_rate_hz", str(spec.sampling_rate_hz))
    hardware.append_child_value("resolution_bits", str(spec.resolution_bits))
    hardware.append_child_value("sensitivity_uv", str(spec.sensitivity_uv))
    hardware.append_child_value("device_delay_ms", str(spec.device_delay_ms))
    hardware.append_child_value("eeg_lsb_uv", str(spec.eeg_lsb_uv))
    hardware.append_child_value("montage", "Fz,C3,Cz,C4,Pz,PO7,Oz,PO8")

    simulation = info.desc().append_child("simulation")
    simulation.append_child_value("seed", str(args.seed))
    simulation.append_child_value("drop_probability", str(args.drop_probability))
    simulation.append_child_value("delivery_jitter_ms", str(args.jitter_ms))
    simulation.append_child_value("device_delay_jitter_ms", str(args.device_delay_jitter_ms))
    simulation.append_child_value("control_host", args.control_host)
    simulation.append_child_value("control_port", str(args.control_port))

    outlet = StreamOutlet(info, chunk_size=args.block, max_buffered=10)
    commands: queue.Queue[str] = queue.Queue()
    if not args.no_stdin:
        threading.Thread(target=_stdin_reader, args=(commands,), daemon=True).start()
    threading.Thread(
        target=_udp_reader,
        args=(commands, args.control_host, args.control_port),
        daemon=True,
    ).start()

    transport_rng = np.random.default_rng(args.seed + 91009)
    attention_gain = 1.0
    silence_until = 0.0
    running = True
    deadline = time.monotonic()
    print(
        f"Publishing {args.name!r} schema={args.schema} channels={len(preview.channel_names)} "
        f"at 250 Hz; synthetic Unicorn device delay={spec.device_delay_ms:g} ms."
    )

    while running:
        while True:
            try:
                command = commands.get_nowait()
            except queue.Empty:
                break
            if command == "0":
                sim.eeg.set_attention(None)
            elif command == "1":
                sim.eeg.set_attention(10.0, attention_gain)
            elif command == "2":
                sim.eeg.set_attention(12.0, attention_gain)
            elif command == "b":
                sim.eeg.inject_artifact("blink", 0.35, 1.0)
            elif command == "j":
                sim.eeg.inject_artifact("jaw", 0.45, 1.0)
            elif command == "c":
                sim.eeg.inject_artifact("controller", 0.50, 1.0)
            elif command == "m":
                sim.eeg.inject_artifact("motion", 0.50, 1.0)
            elif command == "s":
                sim.eeg.inject_artifact("saturation", 0.40, 1.0)
            elif command == "still":
                sim.set_motion((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
            elif command == "turn":
                sim.set_motion((0.10, 0.05, 0.98), (0.0, 55.0, 8.0))
            elif command == "shake":
                sim.set_motion((0.75, -0.35, 1.20), (150.0, -90.0, 65.0))
                sim.eeg.inject_artifact("motion", 0.75, 1.4)
            elif command.startswith("validation:"):
                try:
                    sim.set_validation(int(command.split(":", 1)[1]))
                except ValueError:
                    print("validation must be 0 or 1")
            elif command.startswith("status:"):
                try:
                    sim.set_status(int(command.split(":", 1)[1]))
                except ValueError:
                    print("status must be an integer")
            elif command.startswith("silence:"):
                duration = float(np.clip(_parse_float(command, "silence", 2.0), 0.1, 30.0))
                silence_until = max(silence_until, time.monotonic() + duration)
            elif command.startswith("gain:"):
                attention_gain = float(np.clip(_parse_float(command, "gain", attention_gain), 0.0, 1.5))
                if sim.eeg.target_frequency_hz is not None:
                    sim.eeg.set_attention(sim.eeg.target_frequency_hz, attention_gain)
            elif command == "q":
                running = False

        block = sim.render(args.block)
        dropped = transport_rng.random() < args.drop_probability
        silent = time.monotonic() < silence_until
        if not dropped and not silent:
            if args.jitter_ms > 0:
                time.sleep(float(transport_rng.uniform(0.0, args.jitter_ms)) / 1000.0)
            # LSL's explicit timestamp describes the newest sample in the chunk.
            # Mark it as approximately 40 ms old at host availability, preserving
            # acquisition latency as timestamp age instead of moving neural time.
            newest_sample_timestamp = local_clock() - spec.device_delay_ms / 1000.0
            outlet.push_chunk(block.data.T.tolist(), timestamp=newest_sample_timestamp)

        deadline += args.block / spec.sampling_rate_hz
        time.sleep(max(0.0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()
