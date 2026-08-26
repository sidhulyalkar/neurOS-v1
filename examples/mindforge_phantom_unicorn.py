#!/usr/bin/env python3
"""Publish neurOS synthetic EEG as a controllable Unicorn-like LSL source.

Keyboard or localhost UDP commands:
  1 = attend 10 Hz, 2 = attend 12 Hz, 0 = no target
  b = blink, j = jaw EMG, c = controller EMG, m = motion
  s = saturation, d = Oz dropout, r = restore Oz
  x = two-second stream silence, silence:2.5 = explicit silence duration
  + / - = SSVEP gain step, gain:0.65 = explicit response gain
  q = quit

The UDP control path is intentionally localhost-only by default. It makes Mindforge
calibration/torture rehearsals reproducible without changing the generated EEG or LSL
contract used by the physical-source path.
"""
from __future__ import annotations

import argparse
import queue
import socket
import threading
import time

import numpy as np

from neuros.drivers.synthetic_eeg import SyntheticEEGConfig, SyntheticEEGGenerator


def command_reader(commands: queue.Queue[str]) -> None:
    while True:
        try:
            command = input().strip().lower()
        except EOFError:
            return
        if command:
            commands.put(command)
        if command == "q":
            return


def udp_command_reader(commands: queue.Queue[str], host: str, port: int) -> None:
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


def parse_duration(command: str, prefix: str, default: float) -> float:
    if not command.startswith(prefix + ":"):
        return default
    try:
        return float(command.split(":", 1)[1])
    except ValueError:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="UnicornMock")
    parser.add_argument("--source-id", default="mindforge-phantom-unicorn")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--drop-probability", type=float, default=0.0)
    parser.add_argument("--jitter-ms", type=float, default=0.0,
                        help="uniform extra delivery delay in [0, jitter-ms]")
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=19744)
    parser.add_argument("--no-stdin", action="store_true",
                        help="disable interactive stdin; use UDP control only")
    args = parser.parse_args()
    if not 0.0 <= args.drop_probability < 1.0:
        raise SystemExit("--drop-probability must be in [0, 1)")
    if args.jitter_ms < 0:
        raise SystemExit("--jitter-ms must be non-negative")
    if not 1 <= args.control_port <= 65535:
        raise SystemExit("--control-port must be a valid UDP port")

    try:
        from pylsl import StreamInfo, StreamOutlet, local_clock
    except (ImportError, OSError, RuntimeError) as exc:
        raise SystemExit("Install neuros-drivers[lsl] with a working liblsl runtime") from exc

    config = SyntheticEEGConfig(seed=args.seed)
    generator = SyntheticEEGGenerator(config)
    transport_rng = np.random.default_rng(args.seed + 99173)
    info = StreamInfo(args.name, "EEG", len(config.channel_names), config.sampling_rate_hz,
                      "float32", args.source_id)
    channels = info.desc().append_child("channels")
    for label in config.channel_names:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "microvolts")
        channel.append_child_value("type", "EEG")
    info.desc().append_child_value("manufacturer", "neurOS synthetic source")
    info.desc().append_child_value("synthetic", "true")
    info.desc().append_child_value("transport_drop_probability", str(args.drop_probability))
    info.desc().append_child_value("transport_jitter_ms", str(args.jitter_ms))
    info.desc().append_child_value("control_port", str(args.control_port))
    outlet = StreamOutlet(info, chunk_size=args.block, max_buffered=10)

    commands: queue.Queue[str] = queue.Queue()
    if not args.no_stdin:
        threading.Thread(target=command_reader, args=(commands,), daemon=True,
                         name="PhantomUnicorn-stdin").start()
    threading.Thread(target=udp_command_reader,
                     args=(commands, args.control_host, args.control_port), daemon=True,
                     name="PhantomUnicorn-control").start()

    attention_gain = 1.0
    silence_until = 0.0
    running = True
    print(
        f"Publishing {args.name!r} at {config.sampling_rate_hz:g} Hz; "
        f"drop={args.drop_probability:.3f}, jitter<= {args.jitter_ms:g} ms; "
        f"control=udp://{args.control_host}:{args.control_port}."
    )
    print("Commands: 1/2/0/b/j/c/m/s/d/r/x/silence:SECONDS/+/-/gain:VALUE/q")
    deadline = time.monotonic()

    while running:
        while True:
            try:
                command = commands.get_nowait()
            except queue.Empty:
                break

            if command == "1":
                generator.set_attention(10.0, attention_gain)
            elif command == "2":
                generator.set_attention(12.0, attention_gain)
            elif command == "0":
                generator.set_attention(None)
            elif command == "b":
                generator.inject_artifact("blink", 0.35, 1.0)
            elif command == "j":
                generator.inject_artifact("jaw", 0.45, 1.0)
            elif command == "c":
                generator.inject_artifact("controller", 0.50, 1.0)
            elif command == "m":
                generator.inject_artifact("motion", 0.50, 1.0)
            elif command == "s":
                generator.inject_artifact("saturation", 0.40, 1.0)
            elif command == "d":
                generator.set_channel_gain("Oz", 0.0)
            elif command == "r":
                generator.set_channel_gain("Oz", 1.0)
            elif command == "x" or command.startswith("silence:"):
                duration = float(np.clip(parse_duration(command, "silence", 2.0), 0.1, 10.0))
                silence_until = max(silence_until, time.monotonic() + duration)
            elif command == "+":
                attention_gain = min(1.5, attention_gain + 0.1)
                if generator.target_frequency_hz is not None:
                    generator.set_attention(generator.target_frequency_hz, attention_gain)
            elif command == "-":
                attention_gain = max(0.0, attention_gain - 0.1)
                if generator.target_frequency_hz is not None:
                    generator.set_attention(generator.target_frequency_hz, attention_gain)
            elif command.startswith("gain:"):
                attention_gain = float(np.clip(parse_duration(command, "gain", attention_gain), 0.0, 1.5))
                if generator.target_frequency_hz is not None:
                    generator.set_attention(generator.target_frequency_hz, attention_gain)
            elif command == "q":
                running = False

        block = generator.render(args.block)
        generated_timestamp = local_clock()
        dropped = transport_rng.random() < args.drop_probability
        silent = time.monotonic() < silence_until
        if not dropped and not silent:
            if args.jitter_ms > 0:
                time.sleep(float(transport_rng.uniform(0.0, args.jitter_ms)) / 1000.0)
            outlet.push_chunk(block.data_uv.T.tolist(), timestamp=generated_timestamp)

        deadline += args.block / config.sampling_rate_hz
        time.sleep(max(0.0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()
