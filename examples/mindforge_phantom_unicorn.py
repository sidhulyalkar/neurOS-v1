#!/usr/bin/env python3
"""Publish neurOS synthetic EEG as a controllable Unicorn-like LSL source.

Keyboard or localhost UDP commands:
  1 = attend 10 Hz, 2 = attend 12 Hz, 0 = no target
  b = blink, j = jaw EMG, c = controller EMG, m = motion
  s = source-offset stressor, d = persistent Oz mask, r = restore Oz
  artifact:ID:KIND:START_SAMPLE:DURATION:SEVERITY[:CHANNELS][:SEED]
      = schedule an exact future artifact; CHANNELS is comma-separated or '*'
  cancel:ID = cancel a scheduled artifact that has not completed
  x = two-second stream silence, silence:2.5 = explicit silence duration
  + / - = SSVEP gain step, gain:0.65 = explicit response gain
  q = quit

The UDP control path is intentionally localhost-only by default. Immediate
single-key artifact commands preserve the historical replacement behavior for
manual rehearsals. Explicit ``artifact:...`` commands use the v3 sample-indexed
scheduler so multiple future nuisances can overlap reproducibly.

Source-level ``s``/``saturation`` is a synthetic source-offset stressor, not a
claim about physical Unicorn amplifier saturation. Physical sensitivity clipping
and quantization belong to the Unicorn device simulation layer.
"""
from __future__ import annotations

import argparse
import queue
import socket
import threading
import time

import numpy as np

from neuros.drivers.synthetic_eeg import (
    SUPPORTED_ARTIFACTS,
    ArtifactEvent,
    SyntheticEEGConfig,
    SyntheticEEGGenerator,
)
from neuros.drivers.synthetic_eeg_driver import (
    SYNTHETIC_EEG_ARTIFACT_SCHEDULER_CONTRACT,
    SYNTHETIC_EEG_GENERATOR_CONTRACT,
)


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
        value = float(command.split(":", 1)[1])
    except ValueError:
        return default
    return value if np.isfinite(value) else default


def parse_artifact_schedule_command(command: str) -> dict[str, object]:
    """Parse one exact sample-indexed Phantom artifact command.

    Syntax:

    ``artifact:ID:KIND:START_SAMPLE:DURATION_SECONDS:SEVERITY[:CHANNELS][:SEED]``

    ``CHANNELS`` is a comma-separated scalp-label list or ``*`` for the
    artifact's default support. Parsing is deliberately strict: sample indices
    and seeds are integers, and floating controls must be finite. The generator
    remains authoritative for channel-name and already-rendered-past checks.
    """

    parts = command.strip().split(":")
    if not 6 <= len(parts) <= 8 or parts[0].lower() != "artifact":
        raise ValueError(
            "artifact command must be artifact:ID:KIND:START_SAMPLE:DURATION:SEVERITY"
            "[:CHANNELS][:SEED]"
        )
    _, event_id, kind_text, start_text, duration_text, severity_text, *optional = parts
    event_id = event_id.strip()
    kind = kind_text.strip().lower()
    if not event_id:
        raise ValueError("artifact event ID must be non-empty")
    if kind not in SUPPORTED_ARTIFACTS:
        raise ValueError(f"unsupported artifact kind: {kind}")
    try:
        start_sample = int(start_text)
    except ValueError as exc:
        raise ValueError("artifact START_SAMPLE must be an integer") from exc
    if start_sample < 0:
        raise ValueError("artifact START_SAMPLE must be non-negative")
    try:
        duration_seconds = float(duration_text)
        severity = float(severity_text)
    except ValueError as exc:
        raise ValueError("artifact duration/severity must be numeric") from exc
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("artifact duration must be positive and finite")
    if not np.isfinite(severity) or severity < 0:
        raise ValueError("artifact severity must be non-negative and finite")

    channels: tuple[str, ...] | None = None
    if optional:
        channel_text = optional[0].strip()
        if channel_text and channel_text != "*":
            channels = tuple(
                channel.strip() for channel in channel_text.split(",") if channel.strip()
            )
            if not channels:
                raise ValueError("artifact channel list must not be empty")

    seed: int | None = None
    if len(optional) == 2:
        seed_text = optional[1].strip()
        if not seed_text:
            raise ValueError("artifact seed must not be empty")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError("artifact seed must be an integer") from exc
        if seed < 0:
            raise ValueError("artifact seed must be non-negative")

    return {
        "kind": kind,
        "event_id": event_id,
        "start_sample": start_sample,
        "duration_seconds": duration_seconds,
        "severity": severity,
        "channels": channels,
        "seed": seed,
    }


def schedule_artifact_command(
    generator: SyntheticEEGGenerator,
    command: str,
) -> ArtifactEvent:
    """Parse and apply one exact Phantom artifact schedule command.

    The historical command readers normalize input to lowercase. Resolve channel
    labels case-insensitively back to the generator's canonical montage names so
    `PO7,Oz` and the normalized `po7,oz` address the same channels without making
    the core generator itself case-insensitive.
    """

    parsed = parse_artifact_schedule_command(command)
    channels = parsed["channels"]
    if channels is not None:
        lookup = {name.casefold(): name for name in generator.config.channel_names}
        canonical: list[str] = []
        for channel in channels:
            resolved = lookup.get(str(channel).casefold())
            if resolved is None:
                raise ValueError(f"unknown channel: {channel!r}")
            canonical.append(resolved)
        parsed["channels"] = tuple(canonical)
    return generator.schedule_artifact(**parsed)


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
    if not 0.0 <= args.drop_probability < 1.0 or not np.isfinite(args.drop_probability):
        raise SystemExit("--drop-probability must be finite and in [0, 1)")
    if args.jitter_ms < 0 or not np.isfinite(args.jitter_ms):
        raise SystemExit("--jitter-ms must be non-negative and finite")
    if args.block <= 0:
        raise SystemExit("--block must be positive")
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
    info.desc().append_child_value("generator_contract", SYNTHETIC_EEG_GENERATOR_CONTRACT)
    info.desc().append_child_value(
        "artifact_scheduler_contract",
        SYNTHETIC_EEG_ARTIFACT_SCHEDULER_CONTRACT,
    )
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
    print(
        "Commands: 1/2/0/b/j/c/m/s/d/r/artifact:.../cancel:ID/"
        "x/silence:SECONDS/+/-/gain:VALUE/q"
    )
    deadline = time.monotonic()

    while running:
        while True:
            try:
                command = commands.get_nowait()
            except queue.Empty:
                break

            try:
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
                elif command.startswith("artifact:"):
                    event = schedule_artifact_command(generator, command)
                    print(
                        "scheduled artifact "
                        f"{event.event_id!r} kind={event.kind} "
                        f"samples=[{event.start_sample},{event.end_sample})"
                    )
                elif command.startswith("cancel:"):
                    event_id = command.split(":", 1)[1].strip()
                    if not event_id:
                        raise ValueError("cancel event ID must be non-empty")
                    print(
                        f"cancel artifact {event_id!r}: "
                        f"{'removed' if generator.cancel_artifact(event_id) else 'not-found'}"
                    )
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
                    attention_gain = float(
                        np.clip(parse_duration(command, "gain", attention_gain), 0.0, 1.5)
                    )
                    if generator.target_frequency_hz is not None:
                        generator.set_attention(generator.target_frequency_hz, attention_gain)
                elif command == "q":
                    running = False
                else:
                    print(f"ignored unknown command: {command!r}")
            except (IndexError, TypeError, ValueError) as exc:
                # A malformed control command must not kill the source or mutate
                # itself into a different scenario. Reject it visibly instead.
                print(f"rejected control command {command!r}: {exc}")

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
