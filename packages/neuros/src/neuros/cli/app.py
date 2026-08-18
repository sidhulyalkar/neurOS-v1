"""Argument parsing and dispatch for the neurOS command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neuros.errors import ConfigurationError
from neuros.plugins import PluginKind

from .config_commands import execute_config, validate_config
from .diagnostics import devices, doctor, plugin_inventory
from .legacy import handle as handle_legacy


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="neuros", description="neurOS BCI runtime CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Inspect installation and plugin health")
    _add_json_flag(doctor_parser)

    plugins_parser = subparsers.add_parser("plugins", help="List installed neurOS plugins")
    plugins_parser.add_argument("--kind", choices=[item.value for item in PluginKind])
    _add_json_flag(plugins_parser)

    devices_parser = subparsers.add_parser("devices", help="List installed source/device plugins")
    _add_json_flag(devices_parser)

    validate_parser = subparsers.add_parser("validate", help="Validate and resolve a runtime YAML config")
    validate_parser.add_argument("config", type=str)
    _add_json_flag(validate_parser)

    run_parser = subparsers.add_parser("run", help="Run a YAML runtime or the legacy mock demo")
    run_parser.add_argument("config", nargs="?", default=None, help="Pipeline YAML config")
    run_parser.add_argument("--duration", type=float, default=5.0, help="Run duration in seconds")
    run_parser.add_argument(
        "--until-complete",
        action="store_true",
        help="For finite replay/data sources, ignore --duration and run to source completion",
    )
    run_parser.add_argument("--show-outputs", action="store_true", help="Stream decoder outputs as JSONL")
    _add_json_flag(run_parser)

    bench_parser = subparsers.add_parser("benchmark", help="Benchmark a config or legacy synthetic pipeline")
    bench_parser.add_argument("config", nargs="?", default=None, help="Optional Pipeline YAML config")
    bench_parser.add_argument("--duration", type=float, default=10.0)
    bench_parser.add_argument("--report", type=str, default=None)
    _add_json_flag(bench_parser)

    train_parser = subparsers.add_parser("train", help="Train a legacy model on CSV data")
    train_parser.add_argument("--csv", type=str, required=True)

    save_model_parser = subparsers.add_parser("save-model", help="Save a model to the legacy registry")
    save_model_parser.add_argument("--model-file", type=str, required=True)
    save_model_parser.add_argument("--name", type=str, required=True)
    save_model_parser.add_argument("--version", type=str)
    save_model_parser.add_argument("--tags", nargs="+")
    save_model_parser.add_argument("--accuracy", type=float)

    load_model_parser = subparsers.add_parser("load-model", help="Load a model from the legacy registry")
    load_model_parser.add_argument("--name", type=str, required=True)
    load_model_parser.add_argument("--version", type=str)
    load_model_parser.add_argument("--output", type=str)

    list_models_parser = subparsers.add_parser("list-models", help="List models in the legacy registry")
    list_models_parser.add_argument("--filter", type=str)
    list_models_parser.add_argument("--tags", nargs="+")
    list_models_parser.add_argument("--format", choices=["table", "json"], default="table")

    subparsers.add_parser("dashboard", help="Launch the optional Streamlit dashboard")

    demo_parser = subparsers.add_parser("demo", help="Generate a task demonstration notebook")
    demo_parser.add_argument("--task", type=str, required=True)
    demo_parser.add_argument("--duration", type=float, default=3.0)
    demo_parser.add_argument("--output-dir", type=str, default="notebooks")

    run_tasks_parser = subparsers.add_parser("run-tasks", help="Run multiple legacy task descriptions")
    run_tasks_parser.add_argument("--tasks", nargs="+", required=True)
    run_tasks_parser.add_argument("--duration", type=float, default=3.0)

    serve_parser = subparsers.add_parser("serve", help="Launch the optional API server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    const_parser = subparsers.add_parser("constellation", help="Run the legacy Constellation demo")
    const_parser.add_argument("--duration", type=float, default=10.0)
    const_parser.add_argument("--output-dir", type=str, default="/tmp/constellation_demo")
    const_parser.add_argument("--subject-id", type=str, default="demo_subject")
    const_parser.add_argument("--session-id", type=str, default="demo_session")
    const_parser.add_argument("--fault-injection", action="store_true")
    const_parser.add_argument("--sagemaker-config", type=str, default=None)
    const_parser.add_argument("--kafka-bootstrap", type=str, default="localhost:9092")
    const_parser.add_argument("--topic-prefix", type=str, default="raw")
    const_parser.add_argument("--no-kafka", action="store_true")

    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _emit(value: Any, *, machine: bool = False) -> None:
    if machine:
        print(json.dumps(_jsonable(value), indent=2, sort_keys=True, default=str))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                print(f"{key}: {json.dumps(_jsonable(item), sort_keys=True, default=str)}")
            else:
                print(f"{key}: {item}")
        return
    print(value)


async def _print_output(output: Any) -> None:
    print(json.dumps(_jsonable(output), default=str, separators=(",", ":")))


def main() -> None:
    import neuros.cli as cli_api

    args = _parse_args()
    try:
        if args.command == "doctor":
            result = doctor()
            _emit(result, machine=args.json)
            if not result["healthy"]:
                raise SystemExit(1)
            return

        if args.command == "plugins":
            result = plugin_inventory()
            if args.kind:
                result = [item for item in result if item["kind"] == args.kind]
            _emit(result, machine=args.json)
            return

        if args.command == "devices":
            _emit(devices(), machine=args.json)
            return

        if args.command == "validate":
            _emit(validate_config(args.config), machine=args.json)
            return

        if args.command == "run" and args.config is not None:
            callback = _print_output if args.show_outputs else None
            duration = None if args.until_complete else args.duration
            result = cli_api.asyncio.run(
                execute_config(args.config, duration_s=duration, on_output=callback)
            )
            _emit(result, machine=args.json)
            return

        if args.command == "benchmark" and args.config is not None:
            result = cli_api.asyncio.run(
                execute_config(args.config, duration_s=args.duration)
            )
            text = json.dumps(_jsonable(result), indent=2, sort_keys=True, default=str)
            if args.report:
                Path(args.report).write_text(text, encoding="utf-8")
            else:
                _emit(result, machine=args.json)
            return

        if handle_legacy(args):
            return
        raise ConfigurationError(f"Unsupported command: {args.command}")

    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyError as exc:
        print(f"plugin error: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        raise SystemExit(4) from exc
