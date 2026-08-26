#!/usr/bin/env python3
"""Exercise the maintained neurOS developer-preview path from installed wheels.

This script is intentionally an *orchestrator*, not another runtime. It invokes
public console commands from the active Python environment and records exactly
which public seams succeeded. The repository checkout supplies only maintained
example configurations; the neurOS packages themselves must come from the
active environment.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "neuros.developer_preview_journey.v1"
REQUIRED_DISTRIBUTIONS = (
    "neuros-core",
    "neuros-drivers",
    "neuros-models",
    "neuros",
    "neuros-arena",
    "neuros-example-plugin",
)


@dataclass
class CommandResult:
    name: str
    argv: list[str]
    elapsed_s: float
    stdout: str
    stderr: str
    parsed_json: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": self.argv,
            "elapsed_s": self.elapsed_s,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "parsed_json": self.parsed_json,
        }


def _console_script(name: str) -> Path:
    """Return a console script from the active interpreter environment.

    Virtual-environment Python executables are commonly symlinks to the base
    interpreter. Resolving ``sys.executable`` therefore escapes the venv and can
    point at the host ``bin`` directory. ``sysconfig`` instead reports the
    scripts directory for the active environment on both POSIX and Windows.
    """

    scripts_dir = sysconfig.get_path("scripts")
    if not scripts_dir:
        raise RuntimeError("active Python environment does not expose a scripts directory")
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(scripts_dir) / f"{name}{suffix}"
    if not path.is_file():
        raise RuntimeError(
            f"required console script {name!r} is not installed in the active environment: {path}"
        )
    return path


def _distribution_identity(name: str, repo_root: Path) -> dict[str, Any]:
    dist = metadata.distribution(name)
    direct_url_text = dist.read_text("direct_url.json")
    direct_url: dict[str, Any] | None = None
    editable = False
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        editable = bool(direct_url.get("dir_info", {}).get("editable"))
    location = Path(dist.locate_file("")).resolve()
    if editable:
        raise RuntimeError(f"{name} is installed editable; developer-preview qualification requires wheels")
    try:
        location.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            f"{name} resolves inside the repository checkout ({location}); expected an installed wheel"
        )
    return {
        "name": dist.metadata["Name"],
        "version": dist.version,
        "location": str(location),
        "editable": editable,
        "direct_url": direct_url,
    }


def _run(
    results: list[CommandResult],
    name: str,
    argv: Sequence[str | Path],
    *,
    expect_json: bool = False,
) -> Any | None:
    rendered = [str(item) for item in argv]
    started = time.perf_counter()
    proc = subprocess.run(rendered, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    parsed: Any | None = None
    if expect_json and proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{name} succeeded but did not emit valid JSON: {exc}; stdout={proc.stdout!r}"
            ) from exc
    result = CommandResult(
        name=name,
        argv=rendered,
        elapsed_s=elapsed,
        stdout=proc.stdout,
        stderr=proc.stderr,
        parsed_json=parsed,
    )
    results.append(result)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {proc.returncode}\n"
            f"command: {' '.join(rendered)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return parsed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_journey(repo_root: Path, output_dir: Path, duration_s: float) -> dict[str, Any]:
    results: list[CommandResult] = []
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "resolved_executable": str(Path(sys.executable).resolve()),
            "prefix": sys.prefix,
            "scripts": sysconfig.get_path("scripts"),
            "platform": platform.platform(),
        },
        "distributions": {},
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "duration_s": duration_s,
        "commands": [],
        "artifacts": {},
    }

    try:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")

        mock_config = repo_root / "configs/examples/mock_bci.yaml"
        plugin_config = repo_root / "examples/plugins/neuros-example-plugin/example.yaml"
        for required in (mock_config, plugin_config):
            if not required.is_file():
                raise FileNotFoundError(required)

        neuros = _console_script("neuros")
        models = _console_script("neuros-models")
        arena = _console_script("neuros-arena")
        report["console_scripts"] = {
            "neuros": str(neuros),
            "neuros-models": str(models),
            "neuros-arena": str(arena),
        }

        identities: dict[str, Any] = {}
        report["distributions"] = identities
        for name in REQUIRED_DISTRIBUTIONS:
            identities[name] = _distribution_identity(name, repo_root)

        session_dir = output_dir / "session"
        qualification_dir = output_dir / "qualification"
        arena_report = output_dir / "arena-report.json"

        doctor = _run(results, "doctor", [neuros, "doctor", "--json"], expect_json=True)
        _require(isinstance(doctor, dict) and doctor.get("healthy") is True, "neuros doctor is not healthy")

        plugins = _run(results, "plugins", [neuros, "plugins", "--json"], expect_json=True)
        _require(isinstance(plugins, list), "plugin inventory must be a JSON list")
        discovered = {(item.get("kind"), item.get("name")) for item in plugins if isinstance(item, dict)}
        _require(("source", "example_sine") in discovered, "external example_sine source was not discovered")
        _require(("transform", "example_gain") in discovered, "external example_gain transform was not discovered")

        _run(results, "devices", [neuros, "devices", "--json"], expect_json=True)
        _run(results, "compatibility", [neuros, "compatibility", "--json"], expect_json=True)

        _run(results, "validate-mock", [neuros, "validate", mock_config, "--json"], expect_json=True)
        _run(
            results,
            "run-mock",
            [neuros, "run", mock_config, "--duration", str(duration_s), "--json"],
            expect_json=True,
        )

        _run(
            results,
            "record",
            [
                neuros,
                "record",
                mock_config,
                "--output",
                session_dir,
                "--session-id",
                "developer-preview",
                "--duration",
                str(duration_s),
                "--overwrite",
                "--json",
            ],
            expect_json=True,
        )
        _require(session_dir.is_dir(), "record command did not create a session archive")
        _run(
            results,
            "inspect-recording",
            [neuros, "inspect", session_dir, "--verify", "--json"],
            expect_json=True,
        )
        _run(
            results,
            "replay",
            [neuros, "replay", session_dir, "--config", mock_config, "--json"],
            expect_json=True,
        )

        qualify = _run(
            results,
            "qualify",
            [
                neuros,
                "qualify",
                mock_config,
                "--output",
                qualification_dir,
                "--session-id",
                "developer-preview",
                "--duration",
                str(duration_s),
                "--overwrite",
                "--json",
            ],
            expect_json=True,
        )
        _require(qualification_dir.is_dir(), "qualification command did not create a bundle")
        reproduce = _run(
            results,
            "reproduce",
            [neuros, "reproduce", qualification_dir, "--json"],
            expect_json=True,
        )

        model_card = _run(
            results,
            "model-card",
            [models, "show", "eeg-conformer", "--json"],
            expect_json=True,
        )
        _require(
            isinstance(model_card, dict) and model_card.get("id") == "eeg-conformer",
            "decoder-card inspection did not return eeg-conformer",
        )

        arena_summary = _run(
            results,
            "arena",
            [arena, "--preset", "dual-target-smoke", "--output", arena_report],
            expect_json=True,
        )
        _require(arena_report.is_file(), "Arena did not produce its report")
        arena_payload = json.loads(arena_report.read_text(encoding="utf-8"))
        _require(isinstance(arena_payload, dict) and arena_payload.get("schema"), "Arena report lacks schema identity")

        _run(
            results,
            "validate-external-plugin",
            [neuros, "validate", plugin_config, "--json"],
            expect_json=True,
        )
        _run(
            results,
            "run-external-plugin",
            [neuros, "run", plugin_config, "--duration", str(duration_s), "--json"],
            expect_json=True,
        )

        report["status"] = "pass"
        report["artifacts"] = {
            "session": str(session_dir),
            "qualification": str(qualification_dir),
            "arena_report": str(arena_report),
            "qualification_result": qualify,
            "reproduction_result": reproduce,
            "arena_summary": arena_summary,
        }
        return report
    except Exception as exc:
        report["status"] = "fail"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["commands"] = [item.to_dict() for item in results]
        _write_report(output_dir / "journey-report.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        report = run_journey(repo_root, output_dir, args.duration)
    except Exception as exc:
        print(f"developer preview journey: FAIL: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "schema": report["schema"],
        "status": report["status"],
        "report": str(output_dir / "journey-report.json"),
        "commands": len(report["commands"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
