#!/usr/bin/env python3
"""Clean-room qualification of the default neurOS installed-wheel product path.

This harness deliberately sits above unit/integration tests.  It builds the
policy-selected default release wheels, installs them into a fresh virtual
environment, then exercises public CLI and contract surfaces from outside the
repository checkout.  It also mutates authority-bearing bytes and requires the
public verification/reproduction paths to fail closed.

The output is a machine-readable evidence report.  Passing this harness proves a
software packaging/runtime/replay integrity boundary only; it does not establish
real neural efficacy, hardware validity, closed-loop safety, or clinical value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
import venv
from dataclasses import dataclass
from email.parser import Parser
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence
from zipfile import ZipFile

SCHEMA = "neuros.full_package_qualification.v1"
REQUIRED_DISTRIBUTIONS = (
    "neuros-core",
    "neuros-drivers",
    "neuros-models",
    "neuros",
)
NONDEFAULT_DISTRIBUTIONS = (
    "neuros-arena",
    "neuros-foundation",
    "neuros-mechint",
    "neuros-neurofm",
    "neuros-sourceweigher",
    "neuros-orion",
    "neuros-ui",
    "neuros-cloud",
)


@dataclass(slots=True)
class CommandResult:
    name: str
    argv: list[str]
    returncode: int
    elapsed_s: float
    stdout: str
    stderr: str
    expected: str
    parsed_json: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": self.argv,
            "returncode": self.returncode,
            "elapsed_s": self.elapsed_s,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "expected": self.expected,
            "parsed_json": self.parsed_json,
        }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _wheel_metadata(path: Path) -> dict[str, Any]:
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise RuntimeError(f"{path.name}: expected one METADATA file, found {len(names)}")
        message = Parser().parsestr(archive.read(names[0]).decode("utf-8", errors="strict"))
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise RuntimeError(f"{path.name}: missing Name/Version metadata")
    return {
        "name": str(name),
        "canonical_name": _canonical_name(str(name)),
        "version": str(version),
        "file": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _python_in_venv(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _console_script(name: str) -> Path:
    scripts = sysconfig.get_path("scripts")
    if not scripts:
        raise RuntimeError("active environment does not expose a scripts directory")
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(scripts) / f"{name}{suffix}"
    if not path.is_file():
        raise RuntimeError(f"required console script is missing: {path}")
    return path


def _run(
    results: list[CommandResult],
    name: str,
    argv: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    expect_json: bool = False,
    expected_codes: Iterable[int] = (0,),
) -> Any | None:
    rendered = [str(value) for value in argv]
    expected = tuple(int(value) for value in expected_codes)
    started = time.perf_counter()
    proc = subprocess.run(
        rendered,
        cwd=None if cwd is None else str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    parsed: Any | None = None
    if expect_json and proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{name} returned success without valid JSON: {exc}; stdout={proc.stdout!r}"
            ) from exc
    result = CommandResult(
        name=name,
        argv=rendered,
        returncode=proc.returncode,
        elapsed_s=elapsed,
        stdout=proc.stdout,
        stderr=proc.stderr,
        expected="return code in " + repr(expected),
        parsed_json=parsed,
    )
    results.append(result)
    if proc.returncode not in expected:
        raise RuntimeError(
            f"{name} returned {proc.returncode}, expected one of {expected}\n"
            f"command: {' '.join(rendered)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return parsed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _flip_one_byte(path: Path) -> dict[str, Any]:
    payload = bytearray(path.read_bytes())
    if not payload:
        raise RuntimeError(f"cannot tamper with empty file: {path}")
    index = len(payload) // 2
    before = hashlib.sha256(payload).hexdigest()
    payload[index] ^= 0x01
    path.write_bytes(payload)
    after = hashlib.sha256(payload).hexdigest()
    _require(before != after, "tamper helper did not change file identity")
    return {"path": str(path), "byte_offset": index, "sha256_before": before, "sha256_after": after}


def _append_tamper(path: Path) -> dict[str, Any]:
    before = _sha256_file(path)
    with path.open("ab") as handle:
        handle.write(b" \n")
    after = _sha256_file(path)
    _require(before != after, "tamper helper did not change file identity")
    return {"path": str(path), "sha256_before": before, "sha256_after": after}


def _distribution_identity(name: str, repo_root: Path, expected_sha256: str) -> dict[str, Any]:
    dist = metadata.distribution(name)
    location = Path(dist.locate_file("")).resolve()
    try:
        location.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise RuntimeError(f"{name} resolves inside the repository checkout: {location}")

    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError(f"{name} has no direct_url.json; exact wheel origin is not observable")
    direct_url = json.loads(direct_url_text)
    archive_info = direct_url.get("archive_info", {})
    hashes = archive_info.get("hashes", {})
    observed_sha = hashes.get("sha256")
    if observed_sha is None:
        legacy = archive_info.get("hash")
        if isinstance(legacy, str) and legacy.startswith("sha256="):
            observed_sha = legacy.split("=", 1)[1]
    if observed_sha != expected_sha256:
        raise RuntimeError(
            f"{name} installed wheel digest differs from built artifact: "
            f"expected {expected_sha256}, observed {observed_sha}"
        )

    return {
        "name": str(dist.metadata["Name"]),
        "version": dist.version,
        "location": str(location),
        "editable": bool(direct_url.get("dir_info", {}).get("editable")),
        "wheel_sha256": observed_sha,
        "direct_url": direct_url,
    }


def _runtime_semantics(snapshot: dict[str, Any]) -> dict[str, Any]:
    nodes = snapshot.get("nodes", {})
    edges = snapshot.get("edges", {})
    return {
        "state": snapshot.get("state"),
        "node_processed": {
            str(name): int(value.get("processed", 0)) for name, value in sorted(nodes.items())
        },
        "node_failed": {
            str(name): int(value.get("failed", 0)) for name, value in sorted(nodes.items())
        },
        "edge_accepted": {
            str(name): int(value.get("accepted", 0)) for name, value in sorted(edges.items())
        },
        "edge_dropped": {
            str(name): int(value.get("dropped", 0)) for name, value in sorted(edges.items())
        },
    }


def _assert_clean_runtime(snapshot: dict[str, Any], *, label: str) -> None:
    semantics = _runtime_semantics(snapshot)
    _require(semantics["state"] == "stopped", f"{label} did not stop cleanly")
    _require(
        all(value == 0 for value in semantics["node_failed"].values()),
        f"{label} contains node failures",
    )
    _require(
        all(value == 0 for value in semantics["edge_dropped"].values()),
        f"{label} dropped runtime items",
    )
    _require(
        sum(semantics["node_processed"].values()) > 0,
        f"{label} processed no runtime items",
    )


def _contract_checks() -> dict[str, Any]:
    import numpy as np
    from neuros.contracts import ClockDomain, SignalFrame, StreamDescriptor

    original = np.arange(24, dtype=np.float32).reshape(3, 8)
    frame = SignalFrame(
        stream_id="contract-check",
        sequence_id=1,
        data=original,
        sample_rate_hz=250.0,
        host_receive_time_ns=10,
        clock_domain=ClockDomain.HOST_MONOTONIC,
        metadata={"nested": {"values": [1, 2, 3]}},
    )
    original[:] = -999.0
    _require(not np.all(frame.data == -999.0), "SignalFrame aliases caller-owned sample memory")
    _require(frame.data.flags.writeable is False, "SignalFrame data remains writeable")

    descriptor_a = StreamDescriptor(
        stream_id="eeg",
        modality="eeg",
        sample_rate_hz=250.0,
        channel_names=("C3", "C4"),
        metadata={"b": 2, "a": {"y": 2, "x": 1}},
    )
    descriptor_b = StreamDescriptor(
        stream_id="eeg",
        modality="eeg",
        sample_rate_hz=250.0,
        channel_names=("C3", "C4"),
        metadata={"a": {"x": 1, "y": 2}, "b": 2},
    )
    _require(
        descriptor_a.fingerprint() == descriptor_b.fingerprint(),
        "StreamDescriptor fingerprint depends on mapping insertion order",
    )
    return {
        "signal_frame_copies_input": True,
        "signal_frame_read_only": True,
        "descriptor_fingerprint_mapping_order_invariant": True,
        "descriptor_sha256": descriptor_a.fingerprint(),
    }


def _inside_environment(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    wheel_manifest = json.loads(Path(args.wheel_manifest).read_text(encoding="utf-8"))
    wheel_by_name = {
        _canonical_name(item["name"]): item for item in wheel_manifest.get("wheels", [])
    }
    results: list[CommandResult] = []
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "source_revision": args.source_revision,
        "environment": {
            "python": sys.version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "commands": [],
        "checks": {},
        "tamper_tests": {},
    }

    try:
        identities: dict[str, Any] = {}
        for name in REQUIRED_DISTRIBUTIONS:
            expected = wheel_by_name.get(_canonical_name(name))
            _require(expected is not None, f"wheel manifest is missing {name}")
            identities[name] = _distribution_identity(name, repo_root, expected["sha256"])
            _require(identities[name]["editable"] is False, f"{name} is installed editable")
        report["distributions"] = identities

        installed_neuros = sorted(
            _canonical_name(str(dist.metadata.get("Name", "")))
            for dist in metadata.distributions()
            if _canonical_name(str(dist.metadata.get("Name", ""))) == "neuros"
            or _canonical_name(str(dist.metadata.get("Name", ""))).startswith("neuros-")
        )
        _require(
            installed_neuros == sorted(_canonical_name(name) for name in REQUIRED_DISTRIBUTIONS),
            f"clean-room environment contains unexpected neurOS distributions: {installed_neuros}",
        )
        report["checks"]["only_default_neuros_distributions_installed"] = True

        for name in NONDEFAULT_DISTRIBUTIONS:
            try:
                metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
            raise RuntimeError(f"non-default distribution leaked into clean-room install: {name}")

        import neuros

        required_exports = {"SignalFrame", "Pipeline", "BaseDriver", "BaseModel", "load_plugin"}
        missing_exports = sorted(name for name in required_exports if not hasattr(neuros, name))
        _require(not missing_exports, f"SDK root is missing exports: {missing_exports}")
        neuros_path = Path(neuros.__file__).resolve() if neuros.__file__ else None
        _require(neuros_path is not None, "SDK root has no __file__")
        try:
            neuros_path.relative_to(repo_root)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"SDK import leaked from repository checkout: {neuros_path}")
        report["checks"]["sdk_import_outside_repo"] = str(neuros_path)
        report["checks"]["contracts"] = _contract_checks()

        neuros_cli = _console_script("neuros")
        work = output / "work"
        work.mkdir(parents=True, exist_ok=False)
        starter = work / "starter"

        doctor = _run(results, "doctor", [neuros_cli, "doctor", "--json"], cwd=work, expect_json=True)
        _require(isinstance(doctor, dict) and doctor.get("healthy") is True, "neuros doctor is unhealthy")
        _run(results, "compatibility", [neuros_cli, "compatibility", "--json"], cwd=work, expect_json=True)
        plugins = _run(results, "plugins", [neuros_cli, "plugins", "--json"], cwd=work, expect_json=True)
        _require(isinstance(plugins, list) and plugins, "plugin inventory is empty")

        init_result = _run(
            results,
            "init",
            [neuros_cli, "init", starter, "--json"],
            cwd=work,
            expect_json=True,
        )
        _require(isinstance(init_result, dict) and init_result.get("template") == "mock-bci", "starter init failed")
        config = starter / "neuros.yaml"
        _require(config.is_file(), "starter configuration was not created")

        _run(
            results,
            "init-refuses-overwrite",
            [neuros_cli, "init", starter, "--json"],
            cwd=work,
            expected_codes=(2,),
        )
        _run(results, "validate", [neuros_cli, "validate", config, "--json"], cwd=work, expect_json=True)
        run_result = _run(
            results,
            "run",
            [neuros_cli, "run", config, "--duration", str(args.duration), "--json"],
            cwd=work,
            expect_json=True,
        )
        _require(isinstance(run_result, dict), "runtime did not emit a JSON snapshot")
        _assert_clean_runtime(run_result, label="clean-room run")

        invalid_config = work / "invalid.yaml"
        invalid_config.write_text(
            """schema_version: 1\nstreams:\n  - id: eeg\n    source:\n      plugin: definitely_missing_source\ndecoder:\n  plugin: threshold\nruntime:\n  queue_capacity: 4\n  overflow_policy: drop_oldest\nsinks: []\nmonitors: []\n""",
            encoding="utf-8",
        )
        _run(
            results,
            "invalid-config-rejected",
            [neuros_cli, "validate", invalid_config, "--json"],
            cwd=work,
            expected_codes=(2, 3),
        )

        session = work / "session"
        record_result = _run(
            results,
            "record",
            [
                neuros_cli,
                "record",
                config,
                "--output",
                session,
                "--session-id",
                "full-package",
                "--duration",
                str(args.duration),
                "--json",
            ],
            cwd=work,
            expect_json=True,
        )
        _require(isinstance(record_result, dict) and record_result.get("status") == "complete", "record did not complete")
        frame_files = sorted(session.glob("streams/*/frames/*.npy"))
        _require(frame_files, "recording produced no frame payloads")
        inspect_result = _run(
            results,
            "inspect-verified",
            [neuros_cli, "inspect", session, "--verify", "--json"],
            cwd=work,
            expect_json=True,
        )
        _require(inspect_result.get("integrity") == "verified", "recording integrity was not verified")
        replay_result = _run(
            results,
            "replay",
            [neuros_cli, "replay", session, "--config", config, "--json"],
            cwd=work,
            expect_json=True,
        )
        _assert_clean_runtime(replay_result, label="clean-room replay")
        _run(
            results,
            "record-refuses-overwrite",
            [
                neuros_cli,
                "record",
                config,
                "--output",
                session,
                "--session-id",
                "full-package",
                "--duration",
                str(args.duration),
                "--json",
            ],
            cwd=work,
            expected_codes=(5,),
        )

        tampered_session = work / "session-tampered"
        shutil.copytree(session, tampered_session)
        tampered_frames = sorted(tampered_session.glob("streams/*/frames/*.npy"))
        report["tamper_tests"]["recording_payload"] = _flip_one_byte(tampered_frames[0])
        _run(
            results,
            "tampered-recording-inspect-rejected",
            [neuros_cli, "inspect", tampered_session, "--verify", "--json"],
            cwd=work,
            expected_codes=(5,),
        )
        _run(
            results,
            "tampered-recording-replay-rejected",
            [neuros_cli, "replay", tampered_session, "--config", config, "--json"],
            cwd=work,
            expected_codes=(4,),
        )
        _require(
            "Data hash mismatch" in results[-1].stderr,
            "tampered recording replay failed for an unrelated runtime reason",
        )

        qualification = work / "qualification"
        qualify_result = _run(
            results,
            "qualify",
            [
                neuros_cli,
                "qualify",
                config,
                "--output",
                qualification,
                "--session-id",
                "full-package",
                "--duration",
                str(args.duration),
                "--json",
            ],
            cwd=work,
            expect_json=True,
        )
        _require(isinstance(qualify_result, dict), "qualification result is not JSON")
        root_sha = qualify_result.get("bundle_sha256")
        _require(isinstance(root_sha, str) and len(root_sha) == 64, "qualification bundle SHA is missing")
        boundary = qualify_result.get("claim_boundary", {})
        _require(boundary.get("runtime_record_replay_qualified") is True, "runtime qualification claim is absent")
        for key in ("real_dataset_qualified", "hardware_qualified", "closed_loop_qualified", "clinical_qualified"):
            _require(boundary.get(key) is False, f"qualification overclaims {key}")

        reproduction_1 = _run(
            results,
            "reproduce",
            [neuros_cli, "reproduce", qualification, "--json"],
            cwd=work,
            expect_json=True,
        )
        reproduction_2 = _run(
            results,
            "reproduce-second-pass",
            [neuros_cli, "reproduce", qualification, "--json"],
            cwd=work,
            expect_json=True,
        )
        pinned = _run(
            results,
            "reproduce-pinned",
            [neuros_cli, "reproduce", qualification, "--expected-sha256", root_sha, "--json"],
            cwd=work,
            expect_json=True,
        )
        for item in (reproduction_1, reproduction_2, pinned):
            _require(item.get("reproduced") is True, "qualification reproduction did not complete")
            _require(item.get("bundle_sha256") == root_sha, "qualification root changed during reproduction")
        _require(
            reproduction_1.get("decoder_outputs") == reproduction_2.get("decoder_outputs"),
            "repeated qualification reproductions changed semantic decoder output identity",
        )
        _run(
            results,
            "wrong-external-pin-rejected",
            [neuros_cli, "reproduce", qualification, "--expected-sha256", "0" * 64, "--json"],
            cwd=work,
            expected_codes=(5,),
        )
        _run(
            results,
            "qualification-refuses-overwrite",
            [
                neuros_cli,
                "qualify",
                config,
                "--output",
                qualification,
                "--session-id",
                "full-package",
                "--duration",
                str(args.duration),
                "--json",
            ],
            cwd=work,
            expected_codes=(5,),
        )

        tampered_qualification = work / "qualification-tampered"
        shutil.copytree(qualification, tampered_qualification)
        report["tamper_tests"]["qualification_artifact"] = _append_tamper(
            tampered_qualification / "runtime.json"
        )
        _run(
            results,
            "tampered-qualification-rejected",
            [neuros_cli, "reproduce", tampered_qualification, "--json"],
            cwd=work,
            expected_codes=(5,),
        )
        _run(
            results,
            "tampered-qualification-pinned-rejected",
            [
                neuros_cli,
                "reproduce",
                tampered_qualification,
                "--expected-sha256",
                root_sha,
                "--json",
            ],
            cwd=work,
            expected_codes=(5,),
        )

        soak_elapsed: list[float] = []
        soak_semantics: list[dict[str, Any]] = []
        soak_duration = min(max(args.duration / 4.0, 0.03), 0.08)
        for index in range(args.soak_iterations):
            started = time.perf_counter()
            payload = _run(
                results,
                f"lifecycle-soak-{index:03d}",
                [neuros_cli, "run", config, "--duration", str(soak_duration), "--json"],
                cwd=work,
                expect_json=True,
            )
            soak_elapsed.append(time.perf_counter() - started)
            _assert_clean_runtime(payload, label=f"lifecycle soak {index}")
            soak_semantics.append(_runtime_semantics(payload))
        report["checks"]["lifecycle_soak"] = {
            "iterations": args.soak_iterations,
            "duration_per_iteration_s": soak_duration,
            "elapsed_median_s": statistics.median(soak_elapsed),
            "elapsed_max_s": max(soak_elapsed),
            "all_stopped": all(item["state"] == "stopped" for item in soak_semantics),
            "zero_node_failures": all(
                all(value == 0 for value in item["node_failed"].values()) for item in soak_semantics
            ),
            "zero_dropped_edges": all(
                all(value == 0 for value in item["edge_dropped"].values()) for item in soak_semantics
            ),
        }

        report["checks"]["qualification"] = {
            "bundle_sha256": root_sha,
            "reproduction_exact": True,
            "externally_pinned_reproduction": pinned.get("origin_authenticity") == "externally_pinned",
            "claim_boundary": boundary,
        }
        report["status"] = "pass"
        return 0
    except Exception as exc:
        report["status"] = "fail"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        return 1
    finally:
        report["commands"] = [item.to_dict() for item in results]
        _write_json(output / "environment-qualification.json", report)


def _policy_selected_packages(repo_root: Path) -> list[dict[str, Any]]:
    scripts = repo_root / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from list_release_packages import release_policy

        entries = release_policy()
    finally:
        try:
            sys.path.remove(str(scripts))
        except ValueError:
            pass
    selected = [
        item
        for item in entries
        if bool(item.get("default_release_candidate", item.get("publish_candidate", False)))
    ]
    names = sorted(_canonical_name(item["distribution"]) for item in selected)
    expected = sorted(_canonical_name(name) for name in REQUIRED_DISTRIBUTIONS)
    if names != expected:
        raise RuntimeError(
            "default release policy differs from full-package qualification contract: "
            f"expected {expected}, observed {names}"
        )
    return selected


def _build_wheels(repo_root: Path, wheel_dir: Path, results: list[CommandResult]) -> list[dict[str, Any]]:
    wheel_dir.mkdir(parents=True, exist_ok=False)
    selected = _policy_selected_packages(repo_root)
    for item in selected:
        _run(
            results,
            f"build-{item['distribution']}",
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                wheel_dir,
                repo_root / item["path"],
            ],
            cwd=repo_root,
        )
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != len(REQUIRED_DISTRIBUTIONS):
        raise RuntimeError(
            f"expected {len(REQUIRED_DISTRIBUTIONS)} default wheels, built {len(wheels)}"
        )
    metadata_rows = [_wheel_metadata(path) for path in wheels]
    observed = sorted(row["canonical_name"] for row in metadata_rows)
    expected = sorted(_canonical_name(name) for name in REQUIRED_DISTRIBUTIONS)
    if observed != expected:
        raise RuntimeError(f"built wheel names differ from policy: {observed} != {expected}")
    return metadata_rows


def _outer(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)

    results: list[CommandResult] = []
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "source_revision": args.source_revision,
        "orchestrator_environment": {
            "python": sys.version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "commands": [],
    }

    try:
        with tempfile.TemporaryDirectory(prefix="neuros-full-package-") as temporary:
            temp = Path(temporary)
            wheel_dir = temp / "wheels"
            wheel_rows = _build_wheels(repo_root, wheel_dir, results)
            wheel_manifest_path = output / "wheel-manifest.json"
            wheel_manifest = {
                "schema": "neuros.full_package_wheel_manifest.v1",
                "source_revision": args.source_revision,
                "wheels": wheel_rows,
            }
            _write_json(wheel_manifest_path, wheel_manifest)

            closure_path = output / "release-dependency-closure.json"
            _run(
                results,
                "release-dependency-closure",
                [
                    sys.executable,
                    repo_root / "scripts/check_release_dependency_closure.py",
                    wheel_dir,
                    "--output",
                    closure_path,
                ],
                cwd=repo_root,
            )
            ownership_path = output / "wheel-ownership.json"
            _run(
                results,
                "wheel-ownership",
                [
                    sys.executable,
                    repo_root / "scripts/check_wheel_ownership.py",
                    wheel_dir,
                    "--output",
                    ownership_path,
                ],
                cwd=repo_root,
            )

            env_root = temp / "venv"
            venv.EnvBuilder(with_pip=True, clear=True).create(env_root)
            env_python = _python_in_venv(env_root)
            _run(results, "upgrade-clean-room-pip", [env_python, "-m", "pip", "install", "--upgrade", "pip"])
            _run(
                results,
                "install-exact-default-wheels",
                [env_python, "-m", "pip", "install", *sorted(wheel_dir.glob("*.whl"))],
            )
            _run(results, "pip-check", [env_python, "-m", "pip", "check"])

            child_output = output / "clean-room"
            child_env = dict(os.environ)
            child_env.pop("PYTHONPATH", None)
            child_env.pop("PYTHONHOME", None)
            # The GitHub PR event SHA can be a synthetic merge commit.  Ordinary
            # qualification should not silently mistake that for installed-wheel
            # source identity during this clean-room test.
            child_env.pop("GITHUB_SHA", None)
            child_env.pop("GITHUB_REF", None)
            child_env["PYTHONNOUSERSITE"] = "1"
            outside_cwd = temp / "outside-repository"
            outside_cwd.mkdir()
            child = _run(
                results,
                "installed-wheel-clean-room",
                [
                    env_python,
                    Path(__file__).resolve(),
                    "--inside-env",
                    "--repo-root",
                    repo_root,
                    "--output",
                    child_output,
                    "--wheel-manifest",
                    wheel_manifest_path,
                    "--source-revision",
                    args.source_revision,
                    "--duration",
                    str(args.duration),
                    "--soak-iterations",
                    str(args.soak_iterations),
                ],
                cwd=outside_cwd,
                env=child_env,
            )
            assert child is None
            child_report = json.loads(
                (child_output / "environment-qualification.json").read_text(encoding="utf-8")
            )
            _require(child_report.get("status") == "pass", "clean-room child report is not passing")

            report["wheel_manifest"] = wheel_manifest
            report["release_dependency_closure"] = json.loads(closure_path.read_text(encoding="utf-8"))
            report["wheel_ownership"] = json.loads(ownership_path.read_text(encoding="utf-8"))
            report["clean_room"] = child_report
            report["claim_boundary"] = {
                "installed_wheel_product_path_qualified": True,
                "cross_platform_claim_requires_matrix_aggregation": True,
                "real_dataset_qualified": False,
                "hardware_qualified": False,
                "closed_loop_qualified": False,
                "clinical_qualified": False,
                "statement": (
                    "This report exercises packaging, runtime lifecycle, recording/replay, "
                    "tamper rejection, and software qualification from exact installed wheels. "
                    "It does not establish neural efficacy, physical-device validity, safety, "
                    "or clinical benefit."
                ),
            }
            report["status"] = "pass"
            return 0
    except Exception as exc:
        report["status"] = "fail"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        return 1
    finally:
        report["commands"] = [item.to_dict() for item in results]
        _write_json(output / "full-package-qualification.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", type=str, default="unknown")
    parser.add_argument("--duration", type=float, default=0.2)
    parser.add_argument("--soak-iterations", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inside-env", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wheel-manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.soak_iterations <= 0:
        raise SystemExit("--soak-iterations must be positive")
    if args.inside_env:
        if args.wheel_manifest is None:
            raise SystemExit("--wheel-manifest is required in --inside-env mode")
        return _inside_environment(args)
    return _outer(args)


if __name__ == "__main__":
    raise SystemExit(main())
