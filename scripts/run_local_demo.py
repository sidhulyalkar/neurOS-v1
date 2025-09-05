#!/usr/bin/env python3
"""Convenience wrapper to run the Constellation demo locally.

This script simplifies executing the multi‑modal demo pipeline by
providing a single entry point.  Optionally it can bring up a local
Kafka/Zookeeper stack using the project's ``docker-compose.yml`` and
then invoke the ``neuros constellation`` command with sensible
defaults.  It mirrors the options exposed by the CLI but is easier
to use when developing or testing.

Example usage:

```
python scripts/run_local_demo.py --duration 5 --output-dir /tmp/demo --with-kafka
```

To run the pipeline without Kafka (dry‑run mode), include
``--no-kafka``.  The script will skip starting docker-compose in
that case.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Constellation demo")
    parser.add_argument("--duration", type=float, default=5.0, help="Ingestion duration in seconds")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/constellation_demo_script",
        help="Base directory where data will be stored",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default="demo_subject",
        help="Subject identifier",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="demo_session",
        help="Session identifier",
    )
    parser.add_argument(
        "--with-kafka",
        action="store_true",
        help="Start the local Kafka stack via docker-compose before running",
    )
    parser.add_argument(
        "--no-kafka",
        action="store_true",
        help="Run without a Kafka broker (use NoopWriter)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if args.with_kafka and not args.no_kafka:
        compose_file = project_root / "docker-compose.yml"
        if not compose_file.exists():
            print(
                f"docker-compose.yml not found at {compose_file}; cannot start Kafka",
                file=sys.stderr,
            )
        else:
            print("Starting local Kafka and Zookeeper via docker-compose...", file=sys.stderr)
            subprocess.run([
                "docker-compose",
                "-f",
                str(compose_file),
                "up",
                "-d",
            ], cwd=str(project_root), check=True)

    # Build the neuros CLI command
    cmd = [
        sys.executable,
        "-m",
        "neuros.cli",
        "constellation",
        "--duration",
        str(args.duration),
        "--output-dir",
        args.output_dir,
        "--subject-id",
        args.subject_id,
        "--session-id",
        args.session_id,
        "--topic-prefix",
        "raw",
    ]
    if args.no_kafka:
        cmd.append("--no-kafka")
    # Print the command for visibility
    print("Running command:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)

    print("Constellation demo completed.", file=sys.stderr)


if __name__ == "__main__":
    main()