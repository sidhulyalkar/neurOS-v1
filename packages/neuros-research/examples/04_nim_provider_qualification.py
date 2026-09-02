"""Run and persist bounded NVIDIA hosted-model transport qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neuros.research._canonical import canonical_sha256
from neuros.research.nim_provider import QualifiedNvidiaNimClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("NVIDIA_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1"),
    )
    args = parser.parse_args()

    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise SystemExit("NVIDIA_API_KEY is required")

    client = QualifiedNvidiaNimClient(key, endpoint=args.endpoint)
    failure: str | None = None
    qualified: tuple[str, ...] = ()
    try:
        qualified, _ = client.discover_models()
    except RuntimeError as exc:
        failure = str(exc)

    payload = {
        "kind": "neuros_nim_provider_qualification",
        "schema_version": 1,
        "source_revision": os.environ.get("GITHUB_SHA", "local-unspecified").strip(),
        "provider": "nvidia_nim",
        "qualification": client.provider_qualification(),
        "qualified": bool(qualified),
        "failure": failure,
    }
    payload["fingerprint"] = canonical_sha256(payload)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"NIM_PROVIDER_QUALIFICATION_SHA256={payload['fingerprint']}")
    print("NIM_PROVIDER_MODELS=" + ",".join(qualified))

    if not qualified:
        raise SystemExit("no NVIDIA hosted chat model passed bounded qualification")


if __name__ == "__main__":
    main()
