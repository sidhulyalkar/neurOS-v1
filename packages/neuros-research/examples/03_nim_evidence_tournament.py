"""Run the Algonauts NIM tournament with frozen reviewed aggregate evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

from neuros.research._canonical import canonical_sha256
from neuros.research.nim_provider import QualifiedNvidiaNimClient
from neuros.research.reviewed_context import load_reviewed_aggregate_context

_ROOT = Path(__file__).resolve().parents[1]
_BASE_TOURNAMENT = Path(__file__).with_name("02_nim_algonauts_tournament.py")
_REVIEWED_EVIDENCE = _ROOT / "evidence" / "controlled_representation_noise_sweep_v2.json"


def _load_base_tournament() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "neuros_research_nim_algonauts_tournament",
        _BASE_TOURNAMENT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the base NIM tournament module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _output_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    args, _ = parser.parse_known_args()
    return Path(args.output)


def _write_provider_failure(output: Path) -> None:
    client = QualifiedNvidiaNimClient.latest_instance
    if client is None:
        return
    payload = {
        "kind": "neuros_nim_provider_qualification",
        "schema_version": 1,
        "source_revision": os.environ.get("GITHUB_SHA", "local-unspecified").strip(),
        "provider": "nvidia_nim",
        "qualification": client.provider_qualification(),
    }
    payload["fingerprint"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.with_name("provider-qualification.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"NIM_PROVIDER_QUALIFICATION_SHA256={payload['fingerprint']}")


def _attach_provider_qualification(output: Path) -> None:
    client = QualifiedNvidiaNimClient.latest_instance
    if client is None:
        raise RuntimeError("qualified NVIDIA client instance was not retained")
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["provider_qualification"] = client.provider_qualification()
    payload.pop("fingerprint", None)
    payload["fingerprint"] = canonical_sha256(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    # This validation occurs before base.main() constructs a client or performs model discovery.
    reviewed = load_reviewed_aggregate_context(_REVIEWED_EVIDENCE)
    base = _load_base_tournament()
    base_context = base._public_context
    base.NvidiaNimClient = QualifiedNvidiaNimClient

    def evidence_informed_context():
        context = base_context()
        context["reviewed_prior_evidence"] = reviewed
        context["priority_questions"] = [
            "Does a convergence-matched nonlinear control retain the synthetic noise sensitivity "
            "seen in the four-epoch autoencoder diagnostic?",
            "Can a real temporal representation improve trajectory geometry without relying on "
            "transductive target fitting?",
            "Which development-only geometry measurements prospectively predict validation "
            "decoding gains rather than merely describing the synthetic manifold?",
            "What low-cost matched controls can separate feature dimension, optimization, "
            "temporal prior, and readout capacity?",
        ]
        return context

    base._public_context = evidence_informed_context
    output = _output_path()
    try:
        base.main()
    except Exception:
        _write_provider_failure(output)
        raise
    _attach_provider_qualification(output)


if __name__ == "__main__":
    main()
