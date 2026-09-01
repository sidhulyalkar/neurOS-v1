"""Run the Algonauts NIM tournament with frozen reviewed aggregate evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

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


def main() -> None:
    # This validation occurs before base.main() constructs a client or performs model discovery.
    reviewed = load_reviewed_aggregate_context(_REVIEWED_EVIDENCE)
    base = _load_base_tournament()
    base_context = base._public_context

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
    base.main()


if __name__ == "__main__":
    main()
