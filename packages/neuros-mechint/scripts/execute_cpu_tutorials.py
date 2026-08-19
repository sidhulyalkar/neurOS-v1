"""Execute the maintained CPU-only evidence tutorials for release CI."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

CPU_TUTORIALS = (
    "00_what_counts_as_mechanistic_evidence.ipynb",
    "01_ground_truth_causal_network.ipynb",
    "07_circuit_faithfulness.ipynb",
    "08_held_out_evidence_packs.ipynb",
    "11_hierarchical_replication.ipynb",
    "12_reproducible_evidence_closure.ipynb",
)


def main() -> int:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    repo_root = Path(__file__).resolve().parents[3]
    tutorial_root = repo_root / "tutorials" / "mechint"
    for name in CPU_TUTORIALS:
        path = tutorial_root / name
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=180, kernel_name="python3", allow_errors=False)
        client.execute(cwd=str(repo_root))
        print(f"executed {path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
