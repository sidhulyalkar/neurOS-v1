"""
Federated client for neurOS.

The :class:`FederatedClient` can push run metrics and results to a
central aggregator service in a federated deployment.  In this
placeholder implementation it simply writes metrics and results to a
designated directory, simulating the handoff to an aggregator or
coordinator.  In future versions this class could publish over a
message bus, send HTTP requests or implement secure aggregation
protocols.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple


class FederatedClient:
    """Dummy federated client that writes metrics/results for coordination."""

    def __init__(self, out_dir: str) -> None:
        """Initialise the client.

        Parameters
        ----------
        out_dir : str
            Directory where metrics and results will be staged for
            collection by an aggregator.  The directory is created if
            it does not exist.
        """
        self.out_path = Path(os.path.expanduser(out_dir))
        self.out_path.mkdir(parents=True, exist_ok=True)

    def push_run(self, run_id: str, metrics: Dict[str, float], results: Iterable[Tuple[float, int, float, float]]) -> None:
        """Write a run's metrics and results to the output directory.

        Each run is stored under a subdirectory named after the run ID.
        Metrics are saved as JSON and results as a plain text file.

        Parameters
        ----------
        run_id : str
            Unique identifier of the run.
        metrics : dict
            Run summary metrics to serialise.
        results : iterable
            Per‑sample classification results (timestamp, label,
            confidence, latency).
        """
        run_dir = self.out_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # save metrics
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        # save results
        results_path = run_dir / "results.tsv"
        with results_path.open("w", encoding="utf-8") as f:
            for ts, label, conf, lat in results:
                f.write(f"{ts}\t{label}\t{conf}\t{lat}\n")