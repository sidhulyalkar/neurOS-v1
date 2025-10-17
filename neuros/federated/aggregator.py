"""
Federated aggregator for neurOS.

This module defines a simple class for aggregating metrics and results
from multiple neurOS databases.  In a federated learning scenario,
individual neurOS instances (clients) operate independently and
produce metrics on local neural data.  The aggregator collects these
summaries to compute cross‑site statistics without requiring raw
signal data, thereby preserving privacy.

Usage
-----
Instantiate a :class:`FederatedAggregator` with a list of database
paths.  Call :meth:`aggregate_metrics` to compute overall averages of
throughput, latency, duration, samples and accuracy across all runs
in the provided databases.  You can also inspect per‑site run
distributions via :meth:`gather_runs`.

Example
-------
>>> from neuros.federated import FederatedAggregator
>>> aggregator = FederatedAggregator(["/path/to/site1.db", "/path/to/site2.db"])
>>> summary = aggregator.aggregate_metrics()
>>> print(summary["mean_throughput"])

This class is intentionally simple; production systems might use
secure channels, authentication and distributed aggregation protocols.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from neuros.db.database import Database


class FederatedAggregator:
    """Collect and summarise metrics from multiple neurOS sites."""

    def __init__(self, db_paths: Iterable[str]) -> None:
        self.databases: List[Database] = []
        for path in db_paths:
            try:
                db = Database(path)
                self.databases.append(db)
            except Exception:
                # ignore invalid database paths
                pass

    def gather_runs(self) -> Dict[str, List[str]]:
        """Return a mapping of database path to list of run IDs."""
        runs: Dict[str, List[str]] = {}
        for db in self.databases:
            try:
                run_ids = db.list_runs()
                runs[db.path] = run_ids
            except Exception:
                runs[db.path] = []
        return runs

    def aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate statistics across all runs.

        Returns
        -------
        dict
            A dictionary containing the mean of duration, throughput,
            samples, mean latency and accuracy across all runs and all
            sites.  Missing accuracy values are ignored in the average.
        """
        metrics_accum: Dict[str, List[float]] = {
            "duration": [],
            "throughput": [],
            "samples": [],
            "mean_latency": [],
            "accuracy": [],
            "quality_mean": [],
            "quality_std": [],
        }
        for db in self.databases:
            try:
                run_ids = db.list_runs()
                for rid in run_ids:
                    m = db.get_run_metrics(rid)
                    if not m:
                        continue
                    metrics_accum["duration"].append(m.get("duration", 0.0))
                    metrics_accum["throughput"].append(m.get("throughput", 0.0))
                    metrics_accum["samples"].append(m.get("samples", 0))
                    metrics_accum["mean_latency"].append(m.get("mean_latency", 0.0))
                    # accuracy may be None
                    if m.get("accuracy") is not None:
                        metrics_accum["accuracy"].append(m.get("accuracy"))
                    # include quality metrics if present
                    if m.get("quality_mean") is not None:
                        metrics_accum["quality_mean"].append(m.get("quality_mean"))
                    if m.get("quality_std") is not None:
                        metrics_accum["quality_std"].append(m.get("quality_std"))
            except Exception:
                continue
        summary: Dict[str, float] = {}
        for key, values in metrics_accum.items():
            if not values:
                summary[f"mean_{key}"] = 0.0
            else:
                summary[f"mean_{key}"] = float(sum(values) / len(values))
        return summary
