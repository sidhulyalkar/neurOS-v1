"""
SQLite database integration for neurOS.

This module defines a simple database layer for storing run metrics and
streaming classification results.  It uses Python's built‑in `sqlite3`
module and automatically creates tables on first use.  All methods are
thread‑safe via the SQLite connection's built‑in locking.

The database schema comprises two tables:

* ``runs``: stores per‑run summary metrics such as duration, sample count,
  throughput, mean latency and accuracy.
* ``results``: stores per‑sample classification results with timestamp,
  predicted label, confidence and latency.

The default database path can be overridden via the environment variable
``NEUROS_DB_PATH``.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


class Database:
    """Lightweight SQLite wrapper for storing neurOS runs and results."""

    def __init__(self, db_path: str | None = None) -> None:
        # determine path: environment overrides parameter
        env_path = os.getenv("NEUROS_DB_PATH")
        self.path = env_path or db_path or "neuros.db"
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        # runs table stores summary metrics and tenant ID
        # runs table stores summary metrics, metadata and tenant ID.
        # New columns driver, model and task capture the pipeline
        # configuration for cross‑modality and cross‑model analysis.  If the
        # database already exists without these columns, SQLite will ignore
        # them on creation.  Existing databases may not have these columns;
        # insertion will set them to NULL.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                duration REAL,
                samples INTEGER,
                throughput REAL,
                mean_latency REAL,
                accuracy REAL,
                quality_mean REAL,
                quality_std REAL,
                tenant_id TEXT,
                driver TEXT,
                model TEXT,
                task TEXT
            );
            """
        )
        # results table stores per‑sample outputs and tenant ID
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                run_id TEXT,
                timestamp REAL,
                label INTEGER,
                confidence REAL,
                latency REAL,
                tenant_id TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        self.conn.commit()

    def insert_run_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        tenant_id: str = "default",
        driver: Optional[str] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
    ) -> None:
        """Insert or update a row in the runs table.

        Parameters
        ----------
        run_id : str
            Unique identifier for the run.
        metrics : dict
            Dictionary containing at least ``duration``, ``samples``,
            ``throughput`` and ``mean_latency``.  ``accuracy`` is optional.
        tenant_id : str, optional
            Identifier of the tenant to which this run belongs.  Defaults to "default".
        driver : str, optional
            Name of the driver (modalities) used for this run.
        model : str, optional
            Name of the model used for this run.
        task : str, optional
            Description of the task this pipeline was configured for.
        """
        ts = datetime.utcnow().isoformat()
        accuracy = metrics.get("accuracy")
        q_mean = metrics.get("quality_mean")
        q_std = metrics.get("quality_std")
        # assemble row values, inserting None for missing metrics
        values = (
            run_id,
            ts,
            metrics.get("duration", 0.0),
            int(metrics.get("samples", 0)),
            metrics.get("throughput", 0.0),
            metrics.get("mean_latency", 0.0),
            accuracy,
            q_mean,
            q_std,
            tenant_id,
            driver,
            model,
            task,
        )
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO runs (run_id, timestamp, duration, samples, throughput, mean_latency, accuracy, quality_mean, quality_std, tenant_id, driver, model, task) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
        self.conn.commit()

    def insert_stream_results(self, run_id: str, results: Iterable[Tuple[float, int, float, float]], tenant_id: str = "default") -> None:
        """Insert streaming results for a run.

        Parameters
        ----------
        run_id : str
            Identifier of the run.
        results : iterable
            Iterable of (timestamp, label, confidence, latency) tuples.
        tenant_id : str, optional
            Identifier of the tenant to which these results belong.  Defaults to "default".
        """
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT INTO results (run_id, timestamp, label, confidence, latency, tenant_id) VALUES (?,?,?,?,?,?)",
            [(run_id, ts, int(label), float(conf), float(lat), tenant_id) for ts, label, conf, lat in results],
        )
        self.conn.commit()

    def list_runs(self, tenant_id: Optional[str] = None) -> List[str]:
        """Return a list of run identifiers.

        Parameters
        ----------
        tenant_id : str, optional
            If provided, only runs belonging to this tenant are returned.
        """
        cur = self.conn.cursor()
        if tenant_id:
            rows = cur.execute(
                "SELECT run_id FROM runs WHERE tenant_id = ? ORDER BY timestamp DESC",
                (tenant_id,),
            ).fetchall()
        else:
            rows = cur.execute("SELECT run_id FROM runs ORDER BY timestamp DESC").fetchall()
        return [row[0] for row in rows]

    def get_run_metrics(self, run_id: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Retrieve metrics for a given run.

        Parameters
        ----------
        run_id : str
            Identifier of the run.
        tenant_id : str, optional
            Tenant filter; if provided, ensures the run belongs to this tenant.
        """
        cur = self.conn.cursor()
        if tenant_id:
            row = cur.execute(
                "SELECT * FROM runs WHERE run_id = ? AND tenant_id = ?",
                (run_id, tenant_id),
            ).fetchone()
        else:
            row = cur.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        # extract optional metadata if present; older databases may not have these columns
        keys = row.keys()
        driver = row["driver"] if "driver" in keys else None
        model = row["model"] if "model" in keys else None
        task = row["task"] if "task" in keys else None
        quality_mean = row["quality_mean"] if "quality_mean" in keys else None
        quality_std = row["quality_std"] if "quality_std" in keys else None
        return {
            "run_id": row["run_id"],
            "timestamp": row["timestamp"],
            "duration": row["duration"],
            "samples": row["samples"],
            "throughput": row["throughput"],
            "mean_latency": row["mean_latency"],
            "accuracy": row["accuracy"],
            "quality_mean": quality_mean,
            "quality_std": quality_std,
            "driver": driver,
            "model": model,
            "task": task,
        }

    def get_stream_results(self, run_id: str, tenant_id: Optional[str] = None) -> List[Tuple[float, int, float, float]]:
        """Retrieve streaming results for a run.

        Parameters
        ----------
        run_id : str
            Identifier of the run.
        tenant_id : str, optional
            Tenant filter; if provided, ensures the run belongs to this tenant.
        """
        cur = self.conn.cursor()
        if tenant_id:
            rows = cur.execute(
                "SELECT timestamp, label, confidence, latency FROM results WHERE run_id = ? AND tenant_id = ? ORDER BY timestamp",
                (run_id, tenant_id),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT timestamp, label, confidence, latency FROM results WHERE run_id = ? ORDER BY timestamp",
                (run_id,),
            ).fetchall()
        return [(row["timestamp"], row["label"], row["confidence"], row["latency"]) for row in rows]