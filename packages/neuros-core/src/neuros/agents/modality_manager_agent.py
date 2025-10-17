"""Modality orchestration agent for neurOS.

This agent coordinates the creation, training and execution of multiple
neurOS pipelines across a set of task descriptions and modalities.
It provides a simple interface to benchmark or explore how different
configurations perform on various tasks.  Each task is interpreted
through the auto‑configuration heuristics defined in
:func:`neuros.autoconfig.generate_pipeline_for_task`.

Example
-------

>>> from neuros.agents.modality_manager_agent import ModalityManagerAgent
>>> agent = ModalityManagerAgent(["SSVEP speller", "motor imagery", "video facial"])
>>> results = asyncio.run(agent.run_all())
>>> print(results["motor imagery"]["accuracy"])

Notes
-----

* The agent trains each pipeline using synthetic or dataset data
  similarly to :class:`NotebookAgent`.  If a driver exposes ``data`` and
  ``labels`` attributes (as dataset drivers do), those are used for
  training.  Otherwise random data with an appropriate shape is
  generated.  The training is intentionally lightweight and not
  intended for high accuracy.
* Metrics returned by this agent include the fields produced by
  :meth:`Pipeline.run` as well as the task description.  The caller
  can use these results to populate dashboards or analytics.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Dict, Any

import numpy as np

from neuros.autoconfig import generate_pipeline_for_task
from neuros.agents.notebook_agent import NotebookAgent


class ModalityManagerAgent:
    """Agent that runs pipelines for multiple tasks and aggregates metrics."""

    def __init__(self, tasks: Iterable[str], *, duration: float = 3.0) -> None:
        self.tasks = list(tasks)
        self.duration = duration
        # use a notebook agent internally to reuse its data generation helper
        self._nb_agent = NotebookAgent(output_dir="notebooks")

    async def _run_single(self, task: str) -> Dict[str, Any]:
        """Configure, train and run a single task pipeline.

        Parameters
        ----------
        task : str
            Description of the task.

        Returns
        -------
        dict
            Dictionary of metrics for the task.  Includes the key
            ``"task"`` to identify the originating task.
        """
        pipeline = generate_pipeline_for_task(task)
        # generate training data using helper from notebook agent
        X_train, y_train = self._nb_agent._generate_training_data(pipeline)
        pipeline.train(X_train, y_train)
        try:
            metrics = await pipeline.run(duration=self.duration)
        except Exception as e:
            metrics = {"error": str(e)}
        metrics["task"] = task
        return metrics

    async def run_all(self) -> Dict[str, Dict[str, Any]]:
        """Run pipelines for all tasks asynchronously and collect metrics.

        Returns
        -------
        dict
            Mapping from task description to the metrics returned by the
            corresponding pipeline run.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for task in self.tasks:
            res = await self._run_single(task)
            results[task] = res
        return results