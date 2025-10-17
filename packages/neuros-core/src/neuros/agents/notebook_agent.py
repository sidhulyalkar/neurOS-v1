"""Notebook generation agent for neurOS.

This module defines an agent that can automatically generate Jupyter
notebooks demonstrating how to configure, train and run neurOS pipelines
on a variety of tasks and datasets.  The goal of this agent is to
provide ready‑to‑use tutorials that help users explore the platform and
reproduce experiments.

The agent uses the :mod:`nbformat` library to construct notebooks
programmatically.  It writes the resulting ``.ipynb`` files to a
configured output directory (by default ``notebooks/``).  Each
notebook contains a mixture of Markdown and code cells that import
neurOS modules, assemble a pipeline using
:func:`neuros.autoconfig.generate_pipeline_for_task`, train a model on
a small dataset, run the pipeline for a short duration and display
metrics.  Because Jupyter notebooks execute code sequentially, the
generated notebooks use ``asyncio.run`` to run asynchronous methods.

Example
-------

>>> from neuros.agents.notebook_agent import NotebookAgent
>>> agent = NotebookAgent(output_dir="./tutorials")
>>> agent.generate_demo("2‑class motor imagery")

Notes
-----

* If ``nbformat`` is not available in the runtime environment, the
  agent will raise an ``ImportError``.  This should rarely occur
  because ``nbformat`` is a lightweight dependency of many notebook
  tools.
* The agent attempts to train a model using data generated from the
  selected driver.  For dataset drivers (e.g. Iris), it uses the
  entire dataset for training.  For other drivers, it synthesises a
  small random dataset for demonstration purposes.  The primary goal
  is to produce a working example rather than to achieve high
  accuracy.
* The returned path refers to the written notebook file relative to
  the current working directory.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import nbformat as nbf  # type: ignore
except ImportError as e:  # pragma: no cover
    nbf = None

from ..autoconfig import generate_pipeline_for_task


class NotebookAgent:
    """Agent for generating Jupyter demonstration notebooks.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory in which to save generated notebooks.  If not
        specified, defaults to ``"notebooks"``.
    """

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        if nbf is None:
            raise ImportError(
                "The nbformat package is required to use NotebookAgent. Install it with `pip install nbformat`."
            )
        self.output_dir = Path(output_dir) if output_dir is not None else Path("notebooks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_training_data(self, pipeline, n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for a given pipeline.

        This helper examines the driver and processing configuration of
        the provided pipeline to determine the dimensionality of the
        feature vectors.  It supports dataset pipelines, video/pose
        pipelines, motion sensors, calcium imaging, and default EEG
        pipelines.

        For dataset drivers this returns the full dataset.  For other
        modalities it returns random input/output pairs with the
        correct number of features.  The purpose is to enable
        demonstration of the ``train`` method without requiring
        large datasets or exact replication of the processing chain.

        Parameters
        ----------
        pipeline : Pipeline
            The configured pipeline for which training data should be
            generated.  Both the driver and processing agent class
            influence the feature dimensionality.
        n_samples : int, optional
            Number of random samples to generate when using non‑dataset
            drivers.  Defaults to 200.

        Returns
        -------
        (X, y) : Tuple[np.ndarray, np.ndarray]
            Feature matrix and binary labels for training.
        """
        driver = pipeline.driver
        # If the driver exposes preloaded data/labels, use them directly
        if hasattr(driver, "data") and hasattr(driver, "labels"):
            X = np.asarray(driver.data)
            y = np.asarray(driver.labels)
            return X, y
        # Determine feature dimensionality based on processing agent
        # If a custom processing agent is specified, infer dimensionality
        # from the known agent types.  Otherwise use EEG band‑power logic.
        feat_dim = None
        agent_cls = getattr(pipeline, "processing_agent_class", None)
        # import here to avoid circular imports
        try:
            from ..agents import (
                VideoAgent,
                PoseAgent,
                FacialAgent,
                BlinkAgent,
                MotionAgent,
                CalciumAgent,
            )
        except Exception:
            VideoAgent = PoseAgent = FacialAgent = BlinkAgent = MotionAgent = CalciumAgent = None  # type: ignore
        # Video processing: mean/var per channel -> 2 * channels
        if agent_cls is not None and VideoAgent is not None and issubclass(agent_cls, VideoAgent):
            feat_dim = driver.channels * 2
        # PoseAgent: two centroid coordinates
        elif agent_cls is not None and PoseAgent is not None and issubclass(agent_cls, PoseAgent):
            feat_dim = 2
        # FacialAgent: two ratio features
        elif agent_cls is not None and FacialAgent is not None and issubclass(agent_cls, FacialAgent):
            feat_dim = 2
        # BlinkAgent: number of horizontal bands (default 4)
        elif agent_cls is not None and BlinkAgent is not None and issubclass(agent_cls, BlinkAgent):
            # default number of bands is 4 unless overridden via processing_kwargs
            bands = pipeline.processing_kwargs.get("bands", 4)
            feat_dim = int(bands)
        # MotionAgent: forward raw IMU samples unchanged -> channels
        elif agent_cls is not None and MotionAgent is not None and issubclass(agent_cls, MotionAgent):
            feat_dim = getattr(driver, "channels", 6)
        # CalciumAgent: two summary statistics (mean and std)
        elif agent_cls is not None and CalciumAgent is not None and issubclass(agent_cls, CalciumAgent):
            feat_dim = 2
        # Default EEG processing: use band‑power features
        if feat_dim is None:
            # number of frequency bands: use provided bands or default 5
            bands = pipeline.bands
            n_bands = len(bands) if bands is not None else 5
            # feature dimension = channels * number of bands
            channels = getattr(driver, "channels", 1)
            feat_dim = channels * n_bands
        # Generate random feature matrix and labels
        X = np.random.randn(n_samples, int(feat_dim))
        y = np.random.randint(0, 2, size=n_samples)
        return X, y

    async def _train_and_run_pipeline(self, pipeline, duration: float = 3.0) -> dict:
        """Train and run a pipeline for demonstration.

        The pipeline is trained using synthetic or dataset data via
        :meth:`_generate_training_data`.  After training the model, the
        pipeline is executed for a short duration and the resulting
        metrics are returned.

        Parameters
        ----------
        pipeline : Pipeline
            A configured neurOS pipeline.
        duration : float, optional
            Duration in seconds to run the pipeline.  Defaults to
            3.0 seconds.

        Returns
        -------
        metrics : dict
            Dictionary of performance metrics from :meth:`Pipeline.run`.
        """
        # generate synthetic training data using pipeline configuration
        X, y = self._generate_training_data(pipeline)
        pipeline.train(X, y)
        metrics = await pipeline.run(duration=duration)
        return metrics

    def generate_demo(self, task_description: str, *, duration: float = 3.0, filename: Optional[str] = None) -> str:
        """Generate a demonstration notebook for a given task.

        This method synchronously generates a Jupyter notebook that
        illustrates how to create and run a neurOS pipeline for the
        provided ``task_description``.  It internally trains and runs
        the pipeline to capture example metrics.  The resulting
        notebook is saved to disk.

        Parameters
        ----------
        task_description : str
            Free‑form description of the task to demonstrate.
        duration : float, optional
            Duration in seconds for which to run the pipeline.  Default is
            3.0 seconds.
        filename : str, optional
            Filename to use for the notebook (without extension).  If
            omitted the filename is derived from the task description.

        Returns
        -------
        str
            Path to the generated ``.ipynb`` file.
        """
        # lazily import generate_pipeline_for_task to avoid circular import
        pipeline = generate_pipeline_for_task(task_description)
        # run training and pipeline
        try:
            metrics = asyncio.run(self._train_and_run_pipeline(pipeline, duration=duration))
        except Exception as e:
            # if training or running fails, produce empty metrics
            metrics = {"error": str(e)}
        # derive filename from task description
        safe_name = filename or task_description.lower().strip().replace(" ", "_")
        nb_path = self.output_dir / f"{safe_name}_demo.ipynb"
        # build notebook
        nb = nbf.v4.new_notebook()
        # title
        nb.cells.append(
            nbf.v4.new_markdown_cell(f"# neurOS Demo: {task_description}\n\n"
                                     "This notebook demonstrates how to set up, train and run a neurOS pipeline "
                                     "for the specified task using the auto‑configuration module. The pipeline "
                                     "is trained on synthetic or dataset data and executed for a short duration. "
                                     "The resulting metrics are displayed for reference.")
        )
        # imports, pipeline creation and training code
        nb.cells.append(
            nbf.v4.new_code_cell(
                "from neuros.autoconfig import generate_pipeline_for_task\n"
                "import numpy as np\n"
                "import asyncio\n"
                "\n"
                f"# create pipeline for the task\n"
                f"pipeline = generate_pipeline_for_task('{task_description}', use_brainflow=False)\n"
                "\n"
                "# synthetic training data generation helper\n"
                "def _generate_training_data(pipeline, n_samples=200):\n"
                "    driver = pipeline.driver\n"
                "    # dataset: return stored data and labels\n"
                "    if hasattr(driver, 'data') and hasattr(driver, 'labels'):\n"
                "        return np.asarray(driver.data), np.asarray(driver.labels)\n"
                "    # determine processing agent class\n"
                "    agent_cls = getattr(pipeline, 'processing_agent_class', None)\n"
                "    feat_dim = None\n"
                "    # import agent types (optional)\n"
                "    try:\n"
                "        from neuros.agents import VideoAgent, PoseAgent, FacialAgent, BlinkAgent, MotionAgent, CalciumAgent\n"
                "    except Exception:\n"
                "        VideoAgent = PoseAgent = FacialAgent = BlinkAgent = MotionAgent = CalciumAgent = None\n"
                "    if agent_cls is not None and VideoAgent is not None and issubclass(agent_cls, VideoAgent):\n"
                "        feat_dim = driver.channels * 2\n"
                "    elif agent_cls is not None and PoseAgent is not None and issubclass(agent_cls, PoseAgent):\n"
                "        feat_dim = 2\n"
                "    elif agent_cls is not None and FacialAgent is not None and issubclass(agent_cls, FacialAgent):\n"
                "        feat_dim = 2\n"
                "    elif agent_cls is not None and BlinkAgent is not None and issubclass(agent_cls, BlinkAgent):\n"
                "        bands = pipeline.processing_kwargs.get('bands', 4)\n"
                "        feat_dim = int(bands)\n"
                "    elif agent_cls is not None and MotionAgent is not None and issubclass(agent_cls, MotionAgent):\n"
                "        feat_dim = getattr(driver, 'channels', 6)\n"
                "    elif agent_cls is not None and CalciumAgent is not None and issubclass(agent_cls, CalciumAgent):\n"
                "        feat_dim = 2\n"
                "    if feat_dim is None:\n"
                "        n_bands = len(pipeline.bands) if pipeline.bands is not None else 5\n"
                "        channels = getattr(driver, 'channels', 1)\n"
                "        feat_dim = channels * n_bands\n"
                "    X = np.random.randn(n_samples, int(feat_dim))\n"
                "    y = np.random.randint(0, 2, size=n_samples)\n"
                "    return X, y\n"
                "\n"
                "# generate training data and train the model\n"
                "X_train, y_train = _generate_training_data(pipeline)\n"
                "pipeline.train(X_train, y_train)\n"
                "\n"
                "# run the pipeline asynchronously\n"
                "async def _run():\n"
                "    metrics = await pipeline.run(duration=" + str(duration) + ")\n"
                "    return metrics\n"
                "\n"
                "metrics = asyncio.run(_run())\n"
                "metrics"
            )
        )
        # display captured metrics
        nb.cells.append(
            nbf.v4.new_markdown_cell(
                "## Example Results\n\n"
                "The following metrics were captured when this notebook was generated:\n\n"
                f"``\n{metrics}\n```"
            )
        )
        # save notebook
        nbf.write(nb, nb_path)
        return str(nb_path)