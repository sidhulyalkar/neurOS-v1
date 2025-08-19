"""Tests for NotebookAgent and ModalityManagerAgent."""

import os
import asyncio

from neuros.agents.notebook_agent import NotebookAgent
from neuros.agents.modality_manager_agent import ModalityManagerAgent


def test_notebook_agent_generates_file(tmp_path):
    agent = NotebookAgent(output_dir=tmp_path)
    nb_path = agent.generate_demo("mock task", duration=0.5)
    assert os.path.isfile(nb_path)
    assert nb_path.endswith(".ipynb")


def test_modality_manager_runs_tasks():
    tasks = ["random", "video facial"]
    agent = ModalityManagerAgent(tasks, duration=0.5)
    results = asyncio.run(agent.run_all())
    assert set(results.keys()) == set(tasks)
    for task, metrics in results.items():
        # metrics should include throughput and latency when run succeeds
        assert "task" in metrics
        assert metrics["task"] == task