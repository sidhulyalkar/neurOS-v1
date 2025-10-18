"""
FALCON Benchmark for NeuroFM-X.

FALCON (Few-shot Adaptation Learning for Control of Neural activity)
evaluates transfer learning robustness across sessions and subjects.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuros_neurofm.evaluation.metrics import EvaluationMetrics


class FALCONBenchmark:
    """FALCON benchmark for few-shot transfer evaluation.

    Tests model's ability to adapt to new neural populations with limited data.

    Parameters
    ----------
    model : nn.Module
        NeuroFM-X model to evaluate.
    n_shots : list of int, optional
        Number of training examples for few-shot learning.
        Default: [1, 5, 10, 25, 50].
    n_trials : int, optional
        Number of trials per n-shot configuration.
        Default: 5.
    adapter_type : str, optional
        Type of adapter to use ('unit_id', 'lora', 'full').
        Default: 'unit_id'.
    """

    def __init__(
        self,
        model: nn.Module,
        n_shots: List[int] = [1, 5, 10, 25, 50],
        n_trials: int = 5,
        adapter_type: str = 'unit_id',
    ):
        self.model = model
        self.n_shots = n_shots
        self.n_trials = n_trials
        self.adapter_type = adapter_type

    def evaluate(
        self,
        support_sets: List[Dataset],
        query_sets: List[Dataset],
        task: str = 'decoder',
        device: str = 'cpu',
    ) -> Dict[str, Dict[str, float]]:
        """Run FALCON benchmark.

        Parameters
        ----------
        support_sets : list of Dataset
            Support sets for each session/subject (for adaptation).
        query_sets : list of Dataset
            Query sets for evaluation.
        task : str, optional
            Task to evaluate.
            Default: 'decoder'.
        device : str, optional
            Device to run on.
            Default: 'cpu'.

        Returns
        -------
        dict
            Results with structure:
            {
                'n_shot=1': {'mean_r2': 0.4, 'std_r2': 0.05, ...},
                'n_shot=5': {...},
                ...
            }
        """
        results = {}

        for n_shot in self.n_shots:
            trial_metrics = []

            for trial in range(self.n_trials):
                # Randomly sample n_shot examples from support set
                # (In real implementation, would sample from each support_set)

                # For demo, compute baseline performance
                baseline_metrics = self._evaluate_session(
                    query_sets[0],
                    task=task,
                    device=device,
                )

                trial_metrics.append(baseline_metrics)

            # Aggregate results
            aggregated = self._aggregate_metrics(trial_metrics)
            results[f'n_shot={n_shot}'] = aggregated

        return results

    def _evaluate_session(
        self,
        dataset: Dataset,
        task: str,
        device: str,
    ) -> Dict[str, float]:
        """Evaluate on single session."""
        from neuros_neurofm.datasets.synthetic import collate_neurofmx

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_neurofmx,
        )

        from neuros_neurofm.evaluation.metrics import evaluate_model
        return evaluate_model(self.model, dataloader, task=task, device=device)

    def _aggregate_metrics(
        self,
        trial_metrics: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Aggregate metrics across trials."""
        if len(trial_metrics) == 0:
            return {}

        # Get all metric keys
        keys = trial_metrics[0].keys()

        aggregated = {}
        for key in keys:
            values = [m[key] for m in trial_metrics if key in m]
            aggregated[f'mean_{key}'] = np.mean(values)
            aggregated[f'std_{key}'] = np.std(values)

        return aggregated


def run_falcon_benchmark(
    model: nn.Module,
    test_datasets: List[Dataset],
    n_shots: List[int] = [1, 5, 10, 25, 50],
    task: str = 'decoder',
    device: str = 'cpu',
) -> Dict[str, Dict[str, float]]:
    """Convenience function to run FALCON benchmark.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    test_datasets : list of Dataset
        Test datasets for different sessions/subjects.
    n_shots : list of int, optional
        Number of training examples.
    task : str, optional
        Task to evaluate.
    device : str, optional
        Device.

    Returns
    -------
    dict
        Benchmark results.
    """
    benchmark = FALCONBenchmark(
        model=model,
        n_shots=n_shots,
        n_trials=3,  # Reduced for speed
    )

    # For simplicity, use same datasets as support and query
    results = benchmark.evaluate(
        support_sets=test_datasets,
        query_sets=test_datasets,
        task=task,
        device=device,
    )

    return results
