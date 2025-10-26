"""
Few-Shot Evaluation for NeuroFMX.

Evaluates model adaptation with limited labeled data (K=1, 5, 10, 25, 50 shots).
Uses episode-based sampling, LoRA adapters, and bootstrapping for confidence intervals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, RandomSampler
from tqdm import tqdm

from neuros_neurofm.evaluation.task_registry import EvaluationTask, TaskType
from neuros_neurofm.evaluation.metrics import (
    r2_score,
    pearson_correlation,
    bits_per_spike,
)


@dataclass
class FewShotConfig:
    """Configuration for few-shot evaluation.

    Parameters
    ----------
    k_shots : list
        Number of shots to evaluate (e.g., [1, 5, 10, 25, 50]).
    n_episodes : int
        Number of episodes per k-shot setting.
    n_query_samples : int
        Number of query samples per episode.
    adaptation_method : str
        Method for adaptation ('linear_probe', 'lora', 'full_finetune').
    lora_rank : int
        Rank for LoRA adapters.
    lora_alpha : float
        LoRA scaling parameter.
    adaptation_lr : float
        Learning rate for adaptation.
    adaptation_steps : int
        Number of adaptation steps.
    optimizer : str
        Optimizer type ('adamw', 'sgd').
    freeze_backbone : bool
        Whether to freeze backbone during adaptation.
    use_meta_learning : bool
        Use meta-learning (MAML-style) for adaptation.
    bootstrap_samples : int
        Number of bootstrap samples for confidence intervals.
    device : str
        Computation device.
    random_seed : int
        Random seed for reproducibility.
    """
    k_shots: List[int] = None
    n_episodes: int = 100
    n_query_samples: int = 100
    adaptation_method: str = "lora"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    adaptation_lr: float = 1e-4
    adaptation_steps: int = 100
    optimizer: str = "adamw"
    freeze_backbone: bool = True
    use_meta_learning: bool = False
    bootstrap_samples: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42

    def __post_init__(self):
        if self.k_shots is None:
            self.k_shots = [1, 5, 10, 25, 50]


class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) layer.

    Parameters
    ----------
    original_layer : nn.Linear
        Original linear layer to adapt.
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling parameter.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA parameters
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Adapted output.
        """
        # Original output
        original_out = self.original_layer(x)

        # LoRA adaptation
        lora_out = (x @ self.lora_A) @ self.lora_B
        lora_out = lora_out * self.scaling

        return original_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Apply LoRA to linear layers in model.

    Parameters
    ----------
    model : nn.Module
        Model to adapt.
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling.
    target_modules : list, optional
        Names of modules to apply LoRA to.

    Returns
    -------
    model : nn.Module
        Model with LoRA layers.
    """
    if target_modules is None:
        target_modules = ["query", "key", "value", "output", "fc"]

    for name, module in model.named_modules():
        # Check if module should be adapted
        should_adapt = any(target in name for target in target_modules)

        if should_adapt and isinstance(module, nn.Linear):
            # Replace with LoRA layer
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent = model.get_submodule(parent_name) if parent_name else model
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)

    return model


class FewShotEvaluator:
    """Few-shot evaluator for NeuroFMX.

    Evaluates model adaptation to new tasks with limited labeled data.

    Parameters
    ----------
    model : nn.Module
        Pretrained NeuroFMX model.
    config : FewShotConfig, optional
        Few-shot evaluation configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[FewShotConfig] = None,
    ):
        self.base_model = model
        self.config = config or FewShotConfig()
        self.base_model.to(self.config.device)

        # Set random seeds
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def create_episode(
        self,
        dataset: torch.utils.data.Dataset,
        k_shot: int,
        n_query: int,
    ) -> Tuple[Subset, Subset]:
        """Create a few-shot episode (support + query sets).

        Parameters
        ----------
        dataset : Dataset
            Full dataset.
        k_shot : int
            Number of support examples.
        n_query : int
            Number of query examples.

        Returns
        -------
        support_set : Subset
            Support set.
        query_set : Subset
            Query set.
        """
        n_samples = len(dataset)
        indices = np.random.permutation(n_samples)

        support_indices = indices[:k_shot]
        query_indices = indices[k_shot:k_shot + n_query]

        support_set = Subset(dataset, support_indices)
        query_set = Subset(dataset, query_indices)

        return support_set, query_set

    def adapt_model(
        self,
        model: nn.Module,
        support_loader: DataLoader,
        task_type: TaskType,
    ) -> nn.Module:
        """Adapt model to support set.

        Parameters
        ----------
        model : nn.Module
            Model to adapt.
        support_loader : DataLoader
            Support set dataloader.
        task_type : TaskType
            Type of task.

        Returns
        -------
        adapted_model : nn.Module
            Adapted model.
        """
        model.train()

        # Setup optimizer
        if self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.config.adaptation_lr,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.config.adaptation_lr,
            )

        # Setup loss
        if task_type in [TaskType.CLASSIFICATION, TaskType.MULTI_CLASS]:
            criterion = nn.CrossEntropyLoss()
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        # Adaptation loop
        for step in range(self.config.adaptation_steps):
            total_loss = 0.0
            n_batches = 0

            for batch in support_loader:
                neural_data = batch['neural'].to(self.config.device)
                targets = batch.get('target', batch.get('behavior_target')).to(self.config.device)

                # Forward pass
                if hasattr(model, 'encode'):
                    outputs = model.encode(neural_data)
                else:
                    outputs = model(neural_data)

                # Pool if needed
                if len(outputs.shape) == 3:
                    outputs = outputs.mean(dim=1)

                # Get predictions
                if hasattr(model, 'decode_head'):
                    predictions = model.decode_head(outputs)
                else:
                    predictions = outputs

                # Compute loss
                loss = criterion(predictions, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return model

    def evaluate_episode(
        self,
        model: nn.Module,
        query_loader: DataLoader,
        task_type: TaskType,
    ) -> Dict[str, float]:
        """Evaluate adapted model on query set.

        Parameters
        ----------
        model : nn.Module
            Adapted model.
        query_loader : DataLoader
            Query set dataloader.
        task_type : TaskType
            Type of task.

        Returns
        -------
        metrics : dict
            Evaluation metrics.
        """
        model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in query_loader:
                neural_data = batch['neural'].to(self.config.device)
                targets = batch.get('target', batch.get('behavior_target'))

                # Forward pass
                if hasattr(model, 'encode'):
                    outputs = model.encode(neural_data)
                else:
                    outputs = model(neural_data)

                # Pool if needed
                if len(outputs.shape) == 3:
                    outputs = outputs.mean(dim=1)

                # Get predictions
                if hasattr(model, 'decode_head'):
                    predictions = model.decode_head(outputs)
                else:
                    predictions = outputs

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        # Concatenate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = {}

        if task_type in [TaskType.CLASSIFICATION, TaskType.MULTI_CLASS]:
            predicted_classes = all_predictions.argmax(dim=1)
            metrics['accuracy'] = (predicted_classes == all_targets).float().mean().item()
            metrics['loss'] = F.cross_entropy(all_predictions, all_targets).item()

        elif task_type == TaskType.BINARY_CLASSIFICATION:
            predicted_classes = (torch.sigmoid(all_predictions) > 0.5).long().squeeze()
            metrics['accuracy'] = (predicted_classes == all_targets).float().mean().item()
            metrics['loss'] = F.binary_cross_entropy_with_logits(
                all_predictions, all_targets.float()
            ).item()

        elif task_type == TaskType.ENCODING:
            metrics['bits_per_spike'] = bits_per_spike(all_targets, all_predictions).item()
            metrics['correlation'] = pearson_correlation(all_targets, all_predictions).item()

        else:  # Regression
            metrics['r2'] = r2_score(all_targets, all_predictions).item()
            metrics['correlation'] = pearson_correlation(all_targets, all_predictions).item()
            metrics['mse'] = F.mse_loss(all_predictions, all_targets).item()

        return metrics

    def run_few_shot_evaluation(
        self,
        task: EvaluationTask,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Run complete few-shot evaluation on a task.

        Parameters
        ----------
        task : EvaluationTask
            Task to evaluate.
        output_dir : str or Path, optional
            Directory to save results.

        Returns
        -------
        results : dict
            Results for all k-shot settings.
        """
        print(f"Running few-shot evaluation on task: {task.metadata.name}")

        # Get full dataset
        dataset = task.dataset
        if dataset is None:
            raise ValueError("Task dataset not loaded")

        results = {
            'task_name': task.metadata.name,
            'task_type': task.metadata.task_type.value,
            'species': task.metadata.species.value,
            'modality': task.metadata.modality.value,
            'k_shot_results': {},
        }

        # Evaluate for each k-shot setting
        for k_shot in self.config.k_shots:
            print(f"\nEvaluating {k_shot}-shot learning...")

            episode_results = []

            for episode_idx in tqdm(range(self.config.n_episodes), desc=f"{k_shot}-shot"):
                # Create episode
                support_set, query_set = self.create_episode(
                    dataset, k_shot, self.config.n_query_samples
                )

                # Create dataloaders
                support_loader = DataLoader(
                    support_set, batch_size=min(k_shot, 32), shuffle=True
                )
                query_loader = DataLoader(
                    query_set, batch_size=32, shuffle=False
                )

                # Clone base model
                import copy
                model = copy.deepcopy(self.base_model)

                # Apply adaptation method
                if self.config.adaptation_method == "lora":
                    model = apply_lora_to_model(
                        model,
                        rank=self.config.lora_rank,
                        alpha=self.config.lora_alpha,
                    )
                elif self.config.adaptation_method == "linear_probe":
                    # Freeze all except final layer
                    for name, param in model.named_parameters():
                        if "head" not in name and "decode" not in name:
                            param.requires_grad = False
                # else: full_finetune (all params trainable)

                # Adapt model on support set
                model = self.adapt_model(model, support_loader, task.metadata.task_type)

                # Evaluate on query set
                episode_metrics = self.evaluate_episode(
                    model, query_loader, task.metadata.task_type
                )

                episode_results.append(episode_metrics)

                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Aggregate episode results
            aggregated = self._aggregate_episode_results(episode_results)
            results['k_shot_results'][k_shot] = aggregated

        # Save results
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            import json
            results_path = output_dir / f"{task.metadata.name}_few_shot_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Results saved to {results_path}")

        return results

    def _aggregate_episode_results(
        self, episode_results: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate results across episodes.

        Parameters
        ----------
        episode_results : list
            List of episode metrics.

        Returns
        -------
        aggregated : dict
            Mean, std, and confidence intervals for each metric.
        """
        # Extract all metrics
        metric_names = episode_results[0].keys()
        aggregated = {}

        for metric in metric_names:
            values = np.array([ep[metric] for ep in episode_results])

            # Compute statistics
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

            # Bootstrap confidence intervals
            if len(values) >= 10:
                ci_lower, ci_upper = self._bootstrap_ci(values)
                aggregated[metric]['ci_lower'] = float(ci_lower)
                aggregated[metric]['ci_upper'] = float(ci_upper)

        return aggregated

    def _bootstrap_ci(
        self, values: np.ndarray, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval.

        Parameters
        ----------
        values : np.ndarray
            Sample values.
        alpha : float
            Significance level (default: 0.05 for 95% CI).

        Returns
        -------
        ci_lower : float
        ci_upper : float
        """
        n_samples = len(values)
        bootstrap_means = []

        for _ in range(self.config.bootstrap_samples):
            sample = np.random.choice(values, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(sample))

        ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return ci_lower, ci_upper

    def generate_learning_curve_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate markdown report with learning curves.

        Parameters
        ----------
        results : dict
            Few-shot evaluation results.
        output_path : str or Path, optional
            Path to save report.

        Returns
        -------
        report : str
            Markdown report.
        """
        lines = [
            f"# Few-Shot Learning Curves: {results['task_name']}\n",
            f"## Task Information",
            f"- **Task Type**: {results['task_type']}",
            f"- **Species**: {results['species']}",
            f"- **Modality**: {results['modality']}",
            f"- **Adaptation Method**: {self.config.adaptation_method}",
            f"- **Episodes per K**: {self.config.n_episodes}\n",
            f"## Results by K-Shot\n",
        ]

        # Table header
        metric_names = list(next(iter(results['k_shot_results'].values())).keys())
        primary_metric = metric_names[0]

        lines.append("| K-Shot | " + " | ".join([
            f"{m.title()} (Mean ± Std)" for m in metric_names
        ]) + " |")
        lines.append("|--------|" + "|".join(["--------" for _ in metric_names]) + "|")

        # Table rows
        for k_shot in sorted(results['k_shot_results'].keys()):
            k_results = results['k_shot_results'][k_shot]
            row = f"| {k_shot} |"

            for metric in metric_names:
                mean = k_results[metric]['mean']
                std = k_results[metric]['std']
                row += f" {mean:.4f} ± {std:.4f} |"

            lines.append(row)

        lines.append("\n## Summary\n")

        # Best performance
        best_k = max(
            results['k_shot_results'].keys(),
            key=lambda k: results['k_shot_results'][k][primary_metric]['mean']
        )
        best_score = results['k_shot_results'][best_k][primary_metric]['mean']

        lines.append(f"- **Best {primary_metric}**: {best_score:.4f} at K={best_k}")

        # 1-shot performance
        if 1 in results['k_shot_results']:
            one_shot_score = results['k_shot_results'][1][primary_metric]['mean']
            lines.append(f"- **1-shot {primary_metric}**: {one_shot_score:.4f}")

        report = '\n'.join(lines)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

        return report


def run_few_shot_suite(
    model: nn.Module,
    tasks: List[EvaluationTask],
    config: Optional[FewShotConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run few-shot evaluation on multiple tasks.

    Parameters
    ----------
    model : nn.Module
        Pretrained model.
    tasks : list
        List of evaluation tasks.
    config : FewShotConfig, optional
        Evaluation config.
    output_dir : str or Path, optional
        Output directory.

    Returns
    -------
    all_results : dict
        Results for all tasks.
    """
    evaluator = FewShotEvaluator(model, config)
    all_results = {}

    for task in tasks:
        try:
            results = evaluator.run_few_shot_evaluation(task, output_dir)
            all_results[task.metadata.name] = results

            # Generate learning curve report
            if output_dir:
                report_path = Path(output_dir) / f"{task.metadata.name}_learning_curves.md"
                evaluator.generate_learning_curve_report(results, report_path)

        except Exception as e:
            print(f"Error evaluating task {task.metadata.name}: {e}")
            all_results[task.metadata.name] = {'error': str(e)}

    return all_results
