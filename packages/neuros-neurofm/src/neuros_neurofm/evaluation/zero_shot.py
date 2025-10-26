"""
Zero-Shot Evaluation for NeuroFMX.

Evaluates pretrained representations using frozen embeddings + linear probes.
Tests which layers learn the most transferable representations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuros_neurofm.evaluation.task_registry import EvaluationTask, TaskType
from neuros_neurofm.evaluation.metrics import (
    r2_score,
    pearson_correlation,
    bits_per_spike,
)


@dataclass
class ZeroShotConfig:
    """Configuration for zero-shot evaluation.

    Parameters
    ----------
    layers_to_probe : list, optional
        Layer indices to evaluate. If None, probes all layers.
    probe_lr : float
        Learning rate for linear probe training.
    probe_epochs : int
        Number of epochs to train probe.
    probe_batch_size : int
        Batch size for probe training.
    l2_regularization : float
        L2 regularization strength.
    early_stopping_patience : int
        Patience for early stopping.
    use_layer_norm : bool
        Apply layer normalization to representations.
    cache_representations : bool
        Cache extracted representations to disk.
    device : str
        Device for computation.
    num_workers : int
        DataLoader workers.
    """
    layers_to_probe: Optional[List[int]] = None
    probe_lr: float = 1e-3
    probe_epochs: int = 100
    probe_batch_size: int = 128
    l2_regularization: float = 1e-4
    early_stopping_patience: int = 10
    use_layer_norm: bool = True
    cache_representations: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4


class LinearProbe(nn.Module):
    """Linear probe for zero-shot evaluation.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output dimension.
    task_type : TaskType
        Type of task (regression, classification, etc.).
    n_classes : int, optional
        Number of classes for classification.
    use_layer_norm : bool
        Apply layer normalization to inputs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: TaskType,
        n_classes: Optional[int] = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.n_classes = n_classes

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()

        # Linear layer
        if task_type == TaskType.CLASSIFICATION or task_type == TaskType.MULTI_CLASS:
            self.probe = nn.Linear(input_dim, n_classes)
        else:
            self.probe = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, input_dim).

        Returns
        -------
        output : torch.Tensor
            Predictions.
        """
        x = self.layer_norm(x)
        return self.probe(x)


class ZeroShotEvaluator:
    """Zero-shot evaluator for pretrained NeuroFMX models.

    Extracts frozen representations and trains linear probes to evaluate
    the quality of learned features for downstream tasks.

    Parameters
    ----------
    model : nn.Module
        Pretrained NeuroFMX model.
    config : ZeroShotConfig, optional
        Evaluation configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ZeroShotConfig] = None,
    ):
        self.model = model
        self.config = config or ZeroShotConfig()
        self.model.to(self.config.device)
        self.model.eval()

        # Storage for cached representations
        self._representation_cache: Dict[str, torch.Tensor] = {}

    def extract_representations(
        self,
        dataloader: DataLoader,
        layer_idx: Optional[int] = None,
        pool_method: str = "mean",
        return_targets: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract frozen representations from model.

        Parameters
        ----------
        dataloader : DataLoader
            Data to extract representations from.
        layer_idx : int, optional
            Layer index to extract from. If None, uses final layer.
        pool_method : str
            Pooling method ('mean', 'max', 'cls', 'last').
        return_targets : bool
            Whether to return targets.

        Returns
        -------
        representations : torch.Tensor
            Extracted features, shape (n_samples, feature_dim).
        targets : torch.Tensor, optional
            Target values/labels.
        """
        representations_list = []
        targets_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting representations"):
                # Move to device
                neural_data = batch['neural'].to(self.config.device)

                if return_targets:
                    targets = batch.get('target', batch.get('behavior_target'))
                    if targets is not None:
                        targets_list.append(targets.cpu())

                # Forward pass through model
                if hasattr(self.model, 'encode'):
                    # NeuroFMX with encode method
                    outputs = self.model.encode(neural_data)
                else:
                    # Generic forward pass
                    outputs = self.model(neural_data)

                # Extract from specific layer if requested
                if layer_idx is not None and hasattr(self.model, 'get_layer_output'):
                    outputs = self.model.get_layer_output(neural_data, layer_idx)

                # Pool temporal dimension
                if len(outputs.shape) == 3:  # (batch, time, features)
                    if pool_method == "mean":
                        outputs = outputs.mean(dim=1)
                    elif pool_method == "max":
                        outputs = outputs.max(dim=1)[0]
                    elif pool_method == "last":
                        outputs = outputs[:, -1, :]
                    elif pool_method == "cls":
                        outputs = outputs[:, 0, :]  # CLS token

                representations_list.append(outputs.cpu())

        # Concatenate all batches
        representations = torch.cat(representations_list, dim=0)
        targets = torch.cat(targets_list, dim=0) if targets_list else None

        return representations, targets

    def train_linear_probe(
        self,
        probe: LinearProbe,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task_type: TaskType = TaskType.REGRESSION,
    ) -> Dict[str, List[float]]:
        """Train a linear probe on frozen representations.

        Parameters
        ----------
        probe : LinearProbe
            Linear probe model.
        train_loader : DataLoader
            Training data (representations, targets).
        val_loader : DataLoader, optional
            Validation data.
        task_type : TaskType
            Type of task.

        Returns
        -------
        history : dict
            Training history (losses, metrics).
        """
        probe.to(self.config.device)
        probe.train()

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=self.config.probe_lr,
            weight_decay=self.config.l2_regularization,
        )

        # Setup loss function
        if task_type in [TaskType.CLASSIFICATION, TaskType.MULTI_CLASS]:
            criterion = nn.CrossEntropyLoss()
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': [],
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.probe_epochs):
            # Training
            probe.train()
            train_losses = []

            for batch in train_loader:
                representations = batch[0].to(self.config.device)
                targets = batch[1].to(self.config.device)

                # Forward pass
                predictions = probe(representations)

                # Compute loss
                loss = criterion(predictions, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                val_loss, val_metric = self.evaluate_probe(
                    probe, val_loader, task_type
                )
                history['val_loss'].append(val_loss)
                history['val_metric'].append(val_metric)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def evaluate_probe(
        self,
        probe: LinearProbe,
        dataloader: DataLoader,
        task_type: TaskType,
    ) -> Tuple[float, float]:
        """Evaluate linear probe.

        Parameters
        ----------
        probe : LinearProbe
            Trained probe.
        dataloader : DataLoader
            Evaluation data.
        task_type : TaskType
            Type of task.

        Returns
        -------
        loss : float
            Average loss.
        metric : float
            Task-specific metric.
        """
        probe.eval()

        all_predictions = []
        all_targets = []
        losses = []

        # Setup loss
        if task_type in [TaskType.CLASSIFICATION, TaskType.MULTI_CLASS]:
            criterion = nn.CrossEntropyLoss()
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        with torch.no_grad():
            for batch in dataloader:
                representations = batch[0].to(self.config.device)
                targets = batch[1].to(self.config.device)

                predictions = probe(representations)
                loss = criterion(predictions, targets)

                losses.append(loss.item())
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        # Concatenate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metric
        if task_type in [TaskType.CLASSIFICATION, TaskType.MULTI_CLASS]:
            # Accuracy
            predicted_classes = all_predictions.argmax(dim=1)
            metric = (predicted_classes == all_targets).float().mean().item()
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            # Binary accuracy
            predicted_classes = (torch.sigmoid(all_predictions) > 0.5).long().squeeze()
            metric = (predicted_classes == all_targets).float().mean().item()
        elif task_type == TaskType.ENCODING:
            # Bits per spike
            metric = bits_per_spike(all_targets, all_predictions).item()
        else:
            # R² for regression
            metric = r2_score(all_targets, all_predictions).item()

        return np.mean(losses), metric

    def run_zero_shot_evaluation(
        self,
        task: EvaluationTask,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Run complete zero-shot evaluation on a task.

        Parameters
        ----------
        task : EvaluationTask
            Task to evaluate.
        output_dir : str or Path, optional
            Directory to save results.

        Returns
        -------
        results : dict
            Evaluation results with per-layer metrics.
        """
        print(f"Running zero-shot evaluation on task: {task.metadata.name}")

        # Get dataloaders
        train_loader, val_loader, test_loader = task.get_dataloaders(
            batch_size=self.config.probe_batch_size,
            num_workers=self.config.num_workers,
        )

        # Extract representations (using final layer for now)
        print("Extracting representations...")
        train_reps, train_targets = self.extract_representations(
            train_loader, return_targets=True
        )
        val_reps, val_targets = self.extract_representations(
            val_loader, return_targets=True
        ) if val_loader else (None, None)
        test_reps, test_targets = self.extract_representations(
            test_loader, return_targets=True
        )

        # Create representation dataloaders
        train_rep_loader = DataLoader(
            list(zip(train_reps, train_targets)),
            batch_size=self.config.probe_batch_size,
            shuffle=True,
        )
        val_rep_loader = DataLoader(
            list(zip(val_reps, val_targets)),
            batch_size=self.config.probe_batch_size,
        ) if val_reps is not None else None
        test_rep_loader = DataLoader(
            list(zip(test_reps, test_targets)),
            batch_size=self.config.probe_batch_size,
        )

        # Determine output dimension
        if task.metadata.task_type in [TaskType.CLASSIFICATION, TaskType.MULTI_CLASS]:
            output_dim = task.metadata.n_classes
        else:
            output_dim = task.metadata.output_dim or train_targets.shape[-1]

        # Create and train linear probe
        print("Training linear probe...")
        probe = LinearProbe(
            input_dim=train_reps.shape[-1],
            output_dim=output_dim,
            task_type=task.metadata.task_type,
            n_classes=task.metadata.n_classes,
            use_layer_norm=self.config.use_layer_norm,
        )

        history = self.train_linear_probe(
            probe,
            train_rep_loader,
            val_rep_loader,
            task.metadata.task_type,
        )

        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_metric = self.evaluate_probe(
            probe, test_rep_loader, task.metadata.task_type
        )

        # Compile results
        results = {
            'task_name': task.metadata.name,
            'task_type': task.metadata.task_type.value,
            'species': task.metadata.species.value,
            'modality': task.metadata.modality.value,
            'test_loss': test_loss,
            'test_metric': test_metric,
            'metric_name': self._get_metric_name(task.metadata.task_type),
            'training_history': history,
            'representation_dim': train_reps.shape[-1],
            'n_train_samples': len(train_reps),
            'n_test_samples': len(test_reps),
        }

        # Save results
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save results as JSON
            import json
            results_path = output_dir / f"{task.metadata.name}_zero_shot_results.json"
            with open(results_path, 'w') as f:
                # Convert to JSON-serializable format
                json_results = {k: v for k, v in results.items() if k != 'training_history'}
                json.dump(json_results, f, indent=2)

            print(f"Results saved to {results_path}")

        return results

    def _get_metric_name(self, task_type: TaskType) -> str:
        """Get metric name for task type."""
        if task_type in [TaskType.CLASSIFICATION, TaskType.MULTI_CLASS, TaskType.BINARY_CLASSIFICATION]:
            return "accuracy"
        elif task_type == TaskType.ENCODING:
            return "bits_per_spike"
        else:
            return "r2_score"

    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate markdown report for zero-shot evaluation.

        Parameters
        ----------
        results : dict
            Evaluation results.
        output_path : str or Path, optional
            Path to save report.

        Returns
        -------
        report : str
            Markdown-formatted report.
        """
        lines = [
            f"# Zero-Shot Evaluation Report: {results['task_name']}\n",
            f"## Task Information",
            f"- **Task Type**: {results['task_type']}",
            f"- **Species**: {results['species']}",
            f"- **Modality**: {results['modality']}",
            f"- **Representation Dimension**: {results['representation_dim']}",
            f"- **Training Samples**: {results['n_train_samples']}",
            f"- **Test Samples**: {results['n_test_samples']}\n",
            f"## Results",
            f"- **Test Loss**: {results['test_loss']:.4f}",
            f"- **Test {results['metric_name'].title()}**: {results['test_metric']:.4f}\n",
        ]

        report = '\n'.join(lines)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

        return report


def run_zero_shot_suite(
    model: nn.Module,
    tasks: List[EvaluationTask],
    config: Optional[ZeroShotConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run zero-shot evaluation on multiple tasks.

    Parameters
    ----------
    model : nn.Module
        Pretrained model.
    tasks : list
        List of evaluation tasks.
    config : ZeroShotConfig, optional
        Evaluation config.
    output_dir : str or Path, optional
        Output directory.

    Returns
    -------
    all_results : dict
        Results for all tasks.
    """
    evaluator = ZeroShotEvaluator(model, config)
    all_results = {}

    for task in tasks:
        try:
            results = evaluator.run_zero_shot_evaluation(task, output_dir)
            all_results[task.metadata.name] = results

            # Generate report
            if output_dir:
                report_path = Path(output_dir) / f"{task.metadata.name}_report.md"
                evaluator.generate_report(results, report_path)

        except Exception as e:
            print(f"Error evaluating task {task.metadata.name}: {e}")
            all_results[task.metadata.name] = {'error': str(e)}

    return all_results
