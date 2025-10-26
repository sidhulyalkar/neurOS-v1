"""
Task Registry for NeuroFMX Evaluation.

Provides a comprehensive system for registering, discovering, and managing
evaluation tasks across different species, modalities, and task types.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset


class TaskType(Enum):
    """Types of evaluation tasks."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS = "multi_class"
    FORECASTING = "forecasting"
    ENCODING = "encoding"
    DECODING = "decoding"


class Species(Enum):
    """Species for neural recordings."""
    MOUSE = "mouse"
    MONKEY = "monkey"
    HUMAN = "human"
    RAT = "rat"
    ZEBRAFISH = "zebrafish"
    MULTI_SPECIES = "multi_species"


class Modality(Enum):
    """Neural recording modalities."""
    SPIKES = "spikes"
    LFP = "lfp"
    EEG = "eeg"
    ECOG = "ecog"
    FMRI = "fmri"
    CALCIUM = "calcium"
    VOLTAGE = "voltage"
    MULTI_MODAL = "multi_modal"


@dataclass
class TaskMetadata:
    """Metadata for an evaluation task.

    Parameters
    ----------
    name : str
        Unique task identifier.
    task_type : TaskType
        Type of task (regression, classification, etc.).
    species : Species
        Species of the neural recordings.
    modality : Modality
        Recording modality.
    target : str
        Description of the prediction target.
    metric : str or list
        Primary evaluation metric(s).
    description : str, optional
        Detailed task description.
    n_classes : int, optional
        Number of classes (for classification).
    output_dim : int, optional
        Output dimensionality.
    dataset_path : str, optional
        Path to dataset.
    reference : str, optional
        Citation or reference for the task.
    difficulty : str, optional
        Task difficulty level ('easy', 'medium', 'hard').
    tags : list, optional
        Additional tags for task categorization.
    """
    name: str
    task_type: TaskType
    species: Species
    modality: Modality
    target: str
    metric: Union[str, List[str]]
    description: str = ""
    n_classes: Optional[int] = None
    output_dim: Optional[int] = None
    dataset_path: Optional[str] = None
    reference: Optional[str] = None
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "task_type": self.task_type.value,
            "species": self.species.value,
            "modality": self.modality.value,
            "target": self.target,
            "metric": self.metric,
            "description": self.description,
            "n_classes": self.n_classes,
            "output_dim": self.output_dim,
            "dataset_path": self.dataset_path,
            "reference": self.reference,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMetadata":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            task_type=TaskType(data["task_type"]),
            species=Species(data["species"]),
            modality=Modality(data["modality"]),
            target=data["target"],
            metric=data["metric"],
            description=data.get("description", ""),
            n_classes=data.get("n_classes"),
            output_dim=data.get("output_dim"),
            dataset_path=data.get("dataset_path"),
            reference=data.get("reference"),
            difficulty=data.get("difficulty", "medium"),
            tags=data.get("tags", []),
        )


@dataclass
class EvaluationTask:
    """Complete evaluation task specification.

    Parameters
    ----------
    metadata : TaskMetadata
        Task metadata.
    dataset : Dataset, optional
        PyTorch dataset.
    train_indices : list, optional
        Training set indices.
    val_indices : list, optional
        Validation set indices.
    test_indices : list, optional
        Test set indices.
    metric_fn : callable, optional
        Custom metric function.
    preprocessing_fn : callable, optional
        Preprocessing function.
    """
    metadata: TaskMetadata
    dataset: Optional[Dataset] = None
    train_indices: Optional[List[int]] = None
    val_indices: Optional[List[int]] = None
    test_indices: Optional[List[int]] = None
    metric_fn: Optional[Callable] = None
    preprocessing_fn: Optional[Callable] = None

    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test dataloaders.

        Parameters
        ----------
        batch_size : int
            Batch size.
        num_workers : int
            Number of dataloader workers.
        pin_memory : bool
            Pin memory for faster GPU transfer.

        Returns
        -------
        train_loader : DataLoader
        val_loader : DataLoader
        test_loader : DataLoader
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded for this task")

        # Create subsets
        train_dataset = Subset(self.dataset, self.train_indices) if self.train_indices else self.dataset
        val_dataset = Subset(self.dataset, self.val_indices) if self.val_indices else None
        test_dataset = Subset(self.dataset, self.test_indices) if self.test_indices else None

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ) if val_dataset else None

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ) if test_dataset else None

        return train_loader, val_loader, test_loader

    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = False,
    ):
        """Split dataset into train/val/test sets.

        Parameters
        ----------
        train_ratio : float
            Proportion for training.
        val_ratio : float
            Proportion for validation.
        test_ratio : float
            Proportion for testing.
        random_seed : int
            Random seed for reproducibility.
        stratify : bool
            Whether to stratify split (for classification).
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded")

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        n_samples = len(self.dataset)
        indices = np.arange(n_samples)

        # Get labels for stratification if needed
        stratify_labels = None
        if stratify and hasattr(self.dataset, 'targets'):
            stratify_labels = self.dataset.targets

        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=stratify_labels,
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            random_state=random_seed,
            stratify=stratify_labels[temp_indices] if stratify_labels is not None else None,
        )

        self.train_indices = train_indices.tolist()
        self.val_indices = val_indices.tolist()
        self.test_indices = test_indices.tolist()


class TaskRegistry:
    """Registry for managing evaluation tasks.

    Provides centralized registration, discovery, and loading of
    evaluation tasks from code or YAML configurations.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config file with task definitions.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self._tasks: Dict[str, EvaluationTask] = {}
        self._metadata_cache: Dict[str, TaskMetadata] = {}

        if config_path is not None:
            self.load_from_yaml(config_path)

    def register(
        self,
        task: EvaluationTask,
        overwrite: bool = False,
    ):
        """Register an evaluation task.

        Parameters
        ----------
        task : EvaluationTask
            Task to register.
        overwrite : bool
            Whether to overwrite existing task.
        """
        task_name = task.metadata.name

        if task_name in self._tasks and not overwrite:
            raise ValueError(f"Task '{task_name}' already registered. Use overwrite=True to replace.")

        self._tasks[task_name] = task
        self._metadata_cache[task_name] = task.metadata

    def register_from_metadata(
        self,
        metadata: TaskMetadata,
        dataset: Optional[Dataset] = None,
        **kwargs,
    ) -> EvaluationTask:
        """Register task from metadata.

        Parameters
        ----------
        metadata : TaskMetadata
            Task metadata.
        dataset : Dataset, optional
            PyTorch dataset.
        **kwargs
            Additional task parameters.

        Returns
        -------
        task : EvaluationTask
            Registered task.
        """
        task = EvaluationTask(metadata=metadata, dataset=dataset, **kwargs)
        self.register(task)
        return task

    def get(self, task_name: str) -> EvaluationTask:
        """Get task by name.

        Parameters
        ----------
        task_name : str
            Task identifier.

        Returns
        -------
        task : EvaluationTask
        """
        if task_name not in self._tasks:
            raise KeyError(f"Task '{task_name}' not found in registry")
        return self._tasks[task_name]

    def list_tasks(
        self,
        task_type: Optional[TaskType] = None,
        species: Optional[Species] = None,
        modality: Optional[Modality] = None,
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """List registered tasks with optional filtering.

        Parameters
        ----------
        task_type : TaskType, optional
            Filter by task type.
        species : Species, optional
            Filter by species.
        modality : Modality, optional
            Filter by modality.
        tags : list, optional
            Filter by tags (returns tasks with ANY of these tags).

        Returns
        -------
        task_names : list
            List of matching task names.
        """
        matching_tasks = []

        for name, metadata in self._metadata_cache.items():
            # Apply filters
            if task_type and metadata.task_type != task_type:
                continue
            if species and metadata.species != species:
                continue
            if modality and metadata.modality != modality:
                continue
            if tags and not any(tag in metadata.tags for tag in tags):
                continue

            matching_tasks.append(name)

        return matching_tasks

    def get_metadata(self, task_name: str) -> TaskMetadata:
        """Get task metadata.

        Parameters
        ----------
        task_name : str
            Task identifier.

        Returns
        -------
        metadata : TaskMetadata
        """
        if task_name not in self._metadata_cache:
            raise KeyError(f"Task '{task_name}' not found")
        return self._metadata_cache[task_name]

    def load_from_yaml(self, config_path: Union[str, Path]):
        """Load tasks from YAML configuration.

        Parameters
        ----------
        config_path : str or Path
            Path to YAML config file.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        tasks_config = config.get('tasks', [])

        for task_config in tasks_config:
            # Create metadata
            metadata = TaskMetadata.from_dict(task_config)

            # Register task (without dataset for now)
            task = EvaluationTask(metadata=metadata)
            self.register(task, overwrite=True)

    def save_to_yaml(self, output_path: Union[str, Path]):
        """Save registered tasks to YAML.

        Parameters
        ----------
        output_path : str or Path
            Output YAML file path.
        """
        output_path = Path(output_path)

        config = {
            'tasks': [metadata.to_dict() for metadata in self._metadata_cache.values()]
        }

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def create_task_report(self) -> str:
        """Generate markdown report of all registered tasks.

        Returns
        -------
        report : str
            Markdown-formatted report.
        """
        lines = ["# NeuroFMX Evaluation Tasks\n"]

        # Group by task type
        by_type: Dict[TaskType, List[str]] = {}
        for name, metadata in self._metadata_cache.items():
            if metadata.task_type not in by_type:
                by_type[metadata.task_type] = []
            by_type[metadata.task_type].append(name)

        # Generate sections
        for task_type, task_names in sorted(by_type.items(), key=lambda x: x[0].value):
            lines.append(f"\n## {task_type.value.title()} Tasks\n")

            for name in sorted(task_names):
                metadata = self._metadata_cache[name]
                lines.append(f"\n### {metadata.name}")
                lines.append(f"- **Species**: {metadata.species.value}")
                lines.append(f"- **Modality**: {metadata.modality.value}")
                lines.append(f"- **Target**: {metadata.target}")
                lines.append(f"- **Metric**: {metadata.metric}")
                lines.append(f"- **Difficulty**: {metadata.difficulty}")

                if metadata.description:
                    lines.append(f"\n{metadata.description}\n")

                if metadata.reference:
                    lines.append(f"*Reference: {metadata.reference}*")

                lines.append("")

        # Summary statistics
        lines.append("\n## Summary\n")
        lines.append(f"- **Total tasks**: {len(self._tasks)}")
        lines.append(f"- **Task types**: {', '.join(sorted(set(m.task_type.value for m in self._metadata_cache.values())))}")
        lines.append(f"- **Species**: {', '.join(sorted(set(m.species.value for m in self._metadata_cache.values())))}")
        lines.append(f"- **Modalities**: {', '.join(sorted(set(m.modality.value for m in self._metadata_cache.values())))}")

        return '\n'.join(lines)

    def __len__(self) -> int:
        """Number of registered tasks."""
        return len(self._tasks)

    def __contains__(self, task_name: str) -> bool:
        """Check if task is registered."""
        return task_name in self._tasks

    def __repr__(self) -> str:
        return f"TaskRegistry(n_tasks={len(self._tasks)})"


# Global registry instance
_global_registry = TaskRegistry()


def get_global_registry() -> TaskRegistry:
    """Get the global task registry.

    Returns
    -------
    registry : TaskRegistry
    """
    return _global_registry


def register_task(task: EvaluationTask, **kwargs):
    """Register task in global registry.

    Parameters
    ----------
    task : EvaluationTask
        Task to register.
    **kwargs
        Additional arguments passed to register().
    """
    _global_registry.register(task, **kwargs)


def get_task(task_name: str) -> EvaluationTask:
    """Get task from global registry.

    Parameters
    ----------
    task_name : str
        Task identifier.

    Returns
    -------
    task : EvaluationTask
    """
    return _global_registry.get(task_name)


def list_tasks(**kwargs) -> List[str]:
    """List tasks from global registry.

    Parameters
    ----------
    **kwargs
        Filter arguments (task_type, species, modality, tags).

    Returns
    -------
    task_names : list
    """
    return _global_registry.list_tasks(**kwargs)
