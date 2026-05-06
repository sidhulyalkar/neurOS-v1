"""
Experiment tracking system for neuros-astro experiments.

Enables systematic tracking of experimental conditions, parameters,
and results for reproducibility and comparison.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


@dataclass
class ExperimentConfig:
    """Configuration for a neuros-astro experiment."""

    # Experiment identification
    experiment_id: str
    experiment_name: str
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Data configuration
    dataset_name: str = "unknown"
    session_ids: List[str] = field(default_factory=list)
    data_path: Optional[str] = None

    # Processing parameters
    frame_rate_hz: float = 10.0
    z_threshold: float = 2.5
    min_duration_frames: int = 5
    min_amplitude: float = 0.2

    # Network parameters
    window_size_s: float = 60.0
    coactivation_threshold_s: float = 1.0

    # Model configuration (for ablation studies)
    modalities_enabled: List[str] = field(default_factory=lambda: ['neural', 'astro'])
    model_architecture: Optional[str] = None
    model_parameters: Dict[str, Any] = field(default_factory=dict)

    # Computational
    random_seed: int = 42
    n_workers: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"✓ Saved config to {path}")

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)


@dataclass
class ExperimentResult:
    """Results from a neuros-astro experiment."""

    # Identification
    experiment_id: str
    config: ExperimentConfig

    # Processing results
    n_events_detected: int = 0
    n_regions: int = 0
    n_networks: int = 0

    # Event statistics
    mean_event_duration_s: float = 0.0
    mean_event_amplitude: float = 0.0
    event_rate_hz: float = 0.0

    # Network statistics
    mean_network_density: float = 0.0
    mean_network_clustering: float = 0.0
    network_stability: float = 0.0

    # Model performance (for ablation studies)
    model_metrics: Dict[str, float] = field(default_factory=dict)
    # e.g., {'prediction_loss': 0.123, 'decoding_accuracy': 0.85}

    # Timing
    processing_time_s: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Output paths
    output_dir: Optional[str] = None
    events_path: Optional[str] = None
    networks_path: Optional[str] = None
    figures_dir: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        # Config is nested, convert separately
        result['config'] = self.config.to_dict()
        return result

    def to_json(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"✓ Saved results to {path}")

    @classmethod
    def from_json(cls, path: str | Path) -> 'ExperimentResult':
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct config
        config = ExperimentConfig.from_dict(data.pop('config'))

        return cls(config=config, **data)


class ExperimentTracker:
    """
    Tracker for managing multiple experiments.

    Provides a registry of experiments with their configurations and results.
    """

    def __init__(self, tracker_dir: str | Path = "./experiments"):
        """
        Initialize experiment tracker.

        Args:
            tracker_dir: Directory to store experiment registry and results
        """
        self.tracker_dir = Path(tracker_dir)
        self.tracker_dir.mkdir(parents=True, exist_ok=True)

        self.registry_path = self.tracker_dir / "registry.json"
        self.experiments: Dict[str, ExperimentResult] = {}

        # Load existing registry if available
        if self.registry_path.exists():
            self.load_registry()

    def register_experiment(
        self,
        config: ExperimentConfig,
    ) -> str:
        """
        Register a new experiment.

        Args:
            config: ExperimentConfig object

        Returns:
            Experiment ID
        """
        experiment_id = config.experiment_id

        # Create experiment directory
        exp_dir = self.tracker_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.to_json(exp_dir / "config.json")

        print(f"✓ Registered experiment: {experiment_id}")
        print(f"  Directory: {exp_dir}")

        return experiment_id

    def save_result(
        self,
        result: ExperimentResult,
    ) -> None:
        """
        Save experiment result.

        Args:
            result: ExperimentResult object
        """
        experiment_id = result.experiment_id

        # Create experiment directory if not exists
        exp_dir = self.tracker_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save result
        result.to_json(exp_dir / "result.json")

        # Update registry
        self.experiments[experiment_id] = result
        self.save_registry()

        print(f"✓ Saved result for experiment: {experiment_id}")

    def load_result(
        self,
        experiment_id: str,
    ) -> Optional[ExperimentResult]:
        """
        Load experiment result.

        Args:
            experiment_id: Experiment ID

        Returns:
            ExperimentResult object or None if not found
        """
        result_path = self.tracker_dir / experiment_id / "result.json"

        if not result_path.exists():
            print(f"⚠️  No result found for experiment: {experiment_id}")
            return None

        result = ExperimentResult.from_json(result_path)
        return result

    def list_experiments(self) -> List[str]:
        """
        List all registered experiments.

        Returns:
            List of experiment IDs
        """
        return list(self.experiments.keys())

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric: str = "event_rate_hz",
    ) -> Dict[str, float]:
        """
        Compare metric across experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metric: Metric to compare

        Returns:
            Dict mapping experiment_id -> metric_value
        """
        comparison = {}

        for exp_id in experiment_ids:
            result = self.load_result(exp_id)

            if result is None:
                continue

            # Try to get metric from result
            if hasattr(result, metric):
                value = getattr(result, metric)
            elif metric in result.model_metrics:
                value = result.model_metrics[metric]
            else:
                value = None

            comparison[exp_id] = value

        return comparison

    def save_registry(self) -> None:
        """Save experiment registry to JSON."""
        registry_data = {
            'experiments': {
                exp_id: result.to_dict()
                for exp_id, result in self.experiments.items()
            },
            'last_updated': datetime.now().isoformat(),
        }

        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)

    def load_registry(self) -> None:
        """Load experiment registry from JSON."""
        with open(self.registry_path, 'r') as f:
            registry_data = json.load(f)

        for exp_id, result_data in registry_data.get('experiments', {}).items():
            config_data = result_data.pop('config')
            config = ExperimentConfig.from_dict(config_data)
            result = ExperimentResult(config=config, **result_data)
            self.experiments[exp_id] = result

        print(f"✓ Loaded {len(self.experiments)} experiments from registry")

    def generate_summary_report(
        self,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Generate summary report of all experiments.

        Args:
            output_path: Optional path to save report

        Returns:
            Summary report as string
        """
        report_lines = [
            "=" * 80,
            "EXPERIMENT TRACKER SUMMARY",
            "=" * 80,
            f"Total experiments: {len(self.experiments)}",
            f"Tracker directory: {self.tracker_dir}",
            "",
        ]

        if not self.experiments:
            report_lines.append("No experiments registered yet.")
        else:
            report_lines.append("Experiments:")
            report_lines.append("-" * 80)

            for exp_id, result in self.experiments.items():
                report_lines.extend([
                    f"\n{exp_id}:",
                    f"  Name: {result.config.experiment_name}",
                    f"  Dataset: {result.config.dataset_name}",
                    f"  Events detected: {result.n_events_detected}",
                    f"  Event rate: {result.event_rate_hz:.4f} Hz",
                    f"  Processing time: {result.processing_time_s:.2f}s",
                ])

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"✓ Saved summary report to {output_path}")

        return report
