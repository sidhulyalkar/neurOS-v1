"""
Standardized Pipeline for Mechanistic Interpretability Analysis.

Provides a unified workflow for running comprehensive mechanistic interpretability
analyses, with automatic caching, parallelization, and result aggregation.

Key Features:
- Modular analysis stages (SAE, circuits, dynamics, fractals, etc.)
- Automatic result caching via MechIntDatabase
- Parallel execution of independent analyses
- Provenance tracking across analysis chain
- Progress reporting and error handling
- Configurable analysis depth (quick/standard/comprehensive)
- HTML report generation

Example:
    >>> # Quick analysis
    >>> pipeline = MechIntPipeline(model, db_path="./results")
    >>> results = pipeline.run(inputs, analyses=["sae", "circuits", "info_flow"])
    >>>
    >>> # Comprehensive analysis with all methods
    >>> pipeline = MechIntPipeline(model, depth="comprehensive")
    >>> results = pipeline.run_all(inputs, generate_report=True)
    >>>
    >>> # Resume from checkpoint
    >>> results = pipeline.resume(checkpoint_id="20251030_123456")

Author: NeuroS Team
Date: 2025-10-30
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm

from neuros_mechint.results import (
    MechIntResult,
    CircuitResult,
    DynamicsResult,
    InformationResult,
    AlignmentResult,
    FractalResult,
    ResultCollection,
)
from neuros_mechint.database import MechIntDatabase

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for MechIntPipeline."""

    # Analysis depth: 'quick' | 'standard' | 'comprehensive'
    depth: str = "standard"

    # Which analyses to run
    enabled_analyses: Set[str] = field(default_factory=lambda: {
        "sae", "circuits", "info_flow", "dynamics", "fractals"
    })

    # Parallelization
    parallel: bool = True
    max_workers: int = 4

    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None

    # Progress reporting
    verbose: bool = True
    show_progress: bool = True

    # Error handling
    continue_on_error: bool = True

    # Resource limits
    max_memory_gb: Optional[float] = None
    timeout_per_analysis: Optional[int] = None  # seconds

    # Report generation
    auto_report: bool = False
    report_format: str = "html"  # 'html' | 'pdf' | 'markdown'


@dataclass
class AnalysisStage:
    """Represents a single analysis stage in the pipeline."""

    name: str
    analysis_fn: Callable
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0  # Higher priority runs first
    result_type: type = MechIntResult


class MechIntPipeline:
    """
    Standardized pipeline for mechanistic interpretability analysis.

    Orchestrates multiple analysis methods, handles caching, parallelization,
    and result aggregation.

    Args:
        model: Neural network to analyze
        db_path: Path to MechIntDatabase (optional)
        config: PipelineConfig object
        device: Torch device
    """

    def __init__(
        self,
        model: nn.Module,
        db_path: Optional[Union[str, Path]] = None,
        config: Optional[PipelineConfig] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config or PipelineConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize database if path provided
        self.db = None
        if db_path and self.config.use_cache:
            self.db = MechIntDatabase(db_path, verbose=self.config.verbose)

        # Analysis stages registry
        self.stages: Dict[str, AnalysisStage] = {}
        self._register_default_stages()

        # Results cache
        self.results: Dict[str, MechIntResult] = {}
        self.checkpoint_id: Optional[str] = None

        self._log("Initialized MechIntPipeline")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.config.verbose:
            logger.info(f"[MechIntPipeline] {message}")

    def _register_default_stages(self):
        """Register default analysis stages."""

        # SAE Analysis
        def run_sae(inputs: torch.Tensor, **kwargs) -> MechIntResult:
            from neuros_mechint.sparse_autoencoder import SparseAutoencoder

            # Get layer activations
            activations = self._get_activations(inputs, kwargs.get('layer_name', 'layer_3'))

            # Train SAE
            sae = SparseAutoencoder(
                input_dim=activations.shape[-1],
                hidden_dim=kwargs.get('hidden_dim', 512),
                sparsity_lambda=kwargs.get('sparsity', 0.1)
            ).to(self.device)

            # Simple training loop
            optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
            n_epochs = 10 if self.config.depth == 'quick' else 50

            for epoch in range(n_epochs):
                reconstruction, features = sae(activations)
                loss = sae.loss(activations, reconstruction, features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Extract features
            with torch.no_grad():
                _, features = sae(activations)

            return MechIntResult(
                method="SAE",
                data={'features': features.cpu().numpy(), 'reconstruction': reconstruction.cpu().numpy()},
                metadata={'layer': kwargs.get('layer_name', 'layer_3'), 'sparsity': kwargs.get('sparsity', 0.1)},
                metrics={'reconstruction_loss': float(loss), 'sparsity': float(features.abs().mean())}
            )

        # Circuit Discovery
        def run_circuits(inputs: torch.Tensor, **kwargs) -> CircuitResult:
            from neuros_mechint.circuits import AutomatedCircuitDiscovery

            # Run ACDC
            acdc = AutomatedCircuitDiscovery(
                self.model,
                threshold=kwargs.get('threshold', 0.01),
                device=self.device
            )

            # Need targets for ACDC
            targets = kwargs.get('targets', self.model(inputs))

            circuit = acdc.discover_circuit(inputs, targets, max_iterations=kwargs.get('max_iters', 100))

            # Convert to CircuitResult
            edges = [(e.source, e.target, e.importance) for e in circuit.edges]
            nodes = list(circuit.nodes)

            return CircuitResult(
                method="ACDC",
                data={'circuit': circuit},
                metadata={'threshold': kwargs.get('threshold', 0.01)},
                metrics={'sparsity': circuit.sparsity, 'performance': circuit.performance},
                nodes=nodes,
                edges=edges,
                circuit_type='minimal'
            )

        # Information Flow
        def run_info_flow(inputs: torch.Tensor, **kwargs) -> InformationResult:
            from neuros_mechint.energy_flow import InformationFlowAnalyzer

            analyzer = InformationFlowAnalyzer(device=self.device, verbose=False)

            # Get activations from all layers
            activations_dict = self._get_all_activations(inputs)

            # Get targets
            targets = kwargs.get('targets', self.model(inputs))

            # Compute information plane
            method = 'knn' if self.config.depth == 'quick' else 'mine'
            info_plane = analyzer.information_plane(
                activations_dict,
                inputs.flatten(1),
                targets.flatten(1),
                method=method
            )

            return InformationResult(
                method="InformationFlow",
                data={
                    'I_XZ': info_plane.I_XZ_per_layer,
                    'I_ZY': info_plane.I_ZY_per_layer,
                    'layers': info_plane.layers
                },
                metadata={'method': method},
                metrics={
                    'compression': float(info_plane.I_XZ_per_layer.mean()),
                    'prediction': float(info_plane.I_ZY_per_layer.mean())
                },
                information_flow=info_plane.I_ZY_per_layer - info_plane.I_XZ_per_layer
            )

        # Dynamics Analysis
        def run_dynamics(inputs: torch.Tensor, **kwargs) -> DynamicsResult:
            from neuros_mechint.dynamics import DynamicsAnalyzer

            analyzer = DynamicsAnalyzer(self.model, device=self.device)

            # Collect trajectories
            trajectories = self._collect_trajectories(inputs, n_steps=kwargs.get('n_steps', 100))

            # Compute Lyapunov exponents
            lyapunov = analyzer.compute_lyapunov_spectrum(trajectories)

            # Find fixed points
            fixed_points = analyzer.find_fixed_points(
                n_points=kwargs.get('n_fixed_points', 10),
                tol=kwargs.get('fixed_point_tol', 1e-4)
            )

            return DynamicsResult(
                method="Dynamics",
                data={'trajectories': trajectories.cpu().numpy()},
                metadata={'n_steps': kwargs.get('n_steps', 100)},
                metrics={
                    'max_lyapunov': float(lyapunov.max()) if lyapunov is not None else 0.0,
                    'n_fixed_points': len(fixed_points)
                },
                trajectories=trajectories.cpu().numpy(),
                lyapunov_exponents=lyapunov.cpu().numpy() if lyapunov is not None else None,
                fixed_points=[fp.cpu().numpy() for fp in fixed_points]
            )

        # Fractal Analysis
        def run_fractals(inputs: torch.Tensor, **kwargs) -> FractalResult:
            from neuros_mechint.fractals import FractalMetricsBundle, HiguchiFractalDimension

            # Get activations
            activations = self._get_activations(inputs, kwargs.get('layer_name', 'layer_3'))

            # Compute temporal fractal dimension
            higuchi = HiguchiFractalDimension(k_max=kwargs.get('k_max', 10))
            fd_values = []

            for sample in activations:
                fd = higuchi.compute(sample.cpu().numpy())
                fd_values.append(fd)

            avg_fd = sum(fd_values) / len(fd_values)

            return FractalResult(
                method="Fractals",
                data={'fd_per_sample': fd_values},
                metadata={'k_max': kwargs.get('k_max', 10), 'layer': kwargs.get('layer_name', 'layer_3')},
                metrics={'mean_fd': float(avg_fd), 'std_fd': float(torch.tensor(fd_values).std())},
                fractal_dimension=avg_fd
            )

        # Register stages
        if "sae" in self.config.enabled_analyses:
            self.register_stage("sae", run_sae, priority=1)

        if "circuits" in self.config.enabled_analyses:
            self.register_stage("circuits", run_circuits, priority=2)

        if "info_flow" in self.config.enabled_analyses:
            self.register_stage("info_flow", run_info_flow, priority=1)

        if "dynamics" in self.config.enabled_analyses:
            self.register_stage("dynamics", run_dynamics, dependencies=[], priority=3)

        if "fractals" in self.config.enabled_analyses:
            self.register_stage("fractals", run_fractals, priority=1)

    def register_stage(
        self,
        name: str,
        analysis_fn: Callable,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        result_type: type = MechIntResult
    ):
        """
        Register a custom analysis stage.

        Args:
            name: Unique stage name
            analysis_fn: Function that takes (inputs, **kwargs) and returns MechIntResult
            dependencies: List of stage names that must run first
            priority: Execution priority (higher runs first)
            result_type: Expected result type
        """
        self.stages[name] = AnalysisStage(
            name=name,
            analysis_fn=analysis_fn,
            dependencies=dependencies or [],
            enabled=True,
            priority=priority,
            result_type=result_type
        )
        self._log(f"Registered analysis stage: {name}")

    def run(
        self,
        inputs: torch.Tensor,
        analyses: Optional[List[str]] = None,
        generate_report: bool = False,
        **kwargs
    ) -> ResultCollection:
        """
        Run specified analyses.

        Args:
            inputs: Input data
            analyses: List of analysis names to run (None = all enabled)
            generate_report: Whether to generate HTML report
            **kwargs: Additional arguments passed to analysis functions

        Returns:
            ResultCollection with all results
        """
        self._log(f"Starting pipeline run with depth={self.config.depth}")

        # Determine which analyses to run
        if analyses is None:
            to_run = [name for name, stage in self.stages.items() if stage.enabled]
        else:
            to_run = [name for name in analyses if name in self.stages]

        # Sort by priority and dependencies
        to_run = self._resolve_execution_order(to_run)

        self._log(f"Running {len(to_run)} analyses: {to_run}")

        # Create checkpoint
        self.checkpoint_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Run analyses
        results = []

        if self.config.parallel and len(to_run) > 1:
            # Parallel execution (for independent analyses)
            results = self._run_parallel(inputs, to_run, kwargs)
        else:
            # Sequential execution
            results = self._run_sequential(inputs, to_run, kwargs)

        # Create collection
        collection = ResultCollection(
            results=results,
            name=f"pipeline_{self.checkpoint_id}"
        )

        # Store in database if available
        if self.db:
            for i, result in enumerate(results):
                result_id = self.db.store(
                    result,
                    tags=[to_run[i], f"checkpoint_{self.checkpoint_id}"]
                )
                self._log(f"Stored {to_run[i]} result: {result_id}")

        # Generate report if requested
        if generate_report or self.config.auto_report:
            self._generate_report(collection)

        self._log("Pipeline run complete!")
        return collection

    def run_all(
        self,
        inputs: torch.Tensor,
        generate_report: bool = True,
        **kwargs
    ) -> ResultCollection:
        """
        Run all enabled analyses.

        Args:
            inputs: Input data
            generate_report: Whether to generate report
            **kwargs: Additional arguments

        Returns:
            ResultCollection with all results
        """
        return self.run(inputs, analyses=None, generate_report=generate_report, **kwargs)

    def resume(self, checkpoint_id: str) -> ResultCollection:
        """
        Resume from a previous checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to resume from

        Returns:
            ResultCollection of cached results
        """
        if not self.db:
            raise ValueError("Database required for resume functionality")

        self._log(f"Resuming from checkpoint: {checkpoint_id}")

        # Query results by checkpoint tag
        result_ids = self.db.query(tags=[f"checkpoint_{checkpoint_id}"])

        if not result_ids:
            raise ValueError(f"No results found for checkpoint: {checkpoint_id}")

        # Load results
        results = list(self.db.batch_get(result_ids).values())

        collection = ResultCollection(
            results=results,
            name=f"resumed_{checkpoint_id}"
        )

        self._log(f"Resumed {len(results)} results")
        return collection

    def _run_sequential(
        self,
        inputs: torch.Tensor,
        analyses: List[str],
        kwargs: Dict[str, Any]
    ) -> List[MechIntResult]:
        """Run analyses sequentially."""
        results = []

        iterator = tqdm(analyses, desc="Running analyses") if self.config.show_progress else analyses

        for name in iterator:
            try:
                result = self._run_single_analysis(name, inputs, kwargs)
                results.append(result)
                self.results[name] = result
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
                if not self.config.continue_on_error:
                    raise

        return results

    def _run_parallel(
        self,
        inputs: torch.Tensor,
        analyses: List[str],
        kwargs: Dict[str, Any]
    ) -> List[MechIntResult]:
        """Run independent analyses in parallel."""
        results = []

        # Separate independent and dependent analyses
        independent = [name for name in analyses if not self.stages[name].dependencies]
        dependent = [name for name in analyses if self.stages[name].dependencies]

        # Run independent analyses in parallel
        if independent:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_name = {
                    executor.submit(self._run_single_analysis, name, inputs, kwargs): name
                    for name in independent
                }

                for future in tqdm(as_completed(future_to_name), total=len(independent),
                                  desc="Parallel analyses", disable=not self.config.show_progress):
                    name = future_to_name[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.results[name] = result
                    except Exception as e:
                        logger.error(f"Error in {name}: {e}")
                        if not self.config.continue_on_error:
                            raise

        # Run dependent analyses sequentially
        for name in dependent:
            try:
                result = self._run_single_analysis(name, inputs, kwargs)
                results.append(result)
                self.results[name] = result
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
                if not self.config.continue_on_error:
                    raise

        return results

    def _run_single_analysis(
        self,
        name: str,
        inputs: torch.Tensor,
        kwargs: Dict[str, Any]
    ) -> MechIntResult:
        """Run a single analysis stage."""
        stage = self.stages[name]

        self._log(f"Running {name}...")

        # Check cache if database available
        if self.db and self.config.use_cache:
            # Try to find cached result
            recent = self.db.query(method=name, limit=1)
            if recent:
                cached = self.db.get(recent[0])
                if cached:
                    self._log(f"Using cached result for {name}")
                    return cached

        # Run analysis
        result = stage.analysis_fn(inputs, **kwargs)

        # Add provenance
        if stage.dependencies:
            result.provenance = [self.results[dep] for dep in stage.dependencies if dep in self.results]

        return result

    def _resolve_execution_order(self, analyses: List[str]) -> List[str]:
        """Resolve execution order based on dependencies and priorities."""
        # Topological sort with priority
        ordered = []
        remaining = set(analyses)
        resolved = set()

        while remaining:
            # Find analyses with all dependencies satisfied
            ready = [
                name for name in remaining
                if all(dep in resolved or dep not in analyses
                      for dep in self.stages[name].dependencies)
            ]

            if not ready:
                # Circular dependency or missing dependency
                raise ValueError(f"Cannot resolve dependencies for: {remaining}")

            # Sort ready analyses by priority
            ready.sort(key=lambda x: self.stages[x].priority, reverse=True)

            # Add highest priority
            next_analysis = ready[0]
            ordered.append(next_analysis)
            remaining.remove(next_analysis)
            resolved.add(next_analysis)

        return ordered

    def _get_activations(self, inputs: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Get activations from a specific layer."""
        activations = None

        def hook(module, input, output):
            nonlocal activations
            activations = output.detach()

        # Register hook
        layer = dict(self.model.named_modules())[layer_name]
        handle = layer.register_forward_hook(hook)

        # Forward pass
        with torch.no_grad():
            self.model(inputs.to(self.device))

        handle.remove()

        return activations

    def _get_all_activations(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get activations from all layers."""
        activations = {}

        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        # Register hooks
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):
                handles.append(module.register_forward_hook(make_hook(name)))

        # Forward pass
        with torch.no_grad():
            self.model(inputs.to(self.device))

        # Remove hooks
        for handle in handles:
            handle.remove()

        return activations

    def _collect_trajectories(self, inputs: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
        """Collect activation trajectories over time."""
        # This is a simplified version - would need proper RNN handling
        trajectories = []

        for step in range(n_steps):
            with torch.no_grad():
                output = self.model(inputs.to(self.device))
                trajectories.append(output.flatten(1))

        return torch.stack(trajectories, dim=1)  # (batch, n_steps, dim)

    def _generate_report(self, collection: ResultCollection):
        """Generate analysis report."""
        try:
            from neuros_mechint.reporting import MechIntReporter

            reporter = MechIntReporter()
            report_path = self.config.cache_dir or "."
            report_file = Path(report_path) / f"report_{self.checkpoint_id}.html"

            reporter.generate_report(
                collection,
                output_path=str(report_file),
                format=self.config.report_format
            )

            self._log(f"Generated report: {report_file}")
        except ImportError:
            logger.warning("Reporting module not available")


__all__ = [
    'PipelineConfig',
    'AnalysisStage',
    'MechIntPipeline',
]
