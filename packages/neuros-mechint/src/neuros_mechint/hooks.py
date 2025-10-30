"""
Training and Evaluation Hooks for Automatic Mechanistic Interpretability Integration

Provides seamless integration between NeuroFMX training and mechanistic interpretability:
- Automatic activation sampling during training
- Hook-based architecture for minimal overhead
- PyTorch Lightning callback integration
- FastAPI endpoints for on-demand interpretation
- Unified evaluation runner for comprehensive analysis
- S3/cloud storage support for large-scale experiments

Features:
- Zero-copy activation capture via hooks
- Configurable sampling frequencies
- Automatic storage management
- Multi-analysis orchestration
- Real-time interpretation via API
"""

import os
import json
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pl = None
    def rank_zero_only(fn):
        return fn

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    JSONResponse = None
    BaseModel = object
    Field = lambda *args, **kwargs: None

# Optional S3 support
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None


@dataclass
class MechIntConfig:
    """Configuration for mechanistic interpretability hooks.

    Attributes
    ----------
    sample_layers : List[str]
        Layer names to track activations from.
    save_hidden_every_n_steps : int
        Sampling frequency for saving activations.
    analyses_to_run : List[str]
        List of analyses to run: ['sae', 'circuit', 'neuron', 'causal', 'feature'].
    storage_backend : str
        Storage backend: 'local', 's3', or 'both'.
    storage_path : str
        Local path or S3 bucket for storing activations.
    s3_bucket : Optional[str]
        S3 bucket name if using S3 backend.
    s3_prefix : str
        S3 key prefix.
    max_activations_per_shard : int
        Maximum number of samples per activation shard.
    enable_feature_steering : bool
        Whether to enable feature steering experiments.
    verbose : bool
        Enable verbose logging.
    """
    sample_layers: List[str] = None
    save_hidden_every_n_steps: int = 200
    analyses_to_run: List[str] = None
    storage_backend: str = 'local'
    storage_path: str = './mechint_cache'
    s3_bucket: Optional[str] = None
    s3_prefix: str = 'neurofmx/activations'
    max_activations_per_shard: int = 10000
    enable_feature_steering: bool = False
    verbose: bool = True

    def __post_init__(self):
        if self.sample_layers is None:
            self.sample_layers = ['mamba_backbone.blocks.3', 'popt']
        if self.analyses_to_run is None:
            self.analyses_to_run = ['sae', 'neuron', 'feature']


class ActivationSampler:
    """Captures and saves model activations during training.

    Uses PyTorch hooks to efficiently capture activations with minimal overhead.
    Supports sharding for large-scale experiments.

    Parameters
    ----------
    layers : List[str]
        Layer names to hook.
    save_dir : str
        Directory to save activations.
    max_samples_per_shard : int, optional
        Maximum samples per shard file.
        Default: 10000.
    device : str, optional
        Device for storing activations before saving.
        Default: 'cpu'.

    Examples
    --------
    >>> sampler = ActivationSampler(layers=['layer1', 'layer2'], save_dir='./cache')
    >>> sampler.register_hooks(model)
    >>> # Training loop...
    >>> sampler.save_activations(global_step=100)
    >>> sampler.clear_cache()
    """

    def __init__(
        self,
        layers: List[str],
        save_dir: str,
        max_samples_per_shard: int = 10000,
        device: str = 'cpu'
    ):
        self.layers = layers
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples_per_shard = max_samples_per_shard
        self.device = device

        # Storage
        self.activations = defaultdict(list)
        self.metadata = defaultdict(list)
        self.hooks = []
        self.sample_count = 0
        self.shard_count = 0

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on specified layers.

        Parameters
        ----------
        model : nn.Module
            Model to attach hooks to.
        """
        self.clear_hooks()

        def get_activation_hook(layer_name: str):
            def hook(module, input, output):
                # Handle different output types
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                elif isinstance(output, tuple):
                    act = output[0].detach()
                elif isinstance(output, dict):
                    # Handle dictionary outputs (e.g., from Mamba blocks)
                    act = output.get('hidden_states', output.get('latents', output.get('output')))
                    if act is None:
                        return
                    act = act.detach()
                else:
                    return

                # Move to CPU to avoid GPU memory issues
                act = act.cpu()

                # Store activation
                self.activations[layer_name].append(act)

            return hook

        # Register hooks
        registered = []
        for name, module in model.named_modules():
            if name in self.layers:
                handle = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(handle)
                registered.append(name)

        if len(registered) < len(self.layers):
            missing = set(self.layers) - set(registered)
            warnings.warn(f"Could not register hooks for layers: {missing}")

        return registered

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def save_activations(
        self,
        global_step: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save collected activations to disk.

        Parameters
        ----------
        global_step : int
            Current training step.
        metadata : Dict[str, Any], optional
            Additional metadata to save.

        Returns
        -------
        str
            Path to saved shard file.
        """
        if not self.activations:
            return None

        # Prepare save path
        shard_path = self.save_dir / f'activations_shard_{self.shard_count:06d}_step_{global_step}.pt'

        # Concatenate activations
        save_data = {
            'global_step': global_step,
            'shard_id': self.shard_count,
            'activations': {},
            'metadata': metadata or {}
        }

        for layer_name, acts in self.activations.items():
            if acts:
                # Concatenate all batches
                concatenated = torch.cat(acts, dim=0)
                save_data['activations'][layer_name] = concatenated
                save_data['metadata'][f'{layer_name}_shape'] = list(concatenated.shape)

        # Save to disk
        torch.save(save_data, shard_path)

        # Update counters
        self.sample_count += sum(act.shape[0] for acts in self.activations.values() for act in acts)
        self.shard_count += 1

        return str(shard_path)

    def clear_cache(self) -> None:
        """Clear activation cache."""
        self.activations.clear()
        self.metadata.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about captured activations.

        Returns
        -------
        dict
            Statistics including sample counts, shard counts, etc.
        """
        current_samples = sum(
            sum(act.shape[0] for act in acts)
            for acts in self.activations.values()
        )

        return {
            'total_samples_saved': self.sample_count,
            'current_cache_samples': current_samples,
            'total_shards': self.shard_count,
            'tracked_layers': list(self.activations.keys())
        }


class MechIntHooks:
    """Core hook manager for mechanistic interpretability.

    Orchestrates activation sampling, storage, and analysis triggering
    during training.

    Parameters
    ----------
    config : Union[MechIntConfig, Dict]
        Configuration for mechanistic interpretability.

    Examples
    --------
    >>> config = MechIntConfig(
    ...     sample_layers=['layer1', 'layer2'],
    ...     save_hidden_every_n_steps=100,
    ...     analyses_to_run=['sae', 'neuron']
    ... )
    >>> hooks = MechIntHooks(config)
    >>> hooks.register_hooks(model, trainer)
    """

    def __init__(self, config: Union[MechIntConfig, Dict]):
        if isinstance(config, dict):
            config = MechIntConfig(**config)

        self.config = config
        self.sampler = None
        self.s3_client = None

        # Initialize storage
        if self.config.storage_backend in ['s3', 'both']:
            if not S3_AVAILABLE:
                warnings.warn("boto3 not available, falling back to local storage")
                self.config.storage_backend = 'local'
            else:
                self.s3_client = boto3.client('s3')

        # Analysis history
        self.analysis_history = []
        self.saved_shards = []

    def register_hooks(
        self,
        model: nn.Module,
        trainer: Optional[Any] = None
    ) -> None:
        """Register forward hooks on model layers.

        Parameters
        ----------
        model : nn.Module
            Model to attach hooks to.
        trainer : optional
            Trainer instance (for accessing metadata).
        """
        self.sampler = ActivationSampler(
            layers=self.config.sample_layers,
            save_dir=self.config.storage_path,
            max_samples_per_shard=self.config.max_activations_per_shard
        )

        registered = self.sampler.register_hooks(model)

        if self.config.verbose:
            print(f"[MechInt] Registered hooks on {len(registered)} layers: {registered}")

    def on_training_step(
        self,
        trainer: Any,
        pl_module: nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        global_step: int
    ) -> None:
        """Called after each training step.

        Parameters
        ----------
        trainer : Any
            Trainer instance.
        pl_module : nn.Module
            Lightning module.
        outputs : Any
            Training step outputs.
        batch : Any
            Current batch.
        batch_idx : int
            Batch index.
        global_step : int
            Global training step.
        """
        if self.sampler is None:
            return

        # Check if we should save
        if global_step % self.config.save_hidden_every_n_steps == 0:
            # Save activations
            metadata = {
                'global_step': global_step,
                'epoch': trainer.current_epoch if hasattr(trainer, 'current_epoch') else None,
                'batch_idx': batch_idx
            }

            shard_path = self.sampler.save_activations(
                global_step=global_step,
                metadata=metadata
            )

            if shard_path:
                self.saved_shards.append(shard_path)

                # Upload to S3 if needed
                if self.config.storage_backend in ['s3', 'both']:
                    self._upload_to_s3(shard_path)

                if self.config.verbose:
                    stats = self.sampler.get_statistics()
                    print(f"[MechInt] Saved activations at step {global_step}: {shard_path}")
                    print(f"[MechInt] Stats: {stats}")

                # Clear cache
                self.sampler.clear_cache()

    def on_epoch_end(
        self,
        trainer: Any,
        pl_module: nn.Module
    ) -> None:
        """Called at the end of each epoch.

        Optionally runs lightweight analyses.

        Parameters
        ----------
        trainer : Any
            Trainer instance.
        pl_module : nn.Module
            Lightning module.
        """
        if self.config.verbose:
            epoch = trainer.current_epoch if hasattr(trainer, 'current_epoch') else 'unknown'
            print(f"[MechInt] Epoch {epoch} complete. Saved {len(self.saved_shards)} shards.")

    def on_train_end(
        self,
        trainer: Any,
        pl_module: nn.Module
    ) -> None:
        """Called at the end of training.

        Triggers final comprehensive analysis.

        Parameters
        ----------
        trainer : Any
            Trainer instance.
        pl_module : nn.Module
            Lightning module.
        """
        if self.config.verbose:
            print(f"[MechInt] Training complete. Running final analyses...")

        # Save manifest of all shards
        manifest = {
            'config': asdict(self.config),
            'shards': self.saved_shards,
            'total_shards': len(self.saved_shards),
            'analysis_history': self.analysis_history
        }

        manifest_path = Path(self.config.storage_path) / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        if self.config.verbose:
            print(f"[MechInt] Saved manifest to {manifest_path}")

        # Clean up hooks
        if self.sampler:
            self.sampler.clear_hooks()

    def _upload_to_s3(self, local_path: str) -> Optional[str]:
        """Upload activation shard to S3.

        Parameters
        ----------
        local_path : str
            Local file path.

        Returns
        -------
        Optional[str]
            S3 URI if successful.
        """
        if self.s3_client is None or self.config.s3_bucket is None:
            return None

        try:
            file_name = Path(local_path).name
            s3_key = f"{self.config.s3_prefix}/{file_name}"

            self.s3_client.upload_file(
                local_path,
                self.config.s3_bucket,
                s3_key
            )

            s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"
            if self.config.verbose:
                print(f"[MechInt] Uploaded to {s3_uri}")

            return s3_uri

        except Exception as e:
            warnings.warn(f"Failed to upload to S3: {e}")
            return None


class EvalMechIntRunner:
    """Evaluation-time mechanistic interpretability runner.

    Loads pre-saved activations and runs comprehensive analysis suite.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    config : Union[MechIntConfig, Dict]
        Configuration.
    device : str, optional
        Device for analysis.
        Default: 'cuda'.

    Examples
    --------
    >>> runner = EvalMechIntRunner(model, config)
    >>> results = runner.run_mechint_eval(
    ...     eval_data=dataloader,
    ...     checkpoint_path='./checkpoints/model.pt',
    ...     hidden_shards_path='./mechint_cache'
    ... )
    >>> runner.export_results('./results')
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[MechIntConfig, Dict],
        device: str = 'cuda'
    ):
        if isinstance(config, dict):
            config = MechIntConfig(**config)

        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

        # Results storage
        self.results = {}

    def run_mechint_eval(
        self,
        eval_data: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        hidden_shards_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run full mechanistic interpretability evaluation suite.

        Parameters
        ----------
        eval_data : optional
            Evaluation dataloader.
        checkpoint_path : str, optional
            Path to model checkpoint.
        hidden_shards_path : str, optional
            Path to activation shards.

        Returns
        -------
        dict
            Comprehensive analysis results.
        """
        results = {
            'config': asdict(self.config),
            'analyses': {}
        }

        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Load activation shards
        activations = None
        if hidden_shards_path:
            activations = self._load_activations(hidden_shards_path)

        # Run analyses
        for analysis_name in self.config.analyses_to_run:
            print(f"[MechInt] Running {analysis_name} analysis...")

            try:
                if analysis_name == 'sae':
                    results['analyses']['sae'] = self._run_sae_analysis(activations)
                elif analysis_name == 'neuron':
                    results['analyses']['neuron'] = self._run_neuron_analysis(activations)
                elif analysis_name == 'circuit':
                    results['analyses']['circuit'] = self._run_circuit_analysis(activations)
                elif analysis_name == 'causal':
                    results['analyses']['causal'] = self._run_causal_analysis(activations, eval_data)
                elif analysis_name == 'feature':
                    results['analyses']['feature'] = self._run_feature_analysis(activations)
                else:
                    warnings.warn(f"Unknown analysis: {analysis_name}")

            except Exception as e:
                warnings.warn(f"Failed to run {analysis_name} analysis: {e}")
                results['analyses'][analysis_name] = {'error': str(e)}

        self.results = results
        return results

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"[MechInt] Loaded checkpoint from {checkpoint_path}")

    def _load_activations(self, shards_path: str) -> Dict[str, torch.Tensor]:
        """Load activations from shard files.

        Returns
        -------
        dict
            Dictionary mapping layer names to stacked activations.
        """
        shards_path = Path(shards_path)
        shard_files = sorted(shards_path.glob('activations_shard_*.pt'))

        if not shard_files:
            warnings.warn(f"No activation shards found in {shards_path}")
            return {}

        print(f"[MechInt] Loading {len(shard_files)} activation shards...")

        # Load and concatenate
        activations_by_layer = defaultdict(list)

        for shard_file in shard_files:
            shard_data = torch.load(shard_file, map_location='cpu')

            for layer_name, acts in shard_data['activations'].items():
                activations_by_layer[layer_name].append(acts)

        # Concatenate all shards
        result = {}
        for layer_name, acts_list in activations_by_layer.items():
            result[layer_name] = torch.cat(acts_list, dim=0)
            print(f"[MechInt] Loaded {layer_name}: {result[layer_name].shape}")

        return result

    def _run_sae_analysis(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run sparse autoencoder analysis."""
        from neuros_mechint.sae_training import MultiLayerSAETrainer

        results = {}

        for layer_name, acts in activations.items():
            print(f"  Training SAE on {layer_name}...")

            # Create simple trainer
            trainer = MultiLayerSAETrainer(
                model=self.model,
                layer_names=[layer_name],
                device=self.device
            )

            # Use cached activations directly
            trainer.activation_cache.activations[layer_name] = [acts]
            trainer.initialize_saes()

            # Quick training
            stats = trainer.train(num_epochs=50, batch_size=256)

            results[layer_name] = {
                'training_stats': stats,
                'final_l0': stats[f'{layer_name}/l0_sparsity'][-1],
                'final_loss': stats[f'{layer_name}/loss'][-1]
            }

        return results

    def _run_neuron_analysis(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run neuron activation analysis."""
        from neuros_mechint.neuron_analysis import NeuronActivationAnalyzer

        results = {}

        for layer_name, acts in activations.items():
            print(f"  Analyzing neurons in {layer_name}...")

            # Flatten if needed
            if len(acts.shape) > 2:
                acts = acts.reshape(-1, acts.shape[-1])

            analyzer = NeuronActivationAnalyzer()

            # Compute statistics
            stats = {
                'mean_activation': acts.mean(dim=0).cpu().numpy().tolist(),
                'max_activation': acts.max(dim=0).values.cpu().numpy().tolist(),
                'sparsity': (acts > 0).float().mean(dim=0).cpu().numpy().tolist()
            }

            results[layer_name] = stats

        return results

    def _run_circuit_analysis(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run circuit discovery analysis."""
        from neuros_mechint.circuit_discovery import CircuitDiscovery

        # Placeholder for circuit analysis
        results = {
            'message': 'Circuit analysis requires full model and interventions'
        }

        return results

    def _run_causal_analysis(
        self,
        activations: Dict[str, torch.Tensor],
        eval_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Run causal importance analysis."""
        # Placeholder for causal analysis
        results = {
            'message': 'Causal analysis requires evaluation data and ablation experiments'
        }

        return results

    def _run_feature_analysis(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run feature analysis."""
        from neuros_mechint.feature_analysis import FeatureClusteringAnalyzer

        results = {}

        for layer_name, acts in activations.items():
            print(f"  Analyzing features in {layer_name}...")

            # Flatten if needed
            if len(acts.shape) > 2:
                acts = acts.reshape(-1, acts.shape[-1])

            # Sample for efficiency
            if acts.shape[0] > 10000:
                indices = torch.randperm(acts.shape[0])[:10000]
                acts = acts[indices]

            # Simple clustering
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(50, acts.shape[1]))
            pca_features = pca.fit_transform(acts.cpu().numpy())

            results[layer_name] = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'n_components': pca.n_components_
            }

        return results

    def export_results(self, output_dir: str) -> None:
        """Export all analysis results.

        Parameters
        ----------
        output_dir : str
            Output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        results_path = output_dir / 'mechint_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"[MechInt] Exported results to {results_path}")

        # Generate summary report
        self._generate_report(output_dir)

    def _generate_report(self, output_dir: Path) -> None:
        """Generate human-readable report."""
        report_path = output_dir / 'mechint_report.md'

        with open(report_path, 'w') as f:
            f.write("# Mechanistic Interpretability Analysis Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Configuration\n\n")
            f.write(f"- Sampled layers: {self.config.sample_layers}\n")
            f.write(f"- Analyses run: {self.config.analyses_to_run}\n\n")

            f.write("## Results Summary\n\n")

            for analysis_name, analysis_results in self.results.get('analyses', {}).items():
                f.write(f"### {analysis_name.upper()} Analysis\n\n")
                f.write(f"```json\n{json.dumps(analysis_results, indent=2)}\n```\n\n")

        print(f"[MechInt] Generated report at {report_path}")


if LIGHTNING_AVAILABLE:
    class MechIntCallback(pl.Callback):
        """PyTorch Lightning callback for mechanistic interpretability.

        Automatically integrates with PyTorch Lightning training loop.

        Parameters
        ----------
        config : Union[MechIntConfig, Dict]
            Configuration for mechanistic interpretability.

        Examples
        --------
        >>> from neuros_mechint import MechIntCallback
        >>>
        >>> # Add to trainer
        >>> trainer = pl.Trainer(
        ...     callbacks=[
        ...         MechIntCallback(config={
        ...             'save_hidden_every_n_steps': 200,
        ...             'sample_layers': ['mamba_backbone.blocks.3', 'popt']
        ...         })
        ...     ]
        ... )
        >>>
        >>> trainer.fit(model, dataloader)
        """

        def __init__(self, config: Union[MechIntConfig, Dict]):
            super().__init__()
            self.hooks = MechIntHooks(config)

        def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            """Called when training starts."""
            self.hooks.register_hooks(pl_module.model, trainer)

        def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int
        ) -> None:
            """Called after each training batch."""
            self.hooks.on_training_step(
                trainer=trainer,
                pl_module=pl_module,
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
                global_step=trainer.global_step
            )

        def on_train_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule
        ) -> None:
            """Called at the end of training epoch."""
            self.hooks.on_epoch_end(trainer, pl_module)

        def on_train_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule
        ) -> None:
            """Called when training ends."""
            self.hooks.on_train_end(trainer, pl_module)


if FASTAPI_AVAILABLE:
    # FastAPI integration models
    class InterpretRequest(BaseModel):
        """Request schema for interpretation endpoint."""
        analysis_type: str = Field(..., description="Type of analysis: 'sae', 'neuron', 'circuit', 'causal', 'feature'")
        layer_name: str = Field(..., description="Layer name to analyze")
        config: Dict[str, Any] = Field(default_factory=dict, description="Analysis-specific configuration")


    class InterpretResponse(BaseModel):
        """Response schema for interpretation endpoint."""
        analysis_type: str
        layer_name: str
        results: Dict[str, Any]
        timestamp: float


    class FastAPIIntegrationMixin:
        """Mixin for adding mechanistic interpretability endpoints to FastAPI.

        Usage
        -----
        Add to your FastAPI server:

        >>> from neuros_neurofm.api.server import create_app
        >>> from neuros_mechint.hooks import FastAPIIntegrationMixin
        >>>
        >>> app = create_app(model_path='./model.pt')
        >>>
        >>> # Add interpretation endpoints
        >>> mixin = FastAPIIntegrationMixin(model=model, config=mechint_config)
        >>> mixin.add_routes(app)
        """

        def __init__(
            self,
            model: nn.Module,
            config: Union[MechIntConfig, Dict],
            device: str = 'cuda'
        ):
            if isinstance(config, dict):
                config = MechIntConfig(**config)

            self.model = model.to(device)
            self.model.eval()
            self.config = config
            self.device = device

            # Initialize runner
            self.runner = EvalMechIntRunner(
                model=model,
                config=config,
                device=device
            )

        def add_routes(self, app: FastAPI) -> None:
            """Add interpretation routes to FastAPI app.

            Parameters
            ----------
            app : FastAPI
                FastAPI application instance.
            """

            @app.post("/interpret", response_model=InterpretResponse)
            async def interpret(request: InterpretRequest):
                """Run interpretation analysis.

                Accepts uploaded activations or uses cached activations.
                """
                try:
                    # Load activations if available
                    activations = self.runner._load_activations(self.config.storage_path)

                    if request.layer_name not in activations:
                        raise HTTPException(
                            status_code=404,
                            detail=f"No activations found for layer: {request.layer_name}"
                        )

                    layer_acts = {request.layer_name: activations[request.layer_name]}

                    # Run requested analysis
                    if request.analysis_type == 'sae':
                        results = self.runner._run_sae_analysis(layer_acts)
                    elif request.analysis_type == 'neuron':
                        results = self.runner._run_neuron_analysis(layer_acts)
                    elif request.analysis_type == 'circuit':
                        results = self.runner._run_circuit_analysis(layer_acts)
                    elif request.analysis_type == 'feature':
                        results = self.runner._run_feature_analysis(layer_acts)
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unknown analysis type: {request.analysis_type}"
                        )

                    return InterpretResponse(
                        analysis_type=request.analysis_type,
                        layer_name=request.layer_name,
                        results=results,
                        timestamp=time.time()
                    )

                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/interpret/upload")
            async def interpret_upload(
                file: UploadFile = File(...),
                analysis_type: str = 'neuron'
            ):
                """Accept uploaded activations and run analysis."""
                try:
                    # Save uploaded file
                    upload_path = Path(self.config.storage_path) / 'uploads' / file.filename
                    upload_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(upload_path, 'wb') as f:
                        content = await file.read()
                        f.write(content)

                    # Load activations
                    data = torch.load(upload_path, map_location='cpu')
                    activations = data.get('activations', data)

                    # Run analysis
                    if analysis_type == 'neuron':
                        results = self.runner._run_neuron_analysis(activations)
                    elif analysis_type == 'feature':
                        results = self.runner._run_feature_analysis(activations)
                    else:
                        results = {'message': f'Analysis type {analysis_type} not supported for uploads'}

                    return JSONResponse(content={
                        'analysis_type': analysis_type,
                        'results': results,
                        'timestamp': time.time()
                    })

                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/interpret/layers")
            async def get_available_layers():
                """Get list of available layers with cached activations."""
                try:
                    activations = self.runner._load_activations(self.config.storage_path)
                    return {
                        'layers': list(activations.keys()),
                        'shapes': {k: list(v.shape) for k, v in activations.items()}
                    }
                except Exception as e:
                    return {'layers': [], 'error': str(e)}


# Convenience exports
__all__ = [
    'MechIntConfig',
    'ActivationSampler',
    'MechIntHooks',
    'EvalMechIntRunner',
    'MechIntCallback',
    'FastAPIIntegrationMixin'
]
