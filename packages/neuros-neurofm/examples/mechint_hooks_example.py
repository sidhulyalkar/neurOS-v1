"""
Examples for Mechanistic Interpretability Hooks

Demonstrates how to integrate automatic mech-int into training workflows.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Example 1: PyTorch Lightning Integration
# ============================================================================

def example_lightning_integration():
    """Example of using MechIntCallback with PyTorch Lightning."""

    try:
        import pytorch_lightning as pl
        from neuros_mechint import MechIntCallback, MechIntConfig
        from neuros_neurofm.training.lightning_module import NeuroFMXLightningModule
        from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX

        print("=" * 80)
        print("Example 1: PyTorch Lightning Integration")
        print("=" * 80)

        # Create model
        model = MultiModalNeuroFMX(
            d_model=512,
            n_mamba_blocks=4,
            n_latents=64,
            latent_dim=512
        )

        # Wrap in Lightning module
        pl_module = NeuroFMXLightningModule(
            model=model,
            learning_rate=3e-4,
            max_epochs=100
        )

        # Configure mechanistic interpretability
        mechint_config = MechIntConfig(
            sample_layers=['mamba_backbone.blocks.3', 'popt'],
            save_hidden_every_n_steps=200,
            analyses_to_run=['sae', 'neuron', 'feature'],
            storage_backend='local',
            storage_path='./mechint_cache',
            max_activations_per_shard=10000,
            verbose=True
        )

        # Create callback
        mechint_callback = MechIntCallback(config=mechint_config)

        # Create trainer with callback
        trainer = pl.Trainer(
            max_epochs=10,
            callbacks=[mechint_callback],
            accelerator='auto',
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=10
        )

        # Create dummy dataloader
        dummy_data = {
            'tokens': torch.randn(100, 64, 512),
            'attention_mask': torch.ones(100, 64),
            'behavior': torch.randn(100, 32),
            'neural': torch.randn(100, 96)
        }
        dataset = TensorDataset(*dummy_data.values())
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Train (hooks will automatically capture activations)
        print("\nStarting training with automatic mech-int integration...")
        # trainer.fit(pl_module, dataloader)  # Uncomment to actually train

        print("\nMechInt will automatically:")
        print("  - Capture activations every 200 steps")
        print("  - Save to ./mechint_cache/")
        print("  - Run analyses at epoch end")
        print("  - Generate final report at training end")

    except ImportError as e:
        print(f"Skipping Lightning example: {e}")


# ============================================================================
# Example 2: Manual Hook Integration (without Lightning)
# ============================================================================

def example_manual_hook_integration():
    """Example of using MechIntHooks directly in custom training loop."""

    from neuros_mechint import MechIntHooks, MechIntConfig
    from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX

    print("\n" + "=" * 80)
    print("Example 2: Manual Hook Integration")
    print("=" * 80)

    # Create model
    model = MultiModalNeuroFMX(
        d_model=256,
        n_mamba_blocks=4,
        n_latents=64,
        latent_dim=256
    )

    # Configure hooks
    config = MechIntConfig(
        sample_layers=['mamba_backbone.blocks.1', 'mamba_backbone.blocks.3'],
        save_hidden_every_n_steps=100,
        storage_path='./custom_mechint_cache'
    )

    hooks = MechIntHooks(config)
    hooks.register_hooks(model)

    # Custom training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nRunning custom training loop with mech-int hooks...")

    for epoch in range(3):
        for step in range(50):
            global_step = epoch * 50 + step

            # Dummy batch
            batch = {
                'spike': torch.randn(4, 96, 100),
                'eeg': torch.randn(4, 32, 100)
            }

            # Forward pass (hooks capture activations automatically)
            outputs = model(batch)

            # Dummy loss
            loss = sum(v.mean() for v in outputs.values() if torch.is_tensor(v))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Trigger hook callback
            if global_step % config.save_hidden_every_n_steps == 0:
                # In real usage, hooks.on_training_step would be called automatically
                print(f"  Step {global_step}: Would save activations here")

        # Epoch end
        print(f"Epoch {epoch} complete")
        # hooks.on_epoch_end(trainer=None, pl_module=model)

    # Training end
    print("\nTraining complete!")
    # hooks.on_train_end(trainer=None, pl_module=model)

    print(f"\nActivations saved to: {config.storage_path}")


# ============================================================================
# Example 3: Evaluation-Time Analysis
# ============================================================================

def example_eval_time_analysis():
    """Example of running comprehensive analysis on saved activations."""

    from neuros_mechint import EvalMechIntRunner, MechIntConfig
    from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX

    print("\n" + "=" * 80)
    print("Example 3: Evaluation-Time Analysis")
    print("=" * 80)

    # Create model
    model = MultiModalNeuroFMX(
        d_model=256,
        n_mamba_blocks=4,
        n_latents=64,
        latent_dim=256
    )

    # Configure analysis
    config = MechIntConfig(
        sample_layers=['mamba_backbone.blocks.1', 'mamba_backbone.blocks.3'],
        analyses_to_run=['sae', 'neuron', 'feature'],
        storage_path='./mechint_cache'
    )

    # Create runner
    runner = EvalMechIntRunner(model=model, config=config, device='cpu')

    print("\nRunning comprehensive mech-int evaluation...")

    # Run full analysis suite
    # results = runner.run_mechint_eval(
    #     checkpoint_path='./checkpoints/model.pt',
    #     hidden_shards_path='./mechint_cache'
    # )

    # Export results
    # runner.export_results('./mechint_results')

    print("\nAnalysis would include:")
    print("  - SAE training on each layer")
    print("  - Neuron activation statistics")
    print("  - Feature clustering analysis")
    print("  - Comprehensive report generation")
    print("\nResults would be saved to: ./mechint_results/")


# ============================================================================
# Example 4: FastAPI Integration
# ============================================================================

def example_fastapi_integration():
    """Example of adding mech-int endpoints to FastAPI server."""

    try:
        from fastapi import FastAPI
        from neuros_mechint import FastAPIIntegrationMixin, MechIntConfig
        from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX

        print("\n" + "=" * 80)
        print("Example 4: FastAPI Integration")
        print("=" * 80)

        # Create app
        app = FastAPI(title="NeuroFMX with MechInt")

        # Create model
        model = MultiModalNeuroFMX(
            d_model=256,
            n_mamba_blocks=4,
            n_latents=64,
            latent_dim=256
        )

        # Configure mech-int
        config = MechIntConfig(
            sample_layers=['mamba_backbone.blocks.3'],
            storage_path='./mechint_cache'
        )

        # Add interpretation endpoints
        mixin = FastAPIIntegrationMixin(model=model, config=config, device='cpu')
        mixin.add_routes(app)

        print("\nAdded interpretation endpoints:")
        print("  POST /interpret - Run analysis on cached activations")
        print("  POST /interpret/upload - Upload and analyze activations")
        print("  GET  /interpret/layers - List available layers")

        print("\nExample request:")
        print("""
        curl -X POST "http://localhost:8000/interpret" \\
          -H "Content-Type: application/json" \\
          -d '{
            "analysis_type": "neuron",
            "layer_name": "mamba_backbone.blocks.3",
            "config": {}
          }'
        """)

        # To run the server:
        # import uvicorn
        # uvicorn.run(app, host="0.0.0.0", port=8000)

    except ImportError as e:
        print(f"Skipping FastAPI example: {e}")


# ============================================================================
# Example 5: S3 Storage Integration
# ============================================================================

def example_s3_integration():
    """Example of using S3 for activation storage."""

    from neuros_mechint import MechIntConfig, MechIntHooks

    print("\n" + "=" * 80)
    print("Example 5: S3 Storage Integration")
    print("=" * 80)

    # Configure with S3 backend
    config = MechIntConfig(
        sample_layers=['mamba_backbone.blocks.3'],
        save_hidden_every_n_steps=200,
        storage_backend='s3',  # Use S3 instead of local
        storage_path='./local_cache',  # Local cache for processing
        s3_bucket='my-neurofmx-bucket',
        s3_prefix='experiments/exp001/activations',
        verbose=True
    )

    print("\nS3 Configuration:")
    print(f"  Backend: {config.storage_backend}")
    print(f"  Bucket: {config.s3_bucket}")
    print(f"  Prefix: {config.s3_prefix}")
    print(f"  Local cache: {config.storage_path}")

    print("\nActivations will be:")
    print("  1. Saved locally to ./local_cache/")
    print("  2. Automatically uploaded to S3")
    print("  3. Available at s3://my-neurofmx-bucket/experiments/exp001/activations/")


# ============================================================================
# Example 6: Custom Activation Sampler
# ============================================================================

def example_custom_sampler():
    """Example of using ActivationSampler directly."""

    from neuros_mechint import ActivationSampler
    from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX

    print("\n" + "=" * 80)
    print("Example 6: Custom Activation Sampler")
    print("=" * 80)

    # Create model
    model = MultiModalNeuroFMX(
        d_model=256,
        n_mamba_blocks=4,
        n_latents=64,
        latent_dim=256
    )

    # Create sampler
    sampler = ActivationSampler(
        layers=['mamba_backbone.blocks.0', 'mamba_backbone.blocks.3'],
        save_dir='./custom_activations',
        max_samples_per_shard=5000
    )

    # Register hooks
    registered = sampler.register_hooks(model)
    print(f"\nRegistered hooks on: {registered}")

    # Run some forward passes
    print("\nRunning forward passes...")
    for i in range(10):
        batch = {
            'spike': torch.randn(4, 96, 100),
            'eeg': torch.randn(4, 32, 100)
        }
        _ = model(batch)

    # Save collected activations
    save_path = sampler.save_activations(global_step=100)
    print(f"\nSaved activations to: {save_path}")

    # Get statistics
    stats = sampler.get_statistics()
    print(f"\nStatistics: {stats}")

    # Clear cache
    sampler.clear_cache()
    print("Cache cleared!")


# ============================================================================
# Example 7: Complete End-to-End Workflow
# ============================================================================

def example_complete_workflow():
    """Complete workflow: training -> saving -> analysis -> visualization."""

    print("\n" + "=" * 80)
    print("Example 7: Complete End-to-End Workflow")
    print("=" * 80)

    print("""
COMPLETE WORKFLOW:

1. TRAINING PHASE
   ---------------
   from neuros_mechint import MechIntCallback, MechIntConfig

   config = MechIntConfig(
       sample_layers=['mamba_backbone.blocks.3', 'popt'],
       save_hidden_every_n_steps=200,
       analyses_to_run=['sae', 'neuron', 'feature'],
       storage_backend='both',  # Save locally AND to S3
       s3_bucket='my-bucket'
   )

   trainer = pl.Trainer(
       callbacks=[MechIntCallback(config=config)],
       max_epochs=100
   )

   trainer.fit(model, train_dataloader)
   # Activations automatically saved every 200 steps
   # -> ./mechint_cache/activations_shard_*.pt
   # -> s3://my-bucket/neurofmx/activations/

2. EVALUATION PHASE
   ----------------
   from neuros_mechint import EvalMechIntRunner

   runner = EvalMechIntRunner(model=model, config=config)

   results = runner.run_mechint_eval(
       checkpoint_path='./checkpoints/best.pt',
       hidden_shards_path='./mechint_cache'
   )

   runner.export_results('./results')
   # -> ./results/mechint_results.json
   # -> ./results/mechint_report.md

3. VISUALIZATION PHASE
   -------------------
   from neuros_mechint import SAEVisualizer

   # Load trained SAE from results
   sae = results['analyses']['sae']['layer_name']['sae']

   visualizer = SAEVisualizer(sae)
   visualizer.plot_feature_distribution()
   visualizer.plot_top_features(k=20)
   visualizer.save_all_plots('./visualizations')

4. REAL-TIME INFERENCE
   --------------------
   # Start FastAPI server with mech-int endpoints
   from neuros_mechint import FastAPIIntegrationMixin

   app = create_app(model_path='./model.pt')
   mixin = FastAPIIntegrationMixin(model, config)
   mixin.add_routes(app)

   # Query interpretation
   curl -X POST "http://localhost:8000/interpret" \\
     -d '{"analysis_type": "neuron", "layer_name": "layer3"}'
    """)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MECHANISTIC INTERPRETABILITY HOOKS - EXAMPLES")
    print("=" * 80)

    # Run examples
    example_lightning_integration()
    example_manual_hook_integration()
    example_eval_time_analysis()
    example_fastapi_integration()
    example_s3_integration()
    example_custom_sampler()
    example_complete_workflow()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
