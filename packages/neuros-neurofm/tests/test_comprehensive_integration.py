"""
Comprehensive integration tests for NeuroFM-X.

Tests all components working together end-to-end.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

# Core models
from neuros_neurofm.models.neurofmx import NeuroFMX
from neuros_neurofm.models.neurofmx_complete import NeuroFMXComplete
from neuros_neurofm.models.popt import PopT
from neuros_neurofm.models.heads import MultiTaskHeads

# Tokenizers
from neuros_neurofm.tokenizers.spike_tokenizer import SpikeTokenizer
from neuros_neurofm.tokenizers.binned_tokenizer import BinnedTokenizer
from neuros_neurofm.tokenizers.lfp_tokenizer import LFPTokenizer
from neuros_neurofm.tokenizers.calcium_tokenizer import (
    CalciumTokenizer,
    TwoPhotonTokenizer,
    MiniscopeTokenizer,
)

# Diffusion
from neuros_neurofm.diffusion.latent_diffusion import (
    LatentDiffusionModel,
    DiffusionSchedule,
    SimpleUNet,
)

# Adapters
from neuros_neurofm.adapters import UnitIDAdapter, LoRAAdapter

# Optimization
from neuros_neurofm.optimization.model_compression import (
    ModelQuantizer,
    ModelPruner,
    TorchScriptExporter,
    compute_model_size,
)

# Inference
from neuros_neurofm.inference.realtime_pipeline import (
    RealtimeInferencePipeline,
    DynamicBatcher,
    ModelCache,
    LatencyProfiler,
)


class TestAllTokenizers:
    """Test all tokenizers work correctly."""

    def test_spike_tokenizer(self):
        """Test spike tokenizer."""
        tokenizer = SpikeTokenizer(n_units=100, d_model=256)

        # Create spike times and units (sparse representation)
        spike_times = torch.rand(4, 50) * 100  # 50 spikes per batch
        spike_units = torch.randint(0, 100, (4, 50))  # which unit spiked

        # Tokenize
        tokens, mask = tokenizer(spike_times, spike_units)

        assert tokens.shape[0] == 4  # batch
        assert tokens.shape[2] == 256  # d_model
        assert not torch.isnan(tokens).any()

    def test_binned_tokenizer(self):
        """Test binned tokenizer."""
        tokenizer = BinnedTokenizer(n_units=100, d_model=256)

        # Create binned spikes (batch, n_units, n_bins)
        binned = torch.poisson(torch.ones(4, 100, 100) * 2.0)

        # Tokenize (returns tuple: tokens, mask)
        tokens, mask = tokenizer(binned)

        assert tokens.shape == (4, 100, 256)
        assert not torch.isnan(tokens).any()

    def test_lfp_tokenizer(self):
        """Test LFP tokenizer."""
        tokenizer = LFPTokenizer(n_channels=32, d_model=256)

        # Create LFP signal
        lfp = torch.randn(4, 32, 1000)  # (batch, channels, time)

        # Tokenize (returns tuple: tokens, mask)
        result = tokenizer(lfp)
        if isinstance(result, tuple):
            tokens, mask = result
        else:
            tokens = result

        assert tokens.ndim == 3
        assert tokens.shape[0] == 4
        assert tokens.shape[2] == 256
        assert not torch.isnan(tokens).any()

    def test_calcium_tokenizer(self):
        """Test calcium imaging tokenizer."""
        tokenizer = CalciumTokenizer(
            n_neurons=200,
            d_model=256,
            detect_events=True,
        )

        # Create calcium traces (dF/F)
        calcium = torch.randn(4, 200, 500) * 0.5 + 0.1

        # Tokenize
        tokens, events = tokenizer(calcium)

        # Calcium tokenizer may downsample both time and neurons
        assert tokens.shape[0] == 4  # batch
        assert tokens.shape[2] == 256  # d_model
        assert events.shape[0] == 4  # batch
        assert events.shape[1] > 0  # has neurons (may be downsampled)
        assert not torch.isnan(tokens).any()

    def test_two_photon_tokenizer(self):
        """Test two-photon specific tokenizer."""
        tokenizer = TwoPhotonTokenizer(n_neurons=300, d_model=256)

        calcium = torch.randn(2, 300, 600) * 0.3
        tokens, events = tokenizer(calcium)

        assert tokens.shape[0] == 2  # batch
        assert tokens.shape[2] == 256  # d_model
        assert not torch.isnan(tokens).any()

    def test_miniscope_tokenizer(self):
        """Test miniscope specific tokenizer."""
        tokenizer = MiniscopeTokenizer(n_neurons=150, d_model=256)

        calcium = torch.randn(2, 150, 400) * 0.4
        tokens, events = tokenizer(calcium)

        assert tokens.shape[0] == 2  # batch
        assert tokens.shape[1] <= 400  # Might downsample
        assert tokens.shape[2] == 256  # d_model
        assert not torch.isnan(tokens).any()


class TestDiffusionModel:
    """Test diffusion model components."""

    def test_diffusion_schedule(self):
        """Test diffusion noise schedule."""
        schedule = DiffusionSchedule(n_timesteps=100, schedule_type='cosine')

        # Test forward diffusion
        x_start = torch.randn(4, 128)
        t = torch.randint(0, 100, (4,))
        x_noisy = schedule.q_sample(x_start, t)

        assert x_noisy.shape == x_start.shape
        assert not torch.isnan(x_noisy).any()

    def test_simple_unet(self):
        """Test SimpleUNet denoiser."""
        unet = SimpleUNet(dim=128, condition_dim=64)

        # Test forward pass
        x = torch.randn(4, 128)
        t = torch.randint(0, 100, (4,))
        condition = torch.randn(4, 64)

        noise_pred = unet(x, t, condition)

        assert noise_pred.shape == (4, 128)
        assert not torch.isnan(noise_pred).any()

    def test_latent_diffusion_model(self):
        """Test complete latent diffusion model."""
        model = LatentDiffusionModel(
            latent_dim=256,
            n_timesteps=100,
            condition_dim=128,
        )

        # Test training
        x_start = torch.randn(8, 256)
        condition = torch.randn(8, 128)

        loss, pred_noise = model(x_start, condition)

        assert loss.ndim == 0  # Scalar loss
        assert pred_noise.shape == (8, 256)
        assert not torch.isnan(loss)

    def test_diffusion_sampling(self):
        """Test generating samples from diffusion model."""
        model = LatentDiffusionModel(latent_dim=128, n_timesteps=50)

        # Generate samples (use fewer timesteps for speed)
        samples = model.sample(batch_size=4, device='cpu')

        assert samples.shape == (4, 128)
        assert not torch.isnan(samples).any()

    def test_diffusion_forecasting(self):
        """Test neural forecasting with diffusion."""
        model = LatentDiffusionModel(latent_dim=128, n_timesteps=50)

        # Context from past
        context = torch.randn(4, 128)

        # Forecast future steps
        forecasts = model.forecast(context, n_steps=5)

        assert forecasts.shape == (4, 5, 128)
        assert not torch.isnan(forecasts).any()


class TestModelCompression:
    """Test model compression utilities."""

    def test_model_size_computation(self):
        """Test computing model size."""
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

        size_info = compute_model_size(model)

        assert 'total_mb' in size_info
        assert 'param_mb' in size_info
        assert size_info['total_mb'] > 0

    def test_torchscript_export(self):
        """Test TorchScript export."""
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

        example_inputs = torch.randn(1, 100)

        exporter = TorchScriptExporter(model, example_inputs)
        script_model = exporter.export()

        # Test exported model
        output = script_model(example_inputs)
        assert output.shape == (1, 64)


class TestRealtimeInference:
    """Test real-time inference pipeline."""

    def test_dynamic_batcher(self):
        """Test dynamic batching."""
        from neuros_neurofm.inference.realtime_pipeline import InferenceRequest

        batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100.0)

        # Add requests
        for i in range(3):
            req = InferenceRequest(
                request_id=f"req-{i}",
                data=torch.randn(10, 100),
                timestamp=0.0,
            )
            batcher.add_request(req)

        # Get batch (should return immediately since we have requests)
        batch = batcher.get_batch(timeout=0.5)

        assert batch is not None
        requests, batched_data = batch
        assert len(requests) == 3
        assert batched_data.shape[0] == 3

    def test_model_cache(self):
        """Test model caching and warm-up."""
        model = nn.Linear(100, 64)
        cache = ModelCache(model, device='cpu', warmup_steps=5)

        # Warm up
        example_input = torch.randn(4, 100)
        cache.warmup(example_input)

        # Forward pass
        output = cache.forward(example_input)
        assert output.shape == (4, 64)

    def test_latency_profiler(self):
        """Test latency profiling."""
        profiler = LatencyProfiler(window_size=100)

        # Record some latencies
        for i in range(50):
            profiler.record(10.0 + np.random.randn(), 'inference')

        # Get stats
        stats = profiler.get_stats('inference')

        assert 'mean' in stats
        assert 'p95' in stats
        assert 'p99' in stats
        assert stats['count'] == 50


class TestCompleteIntegration:
    """Test complete end-to-end integration."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from tokenization to prediction."""
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(256, 64)
                self.decoder = nn.Linear(64, 2)

            def forward(self, x):
                # x: (batch, time, 256)
                x = x.mean(dim=1)  # Average over time
                x = self.proj(x)
                return self.decoder(x)

        model = SimpleModel()

        # Create tokenizer (use binned for dense data)
        tokenizer = BinnedTokenizer(n_units=100, d_model=256)

        # Create data (binned spikes) - shape: (batch, time, n_units)
        spikes = torch.poisson(torch.ones(4, 200, 100) * 0.1)

        # Tokenize (returns tuple)
        tokens, mask = tokenizer(spikes)

        # Forward through model
        output = model(tokens)

        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()

    def test_diffusion_with_model_output(self):
        """Test diffusion model with real model outputs."""
        # Simple encoder to latents
        encoder = nn.Linear(256, 128)

        # Diffusion model
        diffusion = LatentDiffusionModel(latent_dim=128, n_timesteps=50)

        # Create input
        tokens = torch.randn(4, 100, 256)

        # Encode to latents
        latents = encoder(tokens.mean(dim=1))  # (4, 128)

        # Train diffusion
        loss, _ = diffusion(latents)

        assert not torch.isnan(loss)

        # Forecast
        forecasts = diffusion.forecast(latents, n_steps=3)
        assert forecasts.shape == (4, 3, 128)

    def test_adapters_integration(self):
        """Test adapters with base model."""
        # Base model
        base_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Add Unit-ID adapter
        adapter = UnitIDAdapter(
            backbone_dim=128,
            n_units=100,
            bottleneck_dim=32,
        )

        # Forward pass
        features = torch.randn(4, 128)
        unit_ids = torch.randint(0, 100, (4,))

        adapted_features = adapter(features, unit_ids)

        assert adapted_features.shape == (4, 128)
        assert not torch.isnan(adapted_features).any()


def test_all_components_importable():
    """Test that all components can be imported."""
    # This test ensures all imports work
    from neuros_neurofm import (
        NeuroFMX,
        NeuroFMXTrainer,
        NWBDataset,
        IBLDataset,
        AllenDataset,
        CalciumTokenizer,
        TwoPhotonTokenizer,
        MiniscopeTokenizer,
        HyperparameterSearch,
        GridSearch,
        ModelQuantizer,
        ModelPruner,
        TorchScriptExporter,
        RealtimeInferencePipeline,
        DynamicBatcher,
        ModelCache,
        LatencyProfiler,
    )

    # All imports successful
    assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
