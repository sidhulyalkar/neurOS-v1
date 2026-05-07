"""Tests for ENGRAM-FMx diagnostics and visualization."""

import pytest
import torch

from neuros_neurofm.backbones.engram_fmx.diagnostics import (
    # Memory metrics
    compute_memory_entropy,
    compute_memory_usage,
    analyze_memory_retrieval,
    MemoryTracker,
    # Latent metrics
    compute_latent_pca,
    compute_latent_similarity,
    track_latent_trajectory,
    LatentTracker,
)


class TestMemoryMetrics:
    """Test memory metric functions."""

    def test_compute_memory_entropy_uniform(self):
        """Uniform weights should have max entropy."""
        # Uniform distribution
        weights = torch.ones(4, 32, 64) / 64  # [B, K, M]
        result = compute_memory_entropy(weights)

        assert "mean_entropy" in result
        assert "mean_normalized_entropy" in result
        # Uniform should have normalized entropy close to 1
        assert result["mean_normalized_entropy"] > 0.99

    def test_compute_memory_entropy_peaked(self):
        """Peaked weights should have low entropy."""
        weights = torch.zeros(2, 8, 32)
        weights[:, :, 0] = 1.0  # All attention on first slot

        result = compute_memory_entropy(weights)
        assert result["mean_normalized_entropy"] < 0.01

    def test_compute_memory_usage(self):
        """Test memory slot usage statistics."""
        weights = torch.randn(4, 16, 64).softmax(dim=-1)
        usage = compute_memory_usage(weights)

        assert "active_slots" in usage
        assert "active_ratio" in usage
        assert "total_slots" in usage
        assert "usage_gini" in usage
        assert usage["total_slots"] == 64

    def test_analyze_memory_retrieval(self):
        """Test full memory retrieval analysis."""
        weights = torch.randn(4, 16, 64).softmax(dim=-1)
        analysis = analyze_memory_retrieval(weights, top_k=5)

        assert "top_k_indices" in analysis
        assert "top_k_weights" in analysis
        assert "top_k_concentration" in analysis
        assert len(analysis["top_k_indices"]) == 5


class TestMemoryTracker:
    """Test MemoryTracker for training monitoring."""

    def test_memory_tracker_init(self):
        """Test tracker initialization."""
        tracker = MemoryTracker(max_history=10)
        assert tracker.max_history == 10
        assert len(tracker.history) == 0

    def test_memory_tracker_update(self):
        """Test tracking memory weights over time."""
        tracker = MemoryTracker(max_history=5)

        for i in range(7):
            weights = torch.randn(2, 8, 32).softmax(dim=-1)
            tracker.update(step=i, memory_weights=weights)

        # Max history should keep only last 5
        assert len(tracker.history) == 5

    def test_memory_tracker_summary(self):
        """Test getting summary statistics."""
        tracker = MemoryTracker(max_history=10)

        for i in range(5):
            weights = torch.randn(2, 8, 32).softmax(dim=-1)
            tracker.update(step=i, memory_weights=weights)

        summary = tracker.summary()
        assert "mean_entropy_mean" in summary
        assert "mean_entropy_final" in summary

    def test_memory_tracker_get_history(self):
        """Test getting history for a metric."""
        tracker = MemoryTracker(max_history=10)

        for i in range(5):
            weights = torch.randn(2, 8, 32).softmax(dim=-1)
            tracker.update(step=i, memory_weights=weights)

        steps, values = tracker.get_history("mean_entropy")
        assert len(steps) == 5
        assert len(values) == 5


class TestLatentMetrics:
    """Test latent space metric functions."""

    def test_compute_latent_pca_basic(self):
        """Test PCA computation on latents."""
        latents = torch.randn(32, 64)  # [N, D]
        result = compute_latent_pca(latents, n_components=3)

        assert "projected" in result
        assert "explained_variance_ratio" in result
        assert "components" in result
        assert result["explained_variance_ratio"].shape == (3,)

    def test_compute_latent_pca_batched(self):
        """Test PCA with batched input."""
        latents = torch.randn(4, 16, 64)  # [B, K, D]
        result = compute_latent_pca(latents, n_components=2)

        # Should reshape to [B, K, n_components]
        assert result["projected"].shape == (4, 16, 2)

    def test_compute_latent_similarity(self):
        """Test latent similarity computation."""
        latents = torch.randn(2, 16, 64)  # [B, K, D]
        result = compute_latent_similarity(latents)

        assert "similarity_matrix" in result
        assert "mean_similarity" in result
        assert result["similarity_matrix"].shape == (2, 16, 16)
        # Diagonal should be 1 (self-similarity with cosine)
        diag = result["similarity_matrix"][0].diag()
        assert torch.allclose(diag, torch.ones(16), atol=0.01)

    def test_track_latent_trajectory(self):
        """Test tracking latent evolution."""
        latents_list = [torch.randn(4, 8, 32) for _ in range(5)]
        trajectory = track_latent_trajectory(latents_list)

        assert "trajectory" in trajectory
        assert "mean_step_size" in trajectory
        assert "drift_norms" in trajectory
        assert "path_length" in trajectory
        assert "displacement" in trajectory
        assert trajectory["trajectory"].shape == (5, 4, 8, 32)


class TestLatentTracker:
    """Test LatentTracker for training monitoring."""

    def test_latent_tracker_init(self):
        """Test tracker initialization."""
        tracker = LatentTracker(max_history=20)
        assert tracker.max_history == 20

    def test_latent_tracker_update(self):
        """Test tracking latents over time."""
        tracker = LatentTracker(max_history=10)

        for i in range(15):
            latents = torch.randn(4, 16, 64)
            tracker.update(step=i, latents=latents)

        # Should keep max_history entries
        assert len(tracker.history) == 10

    def test_latent_tracker_stores_pca_components(self):
        """Test that first PCA components are stored."""
        tracker = LatentTracker(max_history=10)

        latents = torch.randn(4, 16, 64)
        tracker.update(step=0, latents=latents)

        assert tracker.pca_components is not None
        assert tracker.pca_mean is not None

    def test_latent_tracker_project_to_pca(self):
        """Test projecting with stored PCA components."""
        tracker = LatentTracker(max_history=10)

        # First update stores PCA components
        latents1 = torch.randn(4, 16, 64)
        tracker.update(step=0, latents=latents1)

        # Project new latents using stored components
        latents2 = torch.randn(4, 16, 64)
        projected = tracker.project_to_pca(latents2)

        assert projected.shape == (4, 16, 3)

    def test_latent_tracker_get_history(self):
        """Test getting history for a metric."""
        tracker = LatentTracker(max_history=10)

        for i in range(5):
            latents = torch.randn(4, 16, 64)
            tracker.update(step=i, latents=latents)

        steps, values = tracker.get_history("latent_norm")
        assert len(steps) == 5
        assert len(values) == 5


class TestVisualizationImports:
    """Test that visualization functions can be imported."""

    def test_visualization_imports(self):
        """Test all visualization functions are importable."""
        from neuros_neurofm.backbones.engram_fmx.diagnostics import (
            plot_memory_heatmap,
            plot_memory_entropy_over_time,
            plot_latent_pca,
            plot_latent_trajectory_3d,
            plot_gate_activations,
            plot_sparse_anchor_indices,
            create_diagnostic_dashboard,
        )

        # Just check they're callable
        assert callable(plot_memory_heatmap)
        assert callable(plot_memory_entropy_over_time)
        assert callable(plot_latent_pca)
        assert callable(plot_latent_trajectory_3d)
        assert callable(plot_gate_activations)
        assert callable(plot_sparse_anchor_indices)
        assert callable(create_diagnostic_dashboard)


class TestIntegrationWithBackbone:
    """Test diagnostics work with actual backbone output."""

    def test_diagnostics_with_backbone_output(self):
        """Test computing diagnostics from backbone forward pass."""
        from neuros_neurofm.backbones.engram_fmx import (
            ENGRAMFMxConfig,
            ENGRAMBackbone,
        )

        config = ENGRAMFMxConfig(
            input_dim=32,
            hidden_dim=32,
            num_layers=1,
            num_latents=8,
            memory_slots=16,
        )
        backbone = ENGRAMBackbone(config)

        x = torch.randn(2, 16, 32)
        output = backbone(x)

        # Extract memory weights from diagnostics
        memory_key = "layer0_memory_memory_weights"
        assert memory_key in output.diagnostics

        weights = output.diagnostics[memory_key]

        # Compute entropy
        entropy_result = compute_memory_entropy(weights)
        assert "mean_entropy" in entropy_result
        assert isinstance(entropy_result["mean_entropy"], float)

    def test_trackers_during_training_loop(self):
        """Test using trackers in a simulated training loop."""
        from neuros_neurofm.backbones.engram_fmx import (
            ENGRAMFMxConfig,
            ENGRAMBackbone,
        )

        config = ENGRAMFMxConfig(
            input_dim=32,
            hidden_dim=32,
            num_layers=1,
            num_latents=8,
            memory_slots=16,
        )
        backbone = ENGRAMBackbone(config)

        memory_tracker = MemoryTracker(max_history=10)
        latent_tracker = LatentTracker(max_history=10)

        # Simulate training steps
        for step in range(10):
            x = torch.randn(2, 16, 32)
            output = backbone(x)

            # Track memory
            weights = output.diagnostics["layer0_memory_memory_weights"]
            memory_tracker.update(step=step, memory_weights=weights)

            # Track latents
            latent_tracker.update(step=step, latents=output.latent_output)

        # Get summaries
        mem_summary = memory_tracker.summary()
        assert "mean_entropy_mean" in mem_summary

        # Get history
        steps, values = latent_tracker.get_history("latent_norm")
        assert len(steps) == 10
